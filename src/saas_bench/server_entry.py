#!/usr/bin/env python3
"""NovaMind Server — Entry point for the PyInstaller binary.

This is the single executable that manages sessions and runs the simulator.
It is invoked by the `novamind-operation` CLI wrapper.

Commands:
    new-session   Create a new simulation session
    start-server  Start the API server for an existing session
    stop-server   Stop a running API server
    status        Get session status
    list-sessions List all sessions
"""

import argparse
import json
import uuid
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

from numpy.random import Generator, PCG64

from saas_bench.config import BenchmarkConfig, SCENARIO_PACKS, ScenarioPack
from saas_bench.database import init_database
from saas_bench.simulation import Simulator
from saas_bench.customer_llm import CustomerSimulator
from saas_bench.tools import AgentTools
from saas_bench.shocks import ShockManager
from saas_bench.event_logger import EventLogger
from saas_bench.api_server import NovaMindAPIServer
from saas_bench.db_protection import (
    protect_db,
    save_session_db,
    load_session_db,
    snapshot_to_plain,
    AsyncSaver,
)
from saas_bench.docs_generator import initialize_workspace


_SIMULATOR_LLM_CONFIG_FIELDS = (
    "social_post_llm_provider",
    "social_post_llm_model",
    "enterprise_llm_provider",
    "enterprise_llm_model",
)


def _sessions_dir(base: Path) -> Path:
    return base / "sessions"


def _session_dir(base: Path, session_id: str) -> Path:
    return _sessions_dir(base) / session_id


def _session_meta_path(base: Path, session_id: str) -> Path:
    return _session_dir(base, session_id) / "session.json"


def _session_nmdb_path(base: Path, session_id: str) -> Path:
    return _session_dir(base, session_id) / "world.nmdb"


def _session_workspace(base: Path, session_id: str) -> Path:
    return _session_dir(base, session_id) / "workspace"


def _session_history_path(base: Path, session_id: str) -> Path:
    return _session_dir(base, session_id) / "history.jsonl"


def _pid_file(base: Path, session_id: str) -> Path:
    return _session_dir(base, session_id) / ".server.pid"


def _port_file(base: Path, session_id: str) -> Path:
    return _session_dir(base, session_id) / ".server.port"


def _generate_session_id() -> str:
    import hashlib
    raw = f"{time.time()}-{os.getpid()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _get_latest_session(base: Path) -> Optional[str]:
    """Get the most recently created session ID."""
    sessions_dir = _sessions_dir(base)
    if not sessions_dir.exists():
        return None
    sessions = []
    for d in sessions_dir.iterdir():
        meta = d / "session.json"
        if meta.exists():
            try:
                data = json.loads(meta.read_text())
                sessions.append((data.get("created_at", 0), d.name))
            except Exception:
                pass
    if not sessions:
        return None
    sessions.sort(reverse=True)
    return sessions[0][1]


def _resolve_session(base: Path, session_id: Optional[str]) -> str:
    """Resolve session ID (use latest if not specified)."""
    if session_id:
        meta = _session_meta_path(base, session_id)
        if not meta.exists():
            print(f"Error: Session '{session_id}' not found.", file=sys.stderr)
            sys.exit(1)
        return session_id
    latest = _get_latest_session(base)
    if not latest:
        print("Error: No sessions found. Create one with: ./novamind-operation new-session", file=sys.stderr)
        sys.exit(1)
    return latest


def _apply_simulator_llm_config(config: BenchmarkConfig) -> dict:
    """Validate and serialize simulator-side LLM provider/model config."""
    override_provider = os.environ.get("CEOBENCH_SIMULATOR_LLM_PROVIDER")
    override_model = os.environ.get("CEOBENCH_SIMULATOR_LLM_MODEL")
    if override_provider:
        config.social_post_llm_provider = override_provider
        config.enterprise_llm_provider = override_provider
    if override_model:
        config.social_post_llm_model = override_model
        config.enterprise_llm_model = override_model

    valid_providers = {"bedrock", "anthropic", "openai", "deepseek", "opencode"}
    for attr in ("social_post_llm_provider", "enterprise_llm_provider"):
        provider = getattr(config, attr)
        if provider not in valid_providers:
            print(f"Error: invalid simulator LLM provider for {attr}: {provider!r}", file=sys.stderr)
            sys.exit(1)

    if (
        config.social_post_llm_provider == "anthropic"
        or config.enterprise_llm_provider == "anthropic"
    ) and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Error: simulator Anthropic provider requires ANTHROPIC_API_KEY. "
            "It does not use agent-only credentials such as --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    if (
        config.social_post_llm_provider == "openai"
        or config.enterprise_llm_provider == "openai"
    ) and not os.environ.get("OPENAI_API_KEY"):
        print(
            "Error: simulator OpenAI provider requires OPENAI_API_KEY. "
            "It does not use agent-only credentials such as --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    if (
        config.social_post_llm_provider == "deepseek"
        or config.enterprise_llm_provider == "deepseek"
    ) and not os.environ.get("DEEPSEEK_API_KEY"):
        print(
            "Error: simulator DeepSeek provider requires DEEPSEEK_API_KEY. "
            "It does not use agent-only credentials such as --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    if 'opencode' in (config.social_post_llm_provider, config.enterprise_llm_provider) and not os.environ.get('OPENCODE_API_KEY'):
        raise ValueError('Simulator OpenCode provider requires OPENCODE_API_KEY')

    return {field: getattr(config, field) for field in _SIMULATOR_LLM_CONFIG_FIELDS}


def _restore_simulator_llm_config(config: BenchmarkConfig, meta: dict) -> None:
    simulator_llm = meta.get("simulator_llm") or {}
    for attr in _SIMULATOR_LLM_CONFIG_FIELDS:
        value = simulator_llm.get(attr)
        if value:
            setattr(config, attr, value)


def _session_config(seed, total_days, initial_cash, meta=None):
    frozen = (meta or {}).get('benchmark_config')
    manifest_path = os.environ.get('CEOBENCH_RUN_MANIFEST')
    if manifest_path:
        requested = json.loads(Path(manifest_path).read_text())['benchmark_config']
        if frozen and frozen != requested:
            raise ValueError('Session configuration differs from run manifest')
        frozen = requested
    config = BenchmarkConfig(**frozen) if frozen else BenchmarkConfig(
        seed=seed, total_days=total_days, initial_cash=initial_cash)
    if meta and not frozen:
        _restore_simulator_llm_config(config, meta)
    before = {field: getattr(config, field) for field in _SIMULATOR_LLM_CONFIG_FIELDS}
    effective = _apply_simulator_llm_config(config)
    if frozen and effective != before:
        raise ValueError('Simulator environment would change frozen model configuration')
    return config


def _create_simulator_openai_client(config: BenchmarkConfig):
    providers = {config.social_post_llm_provider, config.enterprise_llm_provider}
    compatible = providers.intersection({"openai", "deepseek", "opencode"})
    if not compatible:
        return None
    if len(compatible) > 1:
        raise ValueError('Simulator OpenAI-compatible providers must share one endpoint')

    from openai import OpenAI

    if "opencode" in providers:
        return OpenAI(
            api_key=os.environ.get('OPENCODE_API_KEY'),
            base_url='https://opencode.ai/zen/go/v1',
            default_headers={'User-Agent': 'CEO-Bench/1.0',
                             'x-opencode-session': os.environ.get('CEOBENCH_MODEL_SESSION') or uuid.uuid4().hex},
        )
    if "deepseek" in providers:
        return OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
    return OpenAI(base_url=config.simulator_openai_base_url)


# =========================================================================
# Commands
# =========================================================================

def cmd_new_session(args, base: Path):
    """Create a new simulation session."""
    session_id = _generate_session_id()
    sdir = _session_dir(base, session_id)
    sdir.mkdir(parents=True, exist_ok=True)

    total_days = args.days
    seed = args.seed

    # Initialize RNG and config
    rng = Generator(PCG64(seed))
    config = _session_config(seed, total_days, args.cash)
    simulator_llm = _apply_simulator_llm_config(config)

    # Initialize database in memory (never writes plain SQLite to disk)
    conn = init_database(":memory:")

    # Initialize simulator with customer simulator
    customer_sim = CustomerSimulator(
        client=_create_simulator_openai_client(config),
        conn=conn,
        config=config,
    )
    simulator = Simulator(conn, config, rng, customer_simulator=customer_sim)
    simulator.initialize()

    # Save protected DB (in-memory → obfuscated .nmdb)
    nmdb_path = _session_nmdb_path(base, session_id)
    save_session_db(conn, nmdb_path)
    conn.close()

    # Initialize workspace with docs
    workspace = _session_workspace(base, session_id)
    initialize_workspace(workspace)

    # Save session metadata
    meta = {
        "session_id": session_id,
        "seed": seed,
        "total_days": total_days,
        "initial_cash": args.cash,
        "scenario": getattr(args, 'scenario', 'default'),
        "current_day": 0,
        "created_at": time.time(),
        "status": "created",
        "simulator_llm": simulator_llm,
    }
    from dataclasses import asdict
    meta['benchmark_config'] = asdict(config)
    _session_meta_path(base, session_id).write_text(json.dumps(meta, indent=2))

    # Initialize empty history
    _session_history_path(base, session_id).write_text("")

    result = {
        "session_id": session_id,
        "seed": seed,
        "total_days": total_days,
        "initial_cash": args.cash,
        "workspace": str(workspace),
        "status": "created",
    }
    print(json.dumps(result, indent=2))


def cmd_start_server(args, base: Path):
    """Start the API server for a session (runs in foreground)."""
    session_id = _resolve_session(base, args.session)
    sdir = _session_dir(base, session_id)

    # Load session metadata
    meta = json.loads(_session_meta_path(base, session_id).read_text())
    seed = meta["seed"]
    total_days = meta["total_days"]

    # Load protected DB into memory (no plain SQLite on disk)
    nmdb_path = _session_nmdb_path(base, session_id)

    if not nmdb_path.exists():
        print(f"Error: Session database not found: {nmdb_path}", file=sys.stderr)
        sys.exit(1)

    conn = load_session_db(nmdb_path)

    # Refresh planner stats on the loaded DB. Without this, the planner picks a
    # nested-loop plan for the open_issues dashboard query (scans 166k active subs
    # × ~63k filtered customer_state rows → 200+ seconds). After ANALYZE, it picks
    # an rowid lookup on customer_state and the same query runs in ~10 ms.
    conn.execute("ANALYZE")

    # Run pending migrations on the loaded DB (load_session_db skips init_database)
    try:
        conn.execute("ALTER TABLE agent_social_media_posts ADD COLUMN reasoning_by_group TEXT NOT NULL DEFAULT '{}'")
    except Exception:
        pass  # Column already exists

    # Reconstruct simulator state
    rng = Generator(PCG64(seed))
    config = _session_config(seed, total_days, meta['initial_cash'], meta)
    meta["simulator_llm"] = _apply_simulator_llm_config(config)

    customer_sim = CustomerSimulator(
        client=_create_simulator_openai_client(config),
        conn=conn,
        config=config,
    )
    simulator = Simulator(conn, config, rng, customer_simulator=customer_sim)
    simulator.initialize(resume=True)  # resume=True: skip DB writes, just set up _group_rngs
    current_day = meta.get("current_day", 0)

    workspace = _session_workspace(base, session_id)
    tools = AgentTools(conn, current_day, workspace, rng=rng, config=config, seed=seed)

    # Shock manager for world events
    scenario_name = meta.get("scenario", "default")
    scenario_pack = SCENARIO_PACKS.get(scenario_name, ScenarioPack(
        name='Default', description='Balanced scenario'
    ))
    shock_manager = ShockManager(conn, rng, scenario_pack)
    simulator.shock_manager = shock_manager
    # Construct all random streams before restoring the saved positions.
    restored = simulator.restore_rng_states()
    if current_day > 0 and not restored:
        raise ValueError('Cannot resume: checkpoint has no random states')
    if restored and simulator.current_day != current_day:
        raise ValueError('Checkpoint metadata and simulator day differ')

    # Event logger
    logs_dir = sdir / "logs"
    logs_dir.mkdir(exist_ok=True)
    event_logger = EventLogger(
        run_id=session_id,
        output_dir=logs_dir,
        seed=seed,
        scenario="default",
        config={"seed": seed, "total_days": total_days},
    )
    simulator.set_event_logger(event_logger)
    tools.set_event_logger(event_logger)

    # History logging callback
    history_path = _session_history_path(base, session_id)

    def _log_history(entry: dict):
        with open(history_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    # Async encrypter for the per-day save. The hot path snapshots the
    # in-memory conn to a plain tmp file (~10s on 1.5 GB) and submits it;
    # the worker thread does the ~90s encrypt + atomic-replace off the
    # next-week response path. Drained on shutdown.
    managed_checkpoints = bool(os.environ.get('CEOBENCH_CHECKPOINT_ROOT'))
    async_saver = None if managed_checkpoints else AsyncSaver(nmdb_path)

    # Day callback — save state after each day
    def _day_callback(day, dashboard):
        meta["current_day"] = day
        meta["status"] = "running"
        _session_meta_path(base, session_id).write_text(json.dumps(meta, indent=2))
        # Snapshot synchronously, queue encrypt to background worker.
        if async_saver:
            plain = snapshot_to_plain(conn, nmdb_path.parent)
            async_saver.submit(plain)
        event_logger.save_incremental()
        # Log to history
        _log_history({"type": "next_week", "day": day, "timestamp": time.time()})

    # Create and start API server
    sql_evidence = None
    evidence_path = os.environ.get('CEOBENCH_SQL_EVIDENCE')
    if evidence_path:
        from saas_bench.sql_evidence import SQLEvidenceStore
        identity = json.loads(Path(os.environ['CEOBENCH_RUN_MANIFEST']).read_text())['sql_evidence']
        if Path(evidence_path).resolve().is_relative_to(base.resolve()):
            raise ValueError('SQL evidence must be outside the agent workspace')
        sql_evidence = SQLEvidenceStore(evidence_path, identity)
    api_server = NovaMindAPIServer(
        tools=tools,
        simulator=simulator,
        conn=conn,
        day_callback=_day_callback,
        shock_manager=shock_manager,
        event_logger=event_logger,
        script_workspace=base,
        require_sandbox=os.environ.get('CEOBENCH_RUN_KIND') == 'formal',
        sql_evidence=sql_evidence,
        checkpoint_token=os.environ.get('CEOBENCH_CHECKPOINT_TOKEN'),
    )
    from saas_bench.run_state import write_json
    checkpoint_root = os.environ.get('CEOBENCH_CHECKPOINT_ROOT')
    if checkpoint_root:
        checkpoint_root = Path(checkpoint_root).resolve()
        if checkpoint_root.is_relative_to(base):
            raise ValueError('Checkpoint directory must be outside the agent workspace')

        def _checkpoint():
            import uuid
            snapshot_id = uuid.uuid4().hex
            target = checkpoint_root / snapshot_id
            target.mkdir(parents=True)
            started = time.monotonic()
            simulator.save_rng_states()
            event_logger.save_incremental()
            plain = snapshot_to_plain(conn, target)
            os.replace(plain, target / 'world.plain.sqlite')
            frozen = time.monotonic()
            saved_meta = dict(meta, current_day=simulator.current_day, status='created')
            for field in ('port', 'pid'):
                saved_meta.pop(field, None)
            write_json(target / 'session.json', saved_meta)
            write_json(target / 'server_state.json', {
                'day': simulator.current_day, 'dashboard': api_server.last_dashboard,
                'script_results': api_server.last_script_results,
                'usage': customer_sim.usage_recorder.summary,
                'event_logger': {name: getattr(event_logger, name) for name in
                                 ('current_day', '_event_count', '_total_llm_cost', '_missing_llm_cost')}})
            receipt = {'success': True, 'snapshot_id': snapshot_id, 'day': simulator.current_day,
                       'phase': 'frozen', 'files': ['session.json', 'server_state.json'],
                       'timings': {'world_freeze_s': frozen-started}}
            if sql_evidence:
                receipt['sql_evidence'] = sql_evidence.snapshot(target / 'sql-evidence.sqlite', checksum=False)
                control = sql_evidence.path.with_suffix('.controls.jsonl')
                if control.exists():
                    import shutil
                    shutil.copy2(control, target / control.name)
                    receipt['files'].append(control.name)
            receipt['timings']['evidence_and_metadata_freeze_s'] = time.monotonic()-frozen
            return receipt

        api_server.checkpoint_callback = _checkpoint
    restored_state = os.environ.get('CEOBENCH_RESTORE_SERVER_STATE')
    if restored_state:
        state = json.loads(Path(restored_state).read_text())
        if state['day'] != current_day:
            raise ValueError('Restored dashboard day differs from world')
        api_server._last_dashboard = state['dashboard']
        if sql_evidence and sql_evidence.execution_capture and state['dashboard']:
            from saas_bench.execution_capture import CapturedText
            from saas_bench.sql_evidence import digest
            saved = sql_evidence.load_state('dashboard')
            if saved is None or saved['sha256'] != digest(state['dashboard'].encode()):
                raise ValueError('Restored dashboard source state mismatch')
            api_server._last_dashboard = CapturedText(state['dashboard'], saved['origins'])
        api_server.last_script_results = state['script_results']
        customer_sim.usage_recorder.summary = state['usage']
        for name, value in state['event_logger'].items():
            setattr(event_logger, name, value)
    api_server.start()

    # Set API port on tools so Python sandbox routes queries through HTTP
    tools.api_port = api_server.port

    # Write PID and port files
    _pid_file(base, session_id).write_text(str(os.getpid()))
    _port_file(base, session_id).write_text(str(api_server.port))

    # Update metadata
    meta["status"] = "running"
    meta["port"] = api_server.port
    meta["pid"] = os.getpid()
    _session_meta_path(base, session_id).write_text(json.dumps(meta, indent=2))

    # Print server info
    info = {
        "session_id": session_id,
        "port": api_server.port,
        "pid": os.getpid(),
        "status": "running",
    }
    print(json.dumps(info))
    sys.stdout.flush()

    # Handle shutdown gracefully
    shutdown_requested = False

    def _shutdown(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            return
        shutdown_requested = True
        api_server.stop()
        # Drain the async encrypter, then write a fresh synchronous save so
        # any post-day-callback writes (agent tool calls between days) land
        # before exit.
        if async_saver:
            if not async_saver.shutdown(wait=True, timeout=180.0):
                raise TimeoutError('Background database save did not finish')
            save_session_db(conn, nmdb_path)
        meta["status"] = "stopped"
        meta.pop("port", None)
        meta.pop("pid", None)
        _session_meta_path(base, session_id).write_text(json.dumps(meta, indent=2))
        # Clean up PID/port files
        for f in [_pid_file(base, session_id), _port_file(base, session_id)]:
            if f.exists():
                f.unlink()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Keep running until killed
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown(None, None)


def cmd_stop_server(args, base: Path):
    """Stop a running API server."""
    session_id = _resolve_session(base, args.session)

    pid_path = _pid_file(base, session_id)
    if not pid_path.exists():
        print(json.dumps({"success": False, "error": "No running server found"}))
        return

    pid = int(pid_path.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(json.dumps({"success": True, "stopped_pid": pid}))
    except ProcessLookupError:
        # Already dead, clean up
        pid_path.unlink(missing_ok=True)
        _port_file(base, session_id).unlink(missing_ok=True)
        print(json.dumps({"success": True, "message": "Server was not running, cleaned up stale files"}))


def cmd_status(args, base: Path):
    """Get session status."""
    session_id = _resolve_session(base, args.session)
    meta = json.loads(_session_meta_path(base, session_id).read_text())

    # Check if server is actually running
    pid_path = _pid_file(base, session_id)
    if pid_path.exists():
        pid = int(pid_path.read_text().strip())
        try:
            os.kill(pid, 0)
            meta["server_running"] = True
            port_path = _port_file(base, session_id)
            if port_path.exists():
                meta["port"] = int(port_path.read_text().strip())
        except ProcessLookupError:
            meta["server_running"] = False
            pid_path.unlink(missing_ok=True)
            _port_file(base, session_id).unlink(missing_ok=True)
    else:
        meta["server_running"] = False

    print(json.dumps(meta, indent=2))


def cmd_list_sessions(args, base: Path):
    """List all sessions."""
    sessions_dir = _sessions_dir(base)
    if not sessions_dir.exists():
        print(json.dumps({"sessions": []}))
        return

    sessions = []
    for d in sorted(sessions_dir.iterdir()):
        meta_path = d / "session.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                sessions.append({
                    "session_id": meta.get("session_id", d.name),
                    "current_day": meta.get("current_day", 0),
                    "total_days": meta.get("total_days", 0),
                    "status": meta.get("status", "unknown"),
                    "seed": meta.get("seed", 0),
                })
            except Exception:
                pass

    print(json.dumps({"sessions": sessions}, indent=2))


def cmd_history(args, base: Path):
    """Show session tool call history."""
    session_id = _resolve_session(base, args.session)
    history_path = _session_history_path(base, session_id)

    if not history_path.exists() or history_path.stat().st_size == 0:
        print(json.dumps({"history": [], "count": 0}))
        return

    entries = []
    for line in history_path.read_text().strip().split("\n"):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except Exception:
                pass

    tail = args.tail or 50
    if len(entries) > tail:
        entries = entries[-tail:]

    print(json.dumps({"history": entries, "count": len(entries)}, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        prog="novamind-server",
        description="NovaMind Simulation Server",
    )
    parser.add_argument("--base", type=str, default=".",
                        help="Base directory for sessions (default: current directory)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # new-session
    p_new = subparsers.add_parser("new-session", help="Create a new simulation session")
    p_new.add_argument("--days", type=int, default=365, help="Total simulation days (default: 365)")
    p_new.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p_new.add_argument("--cash", type=float, default=1_000_000.0, help="Initial cash (default: 1000000)")
    p_new.add_argument("--scenario", type=str, default="default", help="Scenario pack (default: default)")

    # start-server
    p_start = subparsers.add_parser("start-server", help="Start API server for a session")
    p_start.add_argument("--session", type=str, default=None, help="Session ID (default: latest)")

    # stop-server
    p_stop = subparsers.add_parser("stop-server", help="Stop a running API server")
    p_stop.add_argument("--session", type=str, default=None, help="Session ID (default: latest)")

    # status
    p_status = subparsers.add_parser("status", help="Get session status")
    p_status.add_argument("--session", type=str, default=None, help="Session ID (default: latest)")

    # list-sessions
    subparsers.add_parser("list-sessions", help="List all sessions")

    # history
    p_hist = subparsers.add_parser("history", help="Show tool call history")
    p_hist.add_argument("--session", type=str, default=None, help="Session ID (default: latest)")
    p_hist.add_argument("--tail", type=int, default=50, help="Number of recent entries (default: 50)")

    args = parser.parse_args()
    base = Path(args.base).resolve()
    if args.command in ('new-session', 'start-server') and os.environ.get('CEOBENCH_RUN_KIND') == 'formal':
        from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
        BashAgentToolExecutor(base, require_sandbox=True).verify_sandbox()

    cmd_map = {
        "new-session": cmd_new_session,
        "start-server": cmd_start_server,
        "stop-server": cmd_stop_server,
        "status": cmd_status,
        "list-sessions": cmd_list_sessions,
        "history": cmd_history,
    }

    cmd_map[args.command](args, base)


if __name__ == "__main__":
    main()
