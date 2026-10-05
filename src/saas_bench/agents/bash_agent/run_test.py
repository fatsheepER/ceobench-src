#!/usr/bin/env python3
"""Test runner for Bash Agent with SaaS Bench.

This script runs a simulation using the bash_agent with any supported LLM provider.
The agent uses bash/file tools and interacts with the simulator via
novamind_api (Python library) and ./novamind-operation (CLI).

The simulation engine runs as a separate subprocess (novamind-server start-server).
The harness communicates with it exclusively via HTTP — no direct DB or simulator
access. This ensures the harness and the public repo have identical interfaces.

Supports OpenAI, xAI/Grok, Anthropic (direct and Bedrock).
"""

import copy
import json
import os
import shutil
import subprocess
import sys
import time as _time
import urllib.request
import urllib.error
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add package to path
package_root = Path(__file__).parent.parent.parent.parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from openai import OpenAI
from saas_bench.config import BenchmarkConfig
from saas_bench.model_usage import usage_delta

try:
    import anthropic
    from anthropic import AnthropicBedrock
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from saas_bench.environment import Action


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def load_env_file(env_path: Path) -> Dict[str, str]:
    env_vars = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                        value = value[1:-1]
                    env_vars[key] = value
    return env_vars


ANTHROPIC_FABLE_FALLBACK_MODEL = "claude-opus-4-8"


def _joined(first, separator, second):
    """Concatenate texts, keeping the evidence source ranges of both parts."""
    from saas_bench.execution_capture import CapturedText, slice_origins
    text = first + separator + second
    origins = list(getattr(first, 'origins', []))
    if getattr(second, 'origins', None):
        origins += slice_origins(second.origins, 0, len(second), target=len(first) + len(separator))
    return CapturedText(text, origins) if origins else text


class BashAgentRunner:
    """Runner for bash_agent with SaaS Bench.

    The simulation runs in a separate subprocess (novamind-server start-server).
    This harness only handles: agent LLM calls, tool execution, timing, and
    checkpoint management. All simulation state is queried via HTTP.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        seed: Optional[int] = None,
        scenario: Optional[str] = None,
        total_days: Optional[int] = None,
        initial_cash: Optional[float] = None,
        workspace_base: Optional[Path] = None,
        reasoning_effort: Optional[str] = None,
        continue_from: Optional[Path] = None,
        label: Optional[str] = None,
        run_kind: Optional[str] = None,
        pricing_file: Optional[Path] = None,
        sql_capture: Optional[bool] = None,
        execution_capture: Optional[bool] = None,
        text_registration: Optional[str] = None,
        pf_stale_checks: Optional[bool] = None,
        stop_after_day: Optional[int] = None,
    ):
        from saas_bench.model_usage import load_pricing
        self.pricing_registration = load_pricing(pricing_file) if pricing_file else None
        default_config = BenchmarkConfig()
        if continue_from:
            saved_manifest = json.loads((Path(continue_from) / 'manifest.json').read_text())
            saved = saved_manifest['configuration']
            saved_registration = saved_manifest.get('text_registration', 'off')
            if text_registration is not None and text_registration != saved_registration:
                raise ValueError('Resume text registration configuration mismatch')
            text_registration = saved_registration
            saved_stale = saved_manifest.get('pf_stale_checks', False)
            if pf_stale_checks is not None and pf_stale_checks != saved_stale:
                raise ValueError('Resume stale check configuration mismatch')
            pf_stale_checks = saved_stale
            saved_capture = bool(saved_manifest.get('sql_evidence'))
            saved_execution = (saved_manifest.get('sql_evidence') or {}).get('capture_scope') == 'execution'
            if execution_capture is not None and execution_capture != saved_execution:
                raise ValueError('Resume execution capture configuration mismatch')
            execution_capture = saved_execution
            if sql_capture is not None and sql_capture != saved_capture:
                raise ValueError('Resume SQL capture configuration mismatch')
            sql_capture = saved_capture
            if pricing_file and self.pricing_registration != saved_manifest.get('pricing'):
                raise ValueError('Resume pricing configuration mismatch')
            self.pricing_registration = saved_manifest.get('pricing')
            supplied = dict(model=model, provider=provider, base_url=base_url,
                            seed=seed, scenario=scenario, total_days=total_days,
                            initial_cash=initial_cash, reasoning_effort=reasoning_effort, run_kind=run_kind)
            for key, value in supplied.items():
                if key == 'total_days' and value is not None:
                    value = value // 7 * 7
                if key == 'base_url' and value is not None:
                    value = value.rstrip('/') + '/'
                expected = saved[key]
                if key == 'base_url' and expected is not None:
                    expected = expected.rstrip('/') + '/'
                if value is not None and value != expected:
                    raise ValueError(f'Resume configuration mismatch: {key}')
            model, provider, base_url = (saved[k] for k in ('model', 'provider', 'base_url'))
            seed, scenario, total_days, initial_cash, reasoning_effort = (
                saved[k] for k in ('seed', 'scenario', 'total_days', 'initial_cash', 'reasoning_effort'))
            run_kind = saved['run_kind']
        self.text_registration = text_registration or 'off'
        self.pf_stale_checks = (self.text_registration == 'pf' if pf_stale_checks is None else pf_stale_checks)
        if self.pf_stale_checks and self.text_registration != 'pf':
            raise ValueError('Automatic stale checks require PF mode')
        if self.text_registration not in ('off', 'git', 'prefix', 'pf'):
            raise ValueError('Invalid text registration mode')
        if self.text_registration in ('prefix', 'pf'):
            if execution_capture is False:
                raise ValueError('Prefix/PF registration requires execution capture')
            execution_capture = True
        if execution_capture:
            if sql_capture is False:
                raise ValueError('Execution capture requires SQL capture')
            sql_capture = True
        self.execution_capture = bool(execution_capture)
        self.evidence_store = None
        self.run_kind = run_kind or 'engineering'
        if self.run_kind not in ('engineering', 'pilot', 'formal'):
            raise ValueError('Invalid run kind')
        seed = 42 if seed is None else seed
        scenario = 'default' if scenario is None else scenario
        total_days = default_config.total_days if total_days is None else total_days
        initial_cash = 1_000_000.0 if initial_cash is None else initial_cash
        self.model = model or default_config.agent_llm_model
        self.provider = provider or default_config.agent_llm_provider
        self.seed = seed
        self.scenario = scenario
        # Round down to nearest full week so the simulation always ends on a
        # week boundary (no partial trailing week). e.g. 500 -> 497.
        self.total_days = (total_days // 7) * 7
        self.initial_cash = initial_cash
        # Harness-only stop at a completed week: never shown to the Agent and not part of
        # the effective configuration, so a stopped prefix can be forked and continued.
        if stop_after_day is not None and (stop_after_day <= 0 or stop_after_day % 7 or
                                           stop_after_day >= self.total_days):
            raise ValueError('stop_after_day must be a whole week before the configured end')
        self.stop_after_day = stop_after_day
        self.reasoning_effort = reasoning_effort or default_config.agent_llm_reasoning_effort
        self.continue_from = continue_from
        self.label = label  # Optional human-readable variant tag — surfaced on the dashboard
        self.anthropic_fallback_model = (
            ANTHROPIC_FABLE_FALLBACK_MODEL
            if self.provider == "anthropic" and "fable" in self.model.lower()
            else None
        )

        # Set in _restore_from_checkpoint when last logged tool was NOT next-week;
        # consumed once by the outer loop to skip force step_day on the resume iter.
        self._suppress_force_step_day_once = False

        if continue_from:
            self.workspace_dir = Path(continue_from).resolve()
            if not self.workspace_dir.exists():
                raise FileNotFoundError(f"Run directory not found: {self.workspace_dir}")
            config_file = self.workspace_dir / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    old_config = json.load(f)
                self.run_id = old_config['run_id']
            else:
                self.run_id = self.workspace_dir.name.replace('run_', '')
            self.workspace_base = self.workspace_dir.parent
        else:
            self.run_id = str(uuid.uuid4())[:8]
            self.workspace_base = (workspace_base or Path('./bash_agent_runs')).resolve()
            self.workspace_dir = self.workspace_base / f"run_{self.run_id}"
            self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Agent working directory (inside the run directory)
        self.agent_workspace = self.workspace_dir / "agent_workspace"
        from saas_bench.sql_evidence import FORMAT
        self.fork_source = saved_manifest.get('fork_source') if continue_from else None
        self.recovery_source = saved_manifest.get('recovery_source') if continue_from else None
        self.sql_evidence_config = (saved_manifest.get('sql_evidence') if continue_from else
            dict(format=FORMAT, run_id=self.run_id, branch_id='prefix',
                 data_source_id=uuid.uuid4().hex) if sql_capture else None)

        if self.sql_evidence_config and self.execution_capture and not continue_from:
            from saas_bench.execution_capture import EXCLUSIONS
            self.sql_evidence_config.update(capture_scope='execution', file_exclusions=EXCLUSIONS)

        # Logs directory
        self.logs_dir = self.workspace_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        # Log file for raw responses
        self.response_log_file = self.logs_dir / f"raw_responses_{self.run_id}.jsonl"

        # Timing log — fine-grained per-turn and per-day timing data
        self.timing_log_file = self.logs_dir / f"timing_{self.run_id}.jsonl"

        # CEOBench dashboard URL for live timing push (set via env var)
        self._dashboard_url = os.environ.get("CEOBENCH_DASHBOARD_URL", "")
        self._timing_queue = None
        if self._dashboard_url:
            import queue, threading
            self._timing_queue = queue.Queue(maxsize=500)
            def _timing_poster():
                batch = []
                while True:
                    try:
                        item = self._timing_queue.get(timeout=5)
                        if item is None:
                            break
                        batch.append(item)
                        # Drain up to 20 more without blocking
                        for _ in range(20):
                            try:
                                batch.append(self._timing_queue.get_nowait())
                            except queue.Empty:
                                break
                    except queue.Empty:
                        pass
                    if batch:
                        try:
                            data = json.dumps(batch).encode()
                            req = urllib.request.Request(
                                self._dashboard_url.rstrip('/') + '/ingest',
                                data=data,
                                headers={'Content-Type': 'application/json'},
                                method='POST',
                            )
                            urllib.request.urlopen(req, timeout=10)
                        except Exception:
                            pass  # Non-critical — dashboard may be down
                        batch = []
            self._timing_thread = threading.Thread(target=_timing_poster, daemon=True)
            self._timing_thread.start()

        # Load API key
        env_file = Path(__file__).parent.parent.parent.parent.parent / ".env"
        env_vars = load_env_file(env_file)

        for key, value in env_vars.items():
            if key not in os.environ:
                os.environ[key] = value

        # The .nmdb session database is SQLCipher-encrypted. The engine resolves
        # the key from saas_bench._embedded_key (committed in the source tree
        # and compiled into the zipapp) or, failing that, the NMDB_KEY env var.
        # Fail fast here only if neither source is available.
        try:
            from saas_bench.db_protection import _get_key
            _get_key()
        except RuntimeError as exc:
            raise RuntimeError(
                "No SQLCipher key available for the .nmdb session database: "
                "neither saas_bench._embedded_key nor the NMDB_KEY env var is "
                "set. Restore src/saas_bench/_embedded_key.py, or set NMDB_KEY "
                "in .env or the environment."
            ) from exc

        self.use_anthropic = self.provider in ("anthropic", "bedrock")
        if api_key:
            self.api_key = api_key
        elif self.provider == "xai":
            self.api_key = env_vars.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY")
        elif self.provider == "google":
            self.api_key = env_vars.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        elif self.provider == "anthropic":
            self.api_key = env_vars.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        elif self.provider == "bedrock":
            self.api_key = None
        elif self.provider == "modal":
            self.api_key = env_vars.get("MODAL_API_KEY") or os.environ.get("MODAL_API_KEY")
        elif self.provider == "together":
            self.api_key = env_vars.get("TOGETHER_API_KEY") or os.environ.get("TOGETHER_API_KEY")
        elif self.provider == "deepseek":
            self.api_key = env_vars.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        elif self.provider == "opencode":
            self.api_key = env_vars.get("OPENCODE_API_KEY") or os.environ.get("OPENCODE_API_KEY")
        elif self.provider == "ai_sandbox":
            self.api_key = env_vars.get("AI_SANDBOX_KEY") or os.environ.get("AI_SANDBOX_KEY")
        else:
            self.api_key = env_vars.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")

        if not self.api_key and self.provider not in ("bedrock",):
            raise ValueError(f"No API key found for provider {self.provider}")

        if base_url:
            self.base_url = base_url
        elif self.provider == "xai":
            self.base_url = "https://api.x.ai/v1"
        elif self.provider == "google":
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        elif self.provider == "modal":
            self.base_url = os.environ.get("MODAL_BASE_URL")
        elif self.provider == "together":
            self.base_url = "https://api.together.xyz/v1"
        elif self.provider == "deepseek":
            self.base_url = "https://api.deepseek.com"
        elif self.provider == "opencode":
            self.base_url = "https://opencode.ai/zen/go/v1"
        else:
            self.base_url = None

        # Create client
        if self.provider == "bedrock":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package required for Bedrock")
            self.client = AnthropicBedrock(
                aws_access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
                aws_region=os.environ.get("AWS_REGION", "us-east-2"),
                base_url=self.base_url,
            )
        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package required")
            self.client = anthropic.Anthropic(api_key=self.api_key, base_url=self.base_url)
        elif self.provider == "ai_sandbox":
            try:
                from portkey_ai import Portkey
            except ImportError as e:
                raise ImportError(
                    "portkey-ai package required for ai_sandbox provider. "
                    "Install with: uv add portkey-ai"
                ) from e
            self.client = Portkey(api_key=self.api_key)
        else:
            import httpx
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            client_kwargs["timeout"] = httpx.Timeout(600.0)  # 10min max per LLM call; retry on timeout
            if self.provider == 'opencode':
                client_kwargs['default_headers'] = {
                    'User-Agent': 'CEO-Bench/1.0',
                    'x-opencode-session': str(uuid.uuid5(uuid.NAMESPACE_URL, str(self.workspace_dir))),
                }
            self.client = OpenAI(**client_kwargs)

        self.base_url = str(self.client.base_url)

        # Components (initialized in setup)
        self.agent = None
        self.tool_executor = None
        self._server_proc = None
        self._server_port = None
        self._session_id = None

    # =========================================================================
    # HTTP helpers — all simulation interaction goes through these
    # =========================================================================

    def _server_url(self, path: str) -> str:
        return f"http://127.0.0.1:{self._server_port}{path}"

    def _http_get(self, path: str, timeout: float = 30) -> Dict:
        req = urllib.request.Request(self._server_url(path))
        if path == '/dashboard' and self.evidence_store:
            from saas_bench.execution_capture import ExecutionCapture, CapturedText, origin
            capture = ExecutionCapture(self.evidence_store)
            capture.begin('dashboard_acquisition', {'path': path})
            if capture.event:
                req.add_header('X-Capture-Context', capture.store.context(capture.event))
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read()
                capture.blob('received', raw, 'program_body')
                result = json.loads(raw)
                text = result.get('dashboard', '')
                source = capture.store.load_state('dashboard') or {}
                from saas_bench.sql_evidence import digest
                if source.get('sha256') != digest(text.encode()):
                    raise ValueError('Dashboard receipt differs from captured generation')
                version = capture.blob('dashboard', text, 'dashboard', derived_from=source.get('version'))
                if version:
                    result['dashboard'] = CapturedText(text, [origin(version, text)] + source.get('origins', []))
                capture.safe(capture.store.complete, capture.event)
                return result
            except Exception as exc:
                capture.store.fail(exc)
                raise
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read())

    def _http_post(self, path: str, data: Optional[Dict] = None, timeout: float = 1800) -> Dict:
        body = json.dumps(data or {}).encode()
        headers = {'Content-Type': 'application/json'}
        if path in ('/checkpoint', '/pf-refresh', '/run-metrics'):
            headers['X-Harness-Token'] = self._checkpoint_token
        req = urllib.request.Request(
            self._server_url(path), data=body,
            headers=headers,
        )
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read())

    def _pf_refresh(self, versions, parent):
        from saas_bench.api_server import NovaMindAPIServer
        # Each source has its own timeout; this transport must let all sources finish.
        timeout = (len(versions) + 1) * NovaMindAPIServer.QUERY_TIMEOUT_SECONDS + 30
        return self._http_post('/pf-refresh', dict(versions=versions, parent=parent), timeout=timeout)['versions']

    def _get_cash(self) -> float:
        """Use the same status receipt as the run loop."""
        return float(self._get_game_status()['cash'])

    def _get_game_status(self) -> Dict:
        """Get game status (day, cash, subs, timeout) via HTTP."""
        result = self._http_get('/game-status')
        if result.get('operation_failed') or result.get('timed_out') or result.get('operation_in_progress'):
            raise RuntimeError('World operation outcome unknown; branch stopped')
        return result

    def _get_dashboard(self) -> str:
        """Get current dashboard via HTTP."""
        try:
            result = self._http_get('/dashboard')
            return result.get('dashboard', '')
        except Exception:
            return "(Dashboard unavailable)"

    def _advance_day_http(self) -> Dict:
        """Force week advancement via HTTP POST /next-week."""
        try:
            return self._http_post('/next-week', timeout=4200)
        except urllib.error.URLError as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Logging
    # =========================================================================

    def _log_response(self, turn: int, day: int, messages: List[Dict], raw_response: Any):
        entry = {
            "timestamp": now(),
            "turn": turn,
            "day": day,
            "messages": messages,
            "raw_response": raw_response,
        }
        with open(self.response_log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

    def _log_tool_result(self, turn: int, day: int, tool_name: str, arguments: Dict, result: str,
                         **metadata):
        tool_results_file = self.logs_dir / f"tool_results_{self.run_id}.jsonl"
        entry = {
            "timestamp": now(),
            "turn": turn,
            "day": day,
            "tool": tool_name,
            "arguments": arguments,
            "result": result,
        }
        entry.update(metadata)
        if hasattr(result, 'pf_calls'):
            entry['pf_calls'] = result.pf_calls
        if getattr(result, 'pf_call', None):
            entry['pf_call'] = result.pf_call
        with open(tool_results_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")

    def _log_timing(self, event: str, day: int, turn: int = 0, **kwargs):
        """Log a timing event to the timing JSONL file and push to dashboard."""
        entry = {
            "timestamp": now(),
            "run_id": self.run_id,
            "event": event,
            "day": day,
            "turn": turn,
            **kwargs,
        }
        with open(self.timing_log_file, 'a') as f:
            f.write(json.dumps(entry) + "\n")
        # Push to ceobench dashboard (non-blocking)
        if self._timing_queue is not None:
            try:
                self._timing_queue.put_nowait(entry)
            except Exception:
                pass

    # =========================================================================
    # Workspace setup
    # =========================================================================

    _GITIGNORE_CONTENT = """\
sessions/
_engine/
*.nmdb
*.db
*.db-journal
*.db-wal
*.db-shm
__pycache__/
*.pyc
.pytest_cache/
.venv/
"""

    def _git(self, *args: str, check: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=str(self.agent_workspace),
            capture_output=True, text=True,
            check=check,
        )

    def _git_init_workspace(self):
        if not (self.agent_workspace / ".git").exists():
            self._git("init", "-q", "-b", "main")
            self._git("config", "user.email", "bash-agent@bossbench.local")
            self._git("config", "user.name", "BashAgent")
        # Git commits must finish maintenance before the sandbox boundary closes.
        self._git("config", "gc.autoDetach", "false", check=True)
        self._git("config", "maintenance.autoDetach", "false", check=True)
        gitignore_path = self.agent_workspace / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text(self._GITIGNORE_CONTENT)

    def _git_commit_workspace(self, message: str, once_key: Optional[str] = None):
        if not (self.agent_workspace / ".git").exists():
            return
        if once_key is not None:
            existing = self._git("log", "--grep", f"[{once_key}]", "--fixed-strings", "--oneline")
            if existing.returncode == 0 and existing.stdout.strip():
                return
            message = f"{message} [{once_key}]"
        self._git("add", "-A")
        status = self._git("status", "--porcelain")
        if status.returncode == 0 and not status.stdout.strip():
            # Empty commit so the tag still lands on the timeline
            self._git("commit", "--allow-empty", "-q", "-m", message)
        else:
            self._git("commit", "-q", "-m", message)

    def _commit_weeks_up_to(self, sim_day: int):
        # Agent advances time via `./novamind-operation next-week ...`, which can cross
        # one or more sim-week boundaries inside a single harness loop iteration. The
        # once_key dedupe makes this safe to call after every sim_day update.
        if sim_day <= 0:
            return
        if not hasattr(self, '_last_committed_week'):
            self._last_committed_week = 0
        target_week = sim_day // 7
        while self._last_committed_week < target_week:
            self._last_committed_week += 1
            wd = self._last_committed_week * 7
            self._git_commit_workspace(
                f"Week {self._last_committed_week} (day {wd})",
                once_key=f"week-{self._last_committed_week}",
            )
            registry = getattr(getattr(self, 'tool_executor', None), 'text_registry', None)
            if registry:
                registry.finalize_week(f'week-{self._last_committed_week}')

    def _initialize_from_public_repo(self):
        """Copy the published layout into the agent workspace and create a session.

        Copy novamind-client as novamind-operation plus docs into the workspace.
        Create the session with the full host novamind-operation zipapp; its
        simulator bytecode and database decryption modules stay on the host.

        public/ must be built first via `uv run python scripts/build_public.py`.
        """
        import stat

        public_dir = self._public_dir()
        self.agent_workspace.mkdir(parents=True, exist_ok=True)
        self._git_init_workspace()

        # docs/ is the only directory the agent needs — it holds the tool/table
        # JSON, cli.md, examples/, and the readable SDK source at
        # docs/novamind_api/ (used for ``import novamind_api`` at runtime).
        src_docs = public_dir / "docs"
        dst_docs = self.agent_workspace / "docs"
        if src_docs.exists():
            if dst_docs.exists():
                shutil.rmtree(dst_docs)
            shutil.copytree(
                src_docs, dst_docs,
                ignore=shutil.ignore_patterns('__pycache__'),
            )

        # Only client code goes into the sandbox, including its initial Git tree.
        src_op = public_dir / "novamind-client"
        dst_op = self.agent_workspace / "novamind-operation"
        if not src_op.exists():
            raise FileNotFoundError(
                f"{src_op} does not exist. Did you run `uv run python scripts/build_public.py`?"
            )
        shutil.copy2(src_op, dst_op)
        dst_op.chmod(dst_op.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        # Create the per-session scratch root (daily_scripts/ inside it is
        # created later by the engine's initialize_workspace()).
        (self.agent_workspace / "daily_scripts").mkdir(exist_ok=True)

        # Run new-session via the HOST-SIDE zipapp (bytecode stays on host).
        env = self._server_environment()
        result = subprocess.run(
            [
                sys.executable, str(public_dir / "novamind-operation"),
                "--base", str(self.agent_workspace),
                "new-session",
                "--days", str(self.total_days),
                "--seed", str(self.seed),
                "--cash", str(self.initial_cash),
                "--scenario", self.scenario,
            ],
            capture_output=True, text=True, env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"novamind-operation new-session failed:\n{result.stderr}\n{result.stdout}"
            )

        session_info = json.loads(result.stdout)
        self._session_id = session_info["session_id"]
        print(f"  Session created via CLI: {self._session_id}")
        self._git_commit_workspace("Initial workspace setup (day 0)")
        return session_info

    def _public_dir(self) -> Path:
        """Location of the host-side public/ bundle (contains _engine/).

        The bash_agent run_test.py lives at:
            <root>/src/saas_bench/agents/bash_agent/run_test.py
        public/ lives at <root>/public/. parent^5 = <root>.

        Variant runs override this via the NOVAMIND_PUBLIC_DIR env var so they
        can launch with a per-variant zipapp (built locally, never pushed). The
        env var is read every call rather than cached so test fixtures can swap
        it on the fly.
        """
        override = os.environ.get("NOVAMIND_PUBLIC_DIR")
        if override:
            public_dir = Path(override).resolve()
            if not public_dir.exists():
                raise FileNotFoundError(
                    f"NOVAMIND_PUBLIC_DIR points to {public_dir} which does not exist."
                )
        else:
            public_dir = Path(__file__).parent.parent.parent.parent.parent / "public"
            if not public_dir.exists():
                raise FileNotFoundError(
                    f"public/ directory not found at {public_dir}. "
                    f"Run 'uv run python scripts/build_public.py' first."
                )
        return public_dir

    def _server_environment(self) -> Dict[str, str]:
        """Environment for host-side simulator processes."""
        import secrets
        if not getattr(self, '_checkpoint_token', None):
            self._checkpoint_token = secrets.token_urlsafe(32)
        env = os.environ.copy()
        env["NOVAMIND_SERVER_MODE"] = "1"
        env['CEOBENCH_RUN_MANIFEST'] = str(self.workspace_dir / 'manifest.json')
        env['CEOBENCH_RUN_KIND'] = self.run_kind
        from .tools import agent_runtime_dir
        env['CEOBENCH_AGENT_RUNTIME'] = str(agent_runtime_dir())
        env['CEOBENCH_CHECKPOINT_ROOT'] = str(self.workspace_dir / 'checkpoints')
        env['CEOBENCH_CHECKPOINT_TOKEN'] = self._checkpoint_token
        env['CEOBENCH_SIMULATOR_USAGE_LOG'] = str(self.logs_dir / 'simulator_requests.jsonl')
        env['CEOBENCH_MODEL_SESSION'] = str(uuid.uuid5(uuid.NAMESPACE_URL, str(self.workspace_dir))) + ':simulator'
        if self.sql_evidence_config:
            env['CEOBENCH_SQL_EVIDENCE'] = str(self.workspace_dir / 'sql-evidence.sqlite')
        else:
            env.pop('CEOBENCH_SQL_EVIDENCE', None)
        if getattr(self, '_restored_snapshot_dir', None):
            env['CEOBENCH_RESTORE_SERVER_STATE'] = str(self._restored_snapshot_dir / 'server_state.json')
        # Use the selected account for both roles unless explicitly overridden.
        if self.provider in ("deepseek", "opencode"):
            env.setdefault("CEOBENCH_SIMULATOR_LLM_PROVIDER", self.provider)
            env.setdefault("CEOBENCH_SIMULATOR_LLM_MODEL", self.model)
        return env

    def _prepare_manifest(self):
        from dataclasses import asdict
        from saas_bench.config import SCENARIO_PACKS, ScenarioPack
        from saas_bench.run_state import verify_build, write_json
        build = verify_build(self._public_dir(), root=package_root.parent)
        configuration = {key: getattr(self, key) for key in (
            'model', 'provider', 'base_url', 'seed', 'scenario', 'total_days',
            'initial_cash', 'reasoning_effort', 'anthropic_fallback_model', 'run_kind')}
        configuration['bedrock_region'] = os.environ.get('AWS_REGION', 'us-east-2')
        config = BenchmarkConfig(seed=self.seed, total_days=self.total_days, initial_cash=self.initial_cash)
        if self.pricing_registration:
            config.model_pricing = self.pricing_registration['rates']
        config.simulator_openai_base_url = os.environ.get('OPENAI_BASE_URL', config.simulator_openai_base_url)
        config.simulator_anthropic_base_url = os.environ.get('ANTHROPIC_BASE_URL', config.simulator_anthropic_base_url)
        env = self._server_environment()
        for suffix in ('provider', 'model'):
            value = env.get('CEOBENCH_SIMULATOR_LLM_' + suffix.upper())
            if value:
                for prefix in ('social_post_llm_', 'enterprise_llm_'):
                    setattr(config, prefix + suffix, value)
        manifest = dict(version=1, build=build, configuration=configuration,
                        benchmark_config=asdict(config), scenario_config=asdict(SCENARIO_PACKS.get(
                            self.scenario, ScenarioPack(name='Default', description='Balanced scenario'))))
        from .tools import GUEST_WORKSPACE, agent_runtime
        # Layout 2: fixed guest paths, dedicated agent runtime, sessions/ hidden.
        manifest['agent_sandbox'] = dict(layout=2, workspace=GUEST_WORKSPACE,
                                         runtime=agent_runtime(verify=True) if shutil.which('bwrap') else None)
        if self.sql_evidence_config:
            manifest['sql_evidence'] = self.sql_evidence_config
        if self.fork_source:
            manifest['fork_source'] = self.fork_source
        if self.recovery_source:
            manifest['recovery_source'] = self.recovery_source
        if self.text_registration != 'off':
            manifest['text_registration'] = self.text_registration
        if self.text_registration == 'pf':
            from saas_bench.payload_tokens import tokenizer_config
            manifest['pf_stale_checks'] = self.pf_stale_checks
            manifest['pf_read_tokenizer'] = tokenizer_config(self.provider, self.model)
        if self.pricing_registration:
            manifest['pricing'] = self.pricing_registration
        manifest = json.loads(json.dumps(manifest))
        self._pricing = manifest['benchmark_config']['model_pricing']
        path = self.workspace_dir / 'manifest.json'
        if self.continue_from:
            if json.loads(path.read_text()) != manifest:
                raise ValueError('Run manifest differs: artifacts or effective configuration changed')
        else:
            write_json(path, manifest)

    def _launch_server(self):
        """Launch the host-side novamind-operation zipapp in server mode.

        The zipapp lives in public/ only. Setting NOVAMIND_SERVER_MODE=1 makes
        its __main__ dispatch to saas_bench.server_entry.main() instead of the
        client-side CLI. The server process runs in the parent environment
        (outside bwrap) so it has access to NMDB_KEY.

        Reads the first line of stdout to get the port, then waits for /health.
        """
        zipapp_path = self._public_dir() / "novamind-operation"
        server_env = self._server_environment()
        # Route api_server stderr to a file rather than a pipe back to bash_agent.
        # bash_agent never drains the pipe during a /call, so a single buffered
        # traceback (>64KB pipe capacity) wedges write() under self._lock and
        # deadlocks every subsequent /call. (run 27c000a5 d105 hang.)
        self._server_stderr_path = self.logs_dir / "api_server_stderr.log"
        self._server_stderr_file = open(self._server_stderr_path, "ab", buffering=0)
        self._server_proc = subprocess.Popen(
            [
                sys.executable, str(zipapp_path),
                "--base", str(self.agent_workspace),
                "start-server",
                "--session", self._session_id,
            ],
            stdout=subprocess.PIPE,
            stderr=self._server_stderr_file,
            env=server_env,
        )

        # Read first line of stdout to get port info
        first_line = self._server_proc.stdout.readline()
        if not first_line:
            try:
                stderr_tail = self._server_stderr_path.read_bytes()[-4096:]
            except Exception:
                stderr_tail = b"<stderr log unavailable>"
            raise RuntimeError(f"Server failed to start:\n{stderr_tail.decode(errors='replace')}")

        server_info = json.loads(first_line)
        self._server_port = server_info["port"]
        print(f"  Server started: port={self._server_port}, pid={server_info['pid']}")

        # Wait for health check
        for i in range(60):
            try:
                self._http_get('/health', timeout=2)
                return
            except Exception:
                _time.sleep(0.5)

        raise RuntimeError("Server did not respond to /health after 30s")

    def _stop_server(self):
        """Stop the server subprocess."""
        if self._server_proc:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
            self._server_proc = None
        f = getattr(self, "_server_stderr_file", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
            self._server_stderr_file = None

    # =========================================================================
    # Checkpoint
    # =========================================================================

    def _check_tamper(self, day: int) -> List[str]:
        """Scan agent workspace for sandbox-escape indicators.

        Looks for files that an agent has no legitimate reason to create —
        primarily duplicate `*.nmdb` files in `sessions/<sid>/` (the engine
        only writes `world.nmdb`; anything else is a backup the agent made
        before tampering).

        Returns a list of suspicious file paths (relative to agent_workspace).
        Empty list = clean.

        Reference: gpt55 v3.4aa run 1267c284 (2026-04-28) created
        `world_before_week31_recovery_patch.nmdb` before running UPDATE
        statements directly against the decrypted DB.
        """
        flagged: List[str] = []
        sessions_dir = self.agent_workspace / "sessions"
        if sessions_dir.exists():
            for session_path in sessions_dir.iterdir():
                if not session_path.is_dir():
                    continue
                for nmdb in session_path.glob("*.nmdb"):
                    if nmdb.name != "world.nmdb":
                        flagged.append(str(nmdb.relative_to(self.agent_workspace)))
        # Anything matching `patch_*.py` or `recover_*.py` at workspace root
        # is also a strong signal of tamper attempts (gpt55 named its script
        # `patch_world_day217_cleanup.py`).
        for suspicious in self.agent_workspace.glob("patch_*.py"):
            flagged.append(str(suspicious.relative_to(self.agent_workspace)))
        for suspicious in self.agent_workspace.glob("recover_*.py"):
            flagged.append(str(suspicious.relative_to(self.agent_workspace)))
        return flagged

    def _wait_checkpoint(self):
        pending = getattr(self, '_checkpoint_future', None)
        if pending is not None:
            self._checkpoint_future = None
            started = _time.monotonic()
            try:
                pending.result()
            finally:
                self._log_timing('checkpoint_wait', self.agent.current_day if self.agent else 0,
                                 elapsed_s=_time.monotonic()-started)

    def _save_checkpoint(self, day: int, fetch_daily_scripts: bool = True, *, wait=True):
        """Freeze while quiescent; one worker publishes only fully verified generations."""
        from concurrent.futures import ThreadPoolExecutor
        from saas_bench.run_state import copy_workspace, file_hash, tree_hash, write_json
        from saas_bench.db_protection import encrypt_plain_atomic
        self._wait_checkpoint()  # At most one immutable generation is being processed.
        if self.tool_executor.text_registry:
            self.tool_executor.text_registry.assert_week_finalized(day)
        started = _time.monotonic()
        if self.agent and self.agent._pending_tool_calls:
            raise RuntimeError('Cannot checkpoint a tool with an unknown outcome')
        if self.agent:
            self.agent._save_conversation_snapshot(strict=True)
        receipt = self._http_post('/checkpoint', {'expected_day': day}, timeout=600)
        if not receipt.get('success') or receipt.get('day') != day:
            raise RuntimeError('Server did not acknowledge the requested checkpoint')
        snapshot_id = receipt['snapshot_id']
        if len(snapshot_id) != 32 or any(c not in '0123456789abcdef' for c in snapshot_id):
            raise ValueError('Invalid server snapshot identifier')
        directory = self.workspace_dir / 'checkpoints' / snapshot_id
        if receipt.get('phase') != 'frozen':
            raise ValueError('Server did not freeze a checkpoint generation')
        shutil.copy2(self.workspace_dir / 'manifest.json', directory / 'manifest.json')
        copying = _time.monotonic()
        copy_workspace(self.agent_workspace, directory / 'agent_workspace', omit_session_world=self._session_id)
        workspace_copy_s = _time.monotonic()-copying
        # Open descriptors and exact cutoffs bind append-only logs without copying them
        # on the model's critical path. A later append cannot enter this generation.
        request_logs = {}
        for name in ('agent_requests.jsonl', 'simulator_requests.jsonl'):
            source = self.logs_dir / name
            if source.exists():
                stream = source.open('rb')
                request_logs[name] = (stream, os.fstat(stream.fileno()).st_size)
        checkpoint = dict(version=2, day=day, run_id=self.run_id,
                          session_id=self._session_id, snapshot_id=snapshot_id,
                          files={}, request_logs={},
                          request_log_cutoffs={n: size for n, (_, size) in request_logs.items()})
        if self.sql_evidence_config:
            evidence = receipt.get('sql_evidence')
            if not evidence or evidence['identity'] != self.sql_evidence_config:
                raise ValueError('Missing or mismatched SQL evidence checkpoint')
            checkpoint['sql_evidence'] = evidence
        checkpoint['context_boundary'] = 'same_week' if self.agent and self.agent.current_day == day else 'new_week'
        progress = self.workspace_dir / 'operation.json'
        checkpoint['operation_id'] = json.loads(progress.read_text())['id'] if progress.exists() else None
        checkpoint['usage'] = json.loads(json.dumps(self.agent.usage_recorder.summary)) if self.agent else None
        if getattr(self, '_last_weekly_day', None) != day:
            write_json(self.workspace_dir / 'usage_summary.json', {
                'snapshot_id': snapshot_id, 'day': day, 'agent': checkpoint['usage'],
                'simulator': json.loads((directory / 'server_state.json').read_text())['usage']})
        for field in ('total_turns', 'total_input_tokens', 'total_output_tokens',
                      'total_cached_tokens', 'total_cache_creation_tokens', 'total_reasoning_tokens', 'total_anthropic_fallbacks'):
            checkpoint[field] = getattr(self.agent, field, 0)
        frozen = _time.monotonic()
        self._log_timing('checkpoint_freeze', day, elapsed_s=round(frozen-started, 3), snapshot_id=snapshot_id,
                         workspace_copy_s=workspace_copy_s, **receipt['timings'])

        def publish():
            try:
                plain = directory / 'world.plain.sqlite'
                encrypt_plain_atomic(plain, directory / 'world.nmdb')
                plain.unlink()
                encrypted = _time.monotonic()
                for name in [*receipt['files'], 'world.nmdb', 'manifest.json']:
                    checkpoint['files'][name] = file_hash(directory / name)
                for name, (stream, remaining) in request_logs.items():
                    target = directory / 'request_logs' / name
                    target.parent.mkdir(exist_ok=True)
                    with target.open('wb') as output:
                        while remaining:
                            data = stream.read(min(1024 * 1024, remaining))
                            if not data:
                                raise ValueError('Request log ended before frozen cutoff')
                            output.write(data)
                            remaining -= len(data)
                    checkpoint['request_logs'][name] = file_hash(target)
                copied = _time.monotonic()
                checkpoint['workspace_sha256'] = tree_hash(directory / 'agent_workspace')
                if self.sql_evidence_config:
                    checkpoint['sql_evidence']['sha256'] = file_hash(directory / 'sql-evidence.sqlite')
                write_json(directory / 'checkpoint.json', checkpoint)
                write_json(self.workspace_dir / 'checkpoint.json', checkpoint)
                self._log_timing('checkpoint_publish', day, elapsed_s=round(_time.monotonic()-frozen, 3),
                                 snapshot_id=snapshot_id, encryption_s=encrypted-frozen,
                                 log_copy_and_file_hash_s=copied-encrypted,
                                 workspace_and_evidence_hash_s=_time.monotonic()-copied)
            finally:
                for stream, _ in request_logs.values():
                    stream.close()

        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='checkpoint')
        self._checkpoint_future = pool.submit(publish)
        pool.shutdown(wait=False)
        if wait:
            self._wait_checkpoint()

    def _restore_checkpoint_files(self, checkpoint):
        from saas_bench.run_state import checkpoint_directory, checkpoint_manifest, copy_workspace, restore_sql_evidence
        directory = checkpoint_directory(self.workspace_dir, checkpoint)
        checkpoint_manifest(self.workspace_dir, directory)
        if self.sql_evidence_config:
            restore_sql_evidence(self.workspace_dir, directory, checkpoint, self.sql_evidence_config)
            controls = directory / 'sql-evidence.controls.jsonl'
            if controls.exists() and not (self.workspace_dir / controls.name).exists():
                shutil.copy2(controls, self.workspace_dir / controls.name)
        staged = self.workspace_dir / ('restore-' + uuid.uuid4().hex)
        copy_workspace(directory / 'agent_workspace', staged)
        self._session_id = checkpoint['session_id']
        session = staged / 'sessions' / self._session_id
        session.mkdir(parents=True, exist_ok=True)
        shutil.copy2(directory / 'world.nmdb', session / 'world.nmdb')
        shutil.copy2(directory / 'session.json', session / 'session.json')
        previous = self.workspace_dir / ('previous-' + uuid.uuid4().hex)
        if self.agent_workspace.exists():
            os.replace(self.agent_workspace, previous)
        try:
            os.replace(staged, self.agent_workspace)
        except Exception:
            if previous.exists():
                os.replace(previous, self.agent_workspace)
            raise
        if previous.exists():
            shutil.rmtree(previous)
        for name in checkpoint['request_logs']:
            shutil.copy2(directory / 'request_logs' / name, self.logs_dir / name)
        self._restored_snapshot_dir = directory

    def _load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint from disk."""
        checkpoint_file = self.workspace_dir / "checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                checkpoint = json.load(f)
            progress = self.workspace_dir / 'operation.json'
            if progress.exists() and json.loads(progress.read_text())['id'] != checkpoint.get('operation_id'):
                raise ValueError('Uncheckpointed operation outcome unknown; use scripts/recover_run.py to preserve this run and start a new attempt')
            return checkpoint
        return None

    def _begin_operation(self, kind, day):
        from saas_bench.run_state import write_json
        self._check_capture_health()
        write_json(self.workspace_dir / 'operation.json', {'id': uuid.uuid4().hex, 'kind': kind, 'day': day})

    def _weekly_record(self, status):
        from saas_bench.run_state import write_json
        metrics = self._http_post('/run-metrics', {})
        if metrics['day'] != status['day']:
            raise ValueError('Weekly metrics day differs from world')
        record = dict(status, agent=self.agent.usage_recorder.summary, simulator=metrics['usage'],
                      model_turns=self.agent.total_turns)
        if self.recovery_source:
            def difference(current, baseline):
                return {k: difference(v, baseline.get(k, {})) if isinstance(v, dict) else
                        v - (baseline.get(k) or 0) if type(v) in (int, float) else v for k, v in current.items()}
            record['attempt_usage'] = {role: difference(record[role], self.recovery_source['usage'][role])
                                       for role in ('agent', 'simulator')}
            record['attempt_id'] = self.recovery_source['attempt_id']
        with (self.workspace_dir / 'weekly.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        write_json(self.workspace_dir / 'usage_summary.json', record)
        self._last_weekly_day = status['day']

    def _check_capture_health(self):
        if self.evidence_store:
            self.evidence_store.assert_healthy(settle=5)
        if self.sql_evidence_config:
            if ((self.workspace_dir / 'sql-evidence.fault.json').exists() or
                    self._http_get('/health').get('status') != 'ok'):
                raise RuntimeError('SQL evidence capture failed; collection stopped')

    def _restore_from_checkpoint(self, checkpoint: Dict):
        """Restore agent accounting after the server has loaded the saved world."""
        if self.agent:
            for field in ('total_turns', 'total_input_tokens', 'total_output_tokens',
                          'total_cached_tokens', 'total_cache_creation_tokens', 'total_reasoning_tokens', 'total_anthropic_fallbacks'):
                setattr(self.agent, field, checkpoint[field])
            self.agent.usage_recorder.summary = checkpoint['usage']
            if checkpoint['context_boundary'] == 'same_week':
                if not self.agent.load_conversation_snapshot(self.agent._snapshot_path):
                    raise ValueError('Cannot restore the checkpoint conversation')
            else:
                self.agent.reset()
        self._suppress_force_step_day_once = True

    # =========================================================================
    # Setup
    # =========================================================================

    def setup(self):
        """Initialize the simulation environment.

        Flow:
        1. Copy public/ into workspace and create session via host-side CLI
        2. Launch host-side 'novamind-server start-server' as subprocess
        3. Create agent and tool executor (communicate via HTTP)

        The simulator bytecode (_engine/) and server launcher NEVER enter the
        workspace — they stay in public/ on the host side.
        """
        from .agent import BashAgent
        from .tools import get_bash_agent_tool_descriptions, BashAgentToolExecutor, NextDayTimeoutError
        if self.run_kind == 'formal':
            BashAgentToolExecutor(self.agent_workspace, require_sandbox=True).verify_sandbox()
            if os.environ.get('BOSSBENCH_LLM_REPLAY_DB') or os.environ.get('ORACLE_MODE') == '1':
                raise ValueError('Formal runs cannot enable replay or oracle mode')
        self._prepare_manifest()
        self.payload_token_counter = None
        if self.text_registration == 'pf':
            from saas_bench.payload_tokens import load_counter
            self.payload_token_counter = load_counter(self.provider, self.model)
        self._NextDayTimeoutError = NextDayTimeoutError

        # ── Step 1: Copy public/ and create session via CLI ──
        if not self.continue_from:
            session_info = self._initialize_from_public_repo()
        else:
            # Resuming — session already exists, find it
            checkpoint = self._load_checkpoint()
            if checkpoint:
                self._restore_checkpoint_files(checkpoint)
            else:
                raise ValueError('Cannot resume without a complete checkpoint')
        if not self._session_id:
            raise RuntimeError("No session ID found. Cannot proceed.")

        # ── Step 2: Launch server subprocess ──
        self._launch_server()

        if self.execution_capture:
            from saas_bench.sql_evidence import SQLEvidenceStore
            self.evidence_store = SQLEvidenceStore(self.workspace_dir / 'sql-evidence.sqlite', self.sql_evidence_config)

        registry = None
        if self.text_registration != 'off':
            from saas_bench.text_registry import TextRegistry
            registry = TextRegistry(self.agent_workspace, self.text_registration, self.evidence_store,
                                    sim_day=lambda: self.agent.current_day)

        # ── Step 3: Create tool executor + agent ──
        # Pass NOVAMIND_API_PORT so the CLI (./novamind-operation) connects to
        # the already-running server instead of trying to start a new one.
        self.tool_executor = BashAgentToolExecutor(
            workspace_path=self.agent_workspace,
            env={"NOVAMIND_API_PORT": str(self._server_port)},
            require_sandbox=self.run_kind == 'formal', stop_on_timeout=True,
            evidence_store=self.evidence_store,
            text_registry=registry,
            pf_stale_checks=self.pf_stale_checks, pf_refresh=self._pf_refresh,
        )

        tool_descriptions = get_bash_agent_tool_descriptions(registry is not None, self.text_registration == 'pf')
        from saas_bench.model_usage import ModelUsage

        self.agent = BashAgent(
            tool_descriptions=tool_descriptions,
            client=self.client,
            model=self.model,
            max_turns_per_day=0,  # No limit
            response_callback=self._log_response,
            reasoning_effort=self.reasoning_effort,
            tool_result_callback=self._log_tool_result,
            workspace_path=self.agent_workspace,
            total_days=self.total_days,
            anthropic_fallback_model=self.anthropic_fallback_model,
            usage_recorder=ModelUsage(self.logs_dir / 'agent_requests.jsonl', 'agent', self._pricing,
                                      self.evidence_store, self.payload_token_counter),
            text_registration=registry is not None,
            pf=self.text_registration == 'pf',
        )

        # Wire the per-session conversation snapshot path. The agent writes
        # this after every LLM call; on resume, _restore_from_checkpoint can
        # load it to recover the exact accumulated context (see agent.py).
        self.agent._snapshot_path = (
            self.agent_workspace / "sessions" / self._session_id / "conversation.json"
        )

        # Save run config
        config = {
            'run_id': self.run_id,
            'model': self.model,
            'provider': self.provider,
            'reasoning_effort': self.reasoning_effort,
            'anthropic_fallback_model': self.anthropic_fallback_model,
            'seed': self.seed,
            'scenario': self.scenario,
            'total_days': self.total_days,
            'initial_cash': self.initial_cash,
            'agent_type': 'bash_agent',
            'api_server_port': self._server_port,
            'session_id': self._session_id,
            'label': self.label,
            'public_dir_override': os.environ.get('NOVAMIND_PUBLIC_DIR') or None,
            'text_registration': self.text_registration,
            'pf_stale_checks': self.pf_stale_checks,
        }
        with open(self.workspace_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a bash_agent tool.

        Raises NextDayTimeoutError if ./novamind-operation next-week times out,
        which triggers run checkpoint + kill in the run loop.
        """
        recorder = self.agent.usage_recorder
        origin = (dict(model_request_event=recorder.last_request_event, model_context_id=recorder.context_id)
                  if recorder.last_request_event else {})
        result = self.tool_executor.execute(tool_name, arguments, **origin)

        # Check if bash output contains a day advancement
        if tool_name == 'bash':
            self.agent.check_day_advanced(result)

        return result

    # =========================================================================
    # Main run loop
    # =========================================================================

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """Stop the branch on any unknown outcome; never publish partial state."""
        from saas_bench.run_state import write_json
        try:
            return self._run(verbose)
        except BaseException as exc:
            write_json(self.workspace_dir / 'branch_stop.json', {'error': type(exc).__name__, 'reason': str(exc)})
            raise
        finally:
            fault_path = self.workspace_dir / 'sql-evidence.fault.json'
            preserve = bool(getattr(self.tool_executor, 'preserved_process', None))
            if fault_path.exists():
                preserve = preserve or json.loads(fault_path.read_text()).get('preserve_scene', False)
            try:
                self._wait_checkpoint()
            except BaseException as exc:
                write_json(self.workspace_dir / 'checkpoint_error.json', {'error': type(exc).__name__, 'reason': str(exc)})
                raise
            finally:
                if not preserve:
                    self._stop_server()

    def _run(self, verbose=True):
        self.setup()
        if not self.continue_from and not (self.workspace_dir / 'checkpoint.json').exists():
            self._save_checkpoint(self._get_game_status()['day'])

        start_day = 1
        if self.continue_from:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                start_day = checkpoint['day'] + 1
                self._restore_from_checkpoint(checkpoint)

                # Check for bankruptcy before resuming
                cash = self._get_cash()
                if cash < 0:
                    print(f"\n{'='*60}")
                    print(f"CANNOT RESUME — COMPANY IS BANKRUPT")
                    print(f"Run ID: {self.run_id}")
                    print(f"Checkpoint: Day {checkpoint['day']}")
                    print(f"Cash balance: ${cash:,.2f}")
                    print(f"{'='*60}\n")
                    raise SystemExit(f"Run {self.run_id} is bankrupt (cash=${cash:,.2f}). Cannot resume.")

                if verbose:
                    print(f"\n{'='*60}")
                    print(f"RESUMING Bash Agent Run from Day {start_day}")
                    print(f"Run ID: {self.run_id}")
                    print(f"Model: {self.model}")
                    print(f"Checkpoint: Day {checkpoint['day']}")
                    print(f"Cash balance: ${cash:,.2f}")
                    print(f"Workspace: {self.workspace_dir}")
                    print(f"{'='*60}\n")
            else:
                print(f"WARNING: No checkpoint found, starting from Day 1")

        if start_day == 1 and verbose:
            print(f"\n{'='*60}")
            print(f"Starting Bash Agent Run")
            print(f"Run ID: {self.run_id}")
            print(f"Model: {self.model}")
            print(f"Provider: {self.provider}")
            print(f"Seed: {self.seed}")
            print(f"API Server Port: {self._server_port}")
            print(f"Agent Workspace: {self.agent_workspace}")
            print(f"Workspace: {self.workspace_dir}")
            print(f"{'='*60}\n")

        current_day = start_day - 1
        game_ended = False
        game_outcome = None
        sim_day = current_day
        _cash = 0
        last_status: Dict[str, Any] = {}

        for day in range(start_day, self.total_days + 1):
            pending = getattr(self, '_checkpoint_future', None)
            if pending is not None and pending.done():
                self._wait_checkpoint()
            _day_start = _time.monotonic()
            current_day = day

            # Get actual simulation day from server (may differ from harness loop counter
            # when agent uses next-week which advances 7 sim days per loop iteration)
            status = self._get_game_status()
            last_status = status
            sim_day = status.get('day', day)

            if verbose:
                print(f"\n{'='*40}")
                print(f"DAY {day} (sim day {sim_day})")
                print(f"{'='*40}")

            # Build dashboard (timed)
            _t0 = _time.monotonic()
            dashboard = self._get_dashboard()
            _dashboard_elapsed = _time.monotonic() - _t0
            self._log_tool_result(0, sim_day, '_dashboard', {}, dashboard)
            self._log_timing("dashboard", sim_day, elapsed_s=round(_dashboard_elapsed, 3))

            # Agent loop for this day
            resumed = getattr(self.agent, '_observation_recorded', False) and self.agent.current_day == sim_day
            observation = self.agent._last_observation if resumed else dashboard
            if not resumed:
                # A new week's context: registered texts whose cited evidence changed follow the dashboard.
                _t0 = _time.monotonic()
                check = self.tool_executor.weekly_check(sim_day)
                if check:
                    self._log_tool_result(0, sim_day, '_weekly_check', {}, check)
                    self._log_timing("weekly_check", sim_day, elapsed_s=round(_time.monotonic() - _t0, 3))
                    observation = _joined(dashboard, '\n\n', check)
                observation = _joined(observation, '\n\n',
                    'Forecast units: All 12 cash forecast values for next-week are USD. '
                    'For 20.7 million USD, submit 20700000 or 20.7e6.')
            info = {'day': sim_day, 'cash': status['cash']}
            turns_today = 0
            day_ended = False
            _day_llm_total = 0.0
            _day_tool_total = 0.0
            _day_usage_before = copy.deepcopy(self.agent.usage_recorder.summary)

            while not day_ended and not game_ended and turns_today < 100:
                turns_today += 1

                # LLM call (timed)
                _t0 = _time.monotonic()
                self._begin_operation('model_and_tool', sim_day)
                _usage_before = copy.deepcopy(self.agent.usage_recorder.summary)
                action = self.agent.act(observation, 0, False, info)
                _llm_elapsed = _time.monotonic() - _t0
                _day_llm_total += _llm_elapsed
                # One act() can regenerate several responses before accepting a tool batch.
                _usage = usage_delta(_usage_before, self.agent.usage_recorder.summary)

                if action is None:
                    # With the agent's retry-with-feedback loop, _call_* should no
                    # longer return None. If we still get here, something is very
                    # wrong — raise so the run fails loudly instead of silently
                    # spamming a broken next-week command.
                    raise RuntimeError(
                        "Agent.act() returned None despite retry-with-feedback loop. "
                        "This indicates a bug in the agent scaffold — please investigate."
                    )

                tool_name = action.tool
                tool_args_preview = ""
                if tool_name == 'bash':
                    tool_args_preview = (action.arguments or {}).get('command', '')[:120]
                else:
                    tool_args_preview = json.dumps(action.arguments or {})[:120]

                self._log_timing("llm_call", sim_day, turn=turns_today,
                                 elapsed_s=round(_llm_elapsed, 2),
                                 tool=tool_name, tool_preview=tool_args_preview,
                                 **_usage['known'], model_calls=_usage['calls'],
                                 usage_missing=_usage['missing'],
                                 requested_model=self.model,
                                 served_model=self.agent.last_serving_model,
                                 anthropic_fallback_used=self.agent.last_anthropic_fallback_used,
                                 anthropic_fallbacks=self.agent.last_anthropic_fallbacks,
                                 total_anthropic_fallbacks=self.agent.total_anthropic_fallbacks)

                batch_size = max(1, len(self.agent._pending_tool_calls))
                self._log_timing('model_tool_batch', sim_day, turn=turns_today, calls=batch_size)
                while action is not None:
                    tool_name = action.tool
                    tool_args_preview = str(action.arguments or {})[:120]
                    # Execute action (timed)
                    if verbose:
                        if tool_name == 'bash':
                            print(f"    [Turn {turns_today}] bash: {tool_args_preview[:100]}")
                        else:
                            print(f"    [Turn {turns_today}] {tool_name}({tool_args_preview[:100]})")

                    _t0 = _time.monotonic()
                    try:
                        result = self._execute_tool(action.tool, action.arguments or {})
                    except self._NextDayTimeoutError as e:
                        _tool_elapsed = _time.monotonic() - _t0
                        print(f"\n⚠️  {tool_name} stopped on sim day {sim_day} ({e})")
                        raise RuntimeError(f'{e}; branch stopped') from e
                    _tool_elapsed = _time.monotonic() - _t0
                    _day_tool_total += _tool_elapsed
                    observation = result if isinstance(result, str) else json.dumps(result)
                    call_id = self.agent._pending_tool_calls[0]['id'] if self.agent._pending_tool_calls else None
                    self.agent.record_tool_result(observation)

                    self._log_timing("tool_exec", sim_day, turn=turns_today,
                                     elapsed_s=round(_tool_elapsed, 3),
                                     tool=tool_name, tool_preview=tool_args_preview)

                    # Log tool result
                    self._log_tool_result(
                        self.agent.total_turns, sim_day,
                        action.tool, action.arguments or {},
                        observation, call_id=call_id
                    )
                    self._check_capture_health()

                    if verbose:
                        print(f"      → {observation[:200]}")
                        print(f"      ⏱ llm={_llm_elapsed:.1f}s tool={_tool_elapsed:.1f}s")

                    # Check if the agent detected a day advancement
                    if self.agent.day_advanced:
                        day_ended = True
                        self.agent.clear_day_advanced()

                    # Check server for timeout (via game-status)
                    status = self._get_game_status()
                    last_status = status
                    # Shell pipelines can trim the dashboard header; the server day
                    # still ends the week and must trigger checkpoint/stop handling.
                    if status.get('day', sim_day) > sim_day:
                        day_ended = True
                    sim_day = status.get('day', sim_day)  # Update sim_day after potential next-week
                    self._commit_weeks_up_to(sim_day)  # Commit any sim-week boundary just crossed

                    # Check if simulation reached total_days (inside inner loop)
                    if sim_day >= self.total_days:
                        game_ended = True
                        game_outcome = 'completed'
                        if verbose:
                            print(f"\n✅ Simulation reached {sim_day} days (target: {self.total_days})")

                    if status.get('timed_out'):
                        print(f"\n⚠️  step_day timed out on sim day {sim_day}")
                        raise RuntimeError('Operation outcome unknown after timeout; branch stopped')

                    _cash_inner = status.get('cash', 0)
                    info = {'day': sim_day, 'cash': _cash_inner}

                    # Check bankruptcy inside inner loop (don't let agent keep playing while bankrupt)
                    if _cash_inner < 0:
                        game_ended = True
                        game_outcome = 'bankrupt'
                        if verbose:
                            print(f"\n💀 BANKRUPT at sim day {sim_day} (cash=${_cash_inner:,.0f})!")

                    if day_ended or game_ended:
                        reason = ('Cancelled: simulation advanced to a new week; read the new state before deciding.'
                                  if day_ended else 'Cancelled: the simulation ended.')
                        while self.agent._pending_tool_calls:
                            pending = self.agent._pending_tool_calls[0]
                            self._log_tool_result(self.agent.total_turns, sim_day, pending['name'],
                                                  pending.get('arguments', {}), reason,
                                                  call_id=pending['id'], outcome='cancelled')
                            self.agent.record_tool_result(reason)
                        break
                    action = self.agent.next_tool_action()

            if game_ended:
                break

            # If the agent did not call next-week within this harness chunk, do
            # not fabricate a /next-week call. The API requires the same
            # rationale/prediction payload that the agent-facing CLI collects.
            # Continuing at the same sim day preserves benchmark semantics and
            # avoids noisy HTTP 400s from an empty forced advance.
            # Exception: on the very first outer iteration after a mid-day
            # resume (agent's last logged tool was NOT next-week), suppress
            # the warning once so the agent can keep planning. Flag is cleared
            # after one iteration so subsequent days behave normally.
            _step_elapsed = 0
            if not day_ended:
                if self._suppress_force_step_day_once:
                    print(f"  [resume] Skipping force step_day on resume iter (last tool was not next-week)")
                    self._suppress_force_step_day_once = False
                else:
                    print(
                        f"\n⚠️  Turn cap reached on sim day {sim_day} without next-week; "
                        f"continuing at the same sim day."
                    )
                    self._log_timing("turn_cap_no_advance", sim_day, turns=turns_today)

            # Log step_day timing
            self._log_timing("step_day", sim_day, elapsed_s=round(_step_elapsed, 2))

            # Log slow step_day as warning
            if _step_elapsed > 300:
                print(f"\n⚠️  step_day took {_step_elapsed:.1f}s on sim day {sim_day} (>300s) — continuing")

            # Get post-day status (also refresh sim_day)
            status = self._get_game_status()
            last_status = status
            sim_day = status.get('day', sim_day)
            self._commit_weeks_up_to(sim_day)  # Commit any sim-week boundary crossed by step_day
            _subs = status.get('subscribers', 0)
            _cash = status.get('cash', 0)

            # Check if simulation reached total_days
            if sim_day >= self.total_days:
                game_ended = True
                game_outcome = 'completed'
                if verbose:
                    print(f"\n✅ Simulation reached {sim_day} days (target: {self.total_days})")
                break

            # Per-day timing summary
            _day_elapsed = _time.monotonic() - _day_start
            _day_other = _day_elapsed - _day_llm_total - _day_tool_total - _step_elapsed - _dashboard_elapsed
            _day_usage = usage_delta(_day_usage_before, self.agent.usage_recorder.summary)
            _day_input_tokens = _day_usage['known']['input_tokens'] or 0
            _day_output_tokens = _day_usage['known']['output_tokens'] or 0
            _day_cached_tokens = _day_usage['known']['cached_tokens'] or 0
            _day_reasoning_tokens = _day_usage['known']['reasoning_tokens'] or 0
            self._log_timing("day_summary", sim_day,
                             elapsed_s=round(_day_elapsed, 1),
                             llm_total_s=round(_day_llm_total, 1),
                             tool_total_s=round(_day_tool_total, 1),
                             step_day_s=round(_step_elapsed, 1),
                             dashboard_s=round(_dashboard_elapsed, 2),
                             other_s=round(max(_day_other, 0), 1),
                             turns=turns_today,
                             subs=_subs,
                             cash=_cash,
                             **{'day_' + field: value for field, value in _day_usage['known'].items()},
                             day_model_calls=_day_usage['calls'], day_usage_missing=_day_usage['missing'],
                             total_input_tokens=self.agent.total_input_tokens,
                             total_output_tokens=self.agent.total_output_tokens,
                             total_cached_tokens=self.agent.total_cached_tokens,
                             total_reasoning_tokens=self.agent.total_reasoning_tokens)

            # Print per-day timing summary to stderr (visible in logs)
            _pct_llm = (_day_llm_total / _day_elapsed * 100) if _day_elapsed > 0 else 0
            _pct_step = (_step_elapsed / _day_elapsed * 100) if _day_elapsed > 0 else 0
            _pct_tool = (_day_tool_total / _day_elapsed * 100) if _day_elapsed > 0 else 0
            _cache_pct = (_day_cached_tokens / _day_input_tokens * 100) if _day_input_tokens > 0 else 0
            print(f"\n⏱ DAY {sim_day} TIMING: total={_day_elapsed:.0f}s | "
                  f"llm={_day_llm_total:.0f}s ({_pct_llm:.0f}%) | "
                  f"step_day={_step_elapsed:.0f}s ({_pct_step:.0f}%) | "
                  f"tools={_day_tool_total:.0f}s ({_pct_tool:.0f}%) | "
                  f"dashboard={_dashboard_elapsed:.1f}s | "
                  f"turns={turns_today} | "
                  f"tokens={_day_input_tokens:,}in/{_day_output_tokens:,}out "
                  f"cached={_day_cached_tokens:,}({_cache_pct:.0f}%) "
                  f"reasoning={_day_reasoning_tokens:,} "
                  f"(cumul: {self.agent.total_input_tokens:,}in/{self.agent.total_output_tokens:,}out)",
                  file=sys.stderr, flush=True)

            if verbose:
                print(f"  📊 End of day: Cash=${_cash:,.0f}, Subs={_subs}")

            # Weekly git commit of agent workspace (before checkpoint so resume re-tries)
            # Idempotent via once_key — _commit_weeks_up_to may have already committed this week.
            self._commit_weeks_up_to(sim_day)

            self._weekly_record(status)
            if self.stop_after_day is not None and sim_day >= self.stop_after_day and _cash >= 0:
                self._save_checkpoint(sim_day)
                checkpoint = self._load_checkpoint()
                if sim_day != self.stop_after_day or checkpoint['context_boundary'] != 'new_week':
                    raise RuntimeError('Harness stop did not land on the requested week boundary')
                game_ended = True
                game_outcome = 'stopped'
                if verbose:
                    print(f"\n⏸ Harness stop at sim day {sim_day} (configured end: {self.total_days})")
                break

            if sim_day % 35 == 0 and _cash >= 0 and not game_ended:
                self._save_checkpoint(sim_day, wait=False)

            # Check bankruptcy
            if _cash < 0:
                game_ended = True
                game_outcome = 'bankrupt'
                if verbose:
                    print(f"\n💀 BANKRUPT at sim day {sim_day}!")
                break

        if not game_outcome:
            game_outcome = 'completed' if sim_day >= self.total_days else 'incomplete'

        # Read final state before shutdown; after _stop_server() the HTTP cash
        # helper intentionally cannot query the in-memory simulator anymore.
        final_status = dict(last_status)
        if self._server_port:
            try:
                queried_status = self._http_get('/game-status')
                if queried_status:
                    final_status = queried_status
                    sim_day = queried_status.get('day', sim_day)
            except Exception:
                pass

        final_cash = final_status.get('cash')
        if final_cash is None:
            final_cash = self._get_cash() if self._server_port else _cash

        # Publish while the server can synchronize its complete state.
        if getattr(self, '_last_weekly_day', None) != sim_day:
            self._weekly_record(final_status)
        if game_outcome != 'stopped':
            self._save_checkpoint(sim_day)
        self._wait_checkpoint()
        self._stop_server()

        if verbose:
            print(f"\n{'='*60}")
            print(f"RUN COMPLETE")
            print(f"{'='*60}")
            print(f"Final Cash: ${final_cash:,.0f}")
            print(f"Sim Days Run: {sim_day}")
            print(f"Outcome: {game_outcome}")
            print(f"Total Turns: {self.agent.total_turns}")
            _total_cache_pct = (self.agent.total_cached_tokens / self.agent.total_input_tokens * 100) if self.agent.total_input_tokens > 0 else 0
            print('Usage (known subtotals and missing counts): ' + json.dumps(self.agent.usage_recorder.summary))
            print(f"Cached Tokens: {self.agent.total_cached_tokens:,} ({_total_cache_pct:.0f}% of input)")
            print(f"Reasoning Tokens: {self.agent.total_reasoning_tokens:,}")
            if self.anthropic_fallback_model or self.agent.total_anthropic_fallbacks:
                print(f"Anthropic Fallbacks: {self.agent.total_anthropic_fallbacks:,}")
            print(f"{'='*60}\n")

        return {
            'run_id': self.run_id,
            'seed': self.seed,
            'scenario': self.scenario,
            'final_cash': final_cash,
            'days_run': sim_day,
            'outcome': game_outcome,
            'total_turns': self.agent.total_turns,
            'total_anthropic_fallbacks': self.agent.total_anthropic_fallbacks,
            'workspace_dir': str(self.workspace_dir),
        }


def main():
    import argparse

    default_config = BenchmarkConfig()
    parser = argparse.ArgumentParser(description="Run bash agent for SaaS Bench")
    parser.add_argument("--model", default=None,
                        help=f"Model name (default: BenchmarkConfig.agent_llm_model={default_config.agent_llm_model})")
    parser.add_argument("--provider", default=None,
                        choices=["openai", "xai", "google", "anthropic", "bedrock", "modal", "together", "deepseek", "opencode", "ai_sandbox"],
                        help=f"API provider (default: BenchmarkConfig.agent_llm_provider={default_config.agent_llm_provider})")
    parser.add_argument("--base-url", help="Custom API base URL")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (new run: 42)")
    parser.add_argument("--scenario", default=None, help="Scenario name (new run: default)")
    parser.add_argument("--days", type=int, default=None, help="Total simulation days shown to the agent (new run: 500)")
    parser.add_argument('--stop-after-day', type=int, default=None,
                        help='Harness-only stop after this completed week; the agent still sees --days')
    parser.add_argument("--workspace", type=Path, help="Workspace base directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--reasoning-effort",
                        choices=["none", "low", "medium", "high", "xhigh", "max"],
                        help="Reasoning effort for reasoning models "
                             f"(default: BenchmarkConfig.agent_llm_reasoning_effort={default_config.agent_llm_reasoning_effort})")
    parser.add_argument("--continue-from", type=Path,
                        help="Path to previous run directory to resume from")
    parser.add_argument("--api-key", help="API key (overrides .env and environment)")
    parser.add_argument("--label",
                        help="Variant tag stored in config.json and shown on the dashboard "
                             "(e.g. 'leads_x1.25'). Lets multiple config variants be "
                             "distinguished without forking the run_id scheme.")
    parser.add_argument('--run-kind', choices=['engineering', 'pilot', 'formal'])
    parser.add_argument('--pricing-file', type=Path, help='JSON with source, basis, and exact-model USD/1k token rates')
    parser.add_argument('--execution-capture', action=argparse.BooleanOptionalAction, default=None, help='Capture public receipts, files, Bash and model source occurrences')
    parser.add_argument('--pf-stale-checks', action=argparse.BooleanOptionalAction, default=None,
                        help='Automatic current-purpose PF dependency checks; default on in PF, disable for offline ablation')
    parser.add_argument('--text-registration', choices=['off', 'git', 'prefix', 'pf'], default=None,
                        help='Shared text tools; prefix privately binds evidence, pf validates delivered evidence; default off')
    parser.add_argument('--sql-capture', action=argparse.BooleanOptionalAction, default=None,
                        help='Privately capture SQL responses; defaults off for new runs')
    args = parser.parse_args()

    runner = BashAgentRunner(
        model=args.model,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
        seed=args.seed,
        scenario=args.scenario,
        total_days=args.days,
        workspace_base=args.workspace,
        reasoning_effort=args.reasoning_effort,
        continue_from=args.continue_from,
        label=args.label,
        run_kind=args.run_kind,
        pricing_file=args.pricing_file,
        sql_capture=args.sql_capture, execution_capture=args.execution_capture, text_registration=args.text_registration,
        pf_stale_checks=args.pf_stale_checks,
        stop_after_day=args.stop_after_day,
    )

    result = runner.run(verbose=not args.quiet)
    print(f"\nResult: {result['outcome']}")
    print(f"Final Cash: ${result['final_cash']:,.0f}")
    print(f"Workspace: {result['workspace_dir']}")


if __name__ == "__main__":
    main()
