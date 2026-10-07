"""Round-five host entry points. No experimental state or answers enter the source tree."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

from saas_bench.agents.bash_agent.run_test import BashAgentRunner, load_env_file
from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
from saas_bench.run_state import (checkpoint_directory, checkpoint_manifest, clone_sql_run,
    file_hash, tree_hash, verify_build, write_json)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'acceptance-materials' / 'round5-first'
MODEL = 'deepseek-v4.1-flash'
ENDPOINT = 'https://opencode.ai/zen/go/v1/'
SEEDS = {42: 'A', 43: 'B', 44: 'C'}


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def locked_write(output, name, value):
    """Shared monitor files use one lock across the independently running groups."""
    import fcntl
    with (output / '.monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write_json(output / name, value)


def environment():
    os.environ['CEOBENCH_SIMULATOR_LLM_PROVIDER'] = 'opencode'
    os.environ['CEOBENCH_SIMULATOR_LLM_MODEL'] = MODEL
    for name in ('BOSSBENCH_LLM_REPLAY_DB', 'ORACLE_MODE', 'CEOBENCH_DASHBOARD_URL',
                 'CEOBENCH_PUBLIC_DIR', 'CEOBENCH_TEST_PUBLIC', 'NOVAMIND_PUBLIC_DIR',
                 'CEOBENCH_SIMULATOR_USAGE_LOG', 'CEOBENCH_MODEL_SESSION'):
        os.environ.pop(name, None)
    os.environ.pop('CEOBENCH_READ_ONLY_TASK', None)


def verify_frozen(output):
    frozen = read(output / 'frozen.json')
    if not frozen or (frozen.get('quota_readiness') or {}).get('status') != 'confirmed':
        raise ValueError('Freeze configuration and quota readiness before launching')
    if verify_build(ROOT / 'public', root=ROOT) != frozen['build']:
        raise ValueError('Public build changed since configuration freeze')
    for path, checksum in frozen['host_scripts'].items():
        if file_hash(path) != checksum:
            raise ValueError('Frozen host script changed: ' + path)
    if read(output / 'pricing.json') != frozen['pricing']:
        raise ValueError('Frozen pricing changed')
    return frozen


def install_pause(runner, request):
    from saas_bench.run_lifecycle import RunCancelled, WorkerLifecycle
    if getattr(runner, 'lifecycle', None):
        runner.lifecycle.hold = request
        if runner.agent:
            runner.agent.lifecycle = runner.lifecycle
        return
    runner.lifecycle = WorkerLifecycle(request, int(os.environ.get('CEOBENCH_SUPERVISOR_PID', os.getppid())))
    if runner.agent:
        runner.agent.lifecycle = runner.lifecycle
    weekly_record = runner._weekly_record
    def weekly(status):
        result = weekly_record(status)
        day = status['day']
        if runner.lifecycle.hold.exists() and 0 < day < runner.total_days and day % 7 == 0:
            raise RunCancelled('hold_at_week_boundary')
        return result
    runner._weekly_record = weekly


def stage_pointer(output, mode, seed, stop, attempt=1):
    if attempt < 1:
        raise ValueError('Attempt must be positive')
    suffix = f'-attempt{attempt}' if attempt > 1 else ''
    return output / f'{mode}-{SEEDS[seed]}-to{stop}{suffix}.json'


def validate_start(mode, seed, stop, source, attempt=1, output=None):
    if mode == 'prefix':
        if stop not in (28, 84) or bool(source) != (stop == 84):
            raise ValueError('Prefix starts fresh to D28, or continues its D28 state to D84')
    elif seed not in SEEDS or stop not in (35, 112, 497) or source is None:
        raise ValueError('Long pairs use a sealed D28 prefix, then gates D35/D112 and end D497')
    if source is None:
        if attempt != 1:
            raise ValueError('An engineering continuation needs its saved source')
        return 0
    cp = read(source / 'checkpoint.json')
    manifest = checkpoint_manifest(source, checkpoint_directory(source, cp))
    expected = 28 if mode == 'prefix' or stop == 35 else 35 if stop == 112 else 112
    if attempt == 1:
        valid_day = cp['day'] == expected
    else:
        previous = read(stage_pointer(output, mode, seed, stop, attempt - 1), {}) if output else {}
        pause = read(source / 'pause-receipt.json', {})
        legacy = (cp['context_boundary'] == 'new_week' and previous.get('status') == 'stopped'
                  and pause.get('day') == cp['day'])
        paused = (previous.get('status') == pause.get('status') == 'paused'
                  and pause.get('day') == cp['day'] and pause.get('snapshot_id') == cp['snapshot_id']
                  and pause.get('context_boundary') == cp['context_boundary']
                  and previous.get('result', {}).get('snapshot_id') == cp['snapshot_id'])
        valid_day = (expected <= cp['day'] < stop and cp['day'] % 7 == 0 and
                     previous.get('path') == str(source.resolve()) and (legacy or paused) and
                     not previous.get('reached_stop') and previous.get('result', {}).get('days_run') == cp['day'])
    if not valid_day or (attempt == 1 and cp['context_boundary'] != 'new_week'):
        raise ValueError('Resume from the specified complete week boundary')
    if manifest.get('text_registration') != mode or manifest['configuration']['seed'] != seed:
        raise ValueError('Resume source group or seed mismatch')
    if mode != 'prefix' and not (manifest.get('fork_source') or manifest.get('sql_evidence', {})).get('source_manifest_sha256'):
        raise ValueError('Long pair must use a fork of the sealed D28 state')
    return cp['day']


def run(args):
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('Start with PYTHONHASHSEED=0')
    output = args.output_dir.resolve()
    verify_frozen(output)
    environment()
    source = args.continue_from.resolve() if args.continue_from else None
    start = validate_start(args.mode, args.seed, args.stop_after_day, source, args.attempt, output)
    pointer = stage_pointer(output, args.mode, args.seed, args.stop_after_day, args.attempt)
    if pointer.exists() or (output / 'hold.json').exists():
        raise ValueError('Run already registered or experiment on hold; inspect before proceeding')
    capture = args.mode in ('prefix', 'pf')
    runner = BashAgentRunner(provider='opencode', model=MODEL, base_url=ENDPOINT,
        reasoning_effort='high', seed=args.seed, total_days=500, scenario='default', initial_cash=1_000_000.,
        workspace_base=output / 'runs' / SEEDS[args.seed] / args.mode, run_kind='formal',
        pricing_file=output / 'pricing.json', continue_from=source, execution_capture=capture,
        sql_capture=capture, text_registration=args.mode, pf_stale_checks=args.mode == 'pf',
        stop_after_day=None if args.stop_after_day == 497 else args.stop_after_day)
    environment()  # Reapply after .env loading; no official-provider override can survive.
    BashAgentToolExecutor(runner.agent_workspace, require_sandbox=True).verify_sandbox()
    runner._prepare_manifest()
    manifest = read(runner.workspace_dir / 'manifest.json')
    for prefix in ('social_post_llm_', 'enterprise_llm_'):
        assert manifest['benchmark_config'][prefix + 'provider'] == 'opencode'
        assert manifest['benchmark_config'][prefix + 'model'] == MODEL
    assert runner.total_days == 497 and runner.base_url == ENDPOINT
    record = dict(group=args.mode, seed=args.seed, state_family=SEEDS[args.seed], run_id=runner.run_id,
        path=str(runner.workspace_dir), status='running', start_day=start, target_day=args.stop_after_day,
        effective_days=497, provider='opencode', model=MODEL, reasoning_effort='high',
        simulator_provider='opencode', simulator_model=MODEL, simulator_reasoning_effort='none',
        cost_basis='opencode_go_quota_usd_both_roles', source_commit=read(output / 'frozen.json')['source_commit'],
        started_at=datetime.now(timezone.utc).isoformat(), attempt=args.attempt)
    write_json(pointer, record)
    print(record['path'], flush=True)
    install_pause(runner, output / 'hold.json')
    try:
        result = runner.run()
        write_json(runner.workspace_dir / 'result.json', result)
        record.update(status=result['outcome'], result=result, finished_at=datetime.now(timezone.utc).isoformat(),
                      reached_stop=result['outcome'] == 'stopped' and result['days_run'] == args.stop_after_day)
        write_json(pointer, record)
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      finished_at=datetime.now(timezone.utc).isoformat())
        write_json(pointer, record)
        raise
    finally:
        runner.client.close()


def seal_state(source, destination, snapshot_id=None):
    """Publish a selector pinned to a generation, never to the moving latest checkpoint."""
    cp = read(source / 'checkpoints' / snapshot_id / 'checkpoint.json') if snapshot_id else read(source / 'checkpoint.json')
    snapshot = checkpoint_directory(source, cp)
    manifest = checkpoint_manifest(source, snapshot)
    if cp['context_boundary'] != 'new_week' or cp['day'] not in (28, 84):
        raise ValueError('Only D28/D84 complete prefix boundaries can be sealed')
    if manifest.get('text_registration') != 'prefix':
        raise ValueError('Seal the original prefix, not a long-run group')
    original_hash = tree_hash(snapshot)
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / 'checkpoints' / cp['snapshot_id']
    target.parent.mkdir()
    shutil.copytree(snapshot, target)
    for name in ('checkpoint.json', 'manifest.json'):
        shutil.copy2(snapshot / name, destination / name)
    shutil.copy2(source / 'config.json', destination / 'config.json')
    assert checkpoint_directory(destination, cp) == target
    assert tree_hash(target) == original_hash == tree_hash(snapshot)
    receipt = dict(source=str(source), destination=str(destination), day=cp['day'], snapshot_id=cp['snapshot_id'],
        snapshot_sha256=original_hash, manifest_sha256=file_hash(destination / 'manifest.json'),
        capture_cutoff=cp['sql_evidence']['cutoff'], workspace_sha256=cp['workspace_sha256'])
    write_json(destination / 'seal.json', receipt)
    # Ordinary writes fail even if the prefix is subsequently extended elsewhere.
    for p in sorted(destination.rglob('*'), reverse=True):
        if not p.is_symlink():
            p.chmod(p.stat().st_mode & ~0o222)
    destination.chmod(destination.stat().st_mode & ~0o222)
    return receipt


def fork_state(source, destination, group, identity):
    cp = read(source / 'checkpoint.json')
    receipt = read(source / 'seal.json')
    snapshot = checkpoint_directory(source, cp)
    if cp['day'] != 28 or not receipt or tree_hash(snapshot) != receipt['snapshot_sha256']:
        raise ValueError('Long pair must fork an intact sealed D28 prefix')
    if read(source / 'manifest.json')['configuration']['seed'] not in SEEDS:
        raise ValueError('Long pair must use a configured prefix seed')
    result = clone_sql_run(source, destination, identity, text_registration=group)
    # Copying the seal preserves its read-only modes. Independent run copies must be writable.
    for p in [destination, *destination.rglob('*')]:
        if not p.is_symlink():
            p.chmod(p.stat().st_mode | 0o200)
    return result


def read_increment(path, cursor, *, prefix=b'', max_bytes=None):
    if not path.exists():
        return [], cursor
    with path.open('rb') as stream:
        stream.seek(cursor)
        rows = []
        # ponytail: budget between records; chunk individual lines if diagnostics exceed 1 MiB.
        while max_bytes is None or stream.tell() - cursor < max_bytes:
            offset = stream.tell()
            line = stream.readline()
            if not line or not line.endswith(b'\n'):
                stream.seek(offset)
                break
            if not line.startswith(prefix):
                continue
            try:
                rows.append(json.loads(line[len(prefix):]))
            except (ValueError, UnicodeDecodeError):
                if not prefix:
                    raise
        return rows, stream.tell()


def sql_failures(run, cursor, *, drain=False):
    path = run / 'logs' / 'api_server_stderr.log'
    if not path.exists():
        return [], {}
    stat = path.stat()
    identity = [stat.st_dev, stat.st_ino]
    offset = cursor.get('offset', 0)
    if cursor.get('path') != str(path) or cursor.get('identity') != identity or offset > stat.st_size:
        offset = 0
    fields = ('day', 'snapshot_ref', 'failure_stage', 'error_type', 'sqlite_errorcode', 'sqlite_errorname')
    issues = []
    while True:
        previous = offset
        rows, offset = read_increment(path, offset, prefix=b'[public_sql] ', max_bytes=1024**2)
        issues.extend({key: row[key] for key in fields if key in row}
                      for row in rows if row.get('error_kind') == 'service')
        if not drain or offset == previous:
            break
    return issues, dict(path=str(path), identity=identity, offset=offset)


def model_issue(row):
    if row['event'] == 'http_request':
        if not row['endpoint'].startswith(ENDPOINT) or row.get('body', {}).get('model') != MODEL:
            return 'model_route_changed'
        if row.get('role') == 'simulator' and (row['body'].get('reasoning_effort') != 'none' or row['body'].get('thinking') != {'type': 'disabled'}):
            return 'simulator_thinking_changed'
    if row['event'] == 'http_error':
        return 'transport_error'
    if row['event'] == 'http_response' and (row.get('error') or row.get('status', 0) >= 400):
        return 'http_or_provider_error'
    if row['event'] == 'response':
        if row.get('error'):
            return 'model_call_error'
        if any(row.get('usage', {}).get(k) is None for k in ('input_tokens', 'output_tokens', 'cached_tokens')):
            return 'missing_usage'
        if row.get('cost_usd') is None:
            return 'missing_cost'
        if row.get('response', {}).get('model') != MODEL:
            return 'served_model_changed'
    return None


def latest_week(run):
    path = run / 'weekly.jsonl'
    if not path.exists():
        return {}
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 32768))
        rows = stream.read().splitlines()
    for line in reversed(rows):
        try:
            return json.loads(line)
        except ValueError:
            pass
    return {}


def quota_health():
    import httpx
    key = os.environ.get('OPENCODE_API_KEY') or load_env_file(ROOT / '.env').get('OPENCODE_API_KEY')
    response = httpx.get(ENDPOINT + 'usage', timeout=15,
        headers={'Authorization': 'Bearer ' + key, 'User-Agent': 'CEO-Bench/1.0',
                 'x-opencode-session': 'round5-quota-monitor'})
    response.raise_for_status()
    usage = response.json()['usage']
    if set(usage) != {'rolling', 'weekly', 'monthly'}:
        raise ValueError('Quota response windows changed')
    return usage


def audit_model_receipts(path, role):
    """Check routes and preserve each unknown failed attempt as an explicit gap."""
    from saas_bench.model_usage import summarize_usage_log
    pending, http_pending, gaps = set(), set(), []
    calls = attempts = failures = 0
    last_success = None
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            event, call = row['event'], row.get('call_id')
            attempt = row.get('attempt_id')
            issue = model_issue(row)
            if issue in ('model_route_changed', 'simulator_thinking_changed', 'served_model_changed'):
                raise ValueError(issue + ': ' + str(call))
            if event == 'request':
                calls += 1
                pending.add(call)
            elif event == 'http_request':
                attempts += 1
                http_pending.add(attempt)
                if role == 'agent' and (row['body'].get('reasoning_effort') != 'high' or row['body'].get('thinking') != {'type': 'enabled'}):
                    raise ValueError('Agent thinking changed')
            elif event in ('http_response', 'http_error'):
                http_pending.discard(attempt)
                if issue:
                    failures += 1
            elif event == 'response':
                pending.discard(call)
                if not issue:
                    last_success = dict(call_id=call, at=row.get('at'), model=row['response']['model'])
            if issue:
                gaps.append(dict(role=role, event=event, call_id=call, attempt_id=attempt, issue=issue))
    if pending or http_pending:
        raise ValueError('Unreturned model requests or HTTP attempts')
    return dict(calls=calls, http_attempts=attempts, failed_http_attempts=failures, gaps=gaps,
                usage=summarize_usage_log(path), last_success=last_success)


def integrity(run, mode, expected_day=None, output=OUT):
    cp = read(run / 'checkpoint.json')
    if expected_day is not None:
        candidates = [p for p in (run / 'checkpoints').iterdir()
            if read(p / 'checkpoint.json', {}).get('day') == expected_day]
        if not candidates:
            raise ValueError(f'Checkpoint D{expected_day} is missing')
        cp = read(sorted(candidates)[-1] / 'checkpoint.json')
    snapshot = checkpoint_directory(run, cp)
    manifest = read(snapshot / 'manifest.json')
    if manifest['text_registration'] != mode:
        raise ValueError('Snapshot group mismatch')
    reviewed = read(output / 'engineering/reviewed-model-gaps.json', {}).get(str(run.resolve()), {})
    model_checks = {}
    for role in ('agent', 'simulator'):
        audit = audit_model_receipts(snapshot / 'request_logs' / f'{role}_requests.jsonl', role)
        approved = reviewed.get(role, [])
        if any(gap not in approved for gap in audit['gaps']):
            raise ValueError('Unreviewed model accounting or transport gaps')
        model_checks[role] = audit
    if mode in ('prefix', 'pf'):
        with sqlite3.connect((snapshot / 'sql-evidence.sqlite').as_uri() + '?mode=ro&immutable=1', uri=True) as conn:
            if conn.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise ValueError('Capture database is corrupt')
            if conn.execute('SELECT count(*) FROM requests WHERE event_id NOT IN (SELECT event_id FROM results)').fetchone()[0]:
                raise ValueError('Capture requests have unresolved results')
            if conn.execute('SELECT count(*) FROM client_calls WHERE received IS NULL').fetchone()[0]:
                raise ValueError('Client calls have unresolved delivery')
    elif list(run.rglob('sql-evidence*')):
        raise ValueError('Private PF material exists in the Git group')
    return dict(status='passed', mode=mode, day=cp['day'], snapshot_id=cp['snapshot_id'],
                model_checks=model_checks,
                cost_completeness='pending_failed_attempt_usage' if any(a['gaps'] for a in model_checks.values()) else 'complete')


def monitor(args):
    """Five-minute local checks; stdout contains only new problems or stage completion."""
    output = args.output_dir.resolve()
    pointer = stage_pointer(output, args.mode, args.seed, args.stop_after_day, args.attempt)
    previous = read(output / f'{stage_pointer(output, args.mode, args.seed, args.stop_after_day, args.attempt - 1).stem}-cursor.json', {}) if args.attempt > 1 else {}
    state = dict(offsets=previous.get('offsets', {}), sql_cursor=previous.get('sql_cursor', {}),
                 alerts=set(), milestones=set())
    argv = [sys.executable, '-u', str(Path(__file__).resolve()), 'run', args.mode,
            '--seed', str(args.seed), '--stop-after-day', str(args.stop_after_day), '--output-dir', str(output),
            '--attempt', str(args.attempt)]
    if args.continue_from:
        argv += ['--continue-from', str(args.continue_from.resolve())]
    log = output / f'{pointer.stem}.log'
    started = time.time()
    ready = output / f'{pointer.stem}-lifecycle-ready.json'
    ready.unlink(missing_ok=True)
    env = dict(os.environ, CEOBENCH_SUPERVISOR_PID=str(os.getpid()), CEOBENCH_LIFECYCLE_READY=str(ready))
    with log.open('xb') as stream:
        process = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env)
    try:
        write_json(output / f'{pointer.stem}-process.json', dict(pid=process.pid, supervisor_pid=os.getpid(), pointer=str(pointer), log=str(log)))
        def event(kind, summary):
            import fcntl
            item = dict(at=datetime.now(timezone.utc).isoformat(), event=kind, **summary)
            with (output / '.monitor.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                with (output / 'events.jsonl').open('a') as stream:
                    stream.write(json.dumps(item) + '\n')
            print(json.dumps(item), flush=True)
        while True:
            rc = process.poll()
            record = read(pointer, {})
            run = Path(record['path']) if record.get('path') else None
            cp = read(run / 'checkpoint.json', {}) if run else {}
            week = latest_week(run) if run else {}
            faults = [name for name in ('branch_stop.json', 'sql-evidence.fault.json', 'checkpoint_error.json')
                      if run and (run / name).exists()]
            issues = []
            counts = {}
            for role in ('agent', 'simulator'):
                path = run / 'logs' / f'{role}_requests.jsonl' if run else None
                if path:
                    rows, state['offsets'][role] = read_increment(path, state['offsets'].get(role, 0))
                    new = [dict(role=role, issue=problem, call_id=row.get('call_id'), attempt_id=row.get('attempt_id'), path=str(path))
                           for row in rows if (problem := model_issue(row))]
                    counts[role] = len({row['call_id'] for row in new})
                    issues.extend(new)
            sql_issues = []
            if run:
                sql_issues, state['sql_cursor'] = sql_failures(run, state['sql_cursor'], drain=rc is not None)
            paths = [log] + (list((run / 'logs').glob('*')) + [run / 'weekly.jsonl', run / 'operation.json', run / 'checkpoint.json'] if run else [])
            idle = time.time() - max([started] + [p.stat().st_mtime for p in paths if p.is_file()])
            status = 'running' if rc is None else record.get('status', 'unexpected_exit')
            warnings = list(faults)
            if rc is not None and status not in ('stopped', 'paused', 'completed', 'bankrupt'):
                warnings.append('unexpected_exit')
            if rc is None and idle >= 1800:
                warnings.append('no_activity_30_minutes')
            disk = shutil.disk_usage(output)
            if disk.free < max(2 * 1024**3, disk.total * .01):
                warnings.append('low_disk_space')
            quota = None
            try:
                quota = quota_health()
                if any(row['status'] != 'ok' or row['percent'] >= 95 for row in quota.values()):
                    warnings.append('go_quota_near_limit')
            except Exception as exc:
                warnings.append('quota_check_failed:' + type(exc).__name__)
            summary = dict(mode=args.mode, family=SEEDS[args.seed], run_id=record.get('run_id'), pid=process.pid,
                day=week.get('day', cp.get('day', 0)), checkpoint_day=cp.get('day'), status=status,
                idle_seconds=round(idle), warnings=warnings, new_errors=counts, issues=issues,
                pointer=str(pointer), path=str(run) if run else None, quota=quota, sql_issues=sql_issues)
            for warning in warnings:
                if warning not in state['alerts']:
                    event('attention', dict(summary, warning=warning))
                    state['alerts'].add(warning)
            if issues:
                event('model_attention', summary)
            if sql_issues:
                event('sql_attention', summary)
            if sql_issues or faults or any(w in warnings for w in ('unexpected_exit', 'low_disk_space', 'go_quota_near_limit')) or any(i['issue'] in ('model_call_error', 'missing_usage', 'missing_cost', 'model_route_changed', 'simulator_thinking_changed', 'served_model_changed') for i in issues):
                locked_write(output, 'hold.json', summary)
            for day in (210, 280, 350, 420):
                if args.stop_after_day == 497 and cp.get('day', 0) >= day and day not in state['milestones']:
                    try:
                        result = integrity(run, args.mode, day, output)
                        write_json(output / f'integrity-{args.mode}-D{day}.json', result)
                    except Exception as exc:
                        locked_write(output, 'hold.json', dict(day=day, error=str(exc)))
                        event('integrity_failed', dict(summary, milestone=day, error=str(exc)))
                    state['milestones'].add(day)
            health = dict(at=datetime.now(timezone.utc).isoformat(), **summary)
            write_json(output / f'{pointer.stem}-health.json', health)
            locked_write(output, 'health.json', health)
            with (output / 'health.jsonl').open('a') as stream:
                stream.write(json.dumps(health) + '\n')
            write_json(output / f'{pointer.stem}-cursor.json', dict(offsets=state['offsets'], sql_cursor=state['sql_cursor'],
                                                                 alerts=sorted(state['alerts']), milestones=sorted(state['milestones'])))
            if rc is not None:
                event('stage_finished', summary)
                return
            time.sleep(300)
    finally:
        if process.poll() is None:
            locked_write(output, 'hold.json', dict(reason='supervisor_exit'))
            deadline = time.monotonic() + 30
            while process.poll() is None and time.monotonic() < deadline:
                if read(ready, {}).get('pid') == process.pid:
                    process.send_signal(__import__('signal').SIGUSR1)
                    break
                time.sleep(.1)
            try:
                process.wait(timeout=max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                write_json(output / f'{pointer.stem}-cleanup-pending.json',
                           dict(pid=process.pid, reason='awaiting_safe_tool_or_checkpoint_boundary'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('run', 'monitor'):
        p = sub.add_parser(command)
        p.add_argument('mode', choices=('prefix', 'git', 'pf'))
        p.add_argument('--seed', type=int, choices=tuple(SEEDS), default=42)
        p.add_argument('--stop-after-day', type=int, required=True)
        p.add_argument('--continue-from', type=Path)
        p.add_argument('--attempt', type=int, default=1)
        p.add_argument('--output-dir', type=Path, default=OUT)
    p = sub.add_parser('seal')
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    p.add_argument('--snapshot-id')
    p = sub.add_parser('fork')
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    p.add_argument('--group', choices=('git', 'pf'), required=True)
    p.add_argument('--branch-id', required=True, help='Evidence identity, not a Git branch')
    args = parser.parse_args(argv)
    if args.command == 'run':
        run(args)
    elif args.command == 'monitor':
        validate_start(args.mode, args.seed, args.stop_after_day, args.continue_from, args.attempt, args.output_dir.resolve())
        verify_frozen(args.output_dir)
        monitor(args)
    elif args.command == 'seal':
        print(json.dumps(seal_state(args.source, args.destination, args.snapshot_id)), flush=True)
    else:
        print(fork_state(args.source, args.destination, args.group, args.branch_id), flush=True)


if __name__ == '__main__':
    main()
