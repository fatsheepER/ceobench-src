import argparse
import base64
import binascii
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time

import brotli

import round5
from team_model_monitor import finish_billing, record_billing
from saas_bench import team_batch as batch
from saas_bench.model_usage import usage_values
from saas_bench.run_state import file_hash, write_json
from saas_bench.team_usage import _body


def tail_hash(path, offset):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        stream.seek(offset)
        while chunk := stream.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def preserve_closed_tail(path, target, output, manifest, expected):
    unitfile = output.parent / 'controller-units.json'
    if not unitfile.exists():
        return None
    units = batch.read(unitfile)
    run = next((run for run in manifest['runs']
        if path.is_relative_to(Path(run['output_dir'])) or
        path.parent == Path(expected['host_logs']) and path.name.startswith(run['run_id'] + '-')), None)
    run_id = run['run_id'] if run else None
    unit = units.get(run_id, {})
    if (not unit.get('scope_empty') or not unit.get('invocation_id') or
            unit.get('exit_code') is None or unit.get('cleanup_error')):
        return None
    # A reaped client and empty invocation-bound scope prove these bytes cannot finish.
    directory = output / 'closed-log-tails'
    directory.mkdir(exist_ok=True)
    name = hashlib.sha256(json.dumps([str(path), target['identity'], target['target_size']]).encode()).hexdigest()
    tail = directory / (name + '.bin')
    with path.open('rb') as source, tail.with_suffix('.tmp').open('wb') as saved:
        source.seek(target['offset'])
        shutil.copyfileobj(source, saved, length=1024**2)
        saved.flush()
        os.fsync(saved.fileno())
    tail.with_suffix('.tmp').replace(tail)
    after = path.stat()
    digest = file_hash(tail)
    if ([after.st_dev, after.st_ino] != target['identity'] or after.st_size != target['target_size'] or
            digest != tail_hash(path, target['offset'])):
        raise ValueError('Closed log changed while preserving its tail')
    history = batch.attempts(run)
    attempt = next((root for root in history if path.is_relative_to(root)), history[-1] if history else None)
    proof = dict(path=str(path), identity=target['identity'], start=target['offset'],
        end=target['target_size'], tail_path=str(tail), tail_sha256=digest, run_id=run_id,
        original_sha256=file_hash(path), attempt_dir=str(attempt) if attempt else None,
        unit=unit, fault_kind='permanent_capture_fault', reason='closed_partial_line')
    write_json(tail.with_suffix('.json'), proof)
    return proof


def physical_body(value):
    if isinstance(value, dict) and set(value) == {'base64'}:
        try:
            raw = base64.b64decode(value['base64'], validate=True)
            decoded = json.loads(brotli.decompress(raw))
        except (binascii.Error, brotli.error, ValueError, UnicodeError, TypeError):
            return _body(value)
        if isinstance(decoded, dict):
            return decoded
    return _body(value)


def identity(path, expected):
    path = Path(path).resolve()
    manifest = batch.validate_manifest(path)
    selection = expected.get('runs')
    if selection is None:
        return manifest
    if (not isinstance(selection, list) or not selection or
            any(not isinstance(run, dict) or set(run) != {'run_id', 'output_dir'} for run in selection) or
            len({run['run_id'] for run in selection}) != len(selection)):
        logging.warning('Monitoring selection has duplicate or mismatched run IDs and output paths')
    runs = [batch.selected(manifest, run['run_id']) for run in selection]
    if any(run['output_dir'] != anchor['output_dir'] for run, anchor in zip(runs, selection)):
        logging.warning('Wrong monitored run output path')
    return dict(manifest, runs=runs)


def _capture_logs(manifest, expected, cursors, checks, progress, faults):
    paths = {}
    for run in manifest['runs']:
        try:
            history = sorted((path for path in Path(run['output_dir']).iterdir()
                              if path.name.startswith('attempt-')),
                             key=lambda path: int(path.name.removeprefix('attempt-')))
        except FileNotFoundError:
            history = []
        except OSError as exc:
            checks.append('run_progress:' + run['run_id'] + ':' + type(exc).__name__)
            continue
        try:
            if history:
                saved = json.loads((history[-1] / 'attempt.json').read_text())
                progress[run['run_id']] = dict(status=saved['status'], **{key: saved.get(key) for key in (
                    'attempt', 'day', 'reason', 'runtime_failure', 'checkpoint_reused_at_stop', 'runtime_phase')})
                if saved['status'] in ('paused', 'failed') and 'model_attempt_limit' in (
                        saved.get('reason'), saved.get('runtime_failure')):
                    checks.append('model_attempt_limit:' + run['run_id'])
            else:
                progress[run['run_id']] = {'status': 'not_started'}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            checks.append('run_progress:' + run['run_id'] + ':' + type(exc).__name__)
        for attempt in history:
            private = attempt / 'runtime/private'

            def raise_error(exc):
                if isinstance(exc, FileNotFoundError) and exc.filename == str(private):
                    return
                raise exc

            try:
                for root, _, files in private.walk(on_error=raise_error):
                    if root == private and 'evidence.fault.json' in files:
                        faults[str(private / 'evidence.fault.json')] = run['run_id']
                    for name in files:
                        if name.endswith('model-usage.jsonl'):
                            paths[str(root / name)] = b''
            except OSError as exc:
                checks.append('private_logs:' + str(private) + ':' + type(exc).__name__)
    logroot = Path(expected['host_logs'])
    try:
        paths.update((str(log), b'[public_sql] ') for log in sorted(logroot.iterdir())
                     if log.name.endswith('.log'))
    except OSError as exc:
        checks.append('host_logs:' + type(exc).__name__)
    for key in cursors:
        paths.setdefault(key, b'' if key.endswith('model-usage.jsonl') else b'[public_sql] ')
    targets = {}
    for key, prefix in sorted(paths.items()):
        prior = cursors.get(key, {})
        target = dict(identity=None, target_size=None, offset=prior.get('offset', 0),
                      backlog_bytes=None, reason=None)
        try:
            stat = Path(key).stat()
            target.update(identity=[stat.st_dev, stat.st_ino], target_size=stat.st_size)
            target['backlog_bytes'] = max(0, stat.st_size - target['offset'])
            if prior and (prior.get('identity', target['identity']) != target['identity'] or
                          prior.get('inode', stat.st_ino) != stat.st_ino):
                target['reason'] = 'log_identity_changed'
            elif max(target['offset'], prior.get('billing_previous_offset', 0)) > stat.st_size:
                target['reason'] = 'log_truncated'
            elif prior.get('closed_tail'):
                proof = prior['closed_tail']
                if (stat.st_size != proof['end'] or tail_hash(key, proof['start']) != proof['tail_sha256'] or
                        file_hash(key) != proof['original_sha256'] or
                        file_hash(proof['tail_path']) != proof['tail_sha256']):
                    target['reason'] = 'closed_log_changed'
                else:
                    target['terminal_fault'] = proof['reason']
                    faults[proof['tail_path']] = proof['run_id']
        except OSError as exc:
            target['reason'] = 'log_io:' + type(exc).__name__
        if target['reason']:
            checks.append(target['reason'] + ':' + key)
        targets[key] = (prefix, target)
    return targets


def poll(manifest_path, output, expected, state, *, quota_fn=None,
         hold_path=None, max_seconds=30, drain=False):
    deadline = time.monotonic() + max_seconds
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    from saas_bench.team_recovery import hold_path as effective_hold
    hold = Path(hold_path) if hold_path else effective_hold(manifest_path)
    checks, sql, routes, transient = [], [], [], []
    quota, manifest, quota_error = None, None, None
    if quota_fn is None:
        def quota_fn():
            if not os.environ.get('OPENCODE_API_KEY'):
                raise ValueError('OPENCODE_API_KEY must already be in the environment')
            return round5.quota_health()
    if state.get('monitor_identity', expected) != expected:
        checks.append('freeze:Monitored identity changed')
    state.setdefault('monitor_identity', batch.canonical(expected))
    try:
        manifest = identity(manifest_path, expected)
    except Exception as exc:
        checks.append('freeze:' + str(exc))
    for key in ('quota_retry_at', 'quota_outage_started'):
        state.pop(key, None)
    try:
        quota = quota_fn()
        if set(quota) != {'rolling', 'weekly', 'monthly'}:
            checks.append('quota_windows_changed')
        if any(row.get('status') != 'ok' or row.get('percent', 100) >= 100 for row in quota.values()):
            checks.append('quota_exhausted')
    except Exception as exc:
        quota_error = 'quota_unavailable:' + type(exc).__name__
        checks.append(quota_error)
        state.update(quota_failures=state.get('quota_failures', 0) + 1, quota_error=quota_error)
    else:
        state.update(quota_failures=0, quota_error=None, quota_last_success=time.time())
    physical_requests = state.setdefault('physical_request_meta', {})
    cursors = state.setdefault('cursors', {})
    seen = state.setdefault('receipt_hashes', {})
    if 'billing_groups' not in state and seen:
        for key, cursor in cursors.items():
            if key.endswith('model-usage.jsonl'):
                cursor['billing_previous_offset'] = cursor['offset']
                cursor['offset'] = 0
        state['billing_rebuild'] = True
    state.setdefault('billing_groups', {})
    billing_changed = state.setdefault('billing_pending', [])
    known = set(state.get('known_faults', []))
    previous = state.get('active_checks', [])
    oldprogress = state.get('progress', {})
    faults, progress, coverage = {}, {}, {}
    pending = quota_error is not None

    def event(*, new_checks=(), sql_errors=(), route_errors=(), new_faults=(),
              transient_errors=(), completed=(), billing=()):
        with (output / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),
                new_checks=list(new_checks), sql_service_errors=list(sql_errors),
                route_errors=list(route_errors), new_faults=list(new_faults),
                transient_receipt_failures=list(transient_errors), completed=list(completed),
                billing_warnings=list(billing))) + '\n')

    targets = _capture_logs(manifest, expected, cursors, checks, progress, faults) if manifest else {}
    while manifest:
        coverage = {key: target for key, (_, target) in targets.items()}
        for key, (prefix, target) in targets.items():
            if target['reason']:
                continue
            path = Path(key)
            while target['offset'] < target['target_size']:
                if time.monotonic() >= deadline:
                    target['reason'] = 'budget_exhausted'
                    pending = True
                    break
                offset = target['offset']
                sql_start, route_start, transient_start = len(sql), len(routes), len(transient)
                try:
                    before = path.stat()
                    if [before.st_dev, before.st_ino] != target['identity']:
                        raise ValueError('log_identity_changed')
                    if before.st_size < target['target_size']:
                        raise ValueError('log_truncated')
                    rows, next_offset = round5.read_increment(path, offset, prefix=prefix,
                        max_bytes=8 * 1024**2, end_offset=target['target_size'], strict=True)
                    after = path.stat()
                    if [after.st_dev, after.st_ino] != target['identity']:
                        raise ValueError('log_identity_changed')
                    if after.st_size < target['target_size']:
                        raise ValueError('log_truncated')
                    for row in rows:
                        if prefix:
                            if row.get('error_kind') == 'service':
                                sql.append(dict(path=key, **{field: row[field] for field in (
                                    'day', 'failure_stage', 'error_type', 'sqlite_errorcode',
                                    'sqlite_errorname') if field in row}))
                            continue
                        receipt_id = row.get('attempt_id') if row['event'].startswith('http_') else row.get('call_id')
                        if not receipt_id:
                            raise ValueError('Immutable model receipt ID is required')
                        receipt_key = row['event'] + ':' + str(receipt_id)
                        fingerprint = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                        group = next((run['run_id'] for run in manifest['runs']
                                      if path.is_relative_to(Path(run['output_dir']))), None)
                        if group is None:
                            raise ValueError('Model usage log outside frozen run outputs')
                        if receipt_key in seen:
                            if seen[receipt_key] != fingerprint:
                                checks.append('conflicting_receipt:' + receipt_key)
                            elif state.get('billing_rebuild'):
                                record_billing(row, state, group, path)
                            continue
                        issue = round5.model_issue(row)
                        if row['event'] == 'http_request':
                            physical_requests[row['attempt_id']] = {
                                field: row.get('body', {}).get(field) for field in ('model', 'stream')}
                        if row['event'] == 'http_response' and 200 <= row.get('status', 0) < 400 and not row.get('error'):
                            body = physical_body(row.get('body'))
                            wire_usage = usage_values(body, 'chat')
                            request_meta = physical_requests.get(row['attempt_id'], {})
                            if body.get('model') != request_meta.get('model'):
                                issue = 'served_model_changed'
                            elif any(wire_usage.get(field) is None for field in (
                                    'input_tokens', 'output_tokens')):
                                issue = 'missing_usage'
                        if row['event'] == 'http_request' and row.get('role') != 'simulator':
                            if (row['body'].get('reasoning_effort') != 'high' or
                                    row['body'].get('thinking') != {'type': 'enabled'}):
                                issue = 'business_thinking_changed'
                        detail = dict(issue=issue, path=key, call_id=row.get('call_id'),
                                      attempt_id=row.get('attempt_id'))
                        if issue in ('model_route_changed', 'served_model_changed',
                                     'simulator_thinking_changed', 'business_thinking_changed',
                                     'missing_usage', 'model_call_error'):
                            routes.append(detail)
                        elif issue:
                            transient.append(detail)
                        if recorded := record_billing(row, state, group, path):
                            billing_changed.append(recorded)
                        seen[receipt_key] = fingerprint
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    target['reason'] = 'log_read:' + type(exc).__name__ + ':' + str(exc)
                    checks.append(target['reason'] + ':' + key)
                if (len(sql), len(routes), len(transient)) != (sql_start, route_start, transient_start):
                    event(sql_errors=sql[sql_start:], route_errors=routes[route_start:],
                          transient_errors=transient[transient_start:])
                if target['reason']:
                    break
                target['offset'] = next_offset
                target['backlog_bytes'] = target['target_size'] - next_offset
                cursor = dict(offset=next_offset, inode=target['identity'][1], identity=target['identity'])
                if 'billing_previous_offset' in cursors.get(key, {}):
                    cursor['billing_previous_offset'] = cursors[key]['billing_previous_offset']
                cursors[key] = cursor
                write_json(output / 'state.json', state)
                if next_offset == offset:
                    try:
                        proof = preserve_closed_tail(path, target, output, manifest, expected)
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        target['reason'] = 'closed_tail_fault:' + type(exc).__name__
                        checks.append(target['reason'] + ':' + key)
                        break
                    if proof:
                        target.update(offset=target['target_size'], backlog_bytes=0,
                            terminal_fault=proof['reason'])
                        cursors[key].update(offset=target['offset'], closed_tail=proof)
                        faults[proof['tail_path']] = proof['run_id']
                        write_json(output / 'state.json', state)
                    else:
                        target['reason'] = 'partial_line'
                        pending = True
                    break
        if time.monotonic() >= deadline:
            pending = True
        if not drain or pending or any(target['reason'] for target in coverage.values()):
            break
        latest = _capture_logs(manifest, expected, cursors, checks, progress, faults)
        if ({key: (target['identity'], target['target_size']) for key, (_, target) in latest.items()} ==
                {key: (target['identity'], target['target_size']) for key, (_, target) in targets.items()}):
            break
        targets = latest

    billing_complete = bool(manifest) and all(target['offset'] == target['target_size'] and not target['reason']
        for prefix, target in targets.values() if not prefix)
    bill_warnings = finish_billing(state, billing_complete)
    if billing_complete:
        for cursor in cursors.values():
            cursor.pop('billing_previous_offset', None)
    if bill_warnings:
        event(billing=bill_warnings)
    checks = list(dict.fromkeys(checks))
    newfaults = sorted(set(faults) - known)
    newchecks = [check for check in checks if check not in previous]
    completed = [key for key, value in progress.items() if value.get('status') in (
        'finished', 'paused', 'failed') and oldprogress.get(key) != value]
    if newchecks or newfaults or completed:
        event(new_checks=newchecks, new_faults=newfaults, completed=completed)
    state.update(active_checks=checks, known_faults=sorted(known | set(faults)), progress=progress)
    units = {}
    unitfile = output.parent / 'controller-units.json'
    if unitfile.exists():
        from saas_bench.systemd_guard import unit_status
        for run_id, saved in json.loads(unitfile.read_text()).items():
            units[run_id] = unit_status(saved['unit'], invocation_id=saved.get('invocation_id'),
                exit_code=saved.get('exit_code'))
            if units[run_id]['result'] == 'oom-kill':
                progress[run_id] = dict(progress.get(run_id, {}), status='failed', reason='oom-kill')
                if run_id not in state.get('known_oom_runs', []):
                    event(new_checks=['unit_oom:' + run_id])
        state['known_oom_runs'] = [run_id for run_id, record in units.items() if record['result'] == 'oom-kill']
    write_json(output / 'state.json', state)
    health = dict(time=datetime.now(timezone.utc).isoformat(), manifest=expected['manifest'],
        manifest_sha256=expected['manifest_sha256'],
        runs=[{key: run[key] for key in ('run_id', 'output_dir')} for run in manifest['runs']] if manifest else [],
        batch_id=expected['batch_id'], status='pending' if pending else 'healthy',
        historical_hold=hold.exists(),
        quota=quota, quota_error=quota_error, checks=checks, sql_service_errors=sql, route_errors=routes,
        billing_warnings=bill_warnings,
        billing_coverage=state['billing_coverage'],
        run_faults=[dict(run_id=run_id, path=path) for path, run_id in sorted(faults.items())],
        progress=progress, units=units, coverage=coverage, state=state)
    write_json(output / 'health.json', health)
    with (output / 'health.jsonl').open('a') as stream:
        stream.write(json.dumps({key: value for key, value in health.items() if key != 'state'}) + '\n')
    return health


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--drain', action='store_true')
    args = parser.parse_args(argv)
    expected = json.loads(args.policy.read_text())['identity']
    directory = args.policy.resolve().parent / 'observer'
    statefile, healthfile = directory / 'state.json', directory / 'health.json'
    state = (json.loads(statefile.read_text()) if statefile.exists() else
             json.loads(healthfile.read_text()).get('state', {}) if healthfile.exists() else {})
    while True:
        health = poll(expected['manifest'], directory, expected, state, drain=args.drain)
        if args.once or args.drain:
            print(json.dumps({key: health[key] for key in ('time', 'status', 'checks', 'quota', 'coverage')}, indent=2))
            return
        time.sleep(300)


if __name__ == '__main__':
    main()
