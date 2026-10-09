import base64
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import brotli
import pytest


@pytest.fixture
def observer(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('team_observer', scripts / 'team_observer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest_path = tmp_path / 'engineering/manifest.json'
    manifest_path.parent.mkdir()
    manifest_path.write_text('{}')
    host_logs = tmp_path / 'host-logs'
    host_logs.mkdir()
    manifest = dict(batch_id='paired-s42', category='engineering', source_root='frozen-source',
        public_dir='frozen-public', source_commit='frozen-commit', runs=[
            dict(run_id=role, output_dir=str(tmp_path / 'runs' / role)) for role in ('git', 'pf')])
    expected = {key: manifest[key] for key in ('batch_id', 'source_root', 'public_dir', 'source_commit')}
    expected.update(manifest=str(manifest_path), manifest_sha256=module.file_hash(manifest_path),
                    host_logs=str(host_logs))
    monkeypatch.setattr(module.batch, 'validate_manifest', lambda path: manifest)
    quota = lambda: {window: dict(status='ok', percent=1) for window in ('rolling', 'weekly', 'monthly')}
    state, output = {}, tmp_path / 'observer'

    def log(attempt='001', role='git'):
        root = tmp_path / 'runs' / role / ('attempt-' + attempt)
        root.mkdir(parents=True, exist_ok=True)
        (root / 'attempt.json').write_text(json.dumps(dict(status='running', attempt=attempt, day=0)))
        path = root / 'runtime/private/ceo/model-usage.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def poll(**kwargs):
        return module.poll(manifest_path, output, expected, kwargs.pop('state', state), quota_fn=quota, **kwargs)

    return SimpleNamespace(module=module, expected=expected, manifest=manifest, manifest_path=manifest_path,
        host_logs=host_logs, output=output, state=state, log=log, poll=poll, quota=quota,
        hold=manifest_path.parent / 'HOLD')


def append(path, *rows, prefix=''):
    with path.open('a') as stream:
        for row in rows:
            stream.write(prefix + json.dumps(row) + '\n')


def request(observer, **kwargs):
    return dict(event='http_request', call_id='call', attempt_id='attempt', role='ceo',
        endpoint=kwargs.get('endpoint', observer.module.round5.ENDPOINT + 'chat/completions'),
        body=dict(model=observer.module.round5.MODEL, reasoning_effort='high', thinking={'type': 'enabled'}))


def large_receipt():
    return dict(event='request', call_id='large', padding='x' * (8 * 1024**2))


def events(observer):
    path = observer.output / 'events.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_first_poll_covers_route_error_after_eight_mib(observer):
    path = observer.log()
    append(path, large_receipt(), request(observer, endpoint='https://wrong.example/chat/completions'))
    health = observer.poll()
    assert health['status'] == 'hold' and observer.hold.exists()
    assert [row['issue'] for row in health['route_errors']] == ['model_route_changed']
    assert health['coverage'][str(path)]['offset'] == path.stat().st_size
    assert health['coverage'][str(path)]['backlog_bytes'] == 0
    assert observer.state['cursors'][str(path)]['offset'] == path.stat().st_size
    saved = json.loads((observer.output / 'health.jsonl').read_text().splitlines()[-1])
    assert saved['coverage'] == health['coverage']


def test_first_poll_covers_sql_service_error_after_eight_mib(observer):
    path = observer.host_logs / 'worker.log'
    path.write_text('x' * (8 * 1024**2) + '\n')
    append(path, dict(error_kind='service', error_type='OperationalError', executed_sql='private-query'),
           prefix='[public_sql] ')
    health = observer.poll()
    assert health['status'] == 'hold'
    assert health['sql_service_errors'] == [dict(path=str(path), error_type='OperationalError')]
    assert health['coverage'][str(path)]['offset'] == path.stat().st_size
    assert 'private-query' not in (observer.output / 'health.json').read_text()


def test_partial_line_stays_pending_and_resumes(observer, monkeypatch):
    path = observer.log()
    append(path, dict(event='request', call_id='complete'))
    complete_offset = path.stat().st_size
    with path.open('ab') as stream:
        stream.write(b'{"event":"request","call_id":"partial"')
    calls = []
    original = observer.module.round5.read_increment

    def reader(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(observer.module.round5, 'read_increment', reader)
    first = observer.poll(drain=True)
    assert first['status'] == 'pending' and not observer.hold.exists()
    assert first['coverage'][str(path)]['reason'] == 'partial_line'
    assert first['coverage'][str(path)]['offset'] == complete_offset
    assert len(calls) == 2
    with path.open('ab') as stream:
        stream.write(b'}\n')
    second = observer.poll()
    assert second['status'] == 'healthy'
    assert second['coverage'][str(path)]['offset'] == path.stat().st_size


def test_budget_pending_then_restart_continues_cursor(observer, monkeypatch):
    path = observer.log()
    append(path, large_receipt(), request(observer, endpoint='https://wrong.example/chat/completions'))
    clock = [0]
    monkeypatch.setattr(observer.module.time, 'monotonic', lambda: clock[0])
    original = observer.module.round5.read_increment

    def reader(*args, **kwargs):
        result = original(*args, **kwargs)
        clock[0] = 31
        return result

    monkeypatch.setattr(observer.module.round5, 'read_increment', reader)
    first = observer.poll(max_seconds=30)
    assert first['status'] == 'pending' and not observer.hold.exists()
    assert first['coverage'][str(path)]['backlog_bytes'] > 0
    assert first['coverage'][str(path)]['reason'] == 'budget_exhausted'
    restarted = json.loads((observer.output / 'state.json').read_text())
    assert restarted['cursors'][str(path)]['offset'] == first['coverage'][str(path)]['offset']
    monkeypatch.setattr(observer.module.round5, 'read_increment', original)
    second = observer.poll(state=restarted)
    assert second['status'] == 'hold'
    assert [row['issue'] for row in second['route_errors']] == ['model_route_changed']
    assert second['coverage'][str(path)]['backlog_bytes'] == 0


def test_inherited_receipts_do_not_repeat_events_and_conflicts_hold(observer):
    original = observer.log()
    append(original, dict(event='http_error', call_id='call', attempt_id='attempt'))
    first = observer.poll()
    assert first['status'] == 'healthy'
    assert len(events(observer)) == 1
    inherited = observer.log('002')
    inherited.write_bytes(original.read_bytes())
    restarted = json.loads((observer.output / 'state.json').read_text())
    assert observer.poll(state=restarted)['status'] == 'healthy'
    assert len(events(observer)) == 1
    append(inherited, dict(event='http_error', call_id='changed', attempt_id='attempt'))
    health = observer.poll(state=restarted)
    assert health['status'] == 'hold'
    assert health['checks'] == ['conflicting_receipt:http_error:attempt']
    assert observer.poll(state=restarted)['status'] == 'hold'


@pytest.mark.parametrize('kind', ['json', 'unicode', 'io'])
def test_log_read_failures_hold(observer, monkeypatch, kind):
    path = observer.host_logs / 'worker.log'
    path.write_bytes(b'[public_sql] ' + (b'not-json\n' if kind == 'json' else b'"\xff"\n'))
    if kind == 'io':
        def failed(*args, **kwargs):
            raise OSError('unreadable')
        monkeypatch.setattr(observer.module.round5, 'read_increment', failed)
    health = observer.poll()
    assert health['status'] == 'hold'
    assert health['checks'][0].startswith('log_read:')
    assert health['coverage'][str(path)]['offset'] == 0
    assert observer.hold.exists()


@pytest.mark.parametrize('location', ['host', 'run', 'private', 'private_nested'])
def test_log_discovery_permission_failures_hold(observer, monkeypatch, location):
    if location == 'host':
        path = observer.host_logs / 'worker.log'
        append(path, dict(error_kind='service'), prefix='[public_sql] ')
        blocked = observer.host_logs
        expected = 'host_logs:PermissionError'
    else:
        path = observer.log()
        append(path, request(observer, endpoint='https://wrong.example/chat/completions'))
        private = path.parent.parent
        blocked = {'run': private.parents[2], 'private': private,
                   'private_nested': path.parent}[location]
        expected = ('run_progress:git:PermissionError' if location == 'run' else
                    'private_logs:' + str(private) + ':PermissionError')
    original = os.scandir

    def scandir(path):
        if path == blocked or path == str(blocked):
            raise PermissionError('cannot enumerate logs')
        return original(path)

    monkeypatch.setattr(os, 'scandir', scandir)
    health = observer.poll()
    assert health['status'] == 'hold'
    assert health['checks'] == [expected]
    assert observer.hold.exists()


def test_not_started_run_and_private_roots_are_allowed(observer):
    path = observer.log()
    path.parent.rmdir()
    path.parent.parent.rmdir()
    health = observer.poll()
    assert health['status'] == 'healthy'
    assert health['checks'] == [] and health['coverage'] == {}
    assert health['progress']['pf'] == {'status': 'not_started'}


def test_missing_nested_private_directory_holds(observer, monkeypatch):
    path = observer.log()
    append(path, request(observer, endpoint='https://wrong.example/chat/completions'))
    original = os.scandir

    def scandir(directory):
        if directory == str(path.parent):
            raise FileNotFoundError(2, 'nested directory disappeared', directory)
        return original(directory)

    monkeypatch.setattr(os, 'scandir', scandir)
    health = observer.poll()
    assert health['status'] == 'hold' and observer.hold.exists()
    assert health['checks'] == ['private_logs:' + str(path.parent.parent) + ':FileNotFoundError']


@pytest.mark.parametrize('kind', ['inode', 'truncate', 'missing'])
def test_changed_or_missing_log_holds_without_resetting_cursor(observer, kind):
    path = observer.log()
    append(path, dict(event='request', call_id='first'))
    assert observer.poll()['status'] == 'healthy'
    offset = observer.state['cursors'][str(path)]['offset']
    if kind == 'inode':
        replacement = path.with_suffix('.replacement')
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    elif kind == 'truncate':
        path.write_bytes(b'')
    else:
        path.unlink()
    health = observer.poll()
    assert health['status'] == 'hold'
    assert observer.state['cursors'][str(path)]['offset'] == offset
    assert health['coverage'][str(path)]['reason'].startswith({
        'inode': 'log_identity_changed', 'truncate': 'log_truncated', 'missing': 'log_io:'}[kind])


@pytest.mark.parametrize('drain', [False, True])
def test_append_respects_initial_target_and_drain_reaches_stable_eof(observer, monkeypatch, drain):
    path = observer.log()
    append(path, dict(event='request', call_id='initial'))
    target = path.stat().st_size
    original = observer.module.round5.read_increment
    appended = []

    def reader(*args, **kwargs):
        if not appended:
            append(path, request(observer, endpoint='https://wrong.example/chat/completions'))
            added = observer.host_logs / 'new.log'
            append(added, dict(error_kind='service'), prefix='[public_sql] ')
            appended.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(observer.module.round5, 'read_increment', reader)
    health = observer.poll(drain=drain)
    if drain:
        assert health['status'] == 'hold'
        assert health['coverage'][str(path)]['offset'] == path.stat().st_size
        assert str(observer.host_logs / 'new.log') in health['coverage']
        assert health['sql_service_errors'] == [dict(path=str(observer.host_logs / 'new.log'))]
    else:
        assert health['status'] == 'healthy'
        assert health['coverage'][str(path)]['target_size'] == target
        assert health['coverage'][str(path)]['offset'] == target < path.stat().st_size
        assert str(observer.host_logs / 'new.log') not in health['coverage']
        assert observer.poll()['status'] == 'hold'


def test_partial_target_does_not_read_newly_appended_line(observer, monkeypatch):
    path = observer.log()
    path.write_bytes(b'{"event":"request","call_id":"first"')
    target = path.stat().st_size
    original = observer.module.round5.read_increment

    def reader(*args, **kwargs):
        with path.open('ab') as stream:
            stream.write(b'}\n')
        return original(*args, **kwargs)

    monkeypatch.setattr(observer.module.round5, 'read_increment', reader)
    health = observer.poll(drain=True)
    assert health['status'] == 'pending'
    assert health['coverage'][str(path)]['target_size'] == target
    assert health['coverage'][str(path)]['offset'] == 0
    assert health['coverage'][str(path)]['reason'] == 'partial_line'
    monkeypatch.setattr(observer.module.round5, 'read_increment', original)
    assert observer.poll()['status'] == 'healthy'


def test_compressed_physical_receipt_retains_served_model_and_usage_checks(observer):
    path = observer.log()
    body = dict(model=observer.module.round5.MODEL,
        usage=dict(prompt_tokens=10, completion_tokens=2, prompt_cache_hit_tokens=0))
    encoded = base64.b64encode(brotli.compress(json.dumps(body).encode())).decode()
    append(path, request(observer), dict(event='http_response', call_id='call', attempt_id='attempt',
                                        status=200, body={'base64': encoded}))
    assert observer.poll()['status'] == 'healthy'
    append(path, dict(event='http_response', call_id='other-call', attempt_id='other-attempt', status=200,
                     body={'base64': encoded}))
    health = observer.poll()
    assert health['status'] == 'hold'
    assert [row['issue'] for row in health['route_errors']] == ['served_model_changed']


def test_fault_hold_and_freeze_and_quota_checks_are_preserved(observer):
    path = observer.log()
    path.write_bytes(b'')
    fault = path.parent.parent / 'evidence.fault.json'
    fault.write_text('{}')
    assert observer.poll()['status'] == 'hold'
    assert events(observer)[0]['new_faults'] == [str(fault)]
    restarted = json.loads((observer.output / 'state.json').read_text())
    assert observer.poll(state=restarted)['status'] == 'hold'
    assert len(events(observer)) == 1
    fault.unlink()
    assert observer.poll()['status'] == 'hold'
    observer.manifest_path.write_text('{"changed":true}')
    health = observer.poll()
    assert health['checks'] == ['freeze:Manifest anchor mismatch']
    observer.manifest_path.write_text('{}')
    health = observer.module.poll(observer.manifest_path, observer.output, observer.expected, observer.state,
                                  quota_fn=lambda: {'rolling': dict(status='ok', percent=100)})
    assert health['checks'] == ['quota_windows_changed', 'quota_exhausted']


def test_main_uses_policy_directory_and_saved_state(observer, monkeypatch, capsys):
    path = observer.log()
    append(path, dict(event='request', call_id='first'))
    observer.poll()
    policy = observer.output.parent / 'policy.json'
    policy.write_text(json.dumps(dict(identity=observer.expected)))
    original = observer.module.poll
    loaded = []

    def poll(*args, **kwargs):
        loaded.append(args[3]['cursors'][str(path)]['offset'])
        return original(*args, quota_fn=observer.quota, **kwargs)

    monkeypatch.setattr(observer.module, 'poll', poll)
    observer.module.main(['--policy', str(policy), '--once'])
    assert loaded == [path.stat().st_size]
    assert json.loads(capsys.readouterr().out)['status'] == 'healthy'


def test_reader_preserves_old_prefix_contract_and_complete_line_budget(observer):
    path = observer.host_logs / 'worker.log'
    path.write_bytes(b'noise\n[public_sql] broken\n[public_sql] {"ok":true}\n')
    reader = observer.module.round5.read_increment
    rows, offset = reader(path, 0, prefix=b'[public_sql] ')
    assert rows == [{'ok': True}] and offset == path.stat().st_size
    with pytest.raises(ValueError):
        reader(path, 0, prefix=b'[public_sql] ', strict=True)
    path.write_bytes(b'{"long":"' + b'x' * 100 + b'"}\n{"next":true}\n')
    rows, offset = reader(path, 0, max_bytes=1)
    assert rows == [{'long': 'x' * 100}]
    assert offset > 1
    assert reader(path, 0, end_offset=offset - 1) == ([], 0)
