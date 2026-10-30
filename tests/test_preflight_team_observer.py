import base64
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import brotli
import httpx
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
            dict(run_id=role, group=role, output_dir=str(tmp_path / 'runs' / role)) for role in ('git', 'pf')])
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
    assert health['status'] == 'healthy' and not observer.hold.exists()
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
    assert health['status'] == 'healthy'
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
    assert second['status'] == 'healthy'
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
    assert health['status'] == 'healthy'
    assert health['checks'] == ['conflicting_receipt:http_error:attempt']
    assert observer.poll(state=restarted)['status'] == 'healthy'


@pytest.mark.parametrize('kind', ['json', 'unicode', 'io'])
def test_log_read_failures_hold(observer, monkeypatch, kind):
    path = observer.host_logs / 'worker.log'
    path.write_bytes(b'[public_sql] ' + (b'not-json\n' if kind == 'json' else b'"\xff"\n'))
    if kind == 'io':
        def failed(*args, **kwargs):
            raise OSError('unreadable')
        monkeypatch.setattr(observer.module.round5, 'read_increment', failed)
    health = observer.poll()
    assert health['status'] == 'healthy'
    assert health['checks'][0].startswith('log_read:')
    assert health['coverage'][str(path)]['offset'] == 0
    assert not observer.hold.exists()


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
    assert health['status'] == 'healthy'
    assert health['checks'] == [expected]
    assert not observer.hold.exists()


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
    assert health['status'] == 'healthy' and not observer.hold.exists()
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
    assert health['status'] == 'healthy'
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
        assert health['status'] == 'healthy'
        assert health['coverage'][str(path)]['offset'] == path.stat().st_size
        assert str(observer.host_logs / 'new.log') in health['coverage']
        assert health['sql_service_errors'] == [dict(path=str(observer.host_logs / 'new.log'))]
    else:
        assert health['status'] == 'healthy'
        assert health['coverage'][str(path)]['target_size'] == target
        assert health['coverage'][str(path)]['offset'] == target < path.stat().st_size
        assert str(observer.host_logs / 'new.log') not in health['coverage']
        assert observer.poll()['status'] == 'healthy'


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
    assert health['status'] == 'healthy'
    assert [row['issue'] for row in health['route_errors']] == ['served_model_changed']






@pytest.mark.parametrize('error', [httpx.ConnectError, httpx.ReadTimeout])
def test_quota_is_observed_once_per_poll_without_retry_or_outage_budget(observer, monkeypatch, error):
    clock, calls = [1000.0], []
    monkeypatch.setattr(observer.module.time, 'time', lambda: clock[0])
    state = dict(quota_retry_at=999999, quota_outage_started=0, quota_error='old')
    def unavailable():
        calls.append(clock[0])
        raise error('unavailable')
    for at in (1000, 1000, 4600, 86400):
        clock[0] = at
        health = observer.module.poll(observer.manifest_path, observer.output, observer.expected,
                                      state, quota_fn=unavailable)
        assert health['status'] == 'pending'
        assert health['checks'] == ['quota_unavailable:' + error.__name__]
        assert 'quota_retry_after' not in health
        assert 'quota_retry_at' not in state and 'quota_outage_started' not in state
        assert not observer.hold.exists()
        state = json.loads((observer.output / 'state.json').read_text())
    assert calls == [1000, 1000, 4600, 86400]
    assert state['quota_failures'] == 4


def test_quota_success_resets_observation_errors_and_exhaustion_is_reported(observer):
    def unavailable():
        raise httpx.ConnectError('unavailable')
    failed = observer.module.poll(observer.manifest_path, observer.output, observer.expected,
                                 observer.state, quota_fn=unavailable)
    assert failed['quota_error'] == 'quota_unavailable:ConnectError'
    healthy = observer.poll()
    assert healthy['status'] == 'healthy' and healthy['quota_error'] is None
    assert observer.state['quota_failures'] == 0
    exhausted = observer.quota()
    exhausted['rolling']['percent'] = 100
    health = observer.module.poll(observer.manifest_path, observer.output, observer.expected,
                                  observer.state, quota_fn=lambda: exhausted)
    assert health['status'] == 'healthy' and health['checks'] == ['quota_exhausted']
    assert not observer.hold.exists()


@pytest.mark.parametrize('status', [401, 403])
def test_quota_confirmed_http_failure_is_reported_without_retries(observer, status):
    def denied():
        httpx.Response(status, request=httpx.Request('GET', 'https://quota.invalid/usage')).raise_for_status()
    result = observer.module.poll(observer.manifest_path, observer.output, observer.expected,
        observer.state, quota_fn=denied)
    assert result['status'] == 'pending'
    assert result['checks'] == ['quota_unavailable:HTTPStatusError']


@pytest.mark.parametrize('status', [408, 429, 500, 502, 503, 504])
def test_quota_http_read_unavailability_and_exhaustion_remain_observations(observer, status):
    def unavailable():
        httpx.Response(status, request=httpx.Request('GET', 'https://quota.invalid/usage')).raise_for_status()
    health = observer.module.poll(observer.manifest_path, observer.output, observer.expected,
        observer.state, quota_fn=unavailable)
    assert health['status'] == 'pending' and not observer.hold.exists()
    exhausted = observer.quota()
    exhausted['rolling']['percent'] = 100
    def confirmed():
        httpx.Response(429, json={'usage': exhausted},
            request=httpx.Request('GET', 'https://quota.invalid/usage')).raise_for_status()
    health = observer.module.poll(observer.manifest_path, observer.output, observer.expected,
        observer.state, quota_fn=confirmed)
    assert health['status'] == 'pending' and not observer.hold.exists()




@pytest.mark.parametrize('usage', [{'rolling': {}}, {'rolling': {}, 'weekly': {}, 'monthly': {'percent': 'broken'}}])
def test_quota_rate_limit_with_malformed_usage_is_permanent(observer, usage):
    def malformed():
        httpx.Response(429, json={'usage': usage},
            request=httpx.Request('GET', 'https://quota.invalid/usage')).raise_for_status()
    health = observer.module.poll(observer.manifest_path, observer.output, observer.expected,
        observer.state, quota_fn=malformed)
    assert health['status'] == 'pending' and not observer.hold.exists()


def test_invalid_quota_response_is_reported(observer):
    def invalid():
        raise ValueError('Quota response windows changed')

    health = observer.module.poll(observer.manifest_path, observer.output, observer.expected,
                                  observer.state, quota_fn=invalid)
    assert health['status'] == 'pending' and not observer.hold.exists()
    assert health['checks'] == ['quota_unavailable:ValueError']


def test_failed_run_does_not_hold_healthy_peer_but_retains_global_fault_checks(observer):
    failed = observer.log(role='git')
    append(failed, dict(event='http_error', call_id='call', attempt_id='attempt'))
    observer.log(role='pf').write_bytes(b'')
    attempt = failed.parents[3] / 'attempt.json'
    attempt.write_text(json.dumps(dict(status='failed', attempt='001', day=14,
                                      reason='ordinary_failure', runtime_phase='ceo')))
    health = observer.poll()
    assert health['status'] == 'healthy' and not observer.hold.exists()
    assert health['checks'] == []
    assert health['progress']['git']['status'] == 'failed'
    assert health['progress']['git']['reason'] == 'ordinary_failure'
    assert health['progress']['pf']['status'] == 'running'
    assert events(observer)[-1]['completed'] == ['git']
    assert [row['issue'] for row in events(observer)[0]['transient_receipt_failures']] == ['transport_error']
    restarted = json.loads((observer.output / 'state.json').read_text())
    assert observer.poll(state=restarted)['status'] == 'healthy'
    assert len(events(observer)) == 2
    append(failed, request(observer, endpoint='https://wrong.example/chat/completions'))
    health = observer.poll(state=restarted)
    assert health['status'] == 'healthy' and not observer.hold.exists()
    assert [row['issue'] for row in health['route_errors']] == ['model_route_changed']


@pytest.mark.parametrize('status,reason,failure,hold', [
    ('paused', 'model_attempt_limit', None, True),
    ('failed', 'model_attempt_limit', None, True),
    ('failed', 'failed', 'model_attempt_limit', True),
    ('running', 'model_attempt_limit', 'model_attempt_limit', False),
])
def test_attempt_limit_is_reported_only_after_harness_exit(observer, status, reason, failure, hold):
    path = observer.log()
    (path.parents[3] / 'attempt.json').write_text(json.dumps(dict(status=status,
        attempt='001', day=0, reason=reason, runtime_failure=failure)))
    state = {}
    for _ in range(2):
        health = observer.poll(state=state)
        assert health['status'] == 'healthy'
        assert not observer.hold.exists()
        assert health['checks'] == (['model_attempt_limit:git'] if hold else [])
        state = json.loads((observer.output / 'state.json').read_text())
    if hold:
        assert sum('model_attempt_limit:git' in event['new_checks'] for event in events(observer)) == 1


@pytest.mark.parametrize('event', ['response', 'http_error'])
@pytest.mark.parametrize('error,issue', [
    ('APIConnectionError', 'transport_error'), ('APITimeoutError', 'transport_error'),
    ('ConnectError', 'transport_error'), ('ReadTimeout', 'transport_error'),
    ('LLMTimeoutError', 'transport_error'), ('ValueError', 'model_call_error'),
    ('UnsupportedProtocol', 'model_call_error'), ('LocalProtocolError', 'model_call_error'),
])
def test_observer_network_and_configuration_receipts(observer, event, error, issue):
    path = observer.log()
    append(path, dict(event=event, call_id='call', attempt_id='attempt', error=error,
                      response=None))
    health = observer.poll()
    assert health['status'] == 'healthy'
    details = (events(observer)[0]['transient_receipt_failures'] if issue == 'transport_error'
               else health['route_errors'])
    assert [row['issue'] for row in details] == [issue]
    assert not observer.hold.exists()


def formal_pair(observer):
    observer.manifest['category'] = 'formal'
    for run, group in zip(observer.manifest['runs'], ('git', 'pf')):
        run.update(pair_id='s101-r1', group=group)
    observer.expected['runs'] = [{key: run[key] for key in ('run_id', 'output_dir')}
                                 for run in observer.manifest['runs']]
    observer.manifest['runs'].extend([
        dict(run_id=group + '-unselected', pair_id='s101-r2', group=group,
             output_dir=str(observer.host_logs.parent / 'runs' / (group + '-unselected')))
        for group in ('git', 'pf')])


def test_formal_selected_pair_monitors_only_anchored_run_paths(observer):
    formal_pair(observer)
    selected = observer.log()
    append(selected, dict(event='request', call_id='selected'))
    excluded = observer.log(role='git-unselected')
    append(excluded, request(observer, endpoint='https://wrong.example/chat/completions'))
    health = observer.poll()
    assert health['status'] == 'healthy' and not observer.hold.exists()
    assert health['runs'] == observer.expected['runs']
    assert set(health['progress']) == {'git', 'pf'}
    assert set(health['coverage']) == {str(selected)}


def test_attempt_missing_status_is_reported(observer):
    path = observer.log()
    path.write_bytes(b'')
    (path.parents[3] / 'attempt.json').write_text('{}')
    health = observer.poll()
    assert health['status'] == 'healthy' and not observer.hold.exists()
    assert health['checks'] == ['run_progress:git:KeyError']


def test_run_fault_scope_and_freeze_and_quota_checks_are_preserved(observer):
    path = observer.log()
    path.write_bytes(b'')
    fault = path.parent.parent / 'evidence.fault.json'
    fault.write_text('{}')
    health = observer.poll()
    assert health['status'] == 'healthy' and not observer.hold.exists()
    assert health['run_faults'] == [dict(run_id='git', path=str(fault))]
    assert not (Path(observer.manifest['runs'][0]['output_dir']) / 'attempt-001/STOP').exists()
    assert not (Path(observer.manifest['runs'][1]['output_dir']) / 'attempt-001/STOP').exists()
    assert events(observer)[0]['new_faults'] == [str(fault)]
    restarted = json.loads((observer.output / 'state.json').read_text())
    assert observer.poll(state=restarted)['status'] == 'healthy'
    assert len(events(observer)) == 1
    fault.unlink()
    assert observer.poll()['status'] == 'healthy'
    observer.manifest_path.write_text('{"changed":true}')
    health = observer.poll()
    assert health['checks'] == []
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
