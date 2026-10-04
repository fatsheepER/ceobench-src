"""Real local HTTP/files/processes with a synthetic world; no provider calls."""
from contextlib import closing
import json
from pathlib import Path
import sys
import shutil
import urllib.request

import pytest

from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
from saas_bench.execution_capture import CapturedText, text_sources, model_request
from saas_bench.sql_evidence import SQLEvidenceStore
from test_sql_evidence import identity, event_ids, settled
from test_public_sql import server


@pytest.fixture
def captured(server, tmp_path):
    store = SQLEvidenceStore(tmp_path / 'private/evidence.sqlite', identity(capture_scope='execution'))
    server.sql_evidence = store
    server.start()
    workspace = tmp_path / 'workspace'
    workspace.mkdir(exist_ok=True)
    shutil.copytree(Path(__file__).parents[1] / 'src/saas_bench/novamind_api', workspace / 'docs/novamind_api', ignore=shutil.ignore_patterns('__pycache__'))
    executor = BashAgentToolExecutor(workspace, evidence_store=store,
        env={'NOVAMIND_API_PORT': str(server.port),
             'PYTHONPATH': str(workspace / 'docs')})
    return server, store, executor


def records(store):
    return [store.read_event(e) for e in event_ids(store)]


def test_bash_sdk_receipts_files_and_model_occurrences(captured, monkeypatch):
    api, store, executor = captured
    import time
    timings = {}
    def timed(obj, name, label):
        function = getattr(obj, name)
        def run(*args, **kwargs):
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                timings[label] = timings.get(label, 0) + time.perf_counter() - started
        monkeypatch.setattr(obj, name, run)
    from saas_bench import execution_capture, sql_evidence
    timed(execution_capture.ExecutionCapture, 'snapshot', 'boundary_scan_inclusive')
    timed(store, 'version', 'version_write_inclusive')
    timed(sql_evidence, 'digest', 'sha256')
    timed(execution_capture, 'digest', 'sha256')
    timed(store, 'received', 'client_callback_storage')
    timed(execution_capture, 'model_request', 'model_request_mapping')
    started = time.perf_counter()
    command = '''python - <<'CODE'
import novamind_api
import json
from pathlib import Path
import time
from novamind_api import _capture
callback_times = []
submit = _capture.submit
def measured(record):
    started = time.perf_counter()
    try:
        return submit(record)
    finally:
        callback_times.append(time.perf_counter() - started)
_capture.submit = measured
first = novamind_api.query('SELECT amount FROM ledger')
second = novamind_api.query('SELECT 99 AS hidden_print')
Path('forecast.json').write_text(json.dumps(first))
print(first['rows'][0]['amount'])
Path('callback-times.json').write_text(json.dumps(callback_times))
CODE'''
    result = executor.execute('bash', {'command': command})
    timings['captured_bash_total'] = time.perf_counter() - started
    capture_timings = dict(timings)
    settled(api)
    store.assert_healthy()
    rs = records(store)
    assert [r['request']['kind'] for r in rs] == ['bash', 'sql_query', 'sql_query']
    root = rs[0]['request']['event_id']
    assert all(r['request']['parent_event_id'] == root for r in rs[1:])
    assert '99' not in result
    assert store.get_content(root + ':stdout_bytes')[1] == result.encode()
    assert 'forecast.json' in rs[0]['result']['changed_paths']
    query = rs[1]['request']['event_id']
    assert json.loads(store.get_content(query + ':client_returned')[1])['row_count']
    timings['client_callback_roundtrip'] = sum(json.loads((executor.workspace_path / 'callback-times.json').read_text()))
    forecast = (executor.workspace_path / 'forecast.json').read_bytes()
    api.sql_evidence = None
    started = time.perf_counter()
    try:
        baseline = BashAgentToolExecutor(executor.workspace_path, env=executor.extra_env).execute('bash', {'command': command})
        timings['uncaptured_bash_total'] = time.perf_counter() - started
        assert baseline == result
        assert (executor.workspace_path / 'forecast.json').read_bytes() == forecast
    finally:
        settled(api)
        api.sql_evidence = store
    import httpx
    from openai import OpenAI
    from saas_bench.model_usage import ModelUsage
    from test_preflight_usage import reply
    recorder = ModelUsage(None, 'agent', evidence_store=store)
    with recorder.attach(OpenAI(api_key='offline-only', max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=reply('chat')))))) as client:
        request = dict(model='test-model', messages=[{'role': 'tool', 'tool_call_id': 'fixture-tool', 'content': result}])
        recorder.call('chat', request, lambda: client.chat.completions.create(**request))
    timings = dict(capture_timings, **{name: timings[name] for name in
                   ('client_callback_roundtrip', 'uncaptured_bash_total', 'model_request_mapping')})
    event = event_ids(store)[-1]
    occurrences = json.loads(store.get_content(event + ':occurrences')[1])
    assert occurrences and all(o['version_id'].startswith(root + ':') for o in occurrences)
    store.assert_healthy()
    import os
    destination = os.environ.get('CEOBENCH_CAPTURE_ARTIFACTS')
    if destination:
        output = Path(destination)
        output.mkdir(parents=True, exist_ok=True)
        store.snapshot(output / 'execution.sqlite')
        (output / 'request-mapping.json').write_text(json.dumps(occurrences, indent=2, ensure_ascii=False))
        (output / 'request.json').write_bytes(store.get_content(event + ':wire')[1])
        (output / 'validation.json').write_text(json.dumps(dict(
            evidence_kind='actually_executed_local_fixture', model_transport='real SDK with MockTransport',
            external_provider_calls=0, python=sys.version, platform=sys.platform,
            timings_seconds=timings, timing_note='inclusive phases overlap; no timeout values changed',
            events=records(store)), indent=2, ensure_ascii=False))


def test_files_boundaries_and_exact_source_ranges(captured, tmp_path):
    api, store, executor = captured
    ws = executor.workspace_path
    (ws / 'x.txt').write_bytes('第一行\r\n第二行\r\n'.encode())
    value = executor.execute('read_file', {'path': 'x.txt', 'offset': 2, 'limit': 1})
    assert value == '     2\t第二行'
    field = value.origins[1]
    assert store.get_content(field['version_id'])[1].decode()[slice(*field['source_range'])] == '第二行'
    assert field['request_range'] == [7, 10]
    assert executor.execute('read_file', {'path': 'x.txt', 'offset': 0}).startswith('Error:')
    assert '(3 bytes)' in executor.execute('write_file', {'path': 'z.txt', 'content': '中'})
    edited = executor.execute('edit_file', {'path': 'x.txt', 'old_string': '第二行', 'new_string': 'changed'})
    assert edited == 'File edited: x.txt'
    assert (ws / 'x.txt').read_bytes() == '第一行\nchanged\n'.encode()
    outside = tmp_path / 'secret.txt'
    outside.write_text('SECRET')
    (ws / 'escape').symlink_to(outside)
    assert 'SECRET' not in executor.execute('search_files', {'pattern': 'SECRET'})
    assert executor.execute('glob_files', {'pattern': '../*'}).startswith('Error:')
    store.assert_healthy()


def test_http_business_failure_and_unknown_bare_client(captured):
    api, store, executor = captured
    req = urllib.request.Request(f'http://127.0.0.1:{api.port}/call',
                                 data=b'{"tool":"does_not_exist"}')
    with urllib.request.urlopen(req) as response:
        raw = response.read()
    settled(api)
    row, = records(store)
    assert json.loads(raw)['success'] is False
    assert row['result']['status'] == 'failed'
    assert row['request']['parent_reason'] == 'unpropagated_context'
    assert row['delivery']['receive_state'] == 'unknown'
    store.assert_healthy()


def test_streams_truncation_failure_and_capture_equivalence(captured):
    _, store, executor = captured
    command = "python -c \"import sys; sys.stdout.write('中'*31000); sys.stderr.write('bad'); sys.exit(3)\""
    actual = executor.execute('bash', {'command': command})
    expected = BashAgentToolExecutor(executor.workspace_path).execute('bash', {'command': command})
    assert actual == expected
    row = records(store)[0]
    assert row['result']['exit_code'] == 3
    assert row['result']['status'] == 'failed'
    root = row['request']['event_id']
    assert len(store.get_content(root + ':stdout')[1].decode()) == 31000
    assert store.get_content(root + ':stderr_bytes')[1] == b'bad'
    for source in actual.origins:
        text = store.get_content(source['version_id'])[1].decode()
        assert text[slice(*source['source_range'])] == actual[slice(*source['request_range'])]


def test_finished_bash_timeout_is_distinct_from_unknown_outcome(captured):
    _, store, executor = captured
    executor.bash_timeout = .1
    result = executor.execute('bash', {'command': 'echo before-timeout; sleep 30'})
    row = records(store)[0]
    assert 'before-timeout' in result and 'timed out' in result
    assert row['result']['status'] == 'timed_out'
    assert row['result']['timed_out'] is True
    assert row['result']['exit_code'] is not None
    store.assert_healthy()


def test_running_week_after_timeout_remains_unknown(captured):
    import threading
    from types import SimpleNamespace
    api, store, _ = captured
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    def step_week():
        started.set()
        try:
            release.wait(5)
            return SimpleNamespace(day=7)
        finally:
            finished.set()
    api.simulator = SimpleNamespace(step_week=step_week)
    api.STEP_WEEK_TIMEOUT = .1
    body = dict(rationale='offline timeout check', predictions={
        name: dict(point=10, lower=0, upper=20) for name in
        ('cash_1wk', 'cash_4wk', 'cash_12wk', 'cash_26wk')})
    try:
        request = urllib.request.Request(f'http://127.0.0.1:{api.port}/next-week',
                                         data=json.dumps(body).encode())
        with urllib.request.urlopen(request, timeout=3) as response:
            assert json.load(response)['error'] == 'step_week_timeout'
        assert started.is_set() and not finished.is_set()
        settled(api)
        row = records(store)[0]
        event = row['request']['event_id']
        assert row['result']['status'] == 'result_unknown'
        assert row['result']['completed_at'] is None
        assert json.loads(store.get_content(event + ':public_response')[1])['error'] == 'step_week_timeout'
        with pytest.raises(RuntimeError, match='Unconfirmed'):
            store.assert_healthy()
        with pytest.raises(RuntimeError, match='Unconfirmed'):
            api.checkpoint(0)
    finally:
        release.set()
        assert finished.wait(3)


@pytest.mark.parametrize('detached', [False, pytest.param(True, marks=pytest.mark.skipif(sys.platform != 'linux', reason='Linux subreaper acceptance'))])
def test_background_descendant_pauses_and_preserves_scene(captured, detached):
    import os
    import signal
    from saas_bench.agents.bash_agent.tools import ProcessBoundaryError
    api, store, executor = captured
    executor.bash_timeout = .3
    try:
        with pytest.raises(ProcessBoundaryError, match='descendants did not finish before the command deadline'):
            command = 'sleep 60 >/dev/null 2>&1 & echo parent-returned'
            if detached:
                command = '''python -c "import subprocess;subprocess.Popen(['sleep','60'], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"'''
            executor.execute('bash', {'command': command})
        assert executor.preserved_process.poll() is None
        row = records(store)[0]
        assert row['result']['status'] == 'result_unknown'
        assert row['result']['process_boundary']['children']
        assert json.loads(store.fault_path.read_bytes())['preserve_scene']
        with pytest.raises(KeyError):
            store.get_content(row['request']['event_id'] + ':tool_return')
        with pytest.raises(RuntimeError, match='capture failed'):
            api.checkpoint(0)
    finally:
        if executor.preserved_process:
            os.killpg(executor.preserved_process.pid, signal.SIGKILL)
            executor.preserved_process.communicate(timeout=5)


@pytest.mark.parametrize('detached', [False, True])
def test_descendants_finish_and_streams_drain_after_bash_error(captured, detached):
    import shlex
    api, store, executor = captured
    executor.bash_timeout = 5
    child = "import time,sys;time.sleep(.08);print('x'*200000);print('child-finished',file=sys.stderr)"
    if detached:
        launch = ('import subprocess;subprocess.Popen(["python", "-c", ' + repr(child)
                  + '], start_new_session=True)')
        command = 'python -c ' + shlex.quote(launch) + '; exit 2'
    else:
        command = 'python -c ' + shlex.quote(child) + ' & exit 2'
    result = executor.execute('bash', {'command': command})
    assert 'child-finished' in result and '[exit code: 2]' in result
    row = records(store)[0]
    assert row['result']['status'] == 'failed'
    assert row['result']['process_boundary']['children'] == []
    assert len(store.get_content(row['request']['event_id'] + ':stdout_bytes')[1]) == 200001
    assert executor.preserved_process is None
    store.assert_healthy()
    assert executor.execute('bash', {'command': 'echo next-command'}) == 'next-command\n'


def test_raw_decode_failure_partial_write_and_limits(captured, monkeypatch):
    api, store, executor = captured
    output = executor.execute('bash', {'command': "python -c \"import sys;sys.stdout.buffer.write(b'\\xff')\""})
    assert output.startswith('Error:')
    event = event_ids(store)[0]
    assert store.get_content(event + ':stdout_bytes')[1] == b'\xff'
    assert store.read_event(event)['result']['exit_code'] == 0
    path = executor.workspace_path / 'partial.txt'
    original = Path.write_text
    def partial(self, text, *args, **kwargs):
        if self == path:
            self.write_bytes(b'partial')
            raise OSError('injected short write')
        return original(self, text, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', partial)
    assert 'injected short write' in executor.execute('write_file', {'path': 'partial.txt', 'content': 'complete'})
    event = event_ids(store)[-1]
    snapshot = json.loads(store.get_content(event + ':workspace_after')[1])
    assert store.get_content(snapshot['partial.txt']['version'])[1] == b'partial'
    assert store.read_event(event)['result']['status'] == 'failed'
    (executor.workspace_path / 'many.txt').write_bytes(b'match\n' * 250)
    text = executor.execute('search_files', {'pattern': 'match', 'path': 'many.txt'})
    assert 'limit' in text.lower() or 'truncated' in text.lower()
    store.assert_healthy()


def test_capture_storage_failure_preserves_business_output(captured, monkeypatch):
    _, store, executor = captured
    original = store.version
    def fail(event, slot, *args, **kwargs):
        if slot == 'stdout_bytes':
            raise OSError('injected evidence disk failure')
        return original(event, slot, *args, **kwargs)
    monkeypatch.setattr(store, 'version', fail)
    assert executor.execute('bash', {'command': 'echo completed > result.txt; echo original-return'}) == 'original-return\n'
    assert (executor.workspace_path / 'result.txt').read_text() == 'completed\n'
    with pytest.raises(RuntimeError, match='capture failed'):
        store.assert_healthy()


def test_unknown_terminal_and_endpoint_is_append_only(captured):
    import urllib.error
    api, store, _ = captured
    for path in ('/_capture', '/_capture?event=test-run/prefix/1'):
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f'http://127.0.0.1:{api.port}' + path)
        assert error.value.code == 404
    settled(api)
    event = store.begin_event('bash', {})
    store.complete(event, 'result_unknown')
    with pytest.raises(RuntimeError, match='Unconfirmed'):
        store.assert_healthy()


def test_checkpoint_drains_parent_while_allowing_child_callbacks(captured, tmp_path):
    import time
    from concurrent.futures import ThreadPoolExecutor
    api, store, executor = captured
    api.checkpoint_callback = lambda: store.snapshot(tmp_path / 'drained.sqlite')
    with ThreadPoolExecutor(3) as pool:
        running = pool.submit(executor.execute, 'bash', {'command': "python -c \"import time;time.sleep(.3);import novamind_api as nm;print(nm.query('SELECT 1 AS n'))\""})
        deadline = time.monotonic() + 3
        while not event_ids(store) and time.monotonic() < deadline:
            time.sleep(.01)
        assert event_ids(store)
        checkpoint = pool.submit(api.checkpoint, 0)
        with api._sql_lock:
            assert api._sql_lock.wait_for(lambda: api._sql_paused, timeout=3)
        later = BashAgentToolExecutor(executor.workspace_path, evidence_store=store)
        queued = pool.submit(later.execute, 'write_file', {'path': 'later.txt', 'content': 'after checkpoint'})
        assert 'row_count' in running.result(timeout=10)
        saved = checkpoint.result(timeout=10)
        assert saved['cutoff'] == 2
        assert queued.result(timeout=10) == 'File written: later.txt (16 bytes)'
    store.assert_healthy()


def test_model_retries_identical_sources_and_unsent_result(captured, monkeypatch):
    import httpx
    from openai import OpenAI
    from saas_bench.model_usage import ModelUsage
    from test_preflight_usage import reply
    _, store, executor = captured
    monkeypatch.setattr('time.sleep', lambda _: None)
    texts = []
    for name in ('a', 'b', 'unsent'):
        (executor.workspace_path / name).write_text('same text')
        texts.append(executor.execute('read_file', {'path': name}))
    attempts = []
    def handle(request):
        attempts.append(request.read())
        return httpx.Response(500, json={'error': {'message': 'retry'}}) if len(attempts) == 1 else httpx.Response(200, json=reply('chat'))
    usage = ModelUsage(None, 'agent', evidence_store=store)
    with usage.attach(OpenAI(api_key='offline-only', max_retries=1,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)))) as client:
        request = dict(model='test-model', messages=[dict(role='user', content=x) for x in texts[:2]])
        usage.call('chat', request, lambda: client.chat.completions.create(**request))
    events = [r for r in records(store) if r['request']['kind'] == 'model_request']
    assert len(events) == 2
    assert [r['result']['http_status'] for r in events] == [500, 200]
    assert len({r['request']['request']['attempt_id'] for r in events}) == 2
    assert len({r['request']['request']['call_id'] for r in events}) == 1
    for row in events:
        mapped = json.loads(store.get_content(row['request']['event_id'] + ':occurrences')[1])
        assert {m['json_pointer'] for m in mapped if m['version_id'] == texts[0].origins[0]['version_id']} == {'/messages/0/content'}
        assert {m['json_pointer'] for m in mapped if m['version_id'] == texts[1].origins[0]['version_id']} == {'/messages/1/content'}
        assert not any(m['version_id'] in {s['version_id'] for s in texts[2].origins} for m in mapped)
    store.assert_healthy()


def test_memory_strip_limit_and_dashboard_slice_keep_exact_origins(captured):
    from types import SimpleNamespace
    from test_preflight_context import agent
    _, store, executor = captured
    value = agent(executor.workspace_path)
    value.usage_recorder = SimpleNamespace(evidence_store=store)
    (executor.workspace_path / 'MEMORY.md').write_bytes((' \r\n' + '中' * 40000 + 'tail \r\n').encode())
    prompt = value._get_system_prompt_with_memory()
    assert '中' * 40000 in prompt and 'tail' not in prompt
    assert len(prompt.origins) == 1
    for item in prompt.origins:
        source = store.get_content(item['version_id'])[1].decode()
        assert source[slice(*item['source_range'])] == prompt[slice(*item['request_range'])]
    result = executor.execute('bash', {'command': "printf 'intro\n=== Week 2 Dashboard (Day 14) ===\nresult'"})
    assert value.check_day_advanced(result)
    for item in value.new_dashboard.origins:
        source = store.get_content(item['version_id'])[1].decode()
        assert source[slice(*item['source_range'])] == value.new_dashboard[slice(*item['request_range'])]
    store.assert_healthy()


def test_second_collection_timeout_retains_raw_streams(captured, monkeypatch):
    import subprocess
    from saas_bench.process_boundary import Boundary
    from saas_bench.agents.bash_agent.tools import NextDayTimeoutError
    _, store, executor = captured
    real_communicate = subprocess.Popen.communicate
    def timeout(self, *args, **kwargs):
        if kwargs.get('timeout') == 5:
            # Reap this fixture's killed process before injecting a failed second collection.
            real_communicate(self, timeout=5)
            raise subprocess.TimeoutExpired(self.args, 5, output=b'partial-out', stderr=b'partial-err')
        return real_communicate(self, *args, **kwargs)
    monkeypatch.setattr(subprocess.Popen, 'communicate', timeout)
    executor.bash_timeout = .1
    with pytest.raises(NextDayTimeoutError, match='cleanup'):
        executor.execute('bash', {'command': 'sleep 60'})
    event = event_ids(store)[0]
    meta, raw = store.get_content(event + ':stdout_bytes')
    assert raw == b'partial-out' and meta['extent'] == 'partial'
    assert store.read_event(event)['result']['status'] == 'result_unknown'
    with pytest.raises(RuntimeError):
        store.assert_healthy()


def test_capture_channel_failure_keeps_business_result_and_refuses_next_call(captured, monkeypatch):
    api, store, executor = captured
    def fail(*args, **kwargs):
        raise OSError('injected capture channel failure')
    monkeypatch.setattr(store, 'received', fail)
    result = executor.execute('bash', {'command': "python -c \"import novamind_api as n;print(n.query('SELECT 7 AS n'))\""})
    assert "'n': 7" in result
    with pytest.raises(RuntimeError, match='capture failed'):
        executor.execute('bash', {'command': 'echo must-not-execute > unexpected'})
    assert not (executor.workspace_path / 'unexpected').exists()
    with pytest.raises(RuntimeError, match='capture failed'):
        api.checkpoint(0)


def test_batch_partial_receipt_and_public_object_fields(captured, monkeypatch):
    from saas_bench.tools import ToolResult
    api, store, _ = captured
    monkeypatch.setattr(api, 'execute_tool', lambda tool, args: ToolResult(True, 'batch result',
        dict(results=[dict(customer_id=42, success=True), dict(customer_id=43, error='rejected')])))
    with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{api.port}/call',
            data=b'{"tool":"reject_enterprise_deal","args":{"deals":[{"customer_id":42}]}}')) as response:
        assert json.load(response)['success']
    settled(api)
    event = event_ids(store)[0]
    assert store.read_event(event)['result']['status'] == 'partially_succeeded'
    meta, _ = store.get_content(event + ':public_response')
    assert any(o['kind'] == 'customer' and o['value'] == 42 and o['basis'] == 'public_response_field' for o in meta['objects'])
    assert any(o['basis'] == 'public_request_target' for o in meta['objects'])


def test_empty_delete_glob_cap_and_invalid_limit(captured):
    _, store, executor = captured
    ws = executor.workspace_path
    (ws / 'empty').touch()
    assert executor.execute('read_file', {'path': 'empty'}) == '     1\t'
    for limit in (True, 0, -1, 1.5):
        assert executor.execute('read_file', {'path': 'empty', 'limit': limit}).startswith('Error:')
    for i in range(201):
        (ws / f'candidate-{i:03}').touch()
    text = executor.execute('glob_files', {'pattern': 'candidate-*'})
    assert len(text.splitlines()) == 201 and 'truncated' in text
    event = event_ids(store)[-1]
    assert store.read_event(event)['outputs'] == [event + ':tool_return']
    executor.execute('bash', {'command': 'rm empty'})
    event = event_ids(store)[-1]
    before = json.loads(store.get_content(event + ':workspace_before')[1])
    after = json.loads(store.get_content(event + ':workspace_after')[1])
    assert 'empty' in before and 'empty' not in after
    assert store.get_content(before['empty']['version'])[1] == b''
    store.assert_healthy()


def test_start_failure_and_flag_conflict(captured, monkeypatch):
    import subprocess
    from saas_bench.agents.bash_agent.run_test import BashAgentRunner
    _, store, executor = captured
    with pytest.raises(ValueError, match='requires SQL'):
        BashAgentRunner(execution_capture=True, sql_capture=False)
    def fail(*args, **kwargs):
        raise OSError('injected process launch failure')
    monkeypatch.setattr(subprocess, 'Popen', fail)
    result = executor.execute('bash', {'command': 'echo cannot-start'})
    assert result == 'Error: injected process launch failure'
    row = records(store)[0]
    assert row['result']['status'] == 'failed' and row['result']['process_started'] is False
    store.assert_healthy()


def test_package_cli_keeps_sdk_value_and_display_projection(captured, monkeypatch, capsys):
    from types import SimpleNamespace
    from saas_bench import novamind_cli
    api, store, executor = captured
    event = store.begin_event('cli_test', {})
    monkeypatch.setenv('NOVAMIND_API_PORT', str(api.port))
    monkeypatch.setenv('NOVAMIND_CAPTURE_CONTEXT', store.context(event))
    novamind_cli._cmd_list_daily_scripts(SimpleNamespace())
    display = capsys.readouterr().out
    settled(api)
    child = event_ids(store)[-1]
    assert json.loads(store.get_content(child + ':client_returned')[1]) == {'success': True, 'data': {'scripts': []}}
    projection = json.loads(store.get_content(child + ':client_projection')[1])
    assert ''.join(item['text'] for item in projection if item['stream'] == 'stdout') == display
    store.complete(event)
    store.assert_healthy()


def test_weekly_script_failure_keeps_child_outcome(captured):
    api, store, executor = captured
    api.script_workspace = executor.workspace_path
    api.set_daily_scripts({'failure': "import sys;print('partial script output');sys.exit(3)"})
    output = api._run_daily_scripts_internal()['failure']
    assert 'partial script output' in output and '[exit code: 3]' in output
    execution = next(r for r in records(store) if r['request']['kind'] == 'registered_script_execution')
    assert execution['result']['status'] == 'failed'
    assert store.read_event(execution['result']['child_event_id'])['result']['exit_code'] == 3
    store.assert_healthy()
