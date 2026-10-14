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


def test_sql_timeout_after_bash_exit_and_broken_headers_has_one_response(captured, monkeypatch):
    import io
    from types import SimpleNamespace
    from saas_bench import api_server
    api, store, _ = captured
    bash = store.begin_event('bash', {'command': 'timeout 280 python script.py'})
    context = store.context(bash)
    raw = b'{"sql":"SELECT amount FROM ledger"}'
    handler = api_server._APIHandler.__new__(api_server._APIHandler)
    handler.server = SimpleNamespace(_api_server=api, _role='ceo')
    handler.path, handler.command, handler.request_version = '/query', 'POST', 'HTTP/1.1'
    handler.headers = {'Content-Length': str(len(raw)), 'X-Capture-Context': context,
                       'X-Capture-Call': 'a' * 32}
    handler.rfile = io.BytesIO(raw)
    handler.connection = SimpleNamespace(settimeout=lambda _: None)
    handler._headers_buffer = []
    statuses = []
    handler.send_response = statuses.append
    handler.send_header = lambda key, value: handler._headers_buffer.append(f'{key}: {value}\r\n'.encode())
    def broken_headers(_):
        raise BrokenPipeError('client killed before SQL timeout response')
    handler.wfile = SimpleNamespace(write=broken_headers)
    def timeout(api, sql, metadata):
        store.complete(bash, 'failed', **closed_bash_result())
        metadata.update(day=105, error_kind='timeout')
        raise TimeoutError('Query time limit exceeded')
    monkeypatch.setattr(api_server, 'execute_query', timeout)
    handler.do_POST()
    assert statuses == [504]
    assert not store.fault_path.exists() and store.fault is None
    query = next(row for row in records(store) if row['request']['kind'] == 'sql_query')
    assert query['request']['parent_event_id'] == bash
    assert query['result']['status'] == 'timed_out'
    assert query['delivery']['send_state'] == 'failed'
    assert query['client_receipt'] == {'receive_state': 'unknown'}
    assert query['client_interruption'] is None
    assert handler.close_connection
    with pytest.raises(RuntimeError, match='Unconfirmed client receive outcome'):
        store.assert_healthy()


@pytest.mark.parametrize('inherited_pipe', [False, True])
def test_public_cli_timeout_has_known_failed_child_and_allows_next_tool(captured, inherited_pipe):
    api, store, executor = captured
    source = Path(__file__).parents[1] / 'src/saas_bench'
    shutil.copy(source / '_public_cli.py', executor.workspace_path / 'cli.py')
    shutil.copy(source / 'novamind_api/_capture.py', executor.workspace_path / '_client_capture.py')
    code = "import sys;sys.stdout.buffer.write(b'partial-out\\xff');sys.stdout.flush();sys.stderr.write('partial-err');sys.stderr.flush();"
    code += ("import subprocess;subprocess.Popen([sys.executable,'-c','import time;time.sleep(.5)'])"
             if inherited_pipe else "\nwhile True: pass")
    driver = '''import os, subprocess
import cli
children = []
communicate = subprocess.Popen.communicate
def shortened(self, *args, **kwargs):
    if kwargs.get('timeout') == 300:
        children.append(self)
        kwargs['timeout'] = .2
    return communicate(self, *args, **kwargs)
subprocess.Popen.communicate = shortened
try:
    cli._execute_python('__env__', int(os.environ['NOVAMIND_API_PORT']), CODE, 'timeout.py')
except subprocess.TimeoutExpired as exc:
    child, = children
    assert child.returncode == EXPECTED
    assert exc.stdout == b'partial-out\\xff' and exc.stderr == b'partial-err'
    try:
        os.kill(child.pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError('Python child was not reaped')
    raise
'''.replace('CODE', repr(code)).replace('EXPECTED', '0' if inherited_pipe else '-9')
    (executor.workspace_path / 'driver.py').write_text(driver)
    executor.bash_timeout = 5
    result = executor.execute('bash', {'command': 'python driver.py'})
    assert 'TimeoutExpired' in result and '[exit code: 1]' in result, result
    assert 'AssertionError' not in result, result
    settled(api)
    rows = records(store)
    child = next(row for row in rows if row['request']['kind'] == 'cli_python')
    assert child['result']['status'] == 'failed'
    assert child['result']['exit_code'] == (0 if inherited_pipe else -9)
    assert child['result']['error'] == 'timeout'
    event = child['request']['event_id']
    assert store.get_content(event + ':stdout_bytes')[1] == b'partial-out\xff'
    assert store.get_content(event + ':stderr_bytes')[1] == b'partial-err'
    assert rows[0]['result']['process_boundary']['children'] == []
    assert executor.preserved_process is None
    store.assert_healthy()
    assert executor.execute('bash', {'command': 'echo next-tool'}) == 'next-tool\n'


@pytest.mark.parametrize('sql_first', [True, False], ids=['sql_before_python', 'sql_after_bash'])
def test_public_cli_timeout_settles_inflight_sql_without_faking_receive(captured, monkeypatch, tmp_path, sql_first):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from saas_bench import public_sql
    api, store, executor = captured
    source = Path(__file__).parents[1] / 'src/saas_bench'
    shutil.copy(source / '_public_cli.py', executor.workspace_path / 'cli.py')
    shutil.copy(source / 'novamind_api/_capture.py', executor.workspace_path / '_client_capture.py')
    started, release = threading.Event(), threading.Event()
    marker = executor.workspace_path / 'sql-started'
    execute_snapshot = public_sql.execute_snapshot
    def blocking_snapshot(conn, sql, deadline, metadata):
        def block():
            if not started.is_set():
                marker.write_text('executing')
                started.set()
                assert release.wait(10), 'SQL gate was not released'
            return 0
        conn.set_progress_handler(block, 1)
        try:
            return execute_snapshot(conn, sql, deadline, metadata)
        finally:
            conn.set_progress_handler(None, 0)
    monkeypatch.setattr(public_sql, 'execute_snapshot', blocking_snapshot)
    complete = store.complete
    def ordered_complete(event, *args, **kwargs):
        if sql_first and store.read_event(event)['request']['kind'] == 'cli_python':
            release.set()
            settled(api)
        return complete(event, *args, **kwargs)
    monkeypatch.setattr(store, 'complete', ordered_complete)
    driver = '''import os, subprocess, time
from pathlib import Path
import cli
communicate = subprocess.Popen.communicate
def shortened(self, *args, **kwargs):
    if kwargs.get('timeout') == 300:
        deadline = time.monotonic() + 5
        while not Path('sql-started').exists():
            assert time.monotonic() < deadline, 'SQL did not start'
            time.sleep(.01)
        kwargs['timeout'] = .1
    return communicate(self, *args, **kwargs)
subprocess.Popen.communicate = shortened
cli._execute_python('__env__', int(os.environ['NOVAMIND_API_PORT']),
    "import novamind_api as nm; nm.query('SELECT amount FROM ledger')", 'blocking.py')
'''
    (executor.workspace_path / 'driver.py').write_text(driver)
    executor.bash_timeout = 5
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(executor.execute, 'bash', {'command': 'python driver.py'})
        try:
            assert started.wait(5), 'SQLite query did not enter its progress handler'
            output = running.result(timeout=10)
            assert 'TimeoutExpired' in output and '[exit code: 1]' in output, output
            assert 'AssertionError' not in output, output
            bash, python, sql = records(store)
            assert python['result']['exit_code'] == -9
            assert python['result']['error'] == 'timeout'
            assert bash['result']['process_boundary']['phase'] == 'closed'
            assert bash['result']['process_boundary']['children'] == []
            assert sql['client_receipt'] == {'receive_state': 'unknown'}
            if not sql_first:
                assert sql['result']['status'] == 'result_unknown'
                assert sql.get('client_interruption') is None
                with pytest.raises(RuntimeError, match='Unconfirmed SQL evidence outcome'):
                    store.assert_healthy()
            release.set()
            settled(api)
            store.assert_healthy()
            sql_event = sql['request']['event_id']
            sql = store.read_event(sql_event)
            assert sql['result']['status'] == 'succeeded'
            assert sql['delivery']['send_state'] in ('sent', 'failed')
            assert sql['client_receipt'] == {'receive_state': 'unknown'}
            interruption = sql['client_interruption']
            assert interruption['state'] == 'interrupted'
            assert interruption['receive_state'] == 'unknown'
            assert interruption['reason'] == 'python_timeout_kill'
            assert interruption['python_event_id'] == python['request']['event_id']
            assert interruption['bash_event_id'] == bash['request']['event_id']
            with closing(store.connect()) as conn:
                received, = conn.execute('SELECT received FROM client_calls WHERE event_id=?', (sql_event,)).fetchone()
            assert received is None
            assert executor.execute('bash', {'command': 'echo next-tool'}) == 'next-tool\n'
            api.checkpoint_callback = lambda: store.snapshot(tmp_path / 'checkpoint.sqlite')
            assert api.checkpoint(0)['cutoff'] == len(event_ids(store))
            restored = SQLEvidenceStore(tmp_path / 'checkpoint.sqlite', store.identity)
            restored.assert_healthy()
            assert restored.read_event(sql_event)['client_interruption'] == interruption
            assert restored.read_event(sql_event)['client_receipt'] == {'receive_state': 'unknown'}
        finally:
            release.set()
            running.result(timeout=10)
            settled(api)


@pytest.mark.parametrize('capture', [False, True])
@pytest.mark.parametrize('state', ['healthy', 'timed_out', 'failed', 'in_progress'])
def test_bash_status_probe_reads_live_flags_without_world_metrics(captured, monkeypatch, capture, state):
    from saas_bench import database
    api, _, _ = captured
    if not capture:
        api.sql_evidence = None
    api._step_day_timed_out = state == 'timed_out'
    api._operation_failed = state == 'failed'
    if state == 'in_progress':
        api._advance_lock.acquire()
    try:
        def unexpected_read(conn):
            pytest.fail('Status flag probe read world metrics')
        with monkeypatch.context() as blocked:
            blocked.setattr(database, 'get_cash', unexpected_read)
            blocked.setattr(database, 'get_active_subscriber_count', unexpected_read)
            request = urllib.request.Request(f'http://127.0.0.1:{api.port}/game-status',
                                             headers={'X-Harness-Status': 'flags'})
            with urllib.request.urlopen(request, timeout=3) as response:
                assert json.load(response) == dict(timed_out=state == 'timed_out',
                    operation_failed=state == 'failed', operation_in_progress=state == 'in_progress')
    finally:
        if state == 'in_progress':
            api._advance_lock.release()
        api._step_day_timed_out = False
        api._operation_failed = False
    with urllib.request.urlopen(f'http://127.0.0.1:{api.port}/game-status', timeout=3) as response:
        status = json.load(response)
    assert status['cash'] == 42 and status['subscribers'] == 0


@pytest.fixture
def timeout_sql_client(tmp_path):
    from saas_bench.public_sql import PUBLIC_POLICY_VERSION
    store = SQLEvidenceStore(tmp_path / 'evidence.sqlite', identity(capture_scope='execution'))
    bash = store.begin_event('bash', {'command': 'python script.py'})
    python = store.begin_event('cli_python', {'source': 'script.py'}, parent=bash)
    sql = store.begin(b'{"sql":"SELECT 1 AS n"}', PUBLIC_POLICY_VERSION, parent=python)
    token = 'a' * 32
    store.bind_client(sql, token)
    return store, bash, python, sql, token


def closed_bash_result():
    return dict(exit_code=1, process_started=True, boundary_closed=True,
        process_boundary=dict(phase='closed', coverage='subreaper_wait',
                              children=[], unknown=None, exit_code=1))


@pytest.mark.parametrize('last', ['sql', 'python', 'bash', 'delivery'])
@pytest.mark.parametrize('sql_status', [200, 400, 500, 504])
@pytest.mark.parametrize('send_state', ['sent', 'failed'])
def test_timeout_sql_client_settles_when_last_terminal_fact_arrives(timeout_sql_client, tmp_path, last, sql_status, send_state):
    store, bash, python, sql, token = timeout_sql_client
    stages = {
        'sql': lambda: store.finish(sql, sql_status, [],
            b'{"success":true,"columns":["n"],"rows":[{"n":1}]}' if sql_status == 200 else
            b'{"success":false,"error":"query failed"}', {}),
        'python': lambda: store.complete(python, 'failed', exit_code=-9, error='timeout'),
        'bash': lambda: store.complete(bash, 'failed', **closed_bash_result()),
        'delivery': lambda: store.delivered(sql, send_state),
    }
    for name, stage in stages.items():
        if name != last:
            stage()
    assert store.read_event(sql)['client_interruption'] is None
    with pytest.raises(RuntimeError, match='Unconfirmed'):
        store.assert_healthy()
    stages[last]()
    store.assert_healthy()
    row = store.read_event(sql)
    assert row['client_interruption']['state'] == 'interrupted'
    assert row['client_interruption']['receive_state'] == 'unknown'
    assert row['client_interruption']['reason'] == 'python_timeout_kill'
    assert row['client_interruption']['python_event_id'] == python
    assert row['client_interruption']['bash_event_id'] == bash
    assert row['client_receipt'] == {'receive_state': 'unknown'}
    with closing(store.connect()) as conn:
        assert conn.execute('SELECT received FROM client_calls WHERE token=?', (token,)).fetchone()[0] is None
    event = store.begin_event('read_file', {'path': 'next.txt'})
    store.complete(event)
    store.snapshot(tmp_path / 'checkpoint.sqlite')
    restored = SQLEvidenceStore(tmp_path / 'checkpoint.sqlite', store.identity)
    restored.assert_healthy()
    assert restored.read_event(sql)['client_interruption'] == row['client_interruption']


def test_timeout_sql_client_health_settles_preexisting_terminal_records(timeout_sql_client, monkeypatch):
    store, bash, python, sql, _ = timeout_sql_client
    with monkeypatch.context() as delayed:
        delayed.setattr(store, '_settle_interrupted_clients', lambda conn: None)
        store.complete(sql)
        store.delivered(sql, 'failed')
        store.complete(python, 'failed', exit_code=-9, error='timeout')
        store.complete(bash, 'failed', **closed_bash_result())
    assert store.read_event(sql)['client_interruption'] is None
    store.assert_healthy()
    assert store.read_event(sql)['client_interruption']['reason'] == 'python_timeout_kill'
    assert store.read_event(sql)['client_receipt'] == {'receive_state': 'unknown'}


@pytest.mark.parametrize('unsafe', [
    'business_write', 'ordinary_lost_receipt', 'kill_without_timeout', 'python_unknown',
    'wrong_python_kind', 'wrong_bash_kind', 'no_python_parent', 'no_bash_parent',
    'missing_cleanup', 'live_descendants', 'unknown_cleanup', 'missing_children',
    'missing_unknown', 'direct_child_only', 'descendants_not_closed', 'descendants_unknown',
    'boundary_open', 'process_not_started',
    'outer_unknown', 'missing_exit_code', 'exit_code_mismatch', 'sql_unknown',
    'sql_missing', 'sql_unfinished', 'sql_capture_missing', 'delivery_unknown', 'delivery_missing',
])
def test_timeout_sql_client_rejects_missing_or_unsafe_proof(timeout_sql_client, unsafe):
    store, bash, python, sql, token = timeout_sql_client
    python_result = dict(status='failed', exit_code=-9, error='timeout')
    bash_result = dict(status='failed', **closed_bash_result())
    sql_result = dict(status='succeeded')
    delivery = 'failed'
    if unsafe == 'business_write':
        with closing(store.connect()) as conn, conn:
            request = store.read_event(sql)['request']
            request.update(kind='public_http', request={'method': 'POST', 'path': '/call',
                           'tool': 'set_daily_spend'})
            conn.execute('UPDATE requests SET request=? WHERE event_id=?', (json.dumps(request).encode(), sql))
    elif unsafe in ('wrong_python_kind', 'wrong_bash_kind', 'no_python_parent', 'no_bash_parent'):
        event = python if unsafe in ('wrong_python_kind', 'no_bash_parent') else sql if unsafe == 'no_python_parent' else bash
        request = store.read_event(event)['request']
        if unsafe.startswith('wrong_'):
            request['kind'] = 'write_file'
        else:
            request['parent_event_id'] = None
        with closing(store.connect()) as conn, conn:
            conn.execute('UPDATE requests SET request=? WHERE event_id=?', (json.dumps(request).encode(), event))
    elif unsafe == 'ordinary_lost_receipt':
        python_result.update(exit_code=1, error=None)
    elif unsafe == 'kill_without_timeout':
        python_result['error'] = None
    elif unsafe == 'python_unknown':
        python_result.update(status='result_unknown', exit_code=None)
    elif unsafe == 'missing_cleanup':
        bash_result.pop('process_boundary')
    elif unsafe in ('live_descendants', 'unknown_cleanup', 'missing_children', 'missing_unknown',
                    'direct_child_only', 'descendants_not_closed', 'descendants_unknown'):
        boundary = bash_result['process_boundary']
        if unsafe == 'live_descendants':
            boundary['children'] = [12345]
        elif unsafe == 'unknown_cleanup':
            boundary['unknown'] = 'cleanup not confirmed'
        elif unsafe == 'direct_child_only':
            boundary['coverage'] = 'direct_child_and_streams_only'
        elif unsafe in ('descendants_not_closed', 'descendants_unknown'):
            boundary['descendants_closed'] = False if unsafe == 'descendants_not_closed' else None
        else:
            boundary.pop('children' if unsafe == 'missing_children' else 'unknown')
    elif unsafe == 'boundary_open':
        bash_result['boundary_closed'] = False
    elif unsafe == 'process_not_started':
        bash_result['process_started'] = False
    elif unsafe == 'outer_unknown':
        bash_result['status'] = 'result_unknown'
    elif unsafe == 'missing_exit_code':
        bash_result['exit_code'] = None
    elif unsafe == 'exit_code_mismatch':
        bash_result['process_boundary']['exit_code'] = 2
    elif unsafe == 'sql_unknown':
        sql_result['status'] = 'result_unknown'
    elif unsafe == 'sql_unfinished':
        sql_result['completed_at'] = None
    elif unsafe == 'sql_capture_missing':
        sql_result['capture_status'] = 'missing'
    elif unsafe == 'delivery_unknown':
        delivery = 'unknown'
    store.complete(python, **python_result)
    store.complete(bash, **bash_result)
    if unsafe != 'sql_missing':
        store.complete(sql, **sql_result)
    if unsafe != 'delivery_missing':
        store.delivered(sql, delivery)
    with pytest.raises(RuntimeError, match='Unconfirmed'):
        store.assert_healthy()
    assert store.read_event(sql)['client_interruption'] is None
    assert store.read_event(sql)['client_receipt'] == {'receive_state': 'unknown'}
    with closing(store.connect()) as conn:
        assert conn.execute('SELECT received FROM client_calls WHERE token=?', (token,)).fetchone()[0] is None


@pytest.mark.parametrize('tampered', ['parent', 'interruption'])
def test_timeout_sql_client_rechecks_stored_interruption_proof(timeout_sql_client, tampered):
    store, bash, python, sql, token = timeout_sql_client
    store.complete(sql)
    store.delivered(sql, 'failed')
    store.complete(python, 'failed', exit_code=-9, error='timeout')
    store.complete(bash, 'failed', **closed_bash_result())
    store.assert_healthy()
    with closing(store.connect()) as conn, conn:
        if tampered == 'parent':
            result = store.read_event(bash)['result']
            result['process_boundary']['children'] = [12345]
            conn.execute('UPDATE results SET record=? WHERE event_id=?', (json.dumps(result).encode(), bash))
        else:
            record = store.read_event(sql)['client_interruption']
            record['python_event_id'] = bash
            conn.execute('UPDATE client_interruptions SET record=? WHERE token=?', (json.dumps(record).encode(), token))
    with pytest.raises(RuntimeError, match='Unconfirmed client receive outcome'):
        store.assert_healthy()
    assert store.read_event(sql)['client_receipt'] == {'receive_state': 'unknown'}


def test_public_cli_cleanup_failure_keeps_child_outcome_unknown(captured, monkeypatch):
    import subprocess
    from saas_bench import _public_cli
    api, store, executor = captured
    parent = store.begin_event('bash', {})
    monkeypatch.setenv('NOVAMIND_API_PORT', str(api.port))
    monkeypatch.setenv('NOVAMIND_CAPTURE_CONTEXT', store.context(parent))
    monkeypatch.delenv('NOVAMIND_API_SOCKET', raising=False)
    monkeypatch.setattr(_public_cli, '_base_dir', lambda: executor.workspace_path)
    communicate, wait = subprocess.Popen.communicate, subprocess.Popen.wait
    children = []
    def shortened(self, *args, **kwargs):
        assert kwargs['timeout'] == 300
        children.append(self)
        return communicate(self, *args, **dict(kwargs, timeout=.1))
    failed = False
    def failed_wait(self, *args, **kwargs):
        nonlocal failed
        result = wait(self, *args, **kwargs)
        if not failed:
            failed = True
            raise OSError('injected cleanup failure')
        return result
    monkeypatch.setattr(subprocess.Popen, 'communicate', shortened)
    monkeypatch.setattr(subprocess.Popen, 'wait', failed_wait)
    with pytest.raises(OSError, match='injected cleanup failure'):
        _public_cli._execute_python('__env__', api.port, 'while True: pass')
    assert children[0].returncode == -9
    settled(api)
    child = next(row for row in records(store) if row['request']['kind'] == 'cli_python')
    assert child['result']['status'] == 'result_unknown'
    assert child['result']['exit_code'] is None
    assert child['result']['error'] == 'OSError'
    with pytest.raises(RuntimeError, match='capture failed'):
        store.assert_healthy()


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
    cleanup = row['result']['process_boundary']
    assert cleanup['phase'] == 'timeout_cleanup'
    assert cleanup['direct_child_reaped'] is True
    assert cleanup['streams_collected'] is True
    assert cleanup['exit_code'] == row['result']['exit_code']
    assert cleanup['coverage'] == 'direct_child_and_streams_only'
    assert cleanup['descendants_closed'] is None
    assert 'last_supervisor_receipt' in cleanup
    assert 'children' not in cleanup
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
    from contextlib import contextmanager
    from saas_bench import workspace_io
    original = workspace_io.open_file
    @contextmanager
    def partial(root, target, mode='rb'):
        with original(root, target, mode) as stream:
            if target == path and mode == 'wb':
                stream.write(b'partial')
                raise OSError('injected short write')
            yield stream
    monkeypatch.setattr(workspace_io, 'open_file', partial)
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
    monkeypatch.setattr(usage.lifecycle, 'sleep', lambda seconds: None)
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


@pytest.mark.parametrize('collection_error', ['timeout', 'oserror'])
@pytest.mark.parametrize('undecodable', [False, True])
def test_second_collection_timeout_retains_raw_streams(captured, monkeypatch, collection_error, undecodable):
    import subprocess
    from saas_bench.process_boundary import Boundary
    from saas_bench.agents.bash_agent.tools import NextDayTimeoutError
    _, store, executor = captured
    expected_stdout = b'partial-out\xff' if undecodable else b'partial-out'
    real_communicate = subprocess.Popen.communicate
    def timeout(self, *args, **kwargs):
        if kwargs.get('timeout') == 5:
            # Reap this fixture's killed process before injecting a failed second collection.
            real_communicate(self, timeout=5)
            if collection_error == 'oserror':
                raise OSError(5, 'injected stream collection failure')
            raise subprocess.TimeoutExpired(self.args, 5, output=expected_stdout, stderr=b'partial-err')
        return real_communicate(self, *args, **kwargs)
    monkeypatch.setattr(subprocess.Popen, 'communicate', timeout)
    executor.bash_timeout = .1
    command = (r"printf 'partial-out\377'" if undecodable else 'printf partial-out')
    with pytest.raises(NextDayTimeoutError, match='cleanup'):
        executor.execute('bash', {'command': command + '; printf partial-err >&2; sleep 60'})
    event = event_ids(store)[0]
    meta, raw = store.get_content(event + ':stdout_bytes')
    assert raw == expected_stdout and meta['extent'] == 'partial'
    assert store.get_content(event + ':stderr_bytes')[1] == b'partial-err'
    facts = store.read_event(event)['result']
    assert facts['status'] == 'result_unknown'
    assert facts['process_boundary']['streams_collected'] is False
    assert facts['process_boundary']['descendants_closed'] is None
    assert facts['boundary_closed'] is False
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
    monkeypatch.setattr(api, 'execute_tool', lambda tool, args, **kwargs: ToolResult(True, 'batch result',
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


@pytest.mark.parametrize('strict, invalid_utf8', [(True, False), (False, True)])
def test_timeout_cleanup_receipt_survives_unknown_outcome_and_decode_failure(captured, strict, invalid_utf8):
    from saas_bench.agents.bash_agent.tools import NextDayTimeoutError
    _, store, executor = captured
    executor.bash_timeout = .2
    executor.stop_on_timeout = strict
    command = ("printf '\\377'; sleep 30" if invalid_utf8 else 'echo partial; sleep 30')
    if strict:
        with pytest.raises(NextDayTimeoutError, match='outcome unknown'):
            executor.execute('bash', {'command': command})
    else:
        assert executor.execute('bash', {'command': command}).startswith('Error:')
    event = event_ids(store)[0]
    facts = store.read_event(event)['result']
    cleanup = facts['process_boundary']
    assert cleanup['phase'] == 'timeout_cleanup'
    assert cleanup['direct_child_reaped'] and cleanup['streams_collected']
    assert cleanup['descendants_closed'] is None
    assert store.get_content(event + ':stdout_bytes')[1] == (b'\xff' if invalid_utf8 else b'partial\n')
    assert facts['status'] == ('result_unknown' if strict else 'failed')


def test_timeout_cleanup_failure_preserves_boundary_receipt(captured, monkeypatch):
    import subprocess
    from saas_bench.agents.bash_agent.tools import NextDayTimeoutError
    _, store, executor = captured
    executor.bash_timeout = .1
    def fail_kill(proc):
        raise OSError('injected direct kill failure')
    monkeypatch.setattr(subprocess.Popen, 'kill', fail_kill)
    try:
        with pytest.raises(NextDayTimeoutError, match='cleanup failed'):
            executor.execute('bash', {'command': 'sleep 30'})
        facts = store.read_event(event_ids(store)[0])['result']
        cleanup = facts['process_boundary']
        assert cleanup['phase'] == 'timeout_cleanup'
        assert cleanup['streams_collected'] is False
        assert cleanup['descendants_closed'] is None
        assert 'injected direct kill failure' in cleanup['actions']['kill']
        assert facts['boundary_closed'] is False
        with pytest.raises(RuntimeError):
            store.assert_healthy()
    finally:
        if executor.preserved_process:
            executor.preserved_process.communicate(timeout=5)


@pytest.mark.parametrize('direct_kill_fails', [False, True])
def test_timeout_signal_oserror_runs_fallback_or_preserves_boundary(captured, monkeypatch, direct_kill_fails):
    import os
    import signal
    import subprocess
    from saas_bench.agents.bash_agent.tools import NextDayTimeoutError, ProcessBoundaryError
    _, store, executor = captured
    executor.bash_timeout = .1
    real_killpg = os.killpg
    processes = []
    real_kill = subprocess.Popen.kill
    real_init = subprocess.Popen.__init__

    def init(proc, *args, **kwargs):
        real_init(proc, *args, **kwargs)
        processes.append(proc)

    def fail_signal(pid, sig):
        raise OSError(5, 'injected group signal failure')

    def kill(proc):
        if direct_kill_fails:
            raise OSError(5, 'injected direct kill failure')
        return real_kill(proc)

    monkeypatch.setattr(subprocess.Popen, '__init__', init)
    monkeypatch.setattr(os, 'killpg', fail_signal)
    monkeypatch.setattr(subprocess.Popen, 'kill', kill)
    try:
        if direct_kill_fails:
            with pytest.raises(NextDayTimeoutError, match='cleanup failed'):
                executor.execute('bash', {'command': 'sleep .3'})
        else:
            result = executor.execute('bash', {'command': 'sleep .3'})
            assert 'Command timed out' in result
        facts = store.read_event(event_ids(store)[0])['result']
        cleanup = facts['process_boundary']
        assert 'injected group signal failure' in cleanup['actions']['killpg']
        assert cleanup['descendants_closed'] is None
        assert processes
        if direct_kill_fails:
            assert executor.preserved_process is processes[0]
            assert facts['boundary_closed'] is False
            assert cleanup['streams_collected'] is False
            assert facts['status'] == 'result_unknown'
            with pytest.raises(RuntimeError):
                store.assert_healthy()
            with pytest.raises(ProcessBoundaryError, match='remains open'):
                executor.execute('bash', {'command': 'true'})
        else:
            assert cleanup['actions']['kill'] == 'returned'
            assert cleanup['direct_child_reaped'] and cleanup['streams_collected']
            store.assert_healthy()
    finally:
        for proc in processes:
            try:
                real_killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate(timeout=5)


@pytest.mark.parametrize('hard, soft', [(0, None), (True, None), (float('inf'), None),
    (float('nan'), None), (1, 0), (1, True), (1, 1), (1, 2), (1, float('nan')), (1, float('inf'))])
def test_bash_threshold_validation(tmp_path, hard, soft):
    with pytest.raises(ValueError, match='bash_'):
        BashAgentToolExecutor(tmp_path, bash_timeout=hard, bash_warn_after=soft)


@pytest.mark.parametrize('elapsed, exit_code, notice', [(119.999, 0, False), (120, 0, True),
    (120, 7, True), (479.999, 0, True), (480, 0, False)])
def test_bash_notice_boundaries_preserve_raw_result(captured, monkeypatch, elapsed, exit_code, notice):
    from saas_bench.process_boundary import Boundary
    from saas_bench.evidence_handles import body_digest
    _, store, executor = captured
    executor.bash_timeout, executor.bash_warn_after = 480, 120
    original = Boundary.communicate
    def communicate(boundary, *args, **kwargs):
        output = original(boundary, *args, **kwargs)
        boundary.record.update(elapsed_seconds=elapsed, timed_out=elapsed >= 480,
            timeout_trigger_seconds=elapsed if elapsed >= 480 else None)
        return output
    monkeypatch.setattr(Boundary, 'communicate', communicate)
    output = executor.execute('bash', dict(command=f'printf "42\\n"; printf "detail\\n" >&2; exit {exit_code}'))
    event = event_ids(store)[0]
    facts = store.read_event(event)['result']
    assert output.count('[Framework timing notice]') == int(notice)
    assert facts['warning_emitted'] is notice
    assert facts['status'] == ('timed_out' if elapsed >= 480 else 'failed' if exit_code else 'succeeded')
    assert facts['exit_code'] == exit_code
    assert store.get_content(event + ':stdout_bytes')[1] == b'42\n'
    assert store.get_content(event + ':stderr_bytes')[1] == b'detail\n'
    metadata, raw = store.get_content(event + ':tool_return')
    assert raw.decode() == output
    assert metadata['body_sha256'] == body_digest(output.split('\n\n[Framework timing notice]')[0])
    for item in output.origins:
        a, b = item['request_range']
        c, d = item['source_range']
        assert output[a:b] == store.get_content(item['version_id'])[1].decode()[c:d]
        if item['version_id'].endswith((':stdout', ':stderr')):
            assert '[Framework timing notice]' not in output[a:b]


@pytest.mark.parametrize('command', [
    "python -c 'import os,time;print(os.getpid(),flush=True);time.sleep(30)'",
    "python -c 'import os,time;print(os.getpid(),flush=True);time.sleep(30)' &",
    "python -c \"import subprocess;subprocess.Popen(['python','-c','import os,time;print(os.getpid(),flush=True);time.sleep(30)'], start_new_session=True)\"",
])
def test_verified_timeout_reaps_foreground_background_and_detached(captured, monkeypatch, command):
    import os
    _, store, executor = captured
    monkeypatch.setattr(executor, '_bwrap', lambda: None)
    executor.bash_timeout, executor.bash_warn_after = .3, .1
    output = executor.execute('bash', dict(command=command))
    pid = int(output.splitlines()[0])
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    facts = records(store)[0]['result']
    assert facts['status'] == 'timed_out' and not facts['warning_emitted']
    assert facts['elapsed_seconds'] >= .3 and facts['timeout_trigger_seconds'] >= .3
    assert 0 <= facts['cleanup_seconds'] < 2
    boundary = facts['process_boundary']
    assert boundary['phase'] == 'closed' and boundary['descendants_closed']
    assert boundary['direct_child_reaped'] and boundary['streams_collected']
    assert pid in boundary['reaped'] and pid in boundary['signaled']
    assert 'Task terminated' in output and 'Output is partial' in output
    assert store.get_content(event_ids(store)[0] + ':stdout_bytes')[0]['extent'] == 'partial'
    assert executor.execute('bash', dict(command='echo next')) == 'next\n'
    store.assert_healthy()


def test_verified_deadline_includes_popen_and_ignores_capture_scans(captured, monkeypatch):
    import time
    import subprocess
    from saas_bench.execution_capture import ExecutionCapture
    _, store, executor = captured
    executor.bash_timeout, executor.bash_warn_after = .2, .1
    original_init = subprocess.Popen.__init__
    original_scan = ExecutionCapture.snapshot
    def init(proc, *args, **kwargs):
        original_init(proc, *args, **kwargs)
        time.sleep(.3)
    def snapshot(capture, *args):
        time.sleep(.2)
        return original_scan(capture, *args)
    monkeypatch.setattr(subprocess.Popen, '__init__', init)
    monkeypatch.setattr(ExecutionCapture, 'snapshot', snapshot)
    result = executor.execute('bash', dict(command='echo partial; sleep 30'))
    facts = records(store)[0]['result']
    assert 'timed out' in result
    assert .3 <= facts['timeout_trigger_seconds'] < .45
    assert facts['host_accounting_seconds'] >= .4
    assert facts['capture_finish_seconds'] >= 0


def test_verified_unknown_receipt_preserves_scene_and_pauses(captured, monkeypatch):
    import os
    import signal
    from saas_bench import process_boundary
    from saas_bench.agents.bash_agent.tools import ProcessBoundaryError
    _, store, executor = captured
    executor.bash_timeout, executor.bash_warn_after = .2, .1
    monkeypatch.setattr(process_boundary, 'SUPERVISOR', process_boundary.SUPERVISOR.replace(
        "report('unknown' if unknown else 'closed')",
        "unknown = 'injected missing cleanup proof'; report('unknown')"))
    try:
        with pytest.raises(ProcessBoundaryError):
            executor.execute('bash', dict(command='echo partial; sleep 30'))
        facts = records(store)[0]['result']
        assert facts['status'] == 'result_unknown' and not facts['warning_emitted']
        assert not facts['boundary_closed']
        assert facts['process_boundary']['unknown'] == 'injected missing cleanup proof'
        assert executor.preserved_process.poll() is None
        with pytest.raises(KeyError):
            store.get_content(event_ids(store)[0] + ':tool_return')
    finally:
        if executor.preserved_process:
            os.killpg(executor.preserved_process.pid, signal.SIGKILL)
            executor.preserved_process.communicate(timeout=5)


def test_verified_timeout_invalid_bytes_stays_timed_out(captured):
    _, store, executor = captured
    executor.bash_timeout, executor.bash_warn_after = .2, .1
    output = executor.execute('bash', dict(command="printf '\\377'; sleep 30"))
    event = event_ids(store)[0]
    assert executor.last_status == 'timed_out'
    assert 'timed out' in output and 'Output is partial' in output
    assert '[Framework timing notice]' not in output
    assert store.get_content(event + ':stdout_bytes')[1] == b'\xff'
    assert store.read_event(event)['result']['process_boundary']['descendants_closed']


@pytest.mark.parametrize('strict, unknown_world', [(True, False), (False, True), (True, True)])
def test_verified_invalid_bytes_cannot_hide_unknown_operation(captured, strict, unknown_world):
    from saas_bench.agents.bash_agent.tools import NextDayTimeoutError
    _, store, executor = captured
    executor.bash_timeout, executor.bash_warn_after = .2, .1
    executor.stop_on_timeout = strict
    executor.world_status = lambda: unknown_world
    with pytest.raises(NextDayTimeoutError, match='outcome unknown'):
        executor.execute('bash', dict(command="printf 'partial-out\\377'; sleep 30"))
    event = event_ids(store)[0]
    assert store.get_content(event + ':stdout_bytes')[1] == b'partial-out\xff'
    assert store.read_event(event)['result']['status'] == 'result_unknown'
    assert executor.last_status == 'result_unknown'
    assert not store.read_event(event)['result']['warning_emitted']
    with pytest.raises(KeyError):
        store.get_content(event + ':tool_return')


def test_verified_timeout_closes_sandbox_descendants(captured):
    _, store, executor = captured
    if executor._bwrap() is None:
        pytest.skip('bubblewrap required')
    executor.bash_timeout, executor.bash_warn_after = .2, .1
    result = executor.execute('bash', dict(command='(sleep 2; echo escaped > marker) & echo parent'))
    facts = records(store)[0]['result']
    assert facts['status'] == 'timed_out' and 'Task terminated' in result
    assert facts['process_boundary']['descendants_closed']
    assert facts['process_boundary']['streams_collected']
    assert facts['process_boundary']['stop_requested']
    assert not (executor.workspace_path / 'marker').exists()
    assert executor.execute('bash', dict(command='echo next')) == 'next\n'


def test_verified_decode_failure_still_checks_http_world_state(captured):
    from saas_bench.agents.bash_agent.tools import NextDayTimeoutError
    api, store, executor = captured
    executor.bash_timeout, executor.bash_warn_after = .2, .1
    api._operation_failed = True
    with pytest.raises(NextDayTimeoutError, match='outcome unknown'):
        executor.execute('bash', dict(command="printf '\\377'; sleep 30"))
    assert records(store)[0]['result']['status'] == 'result_unknown'
