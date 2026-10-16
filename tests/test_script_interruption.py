"""D77 script interruption through the public wrapper, Bash boundary and model wire."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import threading
import time
import urllib.request

import pytest
import httpx
from openai import OpenAI

from saas_bench.model_usage import ModelUsage
from saas_bench.run_lifecycle import RunCancelled
from saas_bench.text_registry import TextRegistry
from test_execution_capture import captured, records
from test_public_sql import server
from test_preflight_usage import reply
from test_sql_evidence import settled
from test_preflight_team_batch import changed_manifest, frozen


@contextmanager
def model_server():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body = json.dumps(reply('chat')).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    api = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=api.serve_forever)
    thread.start()
    try:
        yield f'http://127.0.0.1:{api.server_port}/v1', received
    finally:
        api.shutdown()
        api.server_close()
        thread.join(2)
        assert not thread.is_alive()


def public_script(captured, code):
    api, store, executor = captured
    executor.text_registry = TextRegistry(executor.workspace_path, 'pf', store, sim_day=lambda: 77)
    from saas_bench.pf_queries import PFQueries
    executor.pf_queries = PFQueries(executor.text_registry)
    source = Path(__file__).parents[1] / 'src/saas_bench'
    shutil.copy(source / '_public_cli.py', executor.workspace_path / 'cli.py')
    shutil.copy(source / 'novamind_api/_capture.py', executor.workspace_path / '_client_capture.py')
    (executor.workspace_path / 'mc_w12.py').write_text(code)
    (executor.workspace_path / 'driver.py').write_text(
        "import cli, os\ncli._execute_python('__env__', int(os.environ['NOVAMIND_API_PORT']), "
        "open('mc_w12.py').read(), 'mc_w12.py')\n")
    return api, store, executor


def post_python_end(api, store, event, record):
    parent = store.read_event(event)['request']['parent_event_id']
    with store.connect() as conn:
        context, = conn.execute('SELECT token FROM capture_contexts WHERE event_id=?', (parent,)).fetchone()
        call, = conn.execute('SELECT token FROM capture_contexts WHERE event_id=?', (event,)).fetchone()
    data = json.dumps(dict(context=context, call=call, record=record)).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{api.port}/_capture', data=data)
    try:
        response = urllib.request.urlopen(request, timeout=3)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return response.status, json.load(response)


@pytest.mark.parametrize('prefix', ['', 'echo partial-shell-output; '], ids=['buffered-empty', 'partial'])
def test_bash_timeout_of_public_script_allows_next_model_and_snapshot(captured, prefix):
    _, store, executor = public_script(captured,
        "print('=== BASELINE current settings ===')\nwhile True: pass\n")
    executor.bash_timeout, executor.bash_warn_after = .8, .1
    output = executor.execute('bash', {'command': prefix + 'python driver.py 2>&1 | tail -12'})
    assert 'timed out after 0.8 seconds' in output and 'Output is partial' in output
    assert ('partial-shell-output' in output) == bool(prefix)
    rows = records(store)
    child, = [row for row in rows if row['request']['kind'] == 'cli_python']
    assert child['request']['request']['script'] == 'mc_w12.py'
    assert child['result'].get('completion') == 'unfinished', child['result']
    assert child['result']['capture_status'] == 'complete'
    assert child['result']['completed_at'] is not None
    assert child['result']['exit_code'] is None
    assert rows[0]['result']['boundary_closed'] is True
    assert rows[0]['result']['process_boundary']['children'] == []
    assert executor.preserved_process is None
    assert store.fault is None and not store.fault_path.exists()
    usage = ModelUsage(executor.workspace_path.parent / 'usage.jsonl', 'ops_finance', evidence_store=store)
    with model_server() as (url, sent), usage.attach(OpenAI(api_key='offline', base_url=url, max_retries=0)) as client:
        request = dict(model='test-model', messages=[dict(role='user', content=output)])
        response = usage.call('chat', request, lambda: client.chat.completions.create(**request))
        assert response.choices[0].message.content == 'hello'
        assert len(sent) == 1 and 'Output is partial' in sent[0]['messages'][0]['content']
    store.assert_healthy()
    snapshot = executor.workspace_path.parent / 'eligible.sqlite'
    store.snapshot(snapshot)
    assert snapshot.is_file()
    assert executor.execute('bash', {'command': 'echo next-tool'}).startswith('next-tool\n')


@pytest.mark.parametrize('stdout', [b'', b'partial-out\xff'])
def test_public_python_timeout_keeps_empty_or_undecodable_streams_healthy(captured, stdout, monkeypatch):
    _, store, executor = public_script(captured,
        'import sys\nsys.stdout.buffer.write(' + repr(stdout) + ')\nsys.stdout.flush()\nwhile True: pass\n')
    monkeypatch.setitem(executor.extra_env, 'CEOBENCH_PUBLIC_PYTHON_TIMEOUT_SECONDS', '.15')
    executor.bash_timeout, executor.bash_warn_after = 4, .1
    output = executor.execute('bash', {'command': 'python driver.py 2>&1 | tail -12'})
    assert 'TimeoutExpired' in output
    child, = [row for row in records(store) if row['request']['kind'] == 'cli_python']
    assert child['result']['error'] == 'timeout' and child['result']['exit_code'] == -9
    assert store.get_content(child['request']['event_id'] + ':stdout_bytes')[1] == stdout
    assert store.fault is None and not store.fault_path.exists()
    store.assert_healthy()


def test_promised_script_output_corruption_still_blocks_model(captured, monkeypatch):
    _, store, executor = public_script(captured, "print('saved-output')\n")
    executor.bash_timeout, executor.bash_warn_after = 4, .1
    version = store.version

    def corrupt(event, slot, *args, **kwargs):
        saved = version(event, slot, *args, **kwargs)
        if slot == 'stdout' and store.read_event(event)['request']['kind'] == 'cli_python':
            with store._writer() as conn:
                conn.execute('UPDATE blobs SET content=? WHERE content_hash=(SELECT content_hash FROM versions WHERE version_id=?)',
                             (b'corrupt', saved))
        return saved

    monkeypatch.setattr(store, 'version', corrupt)
    executor.execute('bash', {'command': 'python driver.py'})
    assert store.fault and store.fault_path.exists()
    assert 'checksum mismatch' in store.fault['reason']
    with pytest.raises(RuntimeError, match='capture failed'):
        store.snapshot(executor.workspace_path.parent / 'unsafe.sqlite')
    usage = ModelUsage(None, 'ops_finance', evidence_store=store)
    with model_server() as (url, sent), usage.attach(OpenAI(api_key='offline', base_url=url, max_retries=0)) as client:
        request = dict(model='test-model', messages=[dict(role='user', content='continue')])
        with pytest.raises(RunCancelled, match='capture failed'):
            usage.call('chat', request, lambda: client.chat.completions.create(**request))
        assert sent == []


def test_pre_http_capture_error_is_reported_locally_without_connection_wrapper(captured, monkeypatch):
    from saas_bench import execution_capture
    _, store, executor = captured

    def capture_failed(*args, **kwargs):
        raise OSError('D77 local evidence persistence failure')

    monkeypatch.setattr(execution_capture, 'model_request', capture_failed)
    usage = ModelUsage(executor.workspace_path.parent / 'usage.jsonl', 'ops_finance', evidence_store=store)
    with model_server() as (url, sent), usage.attach(OpenAI(api_key='offline', base_url=url, max_retries=0)) as client:
        request = dict(model='test-model', messages=[dict(role='user', content='continue')])
        with pytest.raises(RunCancelled, match='D77 local evidence persistence failure') as failed:
            usage.call('chat', request, lambda: client.chat.completions.create(**request))
        assert failed.value.reason == 'run_evidence_fault'
        assert isinstance(failed.value.__cause__, OSError)
        assert 'APIConnectionError' not in str(failed.value)
        assert sent == []
    events = [json.loads(line) for line in usage.path.read_text().splitlines()]
    assert not any(row['event'] == 'http_request' for row in events)
    assert store.fault is not None


def test_public_end_recording_finishes_before_boundary_marks_script_unfinished(captured, monkeypatch):
    _, store, executor = public_script(captured, "print('finished-before-wrapper-kill')\n")
    executor.bash_timeout, executor.bash_warn_after = .8, .1
    entered, release = threading.Event(), threading.Event()
    version = store.version

    def slow_receipt(event, slot, *args, **kwargs):
        if slot == 'python_end':
            entered.set()
            assert release.wait(5)
        return version(event, slot, *args, **kwargs)

    monkeypatch.setattr(store, 'version', slow_receipt)
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(executor.execute, 'bash', {'command': 'python driver.py'})
        try:
            assert entered.wait(3)
            deadline = time.monotonic() + 3
            while not executor._bash_facts.get('boundary_closed') and time.monotonic() < deadline:
                time.sleep(.01)
            assert executor._bash_facts['boundary_closed']
        finally:
            release.set()
        output = running.result(timeout=5)
    assert 'timed out' in output
    child, = [row for row in records(store) if row['request']['kind'] == 'cli_python']
    assert child['result']['status'] == 'succeeded'
    assert child['result']['exit_code'] == 0
    assert store.get_content(child['request']['event_id'] + ':stdout')[1] == b'finished-before-wrapper-kill\n'
    assert store.fault is None
    store.assert_healthy()


@pytest.mark.parametrize('late', [False, True], ids=['normal', 'late'])
@pytest.mark.parametrize('corruption', [None, 'python_end', 'stdout_bytes'])
def test_public_end_preserves_result_and_verifies_replayed_receipts(captured, late, corruption):
    api, store, executor = public_script(captured,
        'while True: pass\n' if late else "print('wrapper-output')\n")
    executor.bash_timeout, executor.bash_warn_after = .8, .1
    executor.execute('bash', {'command': 'python driver.py'})
    rows = records(store)
    bash, child = rows[:2]
    original = copy.deepcopy(child['result'])
    assert (original.get('completion') == 'unfinished') == late
    record = dict(kind='python_end', stdout=b'wrapper-output\n'.hex(), stderr='',
                  exit_code=-9 if late else 0, error='timeout' if late else None)

    def post(value):
        return post_python_end(api, store, child['request']['event_id'], value)

    assert post(record) == (200, {'accepted': True})
    assert post(record) == (200, {'accepted': True})
    assert store.read_event(child['request']['event_id'])['result'] == original
    assert store.get_content(child['request']['event_id'] + ':stdout')[1] == b'wrapper-output\n'
    store.assert_healthy()
    if corruption:
        with store.connect() as conn:
            conn.execute('UPDATE blobs SET content=? WHERE content_hash=(SELECT content_hash FROM versions WHERE version_id=?)',
                         (b'corrupt', child['request']['event_id'] + ':' + corruption))
        assert post(record) == (400, {'accepted': False})
        assert store.fault is not None and 'checksum mismatch' in store.fault['reason']
    else:
        assert post(dict(record, exit_code=1)) == (400, {'accepted': False})
        assert store.fault is not None and 'Conflicting Python end receipt' in store.fault['reason']


def test_killed_public_wrapper_settles_sql_after_server_finishes(captured, monkeypatch):
    from saas_bench import api_server
    api, store, executor = public_script(captured,
        "import novamind_api as nm\nnm.query('SELECT amount FROM ledger')\n")
    executor.bash_timeout, executor.bash_warn_after = .8, .1
    entered, release = threading.Event(), threading.Event()
    execute = api_server.execute_query

    def delayed_query(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return execute(*args, **kwargs)

    monkeypatch.setattr(api_server, 'execute_query', delayed_query)
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(executor.execute, 'bash', {'command': 'python driver.py'})
        try:
            assert entered.wait(3)
            output = running.result(timeout=3)
            assert 'Script execution unfinished' in output
            assert store.fault is None
            release.set()
            settled(api)
            query, = [row for row in records(store) if row['request']['kind'] == 'sql_query']
            assert query['result']['status'] == 'succeeded'
            assert query['client_receipt'] == {'receive_state': 'unknown'}
            assert query['client_interruption']['reason'] == 'python_boundary_interruption'
            assert query['client_interruption']['state'] == 'interrupted'
            usage = ModelUsage(None, 'ops_finance', evidence_store=store)
            with model_server() as (url, sent), usage.attach(OpenAI(api_key='offline', base_url=url, max_retries=0)) as client:
                request = dict(model='test-model', messages=[dict(role='user', content=output)])
                usage.call('chat', request, lambda: client.chat.completions.create(**request))
                assert len(sent) == 1
            store.assert_healthy()
            store.snapshot(executor.workspace_path.parent / 'late-sql-checkpoint.sqlite')
        finally:
            release.set()
            running.result(timeout=5)


def test_public_start_arriving_after_bash_closed_settles_before_next_model(captured, monkeypatch):
    _, store, executor = public_script(captured, 'while True: pass\n')
    executor.bash_timeout, executor.bash_warn_after = .8, .1
    entered, release, created = threading.Event(), threading.Event(), threading.Event()
    begin = store.begin_event

    def delayed_start(kind, *args, **kwargs):
        if kind == 'cli_python':
            entered.set()
            assert release.wait(5)
            event = begin(kind, *args, **kwargs)
            created.set()
            return event
        return begin(kind, *args, **kwargs)

    monkeypatch.setattr(store, 'begin_event', delayed_start)
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(executor.execute, 'bash', {'command': 'python driver.py'})
        try:
            assert entered.wait(3)
            output = running.result(timeout=3)
            assert 'timed out' in output
            assert not any(row['request']['kind'] == 'cli_python' for row in records(store))
            release.set()
            assert created.wait(3)
            usage = ModelUsage(None, 'ops_finance', evidence_store=store)
            with model_server() as (url, sent), usage.attach(OpenAI(api_key='offline', base_url=url, max_retries=0)) as client:
                request = dict(model='test-model', messages=[dict(role='user', content=output)])
                usage.call('chat', request, lambda: client.chat.completions.create(**request))
                assert len(sent) == 1
            child, = [row for row in records(store) if row['request']['kind'] == 'cli_python']
            assert child['result']['completion'] == 'unfinished'
            assert store.fault is None
            store.snapshot(executor.workspace_path.parent / 'late-start-checkpoint.sqlite')
        finally:
            release.set()
            running.result(timeout=5)


def test_old_completed_script_accepts_identical_end_without_new_receipt_marker(captured):
    api, store, executor = public_script(captured, "print('legacy-output')\n")
    executor.execute('bash', {'command': 'python driver.py'})
    child, = [row for row in records(store) if row['request']['kind'] == 'cli_python']
    event = child['request']['event_id']
    with store.connect() as conn:
        conn.execute('DELETE FROM versions WHERE version_id=?', (event + ':python_end',))
    record = dict(kind='python_end', stdout=b'legacy-output\n'.hex(), stderr='', exit_code=0, error=None)
    assert post_python_end(api, store, event, record) == (200, {'accepted': True})
    assert store.read_event(event)['result'] == child['result']
    store.assert_healthy()


def test_real_batch_pre_http_evidence_failure_holds_original_attempt(frozen, tmp_path, monkeypatch):
    from saas_bench import execution_capture, team_batch as batch
    from saas_bench.team_checkpoint import validate_snapshot
    path = changed_manifest(frozen, tmp_path, lambda manifest: None)
    entry = next(row for row in batch.read(path)['runs'] if row['group'] == 'pf')
    sent = []

    def client(role):
        def respond(request):
            sent.append(request)
            raise AssertionError('A local evidence failure must stop before model HTTP')
        return batch.checked_client(role, 'local-evidence-failure', transport=httpx.MockTransport(respond))

    def capture_failed(*args, **kwargs):
        raise OSError('D77 local evidence persistence failure')

    monkeypatch.setattr(execution_capture, 'model_request', capture_failed)
    record = batch.prepare(path, entry['run_id'], live=True, client_factory=client,
                           simulator_factory=batch.simulator)
    assert record['status'] == 'paused' and record['hold'] is True
    assert record['runtime_failure'] == 'run_evidence_fault'
    assert len(batch.attempts(entry)) == 1 and sent == []
    attempt, = batch.attempts(entry)
    hold = batch.read(attempt / 'HOLD')
    assert hold['reason'] == 'run_evidence_fault' and hold['manual_resume_required']
    assert 'D77 local evidence persistence failure' in hold['detail']
    assert 'APIConnectionError' not in hold['detail']
    snapshot, state = validate_snapshot(record['checkpoint'])
    assert snapshot.is_dir() and state['day'] == 0 and state['stage'] == 'boundary'


@pytest.mark.parametrize('damage', ['missing_stdout', 'corrupt_raw'])
def test_late_promised_stream_damage_is_not_hidden_by_unfinished_result(captured, monkeypatch, damage):
    api, store, executor = public_script(captured, 'while True: pass\n')
    executor.bash_timeout, executor.bash_warn_after = .8, .1
    decorate = executor.pf_queries.decorate

    def damaged_receipt(capture, text, after):
        child, = [row for row in records(store) if row['request']['kind'] == 'cli_python']
        event = child['request']['event_id']
        assert child['result']['completion'] == 'unfinished'
        receipt = dict(kind='python_end', stdout=b'late-result\n'.hex(), stderr='', exit_code=-9, error='timeout')
        assert post_python_end(api, store, event, receipt) == (200, {'accepted': True})
        with store.connect() as conn:
            conn.execute('DELETE FROM versions WHERE version_id=?', (event + ':stdout',))
            if damage == 'corrupt_raw':
                conn.execute('UPDATE blobs SET content=? WHERE content_hash=(SELECT content_hash FROM versions WHERE version_id=?)',
                             (b'corrupt', event + ':stdout_bytes'))
        return decorate(capture, text, after)

    monkeypatch.setattr(executor.pf_queries, 'decorate', damaged_receipt)
    executor.execute('bash', {'command': 'python driver.py'})
    assert store.fault is not None
    assert store.fault['exception_type'] == ('KeyError' if damage == 'missing_stdout' else 'ValueError')
    if damage == 'corrupt_raw':
        assert 'checksum mismatch' in store.fault['reason']
    usage = ModelUsage(None, 'ops_finance', evidence_store=store)
    with model_server() as (url, sent), usage.attach(OpenAI(api_key='offline', base_url=url, max_retries=0)) as client:
        request = dict(model='test-model', messages=[dict(role='user', content='continue')])
        with pytest.raises(RunCancelled, match='capture failed'):
            usage.call('chat', request, lambda: client.chat.completions.create(**request))
        assert sent == []
