"""SQL cancellation and evidence contention through real HTTP and Bash executors."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
import json
import os
import threading
import time
import urllib.request

import pytest

from saas_bench.execution_capture import ExecutionCapture, model_request
from saas_bench.sql_evidence import SQLEvidenceStore
from test_execution_capture import captured, records
from test_public_sql import server
from test_sql_evidence import identity, settled


def test_shared_fault_during_wire_capture_blocks_model_transport(captured, monkeypatch):
    import httpx
    from openai import APIConnectionError, OpenAI
    from saas_bench.model_usage import ModelUsage
    from test_preflight_usage import reply
    _, store, _ = captured
    competitor = SQLEvidenceStore(store.path, identity('growth'))
    version, sent = store.version, []

    def competing_fault(event, slot, *args, **kwargs):
        result = version(event, slot, *args, **kwargs)
        if slot == 'wire':
            competitor.fail(OSError('competing-role capture failed during wire recording'))
        return result

    def transport(request):
        sent.append(request.read())
        return httpx.Response(200, json=reply('chat'))

    monkeypatch.setattr(store, 'version', competing_fault)
    usage = ModelUsage(None, 'agent', evidence_store=store)
    with usage.attach(OpenAI(api_key='offline-only', max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(transport)))) as client:
        payload = dict(model='test-model', messages=[dict(role='user', content='hello')])
        with pytest.raises(APIConnectionError, match='Connection error') as raised:
            usage.call('chat', payload, lambda: client.chat.completions.create(**payload))
        assert 'capture failed' in str(raised.value.__cause__)
    assert sent == []


@pytest.mark.parametrize('timeout_kind', ['gnu', 'bash'])
def test_closed_bash_timeout_settles_sql_before_next_model_request(captured, monkeypatch, timeout_kind):
    from saas_bench import api_server
    api, store, executor = captured
    entered, release = threading.Event(), threading.Event()
    execute = api_server.execute_query

    def delayed_query(*args, **kwargs):
        entered.set()
        assert release.wait(5), 'query release missing'
        return execute(*args, **kwargs)

    monkeypatch.setattr(api_server, 'execute_query', delayed_query)
    executor.bash_timeout = 5 if timeout_kind == 'gnu' else .5
    executor.bash_warn_after = .1
    client = 'python -c "import novamind_api as nm; nm.query(\'SELECT 1 AS n\')"'
    command = 'timeout --kill-after=.1 .5 ' + client if timeout_kind == 'gnu' else client
    wire = b'{"messages":[]}'
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(executor.execute, 'bash', {'command': command})
        try:
            assert entered.wait(3)
            output = running.result(timeout=3)
            assert ('[exit code: 124]' if timeout_kind == 'gnu' else 'timed out after 0.5 seconds') in output
            with pytest.raises(RuntimeError, match='Unconfirmed SQL evidence outcome'):
                model_request(store, wire, [], 'call', 'attempt', 'context')
            assert not any(r['request']['kind'] == 'model_request' for r in records(store))
            release.set()
            settled(api)
        finally:
            release.set()
            running.result(timeout=5)
    bash, query = records(store)
    print(json.dumps(dict(case=timeout_kind + '_sql_interruption', bash=bash['result'],
        query=query['result'], delivery=query['delivery'], interruption=query['client_interruption'])))
    assert bash['result']['boundary_closed'] is True
    assert bash['result']['process_boundary']['children'] == []
    assert query['result']['status'] == 'succeeded'
    assert query['delivery']['send_state'] in ('sent', 'failed')
    assert query['client_receipt'] == {'receive_state': 'unknown'}
    assert query['client_interruption']['reason'] == 'bash_client_timeout'
    assert query['client_interruption']['bash_event_id'] == bash['request']['event_id']
    with closing(store.connect()) as conn:
        assert conn.execute('SELECT received FROM client_calls').fetchone()[0] is None
    assert store.fault is None
    event = model_request(store, wire, [], 'call', 'attempt', 'context')
    assert store.get_content(event + ':wire')[1] == wire
    store.complete(event)
    store.assert_healthy()
    print(json.dumps(dict(case=timeout_kind + '_model_gate', blocked_model_requests=1, resumed_model_requests=1)))


def test_shared_batch_over_three_seconds_preserves_query_evidence(captured, monkeypatch):
    api, store, _ = captured
    competitor = SQLEvidenceStore(store.path, identity('growth'))
    locked = threading.Event()
    finish = store.finish
    threads = []

    def writer():
        with competitor.batch():
            competitor.save_state('competing_capture', True)
            locked.set()
            time.sleep(3.4)

    def contested_finish(*args):
        thread = threading.Thread(target=writer)
        threads.append(thread)
        thread.start()
        assert locked.wait(3)
        return finish(*args)

    monkeypatch.setattr(store, 'finish', contested_finish)
    try:
        request = urllib.request.Request(f'http://127.0.0.1:{api.port}/query', data=b'{"sql":"SELECT 1 AS n"}')
        with urllib.request.urlopen(request, timeout=6) as response:
            assert response.status == 200
            assert json.load(response)['rows'] == [{'n': 1}]
        settled(api)
        row, = records(store)
        assert row['result']['status'] == 'succeeded'
        assert row['result']['capture_status'] == 'complete'
        assert row['delivery']['send_state'] == 'sent'
        assert store.fault is None and not store.fault_path.exists()
        store.assert_healthy()
        print(json.dumps(dict(case='shared_batch_3.4_seconds', result=row['result'], delivery=row['delivery'])))
    finally:
        for thread in threads:
            thread.join(5)


def test_workspace_hashing_releases_shared_writer(captured, monkeypatch):
    from saas_bench import execution_capture
    _, store, executor = captured
    capture = ExecutionCapture(store)
    capture.begin('read_file', {})
    competitor = SQLEvidenceStore(store.path, identity('growth'))
    hashing, release = threading.Event(), threading.Event()
    digest = execution_capture.digest

    def delayed_digest(raw):
        hashing.set()
        assert release.wait(5), 'hash release missing'
        return digest(raw)

    monkeypatch.setattr(execution_capture, 'digest', delayed_digest)
    with ThreadPoolExecutor(max_workers=2) as pool:
        scan = pool.submit(capture.snapshot, executor.workspace_path, 'before')
        try:
            assert hashing.wait(3)
            write = pool.submit(competitor.save_state, 'during_hash', True)
            write.result(timeout=1)
            assert competitor.load_state('during_hash') is True
        finally:
            release.set()
            scan.result(timeout=5)
    capture.finish('done')
    store.assert_healthy()


def test_snapshot_hashes_equal_size_and_mtime_file_changes(captured):
    _, store, executor = captured
    path = executor.workspace_path / 'changed.txt'
    path.write_bytes(b'before')
    stamp = path.stat()
    capture = ExecutionCapture(store)
    capture.begin('write_file', {})
    before = capture.snapshot(executor.workspace_path, 'before')
    path.write_bytes(b'after!')
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    after = capture.snapshot(executor.workspace_path, 'after')
    assert before['changed.txt']['sha256'] != after['changed.txt']['sha256']
    assert store.get_content(after['changed.txt']['version'])[1] == b'after!'
    capture.finish('done')
    store.assert_healthy()


def test_file_write_during_snapshot_still_stops_capture(captured, monkeypatch):
    from saas_bench import workspace_io
    _, store, executor = captured
    path = executor.workspace_path / 'changed.txt'
    path.write_bytes(b'before')
    original = workspace_io.open_file

    @contextmanager
    def changed_during_read(root, filename):
        with original(root, filename) as stream:
            if str(filename) == str(path):
                class ChangingFile:
                    def fileno(self):
                        return stream.fileno()

                    def read(self):
                        raw = stream.read()
                        stamp = path.stat()
                        path.write_bytes(b'after!')
                        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                        return raw

                yield ChangingFile()
            else:
                yield stream

    monkeypatch.setattr(workspace_io, 'open_file', changed_during_read)
    capture = ExecutionCapture(store)
    capture.begin('read_file', {})
    assert capture.safe(capture.snapshot, executor.workspace_path, 'before') is None
    assert store.fault['reason'] == 'File changed during boundary capture: changed.txt'
    capture.finish('failed scan', status='failed')
    with pytest.raises(RuntimeError, match='capture failed'):
        store.assert_healthy()


def test_contended_write_deadline_keeps_capture_failure(captured, monkeypatch):
    _, store, _ = captured
    competitor = SQLEvidenceStore(store.path, identity('growth'))
    monkeypatch.setattr(store, 'WRITE_TIMEOUT_SECONDS', .05)
    with competitor.batch():
        competitor.save_state('holding_lock', True)
        ExecutionCapture(store).safe(store.save_state, 'blocked', True)
    assert store.fault['sqlite_errorname'] == 'SQLITE_BUSY'
    with pytest.raises(RuntimeError, match='capture failed'):
        store.assert_healthy()
