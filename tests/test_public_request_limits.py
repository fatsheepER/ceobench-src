import json
import socket
import shlex
import threading
import time

import pytest

from saas_bench import public_sql
from saas_bench.public_sql import QueryResultTooLarge, execute_query, query_snapshot
from test_execution_capture import captured, records
from test_public_sql import server, request
from test_sql_evidence import settled


def capture_call_command(endpoint, body, call_key, *, unix_socket=False):
    command = ['curl', '--silent', '--show-error', '--max-time', '5',
               '--write-out', '\nHTTP_STATUS=%{http_code}\n',
               '--header', 'Content-Type: application/json', '--data-raw', body]
    if call_key is not None:
        command += ['--header', 'X-Capture-Call: ' + call_key if call_key else 'X-Capture-Call;']
    command = shlex.join(command) + ' --header "X-Capture-Context: $NOVAMIND_CAPTURE_CONTEXT"'
    if unix_socket:
        return command + ' --unix-socket "$NOVAMIND_API_SOCKET" http://localhost' + endpoint
    return command + ' "http://127.0.0.1:$NOVAMIND_API_PORT' + endpoint + '"'


@pytest.mark.parametrize('endpoint,body,call_key', [
    pytest.param('/call', '{}', 'testcall1', id='actual-call'),
    pytest.param('/query', '{"sql":"SELECT 42 AS n"}', 'testcall1', id='query'),
    pytest.param('/query', '{"sql":"SELECT 42 AS n"}', '', id='empty-header'),
])
def test_invalid_capture_call_is_rejected_and_next_model_request_continues(captured, endpoint, body, call_key):
    import httpx
    from openai import OpenAI
    from saas_bench.model_usage import ModelUsage
    from test_preflight_usage import reply
    api, store, executor = captured
    output = executor.execute('bash', {'command': capture_call_command(endpoint, body, call_key)})
    response, status = output.split('\nHTTP_STATUS=')
    assert status.strip() == '400', (output, store.fault)
    rejected = json.loads(response)
    assert rejected['success'] is False
    assert 'X-Capture-Call' in rejected['error'] and '32' in rejected['error']
    settled(api)
    bash, event = records(store)
    assert event['request']['parent_event_id'] == bash['request']['event_id']
    saved_body = (event['request']['raw_request_hex'] if endpoint == '/query'
                  else event['request']['request']['body_hex'])
    assert bytes.fromhex(saved_body).decode() == body
    assert json.loads(store.get_content(event['request']['event_id'] + ':public_response')[1]) == rejected
    assert event['result']['http_status'] == 400 and event['result']['completed_at'] is not None
    assert event['result']['permanent_error'] is True
    assert event['delivery']['send_state'] == 'sent'
    assert event['client_receipt'] == {'receive_state': 'unknown'}
    assert store.fault is None and not store.fault_path.exists()

    succeeded = executor.execute('bash', {'command':
        'python -c ' + shlex.quote('import novamind_api as nm; print(nm.query("SELECT 43 AS n"))')})
    assert '43' in succeeded and '[exit code:' not in succeeded, succeeded
    settled(api)
    with api._request_lock:
        assert api._request_lock.wait_for(lambda: not api._active_requests, timeout=5)
    queries = [row for row in records(store) if row['request']['kind'] == 'sql_query']
    assert len(queries) == (2 if endpoint == '/query' else 1)
    assert queries[-1]['result']['status'] == 'succeeded'
    assert queries[-1]['client_receipt']['receive_state'] == 'received'
    for row in records(store):
        if row['request']['kind'] == 'bash':
            assert row['result']['process_boundary']['phase'] == 'closed'
            assert row['result']['process_boundary']['children'] == []
    assert executor.preserved_process is None
    sent = []

    def transport(request):
        sent.append(json.loads(request.read()))
        return httpx.Response(200, json=reply('chat'))

    usage = ModelUsage(None, 'agent', evidence_store=store)
    with usage.attach(OpenAI(api_key='offline-only', max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(transport)))) as client:
        payload = dict(model='test-model', messages=[dict(role='user', content=value)
            for value in (output, succeeded)])
        result = usage.call('chat', payload, lambda: client.chat.completions.create(**payload))
    assert result.choices[0].message.content == 'hello'
    assert len(sent) == 1 and [m['content'] for m in sent[0]['messages']] == [output, succeeded]
    store.assert_healthy()
    store.snapshot(executor.workspace_path.parent / 'settled.sqlite')


@pytest.mark.parametrize('failure', ['binding-conflict', 'persistence'])
def test_valid_capture_call_integrity_failures_still_block_model_transport(captured, monkeypatch, failure):
    import httpx
    from openai import OpenAI
    from saas_bench.model_usage import ModelUsage
    from saas_bench.run_lifecycle import RunCancelled
    api, store, executor = captured
    command = capture_call_command('/query', '{"sql":"SELECT 42 AS n"}', 'a' * 32)
    expected = 'Conflicting client call binding'
    if failure == 'binding-conflict':
        command += '\n' + command
    else:
        expected = 'injected response persistence failure'

        def broken_finish(*args, **kwargs):
            raise OSError(expected)

        monkeypatch.setattr(store, 'finish', broken_finish)
    executor.execute('bash', {'command': command})
    settled(api)
    with api._request_lock:
        assert api._request_lock.wait_for(lambda: not api._active_requests, timeout=5)
    assert store.fault['reason'] == expected
    assert store.fault_path.exists()
    assert executor.preserved_process is None
    assert request(api, {'sql': 'SELECT 43 AS n'})[0] == 503
    sent = []

    def transport(request):
        sent.append(request.read())
        raise AssertionError('An unhealthy evidence store reached model transport')

    usage = ModelUsage(None, 'agent', evidence_store=store)
    with usage.attach(OpenAI(api_key='offline-only', max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(transport)))) as client:
        payload = dict(model='test-model', messages=[dict(role='user', content='continue')])
        with pytest.raises(RunCancelled, match='capture failed'):
            usage.call('chat', payload, lambda: client.chat.completions.create(**payload))
    assert sent == []
    with pytest.raises(RuntimeError, match='capture failed'):
        store.snapshot(executor.workspace_path.parent / 'unsafe.sqlite')


def test_incomplete_body_has_terminal_failure_and_releases_admission(captured):
    api, store, _ = captured
    api.QUERY_TIMEOUT_SECONDS = .08
    with socket.create_connection(('127.0.0.1', api.port), timeout=2) as connection:
        connection.sendall(b'POST /query HTTP/1.0\r\nContent-Length: 100\r\n\r\n{"sql":')
        response = connection.makefile('rb').read()
    assert response.startswith(b'HTTP/1.0 504'), response
    assert b'Request input exceeded its time limit' in response
    settled(api)
    row, = records(store)
    assert api._sql_active == 0
    assert row['result']['status'] == 'timed_out'
    assert row['delivery']['send_state'] == 'sent'
    assert store.fault is None
    assert request(api, {'sql': 'SELECT 1 AS n'})[1]['rows'] == [{'n': 1}]


def test_trickled_headers_close_at_total_deadline(server):
    server.QUERY_TIMEOUT_SECONDS = .08
    server.start()
    with socket.create_connection(('127.0.0.1', server.port), timeout=1) as connection:
        connection.sendall(b'POST /query HTTP/1.0\r\nContent-Len')
        response = connection.makefile('rb').read()
        assert response.startswith(b'HTTP/1.0 504'), response
    assert server._sql_active == 0


@pytest.mark.parametrize('length', ['-1', '2097153', 'garbage'])
def test_invalid_body_length_rejected_without_capture_fault(captured, length, monkeypatch):
    api, store, _ = captured
    monkeypatch.setitem(api.runtime_environment, 'CEOBENCH_SQL_CELL_LIMIT_BYTES', '262144')
    with socket.create_connection(('127.0.0.1', api.port), timeout=2) as connection:
        connection.sendall(f'POST /query HTTP/1.0\r\nContent-Length: {length}\r\n\r\n'.encode())
        response = connection.makefile('rb').read()
    assert response.startswith(b'HTTP/1.0 400'), response
    settled(api)
    row, = records(store)
    assert row['result']['status'] == 'rejected'
    assert row['result']['permanent_error'] is True
    assert store.fault is None


def test_admission_wait_uses_request_deadline(captured):
    api, store, _ = captured
    api.QUERY_TIMEOUT_SECONDS = .08
    with api._sql_lock:
        api._sql_paused = True
    try:
        status, body = request(api, {'sql': 'SELECT 1'})
        assert status == 504
        assert body['error'] == 'Request admission exceeded its time limit'
        assert api._sql_active == 0
        row, = records(store)
        assert row['result']['status'] == 'timed_out'
        assert store.fault is None
    finally:
        with api._sql_lock:
            api._sql_paused = False
            api._sql_lock.notify_all()


@pytest.mark.parametrize('endpoint', ['/query', '/call'])
def test_rejected_body_keeps_evidence_after_parent_has_ended(captured, monkeypatch, endpoint):
    from saas_bench import api_server
    api, store, executor = captured
    rejecting, release = threading.Event(), threading.Event()
    reject = api_server._APIHandler._reject_request

    def delayed_reject(handler, exc):
        rejecting.set()
        assert release.wait(5)
        return reject(handler, exc)

    monkeypatch.setattr(api_server._APIHandler, '_reject_request', delayed_reject)
    code = f'''import os,socket
with socket.create_connection(('127.0.0.1',int(os.environ['NOVAMIND_API_PORT']))) as s:
    header = 'POST {endpoint} HTTP/1.0\\r\\nContent-Length: 100\\r\\nX-Capture-Context: ' + os.environ['NOVAMIND_CAPTURE_CONTEXT'] + '\\r\\nX-Capture-Call: ' + 'a'*32 + '\\r\\n\\r\\n'
    s.sendall(header.encode() + b'{{"sql":')
'''
    try:
        assert executor.execute('bash', {'command': 'python -c ' + shlex.quote(code)}) == '(no output)'
        assert rejecting.wait(2)
    finally:
        release.set()
    settled(api)
    bash, rejected = records(store)
    assert rejected['request']['parent_event_id'] == bash['request']['event_id']
    assert rejected['result']['completed_at'] is not None
    assert rejected['result']['http_status'] == 400
    assert rejected['delivery']['send_state'] in ('sent', 'failed')
    assert store.fault is None
    with pytest.raises(RuntimeError, match='Unconfirmed client receive outcome'):
        store.assert_healthy()


def test_body_and_sql_share_one_deadline(server):
    server.QUERY_TIMEOUT_SECONDS = .15
    server.start()
    body = json.dumps({'sql': 'SELECT 1'}).encode()
    with server._lock:
        with socket.create_connection(('127.0.0.1', server.port), timeout=2) as connection:
            connection.sendall(f'POST /query HTTP/1.0\r\nContent-Length: {len(body)}\r\n\r\n'.encode() + body[:1])
            time.sleep(.1)
            connection.sendall(body[1:])
            response = connection.makefile('rb').read()
    assert response.startswith(b'HTTP/1.0 504'), response
    assert server._sql_active == 0


def test_sql_cell_size_is_bounded_before_returning_a_value(server, monkeypatch):
    monkeypatch.setitem(server.runtime_environment, 'CEOBENCH_SQL_CELL_LIMIT_BYTES', '1048576')
    with query_snapshot(server, time.monotonic() + 1) as (connection, _):
        import sqlite3
        assert connection.getlimit(sqlite3.SQLITE_LIMIT_LENGTH) == 1048576
    with pytest.raises(QueryResultTooLarge, match='SQL value or row exceeded 1048576 bytes'):
        execute_query(server, "SELECT replace(printf('%600000s', 'x'), ' ', 'xx') AS text")
    formatted = execute_query(server, "SELECT printf('%1048577s', 'x') AS text")
    assert formatted['rows'] == [{'text': None}]
    assert formatted['warning'] == public_sql.FORMATTING_NULL_WARNING
    assert execute_query(server, 'SELECT printf(NULL) AS n')['rows'] == [{'n': None}]
    empty = execute_query(server, "SELECT printf('') AS n, format('') AS f")
    assert empty['rows'] == [{'n': None, 'f': None}] and 'warning' not in empty
    invalid = execute_query(server, "SELECT printf('%k','x') AS n")
    assert invalid['rows'] == [{'n': None}]
    assert invalid['warning'] == public_sql.FORMATTING_NULL_WARNING
    assert execute_query(server, 'SELECT 1 AS n')['rows'] == [{'n': 1}]


def test_large_result_rejected_with_explicit_http_receipt(captured, monkeypatch):
    api, store, _ = captured
    monkeypatch.setitem(api.runtime_environment, 'CEOBENCH_SQL_RESULT_LIMIT_BYTES', '1024')
    status, body = request(api, {'sql': "WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x WHERE n<100) SELECT n, printf('%40s','x') AS text FROM x"})
    assert status == 413
    assert body['error'] == 'SQL result exceeded 1024 bytes. Narrow or aggregate the query.'
    settled(api)
    row, = records(store)
    assert row['result']['status'] == 'failed'
    assert row['result']['permanent_error'] is True
    assert store.fault is None


def test_serialized_response_budget_rejects_before_success_headers(captured, monkeypatch):
    from saas_bench import api_server
    api, store, _ = captured
    monkeypatch.setitem(api.runtime_environment, 'CEOBENCH_SQL_RESPONSE_LIMIT_BYTES', '256')
    status, body = request(api, {'sql': "SELECT printf('%300s','x') AS text"})
    assert status == 413
    assert body['error'] == 'SQL response exceeded 256 bytes. Narrow the query.'
    settled(api)
    row, = records(store)
    assert row['result']['http_status'] == 413
    assert store.fault is None
