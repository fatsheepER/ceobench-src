"""SQL capture uses synthetic worlds and local HTTP only."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from saas_bench.api_server import NovaMindAPIServer
from saas_bench.public_sql import PUBLIC_POLICY_VERSION
from saas_bench.sql_evidence import FORMAT, SQLEvidenceStore, whole_rows
from test_public_sql import server


def identity(branch='prefix', **kwargs):
    return dict(format=FORMAT, run_id='test-run', branch_id=branch, data_source_id='world-1', **kwargs)


def request(api, sql=None, raw=None):
    body = raw if raw is not None else json.dumps({'sql': sql}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{api.port}/query', data=body)
    try:
        response = urllib.request.urlopen(req)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return response.status, response.read()


def event_ids(store):
    with closing(store.connect()) as conn:
        return [r[0] for r in conn.execute('SELECT event_id FROM requests ORDER BY rowid')]


def settled(api):
    # The body may reach the client before the server commits send completion.
    with api._sql_lock:
        assert api._sql_lock.wait_for(lambda: api._sql_active == 0, timeout=5)


@pytest.fixture
def captured(server, tmp_path):
    server.sql_evidence = SQLEvidenceStore(tmp_path / 'private' / 'evidence.sqlite', identity())
    server.start()
    return server, server.sql_evidence


@pytest.mark.parametrize('sql,raw,expected', [
    ('SELECT amount,note FROM ledger', None, 200),
    ('SELECT amount FROM ledger WHERE 0', None, 200),
    ("SELECT 1 AS i, 1.0 AS f, '1' AS s, NULL AS n UNION ALL SELECT 1,1.0,'1',NULL", None, 200),
    ('SELECT 1 AS x,2 AS x', None, 200),
    ("SELECT X'4142' AS b", None, 200),
    ('SELECT 1e999 AS x', None, 200),
    ('SELECT * FROM subscriptions', None, 200),
    ('WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x WHERE n<5001) SELECT n FROM x', None, 200),
    ('WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x WHERE n<5000) SELECT n FROM x', None, 200),
    ('WITH a AS (SELECT 1) UPDATE ledger SET amount=99', None, 500),
    ('WITH a AS (SELECT 1) UPDATE main.ledger SET amount=99', None, 403),
    ('SELECT * FROM group_insight_snapshots', None, 403),
    ('SELECT actual_completion_day AS x FROM main.research_projects', None, 403),
    ('SELECT missing FROM ledger', None, 500),
    ('SELECT ?', None, 500),
    ('', None, 400),
    (None, b'not json', 400),
])
def test_wire_equivalence_and_saved_states(captured, sql, raw, expected):
    api, store = captured
    before = api.conn.serialize()
    actual = request(api, sql, raw)
    settled(api)
    event, = event_ids(store)
    meta, body = store.get_content(event + ':public_response')
    assert (expected, body) == actual
    record = store.read_event(event)
    assert record['result']['http_status'] == expected
    assert record['delivery']['send_state'] == 'sent'
    assert record['delivery']['receive_state'] == 'unknown'
    assert 'Server:' in record['result']['headers'] and 'Date:' in record['result']['headers']
    assert meta['content_time']['status'] == 'unknown'
    api.sql_evidence = None
    assert request(api, sql, raw) == actual
    assert api.conn.serialize() == before
    if sql == 'SELECT 1 AS x,2 AS x':
        assert record['result']['comparison']['reason'] == 'duplicate_columns'
    if sql == "SELECT X'4142' AS b":
        assert record['result']['comparison']['reason'] == 'blob_coercion'
    if b'"truncated": true' in body:
        assert meta['source_truncated']
        assert record['result']['comparison']['scope'] == 'returned_subset'


def test_repeat_change_identity_and_restart(captured):
    api, store = captured
    sql = ' SELECT amount FROM ledger '
    a = request(api, sql)
    assert request(api, sql) == a
    with api._lock:
        api.conn.execute('UPDATE ledger SET amount=43')
        api.conn.commit()
        api.tools.current_day = 7
    assert request(api, sql) != a
    request(api, sql.strip())
    settled(api)
    events = event_ids(store)
    requests = [store.read_event(e)['request'] for e in events]
    assert len({r['query_id'] for r in requests[:3]}) == 1
    assert requests[3]['query_id'] != requests[0]['query_id']
    first = store.get_content(events[0] + ':public_response')
    second = store.get_content(events[1] + ':public_response')
    assert first[0]['blob_sha256'] == second[0]['blob_sha256']
    assert first[0]['version_id'] != second[0]['version_id']
    assert second[0]['previous_version'] == first[0]['version_id']
    assert store.read_event(events[2])['result']['day'] == 7
    reopened = SQLEvidenceStore(store.path, identity())
    assert reopened.get_content(first[0]['version_id'])[1] == a[1]
    api.sql_evidence = reopened
    request(api, sql)
    settled(api)
    assert event_ids(reopened)[-1].endswith('/5')


def test_comparison_preserves_rows_types_and_order():
    def compare(rows, columns=['x'], **kwargs):
        return whole_rows(json.dumps(dict(success=True, columns=columns, rows=rows, **kwargs)).encode())[1]
    assert compare([{'x': 20}, {'x': 10}, {'x': 20}]) == compare([{'x': 10}, {'x': 20}, {'x': 20}])
    assert compare([{'x': 20}, {'x': 20}]) != compare([{'x': 20}])
    assert len({compare([{'x': x}]) for x in [None, '', 1, 1.0, '1', True]}) == 6
    assert compare([{'a': 1, 'b': 2}], ['a', 'b']) != compare([{'a': 2, 'b': 1}], ['a', 'b'])
    assert compare([{'x': 1}], ['x']) != compare([{'y': 1}], ['y'])


def test_failure_keeps_response_and_stops_checkpoint(captured, monkeypatch):
    api, store = captured
    def fail(*args):
        raise OSError('injected disk failure')
    monkeypatch.setattr(store, 'finish', fail)
    assert request(api, 'SELECT 1')[0] == 200
    settled(api)
    assert store.fault_path.exists()
    assert store.read_event(event_ids(store)[0])['result']['status'] == 'result_unknown'
    with pytest.raises(RuntimeError, match='capture failed'):
        api.checkpoint(0)
    with urllib.request.urlopen(f'http://127.0.0.1:{api.port}/health') as response:
        assert json.load(response)['status'] == 'capture_failed'


def test_timeout_capture_and_following_query(captured):
    api, store = captured
    api.QUERY_TIMEOUT_SECONDS = 0.02
    assert request(api, 'WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x) SELECT sum(n) FROM x')[0] == 504
    api.QUERY_TIMEOUT_SECONDS = 2
    assert request(api, 'SELECT 1')[0] == 200
    settled(api)
    assert store.read_event(event_ids(store)[0])['result']['status'] == 'timed_out'


def test_checkpoint_waits_for_capture_and_blocks_new_admission(captured, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    api, store = captured
    entered, release = threading.Event(), threading.Event()
    finish = store.finish
    def slow(*args):
        entered.set()
        assert release.wait(5)
        return finish(*args)
    monkeypatch.setattr(store, 'finish', slow)
    api.checkpoint_callback = lambda: store.snapshot(tmp_path / 'checkpoint.sqlite')
    with ThreadPoolExecutor(3) as pool:
        first = pool.submit(request, api, 'SELECT 1')
        assert entered.wait(3)
        checkpoint = pool.submit(api.checkpoint, 0)
        with api._sql_lock:
            assert api._sql_lock.wait_for(lambda: api._sql_paused, timeout=3)
        second = pool.submit(request, api, 'SELECT 2')
        assert not checkpoint.done()
        release.set()
        assert first.result(timeout=5)[0] == 200
        assert checkpoint.result(timeout=5)['cutoff'] == 1
        assert second.result(timeout=5)[0] == 200


def test_begin_failure_and_send_failure_preserve_observed_facts(captured, monkeypatch):
    api, store = captured
    from saas_bench.api_server import _APIHandler
    from types import SimpleNamespace
    handler = _APIHandler.__new__(_APIHandler)
    handler.server = SimpleNamespace(_api_server=api, _role="ceo")
    handler.connection = SimpleNamespace(settimeout=lambda _: None)
    handler.request_version = 'HTTP/1.1'
    handler._sql_event = store.begin(b'{"sql":"SELECT 1"}', PUBLIC_POLICY_VERSION)
    handler._sql_execution = {}
    handler._headers_buffer = []
    statuses = []
    handler.send_response = statuses.append
    handler.send_header = lambda key, value: handler._headers_buffer.append(f'{key}: {value}\r\n'.encode())
    def broken(_):
        raise BrokenPipeError('injected socket failure')
    handler.wfile = SimpleNamespace(write=broken)
    with pytest.raises(BrokenPipeError):
        handler._send_query_json(dict(success=True, columns=['x'], rows=[dict(x=1)]))
    assert statuses == [200]
    event = handler._sql_event
    assert store.read_event(event)['result']['status'] == 'succeeded'
    assert store.read_event(event)['delivery']['send_state'] == 'failed'
    assert store.get_content(event + ':public_response')[1]
    def fail(*args):
        raise OSError('request persistence failed')
    monkeypatch.setattr(store, 'begin', fail)
    assert request(api, 'SELECT 2')[0] == 200
    settled(api)
    assert len(event_ids(store)) == 1 and store.fault_path.exists()


def test_parallel_queries_keep_every_event(captured):
    from concurrent.futures import ThreadPoolExecutor
    api, store = captured
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: request(api, 'SELECT 1'), range(8)))
    settled(api)
    assert all(status == 200 for status, _ in results)
    events = event_ids(store)
    assert len(events) == 8
    store.assert_healthy()
    assert len({store.get_content(e + ':public_response')[0]['blob_sha256'] for e in events}) == 1


def test_committed_data_survives_process_exit(tmp_path):
    path = tmp_path / 'evidence.sqlite'
    for phase in ('request', 'result'):
        db = path.with_name(phase + '.sqlite')
        code = '''
import os, sys
from saas_bench.sql_evidence import SQLEvidenceStore, FORMAT
s = SQLEvidenceStore(sys.argv[1], dict(format=FORMAT, run_id='r', branch_id='prefix', data_source_id='w'))
e = s.begin(b'{"sql":"SELECT 1"}', 'public-sql-v1')
if sys.argv[2] == 'result':
    s.finish(e, 200, [], b'{"success":true,"columns":["x"],"rows":[{"x":1}]}', {})
os._exit(0)
'''
        subprocess.run([sys.executable, '-c', code, str(db), phase], check=True)
        store = SQLEvidenceStore(db, dict(format=FORMAT, run_id='r', branch_id='prefix', data_source_id='w'))
        assert store.read_event('r/prefix/1')['delivery']['send_state'] == 'unknown'
        if phase == 'result':
            assert store.get_content('r/prefix/1:public_response')[1]
        else:
            assert store.read_event('r/prefix/1')['result']['status'] == 'result_unknown'
        with pytest.raises(RuntimeError, match='Unconfirmed'):
            store.assert_healthy()


def test_settle_waits_only_for_late_send_records(tmp_path):
    store = SQLEvidenceStore(tmp_path / 'e.sqlite', identity())
    event = store.begin_event('public_http', {'path': '/dashboard'}, requires_delivery=True)
    store.complete(event)
    with pytest.raises(RuntimeError, match='Unconfirmed SQL evidence outcome'):
        store.assert_healthy()
    late = threading.Timer(.2, store.delivered, (event, 'sent'))
    late.start()
    store.assert_healthy(settle=5)
    late.join()
    lost = store.begin_event('public_http', {'path': '/dashboard'}, requires_delivery=True)
    store.complete(lost)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match='Unconfirmed SQL evidence outcome: ' + lost):
        store.assert_healthy(settle=.3)
    assert time.monotonic() - started >= .3
    store.delivered(lost, 'sent')
    running = store.begin_event('public_http', {'path': '/dashboard'}, requires_delivery=True)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match='Unconfirmed SQL evidence outcome: ' + running):
        store.assert_healthy(settle=5)
    assert time.monotonic() - started < 1


def test_snapshot_branch_visibility_and_tampering(captured, tmp_path):
    api, store = captured
    request(api, 'SELECT 1')
    settled(api)
    snap = tmp_path / 'frozen.sqlite'
    saved = store.snapshot(snap)
    request(api, 'SELECT 2')
    settled(api)
    import shutil
    shutil.copy2(snap, tmp_path / 'left.sqlite')
    child = SQLEvidenceStore(tmp_path / 'left.sqlite', identity('left', parent_branch='prefix', fork_seq=saved['cutoff']))
    assert child.get_content('test-run/prefix/1:public_response')[1]
    with pytest.raises(KeyError):
        child.get_content('test-run/prefix/2:public_response')
    e = child.begin(b'{"sql":"SELECT 1"}', PUBLIC_POLICY_VERSION)
    child.finish(e, 200, [], b'{"success":true,"columns":[],"rows":[]}', {})
    child.delivered(e, 'sent')
    with pytest.raises(KeyError):
        store.get_content(e + ':public_response')
    with closing(child.connect()) as conn, conn:
        conn.execute("UPDATE blobs SET content=x'00'")
    with pytest.raises(ValueError, match='checksum'):
        child.get_content(e + ':public_response')


@pytest.mark.skipif(sys.platform != 'linux', reason='Requires real Linux bubblewrap')
def test_private_store_is_inaccessible(captured):
    import shlex
    from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
    api, store = captured
    request(api, 'SELECT 1')
    executor = BashAgentToolExecutor(api.script_workspace, require_sandbox=True)
    executor.verify_sandbox()
    assert executor.execute('bash', {'command': 'test ! -r ' + shlex.quote(str(store.path)) + ' && echo private'}).strip() == 'private'
