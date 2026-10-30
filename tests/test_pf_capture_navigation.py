"""Capture dates and exact saved versions through PF navigation."""
from contextlib import closing
import json

import pytest

from saas_bench.execution_capture import finish_http
from saas_bench.pf_queries import PFQueries
from saas_bench.public_sql import execute_query
from saas_bench.sql_evidence import encoded
from pf_fixtures import chain, cite, forward, record_sql
from test_text_registry import workspace, captured, call, declaration, send
from test_pf_queries import query, read
from test_public_sql import server


def test_pagination_keeps_saved_graph_and_versions_after_world_changes(workspace, tmp_path, server):
    store, registry, executor = chain(workspace, tmp_path, server)
    first = forward(executor, limit=1)
    server.conn.execute('UPDATE ledger SET amount=1')
    server.conn.commit()
    executor.pf_queries = PFQueries(registry)
    page = query(executor, 'pf_dependencies', cursor=first['next_cursor'])
    assert page['items'][0]['target']['version'] == 'query1@v1'
    assert page['items'][0]['target']['sql'] == 'SELECT amount FROM ledger'
    assert query(executor, 'pf_dependencies', cursor=first['next_cursor']) == page
    assert first['root'] == page['root']
    assert not {'stale_check', 'checked_day', 'check_state'} & page.keys()
    assert all('check' not in row for row in first['items'] + page['items'])
    query(executor, 'pf_dependencies', target={'record': 'r2'})
    executor.execute('text_list', {})
    query(executor, 'pf_dependents', target={'record': 'r1'})
    read(executor, target={'record': 'r1'})
    with closing(store.connect()) as conn:
        assert conn.execute("SELECT count(*) FROM requests WHERE json_extract(request, '$.kind')='sql_query'").fetchone()[0] == 1


def test_derived_file_traces_captured_upstream_queries_and_exact_bytes(workspace, tmp_path, server):
    store, registry, executor = captured(workspace, tmp_path)
    server.sql_evidence = store
    script = store.begin_event('registered_script_execution', {'name': 'derive', 'day': 56})
    parent = store.begin_event('bash', {'command': 'python derive.py'}, parent=script)
    store.version(parent, 'code', 'raise RuntimeError("must never replay")', layer='executed_code')
    sqls = ['SELECT amount FROM ledger', 'SELECT COUNT(*) AS n FROM ledger']
    versions = [record_sql(store, sql, execute_query(server, sql), parent=parent) for sql in sqls]
    (workspace / 'derived.json').write_text('{"n":42}')
    version = store.version(parent, 'file', '{"n":42}', layer='file_bytes', object_id='derived.json')
    store.version(parent, 'workspace_after', encoded({'derived.json': {'version': version}}), layer='workspace_boundary')
    store.complete(parent, changed_paths=['derived.json'])
    store.complete(script)
    send(store, executor.execute('read_file', {'path': 'derived.json'}))
    ref = dict(evidence={'path': 'derived.json'}, purpose='current', select={'path': '/n'},
               predicate={'type': 'threshold', 'op': '>', 'value': 40})
    call(registry, 'create', **declaration(references=[ref]))
    server.conn.execute('UPDATE ledger SET amount=1')
    server.conn.commit()
    (workspace / 'derived.json').write_text('{"n":264}')
    before = server.conn.serialize()
    rows = forward(executor, 'r1')['items']
    assert rows[0]['select'] == ref['select'] and rows[0]['predicate'] == ref['predicate']
    assert rows[0]['target']['version'] == 'derived.json@v1'
    sql_rows = [row for row in rows if row['target'].get('sql')]
    assert {row['target']['sql'] for row in sql_rows} == set(sqls)
    assert {row['target']['version'] for row in sql_rows} == {registry.resolver.handle(v) for v in versions}
    assert all(row['origin'] == 'automatic_capture' for row in sql_rows)
    header, body, _ = read(executor, target={'version': 'derived.json@v1'})
    assert body == '{"n":42}' and header['target']['day'] == 56
    assert server.conn.serialize() == before
    with closing(store.connect()) as conn:
        assert conn.execute("SELECT count(*) FROM requests WHERE json_extract(request, '$.kind')='sql_query'").fetchone()[0] == 2
    limited = forward(executor, 'r1', depth=1)
    assert limited['unexpanded'] == ['derived.json@v1']
    assert 'depth limit' in executor.execute('pf_dependencies', {'target': {'record': 'r1'}, 'depth': 1})


def test_saved_receipts_dashboard_and_files_keep_their_captured_identity(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    event = store.begin_event('public_http', dict(method='POST', path='/call',
        parsed=dict(tool='set_prices', args={'B': 99})))
    finish_http(store, event, 200, '', encoded(dict(success=True, data={'B': 99})), {})
    cite(store, registry, executor, event + ':public_response')
    row = forward(executor, 'r1')['items'][0]
    assert row['target']['classification'] == 'write_receipt'
    assert row['target']['version'] == registry.resolver.handle(event + ':public_response')
    for value in ('old summary', 'new summary'):
        event = store.begin_event('dashboard_generation')
        version = store.version(event, 'dashboard', value, layer='dashboard')
        store.complete(event)
        if value == 'old summary':
            saved = version
            cite(store, registry, executor, saved)
    assert forward(executor)['items'][0]['target']['version'] == registry.resolver.handle(saved)
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration({'path': 'evidence.json'}))
    (workspace / 'evidence.json').unlink()
    secret = tmp_path / 'private-secret'
    secret.write_text('NEVER_CAPTURE_PRIVATE_SECRET')
    (workspace / 'evidence.json').symlink_to(secret)
    assert forward(executor, 'r3')['items'][0]['target']['version'] == 'evidence.json@v1'
    _, body, _ = read(executor, target={'version': 'evidence.json@v1'})
    assert body == '{"n":7}'
    with closing(store.connect()) as conn:
        assert not conn.execute("SELECT 1 FROM blobs WHERE instr(content,?)>0", (b'NEVER_CAPTURE_PRIVATE_SECRET',)).fetchone()


def test_cited_dates_follow_the_saved_version_group(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    for day, value in ((56, 7), (63, 8), (70, 7)):
        dashboard = store.begin_event('dashboard_generation', {'day': day})
        store.complete(dashboard)
        (workspace / 'evidence.json').write_text(json.dumps({'n': value}))
        send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration({'path': 'evidence.json'}))
    registry.sim_day = lambda: None
    row = forward(executor, 'r1')['items'][0]
    assert row['target']['version'] == 'evidence.json@v3'
    assert row['target']['day'] == 70


def test_script_before_dashboard_uses_host_day_not_previous_dashboard(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    dashboard = store.begin_event('dashboard_generation', {'day': 0})
    store.complete(dashboard)
    script = store.begin_event('registered_script_execution', {'name': 'report', 'day': 7})
    bash = store.begin_event('bash', {'command': 'report.py', 'note': 'Weekly snapshot'}, parent=script)
    run = store.begin_event('cli_python', {'script': 'report.py'}, parent=bash)
    version = store.version(run, 'file_after', '{"n":7}', layer='file_bytes', object_id='evidence.json')
    store.complete(run)
    store.complete(bash)
    store.complete(script)
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration(references=[dict(
        evidence={'path': 'evidence.json'}, purpose='current', select={'path': '/n'},
        predicate={'type': 'threshold', 'op': '>', 'value': 0})]))
    registry.sim_day = lambda: 14
    row = forward(executor, 'r1')['items'][0]
    assert row['target']['day'] == 7
    assert row['select'] == {'path': '/n'}
    assert row['predicate'] == {'type': 'threshold', 'op': '>', 'value': 0}
    assert query(executor, 'pf_log', target={'path': 'evidence.json'})['items'][0]['day'] == 7
    assert 'evidence.json@v1 · day 7' in executor.execute('bash', {'command': 'pf log evidence.json'})
    shown = executor.execute('bash', {'command': 'pf show evidence.json'})
    assert json.loads(shown.split('\n', 1)[0])['target']['day'] == 7
    assert 'note (day 7): "Weekly snapshot"' in executor.execute('bash', {'command': 'cat evidence.json'})
    dashboard = store.begin_event('dashboard_generation', {'day': 14})
    store.complete(dashboard)
    executor.execute('write_file', {'path': 'evidence.json', 'content': '{"n":8}', 'note': 'Revised snapshot'})
    log = executor.execute('bash', {'command': 'pf log evidence.json'})
    assert 'evidence.json@v2 · day 14' in log and 'evidence.json@v1 · day 7' in log
    diff = executor.execute('bash', {'command': 'pf diff evidence.json'})
    header = json.loads(diff.split('\n', 1)[0])
    assert header['baseline']['day'] == 7 and header['target']['day'] == 14
    assert '-{"n":7}' in diff and '+{"n":8}' in diff


def test_undated_handles_remain_unknown_in_cli_returns_and_notes(workspace, tmp_path):
    from saas_bench.registration_prompt import pf_memory_line
    store, registry, executor = captured(workspace, tmp_path)
    command = {'command': 'printf 7 > MEMORY.md', 'note': 'Undated snapshot'}
    executor.execute('bash', command)
    log = executor.execute('bash', {'command': 'pf log MEMORY.md'})
    assert 'MEMORY.md@v1 · day unknown' in log
    assert query(executor, 'pf_log', target={'path': 'MEMORY.md'})['items'][0]['day'] is None
    shown = executor.execute('bash', {'command': 'pf show MEMORY.md'})
    assert json.loads(shown.split('\n', 1)[0])['target']['day'] is None
    assert 'MEMORY.md@v1 · day unknown' in executor.execute('bash', {'command': 'pf search --text 7'})
    assert 'note (day unknown): "Undated snapshot"' in executor.execute('bash', {'command': 'cat MEMORY.md'})
    again = executor.execute('bash', command)
    assert 'same as day' not in again
    blame = executor.execute('bash', {'command': 'pf blame MEMORY.md'})
    assert 'v1 dunknown' in blame and '(day unknown): "Undated snapshot"' in blame
    version = store.latest_version('MEMORY.md', 'file_bytes')[0]
    assert 'written day unknown' in pf_memory_line(store, version)
    send(store, executor.execute('read_file', {'path': 'MEMORY.md'}))
    receipt = registry.execute('create', declaration({'path': 'MEMORY.md'}))
    assert 'MEMORY.md@v1 (day unknown)' in receipt


def test_handle_date_index_obeys_the_shared_cancellation_budget(workspace, tmp_path):
    from saas_bench.evidence_handles import index
    from saas_bench.pf_deadline import budget_scope
    store, registry, executor = captured(workspace, tmp_path)
    executor.execute('read_file', {'path': 'evidence.json'})
    with pytest.raises(TimeoutError, match='PF call cancelled'):
        with budget_scope() as budget:
            budget.cancelled.set()
            index(store).refresh()
