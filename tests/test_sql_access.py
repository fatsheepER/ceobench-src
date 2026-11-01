"""SQL authorizer observations through real public query and PF clients."""
import json
from pathlib import Path
import shlex
import shutil
import sqlite3

import pytest

from saas_bench.pf_queries import PFQueries
from saas_bench.public_sql import PUBLIC_COLUMNS, QueryDenied, SnapshotUnavailable, execute_query
from saas_bench.text_registry import TextRegistry
from test_execution_capture import captured as http_captured, records
from test_pf_candidates import day, drain, query
from test_public_sql import server
from test_sql_evidence import request, settled
from test_text_registry import captured, workspace


def test_sql_access_records_view_expansion_without_claiming_column_use(server):
    metadata = {}
    before = server.conn.serialize()
    result = execute_query(server, 'SELECT amount AS selected_amount FROM ledger WHERE day=0', metadata=metadata)
    assert result['rows'][0]['selected_amount'] == 42
    access = metadata.get('sql_access')
    assert access is not None, 'SQL authorization access observations must be captured'
    assert access['format'] == 'pf.sql_access.v1'
    assert access['status'] == 'recorded'
    raw = access['raw_reads']
    assert any(row['database'] == 'temp' and row['column'] == 'amount' for row in raw)
    assert any(row['database'] == 'main' and row['column'] == 'note' for row in raw)
    assert any(row['table'] == 'ledger' and row['column'] == 'amount' and
               row['basis'] == 'public_view_reference' for row in access['public_reads'])
    assert any(row['table'] == 'ledger' and row['column'] == 'note' and
               row['basis'] == 'named_source_authorization' for row in access['public_reads'])
    assert server.conn.serialize() == before


def test_sql_access_public_filter_and_directory_preserve_event_and_unknown(http_captured):
    api, store, executor = http_captured
    registry = TextRegistry(executor.workspace_path, 'pf', store, sim_day=lambda: 0)
    executor.text_registry, executor.pf_queries = registry, PFQueries(registry)
    shutil.copy(Path(__file__).parents[1] / 'public/novamind-operation', executor.workspace_path)
    day(store, 0)
    for _ in range(2):
        output = executor.execute('bash', {'command': './novamind-operation query "SELECT amount FROM ledger"'})
        assert '42' in output
    output, page = query(executor, 'pf candidates --table ledger --column amount')
    assert page['total_in_scope'] == 2
    assert len({row['event_ref'] for row in page['items']}) == 2
    assert all(row['sql_access']['status'] == 'recorded' for row in page['items'])
    assert 'authorization' in output.lower() and 'not proof of use' in output
    directory_text, directory = query(executor, 'pf candidates --sql-accesses --table ledger')
    amount = next(row for row in directory['items'] if row['column'] == 'amount')
    assert amount['query_events'] == 2
    assert '--column amount' in amount['navigation']
    assert directory['sql_access_coverage']['unknown_events'] == 0
    assert 'raw_reads' not in directory_text
    persisted = [r for r in records(store) if r['request']['kind'] == 'sql_query']
    assert all(r['result']['sql_access']['raw_reads'] for r in persisted)
    settled(api)
    store.assert_healthy()


@pytest.mark.parametrize('sql,expected', [
    ('SELECT l.amount AS renamed FROM ledger l WHERE l.day=0', {('ledger', 'amount'), ('ledger', 'day')}),
    ('SELECT l.amount FROM ledger l JOIN subscriptions s ON l.day=s.start_day WHERE s.plan=\'A\'',
     {('ledger', 'amount'), ('ledger', 'day'), ('subscriptions', 'start_day'), ('subscriptions', 'plan')}),
    ('SELECT category,sum(amount) FROM ledger GROUP BY category ORDER BY category', {('ledger', 'category'), ('ledger', 'amount')}),
    ('SELECT count(amount) FROM ledger WHERE day=0', {('ledger', 'amount'), ('ledger', 'day')}),
    ('SELECT amount FROM ledger WHERE 0', {('ledger', 'amount')}),
    ('WITH hidden_name AS (SELECT amount,day FROM ledger) SELECT amount FROM hidden_name', {('ledger', 'amount'), ('ledger', 'day')}),
])
def test_sql_access_matches_authorizer_public_reference_observations(server, sql, expected):
    metadata = {}
    execute_query(server, sql, metadata=metadata)
    access = metadata['sql_access']
    observed = {(row['table'], row['column']) for row in access['public_reads'] if row['basis'] == 'public_view_reference'}
    assert expected <= observed
    assert all(row['action'] == sqlite3.SQLITE_READ for row in access['raw_reads'])
    assert all(row['decision'] == 'allowed' for row in access['raw_reads'])
    assert all(row['table'] in PUBLIC_COLUMNS and row['column'] in PUBLIC_COLUMNS[row['table']]
               for row in access['public_reads'] if row['column'] is not None)


def test_sql_access_count_constant_cte_and_denied_states(server):
    count = {}
    execute_query(server, 'SELECT COUNT(*) FROM ledger', metadata=count)
    assert any(row['column'] is None and row['basis'] == 'column_free_authorization' for row in count['sql_access']['public_reads'])
    assert not any(row['column'] == '*' for row in count['sql_access']['public_reads'])
    cte = {}
    execute_query(server, 'WITH ledger AS (SELECT amount FROM main.ledger) SELECT amount FROM ledger', metadata=cte)
    reads = cte['sql_access']['public_reads']
    assert reads and all(row['basis'] == 'named_source_authorization' for row in reads)
    assert {row['column'] for row in reads} == {'amount'}
    literal = {}
    execute_query(server, 'SELECT 1', metadata=literal)
    assert literal['sql_access']['status'] == 'recorded' and literal['sql_access']['raw_reads'] == []
    denied = {}
    with pytest.raises(QueryDenied):
        execute_query(server, 'SELECT actual_completion_day FROM main.research_projects', metadata=denied)
    assert denied['error_kind'] == 'denied'
    assert denied['sql_access']['raw_reads'][-1]['decision'] == 'denied'
    assert denied['sql_access']['public_reads'] == []
    server._operation_failed = True
    missing = {}
    with pytest.raises(SnapshotUnavailable):
        execute_query(server, 'SELECT amount FROM ledger', metadata=missing)
    assert missing['sql_access']['status'] == 'not_started'
    assert missing['sql_access']['raw_reads'] == []


def test_sql_access_oracle_raw_private_and_named_context_stay_private(server):
    server.oracle_mode = True
    metadata = {}
    execute_query(server, 'SELECT snapshot_c_max FROM group_insight_snapshots', metadata=metadata)
    assert any(row['table'] == 'group_insight_snapshots' for row in metadata['sql_access']['raw_reads'])
    assert metadata['sql_access']['public_reads'] == []
    context = {}
    execute_query(server, 'WITH PRIVATE_CONTEXT_SENTINEL AS (SELECT amount FROM ledger) SELECT amount FROM PRIVATE_CONTEXT_SENTINEL', metadata=context)
    assert any(row['source'] == 'PRIVATE_CONTEXT_SENTINEL' for row in context['sql_access']['raw_reads'])
    assert 'PRIVATE_CONTEXT_SENTINEL' not in json.dumps(context['sql_access']['public_reads'])
    assert any(row['source_state'] == 'redacted' for row in context['sql_access']['public_reads'])


def prepare_index(captured):
    api, store, executor = captured
    registry = TextRegistry(executor.workspace_path, 'pf', store, sim_day=lambda: 0)
    executor.text_registry, executor.pf_queries = registry, PFQueries(registry)
    shutil.copy(Path(__file__).parents[1] / 'public/novamind-client', executor.workspace_path / 'novamind-operation')
    day(store, 0)
    return api, store, executor


def test_sql_access_results_rejections_failures_truncation_and_next_call(http_captured, monkeypatch):
    api, store, executor = prepare_index(http_captured)
    monkeypatch.setitem(api.runtime_environment, 'CEOBENCH_SQL_ROW_LIMIT', '1')
    sqls = ['SELECT amount FROM ledger UNION ALL SELECT 99',
            'SELECT amount FROM ledger WHERE abs(amount,1)',
            'SELECT amount,(SELECT snapshot_c_max FROM group_insight_snapshots) FROM ledger',
            'SELECT 1 AS constant']
    for sql in sqls:
        executor.execute('bash', {'command': './novamind-operation query ' + shlex.quote(sql)})
    assert request(api, raw=b'{"sql":null}')[0] == 400
    settled(api)
    _, page = query(executor, 'pf candidates --history --kind sql')
    assert {item['status'] for item in page['items']} >= {'succeeded', 'failed', 'rejected'}
    assert any(item['capture']['source_truncated'] for item in page['items'])
    assert page['sql_access_coverage']['not_started_events'] == 1
    assert page['sql_access_coverage']['recorded_empty_events'] >= 1
    filtered_text, filtered = query(executor, 'pf candidates --history --table ledger --column amount')
    assert all(item['kind'] == 'sql' for item in filtered['items'])
    assert any(item['status'] == 'rejected' for item in filtered['items'])
    _, detail_page = query(executor, 'pf more ' + filtered['detail_cursor'])
    detail = detail_page['detail_text']
    assert 'raw_reads' not in detail and '"decision":' not in detail
    assert '"result_coverage": "truncated"' in detail and '"error_kind": "denied"' in detail
    _, directory = query(executor, 'pf candidates --history --sql-accesses --table ledger --column amount')
    row, = directory['items']
    assert row['truncated_events'] == 1 and row['outcomes']['rejected'] == 1
    assert row['query_events'] == filtered['total_in_scope']
    assert 'next_value' in executor.execute('bash', {'command': './novamind-operation query "SELECT 8 AS next_value"'})
    settled(api)
    store.assert_healthy()


def test_sql_access_directory_frozen_pages_policy_recheck_and_old_unknown(http_captured, monkeypatch):
    api, store, executor = prepare_index(http_captured)
    executor.execute('bash', {'command': './novamind-operation query "SELECT * FROM customers,subscriptions,research_projects LIMIT 0"'})
    old_event = store.begin(b'{"sql":"SELECT old_unrecorded"}', 'public-sql-v2')
    store.finish(old_event, 200, '', b'{"success":true,"columns":[],"rows":[],"row_count":0}', {'day': 0})
    store.delivered(old_event, 'sent')
    _, page = query(executor, 'pf candidates --sql-accesses')
    assert page['has_more'] and page['returned'] == 20
    assert page['sql_access_coverage']['unknown_events'] == 1
    cursor = page['next_cursor']
    before = len([row for row in records(store) if row['request']['kind'] == 'sql_query'])
    executor.execute('bash', {'command': './novamind-operation query "SELECT amount FROM ledger"'})
    items = drain(executor, page)
    assert len(items) == page['total_in_scope'] and not any(row['table'] == 'ledger' for row in items)
    assert len({(row['table'], row['column']) for row in items}) == len(items)
    assert all(row['query_events'] == 1 for row in items)
    assert len([row for row in records(store) if row['request']['kind'] == 'sql_query']) == before + 1
    first = items[0]
    _, navigated = query(executor, first['navigation'])
    assert navigated['total_in_scope'] == 1
    assert navigated['sql_access_coverage']['unknown_events'] == 1
    from saas_bench.sql_access import PUBLIC_COLUMNS as current_policy
    monkeypatch.setitem(current_policy, first['table'], tuple(c for c in current_policy[first['table']] if c != first['column']))
    rejected = executor.execute('bash', {'command': 'pf more ' + cursor})
    assert '[exit code: 1]' in rejected and 'no longer public' in rejected
    assert f"{first['table']}.{first['column']}" not in rejected


@pytest.mark.parametrize('flags', ['--column amount', '--table group_insight_snapshots',
                                  '--table research_projects --column actual_completion_day',
                                  '--sql-accesses --kind file'])
def test_sql_access_bad_filters_are_explicit_without_private_catalog(workspace, tmp_path, flags):
    _, _, executor = captured(workspace, tmp_path)
    result = executor.execute('bash', {'command': 'pf candidates ' + flags})
    assert '[exit code: 1]' in result
    assert 'Run pf help' in result or 'Error:' in result


def test_sql_access_finish_is_atomic_and_conflicting_observation_is_rejected(http_captured):
    api, store, executor = prepare_index(http_captured)
    executor.execute('bash', {'command': './novamind-operation query "SELECT amount FROM ledger"'})
    settled(api)
    row = next(row for row in records(store) if row['request']['kind'] == 'sql_query')
    event = row['request']['event_id']
    record = row['result']
    body = store.get_content(event + ':public_response')[1]
    store.finish(event, record['http_status'], record['headers'], body, record)
    altered = dict(record, sql_access=dict(record['sql_access'], raw_reads=[]))
    with pytest.raises(ValueError, match='Conflicting SQL evidence finish'):
        store.finish(event, record['http_status'], record['headers'], body, altered)
    assert store.read_event(event)['result']['sql_access'] == record['sql_access']
    store.assert_healthy()



def test_sql_access_column_free_navigation_and_same_read_matching(http_captured):
    api, store, executor = prepare_index(http_captured)
    for sql in ('SELECT count(*) FROM ledger', 'SELECT amount FROM ledger'):
        executor.execute('bash', {'command': './novamind-operation query ' + shlex.quote(sql)})
    text, page = query(executor, 'pf candidates --sql-accesses --table ledger')
    free = next(row for row in page['items'] if row['column'] is None)
    assert free['query_events'] == 1 and free['navigation_scope'] == 'all_columns_for_table'
    assert 'Table-wide navigation includes other column observations' in text
    _, whole_table = query(executor, free['navigation'])
    assert whole_table['total_in_scope'] == 2
    executor.execute('bash', {'command': './novamind-operation query "SELECT l.amount,c.day FROM main.ledger l,main.config_history c"'})
    _, direct = query(executor, 'pf candidates --text "main.config_history" --table ledger --column day')
    assert direct['total_in_scope'] == 0
    settled(api)
    store.assert_healthy()


def test_sql_access_old_cursor_and_late_result_keep_frozen_meaning(http_captured):
    api, store, executor = prepare_index(http_captured)
    executor.execute('bash', {'command': './novamind-operation query "SELECT amount FROM ledger"'})
    _, original = query(executor, 'pf candidates --kind sql')
    cursors = store.load_state('pf_cursors')
    old_name = cursors[original['detail_cursor']]['candidate_snapshot']
    old = store.load_state(old_name)
    for field in ('view', 'sql_access_coverage', 'sql_authorization_items'):
        old.pop(field)
    for field in ('table', 'column'):
        old['scope'].pop(field)
    for item in old['items']:
        item.pop('sql_access')
        item.pop('access_match_bases')
    store.save_state(old_name, old)
    old_text, old_page = query(executor, 'pf more ' + original['detail_cursor'])
    assert old_page['cutoff'] == original['cutoff']
    assert 'sql_access' not in old_text and 'SQL access coverage:' not in old_text
    pending = store.begin_event('sql_query', {'sql': 'SELECT amount FROM ledger'})
    metadata = {}
    response = execute_query(api, 'SELECT amount FROM ledger', metadata=metadata)
    _, frozen = query(executor, 'pf candidates --sql-accesses --table ledger --column amount')
    assert frozen['sql_access_coverage']['unknown_events'] == 1
    assert frozen['items'][0]['query_events'] == 1
    store.version(pending, 'public_response', json.dumps(response).encode(), layer='server_public_response')
    store.complete(pending, 'succeeded', **metadata, result_coverage='complete_for_query')
    text, same = query(executor, 'pf more ' + frozen['detail_cursor'])
    assert same['cutoff'] == frozen['cutoff'] and same['sql_access_coverage']['unknown_events'] == 1
    assert json.loads(same['detail_text'])[0]['query_events'] == 1
    _, fresh = query(executor, 'pf candidates --sql-accesses --table ledger --column amount')
    assert fresh['sql_access_coverage']['unknown_events'] == 0
    assert fresh['items'][0]['query_events'] == 2
    settled(api)
    store.assert_healthy()
