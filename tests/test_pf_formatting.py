"""Compact PF returns and saved history formatting."""
import json
import re

from pf_fixtures import chain, cite, record_sql
from saas_bench.public_sql import execute_query
from test_text_registry import workspace, captured, call, declaration, send
from test_public_sql import server
from test_pf_queries import relation_details

CLOCK = re.compile(r'\d{4}-\d\d-\d\dT\d\d:\d\d')


def test_compact_lines_carry_day_source_and_declaration_metadata(workspace, tmp_path, server):
    store, registry, executor = chain(workspace, tmp_path, server, select={'col': 'amount'},
        predicate={'type': 'threshold', 'op': '>=', 'value': 40}, note='Keep at least forty')
    compact = executor.execute('pf_dependencies', {'target': {'record': 'r2'}})
    lines = compact.splitlines()
    assert lines[0] == 'PF relations: saved records only; no replay or validity judgment.'
    scope = json.loads(next(line.removeprefix('Scope: ') for line in lines if line.startswith('Scope: ')))
    assert scope['depth'] == 1 and not scope['all_revisions'] and not scope['include_candidates']
    assert scope['time'] == 'all_saved_days'
    assert scope['current_means'] == 'latest saved revision; no validity judgment'
    counts = json.loads(next(line.removeprefix('Counts: ') for line in lines if line.startswith('Counts: ')))
    assert counts['endpoint_rows'] == counts['declaration_revisions'] == 1
    assert 'Root: r2.1' in lines and 'returned=1; total_in_scope=1; has_more=false.' in lines
    assert 'r1.1 -> r2.1 | author=ceo; revision_status=active; depth=1' in compact
    assert compact.count(' | author_declared | ') == 1
    detail = relation_details(executor, 'pf depend r2 --depth 2 --detail')
    assert detail['scope']['depth'] == 2 and detail['scope']['time'] == 'all_saved_days'
    assert detail['root']['record']['sim_day'] == 7
    assert CLOCK.search(detail['cutoff']['captured_at'])
    direct, leaf = detail['items']
    assert direct['source']['ref']['cite'] == leaf['target']['ref']['cite'] == 'r1.1'
    assert direct['target']['ref']['cite'] == 'r2.1'
    binding = store.load_state('declaration:r1.1')['references'][0]
    assert leaf['source']['ref']['cite'] == registry.resolver.handle(binding['version_id'])
    assert leaf['reference']['select'] == {'col': 'amount'}
    assert leaf['reference']['predicate'] == {'op': '>=', 'type': 'threshold', 'value': 40}
    assert leaf['note'] == 'Keep at least forty'
    assert all(row['sim_day'] == 7 and row['author_role'] == 'ceo'
               and row['reason'] == 'Initial observation' and 'check' not in row
               for row in detail['items'])
    for text in (executor.execute('pf_search', {'object': {'kind': 'plan', 'id': 'B'}, 'detail': True}),
                 executor.execute('pf_read', {'target': {'record': 'r1'}, 'mode': 'history', 'detail': True}),
                 executor.execute('pf_read', {'target': {'record': 'r1'}}), executor.execute('text_list', {}),
                 registry.path.read_text()):
        assert not CLOCK.search(text), text
    header = json.loads(executor.execute('pf_read', {'target': {'record': 'r1'}}).split('\n', 1)[0])
    assert set(header) == {'delivery', 'next_cursor', 'range', 'target', 'total_chars'}
    assert set(header['target']) == {'version', 'day', 'what', 'owner'}
    assert header['target']['owner'] == 'ceo'


def test_saved_query_versions_have_explicit_diff_and_reverse_navigation(workspace, tmp_path, server):
    store, registry, executor = captured(workspace, tmp_path)
    server.sql_evidence = store
    sql = 'SELECT category, SUM(amount) AS total FROM ledger GROUP BY category'
    first = record_sql(store, sql, execute_query(server, sql))
    cite(store, registry, executor, first)
    server.conn.execute('UPDATE ledger SET amount=amount+5')
    server.conn.commit()
    second = record_sql(store, sql, execute_query(server, sql))
    old, new = (registry.resolver.handle(v) for v in (first, second))
    diff = executor.execute('bash', {'command': f'pf diff {old} {new}'})
    assert f'--- {old}\n+++ {new}' in diff
    before = json.loads(next(line[1:] for line in diff.splitlines() if line.startswith('-{')))
    after = json.loads(next(line[1:] for line in diff.splitlines() if line.startswith('+{')))
    assert before['rows'] == [{'category': 'operations', 'total': 42.0}]
    assert after['rows'] == [{'category': 'operations', 'total': 47.0}]
    assert f'Before changing this source: pf rdepend {old} --detail.' in diff
    assert 'r1.1' in executor.execute('bash', {'command': f'pf rdepend {old} --detail'})
    history = executor.execute('pf_read', {'target': {'sql': sql}, 'mode': 'history'})
    assert history.startswith('Versions of SQL: ') and ': 2 saved' in history
    assert history.index(new) < history.index(old)


def test_saved_query_read_keeps_sql_formatting_warning(workspace, tmp_path, server, monkeypatch):
    monkeypatch.setitem(server.runtime_environment, 'CEOBENCH_SQL_CELL_LIMIT_BYTES', '1048576')
    store, registry, executor = chain(workspace, tmp_path, server,
        sql="SELECT printf('%1048577s','x') AS n")
    binding = store.load_state('declaration:r1.1')['references'][0]
    handle = registry.resolver.handle(binding['version_id'])
    text = executor.execute('bash', {'command': f'pf show {handle} --full'})
    assert 'configured value limit' in text
    assert 'SQLite formatting function returned NULL' in text
    delivered = send(store, text)
    wire = json.loads(store.get_content(delivered + ':wire')[1])['messages'][0]['content']
    assert json.loads(wire.split('\n', 1)[1])['rows'] == [{'n': None}]
    assert 'SQLite formatting function returned NULL' in wire


def test_search_and_history_are_newest_first_with_one_optional_detail_line(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    for i in range(3):
        call(registry, 'create', **declaration(text=f'plan {i}'))
    lines = executor.execute('pf_search', {'object': {'kind': 'plan', 'id': 'B'}, 'limit': 2, 'all': True}).splitlines()
    assert lines[0] == 'pf search B: 3 saved, newest first; showing 1–2.'
    assert '"plan 2"' in lines[1] and '"plan 1"' in lines[2] and '1 more older: pf more c1.' in lines[3]
    detail = executor.execute('pf_search', {'object': {'kind': 'plan', 'id': 'B'}, 'detail': True, 'all': True}).splitlines()
    assert len(detail) == 1 + 3 * 2 + 1 and detail[2].startswith('    B declared')
    assert executor.execute('pf_read', {'target': {'record': 'r1'}, 'detail': True}).startswith('Error:')


def test_saved_dependency_keeps_detail_hint(workspace, tmp_path, server):
    _, _, executor = chain(workspace, tmp_path, server)
    result = executor.execute('bash', {'command': 'pf depend r2'})
    cursor = re.search(r'Complete reasons[^\n]+: pf more (c\d+)', result)
    assert cursor, result
    detail = relation_details(executor, 'pf more ' + cursor[1])
    row, = detail['items']
    assert row['source']['ref']['cite'] == 'r1.1' and row['target']['ref']['cite'] == 'r2.1'
    assert row['reason'] == 'Initial observation'
    assert detail['scope']['depth'] == 1
    assert 'See what changed' not in result
