"""Stale checks through real captured evidence, public reads, and PF tool dispatch."""
from contextlib import closing
import json
import os
from pathlib import Path
import random
import re

import pytest

from saas_bench.api_server import NovaMindAPIServer
from saas_bench.execution_capture import finish_http
from saas_bench.pf_queries import PFQueries
from saas_bench.pf_refresh import refresh, READ_TOOLS
from saas_bench.public_sql import PUBLIC_POLICY_VERSION, execute_query
from saas_bench.sql_evidence import encoded
from test_text_registry import workspace, captured, call, declaration, send
from test_pf_queries import query, read
from test_public_sql import server
from test_preflight_integration import offline_runner, packed_public, business_state, advance


def record_sql(store, sql, body, *, parent=None, execution=None):
    event = store.begin(encoded({'sql': sql}), PUBLIC_POLICY_VERSION, parent)
    store.finish(event, 200, '', encoded(body), execution or {})
    store.delivered(event, 'internal')
    return event + ':public_response'


def cite(store, registry, executor, version, **fields):
    handle = registry.resolver.handle(version)
    send(store, executor.execute('pf_read', {'target': {'version': handle}}))
    return call(registry, 'create', **declaration(references=[dict(
        evidence={'version': handle}, purpose='current', **fields)]))


def forward(executor, record='r2', **args):
    return query(executor, 'pf_dependencies', target={'record': record}, purpose='current', **args)


def chain(workspace, tmp_path, server, sql='SELECT amount FROM ledger', **fields):
    store, registry, executor = captured(workspace, tmp_path)
    server.sql_evidence = store
    executor.pf_queries.refresh = lambda versions, parent: refresh(server, versions, parent)
    version = record_sql(store, sql, execute_query(server, sql))
    cite(store, registry, executor, version, **fields)
    send(store, executor.execute('text_list', {}))
    call(registry, 'create', **declaration({'record': 'r1'}))
    return store, registry, executor


def test_cutoff_failure_and_same_day_refresh(workspace, tmp_path, server):
    store, registry, executor = chain(workspace, tmp_path, server, select={'col': 'amount'},
        predicate={'type': 'threshold', 'op': '>=', 'value': 40}, note='Keep at least forty')
    results = []
    for amount, holds in ((43, True), (39, False), (42, True)):
        server.conn.execute('UPDATE ledger SET amount=?', (amount,))
        server.conn.commit()
        before = server.conn.serialize()
        result = forward(executor)
        results.append(result)
        assert server.conn.serialize() == before
        upstream, leaf = result['items']
        assert leaf['check']['predicate_result'] == ('holds' if holds else 'fails')
        assert leaf['check']['notice'] == ('strong_dependency_version_changed' if amount == 43 else
                                          'predicate_failed' if not holds else None)
        assert upstream['check']['version_changed'] is False
        assert upstream['check']['affected'] is not holds
        assert upstream['check']['affected_paths'] == ([] if holds else [leaf['path']])
        assert leaf['note'] == 'Keep at least forty'
    versions = [result['items'][1]['check']['current_version'] for result in results]
    assert len(set(versions)) == 3
    if destination := os.environ.get('CEOBENCH_STALE_ARTIFACTS'):
        folder = Path(destination)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'cutoff-results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
        store.snapshot(folder / 'cutoff-evidence.sqlite')


@pytest.mark.parametrize('mutation, reason', [
    ('UPDATE ledger SET amount=-1', 'selected_value_null'),
    ('DELETE FROM ledger', 'no rows'),
    ("INSERT INTO ledger(day,category,amount) VALUES(0,'operations',43)", 'multiple rows'),
    ("UPDATE ledger SET amount='unknown'", 'not_numeric'),
])
def test_uncheckable_values_propagate_exact_path(workspace, tmp_path, server, mutation, reason):
    store, registry, executor = chain(workspace, tmp_path, server,
        sql='SELECT category,NULLIF(amount,-1) AS amount FROM ledger', select={'row': {'category': 'operations'}, 'col': 'amount'},
        predicate={'type': 'tolerance', 'amount': 1})
    server.conn.execute(mutation)
    server.conn.commit()
    first, leaf = forward(executor)['items']
    assert leaf['check']['predicate_result'] == 'cannot_check'
    assert reason in leaf['check']['reason']
    assert first['check']['affected_paths'] == [leaf['path']]


def test_two_cells_tolerance_and_default_equality(workspace, tmp_path, server):
    store, registry, executor = chain(workspace, tmp_path, server,
        sql="SELECT 'ours' AS kind,NULLIF(amount,-1) AS n FROM ledger UNION ALL SELECT 'theirs',40",
        predicate=dict(type='compare', left={'row': {'kind': 'ours'}, 'col': 'n'}, op='>',
                       right={'row': {'kind': 'theirs'}, 'col': 'n'}))
    for value, result in ((44, 'holds'), (39, 'fails'), (-1, 'cannot_check')):
        server.conn.execute('UPDATE ledger SET amount=?', (value,))
        server.conn.commit()
        assert forward(executor)['items'][1]['check']['predicate_result'] == result
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration(references=[dict(evidence={'path': 'evidence.json'}, purpose='current',
        select={'path': '/n'}, predicate={'type': 'tolerance', 'amount': .5})]))
    for value, result in ((7.5, 'holds'), (7.5001, 'fails'), (7, 'holds')):
        (workspace / 'evidence.json').write_text(json.dumps({'n': value}))
        assert forward(executor, 'r3')['items'][0]['check']['predicate_result'] == result
    call(registry, 'create', **declaration({'path': 'evidence.json'}))
    (workspace / 'evidence.json').write_text('{"n":8}')
    assert forward(executor, 'r4')['items'][0]['check']['notice'] == 'strong_dependency_version_changed'
    (workspace / 'evidence.json').write_text('{"n":7}')
    assert forward(executor, 'r4')['items'][0]['check']['version_changed'] is False


def test_csv_numeric_selection_and_text_revision(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    (workspace / 'cost.csv').write_text('key,n\nB,7\n')
    send(store, executor.execute('read_file', {'path': 'cost.csv'}))
    call(registry, 'create', **declaration(references=[dict(evidence={'path': 'cost.csv'}, purpose='current',
        select={'row': {'key': 'B'}, 'col': 'n'}, predicate={'type': 'threshold', 'op': '<', 'value': 10})]))
    (workspace / 'cost.csv').write_text('key,n\nB,9\n')
    assert forward(executor, 'r1')['items'][0]['check']['predicate_result'] == 'holds'
    send(store, executor.execute('text_list', {}))
    call(registry, 'create', **declaration({'record': 'r1'}))
    call(registry, 'revise', record='r1', reason='Same text, new revision')
    assert forward(executor)['items'][0]['check']['version_changed'] is True
    call(registry, 'retire', record='r1', reason='Stop using')
    assert forward(executor)['items'][0]['check']['version_changed'] is True
    (workspace / 'cost.csv').unlink()
    assert forward(executor)['items'][1]['check']['reason'] == 'file_missing'


def test_pagination_freezes_checks_and_ablation_history_do_not_refresh(workspace, tmp_path, server):
    store, registry, executor = chain(workspace, tmp_path, server)
    registry.sim_day = lambda: 63
    first = forward(executor, limit=1)
    registry.sim_day = lambda: 70
    server.conn.execute('UPDATE ledger SET amount=1')
    server.conn.commit()
    executor.pf_queries = PFQueries(registry, refresh=lambda *_: pytest.fail('Pagination refreshed'))
    page = query(executor, 'pf_dependencies', cursor=first['next_cursor'])
    assert page['checked_day'] == page['items'][0]['check']['checked_day'] == 63
    assert page['items'][0]['check']['version_changed'] is False
    assert query(executor, 'pf_dependencies', cursor=first['next_cursor']) == page
    query(executor, 'pf_dependencies', target={'record': 'r2'})
    executor.execute('text_list', {})
    query(executor, 'pf_dependents', target={'record': 'r1'})
    read(executor, target={'record': 'r1'})
    executor.pf_queries.stale_checks = False
    assert forward(executor)['stale_check'] == 'not_performed'
    executor.pf_queries.stale_checks = True
    executor.pf_queries.refresh = lambda versions, parent: refresh(server, versions, parent)
    assert forward(executor)['items'][1]['check']['version_changed'] is True


def test_derived_file_reports_upstream_query_without_rerunning_script(workspace, tmp_path, server):
    store, registry, executor = captured(workspace, tmp_path)
    server.sql_evidence = store
    executor.pf_queries.refresh = lambda versions, parent: refresh(server, versions, parent)
    dashboard = store.begin_event('dashboard_generation', {'day': 56})
    store.complete(dashboard)
    registry.sim_day = lambda: 63
    script = store.begin_event('registered_script_execution', {'name': 'derive', 'day': 56})
    parent = store.begin_event('bash', {'command': 'python derive.py'}, parent=script)
    store.version(parent, 'code', 'raise RuntimeError("must never replay")', layer='executed_code')
    record_sql(store, 'SELECT amount FROM ledger', execute_query(server, 'SELECT amount FROM ledger'), parent=parent)
    record_sql(store, 'SELECT COUNT(*) AS n FROM ledger', execute_query(server, 'SELECT COUNT(*) AS n FROM ledger'), parent=parent)
    (workspace / 'derived.json').write_text('{"n":42}')
    version = store.version(parent, 'file', '{"n":42}', layer='file_bytes', object_id='derived.json')
    store.version(parent, 'workspace_after', encoded({'derived.json': {'version': version}}), layer='workspace_boundary')
    store.complete(parent, changed_paths=['derived.json'])
    store.complete(script)
    # The agent reads the derived file itself; that observation shares bytes with the Bash output.
    send(store, executor.execute('read_file', {'path': 'derived.json'}))
    call(registry, 'create', **declaration(references=[dict(
        evidence={'path': 'derived.json'}, purpose='current', select={'path': '/n'},
        predicate={'type': 'threshold', 'op': '>', 'value': 40})]))
    server.conn.execute('UPDATE ledger SET amount=1')
    server.conn.commit()
    # Default arguments: current-purpose tracing reruns the derived file's upstream SQL.
    rows = forward(executor, 'r1')['items']
    assert rows[0]['check']['predicate_result'] == 'holds' and rows[0]['check']['version_changed'] is False
    sql_rows = [row for row in rows if row['target'].get('sql')]
    assert len(sql_rows) == 2
    assert sum(row['check']['version_changed'] for row in sql_rows) == 1
    assert any(row['check']['notice'] == 'version_changed' for row in sql_rows)
    assert rows[0]['check']['affected_paths']
    compact = executor.execute('pf_dependencies', {'target': {'record': 'r1'}})
    assert '1 underlying source changed:' in compact
    assert 'of 1 underlying' not in compact
    assert 'stored artifact' in compact and 'not recomputed' in compact
    wire = send(store, compact)
    assert '1 underlying source changed:' in json.loads(store.get_content(wire + ':wire')[1])['messages'][0]['content']
    assert rows[0]['check']['evaluation_scope'] == 'stored_artifact'
    assert rows[0]['check']['checked_day'] == 63
    assert rows[0]['check']['cited_observed_day'] == 56
    assert 'stored bytes 42 (cited 42)' in compact
    assert 'current-world metric unverified' in compact
    low = forward(executor, 'r1', depth=1)
    assert low['unexpanded'] and not low['items'][0]['check']['affected']
    assert 'depth limit' in executor.execute('pf_dependencies', {'target': {'record': 'r1'}, 'depth': 1})

    send(store, executor.execute('text_list', {}))
    for _ in range(2):
        call(registry, 'create', **declaration({'record': 'r1'}))
    send(store, executor.execute('text_list', {}))
    call(registry, 'create', **declaration(references=[
        dict(evidence={'record': r}, purpose='current') for r in ('r2', 'r3')]))
    shared = forward(executor, 'r4', detail=False, depth=8)
    assert len(shared['items']) == 2
    assert all(r['check']['below_content_changed'] == 1 for r in shared['items'])
    assert all('1 underlying source changed:' in r['check']['sources'] for r in shared['items'])

    first = executor.weekly_check(63)
    assert '42→1' in first and 'not recomputed by PF' in first
    assert 'Only added rows' not in first
    server.conn.execute('UPDATE ledger SET amount=2')
    server.conn.commit()
    registry.sim_day = lambda: 70
    second = executor.weekly_check(70)
    assert '42→2' in second and 'checked day 70' in second
    pending = store.load_state('pf_review')['pending']
    assert all(p['last_day'] == 63 for p in pending.values())
    assert all(any(r['check'].get('checked_day') == 70 and r['check'].get('evaluation_scope') == 'refreshed_read'
                   for r in p['conditions']) for p in pending.values())
    assert (workspace / 'derived.json').read_text() == '{"n":42}'
    (workspace / 'derived.json').write_text('{"n":264}')
    changed = forward(executor, 'r1')['items'][0]['check']
    assert changed['cited_observed_day'] == 56 and changed['checked_day'] == 70
    assert changed['summary'] == '> 40: stored bytes 264 (cited 42)'
    from saas_bench.pf_stale import StaleCheck
    legacy = store.load_state(executor.pf_queries.check_state)
    for row in legacy:
        row['check'].pop('evaluation_scope')
    StaleCheck(executor.pf_queries)._propagate(legacy)
    assert legacy[0]['check']['affected_paths']
    (workspace / 'derived.json').write_text('{"n":1}')
    failed = executor.execute('pf_dependencies', {'target': {'record': 'r1'}})
    assert 'PREDICATE FAILS' in failed and 'underlying source changed:' in failed


def test_dedup_shared_snapshot_historical_path_and_original_sql(workspace, tmp_path, server):
    store, registry, executor = chain(workspace, tmp_path, server, sql='SELECT amount FROM ledger WHERE day=0')
    source = store.load_state('declaration:r1.1')['references'][0]['version_id']
    handle = registry.resolver.handle(source)
    call(registry, 'create', **declaration(references=[
        dict(evidence={'version': handle}, purpose='current'),
        dict(evidence={'record': 'r1'}, purpose='current'),
        dict(evidence={'version': handle}, purpose='historical_only')]))
    other = record_sql(store, 'SELECT count(*) AS n FROM ledger', execute_query(server, 'SELECT count(*) AS n FROM ledger'))
    cite(store, registry, executor, other)
    send(store, executor.execute('text_list', {}))
    call(registry, 'create', **declaration(references=[dict(evidence={'record': r}, purpose='current') for r in ('r3', 'r4')]))
    with closing(store.connect()) as conn:
        count = conn.execute('SELECT count(*) FROM queries').fetchone()[0]
    rows = forward(executor, 'r5', depth=4)['items']
    assert any(row['check']['reason'] == 'historical_only' for row in rows)
    with closing(store.connect()) as conn:
        records = [json.loads(r[0]) for r in conn.execute('SELECT record FROM results')]
        assert conn.execute('SELECT count(*) FROM queries').fetchone()[0] == count
    refreshed = [r for r in records if r.get('refresh_of')]
    assert len(refreshed) == 2
    assert len({r['snapshot_ref'] for r in refreshed}) == 1
    assert len({r['day'] for r in refreshed}) == 1


def test_all_sdk_reads_preserve_world_and_all_random_streams(workspace, tmp_path, make_initialized_sim, make_agent_tools):
    from saas_bench.config import ScenarioPack
    from saas_bench.shocks import ShockManager
    conn, sim, config = make_initialized_sim()
    tools = make_agent_tools(conn, config)
    sim.shock_manager = ShockManager(conn, sim.rng, ScenarioPack(name='test', description='test'))
    api = NovaMindAPIServer(tools, simulator=sim, conn=conn)
    store, registry, executor = captured(workspace, tmp_path)
    api.sql_evidence = store
    executor.pf_queries.refresh = lambda versions, parent: refresh(api, versions, parent)
    streams = [tools.rng, sim.rng, sim._macro_rng, sim._competitor_rng, sim._competitor_post_noise_rng,
               sim._competitor_template_rng, sim._quality_rng, sim._customer_quality_noise_rng,
               sim._customer_pick_rng, sim.shock_manager.rng, *sim._group_rngs.values()]
    sim.save_rng_states()
    before, states, python_rng = conn.serialize(), encoded([r.bit_generator.state for r in streams]), random.getstate()
    try:
        for index, tool in enumerate([*sorted(READ_TOOLS), 'current_day'], 1):
            args = {'group_id': 'S1'} if tool == 'get_group_insights' else {}
            request = (dict(method='GET', path='/vars', parsed={}) if tool == 'current_day' else
                       dict(method='POST', path='/call', parsed=dict(tool=tool, args=args)))
            body = {'current_day': tools.current_day} if tool == 'current_day' else api.execute_tool(tool, args).to_json()
            event = store.begin_event('public_http', request)
            finish_http(store, event, 200, '', json.dumps(body).encode(), {})
            cite(store, registry, executor, event + ':public_response')
            result = forward(executor, 'r' + str(index))['items'][0]['check']
            assert result['version_changed'] is False and result['reason'] is None
            assert conn.serialize() == before
            assert encoded([r.bit_generator.state for r in streams]) == states
            assert random.getstate() == python_rng
            assert tools.current_day == sim.current_day == 0
    finally:
        conn.close()


def test_refresh_permission_rejection_timeout_and_lossy_results(workspace, tmp_path, server):
    store, registry, executor = chain(workspace, tmp_path, server)
    for index, (sql, reason) in enumerate([
        ('SELECT * FROM group_insight_snapshots', 'refresh_failed:rejected'),
        ('SELECT actual_completion_day AS done_on FROM main.research_projects', 'refresh_failed:rejected'),
        ('WITH c AS (SELECT 1) UPDATE main.ledger SET amount=99', 'refresh_failed:rejected'),
        ('SELECT 1 AS n,2 AS n', 'duplicate_columns'),
        ('WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x WHERE n<5001) SELECT n FROM x', 'source_truncated'),
        ("SELECT X'41' AS n", 'blob_coercion'),
    ], 3):
        # A formerly captured view may no longer pass today's public policy.
        version = record_sql(store, sql, dict(success=True, columns=['n'], rows=[{'n': 1}], row_count=1))
        cite(store, registry, executor, version, select={'col': 'n'}, predicate={'type': 'threshold', 'op': '>', 'value': 0})
        before = server.conn.serialize()
        check = forward(executor, 'r' + str(index))['items'][0]['check']
        assert check['notice'] == 'cannot_check' and check['reason'] == reason
        assert server.conn.serialize() == before
    server.QUERY_TIMEOUT_SECONDS = .005
    sql = 'WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x) SELECT sum(n) AS n FROM x'
    version = record_sql(store, sql, dict(success=True, columns=['n'], rows=[{'n': 1}], row_count=1))
    record = cite(store, registry, executor, version)
    assert forward(executor, record['id'])['items'][0]['check']['reason'] == 'refresh_failed:timed_out'
    assert not store.fault


def test_equal_rows_preserve_types_duplicates_and_empty_results(workspace, tmp_path, server):
    store, registry, executor = captured(workspace, tmp_path)
    bodies = [dict(success=True, columns=['n'], rows=rows, row_count=len(rows)) for rows in
              ([{'n': 1}, {'n': 2}, {'n': 1}], [{'n': 2}, {'n': 1}, {'n': 1}],
               [{'n': 1}, {'n': 2}], [{'n': 1.0}, {'n': 2}, {'n': 1}], [])]
    version = record_sql(store, 'SELECT n', bodies[0])
    cite(store, registry, executor, version)
    for body, changed in zip(bodies, (False, False, True, True, True)):
        executor.pf_queries.refresh = lambda versions, parent: {
            v: record_sql(store, 'SELECT n', body, parent=parent) for v in versions}
        assert forward(executor, 'r1')['items'][0]['check']['version_changed'] is changed
    cite(store, registry, executor, record_sql(store, 'SELECT n', bodies[-1]))
    assert forward(executor)['items'][0]['check']['version_changed'] is False


def test_missing_snapshot_and_missing_reference_are_not_unchanged(workspace, tmp_path, server):
    store, registry, executor = chain(workspace, tmp_path, server)
    server._operation_failed = True
    rows = forward(executor)['items']
    assert rows[1]['check']['notice'] == 'cannot_check' and rows[1]['check']['current_version']
    assert rows[0]['check']['affected_paths'] == [rows[1]['path']]
    store.assert_healthy()
    call(registry, 'create', **declaration())
    assert forward(executor, 'r3')['items'][0]['check']['reason'] == 'No saved evidence'


def test_file_capture_failure_stops_collection(workspace, tmp_path, monkeypatch):
    store, registry, executor = captured(workspace, tmp_path)
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration({'path': 'evidence.json'}))
    original = store.version
    def fail_file(event, slot, *args, **kwargs):
        if slot == 'file':
            raise OSError('injected capture failure')
        return original(event, slot, *args, **kwargs)
    monkeypatch.setattr(store, 'version', fail_file)
    result = executor.execute('pf_dependencies', {'target': {'record': 'r1'}})
    assert result.startswith('Error:')
    with pytest.raises(RuntimeError, match='Unconfirmed'):
        store.assert_healthy()


def test_receipts_are_neutral_summaries_use_captured_versions_and_files_stay_in_workspace(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    executor.pf_queries.refresh = lambda *_: pytest.fail('Unexpected replay')
    event = store.begin_event('public_http', dict(method='POST', path='/call',
        parsed=dict(tool='set_prices', args={'B': 99})))
    finish_http(store, event, 200, '', encoded(dict(success=True, data={'B': 99})), {})
    cite(store, registry, executor, event + ':public_response')
    check = forward(executor, 'r1')['items'][0]['check']
    assert check['reason'] == 'write_receipt' and check['notice'] is None and not check['affected']
    for value in ('old summary', 'new summary'):
        event = store.begin_event('dashboard_generation')
        version = store.version(event, 'dashboard', value, layer='dashboard')
        store.complete(event)
        if value == 'old summary':
            cite(store, registry, executor, version)
    assert forward(executor)['items'][0]['check']['version_changed'] is True
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration({'path': 'evidence.json'}))
    (workspace / 'evidence.json').unlink()
    secret = tmp_path / 'private-secret'
    secret.write_text('NEVER_CAPTURE_PRIVATE_SECRET')
    (workspace / 'evidence.json').symlink_to(secret)
    assert forward(executor, 'r3')['items'][0]['check']['reason'] == 'file_unavailable'
    with closing(store.connect()) as conn:
        assert not conn.execute("SELECT 1 FROM blobs WHERE instr(content,?)>0", (b'NEVER_CAPTURE_PRIVATE_SECRET',)).fetchone()


def test_current_false_condition_fails_even_when_baseline_was_false(workspace, tmp_path, server):
    _, _, executor = chain(workspace, tmp_path, server, select={'col': 'amount'},
        predicate=dict(type='threshold', op='>=', value=770))
    first, leaf = forward(executor)['items']
    assert leaf['check']['version_changed'] is False
    assert leaf['check']['predicate_result'] == 'fails'
    assert first['check']['affected_paths'] == [leaf['path']]


def test_packed_refresh_restore_ablation_and_private_endpoint(offline_runner, tmp_path):
    from saas_bench.agents.bash_agent.run_test import BashAgentRunner
    from saas_bench.run_state import clone_sql_run
    prefix = offline_runner(text_registration='prefix')
    prefix.agent.current_day = 0
    prefix._execute_tool('bash', {'command': './novamind-operation query "SELECT COUNT(*) AS n FROM ledger" > query.json'})
    assert advance(prefix)['success']
    prefix._save_checkpoint(7)
    prefix._stop_server()
    for mode, enabled in (('pf', True), ('pf', False), ('git', None)):
        branch = 'pf-on' if enabled else 'pf-off' if mode == 'pf' else 'git'
        child = offline_runner(clone_sql_run(prefix.workspace_dir, tmp_path / branch, branch,
                               text_registration=mode, pf_stale_checks=enabled))
        child.agent.current_day = 7
        assert child.pf_stale_checks is bool(enabled)
        probe = ("import os,urllib.request,urllib.error\n"
                 "try: urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:'+os.environ['NOVAMIND_API_PORT']+'/pf-refresh',data=b'{}'))\n"
                 "except urllib.error.HTTPError as e: print(e.code)\n")
        child._execute_tool('write_file', dict(path='probe.py', content=probe))
        assert child._execute_tool('bash', dict(command='python -S probe.py')).strip() == '403'
        child._save_checkpoint(7)
        before = business_state(child)
        args = dict(target={'path': 'query.json'}, include_execution=True, depth=4, limit=1, detail=True)
        output = child._execute_tool('pf_dependencies', args)
        if mode == 'git':
            assert output.startswith('Error: Unknown tool')
            continue
        assert ('Checked day 7; within expanded dependencies; scopes below describe what was evaluated' in output) is enabled, output
        # The packed host runtime must contain the refresh module (it was once left out).
        assert 'refresh_failed' not in output and 'refresh_unavailable' not in output, output
        child._save_checkpoint(7)
        assert business_state(child) == before
        cursor = (re.search(r'\{"cursor": "(c\d+)"\}', output) or [None, None])[1]
        child._stop_server()
        restored = offline_runner(child.workspace_dir)
        assert restored.pf_stale_checks is enabled
        if cursor:
            restored.tool_executor.pf_queries.refresh = lambda *_: pytest.fail('Restored page reran reads')
            assert not restored._execute_tool('pf_dependencies', {'cursor': cursor}).startswith('Error:')
        with pytest.raises(ValueError, match='stale check configuration mismatch'):
            BashAgentRunner(continue_from=child.workspace_dir, pf_stale_checks=not enabled)


@pytest.mark.parametrize('sql,status', [
    ('SELECT * FROM group_insight_snapshots', 403),
    ('WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x) SELECT sum(n) FROM x', 504),
])
def test_mixed_refresh_keeps_reads_independent_within_total_budget(workspace, tmp_path, server, sql, status):
    from saas_bench.tools import AgentTools
    store, _, _ = captured(workspace, tmp_path)
    server.sql_evidence = store
    server.tools = AgentTools(server.conn, 0, server.script_workspace)
    sdk_body = server.execute_tool('list_research_projects', {}).to_json()
    event = store.begin_event('public_http', dict(method='POST', path='/call',
        parsed=dict(tool='list_research_projects', args={})))
    finish_http(store, event, 200, '', encoded(sdk_body), {})
    fast_body = execute_query(server, 'SELECT amount FROM ledger')
    versions = [record_sql(store, sql, {}), record_sql(store, 'SELECT amount FROM ledger', fast_body),
                event + ':public_response']
    parent = store.begin_event('pf_dependencies', {})
    server.QUERY_TIMEOUT_SECONDS = .05
    before = server.conn.serialize()
    refreshed = refresh(server, versions, parent)
    records, bodies = [], []
    for version in versions:
        meta, raw = store.get_content(refreshed[version])
        records.append(store.read_event(meta['created_by_event'])['result'])
        bodies.append(json.loads(raw))
    if status == 504:
        assert [record['http_status'] for record in records] == [504, 504, 504]
        assert [record['attempted'] for record in records] == [True, False, False]
        assert bodies[1:] == [dict(success=False, error='refresh_timed_out')] * 2
    else:
        assert [record['http_status'] for record in records] == [403, 200, 200]
        assert bodies[1:] == [fast_body, sdk_body]
    assert len({record['snapshot_ref'] for record in records}) == 1
    assert {record['day'] for record in records} == {0}
    assert all(record['snapshot_bytes'] == 0 and not record['snapshot_reused'] for record in records[:2])
    assert server.conn.serialize() == before
    assert server.execute_tool('list_research_projects', {}).to_json() == sdk_body


def test_stored_scope_unknown_dates_unchanged_weekly_and_legacy_cursor(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    registry.sim_day = lambda: None
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration(references=[dict(
        evidence={'path': 'evidence.json'}, purpose='current', select={'path': '/n'},
        predicate={'type': 'threshold', 'op': '>', 'value': 0})]))
    send(store, executor.execute('text_list', {}))
    call(registry, 'create', **declaration({'record': 'r1'}))
    first = forward(executor, limit=1)
    assert first['checked_day'] is None
    page = query(executor, 'pf_dependencies', cursor=first['next_cursor'])
    assert page['items'][0]['check']['cited_observed_day'] is None
    assert page['items'][0]['check']['checked_day'] is None
    legacy = store.load_state('pf_cursors')[first['next_cursor']]['check_state']
    rows = store.load_state(legacy)
    for row in rows:
        row['check'].pop('checked_day')
        row['check'].pop('evaluation_scope')
    store.save_state(legacy, rows)
    registry.sim_day = lambda: 70
    text = executor.execute('pf_dependencies', {'cursor': first['next_cursor']})
    assert 'Checked day unknown' in text and 'evaluation scope unknown' in text
    weekly = executor.weekly_check(70)
    assert 'not recomputed by PF' in weekly and 'current-world metrics unverified' in weekly
    assert 'cited observation day unknown; checked day 70; holds on stored bytes' in weekly


def test_refreshed_api_predicate_prunes_but_unavailable_refresh_preserves_source(workspace, tmp_path, server):
    store, registry, executor = captured(workspace, tmp_path)
    server.sql_evidence = store
    sql = 'SELECT amount FROM ledger'
    source = record_sql(store, sql, execute_query(server, sql))
    event = store.begin_event('public_http', dict(method='GET', path='/vars', parsed={}))
    body = encoded({'current_day': 0})
    finish_http(store, event, 200, '', body, {})
    version = store.version(event, 'derived_api', body, layer='server_public_response', object_id='vars.json', derived_from=source)
    cite(store, registry, executor, version, select={'path': '/current_day'},
         predicate=dict(type='threshold', op='>=', value=0))
    server.conn.execute('UPDATE ledger SET amount=1')
    server.conn.commit()
    executor.pf_queries.refresh = lambda versions, parent: refresh(server, versions, parent)
    rows = forward(executor, 'r1')['items']
    assert rows[0]['check']['evaluation_scope'] == 'refreshed_read'
    assert rows[0]['check']['predicate_result'] == 'holds'
    assert rows[0]['check']['affected_paths'] == []
    assert rows[1]['check']['version_changed']
    assert 'underlying source changed:' not in executor.execute('pf_dependencies', {'target': {'record': 'r1'}})
    executor.pf_queries.refresh = None
    failed = forward(executor, 'r1')['items'][0]['check']
    assert failed['evaluation_scope'] == 'not_checked'
    assert failed['predicate_result'] == 'cannot_check' and failed['affected_paths']


def test_cited_observation_dates_follow_the_citable_version_group(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    for day, value in ((56, 7), (63, 8), (70, 7)):
        dashboard = store.begin_event('dashboard_generation', {'day': day})
        store.complete(dashboard)
        (workspace / 'evidence.json').write_text(json.dumps({'n': value}))
        send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration(references=[dict(
        evidence={'path': 'evidence.json'}, purpose='current', select={'path': '/n'},
        predicate={'type': 'threshold', 'op': '>', 'value': 0})]))
    registry.sim_day = lambda: None
    row = forward(executor, 'r1')['items'][0]
    assert row['target']['version'].endswith('@v3')
    assert row['check']['cited_observed_day'] is None
    assert row['check']['cited_context_day'] == 70
    assert row['check']['checked_day'] is None


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
    checked = forward(executor, 'r1')['items'][0]['check']
    assert checked['cited_observed_day'] == 7 and checked['checked_day'] == 14
    assert checked['computation'] == 'not_reexecuted'
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


def test_pending_manual_check_with_unknown_day_remains_renderable(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration({'path': 'evidence.json'}))
    (workspace / 'evidence.json').write_text('{"n":8}')
    executor.weekly_check(63)
    registry.sim_day = lambda: None
    forward(executor, 'r1')
    pending = next(iter(store.load_state('pf_review')['pending'].values()))
    assert pending['last_day'] is None
    assert 'last verified day unknown' in executor.weekly_check(70)
