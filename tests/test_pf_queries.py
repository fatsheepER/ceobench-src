"""Historical PF queries against real capture, model delivery, and fork boundaries."""
from contextlib import closing
import json
import os
from pathlib import Path
import re

import pytest

from saas_bench.pf_queries import MODELS, PFQueries
from saas_bench.registration_evidence import HANDLES
from saas_bench.sql_evidence import SQLEvidenceStore, encoded
from test_sql_evidence import identity, request, settled
from test_text_registry import workspace, captured, call, declaration, receipt, send
from test_public_sql import server
from test_preflight_integration import offline_runner, packed_public, advance


def query(executor, name, **args):
    """Structured page behind a PF tool; dependency queries default to every traced row."""
    if name in ('pf_dependencies', 'pf_dependents') and 'cursor' not in args:
        args.setdefault('detail', True)
        if name == 'pf_dependencies':
            args.setdefault('purpose', 'historical_only')
    return executor.pf_queries.answer(name, args)


def read(executor, **args):
    text = executor.execute('pf_read', args)
    assert not text.startswith('Error:'), text
    head, body = text.split('\n', 1)
    return json.loads(head), body, text


def test_two_layers_revisions_unknown_history_and_reverse_paths(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    ref = dict(evidence={'path': 'evidence.json'}, purpose='current',
               select={'path': '/n'}, predicate={'type': 'threshold', 'op': '>=', 'value': 5}, note='retain baseline')
    call(registry, 'create', **declaration(references=[ref]))
    send(store, executor.execute('text_list', {}))
    call(registry, 'create', **declaration({'record': 'r1.1'}, text='Forecast'))
    call(registry, 'create', **declaration(references=[dict(evidence={'record': 'r1.1'}, purpose='historical_only')]))
    call(registry, 'create', **declaration())
    call(registry, 'revise', record='r1', text='Corrected assumption', reason='Changed interpretation')
    page = query(executor, 'pf_dependencies', target={'record': 'r2'})
    assert len(page['items']) == 2 and page['stale_check'] == 'not_performed'
    assert page['items'][0]['target']['record'] == 'r1.1'
    assert page['items'][1]['note'] == 'retain baseline'
    assert page['items'][1]['predicate']['value'] == 5
    assert all(i['origin'] == 'agent_declaration' for i in page['items'])
    assert page['items'][0]['target']['current_revision'] is False
    assert len(page['items'][1]['path']) == 3
    missing = query(executor, 'pf_dependencies', target={'record': 'r4'})['items'][0]
    assert missing['target'] is None and missing['unexpanded'] == 'missing'
    assert missing['reason'] == 'No saved evidence'
    reverse = query(executor, 'pf_dependents', target={'record': 'r1.1'})
    assert {i['source']['record'] for i in reverse['items']} == {'r2.1', 'r3.1'}
    assert [i['historical_only'] for i in reverse['items']] == [False, True]
    evidence = registry.resolver.handle(store.load_state('declaration:r1.1')['references'][0]['version_id'])
    impact = query(executor, 'pf_dependents', target={'version': evidence})
    assert {i['source']['record'] for i in impact['items']} == {'r1.2', 'r2.1', 'r3.1'}
    assert all(len(i['path']) == 3 for i in impact['items'] if i['source']['record'] != 'r1.2')
    frontier = query(executor, 'pf_dependents', target={'version': evidence}, depth=1)['items']
    assert frontier[0]['source']['record'] == 'r1.1'
    assert frontier[0]['traversal_only'] and frontier[0]['unexpanded'] == 'depth_limit'
    call(registry, 'retire', record='r2', reason='Forecast withdrawn')
    assert len(query(executor, 'pf_dependents', target={'record': 'r1.1'})['items']) == 1
    all_refs = query(executor, 'pf_dependents', target={'record': 'r1.1'}, current_only=False)
    assert {i['source']['record'] for i in all_refs['items']} == {'r2.1', 'r2.2', 'r3.1'}
    history = query(executor, 'pf_read', mode='history', target={'record': 'r1'})
    assert [i['record'] for i in history['items']] == ['r1.2', 'r1.1']  # newest first
    assert history['items'][0]['reason'] == 'Changed interpretation'
    assert not re.search(r'\b[0-9a-f]{8,64}\b', json.dumps(page))
    assert store.identity['run_id'] not in json.dumps(page)
    # The graph is reconstructible without mutable declaration lookup state.
    before_rebuild = query(executor, 'pf_dependencies', target={'record': 'r2.1'})
    with closing(store.connect()) as conn, conn:
        conn.execute("DELETE FROM private_state WHERE name LIKE 'declaration:%'")
    assert query(executor, 'pf_dependencies', target={'record': 'r2.1'}) == before_rebuild
    if destination := os.environ.get('CEOBENCH_PF_ARTIFACTS'):
        folder = Path(destination)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'constructed-query-results.json').write_text(json.dumps(dict(
            evidence_kind='constructed_offline', dependencies=page, reverse=reverse, missing=missing,
            history=history, retired_and_historical=all_refs), ensure_ascii=False, indent=2))
        store.snapshot(folder / 'constructed-evidence.sqlite')


def test_index_is_not_delivery_and_content_read_is_exact_with_ranges(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    raw = b'{\r\n"n":7\r\n}'
    (workspace / 'evidence.json').write_bytes(raw)
    executor.execute('read_file', {'path': 'evidence.json'})
    page = executor.execute('pf_read', {'mode': 'history', 'target': {'path': 'evidence.json'}})
    send(store, page)
    with pytest.raises(ValueError, match='not been delivered'):
        call(registry, 'create', **declaration({'path': 'evidence.json'}))
    header, body, output = read(executor, target={'path': 'evidence.json'})
    assert body.encode() == raw and header['range'] == [0, len(body)]
    send(store, output)
    ref = dict(evidence={'version': header['target']['version']}, purpose='current', select={'path': '/n'})
    assert call(registry, 'create', **declaration(references=[ref]))['id'] == 'r1'
    again = query(executor, 'pf_read', mode='history', target={'path': 'evidence.json'})
    assert again['items'][0]['model_reads']['count'] == 1


def test_projection_handle_binds_its_underlying_file(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    output = executor.execute('read_file', {'path': 'evidence.json'})
    send(store, output)
    projection = next(item['version_id'] for item in output.origins
                      if store.get_content(item['version_id'])[0]['layer'] == 'file_text')
    handle = registry.resolver.handle(projection)
    call(registry, 'create', **declaration({'version': handle}))
    binding = store.load_state('declaration:r1.1')['references'][0]
    assert store.get_content(binding['version_id'])[0]['layer'] == 'file_bytes'


def test_pagination_freezes_snapshot_survives_restore_and_delivers_only_read_chunks(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    for i in range(3):
        call(registry, 'create', **declaration(text=f'plan {i}'))
    page = query(executor, 'pf_search', object={'kind': 'plan', 'id': 'B'}, limit=1, all=True)
    first_cursor = page['next_cursor']
    assert page['remaining'] == 2
    call(registry, 'create', **declaration(text='new plan'))
    executor.pf_queries = PFQueries(registry)
    page2 = query(executor, 'pf_search', cursor=first_cursor)
    assert page2['items'][0]['record'] == 'r2.1' and page2['remaining'] == 1
    assert query(executor, 'pf_search', cursor=first_cursor) == page2
    last = query(executor, 'pf_search', cursor=page2['next_cursor'])
    assert last['items'][0]['record'] == 'r1.1' and last['next_cursor'] is None
    assert len(query(executor, 'pf_search', object={'kind': 'plan', 'id': 'B'}, all=True)['items']) == 4
    (workspace / 'long.txt').write_text('a' * 30000 + 'tail🙂')
    executor.execute('read_file', {'path': 'long.txt', 'limit': 1})
    head, chunk, output = read(executor, target={'path': 'long.txt'})
    send(store, output)
    first, more = chunk.rsplit('\n', 1)
    assert len(first) == 24000 and head['truncated_reason'] == 'character_limit'
    assert more == f"[6,005 more characters: pf more {head['next_cursor']}]"
    call(registry, 'create', **declaration({'path': 'long.txt'}))
    head2, chunk2, output2 = read(executor, cursor=head['next_cursor'])
    send(store, output + output2)  # Plain string concatenation intentionally loses source mappings.
    with pytest.raises(ValueError, match='this request'):
        call(registry, 'create', **declaration({'path': 'long.txt'}))
    call(registry, 'create', **declaration({'version': 'long.txt@v1'}))
    from saas_bench.execution_capture import model_request, text_sources
    body = dict(messages=[dict(role='tool', content=output), dict(role='tool', content=output2)])
    event = model_request(store, json.dumps(body).encode(), text_sources(body), 'call', 'attempt', 'week')
    store.complete(event, send_state='response_received')
    assert call(registry, 'create', **declaration({'path': 'long.txt'}))['id'] == 'r7'
    assert first + chunk2 == (workspace / 'long.txt').read_text()
    assert head2['next_cursor'] is None


def test_diff_direction_newline_and_no_target_delivery(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    executor.execute('read_file', {'path': 'evidence.json'})
    old = query(executor, 'pf_read', mode='history', target={'path': 'evidence.json'})['items'][-1]['version']
    (workspace / 'evidence.json').write_text('{"n":8}\n')
    executor.execute('read_file', {'path': 'evidence.json'})
    header, diff, output = read(executor, mode='diff', baseline={'version': old}, target={'path': 'evidence.json'})
    assert header['direction'] == 'baseline_to_target' and header['raw_equal'] is False
    assert '-{"n":7}' in diff and '+{"n":8}' in diff and 'No newline at end of file' in diff
    send(store, output)
    with pytest.raises(ValueError, match='not been delivered'):
        call(registry, 'create', **declaration({'path': 'evidence.json'}))
    (workspace / 'other.json').write_text('{"n":8}')
    executor.execute('read_file', {'path': 'other.json'})
    assert 'same captured object' in executor.execute('pf_read', dict(mode='diff', baseline={'version': old}, target={'path': 'other.json'}))


def test_sql_diff_preserves_duplicates_types_order_and_cross_branch_identity(workspace, tmp_path, server):
    store, registry, executor = captured(workspace, tmp_path)
    server.sql_evidence = store
    server.start()
    sql = 'SELECT amount FROM ledger'
    request(server, sql)
    settled(server)
    original = query(executor, 'pf_read', mode='history', target={'sql': sql})['items'][0]['version']
    server.conn.execute('UPDATE ledger SET amount=amount+1')
    server.conn.commit()
    request(server, sql)
    settled(server)
    head, diff, _ = read(executor, mode='diff', baseline={'version': original}, target={'sql': sql})
    assert head['rows_equal'] is False and diff
    with closing(store.connect()) as conn:
        cutoff = store.sequence(conn)
    fork = SQLEvidenceStore(store.path, identity('pf', parent_branch='prefix', fork_seq=cutoff, capture_scope='execution'))
    from saas_bench.text_registry import TextRegistry
    branch = PFQueries(TextRegistry(workspace, 'pf', fork))
    server.sql_evidence = fork
    request(server, sql)
    settled(server)
    history = branch.answer('pf_read', dict(target={'sql': sql}, mode='history'))
    assert len(history['items']) == 3
    assert all(i['sql'] == sql for i in history['items'])
    old_handle = history['items'][-1]['version']  # oldest: the prefix result
    send(fork, branch.execute('pf_read', dict(target={'version': old_handle})))
    bound = branch.resolver.resolve({'version': old_handle}, {})
    assert bound['version_id'].startswith('test-run/prefix/')
    assert bound['latest_version_id'].startswith('test-run/pf/')
    # Construct public wire responses with a fixed query definition to control order/types.
    for rows in ([{'x': 1}, {'x': 2}, {'x': 1}], [{'x': 2}, {'x': 1}, {'x': 1}], [{'x': 1}, {'x': 2}]):
        event = store.begin(encoded({'sql': 'SELECT x'}), 'test')
        store.finish(event, 200, '', encoded(dict(success=True, columns=['x'], rows=rows, row_count=len(rows))), {})
        store.delivered(event, 'sent')
    hist = query(executor, 'pf_read', mode='history', target={'sql': 'SELECT x'})['items'][::-1]
    # Reordered but equal rows are the same content: one handle, marked as acquired again.
    assert hist[0]['version'] == hist[1]['version'] and hist[1]['unchanged_since_day'] == 0
    assert hist[2]['version'].endswith('@v2') and hist[2]['previous_version'] == hist[0]['version']
    head, _, _ = read(executor, mode='diff', baseline={'version': hist[0]['version']}, target={'version': hist[2]['version']})
    assert head['raw_equal'] is False and head['rows_equal'] is False


def test_public_objects_are_exact_typed_and_not_inferred_from_prose(workspace, tmp_path):
    from saas_bench.execution_capture import finish_http
    store, registry, executor = captured(workspace, tmp_path)
    event = store.begin_event('public_http', dict(method='POST', path='/call',
        parsed=dict(tool='list_research_projects', args={})))
    finish_http(store, event, 200, '', encoded(dict(success=True, data=[
        dict(project_id='t10_1', tier=10), dict(project_id='t10_2', tier=10)])), {})
    call(registry, 'create', **declaration(objects=[dict(kind='research_project', id='t10_1')], text='Mentions t10_2 in prose'))
    rows = query(executor, 'pf_search', object=dict(kind='research_project', id='t10_2'), all=True)['items']
    assert len(rows) == 1 and rows[0]['classification'] == 'read'
    assert rows[0]['objects'][0]['basis'] == 'public_response_field'
    event = store.begin(encoded({'sql': "SELECT 't10_2' AS project_id"}), 'test')
    store.finish(event, 200, '', encoded(dict(success=True, columns=['project_id'], rows=[{'project_id': 't10_2'}], row_count=1)), {})
    store.delivered(event, 'sent')
    results = query(executor, 'pf_search', object=dict(kind='research_project', id='t10_2'), all=True)['items']
    assert len(results) == 2 and results[0]['objects'][0]['basis'] == 'public_result_column'  # newest first
    assert query(executor, 'pf_search', object=dict(kind='custom', id='t10_2'), all=True)['items'] == []


def test_capture_relations_no_invented_reads_and_cycles(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    parent = store.begin_event('bash', {'command': 'python opaque.py'})
    code = store.version(parent, 'code', 'print(1)', layer='executed_code')
    child = store.begin(encoded({'sql': 'SELECT 1 AS n'}), 'test', parent=parent)
    store.finish(child, 200, '', encoded(dict(success=True, columns=['n'], rows=[{'n': 1}], row_count=1)), {})
    store.delivered(child, 'sent')
    output = store.version(parent, 'file_1_after', '{"n":1}', layer='file_bytes', object_id='derived.json')
    store.version(parent, 'workspace_after', encoded({'derived.json': dict(type='file', version=output)}), layer='workspace_boundary')
    store.complete(parent, changed_paths=['derived.json'], capture_gaps=['unobserved_internal_file_reads'])
    assert query(executor, 'pf_dependencies', target={'path': 'derived.json'})['items'] == []
    automatic = query(executor, 'pf_dependencies', target={'path': 'derived.json'}, include_execution=True)
    assert {i['target']['layer'] for i in automatic['items']} == {'executed_code', 'server_public_response'}
    assert {i['kind'] for i in automatic['items']} == {'same_execution', 'executed_by'}
    assert automatic['root']['capture_gaps'] == ['unobserved_internal_file_reads']
    event = store.begin_event('fixture')
    a = store.version(event, 'a', 'A', layer='stdout', derived_from=event + ':b')
    store.version(event, 'b', 'B', layer='stdout', derived_from=a)
    store.complete(event)
    cyclic = query(executor, 'pf_dependencies', target={'version': registry.resolver.handle(a)}, include_execution=True, depth=10)
    assert len(cyclic['items']) == 2 and cyclic['items'][-1]['unexpanded'] == 'cycle'


@pytest.mark.parametrize('mode', ['git', 'prefix'])
def test_non_pf_modes_cannot_use_any_query_or_receive_handles(workspace, tmp_path, mode):
    store, _, executor = captured(workspace, tmp_path, mode)
    for tool in MODELS:
        assert executor.execute(tool, {}).startswith('Error: Unknown tool')
    assert not executor.execute('read_file', {'path': 'evidence.json'}).endswith(']')
    assert store.load_state(HANDLES) is None


def test_private_layers_siblings_postfork_versions_and_bad_inputs_are_inaccessible(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    executor.execute('read_file', {'path': 'evidence.json'})
    with closing(store.connect()) as conn:
        cutoff = store.sequence(conn)
    sibling = SQLEvidenceStore(store.path, identity('sibling', parent_branch='prefix', fork_seq=cutoff, capture_scope='execution'))
    fork = SQLEvidenceStore(store.path, identity('pf', parent_branch='prefix', fork_seq=cutoff, capture_scope='execution'))
    for source in (store, sibling):
        event = source.begin_event('fixture')
        source.version(event, 'file', 'PRIVATE_SIBLING', layer='file_bytes', object_id='private.txt')
        source.complete(event)
    from saas_bench.text_registry import TextRegistry
    branch_registry = TextRegistry(workspace, 'pf', fork)
    branch = PFQueries(branch_registry)
    with pytest.raises(ValueError, match='No captured version'):
        branch.execute('pf_read', dict(target={'path': 'private.txt'}))
    event = fork.begin_event('fixture')
    private = fork.version(event, 'wire', 'PRIVATE_WIRE', layer='model_request_wire')
    fork.complete(event)
    handle = branch_registry.resolver.handle(private)
    with pytest.raises(ValueError, match='Unknown version handle'):
        branch.execute('pf_read', dict(target={'version': handle}))
    for args in ({'target': {'path': '../private'}}, {'target': {'version': 'v999'}},
                 {'target': {'version': 'f' * 64}}, {'target': {'sql': 'x', 'path': 'x'}},
                 {'cursor': 'c999'}, {'target': {'record': 'r1'}, 'limit': 0}):
        result = executor.execute('pf_read', args)
        assert result.startswith('Error:') and 'f' * 64 not in result
    assert branch.answer('pf_search', {'text': 'PRIVATE_'})['total'] == 0
    assert not store.fault and not fork.fault


def test_invalid_diff_does_not_allocate_unshown_handles(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    event = store.begin_event('fixture')
    store.version(event, 'a', b'A', layer='file_bytes', object_id='a.txt')
    store.version(event, 'b', b'B', layer='file_bytes', object_id='b.txt')
    store.complete(event)
    result = executor.execute('pf_read', dict(mode='diff', baseline={'path': 'a.txt'}, target={'path': 'b.txt'}))
    assert 'same captured object' in result
    assert store.load_state(HANDLES) is None


def test_empty_failed_truncated_and_binary_history_are_distinct(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    for status, body in [(200, dict(success=True, columns=['x'], rows=[], row_count=0)),
                         (403, dict(success=False, error='denied')),
                         (200, dict(success=True, columns=['x'], rows=[], row_count=0, truncated=True))]:
        event = store.begin(encoded({'sql': 'SELECT x'}), 'test')
        store.finish(event, status, '', encoded(body), {})
        store.delivered(event, 'sent')
    items = query(executor, 'pf_read', target={'sql': 'SELECT x'}, mode='history')['items']
    assert [i['status'] for i in items] == ['succeeded', 'rejected', 'succeeded']
    head, _, _ = read(executor, mode='diff', baseline={'version': items[0]['version']}, target={'version': items[2]['version']})
    assert head['rows_equal'] is True and head['rows_scope'] == 'returned_subset'
    head, _, _ = read(executor, mode='diff', baseline={'version': items[1]['version']}, target={'version': items[2]['version']})
    assert head['rows_equal'] is None
    event = store.begin_event('fixture')
    store.version(event, 'binary', b'\xff', layer='file_bytes', object_id='binary.bin')
    store.complete(event)
    assert len(query(executor, 'pf_read', target={'path': 'binary.bin'}, mode='history')['items']) == 1
    assert 'not UTF-8' in executor.execute('pf_read', {'target': {'path': 'binary.bin'}})
    assert not store.fault


def test_packed_pf_query_restore_and_group_boundary(offline_runner, tmp_path):
    from saas_bench.run_state import clone_sql_run
    prefix = offline_runner(text_registration='prefix')
    prefix.agent.current_day = 0
    prefix._execute_tool('write_file', dict(path='facts.txt', content='prefix evidence'))
    prefix._execute_tool('read_file', dict(path='facts.txt'))
    assert advance(prefix)['success']
    prefix._save_checkpoint(7)
    prefix._stop_server()
    for mode in ('git', 'pf'):
        child = offline_runner(clone_sql_run(prefix.workspace_dir, tmp_path / mode, mode, text_registration=mode))
        result = child._execute_tool('pf_read', dict(target={'path': 'facts.txt'}))
        if mode == 'git':
            assert result.startswith('Error: Unknown tool')
            assert child.evidence_store is None and not list(child.workspace_dir.rglob('sql-evidence*'))
            continue
        assert result.split('\n', 1)[1] == 'prefix evidence'
        # PF queries run as the pf command in bash; the function tools are those of Git.
        assert not any(t['name'].startswith('pf_') for t in child.agent.tool_descriptions)
        command = './novamind-operation query "SELECT COUNT(*) AS n FROM ledger" > query.json'
        output = child._execute_tool('bash', {'command': command})
        # The return and the written file are named; the queries behind them are traced, not listed.
        assert re.search(r'\n\[pf: cmd\d+@v1 \| wrote query\.json@v1\]$', output), output
        # A non-zero exit still delivered the SQL result and wrote the file.
        failed = child._execute_tool('bash', {'command': command.replace('query.json', 'failed.json') + '; exit 3'})
        assert re.search(r'\n\[pf: cmd\d+@v1 \| wrote failed\.json@v1\]$', failed), failed
        assert 'SELECT' not in output.rsplit('\n', 1)[1]
        with closing(child.evidence_store.connect()) as conn:
            count = conn.execute('SELECT count(*) FROM requests WHERE query_id IS NOT NULL').fetchone()[0]
        traced = query(child.tool_executor, 'pf_dependencies', target={'path': 'query.json'},
                       include_execution=True, depth=4, purpose='historical_only')
        assert any(item['target'].get('sql') == 'SELECT COUNT(*) AS n FROM ledger' for item in traced['items'])
        with closing(child.evidence_store.connect()) as conn:
            assert conn.execute('SELECT count(*) FROM requests WHERE query_id IS NOT NULL').fetchone()[0] == count
        child._save_checkpoint(7)
        child._stop_server()
        restored = offline_runner(child.workspace_dir)
        header = json.loads(result.split('\n', 1)[0])
        reread = restored._execute_tool('pf_read', dict(target={'version': header['target']['version']}))
        assert reread.split('\n', 1)[1] == 'prefix evidence'


def test_bash_handles_group_queries_and_writes_even_after_a_failed_exit(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    output = executor.execute('bash', {'command': 'printf 1 > one.txt; printf 2 > two.txt; exit 1'})
    assert re.search(r'\n\[pf: cmd1@v1 \| wrote one\.txt@v1, two\.txt@v1\]$', output), output
    many = ' '.join(f'printf {i} > f{i}.txt;' for i in range(10))
    output = executor.execute('bash', {'command': many})
    assert output.endswith(' and 4 more]') and output.count('.txt@v1') == 6, output
    # A changed file gets the next version; the same command returning the same text keeps its handle.
    again = executor.execute('bash', {'command': 'printf 3 > one.txt; exit 1'})
    assert again.endswith('[pf: cmd3@v1 | wrote one.txt@v2]'), again
    (workspace / 'one.txt').write_text('1')
    assert executor.execute('bash', {'command': 'printf 3 > one.txt; exit 1'}).endswith(
        '[pf: cmd3@v1 (same as day 0) | wrote one.txt@v4]')
    history = query(executor, 'pf_read', mode='history', target={'path': 'one.txt'})['items']
    assert history[0]['same_content_as'] == 'one.txt@v2'


def test_handles_name_the_object_and_its_version_in_each_branch(workspace, tmp_path):
    from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
    from saas_bench.text_registry import TextRegistry
    store, registry, executor = captured(workspace, tmp_path)
    for name, content in (('a.txt', 'A'), ('b.txt', 'B')):
        (workspace / name).write_text(content)
    assert executor.execute('read_file', {'path': 'a.txt'}).endswith('[pf: a.txt@v1]')
    assert executor.execute('read_file', {'path': 'b.txt'}).endswith('[pf: b.txt@v1]')
    (workspace / 'a.txt').write_text('A2')
    assert executor.execute('read_file', {'path': 'a.txt'}).endswith('[pf: a.txt@v2]')
    receipt = store.snapshot(tmp_path / 'child.sqlite')
    child_store = SQLEvidenceStore(tmp_path / 'child.sqlite', identity(
        'child', capture_scope='execution', parent_branch='prefix', fork_seq=receipt['cutoff']))
    child = BashAgentToolExecutor(workspace, evidence_store=child_store,
                                  text_registry=TextRegistry(workspace, 'pf', child_store))
    # Handles shown before the fork keep their meaning in the copied branch.
    assert read(child, target={'version': 'a.txt@v1'})[1] == 'A'
    assert read(child, target={'version': 'a.txt@v2'})[1] == 'A2'
    # Each branch numbers its own later versions; nothing is shared between branches.
    (workspace / 'a.txt').write_text('A3')
    assert child.execute('read_file', {'path': 'a.txt'}).endswith('[pf: a.txt@v3]')
    assert executor.execute('read_file', {'path': 'a.txt'}).endswith('[pf: a.txt@v3]')
    with pytest.raises(AssertionError, match='Unknown version handle'):
        read(child, target={'version': 'a.txt@v9'})
    # A handle written as a path is accepted too.
    assert read(child, target={'path': 'b.txt@v1'})[1] == 'B'
    send(store, executor.execute('read_file', {'path': 'b.txt'}))
    created = call(registry, 'create', **declaration({'path': 'b.txt@v1'}))
    assert created['evidence'][0]['version'] == 'b.txt@v1'
    assert store.load_state('declaration:r1.1')['references'][0]['status'] == 'resolved'


def test_path_dependents_survive_boundary_snapshots_of_unchanged_bytes(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration({'path': 'evidence.json'}))
    before = query(executor, 'pf_dependents', target={'path': 'evidence.json'})
    assert [r['source']['record'] for r in before['items']] == ['r1.1']
    # Each Bash boundary stores new versions of unchanged files; the reference still matches.
    executor.execute('bash', {'command': 'echo hi'})
    executor.execute('bash', {'command': 'echo again'})
    for extra in ({}, {'include_execution': True}):
        after = query(executor, 'pf_dependents', target={'path': 'evidence.json'}, **extra)
        assert 'r1.1' in [r['source'].get('record') for r in after['items']]
        assert not any('observed_as' in r for r in after['items'])
    # Changed bytes are a different node: the old reference is not a referrer of the new content.
    (workspace / 'evidence.json').write_text('{"n":8}')
    executor.execute('bash', {'command': 'true'})
    assert query(executor, 'pf_dependents', target={'path': 'evidence.json'})['items'] == []


def test_script_output_handle_cites_printed_result_and_reruns_its_queries(offline_runner):
    """Agents run SDK scripts and read printed summaries, never the raw query response."""
    runner = offline_runner(text_registration='pf')
    runner.agent.current_day = 0
    script = ("cat > calc.py <<'EOF'\nimport novamind_api as nm\n"
              "r = nm.query('SELECT COUNT(*) AS n FROM ledger')\nprint('ledger rows:', r['rows'][0]['n'])\nEOF")
    runner._execute_tool('bash', {'command': script})
    output = runner._execute_tool('bash', {'command': './novamind-operation python calc.py'})
    footer = re.search(r'\n\[pf: (calc\.py\.out@v1)\]$', output)
    assert footer and 'ledger rows:' in output, output
    store = runner.evidence_store
    settled_request = send(store, output)
    assert settled_request
    # The raw query result never reached the model, so SQL alone is still refused, with a pointer.
    refused = runner._execute_tool('text_create', declaration({'sql': 'SELECT COUNT(*) AS n FROM ledger'}))
    assert refused.startswith('Error:') and 'first handle in its [pf: ...] line' in refused, refused
    created = runner._execute_tool('text_create', declaration({'version': footer.group(1)}))
    assert created.startswith('Registered r1.1 (active).\nCited: calc.py.out@v1 (day 0)'), created
    assert runner.tool_executor.text_registry.last_result['evidence'][0]['version'] == footer.group(1)
    traced = query(runner.tool_executor, 'pf_dependencies', target={'record': 'r1'}, purpose='current', depth=4)
    assert traced['stale_check'] == 'performed'
    upstream = [i for i in traced['items'] if (i['target'] or {}).get('sql') == 'SELECT COUNT(*) AS n FROM ledger']
    assert upstream and upstream[0]['kind'] == 'same_execution' and upstream[0]['path'][1] == footer.group(1)
    assert upstream[0]['check']['version_changed'] is False, json.dumps(upstream[0]['check'])
    # Rerun, not reused: an equal result is a new acquisition of the same content, so one handle.
    handle = upstream[0]['target']['version']
    assert upstream[0]['check']['current_version'] == handle
    rerun = query(runner.tool_executor, 'pf_read', mode='history', target={'version': handle})['items']
    assert len(rerun) == 1 and rerun[0]['version'] == handle  # PF reruns are not listed as history
    # Running the same script again with the same printed output keeps the handle; a
    # different output is its next version, whatever the surrounding command looks like.
    output = runner._execute_tool('bash', {'command': './novamind-operation python calc.py 2>&1 | head -5'})
    assert re.search(r'\[pf: calc\.py\.out@v1 \(same as day 0\)\]$', output), output
    runner._execute_tool('bash', {'command': "printf 'print(2)\\nprint(3)\\n' > calc.py"})
    output = runner._execute_tool('bash', {'command': './novamind-operation python calc.py | head -1'})
    assert re.search(r'\[pf: cmd\d+@v1, with part of calc\.py\.out@v2\]$', output), output
    assert send(store, output)
    # Only the first line reached the model; whole capture and actual reading stay distinct.
    refused = runner._execute_tool('text_create', declaration({'version': 'calc.py.out@v2'}))
    assert refused.startswith('Registered r2.1') and 'part of body present in this request' in refused, refused
    shown = re.search(r'\[pf: (cmd\d+@v1),', output).group(1)
    assert receipt(runner._execute_tool('text_create', declaration({'version': shown})))['id'] == 'r3'
    # The script itself is traced too; interpreter caches and session logs get no handles.
    assert '__pycache__' not in output and 'sessions/' not in output
