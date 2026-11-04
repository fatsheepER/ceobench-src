import json
import re
import shlex

from test_preflight_team_flow import final, make_team
from test_pf_candidates import day, query
from test_relation_batch import batch, create
from test_text_registry import captured, declaration, send, workspace


def cli(team, role, *args):
    result = team.roles[role].executor.execute('bash', {'command': shlex.join(args)})
    assert '[exit code:' not in result, result
    assert not result.startswith('Error:'), result
    return result


def details(team, role, *args):
    cli(team, role, 'pf', *args, '--detail')
    page = team.roles[role].executor.pf_queries.last_answer
    fragments = [page['detail_text']]
    while page['next_cursor']:
        cli(team, role, 'pf', 'more', page['next_cursor'])
        page = team.roles[role].executor.pf_queries.last_answer
        fragments.append(page['detail_text'])
    return json.loads(''.join(fragments))


def test_public_depend_and_rdepend_follow_exact_saved_queries():
    with make_team('pf', lambda *args: final('done'), 'relation-public-bidirectional') as team:
        day(team.roles['growth'].store, 0)
        for i in range(3):
            cli(team, 'growth', './novamind-operation', 'query', f'SELECT {i} AS q{i}')
        output, _ = query(team.roles['growth'].executor, 'pf candidates --history --kind sql')
        q = [next(re.match(r'(growth:query\d+@v\d+)', line)[1] for line in output.splitlines()
                  if f' AS q{i} |' in line) for i in range(3)]
        code = "from pathlib import Path; Path('F1.txt').write_text('one'); Path('F2.txt').write_text('two')"
        cli(team, 'growth', 'python', '-c', code)
        f1, f2 = 'growth:F1.txt@v1', 'growth:F2.txt@v1'
        receipt = batch(team, 'growth', [create('f1', f1, [q[1]]), create('f2', f2, [q[2]])], 'first')
        assert receipt['saved'] == 2
        for target, source, other in ((f1, q[1], q[2]), (f2, q[2], q[1])):
            result = cli(team, 'ceo', 'pf', 'depend', target)
            assert source in result and other not in result
            reverse = cli(team, 'ceo', 'pf', 'rdepend', source)
            assert target in reverse
        item = receipt['items'][0]
        revised = batch(team, 'growth', [dict(item_id='add-q0', op='revise', relation=item['relation'],
            expected_revision=item['revision'], sources=[{'ref': {'cite': q[1]}}, {'ref': {'cite': q[0]}}],
            reason='Add earlier saved observation.')], 'add-q0')
        assert revised['saved'] == 1
        result = cli(team, 'ceo', 'pf', 'depend', f1)
        assert q[0] in result and q[1] in result and q[2] not in result
        assert f1 in cli(team, 'ceo', 'pf', 'rdepend', q[0])


def test_public_queries_default_to_current_declarations_and_offer_history():
    with make_team('pf', lambda *args: final('done'), 'relation-public-revisions') as team:
        code = "from pathlib import Path; Path('source.txt').write_text('old'); Path('target.txt').write_text('target')"
        cli(team, 'growth', 'python', '-c', code)
        source1, target = 'growth:source.txt@v1', 'growth:target.txt@v1'
        first = batch(team, 'growth', [create('first', target, [source1])], 'first')['items'][0]
        cli(team, 'growth', 'python', '-c', "from pathlib import Path; Path('source.txt').write_text('new')")
        source2 = 'growth:source.txt@v2'
        second = batch(team, 'growth', [dict(item_id='replace', op='revise', relation=first['relation'],
            expected_revision=first['revision'], sources=[{'ref': {'cite': source2}}], reason='Use the new version.')], 'replace')['items'][0]
        current = cli(team, 'ceo', 'pf', 'depend', target)
        assert source2 in current and source1 not in current
        assert target not in cli(team, 'ceo', 'pf', 'rdepend', source1)
        historical = cli(team, 'ceo', 'pf', 'depend', target, '--all-revisions')
        assert source1 in historical and source2 in historical
        retired = batch(team, 'growth', [dict(item_id='retire', op='retire', relation=first['relation'],
            expected_revision=second['revision'], reason='No longer applicable.')], 'retire')
        assert retired['saved'] == 1
        assert source2 not in cli(team, 'ceo', 'pf', 'depend', target)
        assert source2 in cli(team, 'ceo', 'pf', 'depend', target, '--all-revisions')


def test_public_query_distinguishes_declarers_and_missing_sources():
    with make_team('pf', lambda *args: final('done'), 'relation-public-authors') as team:
        from test_preflight_team_evidence import write
        write(team, 'growth', 'source', 'source.txt')
        write(team, 'growth', 'target', 'target.txt')
        source, target = 'growth:source.txt@v1', 'growth:target.txt@v1'
        first = batch(team, 'growth', [create('own', target, [source])], 'own')['items'][0]
        inferred = batch(team, 'ceo', [create('peer', target, [source], assertion='inferred')], 'peer')['items'][0]
        gap = dict(description='Earlier unavailable input', reason='No saved bytes exist.', last_known_ref='lost-q0')
        missing = create('gap', target, [])
        missing['sources'] = [{'ref': {'missing': gap}}]
        third = batch(team, 'growth', [missing], 'missing')['items'][0]
        detail = json.dumps(details(team, 'ops_finance', 'depend', target)['items'])
        for item in (first, inferred, third):
            assert item['relation'] in detail
        assert 'author_declared' in detail and 'inferred' in detail
        assert 'Earlier unavailable input' in detail and 'lost-q0' in detail
        reverse = json.dumps(details(team, 'ops_finance', 'rdepend', source)['items'])
        assert first['relation'] in reverse and inferred['relation'] in reverse
        assert third['relation'] not in reverse


def test_public_operation_relations_distinguish_same_value_events():
    from test_preflight_team_flow import raw
    from test_responsibility_authority import prepare
    with make_team('pf', lambda *args: final('done'), 'relation-public-operations') as team:
        prepare(team)
        from test_preflight_team_evidence import declare
        declare(team, 'growth', 'Operation source', ['unknown: initial observation'])
        team._set_phase('analysis')
        refs = []
        for _ in range(2):
            status, response = raw(team, 'growth', '/call', {'tool': 'set_capacity_tier', 'args': {'tier': 3}})
            assert status == 200 and response['operation']['outcome'] == 'succeeded'
            refs.append(response['operation']['operation_ref'])
        assert refs[0] != refs[1]
        items = [dict(item_id=str(i), op='create', target={'operation_ref': ref},
                      sources=[{'ref': {'cite': 'growth:r1.1'}}], assertion='author_declared',
                      reason='Link the exact operation.') for i, ref in enumerate(refs)]
        receipt = batch(team, 'growth', items, 'operations')
        assert receipt['saved'] == 2
        team._set_phase('ceo')
        reverse = json.dumps(details(team, 'ceo', 'rdepend', 'growth:r1.1')['items'])
        assert all(ref in reverse for ref in refs)
        for ref, other in (refs, refs[::-1]):
            result = json.dumps(details(team, 'ceo', 'depend', ref)['items'])
            assert 'growth:r1.1' in result and other not in result


def test_git_relations_are_readable_by_ordinary_script():
    import json
    from test_preflight_team_evidence import write
    with make_team('git', lambda *args: final('done'), 'relation-git-readable') as team:
        write(team, 'growth', 'source', 'source.txt')
        write(team, 'growth', 'target', 'target.txt')
        saved = batch(team, 'growth', [create('link', 'growth:target.txt', ['growth:source.txt'])], 'link')['items'][0]
        script = "import json; from pathlib import Path; print(json.dumps(json.loads(Path('documents/relations.json').read_text())))"
        output = cli(team, 'growth', 'python', '-c', script)
        assert saved['relation'] in output
        data = json.loads(output)
        assert data['records']


def test_legacy_text_references_and_detail_do_not_expand_depth():
    from test_preflight_team_evidence import declare, write
    with make_team('pf', lambda *args: final('done'), 'relation-public-legacy') as team:
        write(team, 'growth', 'original source', 'source.txt')
        declare(team, 'growth', 'first legacy statement', ['growth:source.txt@v1'])
        declare(team, 'growth', 'second legacy statement', ['growth:r1.1'])
        direct = json.dumps(details(team, 'ceo', 'rdepend', 'growth:source.txt@v1')['items'])
        assert 'growth:r1.1' in direct and 'growth:r2.1' not in direct
        history = cli(team, 'ceo', 'pf', 'rdepend', 'growth:source.txt@v1', '--depth', '2')
        assert 'growth:r1.1' in history and 'growth:r2.1' in history
        assert 'growth:source.txt@v1' in cli(team, 'ceo', 'pf', 'depend', 'growth:r1.1')


def test_relation_pages_freeze_at_query_and_do_not_replay_operations():
    import json
    from test_preflight_team_evidence import write
    with make_team('pf', lambda *args: final('done'), 'relation-public-pages') as team:
        write(team, 'growth', 'source', 'source.txt')
        for i in range(22):
            write(team, 'growth', f'target {i}', f'target{i}.txt')
        source = 'growth:source.txt@v1'
        items = [create(str(i), f'growth:target{i}.txt@v1', [source]) for i in range(22)]
        receipt = batch(team, 'growth', items, 'initial')
        assert receipt['saved'] == 22
        expected = {item['relation'] for item in receipt['items']}
        before = tuple(team.server.conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                       for table in ('business_operations', 'business_effects'))
        executor = team.roles['ceo'].executor
        cli(team, 'ceo', 'pf', 'rdepend', source)
        page = executor.pf_queries.last_answer
        assert page['next_cursor']
        rows = list(page['items'])
        write(team, 'growth', 'later target', 'later.txt')
        late = batch(team, 'growth', [create('late', 'growth:later.txt@v1', [source])], 'late')['items'][0]
        last = receipt['items'][-1]
        batch(team, 'growth', [dict(item_id='retire', op='retire', relation=last['relation'],
            expected_revision=last['revision'], reason='Retired after the query.')], 'retire')
        while page['next_cursor']:
            cli(team, 'ceo', 'pf', 'more', page['next_cursor'])
            page = executor.pf_queries.last_answer
            rows.extend(page['items'])
        rendered = [json.dumps(row, sort_keys=True) for row in rows]
        assert len(rows) == len(expected)
        assert all(sum(bool(re.search(re.escape(relation) + r'(?!\d)', row)) for row in rendered) == 1
                   for relation in expected)
        assert all(late['relation'] not in row for row in rendered)
        after = tuple(team.server.conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                      for table in ('business_operations', 'business_effects'))
        assert after == before


def test_candidate_edges_require_opt_in_and_history_never_adds_them():
    with make_team('pf', lambda *args: final('done'), 'relation-public-candidates') as team:
        runtime = team.roles['growth']
        day(runtime.store, 0)
        result = runtime.executor.execute('bash', {'command': './novamind-operation query "SELECT 42 AS candidate_marker"; printf derived'})
        assert '[exit code:' not in result and 'candidate_marker' in result
        output, _ = query(runtime.executor, 'pf candidates --history --kind sql')
        source = next(re.match(r'(growth:query\d+@v\d+)', line)[1] for line in output.splitlines()
                      if ' AS candidate_marker |' in line)
        base = cli(team, 'growth', 'pf', 'rdepend', source)
        history = cli(team, 'growth', 'pf', 'depend', source, '--history', '--depth', '2')
        inclusive = cli(team, 'growth', 'pf', 'rdepend', source, '--include-candidates')
        assert 'origin=automatic_capture' not in base and 'origin=automatic_capture' not in history
        assert '| execution_observation |' in inclusive


def test_relation_cursor_rechecks_current_role_permissions(monkeypatch):
    from test_preflight_team_evidence import write
    with make_team('pf', lambda *args: final('done'), 'relation-public-permissions') as team:
        write(team, 'growth', 'secret source', 'source.txt')
        for i in range(22):
            write(team, 'growth', 'target', f'target{i}.txt')
        source = 'growth:source.txt@v1'
        result = batch(team, 'growth', [create(str(i), f'growth:target{i}.txt@v1', [source])
                                       for i in range(22)], 'permissions')
        assert result['saved'] == 22
        executor = team.roles['ceo'].executor
        cli(team, 'ceo', 'pf', 'rdepend', source)
        cursor = executor.pf_queries.last_answer['next_cursor']
        assert cursor
        monkeypatch.setattr('saas_bench.role_policy.READABLE_ROLES',
                            dict(ceo=('ceo',), growth=('growth',), ops_finance=('ops_finance',)))
        denied = executor.execute('bash', {'command': shlex.join(['pf', 'more', cursor])})
        assert '[exit code: 1]' in denied and 'inaccessible' in denied.lower()
        assert 'growth:target' not in denied


def test_pf_queries_keep_legacy_git_commit_and_cross_role_storage():
    import json
    from test_preflight_team_evidence import declare, write
    with make_team('pf', lambda *args: final('done'), 'relation-public-git-ref') as team:
        write(team, 'growth', 'saved bytes', 'git-source.txt')
        commit = team._snapshot_history(team.roles['growth'], 'saved source')
        declare(team, 'ceo', 'Legacy Git evidence', ['growth:git-source.txt@' + commit[:7]])
        record = team.roles['ceo'].registry._load()['records']['r1'][0]
        assert record['references'][0]['evidence']['commit'] == commit
        write(team, 'growth', 'later bytes', 'git-source.txt')
        output = cli(team, 'ops_finance', 'pf', 'depend', 'ceo:r1.1', '--detail')
        short = team._git(team.roles['growth'], 'rev-parse', '--short=7', commit)
        assert short in output and commit not in output and 'saved_content_unavailable' not in output
        assert team._git(team.roles['growth'], 'show', short + ':git-source.txt') == 'saved bytes'
        exact = 'growth:git-source.txt@' + commit
        assert 'ceo:r1.1' in cli(team, 'ops_finance', 'pf', 'rdepend', exact)
        assert 'ceo:r1.1' in cli(team, 'ops_finance', 'pf', 'rdepend', exact[:-33])
        saved = batch(team, 'ceo', [create('copy', 'ceo:r1.1', [exact])], 'copy')['items'][0]
        bound = saved['sources'][0]['ref']
        assert bound['storage_role'] == 'ceo' and bound['source_role'] == 'growth'
        result = cli(team, 'ops_finance', 'pf', 'depend', 'ceo:r1.1', '--detail')
        assert json.dumps(bound['ref']['cite']) in result
        assert '"storage_role": "ceo"' in result and '"source_role": "growth"' in result


def test_full_relation_detail_paginates_long_missing_reason_and_empty_revision():
    import json
    from test_preflight_team_evidence import write
    with make_team('pf', lambda *args: final('done'), 'relation-long-detail') as team:
        write(team, 'growth', 'target', 'target.txt')
        gap = dict(description='Unsaved older source', reason=('gap ' * 1000).rstrip(), last_known_ref='old-output')
        items = []
        for i in range(6):
            item = create(str(i), 'growth:target.txt@v1', [])
            item['sources'] = [{'ref': {'missing': gap}}]
            items.append(item)
        receipt = batch(team, 'growth', items, 'long')
        assert receipt['saved'] == 6
        executor = team.roles['ceo'].executor
        output, page = query(executor, 'pf depend growth:target.txt@v1 --detail')
        parts = [page['detail_text']]
        assert page['has_more'] and len(output) < 28000
        while page['next_cursor']:
            output, page = query(executor, 'pf more ' + page['next_cursor'])
            assert len(output) < 28000
            parts.append(page['detail_text'])
        data = json.loads(''.join(parts))
        assert len(data['items']) == 6
        for item in data['items']:
            assert item['source']['ref']['missing'] == gap
        first = receipt['items'][0]
        changed = batch(team, 'growth', [dict(item_id='clear', op='revise', relation=first['relation'],
            expected_revision=first['revision'], sources=[], reason='Clear all sources.')], 'clear')['items'][0]
        _, current = query(executor, 'pf depend growth:target.txt@v1')
        assert any(item['relation_revision'] == changed['relation_revision'] and item['source'] is None
                   for item in current['items'])


def test_converging_relation_paths_are_not_cycles():
    from test_preflight_team_evidence import write
    with make_team('pf', lambda *args: final('done'), 'relation-diamond') as team:
        for name in ('A', 'B', 'C', 'D'):
            write(team, 'growth', name, name + '.txt')
        ref = lambda name: 'growth:' + name + '.txt@v1'
        result = batch(team, 'growth', [create('b', ref('B'), [ref('A')]),
            create('c', ref('C'), [ref('A')]), create('d', ref('D'), [ref('B'), ref('C')])], 'diamond')
        assert result['saved'] == 3
        rows = details(team, 'ceo', 'rdepend', ref('A'), '--depth', '3')['items']
        assert len(rows) == 4
        assert all(row['unexpanded'] != 'cycle' for row in rows)
        assert any(row['unexpanded'] == 'already_expanded' for row in rows)


def test_pf_redirection_file_remains_queryable_after_relation_registration():
    from test_preflight_team_evidence import write
    with make_team('pf', lambda *args: final('done'), 'relation-retrieval-file') as team:
        write(team, 'growth', 'saved source', 'source.txt')
        output = team.roles['growth'].executor.execute('bash', {'command': 'pf show source.txt@v1 --full > copied.txt'})
        assert '[exit code:' not in output and 'copied.txt@v1' in output
        receipt = batch(team, 'growth', [create('copy', 'growth:copied.txt@v1', ['growth:source.txt@v1'])], 'copy')
        assert receipt['saved'] == 1
        assert 'growth:source.txt@v1' in cli(team, 'ceo', 'pf', 'depend', 'growth:copied.txt@v1')
        assert 'growth:copied.txt@v1' in cli(team, 'ceo', 'pf', 'rdepend', 'growth:source.txt@v1')


def test_peer_execution_membership_navigation_finds_the_saved_event():
    from test_preflight_team_evidence import write
    with make_team('pf', lambda *args: final('done'), 'relation-peer-membership') as team:
        day(team.roles['growth'].store, 0)
        write(team, 'growth', 'exact peer content', 'peer.txt')
        data = details(team, 'ceo', 'rdepend', 'growth:peer.txt@v1', '--include-candidates')
        memberships = [row for row in data['items'] if row.get('basis') == 'execution_membership']
        assert memberships
        for item in memberships:
            output, page = query(team.roles['ceo'].executor, item['navigation'])
            assert page['total_in_scope'] > 0
            assert any(row['content_ref'] == item['content_ref'] for row in page['items'])


def single_cli(executor, *args):
    output = executor.execute('bash', {'command': shlex.join(['pf', *args])})
    assert '[exit code:' not in output and not output.startswith('Error:'), output
    return output


def test_single_role_legacy_returned_refs_round_trip(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    created = executor.execute('text_create', declaration({'path': 'evidence.json'}))
    assert 'Registered r1.1' in created
    single_cli(executor, 'depend', 'r1.1')
    page = executor.pf_queries.last_answer
    assert page['scope']['revisions']['legacy_text_reference'] == 'exact_reachable_saved_revisions'
    source = page['items'][0]['source']['ref']['cite']
    target = page['items'][0]['target']['ref']['cite']
    assert '"n":7' in single_cli(executor, 'show', source, '--full')
    assert 'Keep the plan' in single_cli(executor, 'show', target, '--full')
    single_cli(executor, 'depend', target)
    assert executor.pf_queries.last_answer['items'][0]['source']['ref']['cite'] == source
    single_cli(executor, 'rdepend', source)
    assert executor.pf_queries.last_answer['items'][0]['target']['ref']['cite'] == target
    assert source == 'evidence.json@v1' and target == 'r1.1'


def test_single_role_common_relations_normalize_query_copies(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    created = executor.execute('text_create', declaration({'path': 'evidence.json'}))
    assert 'Registered r1.1' in created
    written = executor.execute('write_file', {'path': 'target.txt', 'content': 'stored target'})
    assert not written.startswith('Error:'), written
    receipt = json.loads(executor.execute('escalate', {'items': [
        create('file', 'target.txt@v1', ['r1.1', 'evidence.json@v1'], assertion='inferred'),
        create('text', 'r1.1', ['evidence.json@v1'], assertion='inferred'),
    ]}, tool_call_id='single-role-relations'))
    assert receipt['saved'] == 2, receipt
    persisted = registry.relations.path.read_bytes()
    file_relation, text_relation = [item['relation_revision'] for item in receipt['items']]
    assert receipt['items'][1]['target']['ref']['cite'] == 'ceo:r1.1'
    single_cli(executor, 'depend', 'target.txt@v1')
    page = executor.pf_queries.last_answer
    rows = [row for row in page['items'] if row.get('relation_revision') == file_relation]
    assert len(rows) == 2
    assert {row['source']['ref']['cite'] for row in rows} == {'r1.1', 'evidence.json@v1'}
    for row in rows:
        source, target = row['source']['ref']['cite'], row['target']['ref']['cite']
        single_cli(executor, 'show', source, '--full')
        assert 'stored target' in single_cli(executor, 'show', target, '--full')
        single_cli(executor, 'rdepend', source)
        assert any(item.get('relation_revision') == file_relation for item in executor.pf_queries.last_answer['items'])
    single_cli(executor, 'depend', 'r1.1')
    assert any(row.get('relation_revision') == text_relation for row in executor.pf_queries.last_answer['items'])
    assert registry.relations.path.read_bytes() == persisted
