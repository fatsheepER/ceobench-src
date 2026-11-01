"""Shared relation declarations through the role's public model tools."""
from copy import deepcopy
import json
import re
import shlex

import pytest

from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from test_preflight_team_evidence import declare, write
from test_preflight_team_flow import final, make_team


def batch(team, role, items, call_id):
    output = team.roles[role].executor.execute(
        'relation_batch', {'items': items}, tool_call_id=call_id)
    assert not output.startswith('Error:'), output
    return json.loads(output)


def create(item_id, target, sources, *, assertion='author_declared', reason='Keep reusable evidence.'):
    return dict(item_id=item_id, op='create', target={'cite': target},
                sources=[{'ref': {'cite': source}} for source in sources],
                assertion=assertion, reason=reason)


def file_ref(mode, role, name, version=1):
    return f'{role}:{name}' + (f'@v{version}' if mode == 'pf' else '')


def items_by_id(receipt):
    return {item['item_id']: item for item in receipt['items']}


def prepare(team):
    for role in ('ceo', 'growth'):
        write(team, role, role + ' source', path='source.txt')
        declare(team, role, role + ' reusable conclusion', ['unknown: initial declaration'])


def test_relation_schema_is_shared_and_discoverable():
    definitions = []
    for pf in (False, True):
        tools = get_bash_agent_tool_descriptions(text_registration=True, pf_queries=pf, team=True)
        relation = next(tool for tool in tools if tool['name'] == 'relation_batch')
        definitions.append(relation)
        schema = json.dumps(relation['parameters'])
        for field in ('item_id', 'op', 'target', 'sources', 'assertion', 'reason',
                      'relation', 'expected_revision', 'operation_ref', 'missing'):
            assert '"' + field + '"' in schema
        assert 'author_declared' in schema and 'inferred' in schema
        assert 'create' in schema and 'revise' in schema and 'retire' in schema
    assert definitions[0]['parameters'] == definitions[1]['parameters']


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_partial_batch_is_atomic_per_target_and_can_be_corrected(mode):
    with make_team(mode, lambda *args: final('done'), 'relations-partial') as team:
        prepare(team)
        registry = team.roles['ceo'].registry
        legacy = registry.path.read_bytes()
        sources = [file_ref(mode, 'ceo', 'source.txt'), file_ref(mode, 'growth', 'source.txt')]
        valid = create('valid', 'ceo:r1.1', sources)
        invalid = create('invalid', 'ceo:r1.1', [sources[0], 'ceo:no-such-output.txt'])
        missing = create('missing', 'ceo:r1.1', [])
        gap = dict(description='Earlier unsaved output', reason='No saved bytes exist.',
                   last_known_ref='query made before saving was enabled')
        missing['sources'] = [{'ref': {'missing': gap}}]
        file_target = create('file', sources[0], ['ceo:r1.1'], assertion='inferred')
        result = batch(team, 'ceo', [valid, invalid, missing, file_target], 'partial')
        assert result['status'] == 'partial'
        assert (result['submitted'], result['saved'], result['rejected'], result['not_attempted']) == (4, 3, 1, 0)
        items = items_by_id(result)
        assert items['invalid']['status'] == 'rejected' and items['invalid']['error_code']
        assert len(items['valid']['sources']) == 2
        assert all(source['ref']['source_state'] == 'available' for source in items['valid']['sources'])
        bound_gap = items['missing']['sources'][0]['ref']
        assert bound_gap['kind'] == 'missing'
        assert bound_gap['source_state'] == 'missing' and bound_gap['readable'] is False
        assert bound_gap['ref'] == {'missing': gap}
        assert items['file']['target']['kind'] == 'file'
        assert len(registry.relations._load()['records']) == 3
        assert registry.path.read_bytes() == legacy
        invalid['sources'][1]['ref']['cite'] = sources[1]
        corrected = batch(team, 'ceo', [invalid], 'corrected')
        assert corrected['status'] == 'saved' and corrected['saved'] == 1
        assert len(registry.relations._load()['records']) == 4


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_revision_replaces_whole_sources_and_keeps_exact_old_target(mode):
    with make_team(mode, lambda *args: final('done'), 'relations-revisions') as team:
        prepare(team)
        registry = team.roles['ceo'].registry
        first = batch(team, 'ceo', [create('first', 'ceo:r1.1', [file_ref(mode, 'ceo', 'source.txt')])], 'first')['items'][0]
        revised_text = team.roles['ceo'].executor.execute('text_revise',
            dict(record='r1', text='A revised conclusion.', reason='New observation.'))
        assert 'ceo:r1.2' in revised_text
        second = batch(team, 'ceo', [dict(item_id='revise', op='revise', relation=first['relation'],
            expected_revision=first['revision'], target={'cite': 'ceo:r1.2'}, sources=[],
            reason='Withdraw every source, while retaining the earlier version.')], 'revise')['items'][0]
        assert second['status'] == 'saved' and second['revision'] == 2
        assert second['sources'] == [] and second['target']['ref']['cite'] == 'ceo:r1.2'
        conflict = batch(team, 'ceo', [dict(item_id='stale', op='revise', relation=first['relation'],
            expected_revision=first['revision'], reason='Stale revision.'),
            dict(item_id='current', op='revise', relation=first['relation'],
            expected_revision=second['relation_revision'], reason='Preserve the now-empty source list.')], 'conflict')
        assert conflict['status'] == 'partial'
        outcome = items_by_id(conflict)
        assert outcome['stale']['status'] == 'rejected' and outcome['stale']['error_code']
        assert outcome['current']['status'] == 'saved' and outcome['current']['revision'] == 3
        assert outcome['current']['sources'] == []
        assert outcome['current']['target'] == second['target']
        retired = batch(team, 'ceo', [dict(item_id='retire', op='retire', relation=first['relation'],
            expected_revision=3, reason='Stop using this declaration.')], 'retire')['items'][0]
        assert retired['record_status'] == 'retired' and retired['revision'] == 4
        history = next(iter(registry.relations._load()['records'].values()))
        assert len(history) == 4
        assert history[0]['target']['ref']['cite'] == 'ceo:r1.1'
        assert len(history[0]['sources']) == 1
        assert history[-1]['target']['ref']['cite'] == 'ceo:r1.2'
        assert history[-1]['sources'] == []
        assert retired['history']['path'] == 'relations.json' and retired['history']['role'] == 'ceo'


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_peer_inference_is_allowed_but_authorship_and_revision_are_not(mode):
    with make_team(mode, lambda *args: final('done'), 'relations-authority') as team:
        prepare(team)
        author = create('author', 'growth:r1.1', ['ceo:r1.1'])
        inference = dict(author, item_id='inference', assertion='inferred')
        forged = dict(create('forged', 'ceo:r1.1', ['growth:r1.1']), author_role='growth')
        result = batch(team, 'ceo', [author, inference, forged], 'authority')
        items = items_by_id(result)
        assert result['status'] == 'partial' and result['saved'] == 1 and result['rejected'] == 2
        assert items['author']['status'] == items['forged']['status'] == 'rejected'
        saved = items['inference']
        assert saved['author_role'] == 'ceo' and saved['assertion'] == 'inferred'
        assert saved['target']['producer_role'] == 'growth'
        peer = batch(team, 'growth', [dict(item_id='change-peer', op='revise', relation=saved['relation'],
            expected_revision=saved['revision'], reason='Try to change another author.'),
            create('own', 'growth:r1.1', ['ceo:r1.1'])], 'peer')
        assert peer['status'] == 'partial'
        assert items_by_id(peer)['change-peer']['status'] == 'rejected'
        assert items_by_id(peer)['own']['status'] == 'saved'
        assert len(next(iter(team.roles['ceo'].registry.relations._load()['records'].values()))) == 1


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_duplicate_logical_tool_call_returns_original_receipt_after_revision(mode):
    with make_team(mode, lambda *args: final('done'), 'relations-idempotent') as team:
        prepare(team)
        request = [create('save', 'ceo:r1.1', ['growth:r1.1'])]
        original = batch(team, 'ceo', request, 'logical-call')
        item = original['items'][0]
        assert batch(team, 'ceo', deepcopy(request), 'logical-call') == original
        batch(team, 'ceo', [dict(item_id='edit', op='revise', relation=item['relation'],
            expected_revision=item['revision'], reason='Later revision.')], 'later-call')
        assert batch(team, 'ceo', deepcopy(request), 'logical-call') == original
        records = team.roles['ceo'].registry.relations._load()['records']
        assert len(records) == 1 and len(next(iter(records.values()))) == 2


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_missing_target_and_private_source_are_rejected_without_partial_edges(mode):
    with make_team(mode, lambda *args: final('done'), 'relations-invalid-target') as team:
        prepare(team)
        private = create('private', 'ceo:r1.1', ['growth:r1.1', 'growth:sessions/private.json'])
        missing = create('missing-target', 'ceo:r1.1', ['growth:r1.1'])
        missing['target'] = {'missing': dict(description='No target', reason='Never existed.')}
        valid = create('valid', 'ceo:r1.1', ['growth:r1.1'])
        result = batch(team, 'ceo', [private, missing, valid], 'invalid-target')
        items = items_by_id(result)
        assert (result['saved'], result['rejected']) == (1, 2)
        assert items['private']['status'] == items['missing-target']['status'] == 'rejected'
        assert items['valid']['status'] == 'saved'
        assert len(team.roles['ceo'].registry.relations._load()['records']) == 1


def test_git_same_named_peer_snapshots_preserve_source_and_storage_roles():
    with make_team('git', lambda *args: final('done'), 'relations-snapshots') as team:
        prepare(team)
        write(team, 'ops_finance', 'ops first', path='source.txt')
        request = create('sources', 'ceo:r1.1', ['growth:source.txt', 'ops_finance:source.txt'])
        saved = batch(team, 'ceo', [request], 'snapshots')['items'][0]
        sources = [source['ref'] for source in saved['sources']]
        assert [source['source_role'] for source in sources] == ['growth', 'ops_finance']
        assert all(source['storage_role'] == 'ceo' for source in sources)
        assert sources[0]['ref']['cite'] != sources[1]['ref']['cite']
        for source, original in zip(sources, ('growth source', 'ops first')):
            owner, path_commit = source['ref']['cite'].split(':', 1)
            path, commit = path_commit.rsplit('@', 1)
            assert len(commit) == 40 and re.fullmatch('[0-9a-f]{40}', commit)
            assert source['snapshot_source']['cite'].startswith(source['source_role'] + ':')
            write(team, source['source_role'], 'overwritten later', path=path)
            command = shlex.join(['git', '-C', str(team.roles[owner].workspace), 'show', commit + ':' + path])
            assert team.roles['growth'].executor.execute('bash', {'command': command}) == original
        assert not team.roles['ceo'].executor.execute('bash', {'command': 'git gc --prune=now'}).startswith('Error:')
        reused = create('reuse', sources[0]['ref']['cite'], ['ceo:r1.1'], assertion='inferred')
        again = batch(team, 'ops_finance', [reused], 'reuse')['items'][0]
        assert again['target']['source_role'] == 'growth'
        assert again['target']['storage_role'] == 'ceo'
        assert again['target']['ref']['cite'] == sources[0]['ref']['cite']


def test_pf_saved_output_is_a_target_and_unread_peer_capture_stays_unread():
    with make_team('pf', lambda *args: final('done'), 'relations-output') as team:
        prepare(team)
        output = team.roles['growth'].executor.execute('bash', {'command': './novamind-operation python-c ' + shlex.quote('import novamind_api as nm; print(nm.query("SELECT 17 AS marker"))')})
        handle = re.search(r'\[pf: ((?:growth:)?cmd\d+@v\d+)', output)
        assert handle, output
        cite = handle.group(1)
        if not cite.startswith('growth:'):
            cite = 'growth:' + cite
        request = create('output', cite, ['ceo:r1.1'], assertion='inferred')
        saved = batch(team, 'ceo', [request], 'output-target')['items'][0]
        assert saved['target']['kind'] == 'output'
        assert saved['target']['ref']['cite'] == cite
        assert saved['target']['readable'] is True
        assert saved['target']['capture']['reading_scope'] == 'not_in_request'
        assert len(saved['target']['sha256']) == 64
        read = team.roles['ops_finance'].executor.execute('bash',
            {'command': shlex.join(['pf', 'show', cite, '--full'])})
        assert 'marker' in read and '17' in read and '[exit code: 1]' not in read
