import json
import shlex
import threading

import pytest

from test_preflight_team_evidence import write
from test_preflight_team_flow import final, make_team, raw
from test_relation_batch import batch, create, prepare
from test_text_registry import git


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_operations_bind_exact_public_results_without_replaying(mode):
    with make_team(mode, lambda *args: final('done'), 'relation-operations') as team:
        prepare(team)
        team._set_phase('analysis')
        refs = []
        for tier in (3, 3, 99):
            _, response = raw(team, 'growth', '/call', dict(tool='set_capacity_tier', args=dict(tier=tier)))
            refs.append(response['operation']['operation_ref'])
        before = team.server.conn.execute('SELECT COUNT(*) FROM business_operations').fetchone()[0]
        items = [dict(item_id=str(i), op='create', target=dict(operation_ref=ref),
                      sources=[dict(ref=dict(cite='growth:r1.1')), dict(ref=dict(operation_ref=refs[0]))],
                      assertion='author_declared', reason='Keep the exact operation and its outcome.') for i, ref in enumerate(refs)]
        result = batch(team, 'growth', items, 'operations')
        assert result['saved'] == 3, result
        targets = [item['target'] for item in result['items']]
        assert [t['ref']['operation_ref'] for t in targets] == refs
        assert [t['operation']['operation']['outcome'] for t in targets] == ['succeeded', 'succeeded', 'rejected']
        assert len(set(refs)) == 3
        assert json.loads(targets[1]['operation']['operation']['result_json'])['data']['changes'] == {}
        for target in targets:
            operation = target['operation']
            assert [e['effect_seq'] for e in operation['effects']] == sorted(e['effect_seq'] for e in operation['effects'])
            assert all(e['operation_ref'] == target['ref']['operation_ref'] for e in operation['effects'])
        team._set_phase('ceo')
        denied = batch(team, 'ceo', items[:1], 'cannot-declare-peer')
        assert denied['rejected'] == 1 and denied['items'][0]['error_code'] == 'author_mismatch'
        items[0]['assertion'] = 'inferred'
        assert batch(team, 'ceo', items[:1], 'infer-peer')['saved'] == 1
        assert team.server.conn.execute('SELECT COUNT(*) FROM business_operations').fetchone()[0] == before


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_malformed_items_reject_individually_without_pausing(mode):
    with make_team(mode, lambda *args: final('done'), 'relation-malformed') as team:
        prepare(team)
        request = [dict(item_id='list-op', op=[]), dict(item_id='object-op', op={}),
                   dict(item_id='missing-target', op='create', target={'missing': {'description': 'no target', 'reason': 'lost'}},
                        sources=[{'ref': {'cite': 'ceo:r1.1'}}], assertion='inferred', reason='invalid'),
                   create('ambiguous-text', 'ceo:r1', ['growth:r1.1']),
                   create('valid', 'ceo:r1.1', ['growth:r1.1'])]
        receipt = batch(team, 'ceo', request, 'malformed')
        assert receipt['status'] == 'partial' and receipt['saved'] == 1 and receipt['rejected'] == 4
        assert batch(team, 'ceo', [create('next', 'ceo:r1.1', ['growth:r1.1'])], 'next')['saved'] == 1
        assert not team._lifecycle.reason


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_private_material_is_not_a_citable_target_or_source(mode):
    with make_team(mode, lambda *args: final('done'), 'relation-private') as team:
        prepare(team)
        peer = team.roles['growth']
        (peer.workspace / 'sessions').mkdir(exist_ok=True)
        (peer.workspace / 'sessions/private.txt').write_text('SECRET-MUST-NOT-APPEAR')
        items = [create('secret-source', 'ceo:r1.1', ['growth:sessions/private.txt']),
                 create('secret-target', 'growth:sessions/private.txt', ['ceo:r1.1'], assertion='inferred'),
                 create('ordinary', 'ceo:r1.1', ['growth:r1.1'])]
        receipt = batch(team, 'ceo', items, 'private')
        assert receipt['saved'] == 1 and receipt['rejected'] == 2
        assert 'SECRET-MUST-NOT-APPEAR' not in json.dumps(receipt)
        assert 'SECRET-MUST-NOT-APPEAR' not in team.roles['ceo'].registry.relations.path.read_text()


def test_explicit_peer_commit_survives_gc_without_writing_peer_repository():
    with make_team('git', lambda *args: final('done'), 'relation-peer-commit') as team:
        prepare(team)
        peer = team.roles['growth']
        commit = team._snapshot_history(peer, 'original file')
        before = {str(p.relative_to(peer.workspace)): p.read_bytes() for p in peer.workspace.rglob('*') if p.is_file()}
        receipt = batch(team, 'ceo', [create('old', 'ceo:r1.1', ['growth:source.txt@' + commit])], 'old')
        source = receipt['items'][0]['sources'][0]['ref']
        after = {str(p.relative_to(peer.workspace)): p.read_bytes() for p in peer.workspace.rglob('*') if p.is_file()}
        assert before == after
        original_path, original_short = source['snapshot_source']['cite'].rsplit('@', 1)
        assert original_path == 'growth:source.txt'
        assert original_short == git(peer.workspace, 'rev-parse', '--short=7', commit)
        assert git(peer.workspace, 'rev-parse', original_short) == commit
        record_id = receipt['items'][0]['relation'].split(':', 1)[1]
        private = team.roles['ceo'].registry.relations._load()['records'][record_id][-1]
        assert private['sources'][0]['ref']['snapshot_source']['cite'] == 'growth:source.txt@' + commit
        assert source['storage_role'] == 'ceo'
        assert source['source_role'] == 'growth'
        assert source['producer_role'] is None
        write(team, 'growth', 'later', path='source.txt')
        output = team.roles['ceo'].executor.execute('bash', {'command': 'git gc --prune=now'})
        assert not output.startswith('Error:') and '[exit code:' not in output
        _, exact = source['ref']['cite'].split(':', 1)
        path, commit = exact.rsplit('@', 1)
        read = team.roles['ceo'].executor.execute('bash', {'command': shlex.join(['git', 'show', commit + ':' + path])})
        assert read == 'growth source'


def test_busy_operation_lookup_is_bounded_and_does_not_reject_other_items():
    with make_team('git', lambda *args: final('done'), 'relation-busy') as team:
        prepare(team)
        team._set_phase('analysis')
        _, response = raw(team, 'growth', '/call', dict(tool='set_capacity_tier', args=dict(tier=3)))
        reference = response['operation']['operation_ref']
        team._set_phase('ceo')
        team.server.QUERY_TIMEOUT_SECONDS = .05
        entered, release = threading.Event(), threading.Event()

        def hold_world():
            with team.server._lock:
                entered.set()
                release.wait(5)

        holder = threading.Thread(target=hold_world)
        holder.start()
        try:
            assert entered.wait(2)
            operation = dict(item_id='busy', op='create', target=dict(operation_ref=reference),
                sources=[dict(ref=dict(cite='growth:r1.1'))], assertion='inferred', reason='Try later.')
            valid = create('valid', 'ceo:r1.1', ['growth:r1.1'])
            receipt = batch(team, 'ceo', [operation, valid], 'busy')
            assert receipt['saved'] == 1 and receipt['rejected'] == 1
            assert receipt['items'][0]['error_code'] == 'operation_unavailable'
            assert not team._lifecycle.reason
        finally:
            release.set()
            holder.join(2)
        assert not holder.is_alive()
        assert batch(team, 'ceo', [operation], 'unblocked')['saved'] == 1


def test_failed_batch_admission_does_not_claim_any_item_was_attempted(monkeypatch):
    from saas_bench.relation_registry import RelationHold
    with make_team('git', lambda *args: final('done'), 'relation-admission') as team:
        prepare(team)
        registry = team.roles['ceo'].registry.relations

        def fail(state):
            raise OSError('Cannot persist batch admission')

        monkeypatch.setattr(registry, '_publish', fail)
        with pytest.raises(RelationHold) as failure:
            batch(team, 'ceo', [create('a', 'ceo:r1.1', ['growth:r1.1']),
                               create('b', 'ceo:r1.1', ['growth:r1.1'])], 'admission')
        receipt = failure.value.receipt
        assert receipt['not_attempted'] == 2 and receipt['unknown'] == 0
        assert [item['status'] for item in receipt['items']] == ['not_attempted', 'not_attempted']
        assert not registry.path.exists()
        assert team._lifecycle.reason == 'relation_persistence_unknown'


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_nested_business_sessions_directory_keeps_existing_visibility(mode):
    with make_team(mode, lambda *args: final('done'), 'relation-nested-directory') as team:
        prepare(team)
        write(team, 'ceo', 'business session notes', path='analysis/sessions/report.md')
        cite = 'ceo:analysis/sessions/report.md' + ('@v1' if mode == 'pf' else '')
        receipt = batch(team, 'ceo', [create('nested', cite, ['ceo:r1.1'])], 'nested')
        assert receipt['saved'] == 1 and receipt['items'][0]['target']['readable']
