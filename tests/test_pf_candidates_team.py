from collections import Counter
import json
import re
import shlex

import pytest

from saas_bench.pf_queries import PFQueries
from saas_bench.sql_evidence import SQLEvidenceStore
from saas_bench.text_registry import TextRegistry
from test_pf_candidates import day, query
from test_preflight_team_evidence import send, write
from test_preflight_team_flow import ADVANCE, calls, final, make_team
from test_receipt_outputs import artifact
from test_sql_evidence import identity
from test_text_registry import captured, workspace


def test_candidate_roles_private_delivery_and_current_request(monkeypatch):
    with make_team('pf', lambda *args: final('done'), 'candidate-permissions') as team:
        for role, runtime in team.roles.items():
            day(runtime.store, 0)
            write(team, role, role + ' exact content', 'same.txt')
        ceo = team.roles['ceo']
        ceo_read = ceo.executor.execute('bash', {'command': 'pf show growth:same.txt@v1 --full'})
        send(team, 'ceo', [ceo_read])
        _, page = query(ceo.executor, 'pf candidates --role growth --kind file --text same.txt')
        assert page['total_in_scope'] == 1
        item = page['items'][0]
        assert item['source_role'] == item['producer_role'] == item['storage_role'] == 'growth'
        assert item['model_delivery']['current']['status'] == 'full'
        first_request = page['model_delivery_scope']['current_request']
        send(team, 'ceo', [ceo.executor.execute('bash', {'command': 'pf show ceo:same.txt@v1 --full'})])
        _, later = query(ceo.executor, 'pf candidates --role growth --kind file --text same.txt')
        assert later['model_delivery_scope']['current_request'] != first_request
        assert later['items'][0]['model_delivery']['historical']['status'] == 'full'
        assert later['items'][0]['model_delivery']['current']['status'] == 'unknown'
        _, peer = query(team.roles['ops_finance'].executor, 'pf candidates --role growth --kind file --text same.txt')
        assert peer['items'][0]['model_delivery']['historical']['status'] == 'unknown'
        assert 'model_request_wire' not in json.dumps(peer)
        private_event = ceo.usage.last_request_event
        denied = team.roles['ops_finance'].executor.execute('bash', {'command': 'pf candidates --history --role all --event ' + private_event})
        assert '[exit code: 1]' in denied and 'inaccessible' in denied
        _, own = query(ceo.executor, 'pf candidates --kind file --text same.txt')
        assert {r['source_role'] for r in own['items']} == {'ceo'}
        _, all_roles = query(ceo.executor, 'pf candidates --role all --kind file --text same.txt')
        assert {r['source_role'] for r in all_roles['items']} == {'ceo', 'growth', 'ops_finance'}
        monkeypatch.setattr('saas_bench.role_policy.READABLE_ROLES', dict(ceo=('ceo',), growth=('growth',), ops_finance=('ops_finance',)))
        failed = ceo.executor.execute('bash', {'command': 'pf more ' + page['detail_cursor']})
        assert '[exit code: 1]' in failed and 'inaccessible' in failed
        assert 'growth exact content' not in failed
        _, restricted = query(ceo.executor, 'pf candidates --role all --kind file --text same.txt')
        assert {r['source_role'] for r in restricted['items']} == {'ceo'}


def test_candidate_model_can_discover_read_and_declare_returned_source():
    turns = Counter()
    chosen = {}

    def respond(team, role, body):
        if role == 'ceo':
            return calls(('bash', {'command': ADVANCE}))
        if role == 'ops_finance':
            return final('No further action.')
        assert 'pf candidates' in body['messages'][0]['content']
        turns[role] += 1
        if turns[role] == 1:
            return calls(('bash', {'command': 'pf candidates --history --kind sql'}))
        results = [m['content'] for m in body['messages'] if m['role'] == 'tool']
        if turns[role] == 2:
            assert 'not proof of use' in results[-1]
            for label in ('q0', 'q1', 'q2'):
                chosen[label] = next(re.match(r'(growth:query\d+@v\d+)', line)[1] for line in results[-1].splitlines() if ' AS ' + label + ' |' in line)
            return calls(*[('bash', {'command': 'pf show ' + chosen[label] + ' --full'}) for label in ('q1', 'q2')])
        if turns[role] == 3:
            assert 'q1' in results[-2] and 'q2' in results[-1]
            code = "from pathlib import Path; Path('F1.txt').write_text('q1=1'); Path('F2.txt').write_text('q2=2')"
            return calls(('bash', {'command': 'python -c ' + shlex.quote(code)}))
        if turns[role] == 4:
            assert 'F1.txt@v1' in results[-1] and 'F2.txt@v1' in results[-1]
            return calls(('escalate', {'items': [dict(item_id=label, op='create', target={'cite': 'growth:F' + str(i) + '.txt@v1'},
                sources=[dict(ref={'cite': chosen[label]})], assertion='author_declared', reason='Keep the selected source.') for i, label in enumerate(('q1', 'q2'), 1)]}))
        if turns[role] == 5:
            result = json.loads(results[-1])
            assert result['saved'] == 2
            chosen['relation'] = result['items'][0]['relation_revision']
            return calls(('escalate', {'items': [dict(item_id='earlier', op='revise', relation=chosen['relation'].rsplit('.', 1)[0],
                expected_revision=chosen['relation'], sources=[dict(ref={'cite': chosen[label]}) for label in ('q1', 'q0')],
                reason='Add the earlier observed source without replacing the retained query.')]}))
        result = json.loads(results[-1])
        assert result['saved'] == 1 and result['items'][0]['sources'][0]['ref']['readable']
        return final('F1 uses the selected saved query. Other possible sources were not declared.')

    with make_team('pf', respond, 'candidate-model') as team:
        runtime = team.roles['growth']
        day(runtime.store, 0)
        for i in range(3):
            sql = runtime.executor.execute('bash', {'command': f'./novamind-operation query "SELECT {i} AS q{i}"'})
            assert f'q{i}' in sql
        result = team.run(stop_after_day=7)
        artifact('candidate-debug.json', dict(reason=result.reason, failure=team.failure, wire=team.test_wire, usage=runtime.usage.path.read_text()))
        assert result.reason == 'observation_end', team.failure
        assert turns['growth'] == 6
        wire = [row['body'] for row in team.test_wire if row['role'] == 'growth']
        usage = [json.loads(line) for line in runtime.usage.path.read_text().splitlines()]
        assert [row['body'] for row in usage if row['event'] == 'http_request'] == wire
        records = runtime.registry.relations._load()['records']
        assert len(records) == 2 and sorted(len(history) for history in records.values()) == [1, 2]
        artifact('candidate-model.json', dict(wire=wire, usage=usage, relations=records))


def test_candidate_checkpoint_copy_preserves_cursor_and_drops_later_history(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    day(store, 7)
    executor.execute('write_file', {'path': 'kept.txt', 'content': 'before checkpoint'})
    _, page = query(executor, 'pf candidates --kind file --text kept.txt')
    cursor = page['detail_cursor']
    snapshot = tmp_path / 'saved.sqlite'
    receipt = store.snapshot(snapshot)
    executor.execute('write_file', {'path': 'discarded.txt', 'content': 'after checkpoint'})
    _, future = query(executor, 'pf candidates --kind file --text discarded.txt')
    discarded_cursor = future['detail_cursor']
    restored = SQLEvidenceStore(snapshot, identity('resumed', parent_branch='prefix', fork_seq=receipt['cutoff'], capture_scope='execution'))
    restored_queries = PFQueries(TextRegistry(workspace, 'pf', restored, sim_day=lambda: 7))
    details = restored_queries.execute('pf_more', {'cursor': cursor})
    assert 'kept.txt' in details and 'discarded.txt' not in details
    assert restored_queries.last_answer['cutoff'] == page['cutoff']
    with pytest.raises(ValueError, match='cursor'):
        restored_queries.execute('pf_more', {'cursor': discarded_cursor})
    fresh = restored_queries.answer('pf_candidates', {'history': True, 'kind': 'file'})
    assert not any(item['name'] == 'discarded.txt' for item in fresh['items'])
    assert 'discarded.txt' in executor.execute('bash', {'command': 'pf candidates --kind file --text discarded.txt'})


@pytest.mark.parametrize('direct_detail', [False, True])
def test_candidate_detail_pages_keep_long_request_and_current_capture_unknown(workspace, tmp_path, direct_detail):
    store, registry, executor = captured(workspace, tmp_path)
    day(store, 7)
    body = 'LONG_DETAIL ' * 6500
    executor.execute('write_file', {'path': 'long.txt', 'content': body})
    output, page = query(executor, 'pf candidates --kind file --text long.txt')
    assert 'LONG_DETAIL' not in output
    assert len(output) < 4000
    assert page['items'][0]['model_delivery']['historical']['status'] == 'unknown'
    first, detail = query(executor, 'pf candidates --kind file --text long.txt --detail' if direct_detail else 'pf more ' + page['detail_cursor'])
    fragments = [detail['detail_text']]
    assert len(first) < 25000 and detail['has_more']
    while detail['next_cursor']:
        output, detail = query(executor, 'pf more ' + detail['next_cursor'])
        assert len(output) < 25000
        fragments.append(detail['detail_text'])
    data = json.loads(''.join(fragments))
    assert data[0]['request']['arguments']['content'] == body
    assert data[0]['model_delivery']['current']['status'] == 'unknown'


def test_candidate_partial_source_delivery_is_not_full(workspace, tmp_path):
    from test_receipt_outputs import model_receives
    store, registry, executor = captured(workspace, tmp_path)
    day(store, 7)
    executor.execute('write_file', {'path': 'partial.txt', 'content': 'visible\nhidden\n'})
    partial = executor.execute('read_file', {'path': 'partial.txt', 'limit': 1})
    model_receives(store, workspace, partial)
    _, page = query(executor, 'pf candidates --kind file --text partial.txt')
    assert any(item['model_delivery']['current']['status'] == 'partial_or_representation' for item in page['items'])
    assert all(item['model_delivery']['current']['status'] != 'full' for item in page['items'])


def test_candidate_real_team_checkpoint_restores_frozen_inventory(tmp_path):
    from saas_bench.multi_agent import MultiAgentRuntime
    from test_preflight_team_checkpoint import clients, dispose, fast_simulator, make_team as checkpoint_team
    original = checkpoint_team(tmp_path / 'original', 'pf')
    restored = None
    try:
        runtime = original.roles['growth']
        write(original, 'growth', 'kept value', 'kept.txt')
        _, page = query(runtime.executor, 'pf candidates --history --kind file --text kept.txt')
        checkpoint = original.checkpoint()
        write(original, 'growth', 'discarded value', 'discarded.txt')
        _, later = query(runtime.executor, 'pf candidates --history --kind file --text discarded.txt')
        original_cursors = runtime.store.load_state('pf_cursors')
        restored = MultiAgentRuntime.restore(checkpoint, tmp_path / 'restored', client_factory=clients(), simulator_factory=fast_simulator)
        resumed = restored.roles['growth']
        detail, frozen = query(resumed.executor, 'pf more ' + page['detail_cursor'])
        assert frozen['cutoff'] == page['cutoff'] and 'kept.txt' in detail
        assert '[exit code: 1]' in resumed.executor.execute('bash', {'command': 'pf more ' + later['detail_cursor']})
        _, after = query(resumed.executor, 'pf candidates --history --kind file --text discarded.txt')
        assert after['total_in_scope'] == 0
        assert restored.run(stop_after_day=7).reason == 'observation_end', restored.failure
        assert runtime.store.load_state('pf_cursors') == original_cursors
        artifact('candidate-team-restore.json', dict(selected=str(checkpoint), kept=page, discarded=later, restored=frozen))
    finally:
        if restored:
            dispose(restored)
        dispose(original)


def test_candidate_repeated_peer_file_reads_keep_acquisition_and_storage_roles():
    with make_team('pf', lambda *args: final('done'), 'candidate-peer-file') as team:
        for runtime in team.roles.values():
            day(runtime.store, 0)
        write(team, 'growth', 'peer saved bytes\r\n', 'peer.txt')
        ceo = team.roles['ceo']
        for _ in range(2):
            result = ceo.executor.execute('read_file', {'path': str(team.roles['growth'].workspace / 'peer.txt')})
            assert 'peer saved bytes' in result
        _, page = query(ceo.executor, 'pf candidates --kind file --text peer.txt')
        artifact('candidate-peer-file.json', page)
        assert len(page['items']) == 2
        assert len({item['event_ref'] for item in page['items']}) == 2
        assert len({item['version'] for item in page['items']}) == 1
        assert all(item['source_role'] == item['producer_role'] == 'growth' for item in page['items'])
        assert all(item['initiator_role'] == item['storage_role'] == 'ceo' for item in page['items'])
        assert all(item['capture']['derived_from'] and item['capture']['transformation'] == 'decode_universal_newlines' for item in page['items'])

        for item in page['items']:
            from saas_bench.evidence_handles import resolve_ref
            assert ceo.store.get_content(resolve_ref(ceo.store, 'capture', item['content_ref']), public=True)[1] == b'peer saved bytes\n'
            assert ceo.store.get_content(resolve_ref(ceo.store, 'capture', item['capture']['derived_from']), public=True)[1] == b'peer saved bytes\r\n'
