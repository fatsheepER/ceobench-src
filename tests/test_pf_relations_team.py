from collections import Counter
from contextlib import closing
import json

from saas_bench.multi_agent import MultiAgentRuntime
from test_pf_candidates import day, query
from test_pf_relations import cli
from test_preflight_team_evidence import write
from test_preflight_team_flow import ADVANCE, calls, final, make_team
from test_receipt_outputs import artifact
from test_relation_batch import batch, create


def business_reads(store):
    with closing(store.connect()) as conn:
        return [row[0] for row in conn.execute("SELECT event_id FROM requests WHERE (query_id IS NOT NULL OR json_extract(request,'$.kind')='public_http') AND json_extract(request,'$.role')='growth' ORDER BY rowid")]


def test_relation_results_reach_model_wire_and_are_used_without_replay():
    turns = Counter()
    chosen = {}

    def respond(team, role, body):
        if role == 'ceo':
            return calls(('bash', {'command': ADVANCE}))
        if role == 'ops_finance':
            return final('No changes needed.')
        turns[role] += 1
        results = [message['content'] for message in body['messages'] if message['role'] == 'tool']
        if turns[role] == 1:
            assert '--all-revisions' in body['messages'][0]['content']
            return calls(('bash', {'command': 'pf depend growth:F1.txt@v1'}))
        if turns[role] == 2:
            assert chosen['source'] in results[-1] and chosen['relation'] in results[-1]
            return calls(('bash', {'command': 'pf rdepend ' + chosen['source']}))
        if turns[role] == 3:
            assert 'growth:F1.txt@v1' in results[-1] and 'author_declared' in results[-1]
            return calls(('bash', {'command': 'pf show ' + chosen['source'] + ' --full'}))
        assert 'source_marker' in results[-1] and '17' in results[-1]
        return final('The saved source_marker is 17. F1 retains this exact source; no new query was necessary.')

    with make_team('pf', respond, 'relation-model-wire') as team:
        growth = team.roles['growth']
        day(growth.store, 0)
        cli(team, 'growth', './novamind-operation', 'query', 'SELECT 17 AS source_marker')
        _, candidates = query(growth.executor, 'pf candidates --history --kind sql')
        chosen['source'] = candidates['items'][0]['version']
        write(team, 'growth', 'source_marker=17', 'F1.txt')
        receipt = batch(team, 'growth', [create('f1', 'growth:F1.txt@v1', [chosen['source']])], 'model-source')
        chosen['relation'] = receipt['items'][0]['relation_revision']
        before = business_reads(growth.store)
        assert len(before) == 1
        result = team.run(stop_after_day=7)
        assert result.reason == 'observation_end', team.failure
        assert turns['growth'] == 4
        assert business_reads(growth.store) == before
        wire = [row['body'] for row in team.test_wire if row['role'] == 'growth']
        usage = [json.loads(line) for line in growth.usage.path.read_text().splitlines()]
        assert [row['body'] for row in usage if row['event'] == 'http_request'] == wire
        artifact('relations-model-wire.json', dict(wire=wire, usage=usage, chosen=chosen, source_calls_before=before, source_calls_after=business_reads(growth.store), result=result.reason))


def test_relation_checkpoint_restores_frozen_pages_and_discards_future(tmp_path):
    from test_preflight_team_checkpoint import clients, dispose, fast_simulator, make_team as checkpoint_team
    original = checkpoint_team(tmp_path / 'original', 'pf')
    restored = None
    try:
        write(original, 'growth', 'old exact input', 'source.txt')
        write(original, 'growth', 'product', 'target.txt')
        first = batch(original, 'growth', [create('kept', 'growth:target.txt@v1', ['growth:source.txt@v1'])], 'kept')['items'][0]
        reader = original.roles['ceo']
        _, page = query(reader.executor, 'pf depend growth:target.txt@v1')
        cursor = page['detail_cursor']
        checkpoint = original.checkpoint()
        batch(original, 'growth', [dict(item_id='retire', op='retire', relation=first['relation'],
            expected_revision=first['revision'], reason='This happens after the selected checkpoint.')], 'discarded')
        _, later = query(reader.executor, 'pf depend growth:target.txt@v1 --all-revisions')
        original_cursors = reader.store.load_state('pf_cursors')
        restored = MultiAgentRuntime.restore(checkpoint, tmp_path / 'restored', client_factory=clients(), simulator_factory=fast_simulator)
        resumed = restored.roles['ceo']
        output, frozen = query(resumed.executor, 'pf more ' + cursor)
        assert first['relation_revision'] in output and 'retired' not in output
        assert frozen['cutoff'] == page['cutoff']
        absent = resumed.executor.execute('bash', {'command': 'pf more ' + later['detail_cursor']})
        assert '[exit code:' in absent and 'cursor' in absent
        current, _ = query(resumed.executor, 'pf depend growth:target.txt@v1')
        assert first['relation_revision'] in current
        assert restored.run(stop_after_day=7).reason == 'observation_end', restored.failure
        assert reader.store.load_state('pf_cursors') == original_cursors
        artifact('relations-checkpoint.json', dict(checkpoint=str(checkpoint), kept=page, discarded=later,
            restored=frozen, original_relations=original.roles['growth'].registry.relations._load()))
    finally:
        if restored:
            dispose(restored)
        dispose(original)
