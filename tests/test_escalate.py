"""The public escalate name preserves the declaration protocol and saved history."""
from collections import Counter
import json

import pytest

from saas_bench.agent_capabilities import AgentCapabilities
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.pf_cli import HELP
from test_pf_candidates import query
from test_pf_relations import details
from test_preflight_team_checkpoint import clients, dispose, fast_simulator, make_team as checkpoint_team
from test_preflight_team_flow import ADVANCE, calls, final, make_team, raw
from test_receipt_outputs import artifact
from test_relation_batch import create


def test_escalate_is_the_only_public_declaration_name():
    for mode in ('git', 'pf'):
        for capabilities in (None, AgentCapabilities.round_three(mode)):
            definitions = get_bash_agent_tool_descriptions(text_registration=True,
                pf_queries=mode == 'pf', team=True, capabilities=capabilities)
            names = {tool['name'] for tool in definitions}
            assert 'relation_batch' not in names
            assert ('escalate' in names) == (capabilities is None or mode == 'pf')
    assert 'escalate' in HELP and 'relation_batch' not in HELP


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_old_name_is_an_audited_failure_and_cannot_bypass_capabilities(mode):
    with make_team(mode, lambda *args: final('done'), 'escalate-old-name',
                   capabilities=AgentCapabilities.round_three(mode)) as team:
        role = team.roles['growth']
        request = {'items': [create('blocked', 'growth:source.txt@v1', ['growth:source.txt@v1'])]}
        result = role.executor.execute('relation_batch', request, tool_call_id='old-name')
        assert result.startswith('Error:')
        expected = 'renamed to escalate' if mode == 'pf' else 'disabled'
        assert expected in result
        assert not role.registry.relations.path.exists()
        if mode == 'git':
            assert 'disabled' in role.executor.execute('escalate', request)
        audit = [json.loads(line) for line in role.audit.path.read_text().splitlines()]
        assert any(row.get('tool') == 'relation_batch' and row['status'] == 'failed' for row in audit)
        assert not role.executor.execute('write_file', {'path': 'continued.txt', 'content': 'normal work'}).startswith('Error:')


def test_third_round_escalate_items_and_followup_queries_reach_model_usage():
    turns, chosen, receipts = Counter(), {}, []

    def respond(team, role, body):
        if role == 'ceo':
            return calls(('bash', {'command': ADVANCE}))
        if role == 'ops_finance':
            return final('No changes needed.')
        turns[role] += 1
        if turns[role] == 1:
            operation = create('operation', 'growth:report.txt@v1', ['growth:source.txt@v1'])
            operation['target'] = {'operation_ref': chosen['operation']}
            missing = {'ref': {'missing': {'description': 'Earlier unsaved forecast',
                'reason': 'No captured or saved version is available.'}}}
            file = create('file', 'growth:report.txt@v1', [chosen['output'], 'ceo:peer.txt@v1'])
            file['sources'].append(missing)
            return calls(('escalate', {'items': [file,
                create('output', chosen['output'], ['growth:source.txt@v1']), operation,
                create('invalid', 'growth:report.txt@v1', ['growth:absent.txt@v1'])]}))
        if turns[role] in (2, 3):
            receipt = json.loads(body['messages'][-1]['content'])
            receipts.append(receipt)
            if turns[role] == 2:
                assert receipt['status'] == 'partial'
                assert [item['status'] for item in receipt['items']] == ['saved', 'saved', 'saved', 'rejected']
                file, output, operation, _ = receipt['items']
                assert file['sources'][-1]['ref']['source_state'] == 'missing'
                assert operation['target']['kind'] == 'operation'
                chosen.update(file=file, output_relation=output)
                return calls(('escalate', {'items': [
                    dict(item_id='revise', op='revise', relation=file['relation'], expected_revision=1,
                        sources=[{'ref': {'operation_ref': chosen['operation']}}], reason='Use the completed operation.'),
                    dict(item_id='retire', op='retire', relation=output['relation'], expected_revision=1,
                        reason='Withdraw this declaration, retaining its history.')]}))
            assert receipt['saved'] == 2 and [item['revision'] for item in receipt['items']] == [2, 2]
            assert receipt['items'][1]['record_status'] == 'retired'
            return calls(('bash', {'command': 'pf depend growth:report.txt@v1 --all-revisions --detail'}))
        if turns[role] == 4:
            result = body['messages'][-1]['content']
            assert chosen['file']['relation_revision'] in result and chosen['operation'] in result
            assert 'Earlier unsaved forecast' in result
            return calls(('bash', {'command': 'pf rdepend ceo:peer.txt@v1 --all-revisions --detail'}))
        if turns[role] == 5:
            assert chosen['file']['relation_revision'] in body['messages'][-1]['content']
            return calls(('bash', {'command': 'pf show growth:source.txt@v1 --full'}))
        assert 'original source' in body['messages'][-1]['content']
        assert 'overwritten source' not in body['messages'][-1]['content']
        return final('Use the completed operation; preserve the earlier missing forecast and exact original source.')

    with make_team('pf', respond, 'escalate-model', capabilities=AgentCapabilities.round_three('pf')) as team:
        growth = team.roles['growth']
        for name, content in [('source.txt', 'original source'), ('report.txt', 'derived report')]:
            growth.executor.execute('write_file', {'path': name, 'content': content})
        team.roles['ceo'].executor.execute('write_file', {'path': 'peer.txt', 'content': 'peer evidence'})
        growth.executor.execute('bash', {'command': './novamind-operation query "SELECT 17 AS evidence_marker"'})
        _, page = query(growth.executor, 'pf candidates --history --kind sql')
        chosen['output'] = page['items'][0]['version']
        status, result = raw(team, 'growth', '/call', {'tool': 'set_capacity_tier', 'args': {'tier': 3}})
        assert status == 200 and result['success']
        chosen['operation'] = result['operation']['operation_ref']
        growth.executor.execute('write_file', {'path': 'source.txt', 'content': 'overwritten source'})
        outcome = team.run(stop_after_day=7)
        assert outcome.reason == 'observation_end', team.failure
        usage = [json.loads(line) for line in growth.usage.path.read_text().splitlines()]
        sent = [row['body'] for row in usage if row['event'] == 'http_request']
        wire = [row['body'] for row in team.test_wire if row['role'] == 'growth']
        assert sent == wire and len(sent) == 6
        for request in sent:
            names = {tool['function']['name'] for tool in request['tools']}
            assert 'escalate' in names and 'relation_batch' not in names
            assert not any(name.startswith('text_') for name in names)
        delivered = [message for message in sent[2]['messages'] if message.get('name') == 'escalate']
        assert [json.loads(message['content']) for message in delivered] == receipts
        state = growth.registry.relations._load()
        assert sorted(len(history) for history in state['records'].values()) == [1, 2, 2]
        artifact('escalate-model.json', dict(wire=wire, usage=usage, receipts=receipts, state=state))


def test_saved_pre_rename_receipt_survives_checkpoint_without_new_declaration(tmp_path):
    original = checkpoint_team(tmp_path / 'original', 'pf', capabilities=AgentCapabilities.round_three('pf'))
    restored = None
    try:
        growth = original.roles['growth']
        growth.executor.execute('write_file', {'path': 'source.txt', 'content': 'saved source'})
        growth.executor.execute('write_file', {'path': 'report.txt', 'content': 'derived report'})
        request = {'items': [create('kept', 'growth:report.txt@v1', ['growth:source.txt@v1'])]}
        # The pre-rename registry protocol is unchanged, including its logical-call key.
        receipt = growth.registry.relations.execute(request, tool_call_id='kept-call')
        checkpoint = original.checkpoint()
        restored = MultiAgentRuntime.restore(checkpoint, tmp_path / 'restored',
            client_factory=clients(), simulator_factory=fast_simulator)
        resumed = restored.roles['growth']
        before = resumed.registry.relations.path.read_bytes()
        repeated = resumed.executor.execute('escalate', request, tool_call_id='kept-call')
        assert json.loads(repeated) == json.loads(receipt)
        assert resumed.registry.relations.path.read_bytes() == before
        page = details(restored, 'ceo', 'depend', 'growth:report.txt@v1', '--all-revisions')
        assert page['scope']['revisions']['relation_batch'] == 'all_saved'
        assert page['items'] and all(item['origin'] == 'relation_batch' for item in page['items'])
        assert restored.run(stop_after_day=7).reason == 'observation_end', restored.failure
        state = resumed.registry.relations._load()
        assert len(state['records']) == len(state['calls']) == 1
        artifact('escalate-restore.json', dict(receipt=json.loads(repeated), state=state, page=page))
    finally:
        if restored:
            dispose(restored)
        dispose(original)
