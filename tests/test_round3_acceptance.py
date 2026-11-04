"""Cross-task third-round public references, rejection recovery and model delivery."""
from collections import Counter
import json
import re

from saas_bench.agent_capabilities import AgentCapabilities
from test_pf_candidates import query
from test_preflight_team_flow import ADVANCE, calls, final, make_team
from test_receipt_outputs import artifact
from test_relation_batch import create


def test_nontext_public_references_and_old_call_recovery_for_all_roles():
    turns, chosen, receipts, public = Counter(), {}, {}, {}

    def respond(team, name, body):
        turns[name] += 1
        artifact(name + '-turn-' + str(turns[name]) + '.json', body)
        role = team.roles[name]
        refs = chosen[name]
        item = create('report', name + ':report.txt@v1',
                      [refs['version'], name + ':source.txt@v1'])
        if turns[name] == 1:
            return calls(('relation_batch', {'items': [item]}), ('text_list', {}))
        outputs = [m['content'] for m in body['messages'] if m['role'] == 'tool']
        if turns[name] == 2:
            assert 'renamed to escalate' in outputs[-2]
            assert 'disabled' in outputs[-1]
            assert not role.registry.relations.path.exists()
            return calls(('escalate', {'items': [item]}))
        if turns[name] == 3:
            receipt = json.loads(outputs[-1])
            assert receipt['saved'] == 1 and receipt['rejected'] == 0
            receipts[name] = receipt
            return calls(('bash', {'command': 'pf depend ' + name + ':report.txt@v1 --detail'}))
        if turns[name] == 4:
            assert refs['version'] in outputs[-1]
            assert name + ':source.txt@v1' in outputs[-1]
            return calls(('bash', {'command': 'pf show ' + refs['content_ref'] + ' --full'}),
                         ('bash', {'command': 'pf show ' + name + ':source.txt@v1 --full'}),
                         ('read_file', {'path': 'documents/relations.json'}))
        assert turns[name] == 5
        assert 'source_marker' in outputs[-3] and '17' in outputs[-3]
        assert 'ORIGINAL ' + name in outputs[-2] and 'OVERWRITTEN ' + name not in outputs[-2]
        public[name] = outputs[-1]
        assert receipts[name]['items'][0]['relation'] in public[name]
        for path in (role.registry.relations.path, role.executor.private_root / 'tool-notes.jsonl'):
            assert role.executor.execute('read_file', {'path': str(path)}).startswith('Error:')
        if name == 'ceo':
            assert 'growth full handoff: source_marker=17' in json.dumps(body)
            assert 'ops_finance full handoff: source_marker=17' in json.dumps(body)
            for peer in ('growth', 'ops_finance'):
                exact = role.executor.execute('bash', {'command': 'pf show ' + peer + ':source.txt@v1 --full'})
                assert 'ORIGINAL ' + peer in exact and 'OVERWRITTEN ' + peer not in exact
            return calls(('bash', {'command': ADVANCE}))
        return final(name + ' full handoff: source_marker=17 from the exact public capture.')

    with make_team('pf', respond, 'round3-public-closure',
                   capabilities=AgentCapabilities.round_three('pf')) as team:
        for name, role in team.roles.items():
            for path, content in [('source.txt', 'ORIGINAL ' + name), ('report.txt', 'derived report')]:
                assert not role.executor.execute('write_file', dict(path=path, content=content)).startswith('Error:')
            for _ in range(2):
                output = role.executor.execute('bash', dict(
                    command='./novamind-operation query "SELECT 17 AS source_marker"',
                    note='Exact acquisition for the third-round report.'))
                assert '17' in output
            _, page = query(role.executor, 'pf candidates --history --kind sql --text source_marker')
            first, second = page['items']
            for field, pattern in [('event_ref', r':event[1-9][0-9]*'),
                                   ('content_ref', r':capture[1-9][0-9]*')]:
                assert re.fullmatch(name + pattern, first[field])
                assert first[field] != second[field]
            assert first['version'] == second['version']
            chosen[name] = first
            role.executor.execute('write_file', dict(path='source.txt', content='OVERWRITTEN ' + name))

        outcome = team.run(stop_after_day=7)
        artifact('outcome.json', dict(reason=outcome.reason, failure=team.failure, turns=dict(turns),
            usage={name: role.usage.path.read_text() for name, role in team.roles.items()}))
        assert outcome.reason == 'observation_end', team.failure
        for name, role in team.roles.items():
            wire = [row['body'] for row in team.test_wire if row['role'] == name]
            usage = [json.loads(line) for line in role.usage.path.read_text().splitlines()]
            assert [row['body'] for row in usage if row['event'] == 'http_request'] == wire
            state = role.registry.relations._load()
            assert len(state['records']) == len(state['calls']) == 1
            delivered = '\n'.join(m['content'] for request in wire for m in request['messages'] if m['role'] == 'tool')
            for secret in (role.identity.run_id, role.identity.world_id,
                           role.identity.agent_id, role.identity.session_id):
                assert secret not in delivered + public[name]
            assert json.loads(next(m['content'] for request in wire for m in request['messages']
                if m.get('name') == 'escalate')) == receipts[name]
            for request in wire:
                names = {tool['function']['name'] for tool in request['tools']}
                assert 'escalate' in names and 'relation_batch' not in names
                assert not any(tool.startswith('text_') for tool in names)
                assert 'text_create' not in request['messages'][0]['content']
            assert '[week-1]' in team._git(role, 'log', '--format=%s')
            artifact(name + '-round3-public-closure.json', dict(
                wire=wire, usage=usage, references=chosen[name], receipt=receipts[name], public=public[name]))
