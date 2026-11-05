"""Capability-matched guidance at the public model and PF shell boundaries."""
from collections import Counter
import json
import re
import shlex
from types import SimpleNamespace

import pytest

from saas_bench.agent_capabilities import AgentCapabilities
from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.team_prompts import REGISTERED_SCRIPT_CONTRACT
from test_preflight_team_checkpoint import clients, dispose, fast_simulator, make_team as checkpoint_team
from test_preflight_team_flow import ADVANCE, calls, final, make_team
from test_receipt_outputs import artifact


def check_guidance(text, mode):
    assert not any(name in text for name in ('text_create', 'text_revise', 'text_retire', 'text_list'))
    assert 'D70' not in text
    assert 'judgments, assumptions, plans and change reasons in ordinary files' in text
    if mode == 'git':
        assert 'escalate' not in text
        assert 'callN.txt' not in text and 'snapshots its current bytes immediately' not in text
    else:
        for command in ('search', 'log', 'show', 'diff', 'blame', 'depend', 'rdepend', 'candidates', 'more'):
            assert 'pf ' + command in text
        assert 'not a dependency declaration' in text
        assert 'no weekly registration' in text
        assert 'not refreshed in PF until it ends' in text
        assert 'unknown' in text and 'missing' in text and '--role all' in text
        assert '```json\n' in text and '"op":"revise"' in text


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_standalone_prompt_matches_enabled_capabilities(mode):
    caps = AgentCapabilities.round_three(mode)
    agent = BashAgent(get_bash_agent_tool_descriptions(pf_queries=mode == 'pf', capabilities=caps),
                      SimpleNamespace(), pf=mode == 'pf', capabilities=caps)
    check_guidance(agent.system_prompt, mode)
    artifact(mode + '-standalone-system.txt', agent.system_prompt)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_initial_three_role_model_inputs_and_public_help(mode):
    with make_team(mode, lambda *args: calls(('bash', {'command': 'pf help' if mode == 'pf' else 'git log -1 --oneline'})), 'r3-guidance',
                   capabilities=AgentCapabilities.round_three(mode)) as team:
        for name, role in team.roles.items():
            action = role.agent.act('Synthetic dashboard', 0, False, {'day': 0})
            body = team.test_wire[-1]['body']
            system = body['messages'][0]['content']
            check_guidance(system, mode)
            assert system.count(REGISTERED_SCRIPT_CONTRACT) == 1
            assert 'ask_agent' in system and 'normal write permissions remain' in system
            assert 'host creates that commit after the analyst' in system
            assert 'register-daily-script' in system
            names = {item['function']['name'] for item in body['tools']}
            assert ('escalate' in names) == (mode == 'pf')
            assert not any(n.startswith('text_') for n in names)
            usage = [json.loads(line)['body'] for line in role.usage.path.read_text().splitlines()
                     if json.loads(line)['event'] == 'http_request']
            assert usage[-1] == body
            artifact(mode + '-' + name + '-initial.json', body)
            if mode == 'pf':
                delivered_help = role.executor.execute(action.tool, action.arguments)
                role.agent.act(delivered_help, 0, False, {'day': 0})
                assert team.test_wire[-1]['body']['messages'][-1]['content'] == delivered_help
                artifact(name + '-help-model.json', team.test_wire[-1]['body'])
                for command in ('pf help', 'pf log --help', 'pf show --help'):
                    help_text = role.executor.execute('bash', {'command': command})
                    check_guidance(help_text, mode)
                    artifact(name + '-' + command.replace(' ', '-') + '.txt', help_text)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_restored_three_role_model_inputs_keep_guidance_and_scripts(tmp_path, mode):
    original = checkpoint_team(tmp_path / 'original', mode, capabilities=AgentCapabilities.round_three(mode))
    restored = None
    try:
        assert original.run(stop_after_day=7).reason == 'observation_end', original.failure
        for role in original.roles.values():
            role.agent.system_prompt += '\nSTALE use text_create'
        selected = original.checkpoint()
        restored = MultiAgentRuntime.restore(selected, tmp_path / 'restored',
            client_factory=clients(), simulator_factory=fast_simulator)
        sizes = {name: len(role.usage.path.read_text().splitlines()) for name, role in restored.roles.items()}
        assert restored.run(stop_after_day=14).reason == 'observation_end', restored.failure
        for name, role in restored.roles.items():
            requests = [json.loads(line)['body'] for line in role.usage.path.read_text().splitlines()[sizes[name]:]
                        if json.loads(line)['event'] == 'http_request']
            assert requests
            for body in requests:
                check_guidance(body['messages'][0]['content'], mode)
                names = {item['function']['name'] for item in body['tools']}
                assert ('escalate' in names) == (mode == 'pf')
                assert not any(n.startswith('text_') for n in names)
            assert (role.workspace / 'script-days.txt').read_text() == '0\n7\n'
            artifact(mode + '-' + name + '-restored.json', requests)
    finally:
        if restored:
            dispose(restored)
        dispose(original)


def test_pf_prompt_examples_use_publicly_obtained_references_and_reach_model():
    turns, refs, receipts = Counter(), {}, []

    def examples(system, **values):
        blocks = re.findall(r'```json\n(.*?)\n```', system, re.S)
        result = []
        for block in blocks:
            for key, value in values.items():
                block = block.replace(key, str(value))
            result.append(json.loads(block))
        return result

    def respond(team, role, body):
        if role == 'ceo':
            return calls(('bash', {'command': ADVANCE}))
        if role == 'ops_finance':
            return final('No changes needed.')
        turns[role] += 1
        results = [m['content'] for m in body['messages'] if m['role'] == 'tool']
        system = body['messages'][0]['content']
        if turns[role] == 1:
            program = 'import json, novamind_api as n; print(json.dumps(n.infrastructure.set_capacity_tier(3)))'
            return calls(('bash', {'command': './novamind-operation query "SELECT 17 AS evidence_marker"'}),
                ('write_file', {'path': 'report.md', 'content': 'The saved evidence_marker is 17.'}),
                ('bash', {'command': './novamind-operation python-c ' + shlex.quote(program)}))
        if turns[role] == 2:
            refs['SOURCE_HANDLE'] = re.search(r'growth:query\d+@v\d+', results[0])[0]
            refs['TARGET_HANDLE'] = re.search(r'growth:report.md@v\d+', results[1])[0]
            refs['OPERATION_REF'] = re.search(r'"operation_ref"\s*:\s*"(growth:\d+)"', results[2])[1]
            return calls(('bash', {'command': 'pf log report.md'}),
                         ('bash', {'command': 'pf show ' + refs['SOURCE_HANDLE'] + ' --full'}))
        if turns[role] == 3:
            assert refs['TARGET_HANDLE'] in results[-2]
            assert 'evidence_marker' in results[-1] and '17' in results[-1]
            return calls(('escalate', examples(system, **refs)[0]))
        if turns[role] == 4:
            receipt = json.loads(results[-1])
            receipts.append(receipt)
            assert receipt['saved'] == 2
            first = receipt['items'][0]
            refs.update(RELATION_ID=first['relation'], RELATION_REVISION=first['relation_revision'])
            return calls(('escalate', examples(system, **refs)[1]))
        if turns[role] == 5:
            receipt = json.loads(results[-1])
            receipts.append(receipt)
            assert receipt['saved'] == 1 and receipt['items'][0]['revision'] == 2
            assert receipt['items'][0]['source_state'] == 'missing'
            return calls(('bash', {'command': 'pf depend ' + refs['TARGET_HANDLE'] + ' --all-revisions --detail'}),
                         ('bash', {'command': 'pf rdepend ' + refs['SOURCE_HANDLE'] + ' --detail'}))
        assert refs['RELATION_ID'] + '.2' in results[-2] and 'Earlier calculation' in results[-2]
        assert refs['TARGET_HANDLE'] in results[-1]
        return final('The saved result is 17; the earlier calculation remains missing.')

    with make_team('pf', respond, 'r3-example', capabilities=AgentCapabilities.round_three('pf')) as team:
        outcome = team.run(stop_after_day=7)
        artifact('example-attempt.json', dict(wire=team.test_wire, refs=refs, failure=team.failure,
            usage=team.roles['growth'].usage.path.read_text()))
        assert outcome.reason == 'observation_end', team.failure
        role = team.roles['growth']
        usage = [json.loads(line)['body'] for line in role.usage.path.read_text().splitlines()
                 if json.loads(line)['event'] == 'http_request']
        wire = [row['body'] for row in team.test_wire if row['role'] == 'growth']
        assert usage == wire and len(wire) == 6
        delivered = [m['content'] for m in wire[-1]['messages'] if m.get('name') == 'escalate']
        assert [json.loads(value) for value in delivered] == receipts
        artifact('example-public-requests.json', dict(refs=refs, receipts=receipts, wire=wire))


@pytest.mark.parametrize('notes,declarations', [(False, True), (True, False)])
def test_nontext_help_keeps_notes_and_declarations_independent(notes, declarations):
    from saas_bench import pf_cli, registration_prompt
    caps = AgentCapabilities(False, declarations, notes, True)
    with make_team('pf', lambda *args: final('done'), 'r3-independent', capabilities=caps) as team:
        role = team.roles['growth']
        for text in (role.agent.system_prompt, role.executor.execute('bash', {'command': 'pf help'})):
            assert (registration_prompt.PF_NOTES in text) == notes
            assert ('escalate' in text) == declarations
            assert 'text_create' not in text
        assert pf_cli.command_help(AgentCapabilities()) == pf_cli.HELP


def test_public_guide_matches_enabled_examples():
    from pathlib import Path
    from saas_bench.registration_prompt import capability_history
    from saas_bench.workspace_guides import render_guides
    guide = (Path(__file__).resolve().parents[1] / 'public/docs/workspace-history.md').read_text()
    assert '\\n' not in guide and 'escalate' not in guide
    for mode in ('git', 'pf'):
        caps = AgentCapabilities.round_three(mode)
        assert capability_history(mode == 'pf', caps) in render_guides(mode, caps)['workspace-history.md']
