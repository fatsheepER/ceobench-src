"""R4-1.2: actual role inputs and readable guides follow frozen capabilities."""
import json
from pathlib import Path
import re
import shlex

import httpx
from openai import OpenAI
import pytest

from saas_bench.agent_capabilities import AgentCapabilities
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.tools import TOOL_DOCS, get_tool_descriptions
from test_preflight_team_checkpoint import dispose, fast_simulator, make_team as checkpoint_team
from test_preflight_team_flow import ADVANCE, calls, completion, final, make_team, opening_response, raw
from test_receipt_outputs import artifact


ROOT = Path(__file__).resolve().parents[1]
FORMULA = '0.0225 × ln(1 + spend/5000)'
REFRESHED_DOCS = ('api/analytics.json', 'tools-reference.md', 'novamind_api/analytics.py')


def check_history(text, mode):
    assert r'\n\n' not in text[:100]
    assert 'reports/w4_calc.out' not in text
    assert 'text_create' not in text and 'text_revise' not in text
    if mode == 'git':
        assert 'pf ' not in text and 'escalate' not in text
        assert 'git -C WORKSPACE show COMMIT:PATH' in text
    else:
        assert 'git show' not in text and '@COMMIT' not in text
        assert 'not refreshed in PF until it ends' in text
        assert 'pf log growth:' in text
        assert 'pf show growth:' in text
        assert 'ask_agent' in text
        assert 'recorded' in text and 'all users' in text
        assert 'Use handles already returned' in text


def check_ledger(text, declarations, mode):
    for required in ('business_operations', 'business_effects', 'OP_HIGH', 'EFFECT_HIGH',
                     'result_json', "operation_ref = 'growth:1'", 'novamind_api.query'):
        assert required in text
    assert ('escalate' in text) == declarations
    if mode == 'git':
        assert 'pf depend' not in text


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_three_roles_read_only_their_mode_guides_and_deliver_them(mode):
    with make_team(mode, lambda *args: calls(('bash', {'command':
            'cat docs/workspace-history.md docs/business-ledger.md'})), 'r4-guides',
            capabilities=AgentCapabilities.round_three(mode)) as team:
        for name, role in team.roles.items():
            action = role.agent.act('Read the workspace guides.', 0, False, {'day': 0})
            body = team.test_wire[-1]['body']
            check_history(body['messages'][0]['content'], mode)
            assert 'docs/workspace-history.md' in body['messages'][0]['content']
            if mode == 'pf':
                description = next(t['function']['description'] for t in body['tools']
                                   if t['function']['name'] == 'escalate')
                assert '@COMMIT' not in description and 'ROLE:rN.K' not in description
                assert 'ROLE:HANDLE@vK' in description
            history = (role.workspace / 'docs/workspace-history.md').read_text()
            ledger = (role.workspace / 'docs/business-ledger.md').read_text()
            check_history(history, mode)
            check_ledger(ledger, mode == 'pf', mode)
            output = role.executor.execute(action.tool, action.arguments)
            role.agent.act(output, 0, False, {'day': 0})
            delivered = team.test_wire[-1]['body']
            assert history in delivered['messages'][-1]['content']
            assert ledger in delivered['messages'][-1]['content']
            usage = [json.loads(line)['body'] for line in role.usage.path.read_text().splitlines()
                     if json.loads(line)['event'] == 'http_request']
            assert usage[-1] == delivered
            artifact(mode + '-' + name + '-guides-and-model.json', delivered)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_restored_guides_replace_old_mixed_material_without_changing_checkpoint(tmp_path, mode):
    original = checkpoint_team(tmp_path / 'original', mode, capabilities=AgentCapabilities.round_three(mode))
    restored = None
    try:
        assert original.run(stop_after_day=7).reason == 'observation_end', original.failure
        for role in original.roles.values():
            for name in ('workspace-history.md', 'business-ledger.md'):
                (role.workspace / 'docs' / name).write_text('STALE mixed Git PF text_create escalate\n')
            for name in REFRESHED_DOCS:
                (role.workspace / 'docs' / name).write_text('# STALE development formula\n')
        selected = original.checkpoint()
        def reader_client(role):
            count = 0
            def respond(request):
                nonlocal count
                body = json.loads(request.content)
                if response := opening_response(role, body):
                    return response
                count += 1
                command = ('cat docs/workspace-history.md docs/business-ledger.md; '
                    './novamind-operation python-c ' + shlex.quote(
                        "import json, novamind_api as n; "
                        "doc = next(d for d in json.load(open('docs/api/analytics.json')) if d['name'] == 'set_targeted_dev_spend'); "
                        "print('DOC_FORMULA: ' + doc['description']); "
                        "print(n.analytics.set_targeted_dev_spend({'E1': 5000}))"))
                action = (calls(('bash', {'command': command}))
                          if count == 1 else calls(('bash', {'command': ADVANCE})) if role == 'ceo'
                          else final('Read the refreshed guides.'))
                return completion(action, body.get('stream', False))
            return OpenAI(api_key='OFFLINE-ONLY', base_url='https://opencode.ai/zen/go/v1', max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        restored = MultiAgentRuntime.restore(selected, tmp_path / 'restored',
            client_factory=reader_client, simulator_factory=fast_simulator)
        cutoffs = {name: len(role.usage.path.read_text().splitlines()) for name, role in restored.roles.items()}
        assert restored.run(stop_after_day=14).reason == 'observation_end', restored.failure
        for name, role in restored.roles.items():
            history = (role.workspace / 'docs/workspace-history.md').read_text()
            ledger = (role.workspace / 'docs/business-ledger.md').read_text()
            check_history(history, mode)
            check_ledger(ledger, mode == 'pf', mode)
            public_docs = {file: (role.workspace / 'docs' / file).read_text() for file in REFRESHED_DOCS}
            artifact(mode + '-' + name + '-restored-public-docs.json', public_docs)
            for file, text in public_docs.items():
                assert text == (ROOT / 'public/docs' / file).read_text()
                assert (Path(selected) / 'roles' / name / 'docs' / file).read_text().startswith('# STALE')
            for file in ('workspace-history.md', 'business-ledger.md'):
                assert (Path(selected) / 'roles' / name / 'docs' / file).read_text().startswith('STALE')
            requests = [json.loads(line)['body'] for line in role.usage.path.read_text().splitlines()[cutoffs[name]:]
                        if json.loads(line)['event'] == 'http_request']
            assert requests
            for body in requests:
                check_history(body['messages'][0]['content'], mode)
            assert any(history in m['content'] for b in requests for m in b['messages'])
            assert any(ledger in m['content'] for b in requests for m in b['messages'])
            assert any('DOC_FORMULA:' in m['content'] and FORMULA in m['content'] and 'quality_effect' in m['content']
                       for b in requests for m in b['messages'])
            artifact(mode + '-' + name + '-restored-guides.json', requests)
    finally:
        if restored:
            dispose(restored)
        dispose(original)


@pytest.mark.parametrize('pf', [False, True])
@pytest.mark.parametrize('texts', [False, True])
def test_escalate_description_only_advertises_mode_endpoints(pf, texts):
    caps = AgentCapabilities(texts, True, True, True)
    definitions = get_bash_agent_tool_descriptions(team=True, pf_queries=pf, capabilities=caps)
    description = next(t['description'] for t in definitions if t['name'] == 'escalate')
    if pf:
        assert all('path@COMMIT' not in t['description'] and 'path@week-N' not in t['description'] for t in definitions)
    assert ('ROLE:HANDLE@vK' in description) == pf
    assert ('ROLE:path@COMMIT' in description) == (not pf)
    assert ('ROLE:rN.K' in description) == texts
    if pf and not texts:
        assert 'require an explicit @vK' in description and 'bare file path' not in description


def test_development_formula_in_definition_static_docs_and_public_receipt():
    definition = next(t for t in get_tool_descriptions() if t['name'] == 'set_targeted_dev_spend')
    static = json.loads((ROOT / 'src/saas_bench/tool_docs.json').read_text())
    for text in (TOOL_DOCS['set_targeted_dev_spend']['mechanics']['quality_bonus'],
                 definition['description'], json.dumps(static, ensure_ascii=False),
                 (ROOT / 'public/docs/tools-reference.md').read_text(),
                 json.dumps(json.loads((ROOT / 'public/docs/api/analytics.json').read_text()), ensure_ascii=False)):
        assert FORMULA in text
        assert '0.005 × log(1 + spend/1000)' not in text
    with make_team('pf', lambda *args: calls(('bash', {'command': 'true'})), 'r4-dev',
                   capabilities=AgentCapabilities.round_three('pf')) as team:
        status, receipt = raw(team, 'ops_finance', '/call',
            {'tool': 'set_targeted_dev_spend', 'args': {'targeted_spend': {'E1': 5000}}})
        assert status == 200 and receipt['success'], receipt
        assert FORMULA in receipt['data']['quality_effect']
        assert 'USD per day' in receipt['data']['quality_effect']
        assert team.server.tools.config.targeted_dev_spend == {'E1': 5000}
        artifact('development-receipt.json', receipt)


def test_peer_example_reaches_model_and_guides_a_revision_question():
    from collections import Counter
    turns, refs = Counter(), {}

    def respond(team, role, body):
        if role == 'ceo':
            return calls(('bash', {'command': ADVANCE}))
        if role == 'growth':
            if not any(m['role'] == 'user' and 'REVIEW_PEER_FORECAST' in m['content'] for m in body['messages']):
                return final('Use the saved forecast and its exact version.')
            turns['review'] += 1
            if turns['review'] == 1:
                return calls(('write_file', {'path': 'forecast.md', 'content': 'Revised acquisition assumption: 40 leads/day.'}))
            if turns['review'] == 2:
                return calls(('bash', {'command': 'pf rdepend ' + refs['peer'] + ' --detail'}))
            assert refs['target'] in body['messages'][-1]['content']
            return final('The saved cash plan cites the earlier forecast. The revised assumption is 40 leads/day; review the cash plan.')
        turns[role] += 1
        if turns[role] == 1:
            examples = re.findall(r'```json\n(.*?)\n```', body['messages'][0]['content'], re.S)
            example = examples[-1].replace('growth:forecast.md@v3', refs['peer']).replace(
                'ops_finance:cash-plan.md@v2', refs['target'])
            return calls(('escalate', json.loads(example)))
        if turns[role] == 2:
            receipt = json.loads(body['messages'][-1]['content'])
            assert receipt['saved'] == 1 and receipt['items'][0]['sources'][0]['ref']['ref']['cite'] == refs['peer']
            return calls(('ask_agent', {'role': 'growth', 'message': 'REVIEW_PEER_FORECAST: check the assumption used by ' + refs['target']}))
        if turns[role] == 3:
            assert '40 leads/day' in body['messages'][-1]['content']
            return calls(('bash', {'command': 'pf show ' + refs['peer'] + ' --full'}),
                         ('bash', {'command': 'pf log growth:forecast.md'}))
        results = [m['content'] for m in body['messages'] if m['role'] == 'tool']
        assert '20 leads/day' in results[-2] and '40 leads/day' not in results[-2]
        assert 'growth:forecast.md@v4' in results[-1]
        return final('The retained source is 20 leads/day; Growth supplied a 40 leads/day revision for review.')

    with make_team('pf', respond, 'r4-peer-example', capabilities=AgentCapabilities.round_three('pf')) as team:
        for content in ('10 leads/day', '15 leads/day', '20 leads/day'):
            receipt = team.roles['growth'].executor.execute('write_file', {'path': 'forecast.md', 'content': content})
        refs['peer'] = re.search(r'growth:forecast.md@v[0-9]+', receipt)[0]
        for content in ('Draft cash plan.', 'Cash plan assumes 20 leads/day.'):
            receipt = team.roles['ops_finance'].executor.execute('write_file', {'path': 'cash-plan.md', 'content': content})
        refs['target'] = re.search(r'ops_finance:cash-plan.md@v[0-9]+', receipt)[0]
        assert team.run(stop_after_day=7).reason == 'observation_end', team.failure
        assert turns['review'] == 3
        for name, role in team.roles.items():
            usage = [json.loads(line)['body'] for line in role.usage.path.read_text().splitlines()
                     if json.loads(line)['event'] == 'http_request']
            wire = [row['body'] for row in team.test_wire if row['role'] == name]
            assert usage == wire
            artifact(name + '-peer-example.json', dict(refs=refs, wire=wire))


@pytest.mark.parametrize('texts', [False, True])
def test_disabled_declarations_do_not_leak_into_help_or_guides(texts):
    from saas_bench.pf_cli import command_help
    from saas_bench.workspace_guides import render_guides
    caps = AgentCapabilities(texts, False, False, False)
    assert 'escalate' not in command_help(caps)
    for mode in ('git', 'pf'):
        guides = render_guides(mode, caps)
        assert 'escalate' not in guides['workspace-history.md']
        assert 'escalate' not in guides['business-ledger.md']
        check_ledger(guides['business-ledger.md'], False, mode)


def test_empty_reverse_lookup_is_about_recorded_relationships():
    from saas_bench.pf_render import render_dependents
    with make_team('pf', lambda *args: calls(('bash', {'command': 'true'})), 'r4-empty',
                   capabilities=AgentCapabilities.round_three('pf')) as team:
        role = team.roles['growth']
        receipt = role.executor.execute('write_file', {'path': 'uncited.txt', 'content': 'Uncited result.'})
        handle = re.search(r'growth:uncited.txt@v[0-9]+', receipt)[0]
        output = role.executor.execute('bash', {'command': 'pf rdepend ' + handle})
        assert 'No saved relationships in this scope' in output and 'all users' in output
        assert 'registered text' not in output
        artifact('empty-rdepend.txt', output)
    legacy = render_dependents(dict(root={'version': 'growth:uncited.txt@v1', 'what': 'file uncited.txt'}, items=[], detail=True))
    assert 'No recorded referrers' in legacy and 'registered text' not in legacy


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_public_development_receipt_reaches_all_roles_without_changing_formula(mode):
    import math
    from saas_bench.database import get_global_state
    from saas_bench.simulation import Simulator
    command = "./novamind-operation python-c \"import novamind_api as n; print(n.analytics.set_targeted_dev_spend({'E1': 5000}))\""
    with make_team(mode, lambda *args: calls(('bash', {'command': command})), 'r4-dev-model',
                   capabilities=AgentCapabilities.round_three(mode)) as team:
        for name, role in team.roles.items():
            action = role.agent.act('Set targeted development spend.', 0, False, {'day': 0})
            output = role.executor.execute(action.tool, action.arguments)
            assert FORMULA in output and 'USD per day' in output
            role.agent.act(output, 0, False, {'day': 0})
            body = team.test_wire[-1]['body']
            assert output == body['messages'][-1]['content']
            recorded = [json.loads(line)['body'] for line in role.usage.path.read_text().splitlines()
                        if json.loads(line)['event'] == 'http_request'][-1]
            assert body == recorded
            artifact(mode + '-' + name + '-development-model.json', body)
        simulator = Simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        before = get_global_state(team.server.conn, 'q_group_bonus_E1', 0.0)
        simulator._update_global_state({'spend_operations': 0, 'spend_development': 0})
        after = get_global_state(team.server.conn, 'q_group_bonus_E1', 0.0)
        assert after - before == pytest.approx(0.0225 * math.log(2))


def test_runtime_guide_replacement_does_not_follow_file_or_directory_symlinks(tmp_path):
    from saas_bench.workspace_guides import write_guides
    workspace, external = tmp_path / 'workspace', tmp_path / 'external'
    workspace.mkdir()
    external.mkdir()
    docs = workspace / 'docs'
    docs.mkdir()
    target = external / 'source.md'
    target.write_text('Original external evidence')
    guide = docs / 'workspace-history.md'
    guide.symlink_to(target)
    write_guides(workspace, 'git', AgentCapabilities.round_three('git'))
    assert not guide.is_symlink() and 'git log' in guide.read_text()
    assert target.read_text() == 'Original external evidence'
    for path in docs.iterdir():
        path.unlink()
    docs.rmdir()
    docs.symlink_to(external, target_is_directory=True)
    with pytest.raises(OSError):
        write_guides(workspace, 'git', AgentCapabilities.round_three('git'))
    assert list(external.iterdir()) == [target]
