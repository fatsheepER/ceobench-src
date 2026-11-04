"""Public third-round capability contracts and legacy compatibility."""
import json
import shlex
from collections import Counter

import pytest

from saas_bench.agent_capabilities import AgentCapabilities
from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from test_preflight_team_flow import final, make_team
from test_preflight_team_flow import ADVANCE, calls, raw
from test_receipt_outputs import artifact
from test_preflight_team_batch import frozen


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_third_round_definitions_and_direct_agent(mode):
    capabilities = AgentCapabilities.round_three(mode)
    definitions = get_bash_agent_tool_descriptions(
        text_registration=True, pf_queries=mode == 'pf', team=True, capabilities=capabilities)
    names = {item['name'] for item in definitions}
    assert not names & {'text_create', 'text_revise', 'text_retire', 'text_list'}
    assert ('relation_batch' in names) == (mode == 'pf')
    for item in definitions:
        if item['name'] in ('bash', 'write_file', 'edit_file'):
            assert ('note' in item['parameters']['properties']) == (mode == 'pf')
    from types import SimpleNamespace
    agent = BashAgent(definitions, SimpleNamespace(), pf=mode == 'pf', capabilities=capabilities)
    assert 'text_create' not in agent.system_prompt


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_team_capabilities_reject_before_side_effects_and_continue(mode):
    caps = AgentCapabilities.round_three(mode)
    with make_team(mode, lambda *args: final('done'), 'capabilities-' + mode,
                   capabilities=caps.to_dict()) as team:
        for role in team.roles.values():
            executor = role.executor
            assert executor.text_registry is not None
            assert executor.execute('text_create', {}).startswith('Error:')
            assert not executor.text_registry.path.exists()
            assert 'text_create' not in role.agent.system_prompt
            if mode == 'git':
                assert executor.execute('relation_batch', {}).startswith('Error:')
                for tool, args in (
                    ('bash', dict(command='touch forbidden.txt')),
                    ('write_file', dict(path='forbidden.txt', content='blocked')),
                    ('edit_file', dict(path='forbidden.txt', old_string='a', new_string='b')),
                ):
                    assert executor.execute(tool, dict(args, note='disabled')).startswith('Error:')
                assert not (role.workspace / 'forbidden.txt').exists()
                assert not (role.workspace / 'tool-outputs').exists()
                assert not (executor.private_root / 'tool-notes.jsonl').exists()
                assert not executor.text_registry.relations.path.exists()
            result = executor.execute('write_file', dict(path='allowed.txt', content='normal work'))
            assert not result.startswith('Error:'), result
            assert 'normal work' in executor.execute('read_file', dict(path='allowed.txt'))


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_disabled_calls_reach_model_and_normal_week_still_finishes(mode):
    turns = Counter()

    def respond(team, role, body):
        turns[role] += 1
        results = [m['content'] for m in body['messages'] if m['role'] == 'tool']
        if turns[role] == 1:
            actions = [('text_list', {})]
            if mode == 'git':
                actions += [('relation_batch', {'items': []}),
                            ('bash', dict(command='touch forbidden', note='disabled'))]
            return calls(*actions)
        if turns[role] == 2:
            assert results and all(result.startswith('Error:') for result in results)
            return calls(('write_file', dict(path='MEMORY.md', content=role + ' retained fact')),
                         ('bash', dict(command='./novamind-operation query "SELECT 17 AS useful_fact"')))
        assert any('17' in result for result in results)
        if role == 'ceo':
            if turns[role] == 3:
                return calls(('ask_analyst', dict(role='growth', message='Confirm the retained fact.')))
            return calls(('bash', dict(command=ADVANCE)))
        return final(role + ' full handoff: useful_fact=17; files retain the evidence.')

    with make_team(mode, respond, 'capability-week', capabilities=AgentCapabilities.round_three(mode)) as team:
        for name, role in team.roles.items():
            status, result = raw(team, name, '/daily-scripts', dict(name='observe.py', content=
                "import novamind_api as nm\nopen('script.txt','w').write(str(nm.vars.current_day))\nprint('SCRIPT-OBSERVATION')"))
            assert status == 200 and result['success']
        outcome = team.run(stop_after_day=7)
        assert outcome.reason == 'observation_end', team.failure
        for name, role in team.roles.items():
            assert (role.workspace / 'script.txt').read_text() == '0'
            assert (role.workspace / 'MEMORY.md').read_text() == name + ' retained fact'
            assert not (role.workspace / 'forbidden').exists()
            assert '[week-1]' in team._git(role, 'log', '--format=%s')
            wire = [item['body'] for item in team.test_wire if item['role'] == name]
            usage = [json.loads(line) for line in role.usage.path.read_text().splitlines()]
            assert [item['body'] for item in usage if item['event'] == 'http_request'] == wire
            for request in wire:
                definitions = [item['function'] for item in request['tools']]
                assert not any(item['name'].startswith('text_') for item in definitions)
                assert 'text_create' not in request['messages'][0]['content']
            audit = [json.loads(line) for line in role.audit.path.read_text().splitlines()]
            assert any(item.get('tool') == 'text_list' and item['status'] == 'failed' for item in audit)
            artifact(name + '-wire.json', wire)
        ceo_wire = json.dumps([item for item in team.test_wire if item['role'] == 'ceo'])
        assert 'growth full handoff' in ceo_wire and 'ops_finance full handoff' in ceo_wire
        assert 'Team reply' in ceo_wire


def test_pf_nontext_declarations_versions_operations_and_model_use():
    from test_pf_candidates import query
    from test_relation_batch import create, batch
    chosen = {}
    turns = Counter()

    def respond(team, role, body):
        if role == 'ceo':
            return calls(('bash', dict(command=ADVANCE)))
        if role == 'ops_finance':
            return final('No additional changes.')
        turns[role] += 1
        if turns[role] == 1:
            return calls(('relation_batch', dict(items=[create('model-file', 'growth:report.txt@v1', [chosen['output']])])) )
        if turns[role] == 2:
            receipt = json.loads(body['messages'][-1]['content'].split('\n[pf:', 1)[0])
            assert receipt['saved'] == 1
            return calls(('bash', dict(command='pf depend growth:report.txt@v1 --detail')))
        if turns[role] == 3:
            assert chosen['output'] in body['messages'][-1]['content']
            return calls(('bash', dict(command='pf show ' + chosen['output'] + ' --full')))
        assert 'source_marker' in body['messages'][-1]['content'] and '17' in body['messages'][-1]['content']
        return final('Using saved source_marker=17 from the declared output.')

    with make_team('pf', respond, 'capability-pf-sources', capabilities=AgentCapabilities.round_three('pf')) as team:
        owner = team.roles['growth']
        for name, content in [('source.txt', 'old exact source'), ('report.txt', 'derived report')]:
            owner.executor.execute('write_file', dict(path=name, content=content, note='exact purpose'))
        output = owner.executor.execute('bash', dict(command='./novamind-operation query "SELECT 17 AS source_marker"', note='SQL purpose'))
        _, candidates = query(owner.executor, 'pf candidates --history --kind sql')
        chosen['output'] = candidates['items'][0]['version']
        program = 'import novamind_api as n; print(n.infrastructure.set_capacity_tier(3))'
        output = owner.executor.execute('bash', dict(command='./novamind-operation python-c ' + shlex.quote(program)))
        operation = team.server.conn.execute("SELECT operation_ref FROM business_operations WHERE tool='set_capacity_tier'").fetchone()[0]
        item = create('operation', 'growth:report.txt@v1', [])
        item['target'] = {'operation_ref': operation}
        item['sources'] = [{'ref': {'cite': 'growth:source.txt@v1'}}, {'ref': {'cite': chosen['output']}}]
        receipt = batch(team, 'growth', [item, create('operation-source', 'growth:report.txt@v1', []) | {
            'sources': [{'ref': {'operation_ref': operation}}]}, create('bad-text', 'growth:r1.1', ['growth:source.txt@v1'])], 'initial')
        assert receipt['saved'] == 2 and receipt['rejected'] == 1
        assert receipt['items'][-1]['error_code'] == 'disabled_reference'
        owner.executor.execute('write_file', dict(path='source.txt', content='new source'))
        reader = team.roles['ceo'].executor
        shown = reader.execute('bash', dict(command='pf show growth:source.txt@v1 --full'))
        assert 'old exact source' in shown and 'exact purpose' in shown and 'new source' not in shown
        shown = reader.execute('bash', dict(command='pf rdepend growth:source.txt@v1 --detail'))
        assert operation in shown
        assert 'session_id' not in shown
        assert team.run(stop_after_day=7).reason == 'observation_end', team.failure
        usage = [json.loads(line) for line in owner.usage.path.read_text().splitlines()]
        wire = [item['body'] for item in team.test_wire if item['role'] == 'growth']
        assert [item['body'] for item in usage if item['event'] == 'http_request'] == wire
        artifact('pf-nontext-wire.json', wire)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_checkpoint_keeps_capabilities_and_replaces_saved_disabled_guidance(tmp_path, mode):
    from saas_bench.multi_agent import MultiAgentRuntime
    from test_preflight_team_checkpoint import clients, dispose, fast_simulator, make_team as checkpoint_team
    caps = AgentCapabilities.round_three(mode)
    original = checkpoint_team(tmp_path / 'original', mode, capabilities=caps)
    restored = None
    selected = []
    frozen_systems = {}
    try:
        for role in original.roles.values():
            role.agent.system_prompt += '\nSTALE-CAPABILITY-GUIDANCE: use text_create and note snapshots.'

        def save(runtime, boundary):
            nonlocal restored
            if boundary == 'answers_ready' and not selected:
                for name, role in runtime.roles.items():
                    frozen_systems[name] = role.agent._context_system_prompt()
                    if name == 'ceo':
                        output = role.executor.execute('write_file', dict(path='MEMORY.md', content='AFTER_SYSTEM_FREEZE'))
                        assert not output.startswith('Error:'), output
                selected.append(runtime.checkpoint())
                restored = MultiAgentRuntime.restore(selected[0], tmp_path / 'restored',
                    client_factory=clients(), simulator_factory=fast_simulator)

        original.checkpoint_callback = save
        assert original.run(stop_after_day=7).reason == 'observation_end', original.failure
        saved = json.loads((selected[0] / 'team.json').read_text())
        assert saved['configuration']['capabilities'] == caps.to_dict()
        assert restored.configuration['capabilities'] == caps.to_dict()
        for name, role in restored.roles.items():
            assert role.registry.capabilities == role.executor.capabilities == role.agent.capabilities == caps
            assert 'STALE-CAPABILITY-GUIDANCE' not in role.agent.system_prompt
            assert all('STALE-CAPABILITY-GUIDANCE' not in message.content
                       for message in role.agent.conversation if message.role == 'system')
            prompt = role.agent._context_system_prompt()
            assert name + ' memory' in prompt and 'AFTER_SYSTEM_FREEZE' not in prompt
            expected_memory = 'AFTER_SYSTEM_FREEZE' if name == 'ceo' else name + ' memory'
            assert (role.workspace / 'MEMORY.md').read_text() == expected_memory
            before = frozen_systems[name]
            def origins(text):
                return [(item['version_id'], item['source_range'], text[slice(*item['request_range'])])
                        for item in getattr(text, 'origins', [])]
            assert origins(prompt) == origins(before)
            assert role.executor.execute('text_list', {}).startswith('Error:')
            # The script batch from the selected week must not run again.
            assert (role.workspace / 'script-days.txt').read_text() == '0\n'
        sizes = {name: len(role.usage.path.read_text().splitlines()) for name, role in restored.roles.items()}
        assert restored.run(stop_after_day=14).reason == 'observation_end', restored.failure
        for name, role in restored.roles.items():
            requests = [json.loads(line)['body'] for line in role.usage.path.read_text().splitlines()[sizes[name]:]
                        if json.loads(line)['event'] == 'http_request']
            assert requests
            if name in ('ceo', 'growth'):
                assert name + ' memory' in requests[0]['messages'][0]['content']
                assert 'AFTER_SYSTEM_FREEZE' not in requests[0]['messages'][0]['content']
            for request in requests:
                assert 'STALE-CAPABILITY-GUIDANCE' not in request['messages'][0]['content']
                assert not any(item['function']['name'].startswith('text_') for item in request['tools'])
            assert (role.workspace / 'script-days.txt').read_text() == '0\n7\n'
            artifact(name + '-restored-wire.json', requests)
    finally:
        if restored:
            dispose(restored)
        dispose(original)


def test_old_pf_text_history_stays_readable_but_cannot_be_redeclared(tmp_path):
    from saas_bench.multi_agent import MultiAgentRuntime
    from test_preflight_team_checkpoint import clients, dispose, fast_simulator, make_team as checkpoint_team
    from test_preflight_team_evidence import declare, write
    from test_relation_batch import batch, create
    original = checkpoint_team(tmp_path / 'legacy', 'pf')
    restored = None
    try:
        declare(original, 'growth', 'historical claim', ['unknown: old evidence'])
        write(original, 'growth', 'report', 'report.txt')
        old = batch(original, 'growth', [create('old', 'growth:report.txt@v1', ['growth:r1.1'])], 'old')['items'][0]
        checkpoint = original.checkpoint()
        restored = MultiAgentRuntime.restore(checkpoint, tmp_path / 'restored', client_factory=clients(),
            simulator_factory=fast_simulator, capabilities=AgentCapabilities.round_three('pf'))
        owner = restored.roles['growth']
        assert 'historical claim' in owner.executor.execute('bash', dict(command='pf show growth:r1.1 --full'))
        assert old['relation_revision'] in owner.executor.execute('bash', dict(command='pf depend growth:report.txt@v1 --all-revisions'))
        revise = dict(item_id='revision', op='revise', relation=old['relation'], expected_revision=old['revision'], reason='Keep unchanged sources')
        rejected = batch(restored, 'growth', [revise], 'rejected')
        assert rejected['items'][0]['error_code'] == 'disabled_reference'
        revised = batch(restored, 'growth', [dict(revise, sources=[{'ref': {'missing': {
            'description': 'replacement source', 'reason': 'not saved'}}}])], 'nontext')
        assert revised['saved'] == 1
        assert old['relation_revision'] in owner.executor.execute('bash', dict(command='pf depend growth:report.txt@v1 --all-revisions'))
    finally:
        if restored:
            dispose(restored)
        dispose(original)


def test_pf_agent_can_disable_text_without_disabling_history():
    from types import SimpleNamespace
    agent = BashAgent(get_bash_agent_tool_descriptions(pf_queries=True), SimpleNamespace(),
                      text_registration=False, pf=True)
    assert 'text_create' not in agent.system_prompt
    assert 'pf show HANDLE --full' in agent.system_prompt


@pytest.mark.parametrize('source', ['entry', 'parameters'])
def test_batch_factory_passes_explicit_capabilities(frozen, tmp_path, source):
    from copy import deepcopy
    from saas_bench import team_batch
    from test_preflight_team_checkpoint import clients, dispose
    manifest = deepcopy(frozen[1])
    for mode in ('git', 'pf'):
        entry = deepcopy(next(row for row in manifest['runs'] if row['group'] == mode))
        entry['output_dir'] = str(tmp_path / (mode + '-records'))
        caps = AgentCapabilities.round_three(mode).to_dict()
        if source == 'entry':
            entry['capabilities'] = caps
        else:
            manifest['parameters']['capabilities'] = {mode: caps}
        team = team_batch.make_team(manifest, entry, tmp_path / mode, client_factory=clients())
        try:
            assert team.configuration['capabilities'] == caps
            for role in team.roles.values():
                assert role.executor.execute('text_list', {}).startswith('Error:')
                assert role.agent.capabilities.to_dict() == caps
                assert 'text_create' not in role.agent.system_prompt
        finally:
            dispose(team)


def test_git_recovery_does_not_inject_old_notes_or_save_new_artifacts(tmp_path):
    from saas_bench.multi_agent import MultiAgentRuntime
    from test_preflight_team_checkpoint import clients, dispose, fast_simulator, make_team as checkpoint_team
    original = checkpoint_team(tmp_path / 'legacy', 'git')
    restored = None
    try:
        role = original.roles['growth']
        role.executor.execute('write_file', dict(path='report.txt', content='original bytes', note='OLD-NOTE-ONLY'))
        checkpoint = original.checkpoint()
        restored = MultiAgentRuntime.restore(checkpoint, tmp_path / 'restored', client_factory=clients(),
            simulator_factory=fast_simulator, capabilities=AgentCapabilities.round_three('git').to_dict())
        for reader in (restored.roles['growth'], restored.roles['ceo']):
            path = str(restored.roles['growth'].workspace / 'report.txt')
            for tool, args in [('read_file', {'path': path}), ('bash', {'command': 'cat ' + shlex.quote(path)})]:
                output = reader.executor.execute(tool, args)
                assert 'original bytes' in output and 'OLD-NOTE-ONLY' not in output
        role = restored.roles['growth']
        before = {path.name: path.read_bytes() for path in (role.workspace / 'tool-outputs').iterdir()}
        refs = restored._git(role, 'show-ref')
        assert role.executor.execute('write_file', dict(path='report.txt', content='new', note='disabled')).startswith('Error:')
        assert restored._git(role, 'show-ref') == refs
        assert before == {path.name: path.read_bytes() for path in (role.workspace / 'tool-outputs').iterdir()}
        output = role.executor.execute('bash', dict(command="printf 'manual' > manual.txt; git add manual.txt; git -c user.name=Test -c user.email=test@example.invalid commit -qm manual"))
        assert '[exit code:' not in output
        assert restored._git(role, 'show', 'HEAD:manual.txt') == 'manual'
    finally:
        if restored:
            dispose(restored)
        dispose(original)


def test_git_extra_saves_can_be_disabled_independently(tmp_path):
    from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
    from saas_bench.text_registry import TextRegistry
    from test_text_registry import git
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    git(workspace, 'init', '-q')
    git(workspace, 'config', 'user.name', 'Test')
    git(workspace, 'config', 'user.email', 'test@example.invalid')
    (workspace / 'source.txt').write_text('source')
    git(workspace, 'add', '.')
    git(workspace, 'commit', '-qm', 'initial')
    caps = AgentCapabilities(git_extra_saves=False)
    registry = TextRegistry(workspace, 'git', capabilities=caps)
    executor = BashAgentToolExecutor(workspace, text_registry=registry)
    assert 'Error:' not in executor.execute('write_file', dict(path='source.txt', content='changed', note='private note'))
    assert not (workspace / 'tool-outputs').exists()
    assert 'private note' not in executor.execute('read_file', {'path': 'source.txt'})
    assert (executor.private_root / 'tool-notes.jsonl').exists()
    from test_relation_batch import create
    result = json.loads(executor.execute('relation_batch', {'items': [create('source', 'source.txt', ['source.txt'])]}))
    assert result['rejected'] == 1 and 'Immediate Git snapshots are disabled' in result['items'][0]['error']
    assert not git(workspace, 'for-each-ref', '--format=%(refname)', 'refs/ceobench/snapshots')
    assert git(workspace, 'show', 'HEAD:source.txt') == 'source'
