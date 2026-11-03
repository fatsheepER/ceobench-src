"""Private authority stays outside the actual role file and model interfaces."""
import json
import shlex

import pytest

from test_preflight_team_evidence import declare, write
from test_preflight_team_flow import final, make_team
from test_relation_batch import batch, create


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_private_registry_and_notes_are_not_public_files(mode):
    with make_team(mode, lambda *args: final('done'), 'private-records') as team:
        role = team.roles['ceo']
        write(team, 'ceo', '{"sha256":"business-value","session_id":"customer-session"}', path='business.json')
        declare(team, 'ceo', 'Keep the current plan', ['unknown: initial observation'])
        role.executor.execute('text_revise', dict(record='r1', text='Revised plan', reason='New figures'))
        receipt = batch(team, 'ceo', [create('source', 'ceo:r1.2', ['ceo:r1.1'])], 'private-call')
        assert receipt['saved'] == 1
        private = json.loads(role.registry.path.read_text())
        assert private['records']['r1'][0]['session_id'] == role.identity.session_id
        assert not role.registry.path.is_relative_to(role.workspace)
        assert not role.registry.relations.path.is_relative_to(role.workspace)
        for name in ('registrations.json', 'relations.json', '.tool-notes.jsonl'):
            assert not (role.workspace / name).exists()
        public = role.executor.execute('read_file', {'path': 'documents/registrations.json'})
        assert 'Keep the current plan' in public and 'Revised plan' in public
        raw = role.executor.execute('text_list', dict(raw=True))
        relation = role.executor.execute('read_file', {'path': 'documents/relations.json'})
        for output in (public, raw, relation, json.dumps(receipt)):
            for secret in (role.identity.run_id, role.identity.world_id, role.identity.agent_id, role.identity.session_id):
                assert secret not in output
            assert 'sha256' not in output and '"calls"' not in output
        for path in (role.registry.path, role.registry.relations.path):
            assert role.executor.execute('read_file', {'path': str(path)}).startswith('Error:')
            output = role.executor.execute('bash', {'command': 'cat ' + shlex.quote(str(path))})
            assert role.identity.session_id not in output and 'No such file' in output
        request = dict(model=role.agent.model, messages=[dict(role='user', content='\n'.join((public, raw, relation)))])
        role.agent._llm_attempt = 0
        role.agent._request_model('chat', request, lambda: role.agent.client.chat.completions.create(**request))
        actual = json.dumps(team.test_wire[-1]['body'])
        assert role.identity.session_id not in actual and role.identity.run_id not in actual
        business = role.executor.execute('read_file', {'path': 'business.json'})
        assert '"sha256":"business-value"' in business and '"session_id":"customer-session"' in business
        role.executor.execute('bash', {'command': 'git add -f . && git -c user.name=Test -c user.email=test@example.invalid commit -qm public'})
        objects = role.executor.execute('bash', {'command': 'git log --all --format=%H | while read c; do git grep -n ' + shlex.quote(role.identity.session_id) + ' "$c" -- || true; done'})
        assert role.identity.session_id not in objects


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_public_views_are_disposable_and_peer_revision_uses_private_authority(mode):
    with make_team(mode, lambda *args: final('done'), 'private-record-peer') as team:
        declare(team, 'growth', 'Original fact', ['unknown: initial observation'])
        growth = team.roles['growth']
        growth.executor.execute('text_revise', dict(record='r1', text='Second fact', reason='Update'))
        target = growth.workspace / 'documents/registrations.json'
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps({'format': 'ceobench.text-register.v1', 'records': {'r1': []}}))
        declare(team, 'ceo', 'Adopt the original revision', ['growth:r1.1'])
        assert team.roles['ceo'].registry._load()['records']['r1'][0]['references'][0]['evidence']['record'] == 'growth:r1.1'
        assert 'Original fact' in growth.registry.path.read_text()
        growth.executor.execute('text_revise', dict(record='r1', reason='Retain the second fact'))
        shown = json.loads(target.read_text())['records']['r1']
        assert [r['version'] for r in shown] == ['r1.1', 'r1.2', 'r1.3']
        assert shown[0]['text'] == 'Original fact'


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_old_checkpoint_restores_world_with_mixed_history_unavailable(tmp_path, mode):
    from saas_bench.multi_agent import MultiAgentRuntime
    from saas_bench.run_state import tree_hash, write_json
    from test_preflight_team_checkpoint import clients, dispose, fast_simulator, make_team as checkpoint_team
    team = checkpoint_team(tmp_path / 'source', mode)
    secret = 'old-internal-identity-0123456789abcdef'
    try:
        role = team.roles['ceo']
        write(team, 'ceo', secret, path='apparently-business.json')
        role.executor.execute('bash', {'command': 'git add -f . && git -c user.name=Test -c user.email=test@example.invalid commit -qm legacy'})
        old_head = team._git(role, 'rev-parse', 'HEAD')
        snapshot = team.checkpoint()
    finally:
        dispose(team)
    state = json.loads((snapshot / 'team.json').read_text())
    state.pop('public_history_format', None)
    write_json(snapshot / 'team.json', state)
    original = tree_hash(snapshot, exact=True)
    restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored',
        client_factory=clients(), simulator_factory=fast_simulator)
    try:
        role = restored.roles['ceo']
        assert restored.server.tools.current_day == state['day']
        shown = role.executor.execute('read_file', {'path': 'apparently-business.json'})
        assert secret not in shown and shown.startswith('Error:')
        old = role.executor.execute('bash', {'command': 'git cat-file -p ' + old_head + ':apparently-business.json'})
        assert secret not in old
        assert 'unavailable' in (role.workspace / 'MEMORY.md').read_text().lower()
        archived = restored.root / 'private/legacy-history/roles/ceo/apparently-business.json'
        assert archived.read_text() == secret
        if mode == 'pf':
            shown = role.executor.execute('bash', {'command': 'pf show ceo:apparently-business.json@v1 --full'})
            assert secret not in shown
        assert all(not scripts for scripts in restored.server._role_scripts.values())
        restored.run(stop_after_day=7)
        for runtime in restored.roles.values():
            assert secret not in runtime.usage.path.read_text()
    finally:
        dispose(restored)
    assert tree_hash(snapshot, exact=True) == original
