import hashlib
import json
import stat
from pathlib import Path

import httpx
from openai import OpenAI
import pytest

from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.role_policy import ROLES
from saas_bench.team_prompts import CEO_OPENING
from saas_bench.team_checkpoint import CheckpointDataError
from test_preflight_team_checkpoint import dispose, fast_simulator
from test_preflight_team_flow import ADVANCE, calls, completion, final, make_team


FIXTURES = Path(__file__).parent / 'fixtures'
PUBLIC = Path(__file__).resolve().parents[1] / 'public'
LEGACY_CLIENT = FIXTURES / 'registered-script-client-23a90ff.pyz'
LEGACY_HELP = FIXTURES / 'registered-script-cli-23a90ff.md'
OLD_GRANT = 'Registered scripts\nhave the same permissions as their owner.'


def install_legacy_client(workspace):
    (workspace / 'novamind-operation').write_bytes(LEGACY_CLIENT.read_bytes())
    (workspace / 'novamind-operation').chmod(0o755)
    (workspace / 'docs/cli.md').write_bytes(LEGACY_HELP.read_bytes())


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_legacy_checkpoint_refreshes_client_and_help_without_replaying_scripts(tmp_path, mode):
    snapshots, saved_heads, wire, holder, turns = [], {}, [], {}, {}
    nested = """from pathlib import Path
import hashlib
import novamind_api as nm
source = Path(__file__)
day = nm.vars.current_day
with open('nested-days.txt', 'a') as stream: stream.write(str(day) + '\\n')
print('NESTED', day, (source.parent / 'data.txt').read_text(), hashlib.sha256(source.read_bytes()).hexdigest())
"""
    digest = hashlib.sha256(nested.encode()).hexdigest()

    def original_response(team, role, body):
        return calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final(role + ' saved report')

    with make_team(mode, original_response, 'legacy-client-checkpoint') as original:
        original.server.simulator = fast_simulator(original.server.conn, original.server.tools.config, original.server.tools.rng)
        for role, runtime in original.roles.items():
            install_legacy_client(runtime.workspace)
            executor = runtime.executor
            legacy = "import novamind_api as nm\nwith open('legacy-days.txt', 'a') as stream: stream.write(str(nm.vars.current_day) + '\\n')\nprint('LEGACY-" + role + "')\n"
            for path, content in (('legacy.py', legacy), ('weekly_scripts/nested.py', nested),
                                  ('weekly_scripts/data.txt', 'DATA-' + role), ('MEMORY.md', 'MEMORY-' + role)):
                executor.execute('write_file', dict(path=path, content=content))
            receipt = executor.execute('bash', dict(command='./novamind-operation register-daily-script weekly_scripts/nested.py'))
            assert '"name": "nested.py"' in receipt, receipt
            assert '"removed": "nested.py"' in executor.execute('bash', dict(command='./novamind-operation remove-daily-script nested.py'))
            assert '"name": "legacy.py"' in executor.execute('bash', dict(command='./novamind-operation register-daily-script legacy.py'))

        def checkpoint(runtime, boundary):
            if boundary == 'answers_ready':
                for role, owner in runtime.roles.items():
                    legacy = '\nUNRELATED-SAVED-' + role + '\n' + OLD_GRANT
                    owner.agent.system_prompt += legacy
                    for message in owner.agent.conversation:
                        if message.role == 'system':
                            message.content += legacy
                    saved_heads[role] = runtime._git(owner, 'rev-parse', 'HEAD')
                snapshots.append(runtime.checkpoint())

        original.checkpoint_callback = checkpoint
        assert original.run(stop_after_day=7).reason == 'observation_end', original.failure
        selected, = snapshots
        saved_state = json.loads((selected / 'team.json').read_text())
        assert all(week['scripts_status'] == 'succeeded' for week in saved_state['role_weeks'].values())

    def clients(role):
        def respond(request):
            body = json.loads(request.content)
            day = holder['runtime'].server.tools.current_day
            wire.append(dict(role=role, day=day, body=body))
            if role == 'ceo' and CEO_OPENING in str(body['messages'][-1].get('content', '')):
                return completion(final('CURRENT-CEO-DIRECTIONS'), body.get('stream', False))
            key = (role, day)
            turns[key] = turns.get(key, 0) + 1
            if day == 0 and turns[key] == 1:
                message = calls(('read_file', dict(path='docs/cli.md')),
                    ('bash', dict(command='./novamind-operation register-daily-script weekly_scripts/nested.py')),
                    ('write_file', dict(path='weekly_scripts/nested.py', content="raise AssertionError('LIVE-EDIT-MUST-NOT-RUN')\n")))
            elif day == 7 and turns[key] == 1:
                message = calls(('read_file', dict(path='docs/cli.md')))
            else:
                message = calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final(role + ' current report')
            return completion(message, body.get('stream', False))
        return OpenAI(api_key='OFFLINE-ONLY', base_url='https://opencode.ai/zen/go/v1', max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    resumed = MultiAgentRuntime.restore(selected, tmp_path / 'resumed', client_factory=clients,
                                       simulator_factory=fast_simulator)
    holder['runtime'] = resumed
    try:
        updates = json.loads((resumed.root / 'recovery.json').read_text())['workspace_runtime_updates']
        assert len(updates) == 2 * len(ROLES)
        updated = {(record['role'], record['path']): record for record in updates}
        for role, runtime in resumed.roles.items():
            for relative, current, fixture in (('novamind-operation', PUBLIC / 'novamind-client', LEGACY_CLIENT),
                                               ('docs/cli.md', PUBLIC / 'docs/cli.md', LEGACY_HELP)):
                record = updated[role, relative]
                assert record['previous'] == dict(kind='file', sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                    mode=stat.S_IMODE((selected / 'roles' / role / relative).stat().st_mode))
                assert record['sha256'] == hashlib.sha256(current.read_bytes()).hexdigest()
                assert record['mode'] == stat.S_IMODE(current.stat().st_mode)
            assert (runtime.workspace / 'novamind-operation').read_bytes() == (PUBLIC / 'novamind-client').read_bytes()
            assert (runtime.workspace / 'docs/cli.md').read_bytes() == (PUBLIC / 'docs/cli.md').read_bytes()
            assert (runtime.workspace / 'MEMORY.md').read_text() == 'MEMORY-' + role
            assert (runtime.workspace / 'weekly_scripts/nested.py').read_text() == nested
            assert (runtime.workspace / 'legacy-days.txt').read_text() == '0\n'
            assert resumed._git(runtime, 'rev-parse', 'HEAD') == saved_heads[role]
            assert resumed.server.get_daily_scripts(role) == saved_state['scripts'][role]
            assert (runtime.workspace / 'legacy.py').read_text() == saved_state['scripts'][role]['legacy.py']
        assert resumed.run(stop_after_day=7).reason == 'observation_end', resumed.failure
        assert {row['role'] for row in wire} == {'ceo'}
        for runtime in resumed.roles.values():
            assert (runtime.workspace / 'legacy-days.txt').read_text() == '0\n'
            assert not (runtime.workspace / 'nested-days.txt').exists()
        assert resumed.server.get_daily_scripts('ceo')['weekly_scripts/nested.py'] == nested
        assert 'nested.py' not in resumed.server.get_daily_scripts('ceo')
        assert resumed.run(stop_after_day=14).reason == 'observation_end', resumed.failure
        assert (resumed.roles['ceo'].workspace / 'nested-days.txt').read_text() == '7\n'
        for role, runtime in resumed.roles.items():
            assert (runtime.workspace / 'legacy-days.txt').read_text() == '0\n7\n'
            assert (runtime.workspace / 'MEMORY.md').read_text() == 'MEMORY-' + role
            assert saved_heads[role] in resumed._git(runtime, 'log', '--format=%H')
            frozen = selected / 'roles' / role
            assert (frozen / 'novamind-operation').read_bytes() == LEGACY_CLIENT.read_bytes()
            assert (frozen / 'docs/cli.md').read_bytes() == LEGACY_HELP.read_bytes()
            assert (frozen / 'weekly_scripts/nested.py').read_text() == nested
            assert (frozen / 'legacy-days.txt').read_text() == '0\n'
            assert (frozen / 'MEMORY.md').read_text() == 'MEMORY-' + role
            requests = [row['body'] for row in wire if row['role'] == role]
            usage = [json.loads(line) for line in runtime.usage.path.read_text().splitlines()]
            sent = [row['body'] for row in usage if row['event'] == 'http_request']
            assert sent[-len(requests):] == requests
            rendered = json.dumps(requests)
            assert 'observation-only' in rendered and 'temporary Python snapshot' in rendered
            assert "analyst scripts use their owner's business permissions" not in rendered
            for body in requests:
                system, = [message['content'] for message in body['messages'] if message['role'] == 'system']
                assert OLD_GRANT not in system and 'same business permissions' not in system
                assert 'UNRELATED-SAVED-' + role in system
                assert 'MEMORY-' + role in system
        ceo_day7 = json.dumps([row['body'] for row in wire if row['role'] == 'ceo' and row['day'] == 7])
        assert 'NESTED 7 DATA-ceo ' + digest in ceo_day7
        assert json.loads((selected / 'team.json').read_text()) == saved_state
    finally:
        dispose(resumed)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_restore_replaces_artifact_leaf_symlinks_without_touching_targets(tmp_path, mode):
    wire = []
    with make_team(mode, lambda team, role, body: final('unused'), 'legacy-client-symlink') as original:
        original.server.simulator = fast_simulator(original.server.conn, original.server.tools.config, original.server.tools.rng)
        workspace = original.roles['ceo'].workspace
        external = tmp_path / 'external'
        external.mkdir()
        targets = {'novamind-operation': LEGACY_CLIENT, 'docs/cli.md': LEGACY_HELP}
        for index, (name, fixture) in enumerate(targets.items()):
            target = external / str(index)
            target.write_bytes(fixture.read_bytes())
            path = workspace / name
            path.unlink()
            path.symlink_to(target)
        selected = original.checkpoint()

    def clients(role):
        def unexpected(request):
            wire.append(role)
            raise AssertionError('Restore must not call a model')
        return OpenAI(api_key='OFFLINE-ONLY', base_url='https://opencode.ai/zen/go/v1', max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(unexpected)))

    resumed = MultiAgentRuntime.restore(selected, tmp_path / 'resumed', client_factory=clients,
                                       simulator_factory=fast_simulator)
    try:
        updates = json.loads((resumed.root / 'recovery.json').read_text())['workspace_runtime_updates']
        assert len(updates) == 2
        updated = {(record['role'], record['path']): record for record in updates}
        for index, (name, fixture) in enumerate(targets.items()):
            path = resumed.roles['ceo'].workspace / name
            assert not path.is_symlink()
            current = PUBLIC / ('novamind-client' if name == 'novamind-operation' else name)
            assert path.read_bytes() == current.read_bytes()
            record = updated['ceo', name]
            assert record['previous'] == dict(kind='symlink', target=str(external / str(index)), sha256=None)
            assert record['sha256'] == hashlib.sha256(current.read_bytes()).hexdigest()
            assert record['mode'] == stat.S_IMODE(current.stat().st_mode)
            assert (external / str(index)).read_bytes() == fixture.read_bytes()
            assert (selected / 'roles/ceo' / name).is_symlink()
        assert not wire
    finally:
        dispose(resumed)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_restore_rejects_artifact_parent_symlink_before_model(tmp_path, mode):
    wire = []
    with make_team(mode, lambda team, role, body: final('unused'), 'legacy-client-parent-symlink') as original:
        original.server.simulator = fast_simulator(original.server.conn, original.server.tools.config, original.server.tools.rng)
        docs = original.roles['ceo'].workspace / 'docs'
        external = tmp_path / 'external-docs'
        docs.rename(external)
        docs.symlink_to(external, target_is_directory=True)
        before = {str(path.relative_to(external)): path.read_bytes() for path in external.rglob('*') if path.is_file()}
        selected = original.checkpoint()

    def clients(role):
        def unexpected(request):
            wire.append(role)
            raise AssertionError('Restore must not call a model')
        return OpenAI(api_key='OFFLINE-ONLY', base_url='https://opencode.ai/zen/go/v1', max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(unexpected)))

    resumed = None
    try:
        with pytest.raises(CheckpointDataError, match='Cannot refresh restored public client file') as refusal:
            resumed = MultiAgentRuntime.restore(selected, tmp_path / 'resumed', client_factory=clients,
                                               simulator_factory=fast_simulator)
        assert 'docs/cli.md' in str(refusal.value)
        assert not wire
        assert {str(path.relative_to(external)): path.read_bytes() for path in external.rglob('*') if path.is_file()} == before
        assert (selected / 'roles/ceo/docs').is_symlink()
    finally:
        if resumed is not None:
            dispose(resumed)
