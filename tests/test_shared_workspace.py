import json
from pathlib import Path
import shlex
import threading
from contextlib import contextmanager
from saas_bench.agent_capabilities import AgentCapabilities
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.role_policy import ROLES
from test_preflight_team_flow import ADVANCE, calls, final, make_team, raw
from test_preflight_team_checkpoint import clients, dispose, fast_simulator
from test_receipt_outputs import artifact
from concurrent.futures import ThreadPoolExecutor
import re
import uuid

import pytest

from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
from saas_bench.api_server import registered_script_source
from saas_bench.public_refs import PublicRefs
from saas_bench.role_policy import AgentIdentity


def executors(tmp_path):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    refs = PublicRefs(tmp_path / 'refs.sqlite')
    return [BashAgentToolExecutor(workspace, identity=AgentIdentity('run', 'world', role,
        uuid.uuid4().hex, uuid.uuid4().hex), require_sandbox=True,
        public_refs=refs, private_root=tmp_path / 'private' / role)
        for role in ('growth', 'ops_finance')]


def test_shared_complete_outputs_do_not_overwrite_other_role(tmp_path):
    workers = executors(tmp_path)
    def output(worker):
        return worker.execute('bash', {'command': "python -c \"print('" + worker.role + "' * 20000)\""})
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(output, workers))
    paths = [re.search(r'full output: ([^\]]+)', result).group(1) for result in results]
    assert paths[0] != paths[1], results
    for worker, path in zip(workers, paths):
        assert (worker.workspace_path / path).read_text() == worker.role * 20000 + '\n'


def test_shared_registered_sources_keep_both_live_snapshots(tmp_path):
    first, second = executors(tmp_path)
    first._bash_facts = second._bash_facts = {'boundary_closed': True}
    with registered_script_source(first, 'scripts/report.py', 'FIRST') as a:
        with registered_script_source(second, 'scripts/report.py', 'SECOND') as b:
            assert a != b
            assert a.read_text() == 'FIRST'
            assert b.read_text() == 'SECOND'
        assert a.read_text() == 'FIRST'
        assert not b.exists()
    assert not a.exists()


@contextmanager
def shared_team(mode, responder=None, **options):
    with make_team(mode, responder or (lambda *args: final('done')), 'shared-workspace',
                     workspace_layout='shared-v1', capabilities=AgentCapabilities.round_three(mode), **options) as team:
        try:
            yield team
        finally:
            artifact('observations.json', [json.loads(path.read_text()) for path in (team.root / 'private/file-observations').glob('*.json')])
            artifact('model-inputs.json', {role: [json.loads(line) for line in runtime.usage.path.read_text().splitlines()] if runtime.usage.path.exists() else [] for role, runtime in team.roles.items()})
            artifact('timeline.json', team.test_timeline)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_shared_real_bash_and_file_tools_are_concurrent(mode):
    with shared_team(mode) as team:
        team._set_phase('analysis')
        workers = [team.roles[role].executor for role in ('growth', 'ops_finance')]
        assert len({runtime.workspace for runtime in team.roles.values()}) == 1
        for worker in workers:
            assert worker.guest_root == '/workspace'
        assert 'Error:' not in workers[0].execute('write_file', dict(path='/workspace/scripts/report.py', content='FIRST'))
        assert 'FIRST' in workers[1].execute('read_file', dict(path='scripts/report.py'))
        assert 'Error:' not in workers[1].execute('edit_file', dict(path='scripts/report.py', old_string='FIRST', new_string='SECOND'))
        assert 'SECOND' in workers[0].execute('read_file', dict(path='/workspace/scripts/report.py'))
        program = """import sys,time
from pathlib import Path
mine,peer=sys.argv[1:]
Path('plans/'+mine).write_text(mine)
end=time.monotonic()+10
while not Path('plans/'+peer).exists():
    assert time.monotonic()<end, 'partner Bash did not execute concurrently'
    time.sleep(.01)
print(Path('plans/'+peer).read_text())
"""
        def execute(worker):
            peer = 'ops_finance' if worker.role == 'growth' else 'growth'
            return worker.execute('bash', dict(command='python -c ' + shlex.quote(program) + ' ' + worker.role + ' ' + peer))
        with ThreadPoolExecutor(2) as pool:
            outputs = list(pool.map(execute, workers))
        assert 'ops_finance' in outputs[0] and 'growth' in outputs[1], outputs
        assert all(worker.last_status == 'succeeded' for worker in workers), outputs
        records = [json.loads(p.read_text()) for p in (team.root / 'private/file-observations').glob('*.json')]
        overlap = [row for row in records if row['tool'] == 'bash' and row['concurrent_executions']]
        assert len(overlap) == 2
        assert all('before' in row and 'after' in row for row in overlap), records
        if mode == 'pf':
            assert all(row['before']['reused_capture'] for row in overlap)
        else:
            assert all(row['before']['coverage'] == 'file_acquisitions_not_atomic_tree' for row in overlap)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_shared_week_saves_inputs_and_checkpoint(mode, tmp_path):
    turns = {}
    def respond(team, role, body):
        rendered = json.dumps(body['messages'])
        assert 'MEMORY-ONLY-' + role in rendered
        assert all('MEMORY-ONLY-' + other not in rendered for other in ROLES if other != role)
        assert '/workspace' in body['messages'][0]['content']
        if 'VERIFY-SHARED' in str(body['messages'][-1].get('content')):
            return final('SHARED-REPLY')
        turns[role] = turns.get(role, 0) + 1
        if role != 'ceo':
            if turns[role] == 1:
                return calls(('write_file', dict(path='plans/' + role + '.txt', content=role)))
            return final('ADVICE-' + role)
        if turns[role] == 1:
            assert 'ADVICE-growth' in rendered and 'ADVICE-ops_finance' in rendered
            return calls(('ask_analyst', dict(role='growth', message='VERIFY-SHARED')))
        assert 'SHARED-REPLY' in rendered
        return calls(('bash', dict(command=ADVANCE)))
    restored = None
    with shared_team(mode, respond) as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        root = team.roles['ceo'].workspace
        for role in ROLES:
            (root / 'memory' / (role + '.md')).write_text('MEMORY-ONLY-' + role)
        outcome = team.run(stop_after_day=7)
        assert (outcome.reason, team.failure) == ('observation_end', None), getattr(team, 'failure_detail', None)
        log = team._git(team.roles['ceo'], 'log', '--format=%s').splitlines()
        assert log.count('Initial workspace') == 1
        assert sum(line.startswith('Analyst handoff ') for line in log) == 2
        assert sum(line.startswith('Team reply ') for line in log) == 1
        assert sum('[week-1]' in line for line in log) == 1
        for message in team.messages:
            if message.handoff_commit:
                tree = team._git(team.roles['ceo'], 'ls-tree', '-r', '--name-only', message.handoff_commit)
                assert 'memory/growth.md' in tree and 'memory/ops_finance.md' in tree
        usage = [json.loads(line) for line in team.roles['ceo'].usage.path.read_text().splitlines()]
        assert any('SHARED-REPLY' in json.dumps(row) for row in usage)
        checkpoint = team.checkpoint()
        assert sorted(p.name for p in (checkpoint / 'roles').iterdir()) == ['shared']
        try:
            restored = MultiAgentRuntime.restore(checkpoint, tmp_path / 'restored', client_factory=clients(), simulator_factory=fast_simulator)
            assert restored.configuration['workspace_layout'] == 'shared-v1'
            assert len({runtime.workspace for runtime in restored.roles.values()}) == 1
            assert (restored.roles['growth'].workspace / 'plans/ops_finance.txt').read_text() == 'ops_finance'
            assert len({runtime.identity.agent_id for runtime in restored.roles.values()}) == 3
            with pytest.raises(ValueError, match='layout'):
                MultiAgentRuntime.restore(checkpoint, tmp_path / 'wrong-layout', client_factory=clients(), workspace_layout='role-v1')
        finally:
            if restored:
                dispose(restored)


def test_git_manual_errors_and_simultaneous_host_commits_remain_real():
    with shared_team('git') as team:
        team._set_phase('analysis')
        roles = [team.roles[role] for role in ('growth', 'ops_finance')]
        for runtime in roles:
            assert runtime.executor.execute('bash', dict(command='git status --porcelain')).strip() == '(no output)'
        barrier = threading.Barrier(2)
        def save(runtime):
            runtime.executor.execute('write_file', dict(path='plans/' + runtime.identity.role, content='saved'))
            barrier.wait(timeout=10)
            return team._snapshot_history(runtime, 'Concurrent ' + runtime.identity.role)
        with ThreadPoolExecutor(2) as pool:
            commits = list(pool.map(save, roles))
        assert len(set(commits)) == 2
        assert team._git(roles[0], 'rev-list', '--count', 'HEAD') == '3'
        executor = roles[0].executor
        result = executor.execute('bash', dict(command="git -c user.name=Manual -c user.email=manual@example.invalid commit --allow-empty -m manual"))
        assert executor.last_status == 'succeeded', result
        lock = roles[0].workspace / '.git/index.lock'
        lock.write_text('manual lock')
        try:
            result = executor.execute('bash', dict(command='git add plans'))
            assert 'index.lock' in result and executor.last_status != 'succeeded', result
            assert 'conflict' not in result.lower()
        finally:
            lock.unlink()


def test_git_observation_race_is_private_gap_and_business_continues(monkeypatch):
    from saas_bench import workspace_observation
    with shared_team('git') as team:
        team._set_phase('analysis')
        runtime = team.roles['growth']
        path = runtime.workspace / 'plans/race.txt'
        path.write_text('first')
        original = workspace_observation.open_file
        changed = []
        @contextmanager
        def racing(root, candidate, *args, **kwargs):
            with original(root, candidate, *args, **kwargs) as stream:
                yield stream
            if candidate == path and not changed:
                path.write_text('second')
                changed.append(True)
        monkeypatch.setattr(workspace_observation, 'open_file', racing)
        result = runtime.executor.execute('bash', dict(command='printf BUSINESS-OK'))
        assert result == 'BUSINESS-OK'
        records = [json.loads(p.read_text()) for p in (team.root / 'private/file-observations').glob('*.json')]
        assert any(gap['path'] == 'plans/race.txt' for row in records for gap in row['before']['gaps'])
        assert runtime.executor.execute('bash', dict(command='printf NEXT-OK')) == 'NEXT-OK'
        assert team.failure is None


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_shared_registered_snapshot_is_private_but_adjacent_data_stays_live(mode):
    with shared_team(mode) as team:
        team._set_phase('analysis')
        growth = team.server.role_executors['growth']
        peer = team.roles['ops_finance'].executor
        root = team.roles['growth'].workspace
        (root / 'scripts/data.txt').write_text('LIVE-DATA')
        (root / 'scripts/adjacent.py').write_text("VALUE='LIVE-MODULE'")
        source = "from pathlib import Path\nimport adjacent,novamind_api as nm\nprint('PRIVATE-SOURCE')\nprint(Path(__file__).read_text())\nprint(Path(__file__).parent.joinpath('data.txt').read_text(),adjacent.VALUE,nm.vars.current_day)\n"
        status, response = raw(team, 'growth', '/daily-scripts', dict(name='scripts/report.py', content=source))
        assert status == 200 and response['success']
        assert raw(team, 'ops_finance', '/daily-scripts')[1] != raw(team, 'growth', '/daily-scripts')[1]
        with registered_script_source(growth, 'scripts/report.py', source) as guest:
            result = growth.execute('bash', dict(command='python ' + shlex.quote(str(guest))))
            assert 'PRIVATE-SOURCE' in result and 'LIVE-DATA LIVE-MODULE 0' in result, result
            assert 'PRIVATE-SOURCE' not in peer.execute('bash', dict(command='cat ' + shlex.quote(str(guest))))
            assert all(b'PRIVATE-SOURCE' not in p.read_bytes() for p in root.rglob('*') if p.is_file() and '.git' not in p.parts)
            assert team._git(team.roles['growth'], 'ls-files', '--others', '--exclude-standard').find('.registered-') < 0 or (root / guest.relative_to('/workspace')).read_bytes() == b''
        assert not list((team.root / 'private/growth/script-sources').glob('*.py'))
        assert not list(root.rglob('.registered-*'))
        assert peer.execute('bash', dict(command='printf CONTINUES')).startswith('CONTINUES')


def test_git_observation_failure_does_not_change_tool_result(monkeypatch):
    with shared_team('git') as team:
        team._set_phase('analysis')
        observer = team._workspace_observer
        def fail(_):
            raise OSError('injected observer read failure')
        monkeypatch.setattr(observer, '_scan', fail)
        executor = team.roles['growth'].executor
        assert executor.execute('write_file', dict(path='plans/works', content='saved')).startswith('File written:')
        assert (executor.workspace_path / 'plans/works').read_text() == 'saved'
        assert executor.execute('bash', dict(command='printf NEXT')) == 'NEXT'
        records = [json.loads(p.read_text()) for p in observer.root.glob('*.json')]
        assert all(row['gaps'] for row in records)
        assert team.failure is None


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_initialized_checkpoint_preserves_frozen_role_memory(mode, tmp_path):
    checkpoints = []
    def respond(team, role, body):
        return calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final('Advice')
    with shared_team(mode, respond) as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        for role in ROLES:
            (team.roles[role].workspace / 'memory' / (role + '.md')).write_text('FROZEN-' + role)
        def save(runtime, boundary):
            if boundary == 'answers_ready' and not checkpoints:
                for role in ROLES:
                    (runtime.roles[role].workspace / 'memory' / (role + '.md')).write_text('LATER-ON-DISK-' + role)
                checkpoints.append(runtime.checkpoint())
        team.checkpoint_callback = save
        assert team.run(stop_after_day=7).reason == 'observation_end'
        restored = MultiAgentRuntime.restore(checkpoints[0], tmp_path / 'frozen', client_factory=clients(), simulator_factory=fast_simulator)
        try:
            for role, runtime in restored.roles.items():
                system = runtime.agent.conversation[0].content
                assert 'FROZEN-' + role in system and 'LATER-ON-DISK' not in system
                assert (runtime.workspace / 'memory' / (role + '.md')).read_text() == 'LATER-ON-DISK-' + role
                if mode == 'pf':
                    assert any('file_' in source['version_id'] for source in system.origins)
            assert restored.run(stop_after_day=7).reason == 'observation_end', getattr(restored, 'failure_detail', None)
            for role in ('ceo', 'growth'):
                requests = [json.loads(line)['body'] for line in restored.roles[role].usage.path.read_text().splitlines() if json.loads(line)['event'] == 'http_request']
                assert requests and all('FROZEN-' + role in json.dumps(body) and 'LATER-ON-DISK' not in json.dumps(body) for body in requests)
        finally:
            dispose(restored)


def test_git_observer_records_root_race_and_recovers_failed_blob_publish(monkeypatch):
    import hashlib
    from saas_bench import workspace_observation
    with shared_team('git') as team:
        observer = team._workspace_observer
        root = team.roles['growth'].workspace
        (root / 'plans/one').write_text('durable-observed-bytes')
        walk = workspace_observation.os.walk
        def racing(*args, **kwargs):
            for index, value in enumerate(walk(*args, **kwargs)):
                yield value
                if index == 0:
                    (root / 'new-root-file').write_text('created after root enumeration')
        monkeypatch.setattr(workspace_observation.os, 'walk', racing)
        result = observer._scan(root)
        assert dict(path='.', reason='directory_changed_during_tree_scan') in result['gaps']
        monkeypatch.setattr(workspace_observation.os, 'walk', walk)
        target = observer.blobs / hashlib.sha256(b'durable-observed-bytes').hexdigest()
        target.write_bytes(b'broken')
        replace = workspace_observation.os.replace
        def fail_publish(source, destination, *args, **kwargs):
            if Path(destination) == target:
                raise OSError('injected observation publish failure')
            return replace(source, destination, *args, **kwargs)
        monkeypatch.setattr(workspace_observation.os, 'replace', fail_publish)
        result = observer._scan(root)
        assert 'plans/one' not in result['entries']
        assert any(gap['path'] == 'plans/one' for gap in result['gaps'])
        assert not list(observer.blobs.glob('.blob-*'))
        monkeypatch.setattr(workspace_observation.os, 'replace', replace)
        result = observer._scan(root)
        assert result['entries']['plans/one']['sha256'] == target.name
        assert target.read_bytes() == b'durable-observed-bytes'


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_registered_snapshot_cleanup_survives_shared_parent_rename(mode):
    with shared_team(mode) as team:
        team._set_phase('analysis')
        script = team.server.role_executors['growth']
        peer = team.roles['ops_finance'].executor
        root = team.roles['growth'].workspace
        script._bash_facts = dict(boundary_closed=True)
        with registered_script_source(script, 'scripts/report.py', "print('SOURCE-OK')") as guest:
            output = script.execute('bash', dict(command='python ' + shlex.quote(str(guest))))
            assert 'SOURCE-OK' in output
            renamed = peer.execute('bash', dict(command='mv scripts moved-scripts'))
            assert peer.last_status == 'succeeded', renamed
        assert not list(root.rglob('.registered-*'))
        assert not list((team.root / 'private/growth/script-sources').glob('*.py'))
        assert 'NEXT-OK' in peer.execute('bash', dict(command='printf NEXT-OK'))


def test_shared_pf_help_and_role_prompts_use_the_real_file_namespace():
    with shared_team('pf') as team:
        for role, runtime in team.roles.items():
            prompt = runtime.agent.system_prompt
            assert 'shared:plans/forecast.md@v3' in prompt
            assert 'growth:forecast.md' not in prompt and 'ops_finance:cash-plan.md' not in prompt
            assert 'Only memory/' + role + '.md is auto-loaded' in prompt
        help_text = team.roles['growth'].executor.execute('bash', dict(command='pf help'))
        assert 'shared:plans/forecast.md@v3' in help_text and 'growth:forecast.md' not in help_text
        assert 'growth:report.md' not in help_text
        assert 'growth:forecast.md' not in (team.roles['growth'].workspace / 'docs/workspace-history.md').read_text()


def test_shared_note_scan_race_does_not_cancel_business(monkeypatch):
    with shared_team('pf') as team:
        root = team.roles['growth'].workspace
        path = root / 'plans/fragile'
        path.write_text('peer will delete')
        original = Path.stat
        reads, deleted = [], []
        def racing(candidate, *args, **kwargs):
            if candidate == path and kwargs.get('follow_symlinks', True):
                reads.append(True)
                if len(reads) == 2 and not deleted:
                    deleted.append(True)
                    result = team.roles['ops_finance'].executor.execute('bash', dict(command='rm plans/fragile'))
                    assert team.roles['ops_finance'].executor.last_status == 'succeeded', result
            return original(candidate, *args, **kwargs)
        monkeypatch.setattr(Path, 'stat', racing)
        output = team.roles['growth'].executor.execute('bash', dict(command='printf BUSINESS-AFTER-NOTE-RACE', note='check shared work'))
        assert deleted and 'BUSINESS-AFTER-NOTE-RACE' in output
        assert team.roles['growth'].executor.last_status == 'succeeded', output
        notes = [json.loads(line) for line in (team.root / 'private/growth/tool-notes.jsonl').read_text().splitlines()]
        assert any(gap['path'] == 'plans/fragile' for gap in notes[-1]['file_observation_gaps'])
        assert 'plans/fragile' not in notes[-1]['files']
        assert 'NEXT' in team.roles['growth'].executor.execute('bash', dict(command='printf NEXT'))


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('peer_removed_placeholder', [False, True])
def test_shared_registered_preparation_failure_cleans_placeholder_and_continues(mode, peer_removed_placeholder, monkeypatch):
    import os
    import errno
    from saas_bench import api_server
    def respond(team, role, body):
        return calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final('ADVICE')
    with shared_team(mode, respond) as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        status, reply = raw(team, 'growth', '/daily-scripts', dict(name='scripts/preparation.py', content="print('SHOULD-NOT-RUN')"))
        assert status == 200 and reply['success']
        original = os.open
        failed = []
        def fail_private_source(path, flags, *args, **kwargs):
            directory = kwargs.get('dir_fd')
            if str(path).startswith('.registered-') and directory is not None and not failed:
                if os.readlink('/proc/self/fd/' + str(directory)).endswith('/private/growth/script-sources'):
                    failed.append(True)
                    if peer_removed_placeholder:
                        with ThreadPoolExecutor(1) as pool:
                            pool.submit(team.roles['ops_finance'].executor.execute, 'bash', dict(command='rm scripts/.registered-*')).result(timeout=10)
                    raise OSError(errno.ENOSPC, 'injected private source creation failure')
            return original(path, flags, *args, **kwargs)
        monkeypatch.setattr(api_server.os, 'open', fail_private_source)
        outcome = team.run(stop_after_day=7)
        assert failed and outcome.reason == 'observation_end', (team.failure, getattr(team, 'failure_detail', None))
        assert team.server.role_script_results['growth'][0]['classification'] == 'script_error'
        assert not list(team.roles['growth'].workspace.rglob('.registered-*'))
        assert not list((team.root / 'private/growth/script-sources').glob('*'))
