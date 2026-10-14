import errno
import json
from pathlib import Path

import pytest

from saas_bench.role_policy import ROLES
from test_preflight_team_checkpoint import fast_simulator
from test_preflight_team_flow import ADVANCE, calls, final, make_team


def register_actions(role, code):
    return calls(('write_file', dict(path='weekly.py', content=code)),
                 ('bash', dict(command='./novamind-operation register-daily-script weekly.py')))


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('exit_code', [3, 137])
def test_script_error_reaches_owner_verbatim_and_owner_repairs(mode, exit_code):
    turns = {}
    code = "import sys\nprint('OUT-' + 'x' * 31000)\nprint('ERR-END No space left on device MemoryError Cannot allocate memory', file=sys.stderr)\n" + f"sys.exit({exit_code})\n"
    def responder(team, role, body):
        day = team.server.tools.current_day
        turn = turns[role, day] = turns.get((role, day), 0) + 1
        if day == 0 and turn == 1:
            return register_actions(role, code)
        if day == 7 and turn == 1:
            rendered = json.dumps(body['messages'])
            assert 'OUT-' + 'x' * 31000 in rendered
            assert 'ERR-END' in rendered
            assert 'Command reported an allocation failure' not in rendered
            return calls(('write_file', dict(path='weekly.py', content="print('repaired')\n")),
                         ('bash', dict(command='./novamind-operation register-daily-script weekly.py')))
        return final(role + ' advice') if role != 'ceo' else calls(('bash', dict(command=ADVANCE)))
    with make_team(mode, responder, 'registered-error') as team:
        outcome = team.run(stop_after_day=14)
        assert outcome.reason == 'observation_end', team.failure
        for role in ROLES:
            record = team.server.role_script_results[role][0]
            assert record['classification'] == 'script_error'
            evidence = Path(record['evidence'])
            assert (evidence / 'stdout.bin').read_bytes() == ('OUT-' + 'x' * 31000 + '\n').encode()
            assert (evidence / 'stderr.bin').read_bytes() == b'ERR-END No space left on device MemoryError Cannot allocate memory\n'
            usage = [json.loads(row) for row in team.roles[role].usage.path.read_text().splitlines()]
            assert any('OUT-' + 'x' * 31000 in json.dumps(row) for row in usage)
            assert team.server.get_daily_scripts(role) == {'weekly.py': "print('repaired')\n"}


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_prestart_enospc_pauses_at_boundary_without_script_error_model_input(mode, monkeypatch):
    from saas_bench.process_boundary import Boundary
    turns = {}
    def responder(team, role, body):
        day = team.server.tools.current_day
        assert day == 0
        turn = turns[role] = turns.get(role, 0) + 1
        if turn == 1:
            return register_actions(role, "print('read-only')\n")
        return final(role + ' advice') if role != 'ceo' else calls(('bash', dict(command=ADVANCE)))
    with make_team(mode, responder, 'registered-enospc') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        snapshots = []
        def checkpoint(runtime, boundary):
            snapshots.append(runtime.checkpoint())
        team.checkpoint_callback = checkpoint
        original = Boundary.__init__
        def fail_at_week(self, *args, **kwargs):
            if team.server.tools.current_day == 7:
                raise OSError(errno.ENOSPC, 'No space left on device')
            return original(self, *args, **kwargs)
        monkeypatch.setattr(Boundary, '__init__', fail_at_week)
        outcome = team.run(stop_after_day=14)
        assert (outcome.day, outcome.reason) == (7, 'paused')
        assert team.failure == 'registered_script_transient_infrastructure'
        record = team.server.role_script_results['ceo'][0]
        assert record['classification'] == 'transient_infrastructure'
        assert record['facts']['process_started'] is False
        assert (Path(record['evidence']) / 'stderr.bin').read_bytes() == b''
        from saas_bench.team_checkpoint import validate_recovery_source, validate_snapshot
        _, state = validate_snapshot(snapshots[-1])
        validate_recovery_source(snapshots[-1], state, current_root=team.root)
        assert all(item['day'] == 0 for item in team.test_wire)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_readonly_script_keeps_evidence_without_business_wrapper(mode):
    with make_team(mode, lambda team, role, body: final('unused'), 'registered-readonly') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        role = team.roles['ceo']
        result = role.executor.execute('write_file', dict(path='weekly.py', content='import novamind_api as nm\nprint(nm.vars.current_day)\n'))
        assert 'Error:' not in result
        assert 'registered' in role.executor.execute('bash', dict(command='./novamind-operation register-daily-script weekly.py')).lower()
        snapshot = team.checkpoint()
        assert '0' in team._script_output('ceo')
        operations = [json.loads(path.read_text()) for path in (team.root / 'private/operations').glob('*.json')]
        assert not any(row['kind'] == 'ceo_scripts' for row in operations)
        record = team.server.role_script_results['ceo'][0]
        assert record['classification'] == 'succeeded'
        assert (Path(record['evidence']) / 'stdout.bin').read_bytes() == b'0\n'
        assert not list(role.workspace.rglob('__pycache__'))
        from saas_bench.team_checkpoint import validate_recovery_source, validate_snapshot
        _, state = validate_snapshot(snapshot)
        validate_recovery_source(snapshot, state, current_root=team.root)


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('role', ROLES)
@pytest.mark.parametrize('failure', ['timeout', 'killed'])
def test_resource_failure_keeps_partial_streams_and_never_reaches_model(mode, role, failure):
    code = "import time, os, signal\nprint('PARTIAL-OUT', flush=True)\n"
    code += "time.sleep(60)\n" if failure == 'timeout' else "os.kill(os.getpid(), signal.SIGKILL)\n"
    def responder(team, sender, body):
        assert sender != role, 'Failed script owner received a model request'
        return final(sender + ' advice') if sender != 'ceo' else calls(('bash', dict(command=ADVANCE)))
    with make_team(mode, responder, 'registered-resource') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        executor = team.roles[role].executor
        assert 'Error:' not in executor.execute('write_file', dict(path='weekly.py', content=code))
        assert 'registered' in executor.execute('bash', dict(command='./novamind-operation register-daily-script weekly.py')).lower()
        source = team.checkpoint()
        script_timeout = team.server.role_executors[role].bash_timeout
        if failure == 'timeout':
            team.server.role_executors[role].bash_timeout = .3
        outcome = team.run(stop_after_day=7)
        team.server.role_executors[role].bash_timeout = script_timeout
        assert (outcome.day, outcome.reason) == (0, 'paused'), team.failure
        assert team.failure == 'registered_script_transient_infrastructure'
        record = team.server.role_script_results[role][0]
        assert record['classification'] == 'transient_infrastructure'
        assert record['facts']['target_started'] is True
        assert record['facts']['boundary_closed'] is True
        assert (Path(record['evidence']) / 'stdout.bin').read_bytes() == b'PARTIAL-OUT\n'
        assert all(not executor.preserved_process for executor in team.server.role_executors.values())
        from saas_bench.team_checkpoint import validate_recovery_source, validate_snapshot
        from saas_bench.run_state import tree_hash
        _, state = validate_snapshot(source)
        validate_recovery_source(source, state, current_root=team.root)
        paused = team.checkpoint()
        assert tree_hash(paused / 'roles', exact=True) == tree_hash(team.root / 'roles', exact=True)
        _, state = validate_snapshot(paused)
        validate_recovery_source(paused, state, current_root=team.root)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_failed_script_partial_business_write_blocks_old_checkpoint(mode):
    with make_team(mode, lambda team, role, body: final('unused'), 'registered-write') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        code = "import novamind_api as nm\nnm.pricing.set_prices(A=37)\nprint('WRITE-DONE')\nraise ValueError('repair me')\n"
        executor = team.roles['ceo'].executor
        executor.execute('write_file', dict(path='weekly.py', content=code))
        assert 'registered' in executor.execute('bash', dict(command='./novamind-operation register-daily-script weekly.py')).lower()
        snapshot = team.checkpoint()
        output = team._script_output('ceo')
        assert 'WRITE-DONE' in output and 'ValueError: repair me' in output
        assert team.server.role_script_results['ceo'][0]['classification'] == 'script_error'
        assert team.server.conn.execute('SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] == 37
        from saas_bench.team_checkpoint import validate_recovery_source, validate_snapshot
        _, state = validate_snapshot(snapshot)
        with pytest.raises(RuntimeError, match='after checkpoint'):
            validate_recovery_source(snapshot, state, current_root=team.root)


def test_permanent_pf_capture_fault_retains_raw_script_evidence_and_blocks_recovery(monkeypatch):
    with make_team('pf', lambda team, role, body: final('unused'), 'registered-capture') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        executor = team.roles['ceo'].executor
        executor.execute('write_file', dict(path='weekly.py', content="print('PERSIST-ME')\n"))
        assert 'registered' in executor.execute('bash', dict(command='./novamind-operation register-daily-script weekly.py')).lower()
        snapshot = team.checkpoint()
        store = team.roles['ceo'].store
        version = store.version
        def fail_stdout(event, slot, *args, **kwargs):
            if slot == 'stdout_bytes':
                raise OSError(errno.EIO, 'permanent capture fault')
            return version(event, slot, *args, **kwargs)
        monkeypatch.setattr(store, 'version', fail_stdout)
        outcome = team.run(stop_after_day=7)
        assert outcome.reason == 'failed'
        record = team.server.role_script_results['ceo'][0]
        assert record['classification'] == 'unknown'
        assert record['facts']['capture_fault']['reason'] == '[Errno 5] permanent capture fault'
        assert (Path(record['evidence']) / 'stdout.bin').read_bytes() == b'PERSIST-ME\n'
        with pytest.raises(RuntimeError, match='recoverable handoff'):
            team.checkpoint()
        with pytest.raises(RuntimeError, match='capture failed'):
            store.assert_healthy()
        assert not team.test_wire


from test_preflight_team_readonly_recovery import managed_git
from test_preflight_team_batch import frozen


@pytest.mark.parametrize('managed_git', ['git', 'pf'], indirect=True)
def test_transient_weekly_script_pause_resumes_real_attempt(managed_git, tmp_path, monkeypatch):
    from saas_bench import team_batch as batch
    from saas_bench.process_boundary import Boundary
    from saas_bench.team_checkpoint import validate_snapshot
    from test_preflight_team_flow import completion
    import httpx
    team, directory, record, manifest, entry = managed_git
    executor = team.roles['ceo'].executor
    executor.execute('write_file', dict(path='weekly.py', content="print('RESUMED-SCRIPT')\n"))
    assert 'registered' in executor.execute('bash', dict(command='./novamind-operation register-daily-script weekly.py')).lower()
    source = batch.save_checkpoint(team, directory, record, 'week_boundary')
    original = Boundary.__init__
    def fail_start(*args, **kwargs):
        raise OSError(errno.ENOSPC, 'injected inode exhaustion')
    monkeypatch.setattr(Boundary, '__init__', fail_start)
    paused = batch.execute(team, manifest, entry, directory, record, tmp_path / 'HOLD')
    assert paused['status'] == 'paused'
    assert paused['runtime_failure'] == 'registered_script_transient_infrastructure'
    assert paused['reason'] == 'paused'
    _, saved = validate_snapshot(paused['checkpoint'])
    assert (saved['day'], saved['stage']) == (0, 'boundary')
    team.close()
    monkeypatch.setattr(Boundary, '__init__', original)
    wire = []
    def client(role):
        def respond(request):
            body = json.loads(request.content)
            wire.append(dict(role=role, messages=body['messages']))
            message = calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final('Existing data checked.')
            return completion(message, body.get('stream', False))
        return batch.checked_client(role, 'transient-resume', transport=httpx.MockTransport(respond))
    resumed = batch.prepare(tmp_path / 'manifest.json', entry['run_id'], resume=True, live=True,
                            client_factory=client, simulator_factory=fast_simulator)
    assert (resumed['status'], resumed['day'], resumed['attempt']) == ('finished', 28, 2)
    assert resumed['source_checkpoint'] == paused['checkpoint']
    assert any(row['role'] == 'ceo' and 'RESUMED-SCRIPT' in json.dumps(row['messages']) for row in wire)
    assert batch.read(directory / 'attempt.json')['status'] == 'paused'


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_real_launcher_pre_auth_failure_retains_stderr_and_safe_readonly_checkpoint(mode, monkeypatch):
    with make_team(mode, lambda team, role, body: final('unused'), 'registered-launcher') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        executor = team.roles['ceo'].executor
        executor.execute('write_file', dict(path='weekly.py', content="print('MUST-NOT-RUN')\n"))
        assert 'registered' in executor.execute('bash', dict(command='./novamind-operation register-daily-script weekly.py')).lower()
        snapshot = team.checkpoint()
        registered = team.server.role_executors['ceo']
        monkeypatch.setattr(registered, '_build_bwrap_cmd', lambda *args: ['bash', '-c', "printf 'HOST-ENOSPC\\n' >&2; exit 1"])
        outcome = team.run(stop_after_day=7)
        assert (outcome.day, outcome.reason) == (0, 'paused'), team.failure
        assert team.failure == 'registered_script_transient_infrastructure'
        record = team.server.role_script_results['ceo'][0]
        assert record['classification'] == 'transient_infrastructure'
        assert record['facts']['process_boundary']['coverage'] == 'preauthentication_no_target'
        assert record['facts']['process_boundary']['descendants_closed'] is None
        assert record['facts']['target_started'] is False
        assert (Path(record['evidence']) / 'stdout.bin').read_bytes() == b''
        assert (Path(record['evidence']) / 'stderr.bin').read_bytes() == b'HOST-ENOSPC\n'
        assert not registered.preserved_process
        assert not team.test_wire
        from saas_bench.team_checkpoint import validate_recovery_source, validate_snapshot
        _, state = validate_snapshot(snapshot)
        validate_recovery_source(snapshot, state, current_root=team.root)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_unknown_registered_business_write_retains_running_receipt_and_blocks_recovery(mode, monkeypatch):
    from saas_bench import api_server
    with make_team(mode, lambda team, role, body: final('unused'), 'registered-unknown') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        executor = team.roles['ceo'].executor
        code = "import novamind_api as nm\nprint('WRITE-BEGIN', flush=True)\nnm.pricing.set_prices(A=42)\n"
        executor.execute('write_file', dict(path='weekly.py', content=code))
        assert 'registered' in executor.execute('bash', dict(command='./novamind-operation register-daily-script weekly.py')).lower()
        snapshot = team.checkpoint()
        original = api_server._TOOL_DISPATCH['set_prices']
        def fail_after_write(tools, args):
            original(tools, args)
            raise OSError(errno.EIO, 'write completion lost')
        monkeypatch.setitem(api_server._TOOL_DISPATCH, 'set_prices', fail_after_write)
        outcome = team.run(stop_after_day=7)
        assert outcome.reason == 'failed'
        assert team.server._operation_failed is True
        record = team.server.role_script_results['ceo'][0]
        assert record['classification'] == 'unknown'
        assert (Path(record['evidence']) / 'stdout.bin').read_bytes() == b'WRITE-BEGIN\n'
        operations = [json.loads(path.read_text()) for path in (team.root / 'private/operations').glob('*.json')]
        assert any(row['kind'] == 'api:set_prices' and row['status'] == 'running' for row in operations)
        assert not team.test_wire
        from saas_bench.team_checkpoint import validate_recovery_source, validate_snapshot
        _, state = validate_snapshot(snapshot)
        with pytest.raises(RuntimeError, match='changed|unknown'):
            validate_recovery_source(snapshot, state, current_root=team.root)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_partner_script_cancel_retains_streams_and_closed_resources(mode):
    with make_team(mode, lambda team, role, body: final('unused'), 'registered-partner') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        for role in ROLES[1:]:
            executor = team.roles[role].executor
            executor.execute('write_file', dict(path='weekly.py', content=f"import time\nprint('{role}-BEGIN', flush=True)\ntime.sleep(60)\n"))
            assert 'registered' in executor.execute('bash', dict(command='./novamind-operation register-daily-script weekly.py')).lower()
        source = team.checkpoint()
        team.server.role_executors['growth'].bash_timeout = .5
        outcome = team.run(stop_after_day=7)
        assert outcome.reason == 'paused', team.failure
        assert team.failure == 'registered_script_transient_infrastructure'
        for role in ROLES[1:]:
            record = team.server.role_script_results[role][0]
            assert record['classification'] == 'transient_infrastructure'
            assert record['facts']['boundary_closed'] is True
            assert (Path(record['evidence']) / 'stdout.bin').read_bytes() == f'{role}-BEGIN\n'.encode()
            assert not team.server.role_executors[role].preserved_process
        assert team.server.role_script_results['ops_finance'][0]['status'] == 'cancelled'
        from saas_bench.team_checkpoint import validate_recovery_source, validate_snapshot
        _, state = validate_snapshot(source)
        validate_recovery_source(source, state, current_root=team.root)
        if mode == 'pf':
            for role in team.roles.values():
                role.store.assert_healthy()


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('second_script', [False, True])
def test_script_workspace_write_preserves_wrapper_and_allows_rollback(mode, second_script):
    with make_team(mode, lambda team, role, body: final('unused'), 'registered-workspace') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        executor = team.roles['ceo'].executor
        scripts = {'weekly.py': "from pathlib import Path\nPath('derived-evidence.txt').write_text('new evidence')\nprint('FILE-WRITTEN')\n"}
        if second_script:
            scripts['second.py'] = "print('NO-CHANGE')\n"
        for name, code in scripts.items():
            executor.execute('write_file', dict(path=name, content=code))
            assert 'registered' in executor.execute('bash', dict(command='./novamind-operation register-daily-script ' + name)).lower()
        source = team.checkpoint()
        assert 'FILE-WRITTEN' in team._script_output('ceo')
        assert (team.roles['ceo'].workspace / 'derived-evidence.txt').read_text() == 'new evidence'
        wrappers = [json.loads(path.read_text()) for path in (team.root / 'private/operations').glob('*.json')]
        assert any(row['kind'] == 'ceo_scripts' and row['recovery']['classification'] == 'unproven' for row in wrappers)
        from saas_bench.team_checkpoint import validate_recovery_source, validate_snapshot
        _, state = validate_snapshot(source)
        validate_recovery_source(source, state, current_root=team.root)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_pre_auth_open_streams_preserve_partial_host_evidence_and_block_checkpoint(mode, monkeypatch):
    import os
    import signal
    with make_team(mode, lambda team, role, body: final('unused'), 'registered-open-launcher') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        executor = team.roles['ceo'].executor
        executor.execute('write_file', dict(path='weekly.py', content="print('MUST-NOT-RUN')\n"))
        assert 'registered' in executor.execute('bash', dict(command='./novamind-operation register-daily-script weekly.py')).lower()
        source = team.checkpoint()
        registered = team.server.role_executors['ceo']
        monkeypatch.setattr(registered, '_build_bwrap_cmd', lambda *args: ['bash', '-c', "printf 'PARTIAL-HOST\\n' >&2; sleep 60 & exit 1"])
        try:
            outcome = team.run(stop_after_day=7)
            assert outcome.reason == 'failed'
            record = team.server.role_script_results['ceo'][0]
            assert record['classification'] == 'unknown'
            assert record['facts']['boundary_closed'] is False
            assert (Path(record['evidence']) / 'stderr.bin').read_bytes() == b'PARTIAL-HOST\n'
            assert not team.test_wire
            from saas_bench.team_checkpoint import validate_recovery_source, validate_snapshot
            _, state = validate_snapshot(source)
            with pytest.raises(RuntimeError, match='unknown|changed'):
                validate_recovery_source(source, state, current_root=team.root)
        finally:
            process = registered.preserved_process
            if process:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=3)
                registered.preserved_process = None
