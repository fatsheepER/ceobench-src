import errno
import json
import shlex
from pathlib import Path

import pytest

from saas_bench.role_policy import ROLES
from test_preflight_team_checkpoint import fast_simulator
from test_preflight_team_flow import ADVANCE, calls, final, make_team, opening_response


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
def test_prestart_enospc_reaches_owner_and_all_roles_continue(mode, monkeypatch):
    from saas_bench.process_boundary import Boundary
    turns = {}
    def responder(team, role, body):
        day = team.server.tools.current_day
        turn = turns[role, day] = turns.get((role, day), 0) + 1
        if day == 0 and turn == 1:
            return register_actions(role, "print('read-only')\n")
        if day == 7:
            assert 'No space left on device' in json.dumps(body['messages'])
            assert team.roles[role].week.scripts_status == 'script_error'
            if turn == 1:
                return calls(('bash', dict(command='echo NEXT-' + role)))
            assert 'NEXT-' + role in body['messages'][-1]['content']
        return final(role + ' advice') if role != 'ceo' else calls(('bash', dict(command=ADVANCE)))
    with make_team(mode, responder, 'registered-enospc') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        snapshots = []
        def checkpoint(runtime, boundary):
            snapshots.append(runtime.checkpoint())
        team.checkpoint_callback = checkpoint
        original = Boundary.__init__
        def fail_at_week(self, *args, **kwargs):
            if team.server.tools.current_day == 7 and '/.registered-' in args[0]:
                raise OSError(errno.ENOSPC, 'No space left on device')
            return original(self, *args, **kwargs)
        monkeypatch.setattr(Boundary, '__init__', fail_at_week)
        outcome = team.run(stop_after_day=14)
        assert (outcome.day, outcome.reason) == (14, 'observation_end'), team.failure
        assert team.failure is None
        for role in ROLES:
            record = team.server.role_script_results[role][0]
            assert record['classification'] == 'script_error'
            assert record['facts']['process_started'] is False
            assert record['facts']['boundary_closed'] is True
            assert (Path(record['evidence']) / 'stdout.bin').read_bytes() == b''
            assert (Path(record['evidence']) / 'stderr.bin').read_bytes() == b''
            assert turns[role, 7] == 2
            assert team.roles[role].week.weekly_status == 'completed'
        from saas_bench.team_checkpoint import validate_snapshot
        _, state = validate_snapshot(snapshots[-1])
        assert state['day'] == 14


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_readonly_script_keeps_evidence_without_business_wrapper(mode):
    with make_team(mode, lambda team, role, body: final('unused'), 'registered-readonly') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        role = team.roles['ceo']
        result = role.executor.execute('write_file', dict(path='weekly.py', content='import novamind_api as nm\nprint(nm.vars.current_day)\n'))
        assert 'Error:' not in result
        assert 'registered' in role.executor.execute('bash', dict(command='./novamind-operation register-daily-script weekly.py')).lower()
        assert '0' in team._script_output('ceo')
        operations = [json.loads(path.read_text()) for path in (team.root / 'private/operations').glob('*.json')]
        assert not any(row['kind'] == 'ceo_scripts' for row in operations)
        record = team.server.role_script_results['ceo'][0]
        assert record['classification'] == 'succeeded'
        assert (Path(record['evidence']) / 'stdout.bin').read_bytes() == b'0\n'
        assert not list(role.workspace.rglob('__pycache__'))


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('role', ROLES)
@pytest.mark.parametrize('failure', ['timeout', 'killed'])
def test_closed_resource_failure_reaches_owner_and_partner_models(mode, role, failure):
    code = "import time, os, signal\nprint('PARTIAL-OUT', flush=True)\n"
    code += "time.sleep(60)\n" if failure == 'timeout' else "os.kill(os.getpid(), signal.SIGKILL)\n"
    turns = {}
    def responder(team, sender, body):
        turns[sender] = turns.get(sender, 0) + 1
        if sender == role:
            assert 'PARTIAL-OUT' in json.dumps(body['messages'])
            assert team.roles[role].week.scripts_status == 'script_error'
            assert ('timed out' if failure == 'timeout' else 'exit code') in json.dumps(body['messages'])
        if turns[sender] == 1:
            return calls(('bash', dict(command='echo NEXT-' + sender)))
        assert 'NEXT-' + sender in body['messages'][-1]['content']
        return final(sender + ' advice') if sender != 'ceo' else calls(('bash', dict(command=ADVANCE)))
    with make_team(mode, responder, 'registered-resource') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        executor = team.roles[role].executor
        assert 'Error:' not in executor.execute('write_file', dict(path='weekly.py', content=code))
        assert 'registered' in executor.execute('bash', dict(command='./novamind-operation register-daily-script weekly.py')).lower()
        script_timeout = team.server.role_executors[role].bash_timeout
        if failure == 'timeout':
            team.server.role_executors[role].bash_timeout = .3
        outcome = team.run(stop_after_day=7)
        team.server.role_executors[role].bash_timeout = script_timeout
        assert (outcome.day, outcome.reason) == (7, 'observation_end'), team.failure
        assert team.failure is None
        record = team.server.role_script_results[role][0]
        assert record['classification'] == 'script_error'
        assert record['facts']['target_started'] is True
        assert record['facts']['boundary_closed'] is True
        assert (Path(record['evidence']) / 'stdout.bin').read_bytes() == b'PARTIAL-OUT\n'
        assert all(not executor.preserved_process for executor in team.server.role_executors.values())
        assert turns == {peer: 2 for peer in ROLES}
        assert team.roles[role].week.scripts_status == 'script_error'
        assert all(runtime.week.weekly_status == 'completed' for runtime in team.roles.values())
        assert team.checkpoint().is_dir()


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_model_script_retains_partial_business_write_and_error(mode):
    turns = []
    code = "import novamind_api as nm\nnm.pricing.set_prices(A=37)\nprint('WRITE-DONE')\nraise ValueError('repair me')\n"

    def respond(team, role, body):
        if role != 'ceo':
            return final(role + ' report')
        turns.append(body)
        if len(turns) == 1:
            return calls(('bash', dict(command='python weekly.py')))
        rendered = json.dumps(body['messages'])
        assert 'WRITE-DONE' in rendered and 'ValueError: repair me' in rendered
        return calls(('bash', dict(command=ADVANCE)))

    with make_team(mode, respond, 'model-script-write') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        team.roles['ceo'].executor.execute('write_file', dict(path='weekly.py', content=code))
        assert team.run(stop_after_day=7).reason == 'observation_end', team.failure
        assert len(turns) == 2
        assert not team.server._operation_failed
        assert team.server.conn.execute('SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] == 37


def test_permanent_pf_capture_fault_retains_raw_evidence_and_prior_checkpoint(monkeypatch):
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
        assert [item['role'] for item in team.test_wire] == ['ceo']


from test_preflight_team_readonly_recovery import managed_git
from test_preflight_team_batch import frozen


@pytest.mark.parametrize('managed_git', ['git', 'pf'], indirect=True)
def test_script_receipt_failure_holds_original_attempt_and_resumes_checkpoint(managed_git, tmp_path, monkeypatch):
    from saas_bench import run_state, team_batch as batch
    from saas_bench.team_checkpoint import validate_snapshot
    from test_preflight_team_flow import completion
    import httpx
    team, directory, record, manifest, entry = managed_git
    executor = team.roles['ceo'].executor
    executor.execute('write_file', dict(path='weekly.py', content="print('RESUMED-SCRIPT')\n"))
    assert 'registered' in executor.execute('bash', dict(command='./novamind-operation register-daily-script weekly.py')).lower()
    source = batch.save_checkpoint(team, directory, record, 'week_boundary')
    ceo = team.roles['ceo']
    ceo.agent.client.close()
    ceo.agent.client = batch.checked_client('ceo', 'transient-opening', transport=httpx.MockTransport(
        lambda request: opening_response('ceo', json.loads(request.content))))
    ceo.usage.attach(ceo.agent.client)
    original = run_state.write_json
    def fail_script_receipt(path, value):
        if Path(path).name == 'execution.json':
            raise OSError(errno.EIO, 'injected script receipt persistence failure')
        return original(path, value)
    monkeypatch.setattr(run_state, 'write_json', fail_script_receipt)
    paused = batch.execute(team, manifest, entry, directory, record, tmp_path / 'HOLD')
    assert paused['status'] == 'paused'
    assert 'injected script receipt persistence failure' in paused['runtime_failure']
    assert paused['hold'] is True
    assert paused['reason'] == 'paused'
    assert paused['checkpoint'] == str(source)
    assert any(path.read_bytes() == b'RESUMED-SCRIPT\n'
               for path in (team.root / 'private/ceo/scripts').glob('*/stdout.bin'))
    assert not any(executor.preserved_process for executor in team.server.role_executors.values())
    _, saved = validate_snapshot(paused['checkpoint'])
    assert (saved['day'], saved['stage']) == (0, 'boundary')
    team.close()
    monkeypatch.setattr(run_state, 'write_json', original)
    wire = []
    def client(role):
        def respond(request):
            body = json.loads(request.content)
            wire.append(dict(role=role, messages=body['messages']))
            if response := opening_response(role, body):
                return response
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
def test_real_launcher_pre_auth_failure_retains_stderr_and_closes_resources(mode, monkeypatch):
    def responder(team, role, body):
        if role == 'ceo':
            assert 'HOST-ENOSPC' in json.dumps(body['messages'])
            assert team.roles[role].week.scripts_status == 'script_error'
            return calls(('bash', dict(command=ADVANCE)))
        return final(role + ' advice')
    with make_team(mode, responder, 'registered-launcher') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        executor = team.roles['ceo'].executor
        executor.execute('write_file', dict(path='weekly.py', content="print('MUST-NOT-RUN')\n"))
        assert 'registered' in executor.execute('bash', dict(command='./novamind-operation register-daily-script weekly.py')).lower()
        registered = team.server.role_executors['ceo']
        monkeypatch.setattr(registered, '_build_bwrap_cmd', lambda *args: ['bash', '-c', "printf 'HOST-ENOSPC\\n' >&2; exit 1"])
        outcome = team.run(stop_after_day=7)
        assert (outcome.day, outcome.reason) == (7, 'observation_end'), team.failure
        assert team.failure is None
        record = team.server.role_script_results['ceo'][0]
        assert record['classification'] == 'script_error'
        assert record['facts']['process_boundary']['coverage'] == 'preauthentication_no_target'
        assert record['facts']['process_boundary']['descendants_closed'] is None
        assert record['facts']['target_started'] is False
        assert (Path(record['evidence']) / 'stdout.bin').read_bytes() == b''
        assert (Path(record['evidence']) / 'stderr.bin').read_bytes() == b'HOST-ENOSPC\n'
        assert not registered.preserved_process
        assert set(item['role'] for item in team.test_wire) == set(ROLES)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_unknown_model_script_write_retains_evidence_and_skips_followup(mode, monkeypatch):
    from saas_bench import api_server
    code = "import novamind_api as nm\nprint('WRITE-BEGIN', flush=True)\nnm.pricing.set_prices(A=42)\n"

    def respond(team, role, body):
        if role == 'ceo':
            return calls(('bash', dict(command='python -c ' + shlex.quote(code))))
        return final(role + ' report')

    with make_team(mode, respond, 'model-script-unknown') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        original = api_server._TOOL_DISPATCH['set_prices']

        def fail_after_write(tools, args):
            original(tools, args)
            raise OSError(errno.EIO, 'write completion lost')

        monkeypatch.setitem(api_server._TOOL_DISPATCH, 'set_prices', fail_after_write)
        outcome = team.run(stop_after_day=7)
        assert (outcome.day, outcome.reason, team.failure) == (0, 'paused', 'business_operation_unknown')
        assert team.server._operation_failed is True
        executor = team.roles['ceo'].executor
        assert executor._raw_streams[0] == b'WRITE-BEGIN\n'
        assert executor._bash_facts['boundary_closed'] is True
        assert not executor.preserved_process
        operations = [json.loads(path.read_text()) for path in (team.root / 'private/operations').glob('*.json')]
        assert any(row['kind'] == 'api:set_prices' and row['status'] == 'running' for row in operations)
        assert [item['role'] for item in team.test_wire].count('ceo') == 2
        assert team.server.role_script_results['ceo'] == []


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_unknown_model_partner_write_cancels_script_and_retains_closed_partial_streams(mode, monkeypatch):
    import time
    from saas_bench import api_server
    model_code = "print('growth-BEGIN', flush=True)\nimport novamind_api as n\nn.infrastructure.set_capacity_tier(2)\n"

    def respond(team, role, body):
        assert role == 'growth'
        return calls(('bash', dict(command='python -c ' + shlex.quote(model_code))))

    with make_team(mode, respond, 'registered-partner') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        executor = team.roles['ops_finance'].executor
        code = ("print('ops_finance-BEGIN', flush=True)\nimport time\nfrom pathlib import Path\n"
                "Path('partner-started').touch()\ntime.sleep(60)\n")
        executor.execute('write_file', dict(path='weekly.py', content=code))
        assert 'registered' in executor.execute('bash', dict(command='./novamind-operation register-daily-script weekly.py')).lower()
        write = api_server._TOOL_DISPATCH['set_capacity_tier']

        def lose_write_completion(tools, args):
            deadline = time.monotonic() + 5
            while not (team.roles['ops_finance'].workspace / 'partner-started').exists():
                assert time.monotonic() < deadline, 'Partner script did not start'
                time.sleep(.01)
            write(tools, args)
            raise OSError(errno.EIO, 'partner write completion lost')

        monkeypatch.setitem(api_server._TOOL_DISPATCH, 'set_capacity_tier', lose_write_completion)
        outcome = team.run(stop_after_day=7)
        assert outcome.reason in ('paused', 'failed'), team.failure
        assert 'unknown' in team.failure.lower()
        assert team.server._operation_failed is True
        growth = team.roles['growth'].executor
        assert growth._raw_streams[0] == b'growth-BEGIN\n'
        assert growth._bash_facts['boundary_closed'] is True
        assert not growth.preserved_process
        record, = team.server.role_script_results['ops_finance']
        assert record['classification'] == 'unknown'
        assert record['facts']['business_operation_unknown'] is True
        assert record['facts']['boundary_closed'] is True
        assert (Path(record['evidence']) / 'stdout.bin').read_bytes() == b'ops_finance-BEGIN\n'
        assert not team.server.role_executors['ops_finance'].preserved_process
        assert record['status'] == 'cancelled'
        assert [item['role'] for item in team.test_wire] == ['ceo', 'growth']
        with pytest.raises(RuntimeError, match='recoverable handoff'):
            team.checkpoint()
        if mode == 'pf':
            store = team.roles['ops_finance'].store
            assert store.fault is None, store.fault
            fault = json.loads(store.fault_path.read_text())
            assert fault['reason'] == 'Execution outcome unknown; branch paused'
            with pytest.raises(RuntimeError, match='collection stopped'):
                store.assert_healthy()


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('second_script', [False, True])
def test_script_workspace_write_preserves_file_and_wrapper(mode, second_script):
    with make_team(mode, lambda team, role, body: final('unused'), 'registered-workspace') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        executor = team.roles['ceo'].executor
        scripts = {'weekly.py': "from pathlib import Path\nPath('derived-evidence.txt').write_text('new evidence')\nprint('FILE-WRITTEN')\n"}
        if second_script:
            scripts['second.py'] = "print('NO-CHANGE')\n"
        for name, code in scripts.items():
            executor.execute('write_file', dict(path=name, content=code))
            assert 'registered' in executor.execute('bash', dict(command='./novamind-operation register-daily-script ' + name)).lower()
        assert 'FILE-WRITTEN' in team._script_output('ceo')
        assert (team.roles['ceo'].workspace / 'derived-evidence.txt').read_text() == 'new evidence'
        wrappers = [json.loads(path.read_text()) for path in (team.root / 'private/operations').glob('*.json')]
        assert any(row['kind'] == 'ceo_scripts' and row['recovery']['classification'] == 'unproven' for row in wrappers)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_pre_auth_open_streams_preserve_partial_host_evidence_and_skip_model(mode, monkeypatch):
    import os
    import signal
    with make_team(mode, lambda team, role, body: final('unused'), 'registered-open-launcher') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        executor = team.roles['ceo'].executor
        executor.execute('write_file', dict(path='weekly.py', content="print('MUST-NOT-RUN')\n"))
        assert 'registered' in executor.execute('bash', dict(command='./novamind-operation register-daily-script weekly.py')).lower()
        registered = team.server.role_executors['ceo']
        monkeypatch.setattr(registered, '_build_bwrap_cmd', lambda *args: ['bash', '-c', "printf 'PARTIAL-HOST\\n' >&2; sleep 60 & exit 1"])
        try:
            outcome = team.run(stop_after_day=7)
            assert (outcome.day, outcome.reason, team.failure) == (0, 'paused', 'business_operation_unknown')
            record = team.server.role_script_results['ceo'][0]
            assert record['classification'] == 'unknown'
            assert record['facts']['boundary_closed'] is False
            assert (Path(record['evidence']) / 'stderr.bin').read_bytes() == b'PARTIAL-HOST\n'
            assert [item['role'] for item in team.test_wire] == ['ceo']
        finally:
            process = registered.preserved_process
            if process:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=3)
                registered.preserved_process = None
