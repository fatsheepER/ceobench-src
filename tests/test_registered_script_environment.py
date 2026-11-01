import hashlib
import json
from pathlib import Path

import pytest

from saas_bench.role_policy import ROLES
from test_preflight_team_checkpoint import fast_simulator
from test_preflight_team_flow import ADVANCE, calls, final, make_team, raw


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('name', ['weekly.py', 'weekly_scripts/weekly.py'])
def test_automatic_scripts_use_real_snapshot_beside_owner_data_and_continue(mode, name):
    turns = {}
    received = {}

    def responder(team, role, body):
        turns[role] = turns.get(role, 0) + 1
        received.setdefault(role, []).append(json.dumps(body['messages']))
        if turns[role] == 1:
            return calls(('bash', dict(command='echo NEXT-' + role)))
        return calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final(role + ' advice')

    code = """from pathlib import Path
import hashlib
import json
import sys
import __main__
import adjacent_module
import novamind_api as nm
source = Path(__file__)
print('ENV=' + json.dumps(dict(file=str(source), cwd=str(Path.cwd()), argv=sys.argv,
    main=__name__, main_file=__main__.__file__, module=adjacent_module.VALUE,
    data=(source.parent / 'data.txt').read_text(), relative=Path('root-data.txt').read_text(),
    sha256=hashlib.sha256(source.read_bytes()).hexdigest(), day=nm.vars.current_day)))
print('COMPLETE-' + 'x' * 31000)
"""
    digest = hashlib.sha256(code.encode()).hexdigest()
    with make_team(mode, responder, 'registered-environment') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        for role in ROLES:
            executor = team.roles[role].executor
            parent = Path(name).parent
            for path, content in ((name, code), (str(parent / 'data.txt'), 'DATA-' + role),
                                  (str(parent / 'adjacent_module.py'), "VALUE = 'MODULE-' + " + repr(role)),
                                  ('root-data.txt', 'ROOT-' + role),
                                  ('failed.py', "import sys\nprint('PARTIAL-' + " + repr(role) + ")\nprint('ERR-END', file=sys.stderr)\nsys.exit(3)\n"),
                                  ('following.py', "print('FOLLOWING-' + " + repr(role) + ")\n")):
                assert 'Error:' not in executor.execute('write_file', dict(path=path, content=content))
            for script in (name, 'failed.py', 'following.py'):
                receipt = executor.execute('bash', dict(command='./novamind-operation register-daily-script ' + script))
                assert '"registered": true' in receipt, receipt
            live = team.roles[role].workspace / name
            if role == 'growth':
                live.unlink()
            else:
                live.write_text("raise AssertionError('LIVE-EDIT-MUST-NOT-EXECUTE')\n")

        outcome = team.run(stop_after_day=7)
        assert (outcome.day, outcome.reason, team.failure) == (7, 'observation_end', None)
        for role in ROLES:
            records = team.server.role_script_results[role]
            assert records[0]['classification'] == 'succeeded', records[0]['output']
            assert [record['name'] for record in records] == [name, 'failed.py', 'following.py']
            assert [record['classification'] for record in records] == ['succeeded', 'script_error', 'succeeded']
            evidence = Path(records[0]['evidence'])
            stdout = (evidence / 'stdout.bin').read_text()
            env = json.loads(stdout.splitlines()[0].removeprefix('ENV='))
            source = Path(env['file'])
            workspace = team.roles[role].workspace
            assert source.parent == (workspace / name).parent
            assert source != workspace / name
            assert env == dict(file=str(source), cwd=str(workspace), argv=[str(source)],
                               main='__main__', main_file=str(source), module='MODULE-' + role,
                               data='DATA-' + role, relative='ROOT-' + role, sha256=digest, day=0)
            assert records[0]['execution_path'] == str(source)
            assert records[0]['cwd'] == str(workspace)
            assert records[0]['sha256'] == digest
            assert (evidence / 'code.py').read_text() == code
            assert stdout.endswith('COMPLETE-' + 'x' * 31000 + '\n')
            assert (evidence / 'stderr.bin').read_bytes() == b''
            assert not source.exists()
            assert not list(workspace.rglob('__pycache__'))
            live = workspace / name
            if role == 'growth':
                assert not live.exists()
            else:
                assert 'LIVE-EDIT' in live.read_text()
            assert all(record['facts']['boundary_closed'] is True for record in records)
            assert all(not executor.preserved_process for executor in team.server.role_executors.values())
            assert turns[role] == 2
            assert team.roles[role].week.scripts_status == 'script_error'
            assert team.roles[role].week.weekly_status == 'completed'
            delivered = received[role][0]
            assert 'COMPLETE-' + 'x' * 31000 in delivered
            assert 'PARTIAL-' + role in delivered and 'ERR-END' in delivered
            assert 'FOLLOWING-' + role in delivered
            assert 'NEXT-' + role in received[role][1]
            assert all('DATA-' + peer not in delivered for peer in ROLES if peer != role)
            usage = [json.loads(line) for line in team.roles[role].usage.path.read_text().splitlines()]
            assert any('COMPLETE-' + 'x' * 31000 in json.dumps(row) for row in usage)
        assert team.checkpoint().is_dir()


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_registration_paths_stay_inside_owner_workspace(mode):
    with make_team(mode, lambda team, role, body: final('unused'), 'registered-paths') as team:
        executor = team.roles['ceo'].executor
        executor.execute('write_file', dict(path='nested/weekly.py', content="print('NESTED')\n"))
        output = executor.execute('bash', dict(command='cd nested && ../novamind-operation register-daily-script weekly.py'))
        assert '"name": "nested/weekly.py"' in output, output
        assert team.server.get_daily_scripts('ceo') == {'nested/weekly.py': "print('NESTED')\n"}
        for name in ('../escape.py', '/absolute.py', 'nested/../escape.py', '.', 'nested/', 'nul\x00.py'):
            status, response = raw(team, 'ceo', '/daily-scripts', dict(name=name, content="print('BAD')\n"))
            assert status == 400 and response['success'] is False, (name, status, response)
        foreign = team.roles['growth'].workspace / 'foreign.py'
        foreign.write_text("print('FOREIGN')\n")
        output = executor.execute('bash', dict(command='./novamind-operation register-daily-script ' + str(foreign)))
        assert 'Error:' in output and 'owner workspace' in output, output
        assert team.server.get_daily_scripts('ceo') == {'nested/weekly.py': "print('NESTED')\n"}


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('closed', [True, False])
def test_snapshot_cleanup_waits_for_proven_process_closure(mode, closed, monkeypatch):
    import os
    import signal

    def responder(team, role, body):
        return calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final(role + ' advice')

    with make_team(mode, responder, 'registered-snapshot-lifetime') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        code = "import time\nprint('STARTED', flush=True)\ntime.sleep(60)\n"
        executor = team.roles['ceo'].executor
        executor.execute('write_file', dict(path='weekly.py', content=code))
        receipt = executor.execute('bash', dict(command='./novamind-operation register-daily-script weekly.py'))
        assert '"registered": true' in receipt
        registered = team.server.role_executors['ceo']
        registered.bash_timeout = .3
        if not closed:
            monkeypatch.setattr(registered, '_build_bwrap_cmd', lambda *args:
                ['bash', '-c', "printf 'PARTIAL-HOST\\n' >&2; sleep 60 & exit 1"])
        try:
            outcome = team.run(stop_after_day=7)
            record = team.server.role_script_results['ceo'][0]
            source = Path(record['execution_path'])
            assert record['facts']['boundary_closed'] is closed
            if closed:
                assert (outcome.day, outcome.reason, team.failure) == (7, 'observation_end', None)
                assert record['classification'] == 'script_error'
                assert record['facts']['timed_out'] is True
                assert (Path(record['evidence']) / 'stdout.bin').read_bytes() == b'STARTED\n'
                assert not source.exists()
                assert not registered.preserved_process
                assert team.checkpoint().is_dir()
            else:
                assert (outcome.day, outcome.reason, team.failure) == (0, 'paused', 'business_operation_unknown')
                assert record['classification'] == 'unknown'
                assert source.read_text() == code
                assert (Path(record['evidence']) / 'stderr.bin').read_bytes() == b'PARTIAL-HOST\n'
                assert registered.preserved_process is not None
                assert [item['role'] for item in team.test_wire] == ['ceo']
        finally:
            if registered.preserved_process:
                process = registered.preserved_process
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=3)
                registered.preserved_process = None


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('stage', ['create', 'write'])
def test_snapshot_preparation_failure_reaches_owner_and_all_roles_continue(mode, stage, monkeypatch):
    import errno
    import os
    from saas_bench import api_server

    received, turns = {}, {}

    def responder(team, role, body):
        turns[role] = turns.get(role, 0) + 1
        received.setdefault(role, []).append(json.dumps(body['messages']))
        if turns[role] == 1:
            return calls(('bash', dict(command='echo NEXT-' + role)))
        return calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final(role + ' advice')

    with make_team(mode, responder, 'registered-source-failure') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        executor = team.roles['ceo'].executor
        for name, code in (('weekly.py', "print('MUST-NOT-RUN')\n"), ('following.py', "print('FOLLOWING')\n")):
            executor.execute('write_file', dict(path=name, content=code))
            assert '"registered": true' in executor.execute('bash', dict(command='./novamind-operation register-daily-script ' + name))
        injected = []
        if stage == 'create':
            original = os.open

            def fail_create(path, flags, *args, **kwargs):
                if str(path).startswith('.registered-') and not injected:
                    injected.append(stage)
                    raise OSError(errno.ENOSPC, 'injected snapshot create failure')
                return original(path, flags, *args, **kwargs)

            monkeypatch.setattr(api_server.os, 'open', fail_create)
        else:
            original = os.fdopen

            class FailedWrite:
                def __init__(self, stream):
                    self.stream = stream

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return self.stream.__exit__(*args)

                def write(self, content):
                    self.stream.write(content[:5])
                    self.stream.flush()
                    raise OSError(errno.ENOSPC, 'injected snapshot write failure')

            def fail_write(descriptor, *args, **kwargs):
                stream = original(descriptor, *args, **kwargs)
                if '/.registered-' in os.readlink(f'/proc/self/fd/{descriptor}') and not injected:
                    injected.append(stage)
                    return FailedWrite(stream)
                return stream

            monkeypatch.setattr(api_server.os, 'fdopen', fail_write)

        outcome = team.run(stop_after_day=7)
        assert injected == [stage]
        assert (outcome.day, outcome.reason, team.failure) == (7, 'observation_end', None)
        failed, following = team.server.role_script_results['ceo']
        assert failed['classification'] == 'script_error'
        assert failed['facts']['process_started'] is False
        assert failed['facts']['target_started'] is False
        assert failed['facts']['boundary_closed'] is True
        assert failed['facts']['error_errno'] == errno.ENOSPC
        assert (Path(failed['evidence']) / 'stdout.bin').read_bytes() == b''
        assert (Path(failed['evidence']) / 'stderr.bin').read_bytes() == b''
        assert following['classification'] == 'succeeded'
        assert 'injected snapshot ' + stage + ' failure' in received['ceo'][0]
        assert 'FOLLOWING' in received['ceo'][0]
        assert all(turns[role] == 2 and 'NEXT-' + role in received[role][1] for role in ROLES)
        assert all(not executor.preserved_process for executor in team.server.role_executors.values())
        assert not list(team.roles['ceo'].workspace.rglob('.registered-*.py'))
        assert team.roles['ceo'].week.scripts_status == 'script_error'
        assert team.checkpoint().is_dir()


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_snapshot_cleanup_failure_preserves_source_and_stops_batch(mode, monkeypatch):
    import errno
    import os
    from saas_bench import api_server

    with make_team(mode, lambda team, role, body: final('unused'), 'registered-source-cleanup') as team:
        executor = team.roles['ceo'].executor
        code = "print('EXECUTED-BEFORE-CLEANUP')\n"
        for name, content in (('weekly.py', code), ('following.py', "print('MUST-NOT-FOLLOW')\n")):
            executor.execute('write_file', dict(path=name, content=content))
            assert '"registered": true' in executor.execute('bash', dict(command='./novamind-operation register-daily-script ' + name))
        original = os.unlink

        def fail_unlink(path, *args, **kwargs):
            if str(path).startswith('.registered-'):
                raise OSError(errno.EIO, 'injected snapshot cleanup failure')
            return original(path, *args, **kwargs)

        monkeypatch.setattr(api_server.os, 'unlink', fail_unlink)
        outcome = team.run(stop_after_day=7)
        assert (outcome.day, outcome.reason) == (0, 'failed')
        assert 'injected snapshot cleanup failure' in team.failure
        record, = team.server.role_script_results['ceo']
        assert record['facts']['boundary_closed'] is True
        assert (Path(record['evidence']) / 'stdout.bin').read_bytes() == b'EXECUTED-BEFORE-CLEANUP\n'
        assert Path(record['execution_path']).read_text() == code
        assert [item['role'] for item in team.test_wire] == ['ceo']
        assert all(not executor.preserved_process for executor in team.server.role_executors.values())
        with pytest.raises(RuntimeError):
            team.checkpoint()
