import json
import os
import resource
import shlex
from pathlib import Path
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
from test_preflight_team_observer import observer
from test_preflight_team_controller import controller
from test_execution_capture import captured
from test_public_sql import server
from test_sql_evidence import event_ids


@pytest.fixture(autouse=True)
def require_test_memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    assert 0 < soft <= 2 * 1024**3 and 0 < hard <= 2 * 1024**3, 'Run this suite under prlimit --as=2147483648'


@pytest.mark.parametrize('sandboxed', [False, True])
@pytest.mark.parametrize('verified', [False, True])
def test_bash_huge_dictionary_is_limited_and_next_command_works(tmp_path, monkeypatch, verified, sandboxed):
    ceiling = 64 * 1024**2
    executor = BashAgentToolExecutor(tmp_path, bash_memory_limit_bytes=ceiling,
        bash_timeout=10, bash_warn_after=5 if verified else None)
    if sandboxed:
        if executor._bwrap() is None:
            pytest.skip('bubblewrap required')
    else:
        monkeypatch.setattr(executor, '_bwrap', lambda: None)
    script = '''import resource
limit = resource.getrlimit(resource.RLIMIT_AS)[1]
assert 0 < limit <= 67108864, 'missing command memory guard; allocation skipped'
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
dict.fromkeys(range(10**9))
'''
    (tmp_path / 'huge.py').write_text(script)
    with ThreadPoolExecutor(max_workers=1) as pool:
        output = pool.submit(executor.execute, 'bash', dict(command='python huge.py')).result(timeout=15)
    assert 'MemoryError' in output, output
    assert 'allocation failure' in output and '67108864 bytes' in output, output
    assert executor.last_status == 'failed'
    assert executor._bash_facts['process_boundary']['memory_limit_bytes'] == ceiling
    assert executor._bash_facts['boundary_closed']
    assert executor.preserved_process is None
    assert executor.execute('bash', dict(command='echo next')) == 'next\n'


def test_bash_memory_environment_and_inherited_hard_limit(tmp_path, monkeypatch):
    monkeypatch.setenv('CEOBENCH_BASH_MEMORY_MAX_BYTES', '67108864')
    executor = BashAgentToolExecutor(tmp_path)
    monkeypatch.setattr(executor, '_bwrap', lambda: None)
    output = executor.execute('bash', dict(command=
        "python -c 'import resource; print(resource.getrlimit(resource.RLIMIT_AS))'"))
    assert output == '(67108864, 67108864)\n'
    script = '''import resource, subprocess, sys
from saas_bench.process_boundary import Boundary
resource.setrlimit(resource.RLIMIT_AS, (67108864, 67108864))
b = Boundary("python -c 'import resource; print(resource.getrlimit(resource.RLIMIT_AS))'", memory_limit_bytes=134217728)
try:
    p = subprocess.Popen(['bash', '-c', b.command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = b.communicate(p, 5)
    assert p.returncode == 0 and out == b'(67108864, 67108864)\\n', (out, err)
    assert b.record['memory_limit_bytes'] == 67108864
finally:
    b.close()
'''
    subprocess.run([sys.executable, '-c', script], check=True, timeout=10)


def test_default_guard_keeps_scientific_imports_working(tmp_path):
    executor = BashAgentToolExecutor(tmp_path, require_sandbox=True)
    code = '''import os, numpy as np, pandas, sklearn
print(np.array([1, 2]).dot(np.array([3, 4])))
print(os.environ['OPENBLAS_NUM_THREADS'], os.environ['OMP_NUM_THREADS'])
'''
    assert executor.execute('bash', dict(command=shlex.join([executor.python, '-c', code]))) == '11\n1 1\n'


def test_explicit_command_thread_settings_are_preserved(tmp_path, monkeypatch):
    executor = BashAgentToolExecutor(tmp_path, env=dict(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2'))
    monkeypatch.setattr(executor, '_bwrap', lambda: None)
    assert executor.execute('bash', dict(command='echo "$OPENBLAS_NUM_THREADS $OMP_NUM_THREADS"')) == '2 2\n'


def test_memory_failure_keeps_capture_and_allows_next_tool(captured):
    _, store, executor = captured
    executor.bash_memory_limit_bytes = 64 * 1024**2
    executor.bash_timeout, executor.bash_warn_after = 10, 5
    code = '''import resource
assert 0 < resource.getrlimit(resource.RLIMIT_AS)[1] <= 67108864, 'unsafe allocation skipped'
dict.fromkeys(range(10**9))
'''
    output = executor.execute('bash', dict(command=shlex.join([executor.python, '-c', code])))
    assert 'MemoryError' in output and '67108864 bytes' in output
    event = store.read_event(event_ids(store)[0])
    assert event['result']['status'] == 'failed'
    assert event['result']['process_boundary']['memory_limit_bytes'] == 67108864
    assert event['result']['process_boundary']['descendants_closed']
    assert executor.execute('bash', dict(command='echo next')) == 'next\n'
    store.assert_healthy()


@pytest.mark.parametrize('value', [-1, True, 1.5, 'bad'])
def test_invalid_bash_memory_limit_is_rejected(tmp_path, value):
    with pytest.raises(ValueError, match='memory'):
        BashAgentToolExecutor(tmp_path, bash_memory_limit_bytes=value)


def test_scope_command_has_separate_units_and_memory_properties():
    from saas_bench.systemd_guard import scope_command
    commands = [scope_command(['python', 'worker.py'], f'ceobench-test-{role}.scope',
        memory_high=128 * 1024**2, memory_max=256 * 1024**2, memory_swap_max=0)
        for role in ('git', 'pf')]
    assert commands[0] != commands[1]
    for command in commands:
        assert command[:3] == ['systemd-run', '--user', '--scope']
        assert '--slice=ceobench.slice' in command
        assert '--property=ManagedOOMPreference=omit' in command
        assert '--property=MemoryHigh=134217728' in command
        assert '--property=MemoryMax=268435456' in command
        assert '--property=MemorySwapMax=0' in command
        assert '--property=OOMPolicy=kill' in command
        assert command[-2:] == ['python', 'worker.py']


@pytest.mark.parametrize('current_id', ['', 'old'])
def test_journal_oom_overrides_unloaded_or_wrong_invocation_success(monkeypatch, current_id):
    from saas_bench import systemd_guard
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        stdout = (f'LoadState=not-found\nActiveState=inactive\nResult=success\nInvocationID={current_id}\n'
            if command[0] == 'systemctl' else '\n'.join(json.dumps(row) for row in [
                dict(USER_UNIT='work.service', USER_INVOCATION_ID='old', UNIT_RESULT='success'),
                dict(USER_UNIT='work.service', USER_INVOCATION_ID='current', UNIT_RESULT='oom-kill')]))
        return subprocess.CompletedProcess(command, 0, stdout, '')
    monkeypatch.setattr(systemd_guard.subprocess, 'run', run)
    result = systemd_guard.unit_status('work.service', invocation_id='current', exit_code=0)
    assert result['result'] == 'oom-kill'
    assert result['invocation_id'] == 'current'
    assert result['source'] == 'journal'
    assert len(calls) == 2


def test_unit_status_does_not_treat_missing_evidence_as_success(monkeypatch):
    from saas_bench import systemd_guard
    def run(command, **kwargs):
        stdout = 'LoadState=not-found\nActiveState=inactive\nResult=success\n' if command[0] == 'systemctl' else ''
        return subprocess.CompletedProcess(command, 0, stdout, '')
    monkeypatch.setattr(systemd_guard.subprocess, 'run', run)
    assert systemd_guard.unit_status('gone.scope', exit_code=0)['result'] == 'unknown'


def test_unit_status_ignores_older_oom_and_command_text(monkeypatch):
    from saas_bench import systemd_guard
    def run(command, **kwargs):
        stdout = ('LoadState=loaded\nActiveState=active\nResult=success\nInvocationID=current\n'
            if command[0] == 'systemctl' else '\n'.join(json.dumps(row) for row in [
                dict(USER_UNIT='work.service', USER_INVOCATION_ID='old', UNIT_RESULT='oom-kill'),
                dict(USER_UNIT='work.service', USER_INVOCATION_ID='current',
                     MESSAGE='Started python -c "print(\'oom-kill\')".')]))
        return subprocess.CompletedProcess(command, 0, stdout, '')
    monkeypatch.setattr(systemd_guard.subprocess, 'run', run)
    assert systemd_guard.unit_status('work.service', invocation_id='current')['result'] == 'running'


@pytest.mark.parametrize('message_id', ['d989611b15e44c9dbf31e3c81256e4ed', 'fe6faa94e7774663a0da52717891d8ef'])
def test_systemd_oom_event_ids_override_success(monkeypatch, message_id):
    from saas_bench import systemd_guard
    def run(command, **kwargs):
        stdout = ('LoadState=loaded\nActiveState=inactive\nResult=success\nInvocationID=current\n'
            if command[0] == 'systemctl' else json.dumps(dict(USER_UNIT='work.service',
                USER_INVOCATION_ID='current', MESSAGE_ID=message_id)))
        return subprocess.CompletedProcess(command, 0, stdout, '')
    monkeypatch.setattr(systemd_guard.subprocess, 'run', run)
    assert systemd_guard.unit_status('work.service', invocation_id='current', exit_code=0)['result'] == 'oom-kill'


def test_watch_pins_invocation_and_exits_nonzero_on_oom(monkeypatch, capsys):
    from saas_bench import systemd_guard
    calls = []
    def status(unit, **kwargs):
        calls.append(kwargs)
        return dict(unit=unit, invocation_id='current', result='running' if len(calls) == 1 else 'oom-kill')
    monkeypatch.setattr(systemd_guard, 'unit_status', status)
    monkeypatch.setattr(systemd_guard.time, 'sleep', lambda _: None)
    assert systemd_guard.main(['--unit', 'work.service', '--watch']) == 2
    assert calls == [dict(invocation_id=None, exit_code=None), dict(invocation_id='current', exit_code=None)]
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['result'] == 'oom-kill'


def test_observer_reports_oom_for_one_group_without_stopping_peer(observer, monkeypatch):
    from saas_bench import systemd_guard
    units = dict(git=dict(unit='git.scope', invocation_id='git-id'),
                 pf=dict(unit='pf.scope', invocation_id='pf-id'))
    (observer.output.parent / 'controller-units.json').write_text(json.dumps(units))
    monkeypatch.setattr(systemd_guard, 'unit_status', lambda unit, **kwargs:
        dict(unit=unit, invocation_id=kwargs['invocation_id'],
             result='oom-kill' if unit == 'git.scope' else 'running', source='journal'))
    health = observer.poll()
    assert health['units']['git']['result'] == 'oom-kill'
    assert health['progress']['git']['reason'] == 'oom-kill'
    assert health['progress']['git']['status'] == 'failed'
    assert health['units']['pf']['result'] == 'running'
    assert health['status'] == 'healthy' and not observer.hold.exists()


@pytest.mark.parametrize('failed', [False, True])
def test_controller_main_returns_failure_and_passes_memory_settings(controller, monkeypatch, failed):
    calls = []
    def supervise(policy, **kwargs):
        calls.append(kwargs)
        (policy.parent / 'controller-exit.json').write_text(json.dumps(dict(failed=failed, hold=False)))
    monkeypatch.setattr(controller.module, 'supervise', supervise)
    monkeypatch.setenv('CEOBENCH_BASH_MEMORY_MAX_BYTES', '1073741824')
    assert controller.module.main(['--policy', str(controller.policy), '--live',
        '--memory-high', '128M', '--memory-max', '256M', '--memory-swap-max', '0',
        '--bash-memory-max', '64M']) == (2 if failed else 0)
    assert calls == [dict(resume=False, force_resume=False, memory_high=134217728, memory_max=268435456, memory_swap_max=0)]
    assert os.environ['CEOBENCH_BASH_MEMORY_MAX_BYTES'] == '67108864'


@pytest.mark.parametrize('high,maximum,swap', [(0, 1, 0), (2, 1, 0), (1, 2, -1)])
def test_invalid_cgroup_limits_are_rejected(high, maximum, swap):
    from saas_bench.systemd_guard import scope_command
    with pytest.raises(ValueError, match='MemoryHigh'):
        scope_command([], 'test.scope', memory_high=high, memory_max=maximum, memory_swap_max=swap)


@pytest.mark.parametrize('value', ['max', 'infinity'])
def test_unlimited_worker_memory_is_passed_to_controller(controller, monkeypatch, value):
    calls = []
    def supervise(policy, **kwargs):
        calls.append(kwargs)
        (policy.parent / 'controller-exit.json').write_text(json.dumps(dict(failed=False, hold=False)))
    monkeypatch.setattr(controller.module, 'supervise', supervise)
    assert controller.module.main(['--policy', str(controller.policy), '--live',
        '--memory-high', value, '--memory-max', value]) == 0
    assert calls == [dict(resume=False, force_resume=False,
        memory_high='max', memory_max='max', memory_swap_max=0)]
    from saas_bench.systemd_guard import scope_command
    command = scope_command([], 'test.scope', memory_high='max', memory_max='max')
    assert '--property=MemoryHigh=infinity' in command and '--property=MemoryMax=infinity' in command


def test_real_cgroup_oom_is_detected_and_peer_survives(tmp_path):
    from saas_bench.systemd_guard import scope_command, unit_status
    manager = subprocess.run(['systemctl', '--user', 'show', '--property=Version'],
        capture_output=True, timeout=5)
    if manager.returncode:
        pytest.skip('A running user systemd manager is required for the cgroup integration test')
    suffix = uuid.uuid4().hex
    units = [f'ceobench-oom-test-{suffix}-{role}.scope' for role in ('peer', 'oom')]
    children = []
    peer_ready = tmp_path / 'peer-ready'
    peer_code = f"import pathlib,time; pathlib.Path({str(peer_ready)!r}).touch(); time.sleep(20)"
    allocator = '''import resource
resource.setrlimit(resource.RLIMIT_AS, (268435456, 268435456))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
blocks = [bytearray(1024**2) for _ in range(128)]
'''
    try:
        for unit, code in zip(units, (peer_code, allocator)):
            command = scope_command(['/usr/bin/prlimit', '--as=268435456', '--cpu=20', '--',
                sys.executable, '-c', code], unit, memory_high=64 * 1024**2,
                memory_max=64 * 1024**2, memory_swap_max=0)
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            children.append(child)
            if unit == units[0]:
                deadline = time.monotonic() + 10
                while not peer_ready.exists() and child.poll() is None and time.monotonic() < deadline:
                    time.sleep(.01)
                assert peer_ready.exists(), child.communicate(timeout=1)
        stdout, stderr = children[1].communicate(timeout=15)
        assert children[1].returncode != 0, (stdout, stderr)
        deadline = time.monotonic() + 5
        while True:
            record = unit_status(units[1], exit_code=children[1].returncode)
            if record['result'] == 'oom-kill' or time.monotonic() >= deadline:
                break
            time.sleep(.05)
        assert record['result'] == 'oom-kill', record
        assert record['invocation_id'], record
        assert children[0].poll() is None
        peer = unit_status(units[0])
        assert peer['result'] == 'running', peer
        properties = subprocess.run(['systemctl', '--user', 'show', units[0],
            '--property=MemoryHigh,MemoryMax,MemorySwapMax,ControlGroup'], capture_output=True,
            text=True, check=True, timeout=5).stdout
        assert 'MemoryHigh=67108864' in properties and 'MemoryMax=67108864' in properties
        assert 'MemorySwapMax=0' in properties and units[0] in properties
    finally:
        subprocess.run(['systemctl', '--user', 'stop', *units], capture_output=True, timeout=10)
        for child in children:
            child.communicate(timeout=10)
        subprocess.run(['systemctl', '--user', 'reset-failed', *units], capture_output=True, timeout=10)
