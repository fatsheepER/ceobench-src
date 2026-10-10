import os
import select
import signal
import subprocess
import sys

import pytest

from saas_bench import team_batch as batch
from saas_bench.run_lifecycle import WorkerLifecycle
from saas_bench.run_state import write_json
from test_preflight_team_batch import frozen, managed_team


def test_nested_lifecycle_protects_finalization(tmp_path, monkeypatch):
    ready = tmp_path / 'ready.json'
    monkeypatch.setenv('CEOBENCH_LIFECYCLE_READY', str(ready))
    before = signal.getsignal(signal.SIGUSR1)
    try:
        with WorkerLifecycle() as life:
            with life:
                pass
            assert callable(signal.getsignal(signal.SIGUSR1)), 'Inner run removed finalization protection'
            assert ready.exists()
        assert not ready.exists(), 'Exited worker still advertises cancellation readiness'
    finally:
        signal.signal(signal.SIGUSR1, before)


def test_delayed_and_duplicate_sigusr1_is_harmless_after_scope(tmp_path):
    code = '''
import os, sys
from saas_bench.run_lifecycle import WorkerLifecycle
with WorkerLifecycle() as life:
 print('active', flush=True)
 sys.stdin.readline()
 print(life.reason, flush=True)
 sys.stdin.readline()
print('finalizing', flush=True)
sys.stdin.readline()
print('persisted', flush=True)
'''
    env = dict(os.environ, CEOBENCH_LIFECYCLE_READY=str(tmp_path / 'ready.json'))
    child = subprocess.Popen([sys.executable, '-c', code], env=env, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    def receive():
        assert select.select([child.stdout], [], [], 5)[0], 'worker synchronization timeout'
        return child.stdout.readline().strip()
    try:
        assert receive() == 'active'
        os.kill(child.pid, signal.SIGUSR1)
        child.stdin.write('\n'); child.stdin.flush()
        assert receive() == 'supervisor_cancelled'
        child.stdin.write('\n'); child.stdin.flush()
        assert receive() == 'finalizing'
        for _ in range(3):
            os.kill(child.pid, signal.SIGUSR1)
        out, err = child.communicate('\n', timeout=5)
        assert child.returncode == 0, f'Finalization killed by delayed SIGUSR1: exit={child.returncode}, stderr={err}'
        assert out.strip() == 'persisted'
        assert not (tmp_path / 'ready.json').exists()
    finally:
        if child.poll() is None:
            child.kill(); child.communicate(timeout=5)


def test_execute_protects_result_and_attempt_persistence(managed_team, tmp_path, monkeypatch):
    import saas_bench.team_results as results
    team, directory, record, manifest, entry = managed_team
    monkeypatch.setattr(team, 'run', lambda **kw: type('Outcome', (), dict(day=0, reason='paused'))())
    team.failure = 'hold'
    summarize = results.summarize_run
    def protected(*args, **kw):
        assert callable(signal.getsignal(signal.SIGUSR1)), 'Result persistence has default SIGUSR1 handler'
        os.kill(os.getpid(), signal.SIGUSR1)
        return summarize(*args, **kw)
    monkeypatch.setattr(results, 'summarize_run', protected)
    result = batch.execute(team, manifest, entry, directory, record, tmp_path / 'HOLD')
    assert result['status'] == 'paused'
    assert (directory / 'result.json').exists()
    assert batch.read(directory / 'attempt.json')['status'] == 'paused'


@pytest.mark.parametrize('operation', ['covered', 'new_completed', 'unknown'])
def test_analysis_hold_preserves_error_and_audits_safe_boundary(managed_team, tmp_path, monkeypatch, operation):
    team, directory, record, manifest, entry = managed_team
    checkpoint = record['checkpoint']
    team._stage = 'analysis'
    team.failure = 'hold'
    monkeypatch.setattr(team, 'run', lambda **kw: type('Outcome', (), dict(day=0, reason='paused'))())
    if operation != 'covered':
        write_json(team.root / 'private/operations/post-snapshot.json',
            dict(status='completed' if operation == 'new_completed' else 'started'))
    result = batch.execute(team, manifest, entry, directory, record, tmp_path / 'HOLD')
    assert 'outside a recoverable handoff' in result['checkpoint_error']
    assert result['checkpoint'] == checkpoint
    if operation == 'covered':
        assert result.get('recovery_validation', {}).get('eligible') is True, 'Safe boundary coverage was not audited'
        assert result['recovery_validation']['checkpoint'] == checkpoint
    else:
        assert result.get('recovery_validation', {}).get('eligible') is False
        assert result['recovery_validation']['error']
