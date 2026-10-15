from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace

import httpx
import pytest

from saas_bench import team_batch as batch
from saas_bench.agents.bash_agent.agent import BashAgent, FinalText
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from saas_bench.run_lifecycle import RunCancelled
from saas_bench.run_lifecycle import WorkerLifecycle
from saas_bench.run_state import write_json
from saas_bench.model_usage import ModelUsage
from test_preflight_team_flow import completion, final
from test_preflight_team_observer import observer


@pytest.fixture
def controller(observer, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/team_controller.py'
    spec = importlib.util.spec_from_file_location('team_controller', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Supervisor tests below replace systemd-run with local Popen workers.
    def scope(saved, *, exit_code=None):
        record = module.unit_status(saved['unit'], invocation_id=saved.get('invocation_id'), exit_code=exit_code)
        record.update(invocation_id=record.get('invocation_id') or saved['unit'],
                      control_group='/test/' + saved['unit'], scope_empty=exit_code is not None)
        return record
    monkeypatch.setattr(module, 'scope_state', scope)
    class Sampler:
        def __init__(self, path):
            self.path = str(path)
        def start(self):
            return dict(sequence=1)
        def check(self):
            return self.start()
        def stop(self):
            return dict(closed=True, samples_path=self.path)
    monkeypatch.setattr(module, 'MemorySampler', Sampler)
    policy = observer.output.parent / 'policy.json'
    write_json(policy, dict(identity=observer.expected))
    return SimpleNamespace(module=module, policy=policy, gate=module.RequestGate(policy))


def test_historical_stop_warns_and_current_stop_cancels_only_own_request(observer, controller):
    observer.poll()
    entry = observer.manifest['runs'][0]
    attempt = Path(entry['output_dir']) / 'attempt-001'
    attempt.mkdir(parents=True)
    (attempt / 'STOP').touch()
    stopped = controller.module.RequestGate(controller.policy, run_id=entry['run_id'])
    peer = controller.module.RequestGate(controller.policy, run_id=observer.manifest['runs'][1]['run_id'])
    stopped(None)
    peer(None)
    current = attempt.parent / 'attempt-002'
    write_json(current / 'attempt.json', dict(attempt=2))
    (current / 'STOP').touch()
    with pytest.raises(RunCancelled, match='run_evidence_fault'):
        stopped(None)
    peer(None)
    assert not observer.hold.exists()


def test_explicit_stop_cancels_before_model_request(observer, controller):
    (controller.policy.parent / 'STOP').touch()
    requests = []
    with batch.checked_client('growth', 'offline',
            transport=httpx.MockTransport(lambda request: requests.append(request)),
            request_guard=controller.gate, hold=observer.hold) as client:
        with pytest.raises(RunCancelled):
            client.chat.completions.create(model=batch.MODEL, messages=[])
    assert requests == [] and not observer.hold.exists()


def test_controller_requires_explicit_live_flag(controller):
    with pytest.raises(SystemExit) as exc:
        controller.module.main(['--policy', str(controller.policy)])
    assert exc.value.code == 2


@pytest.mark.parametrize('week', [None, 4])
def test_resume_skips_finished_group_unless_week_selected(observer, controller, monkeypatch, caplog, week):
    from test_preflight_group_join import group_batch
    launches, children = group_batch(observer, controller, monkeypatch, paired=True)
    entry = observer.manifest['runs'][0]
    previous = Path(entry['output_dir']) / 'attempt-001/attempt.json'
    write_json(previous, dict(status='finished', attempt=1, day=28))
    original = previous.read_bytes()
    controller.module.supervise(controller.policy, resume=True, week=week)
    assert launches == (['s101-pf'] if week is None else ['s101-git', 's101-pf'])
    assert all(child.poll() is not None for _, child in children)
    assert previous.read_bytes() == original
    if week is None:
        assert 'Skipping finished group s101-git' in caplog.text
    receipt = batch.read(controller.policy.parent / 'controller-exit.json')
    assert not receipt['failed'] and not receipt['hold']


@pytest.mark.parametrize('failure', ['popen', 'worker'])
def test_startup_failure_preserves_finished_attempt_and_runs_peer(observer, controller, monkeypatch, failure):
    from test_preflight_group_join import group_batch
    popen = subprocess.Popen
    launches, children = group_batch(observer, controller, monkeypatch, paired=True)
    entry = observer.manifest['runs'][0]
    previous = Path(entry['output_dir']) / 'attempt-001/attempt.json'
    write_json(previous, dict(status='finished', attempt=1, day=28))
    original = previous.read_bytes()
    launch = controller.module.subprocess.Popen
    failed_children = []

    def fail_start(command, **kwargs):
        if '--worker' in command and command[command.index('--worker') + 1] == entry['run_id']:
            if failure == 'popen':
                raise OSError('injected worker startup failure')
            child = popen([sys.executable, '-c', 'raise SystemExit(2)'], **kwargs)
            failed_children.append(child)
            return child
        return launch(command, **kwargs)

    monkeypatch.setattr(controller.module.subprocess, 'Popen', fail_start)
    controller.module.supervise(controller.policy, resume=True, week=4)
    assert previous.read_bytes() == original
    assert not (previous.parent / 'HOLD').exists()
    assert launches == ['s101-pf']
    assert all(child.poll() is not None for _, child in children)
    assert all(child.poll() is not None for child in failed_children)
    notice = batch.read(previous.parent.parent / 'HOLD')
    assert notice['manual_resume_required'] and notice['run_id'] == entry['run_id']
    terminal = batch.read(controller.policy.parent / 'controller-terminal.json')
    assert terminal[entry['run_id']]['disposition'] == 'hold'
    assert terminal['s101-pf']['status'] == 'finished'


@pytest.mark.parametrize('source', ['attempt', 'units', 'events'])
def test_corrupt_management_history_warns_preserves_original_and_runs_peer(
        observer, controller, monkeypatch, caplog, source):
    from test_preflight_group_join import group_batch
    launches, children = group_batch(observer, controller, monkeypatch, paired=True)
    entry = observer.manifest['runs'][0]
    corrupt = (Path(entry['output_dir']) / 'attempt-001/attempt.json' if source == 'attempt' else
        controller.policy.parent / ('controller-units.json' if source == 'units' else 'controller-events.jsonl'))
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    original = b'{"unfinished":'
    corrupt.write_bytes(original)
    controller.module.supervise(controller.policy, resume=True)
    assert launches == ['s101-git', 's101-pf']
    assert all(child.poll() is not None for _, child in children)
    assert corrupt.read_bytes() == original
    assert str(corrupt) in caplog.text and 'unavailable' in caplog.text
    receipt = batch.read(controller.policy.parent / 'controller-exit.json')
    assert receipt['children_reaped'] and not receipt['hold'] and not receipt['failed']
    assert set(batch.read(receipt['units_path'])) == set(launches)
    events = [json.loads(line) for line in Path(receipt['events_path']).read_text().splitlines()]
    assert {row['run_id'] for row in events if row['event'] == 'worker_launched'} == set(launches)


def test_worker_entry_uses_own_tmpdir_before_prepare(observer, controller, monkeypatch):
    entry = observer.manifest['runs'][0]
    monkeypatch.setenv('CEOBENCH_WORKER_LAUNCH_ID', 'offline-launch')
    temporary = observer.host_logs / 'tmp/offline-launch'
    monkeypatch.setattr(os, 'environ', dict(os.environ, OPENCODE_API_KEY='offline'))
    monkeypatch.setattr(tempfile, 'tempdir', tempfile.tempdir)
    calls = []

    def prepare(manifest, run_id, **kwargs):
        calls.append((manifest, run_id, kwargs))
        assert os.environ['TMPDIR'] == tempfile.gettempdir() == str(temporary)
        with tempfile.TemporaryFile() as probe:
            probe.write(b'worker-owned')
        return dict(status='finished', run_id=run_id)

    def forbidden_env_read(*args):
        raise AssertionError('Offline worker must not read credentials')

    monkeypatch.setattr(batch, 'prepare', prepare)
    monkeypatch.setenv('CEOBENCH_WORKER_SCOPE_POLICY', '{}')
    monkeypatch.setattr(controller.module, 'verify_worker_scope', lambda policy: dict(unit='offline.scope'))
    monkeypatch.setattr('saas_bench.agents.bash_agent.run_test.load_env_file', forbidden_env_read)
    assert controller.module.worker(controller.policy, entry['run_id'], resume=False,
                                    parent_pid=None) == 0
    assert temporary.is_dir() and list(temporary.iterdir()) == []
    assert len(calls) == 1 and calls[0][2]['live'] is True


@pytest.mark.parametrize('bytes_available,inodes', [(1, 999999), (999999999, 1)])
def test_controller_storage_precheck_only_warns(observer, controller, monkeypatch, caplog,
                                              bytes_available, inodes):
    from test_preflight_group_join import group_batch
    monkeypatch.setattr(os, 'statvfs', lambda path: SimpleNamespace(f_bavail=bytes_available,
        f_frsize=1, f_favail=inodes, f_files=1000000))
    launches, children = group_batch(observer, controller, monkeypatch)
    controller.module.supervise(controller.policy)
    assert len(launches) == 4 and all(child.poll() is not None for _, child in children)
    assert 'Insufficient temporary storage' in caplog.text
    for entry in observer.manifest['runs']:
        record = batch.read(batch.attempts(entry)[0] / 'attempt.json')
        assert any('Insufficient temporary storage' in row['message'] for row in record['validation_warnings'])
