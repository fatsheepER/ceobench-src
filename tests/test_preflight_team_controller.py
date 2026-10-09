from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
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
from test_preflight_team_flow import completion, final
from test_preflight_team_observer import observer


@pytest.fixture
def controller(observer):
    path = Path(__file__).resolve().parents[1] / 'scripts/team_controller.py'
    spec = importlib.util.spec_from_file_location('team_controller', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = observer.output.parent / 'policy.json'
    write_json(policy, dict(identity=observer.expected))
    return SimpleNamespace(module=module, policy=policy, gate=module.RequestGate(policy))


@pytest.mark.parametrize('pending', ['quota', 'coverage', 'stale'])
def test_real_act_waits_then_resumes_same_request_without_hold(observer, controller, monkeypatch, pending):
    if pending == 'quota':
        def unavailable():
            raise httpx.ReadTimeout('temporarily unavailable')
        observer.module.poll(observer.manifest_path, observer.output, observer.expected,
                             observer.state, quota_fn=unavailable)
    elif pending == 'coverage':
        path = observer.log()
        path.write_text('{"event":"request","call_id":"partial"')
        assert observer.poll()['status'] == 'pending'
    else:
        health = observer.poll()
        health['time'] = (datetime.now(timezone.utc) - timedelta(seconds=331)).isoformat()
        write_json(observer.output / 'health.json', health)
    waiting, release = threading.Event(), threading.Event()

    def wait(seconds):
        waiting.set()
        assert release.wait(5), 'Pending request did not receive resumed health'

    monkeypatch.setattr(controller.module.time, 'sleep', wait)
    requests = []

    def respond(request):
        requests.append(request.extensions['timeout'])
        return completion(final('recovered advice'), stream=True)

    client = batch.checked_client('growth', 'offline', transport=httpx.MockTransport(respond),
        request_guard=controller.gate, response_guard=controller.gate.response, hold=observer.hold)
    agent = BashAgent(get_bash_agent_tool_descriptions(), client, model=batch.MODEL,
        system_prompt='Give advice.', workspace_path=observer.output.parent,
        identity=SimpleNamespace(role='growth'), reasoning_effort='high', allow_final_text=True)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(agent.act, 'dashboard', 0, False, {'day': 0})
            try:
                assert waiting.wait(5)
                assert requests == [] and not observer.hold.exists() and not result.done()
                if pending == 'coverage':
                    with path.open('a') as stream:
                        stream.write('}\n')
                assert observer.poll()['status'] == 'healthy'
            finally:
                release.set()
            assert result.result(timeout=5) == FinalText('recovered advice')
        assert requests == [dict(connect=60, read=180, write=60, pool=60)]
        assert not observer.hold.exists()
        assert agent.total_turns == 1
    finally:
        client.close()


@pytest.mark.parametrize('retry', [False, True])
@pytest.mark.parametrize('threaded', [False, True])
def test_pending_over_six_hundred_seconds_excludes_model_deadlines(observer, controller, monkeypatch, retry, threaded):
    health = observer.poll()
    pending = dict(health, status='pending')
    if not retry:
        write_json(observer.output / 'health.json', pending)
    clock, requests, waited = [0], [], []

    def wait(seconds):
        if seconds != .25:
            return
        waited.append(True)
        if not threaded:
            assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
        clock[0] += 701
        write_json(observer.output / 'health.json', health)

    def respond(request):
        requests.append(request)
        if retry and len(requests) == 1:
            write_json(observer.output / 'health.json', pending)
            return httpx.Response(500, json={'error': {'message': 'temporary provider error'}})
        if not threaded:
            assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
        return completion(final('advice after waiting'), stream=True)

    monkeypatch.setattr(controller.module.time, 'sleep', wait)
    monkeypatch.setattr(controller.module.time, 'monotonic', lambda: clock[0])
    with batch.checked_client('growth', 'offline', transport=httpx.MockTransport(respond),
            request_guard=controller.gate, hold=observer.hold) as client:
        agent = BashAgent(get_bash_agent_tool_descriptions(), client, model=batch.MODEL,
            system_prompt='Give advice.', workspace_path=observer.output.parent,
            identity=SimpleNamespace(role='growth'), reasoning_effort='high', allow_final_text=True)
        if threaded:
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(agent.act, 'dashboard', 0, False, {'day': 0}).result(timeout=5)
        else:
            result = agent.act('dashboard', 0, False, {'day': 0})
        assert result == FinalText('advice after waiting')
        assert len(requests) == (2 if retry else 1) and len(waited) == 1
        assert agent._llm_attempt == 1 and agent.usage_recorder.summary['calls'] == 1
        assert not observer.hold.exists()
    if not threaded:
        assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


@pytest.mark.parametrize('cause', ['parent', 'sibling'])
def test_pending_request_exits_when_parent_dies_or_sibling_cancels(observer, controller, monkeypatch, cause):
    health = observer.poll()
    write_json(observer.output / 'health.json', dict(health, status='pending'))
    waiting, release = threading.Event(), threading.Event()
    lifecycle = WorkerLifecycle()
    parent = [100]
    gate = controller.module.RequestGate(controller.policy, parent_pid=100 if cause == 'parent' else None)
    monkeypatch.setattr(controller.module.os, 'getppid', lambda: parent[0])
    waits = []

    def wait(seconds):
        waits.append(True)
        assert len(waits) == 1, 'Cancelled request entered the wait again'
        waiting.set()
        assert release.wait(5)

    monkeypatch.setattr(controller.module.time, 'sleep', wait)
    requests = []
    with batch.checked_client('growth', 'offline', transport=httpx.MockTransport(lambda req: requests.append(req)),
            request_guard=lambda req: gate(req, check=lifecycle.check), hold=observer.hold) as client:
        agent = BashAgent(get_bash_agent_tool_descriptions(), client, model=batch.MODEL,
            system_prompt='Give advice.', workspace_path=observer.output.parent,
            identity=SimpleNamespace(role='growth'), reasoning_effort='high', allow_final_text=True)
        agent.lifecycle = lifecycle
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(agent.act, 'dashboard', 0, False, {'day': 0})
            try:
                assert waiting.wait(5)
                if cause == 'parent':
                    parent[0] = 101
                else:
                    lifecycle.cancel('analyst_failure')
            finally:
                release.set()
            with pytest.raises(RunCancelled, match='supervisor_exited' if cause == 'parent' else 'analyst_failure'):
                result.result(timeout=5)
    assert requests == [] and not observer.hold.exists()


def test_run_evidence_stop_cancels_only_own_model_request(observer, controller):
    observer.poll()
    entry = observer.manifest['runs'][0]
    attempt = Path(entry['output_dir']) / 'attempt-001'
    attempt.mkdir(parents=True)
    (attempt / 'STOP').touch()
    stopped = controller.module.RequestGate(controller.policy, run_id=entry['run_id'])
    peer = controller.module.RequestGate(controller.policy, run_id=observer.manifest['runs'][1]['run_id'])
    with pytest.raises(RunCancelled, match='run_evidence_fault'):
        stopped(None)
    peer(None)
    assert not observer.hold.exists()


@pytest.mark.parametrize('stop', ['HOLD', 'STOP', 'QUOTA_EXHAUSTED', 'second_quota', 'identity'])
def test_global_conditions_cancel_before_model_request(observer, controller, stop):
    observer.poll()
    if stop == 'second_quota':
        def unavailable():
            raise httpx.ConnectError('unavailable')
        for _ in range(2):
            observer.module.poll(observer.manifest_path, observer.output, observer.expected,
                                 observer.state, quota_fn=unavailable)
    elif stop == 'identity':
        health = batch.read(observer.output / 'health.json')
        health['batch_id'] = 'another-batch'
        write_json(observer.output / 'health.json', health)
    elif stop == 'HOLD':
        observer.hold.touch()
    else:
        (controller.policy.parent / stop).touch()
    requests = []
    client = batch.checked_client('growth', 'offline',
        transport=httpx.MockTransport(lambda request: requests.append(request)),
        request_guard=controller.gate, hold=observer.hold)
    try:
        with pytest.raises(RunCancelled):
            client.chat.completions.create(model=batch.MODEL, messages=[])
        assert requests == [] and observer.hold.exists()
    finally:
        client.close()


def test_quota_response_sets_hold_and_frozen_request_drift_holds_immediately(observer, controller):
    observer.poll()
    requests = []

    def exhausted(request):
        requests.append(request)
        return httpx.Response(429, json={'error': {'message': 'weekly usage limit exceeded'}})

    kwargs = dict(model=batch.MODEL, reasoning_effort='high', temperature=1., max_tokens=16384,
        stream=True, stream_options={'include_usage': True}, extra_body={'thinking': {'type': 'enabled'}},
        messages=[dict(role='user', content='offline')])
    with batch.checked_client('growth', 'offline', transport=httpx.MockTransport(exhausted),
            request_guard=controller.gate, response_guard=controller.gate.response, hold=observer.hold) as client:
        with pytest.raises(RunCancelled):
            client.chat.completions.create(**kwargs)
    assert len(requests) == 1 and observer.hold.exists()
    assert (controller.policy.parent / 'QUOTA_EXHAUSTED').exists()
    observer.hold.unlink()
    (controller.policy.parent / 'QUOTA_EXHAUSTED').unlink()
    with batch.checked_client('growth', 'offline', transport=httpx.MockTransport(exhausted),
            request_guard=controller.gate, hold=observer.hold) as client:
        with pytest.raises(RunCancelled, match='frozen_identity_changed'):
            client.chat.completions.create(**dict(kwargs, timeout=60))
    assert len(requests) == 1 and observer.hold.exists()


@pytest.mark.parametrize('initial_pending', [False, True])
@pytest.mark.parametrize('missing_record', [False, True])
def test_controller_preserves_live_peer_when_other_worker_exits(observer, controller, monkeypatch,
                                                               initial_pending, missing_record):
    processes, signals, checks = [], [], []
    popen, killpg, sleep = subprocess.Popen, controller.module.os.killpg, controller.module.time.sleep
    source = str(Path(__file__).resolve().parents[1])
    observer.expected['source_root'] = observer.manifest['source_root'] = source
    write_json(controller.policy, dict(identity=observer.expected))
    for entry in observer.manifest['runs']:
        entry.update(pair_id='s42-r1', group=entry['run_id'])
    monkeypatch.setattr(controller.module, 'identity', observer.module.identity)

    def launch(command, **kwargs):
        role = command[command.index('--worker') + 1]
        entry = batch.selected(observer.manifest, role)
        record = Path(entry['output_dir']) / 'attempt-001/attempt.json'
        child_code = (
            'import json,sys,time; from pathlib import Path; '
            'p=Path(sys.argv[1]); '
            'time.sleep(float(sys.argv[2])); '
            'p.parent.mkdir(parents=True,exist_ok=True) if sys.argv[3]!="missing" else None; '
            'p.write_text(json.dumps(dict(status=sys.argv[3],day=28))) if sys.argv[3]!="missing" else None; '
            'sys.exit(int(sys.argv[4]))')
        status = ('missing' if missing_record else 'failed') if role == 'git' else 'finished'
        child = popen([sys.executable, '-c', child_code, str(record),
            '0' if role == 'git' else '.25', status,
            '2' if role == 'git' else '0'], **kwargs)
        processes.append((role, child))
        return child

    def signal_group(pid, sig):
        signals.append((pid, sig))
        return killpg(pid, sig)

    def poll(*args):
        checks.append(True)
        if initial_pending and len(checks) == 1:
            def unavailable():
                raise httpx.ConnectError('unavailable')
            return observer.module.poll(*args, quota_fn=unavailable)
        return observer.module.poll(*args, quota_fn=observer.quota)

    clock = [0]

    def tick(_):
        clock[0] += 300 if not processes else .01
        sleep(.01)

    monkeypatch.setattr(controller.module.subprocess, 'Popen', launch)
    monkeypatch.setattr(controller.module.os, 'killpg', signal_group)
    monkeypatch.setattr(controller.module.time, 'sleep', tick)
    monkeypatch.setattr(controller.module.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(controller.module, 'poll', poll)
    results = controller.module.supervise(controller.policy)
    assert [record['status'] if record else None for record in results] == [
        None if missing_record else 'failed', 'finished']
    assert not observer.hold.exists()
    assert len(processes) == 2 and all(child.poll() is not None for _, child in processes)
    assert all(pid == processes[0][1].pid and sig == signal.SIGTERM for pid, sig in signals)
    assert len(checks) == (3 if initial_pending else 2)
    assert batch.read(controller.policy.parent / 'controller-exit.json')['children_reaped'] is True
    with pytest.raises(ValueError, match='explicit --resume'):
        controller.module.supervise(controller.policy)
    assert len(processes) == 2


def test_controller_requires_explicit_live_flag(controller):
    with pytest.raises(SystemExit) as exc:
        controller.module.main(['--policy', str(controller.policy)])
    assert exc.value.code == 2


def test_pf_evidence_fault_stops_its_worker_while_git_finishes(observer, controller, monkeypatch):
    processes, signals = [], []
    popen, kill, sleep = subprocess.Popen, controller.module.os.kill, controller.module.time.sleep
    source = str(Path(__file__).resolve().parents[1])
    observer.expected['source_root'] = observer.manifest['source_root'] = source
    write_json(controller.policy, dict(identity=observer.expected))
    child_code = '''import json,os,signal,sys,time
from pathlib import Path
from saas_bench.run_state import write_json
from saas_bench.sql_evidence import SQLEvidenceStore,FORMAT
role,out=sys.argv[1:]
root=Path(out);stage=root/'.staging';stage.mkdir(parents=True)
write_json(stage/'attempt.json',dict(status='running',day=0))
stage.rename(root/'attempt-001')
record=root/'attempt-001/attempt.json'
if role=='pf':
    def stop(*_):
        write_json(record,dict(status='failed',day=0,reason='run_evidence_fault'))
        sys.exit(2)
    signal.signal(signal.SIGUSR1,stop)
    store=SQLEvidenceStore(record.parent/'runtime/private/evidence.sqlite',
        dict(run_id='pf',branch_id='offline',data_source_id='world',format=FORMAT))
    store.fail(RuntimeError('offline source capture failed'))
    Path(os.environ['CEOBENCH_LIFECYCLE_READY']).write_text('{}')
    time.sleep(5)
    sys.exit(3)
time.sleep(.5)
write_json(record,dict(status='finished',day=28))
'''

    def launch(command, **kwargs):
        role = command[command.index('--worker') + 1]
        entry = batch.selected(observer.manifest, role)
        child = popen([sys.executable, '-c', child_code, role, entry['output_dir']], **kwargs)
        processes.append((role, child))
        return child

    def signal_worker(pid, sig):
        signals.append((pid, sig))
        kill(pid, sig)

    clock = [0]

    def tick(_):
        health = batch.read(observer.output / 'health.json')
        clock[0] += .01 if health.get('run_faults') else 300
        sleep(.01)

    monkeypatch.setattr(controller.module.subprocess, 'Popen', launch)
    monkeypatch.setattr(controller.module.os, 'kill', signal_worker)
    monkeypatch.setattr(controller.module.time, 'sleep', tick)
    monkeypatch.setattr(controller.module.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(controller.module, 'poll',
        lambda *args: observer.module.poll(*args, quota_fn=observer.quota))
    results = controller.module.supervise(controller.policy)
    assert [row['status'] for row in results] == ['finished', 'failed']
    assert not observer.hold.exists()
    assert signals == [(processes[1][1].pid, signal.SIGUSR1)]
    health = batch.read(observer.output / 'health.json')
    assert health['status'] == 'healthy' and health['run_faults'][0]['run_id'] == 'pf'
    assert (Path(observer.manifest['runs'][1]['output_dir']) / 'attempt-001/STOP').exists()
    assert not (Path(observer.manifest['runs'][0]['output_dir']) / 'attempt-001/STOP').exists()
    assert all(child.poll() is not None for _, child in processes)
