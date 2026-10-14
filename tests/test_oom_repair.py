import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import httpx
import pytest

from saas_bench import team_batch as batch
from saas_bench.model_usage import ModelUsage
from saas_bench.run_lifecycle import RunCancelled
from saas_bench.run_state import write_json
from test_preflight_team_batch import frozen, managed_team
from test_preflight_team_controller import controller
from test_preflight_team_flow import completion, final
from test_preflight_team_observer import observer


def test_worker_refuses_invalid_scope_before_model_transport(observer, controller, monkeypatch, tmp_path):
    observer.poll()
    monkeypatch.setenv('OPENCODE_API_KEY', 'offline')
    monkeypatch.setenv('CEOBENCH_WORKER_SCOPE_POLICY', '{}')
    wires = []
    def respond(request):
        wires.append(request)
        return completion(final('unsafe request'), stream=True)
    def prepare(manifest, run_id, **options):
        client = batch.checked_client('ceo', 'offline', transport=httpx.MockTransport(respond),
            request_guard=options['request_guard'])
        usage = ModelUsage(tmp_path / 'usage.jsonl', 'ceo', {})
        usage.attach(client)
        try:
            params = dict(model=batch.MODEL, messages=[dict(role='user', content='offline')],
                reasoning_effort='high', extra_body={'thinking': {'type': 'enabled'}},
                temperature=1., max_tokens=16384, stream=True, stream_options={'include_usage': True})
            def invoke():
                with client.chat.completions.create(**params) as stream:
                    list(stream)
                return completion(final('unsafe request'), stream=False).json()
            usage.call('chat', params, invoke, day=0)
        finally:
            client.close()
        return dict(status='finished')
    monkeypatch.setattr(batch, 'prepare', prepare)
    failure = None
    try:
        controller.module.worker(controller.policy, observer.manifest['runs'][0]['run_id'],
            resume=False, parent_pid=None)
    except RunCancelled as exc:
        failure = str(exc)
    assert wires == [], 'Invalid worker scope reached the model transport'
    assert failure == 'worker_scope_invalid'
    assert observer.hold.exists()
    assert not (tmp_path / 'usage.jsonl').exists()


@pytest.mark.parametrize('partial_tail', [False, True])
def test_controller_oom_records_safe_checkpoint_after_real_scope_close(
        managed_team, observer, controller, monkeypatch, partial_tail):
    team, directory, record, manifest, entry = managed_team
    record.update(status='failed', reason='host_interrupted')
    write_json(directory / 'attempt.json', record)
    partner = dict(entry, run_id='finished-pf-partner', group='pf',
        output_dir=str(controller.policy.parent / 'finished-partner'))
    write_json(Path(partner['output_dir']) / 'attempt-001/attempt.json', dict(status='finished'))
    observer.manifest.update(source_root=str(batch.ROOT), runs=[entry, partner])
    observer.expected.update(source_root=str(batch.ROOT),
        runs=[{key: run[key] for key in ('run_id', 'output_dir')} for run in (entry, partner)])
    write_json(controller.policy, dict(identity=observer.expected))
    monkeypatch.setattr(controller.module, 'identity', lambda *args: observer.manifest)
    before_capture = []
    def poll(*args, **options):
        current = batch.read(directory / 'attempt.json')
        if current.get('reason') == 'oom-kill':
            before_capture.append(current.get('recovery_validation', {}).get('eligible'))
        return observer.module.poll(*args, **options, quota_fn=observer.quota)
    monkeypatch.setattr(controller.module, 'poll', poll)
    spec = importlib.util.spec_from_file_location('oom_actual_controller', batch.ROOT / 'scripts/team_controller.py')
    actual = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(actual)
    monkeypatch.setattr(controller.module, 'scope_state', actual.scope_state)
    monkeypatch.setattr(controller.module, 'MemorySampler', actual.MemorySampler)
    popen = subprocess.Popen
    children, units = [], []
    def launch(command, **options):
        if '--worker' not in command:
            return popen(command, **options)
        units.append(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--unit=')))
        code = '''import json,pathlib,resource,sys,time
resource.setrlimit(resource.RLIMIT_AS, (268435456, 268435456))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
p=pathlib.Path(sys.argv[1]);r=json.loads(p.read_text());r['status']='running';p.write_text(json.dumps(r))
if sys.argv[2]=='tail':
 log=p.parent/'runtime/private/ceo/model-usage.jsonl'
 log.write_bytes(b'{"event":"request","call_id":"closed-tail"')
time.sleep(.3)
blocks=[bytearray(1024**2) for _ in range(128)]
'''
        child = popen([*command[:command.index(sys.executable)], sys.executable,
            '-c', code, str(directory / 'attempt.json'), 'tail' if partial_tail else 'clean'], **options)
        children.append(child)
        return child
    monkeypatch.setattr(controller.module.subprocess, 'Popen', launch)
    try:
        results = controller.module.supervise(controller.policy, resume=True,
            memory_high=64 * 1024**2, memory_max=64 * 1024**2)
        assert children[0].returncode != 0
        exit_record = batch.read(controller.policy.parent / 'controller-exit.json')
        assert exit_record['children_reaped']
        assert exit_record['units'][entry['run_id']]['result'] == 'oom-kill'
        assert exit_record['units'][entry['run_id']]['scope_empty']
        assert True in before_capture
        assert results[0].get('recovery_validation', {}).get('eligible') is not partial_tail
        assert results[0]['recovery_validation']['current_operation_coverage'] is True
        assert results[0]['recovery_validation']['checkpoint'] == record['checkpoint']
        assert exit_record['memory_sampler']['closed']
        samples = Path(exit_record['memory_sampler']['samples_path'])
        assert json.loads(samples.read_text().splitlines()[0])['sequence'] == 1
        if partial_tail:
            assert results[0]['recovery_validation']['capture_faults']
            assert results[0]['controller_capture_faults']
    finally:
        subprocess.run(['systemctl', '--user', 'stop', *units], capture_output=True, timeout=10)
        for child in children:
            child.wait(timeout=10)
        subprocess.run(['systemctl', '--user', 'reset-failed', *units], capture_output=True, timeout=5)


@pytest.mark.parametrize('fault', ['workspace', 'unknown-write', 'completed-write', 'evidence', 'missing', 'corrupt'])
def test_oom_validation_refuses_unsafe_sources(managed_team, controller, fault):
    team, directory, record, _, entry = managed_team
    record['status'] = 'running'
    if fault == 'workspace':
        (team.root / 'roles/ceo/late.txt').write_text('after checkpoint')
    elif fault in ('unknown-write', 'completed-write'):
        write_json(team.root / 'private/operations/late.json',
            dict(kind='api:set_prices', status='running' if fault == 'unknown-write' else 'completed'))
    elif fault == 'evidence':
        write_json(team.root / 'private/evidence.fault.json', dict(error='permanent'))
    elif fault == 'missing':
        record['checkpoint'] = None
    else:
        (Path(record['checkpoint']) / 'team.json').write_text('{}')
    write_json(directory / 'attempt.json', record)
    unit = dict(unit='test.scope', invocation_id='test-invocation', control_group='/test/cgroup',
        result='oom-kill', scope_empty=True)
    result = controller.module.finalize_oom(entry, unit)
    assert result['status'] == 'failed' and result['reason'] == 'oom-kill'
    assert result['checkpoint'] == record['checkpoint']
    assert result['recovery_validation']['eligible'] is False
    assert result['recovery_validation']['error']


def test_oom_finalization_requires_closure_and_repeated_binding(managed_team, controller):
    _, directory, record, _, entry = managed_team
    record['status'] = 'running'
    write_json(directory / 'attempt.json', record)
    unit = dict(unit='test.scope', invocation_id='test-invocation', control_group='/test/cgroup',
        result='oom-kill', scope_empty=False)
    assert controller.module.finalize_oom(entry, unit) == record
    unit['scope_empty'] = True
    result = controller.module.finalize_oom(entry, unit)
    assert result['recovery_validation']['eligible'] is True
    original = (directory / 'attempt.json').read_bytes()
    assert controller.module.finalize_oom(entry, dict(unit, active_state='inactive')) == result
    assert (directory / 'attempt.json').read_bytes() == original
    with pytest.raises(RuntimeError, match='Conflicting OOM'):
        controller.module.finalize_oom(entry, dict(unit, invocation_id='other'))
    with pytest.raises(RuntimeError, match='Conflicting OOM'):
        controller.module.finalize_oom(entry, dict(unit, invocation_id='other', result='unknown', scope_empty=False))
    assert (directory / 'attempt.json').read_bytes() == original


@pytest.mark.parametrize('fault', ['slice', 'invocation', 'omit', 'limit', 'user-ancestor', 'system-ancestor'])
def test_scope_startup_rejects_incomplete_or_unsafe_properties(monkeypatch, fault):
    from saas_bench import systemd_guard
    policy = dict(unit='test.scope', memory_high=2 * 1024**3, memory_max=3 * 1024**3, memory_swap_max=0)
    observed = dict(Id='test.scope', InvocationID='scope-invocation', Slice='ceobench.slice',
        ControlGroup=f'/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/ceobench.slice/test.scope',
        OOMPolicy='kill', ManagedOOMPreference='omit', ManagedOOMMemoryPressure='auto', ManagedOOMSwap='auto',
        MemoryHigh=str(policy['memory_high']), MemoryMax=str(policy['memory_max']), MemorySwapMax='0')
    if fault in ('slice', 'invocation', 'omit', 'limit'):
        key, value = {'slice': ('Slice', 'app.slice'), 'invocation': ('InvocationID', ''),
            'omit': ('ManagedOOMPreference', 'none'), 'limit': ('MemoryMax', '1')}[fault]
        observed[key] = value
    calls = []
    def show(command, **options):
        calls.append(options['timeout'])
        rows = []
        for name in command[command.index('--') + 1:]:
            row = dict(observed) if name == 'test.scope' else dict(Id=name,
                ManagedOOMMemoryPressure='auto', ManagedOOMSwap='auto')
            if name == '-.slice' and ((fault == 'user-ancestor' and '--user' in command) or
                                      (fault == 'system-ancestor' and '--user' not in command)):
                row['ManagedOOMMemoryPressure'] = 'kill'
            rows.append('\n'.join(f'{key}={value}' for key, value in row.items()))
        return SimpleNamespace(stdout='\n\n'.join(rows))
    monkeypatch.setattr(systemd_guard.subprocess, 'run', show)
    monkeypatch.setattr(systemd_guard, 'check_worker_cgroup', lambda proof: None)
    with pytest.raises(RuntimeError, match='isolation policy|monitored ancestor'):
        systemd_guard.verify_worker_scope(policy)
    assert calls and all(0 < value <= 5 for value in calls)


@pytest.mark.parametrize('omit', [False, True])
def test_real_worker_scope_proof_precedes_model_usage(observer, controller, tmp_path, omit):
    from saas_bench.systemd_guard import scope_command
    observer.expected['source_root'] = observer.manifest['source_root'] = str(batch.ROOT)
    write_json(controller.policy, dict(identity=observer.expected))
    observer.poll()
    unit = 'ceobench-134-model-proof-' + uuid.uuid4().hex + '.scope'
    scope_policy = dict(unit=unit, memory_high=2 * 1024**3, memory_max=3 * 1024**3, memory_swap_max=0)
    manifest_file = tmp_path / 'offline-manifest.json'
    write_json(manifest_file, observer.manifest)
    usage_path = tmp_path / 'worker-model.jsonl'
    code = '''import json,os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]+'/scripts');sys.path.insert(0,sys.argv[1]+'/tests')
import team_controller,httpx
from saas_bench import team_batch as batch
from saas_bench.model_usage import ModelUsage
from saas_bench.run_lifecycle import RunCancelled
from saas_bench.agents.bash_agent import run_test
from test_preflight_team_flow import completion,final
manifest=json.loads(Path(sys.argv[3]).read_text());wires=[]
team_controller.identity=lambda *args:manifest
def forbidden(*args):raise AssertionError('Scope proof must not read .env')
run_test.load_env_file=forbidden
def respond(request):wires.append(request);return completion(final('offline'),stream=True)
def prepare(path,run_id,**options):
 client=batch.checked_client('ceo','offline',transport=httpx.MockTransport(respond),request_guard=options['request_guard'])
 usage=ModelUsage(sys.argv[4],'ceo',{});usage.attach(client)
 params=dict(model=batch.MODEL,messages=[dict(role='user',content='offline')],reasoning_effort='high',extra_body={'thinking':{'type':'enabled'}},temperature=1.,max_tokens=16384,stream=True,stream_options={'include_usage':True})
 def invoke():
  with client.chat.completions.create(**params) as stream:list(stream)
  return completion(final('offline'),stream=False).json()
 try:usage.call('chat',params,invoke,day=0)
 finally:client.close()
 return dict(status='finished',run_id=run_id)
batch.prepare=prepare
try:team_controller.worker(sys.argv[2],manifest['runs'][0]['run_id'],resume=False,parent_pid=None)
except RunCancelled as exc:print(json.dumps(dict(failure=str(exc),wires=len(wires))))
else:print(json.dumps(dict(wires=len(wires))))
'''
    env = dict(os.environ, OPENCODE_API_KEY='offline', CEOBENCH_WORKER_SCOPE_POLICY=json.dumps(scope_policy))
    command = scope_command([sys.executable, '-c', code, str(batch.ROOT), str(controller.policy),
        str(manifest_file), str(usage_path)], unit)
    if not omit:
        command.remove('--property=ManagedOOMPreference=omit')
    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
        assert result.returncode == 0, result.stderr
        output = json.loads(result.stdout.splitlines()[-1])
        assert output['wires'] == int(omit)
        assert usage_path.exists() is omit
        proof = batch.read(observer.host_logs / 'worker-scope-git-run.json')
        assert proof['verified'] is omit
        if omit:
            assert proof['slice'] == 'ceobench.slice' and '/app.slice/' not in proof['control_group']
            assert proof['memory_high'] == 2 * 1024**3 and proof['memory_max'] == 3 * 1024**3
            assert proof['memory_swap_max'] == 0
        else:
            assert output['failure'] == 'worker_scope_invalid' and observer.hold.exists()
    finally:
        subprocess.run(['systemctl', '--user', 'stop', unit], capture_output=True, timeout=5)
        subprocess.run(['systemctl', '--user', 'reset-failed', unit], capture_output=True, timeout=5)


@pytest.mark.parametrize('fault', ['dead', 'stale', 'forced'])
def test_controller_requires_healthy_sampler_at_finalization(observer, controller, monkeypatch, fault):
    for entry in observer.manifest['runs']:
        write_json(Path(entry['output_dir']) / 'attempt-001/attempt.json', dict(status='finished'))
    polls = []
    def poll(*args):
        polls.append(None)
        return dict(status='healthy', quota_error=None, coverage={}, units={})
    class Sampler:
        def __init__(self, path):
            pass
        def start(self):
            return dict(sequence=1)
        def check(self):
            if len(polls) == 2 and fault != 'forced':
                raise RuntimeError('Memory sampler ' + fault)
        def stop(self):
            return dict(closed=True, exit_code=1 if fault == 'dead' else 0,
                forced=fault == 'forced', unexpected_exit=fault == 'dead')
    monkeypatch.setattr(controller.module, 'poll', poll)
    monkeypatch.setattr(controller.module, 'MemorySampler', Sampler)
    if fault == 'forced':
        controller.module.supervise(controller.policy, resume=True)
    else:
        with pytest.raises(RuntimeError, match='Memory sampler'):
            controller.module.supervise(controller.policy, resume=True)
    assert observer.hold.exists()
    receipt = batch.read(controller.policy.parent / 'controller-exit.json')
    assert receipt['failed'] and receipt['hold'] and receipt['memory_sampler']['closed']


def test_controller_detects_real_sampler_exit_during_final_poll(observer, controller, monkeypatch):
    from saas_bench.host_memory import MemorySampler
    for entry in observer.manifest['runs']:
        write_json(Path(entry['output_dir']) / 'attempt-001/attempt.json', dict(status='finished'))
    samplers, polls = [], []
    def sampler(path):
        samplers.append(MemorySampler(path))
        return samplers[-1]
    def poll(*args):
        polls.append(None)
        if len(polls) == 2:
            samplers[0].child.kill()
            samplers[0].child.wait(timeout=5)
        return dict(status='healthy', quota_error=None, coverage={}, units={})
    monkeypatch.setattr(controller.module, 'poll', poll)
    monkeypatch.setattr(controller.module, 'MemorySampler', sampler)
    with pytest.raises(RuntimeError, match='Memory sampler exited'):
        controller.module.supervise(controller.policy, resume=True)
    receipt = batch.read(controller.policy.parent / 'controller-exit.json')
    assert receipt['failed'] and receipt['hold']
    assert receipt['memory_sampler']['closed'] and receipt['memory_sampler']['unexpected_exit']
    assert receipt['memory_sampler']['exit_code'] != 0


def test_sampler_failure_during_pending_poll_prevents_worker_launch(observer, controller, monkeypatch):
    clock, polls = [1000.], []
    def poll(*args):
        polls.append(None)
        return dict(status='pending' if len(polls) == 1 else 'healthy',
            quota_error='retry' if len(polls) == 1 else None, quota_retry_after=1)
    class Sampler:
        def __init__(self, path):
            pass
        def start(self):
            return dict(sequence=1)
        def check(self):
            if len(polls) > 1:
                raise RuntimeError('Memory sampler exited during poll')
        def stop(self):
            return dict(closed=True, exit_code=1, unexpected_exit=True)
    monkeypatch.setattr(controller.module, 'poll', poll)
    monkeypatch.setattr(controller.module, 'MemorySampler', Sampler)
    monkeypatch.setattr(controller.module.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(controller.module.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(controller.module.subprocess, 'Popen', lambda *a, **k: pytest.fail('Sampler failure launched worker'))
    with pytest.raises(RuntimeError, match='Memory sampler exited during poll'):
        controller.module.supervise(controller.policy)
    assert observer.hold.exists()
    assert not (controller.policy.parent / 'controller-units.json').exists()
    assert batch.read(controller.policy.parent / 'controller-exit.json')['failed']
