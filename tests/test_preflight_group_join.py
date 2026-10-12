import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from saas_bench.run_state import write_json
from test_preflight_team_controller import controller
from test_preflight_team_observer import observer
from test_preflight_team_batch import frozen


def group_batch(observer, controller, monkeypatch, *, first_status='finished', second_status='finished', hold=False, unknown=False):
    module = controller.module
    observer.manifest.update(category='formal', source_root=str(Path(__file__).resolve().parents[1]))
    observer.expected['source_root'] = observer.manifest['source_root']
    runs = [dict(run_id=f's{seed}-{group}', group=group, pair_id=f's{seed}-r1',
        output_dir=str(observer.host_logs.parent / 'runs' / f's{seed}-{group}'))
        for seed in (101, 102) for group in ('git', 'pf')]
    observer.manifest['runs'] = runs
    observer.manifest['scheduling'] = dict(mode='group_independent',
        queues={group: [row['run_id'] for row in runs if row['group'] == group] for group in ('git', 'pf')})
    observer.expected['runs'] = [{key: row[key] for key in ('run_id', 'output_dir')} for row in runs]
    write_json(controller.policy, dict(identity=observer.expected))
    monkeypatch.setattr(module, 'identity', observer.module.identity)
    monkeypatch.setattr(module, 'poll', lambda *args, **kwargs: observer.module.poll(*args,
        quota_fn=observer.quota, **kwargs))
    status = lambda unit, **kwargs: dict(unit=unit, exit_code=kwargs.get('exit_code'),
        invocation_id=kwargs.get('invocation_id') or unit, result='running' if kwargs.get('exit_code') is None
        else 'success' if kwargs['exit_code'] == 0 else 'oom-kill' if kwargs['exit_code'] == 137 else 'exit-code')
    monkeypatch.setattr(module, 'unit_status', status)
    from saas_bench import systemd_guard
    monkeypatch.setattr(systemd_guard, 'unit_status', status)
    popen, sleep = subprocess.Popen, time.sleep
    children, launches = [], []

    def launch(command, **kwargs):
        run_id = command[command.index('--worker') + 1]
        entry = module.batch.selected(observer.manifest, run_id)
        rows = [json.loads(line) for line in (controller.policy.parent / 'controller-events.jsonl').read_text().splitlines()]
        intent = rows[-1]
        assert intent['event'] == 'worker_launch_intent' and intent['run_id'] == run_id
        assert intent['launch_ordinal'] == len(launches) + 1
        assert intent['source_commit'] == observer.manifest['source_commit']
        if run_id == 's102-pf' and run_id not in launches:
            assert children[0][1].poll() is None, 'PF must join before the first Git worker exits'
        launches.append(run_id)
        code = '''import json,sys,time
from pathlib import Path
run,out,status,hold,unknown=sys.argv[1:]
p=Path(out)/'attempt-001/attempt.json';p.parent.mkdir(parents=True)
p.write_text(json.dumps(dict(status='running',execution_started=True)))
time.sleep(.7 if run=='s101-git' else .03)
if unknown=='yes':
 q=p.parent/'runtime/private/operations/unknown.json';q.parent.mkdir(parents=True)
 q.write_text(json.dumps(dict(kind='api:set_prices',day=0,status='running')))
if hold!='-':Path(hold).touch()
if status=='oom-kill':sys.exit(137)
p.write_text(json.dumps(dict(status=status,reason='offline_terminal_failure' if status=='failed' else status,
    execution_started=True,day=28)))
sys.exit(0 if status=='finished' else 2)
'''
        status = first_status if run_id == 's101-pf' else second_status if run_id == 's102-pf' else 'finished'
        child = popen([sys.executable, '-c', code, run_id, entry['output_dir'], status,
            str(observer.hold) if hold and run_id == 's101-pf' else '-',
            'yes' if unknown and run_id == 's101-pf' else 'no'], **kwargs)
        children.append((run_id, child))
        return child

    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    def signal_local_scope(saved, signum):
        units = module.batch.read(controller.policy.parent / 'controller-units.json')
        run_id = next(run_id for run_id, unit in units.items() if unit['unit'] == saved['unit'])
        child = next(child for run, child in reversed(children) if run == run_id)
        if child.poll() is None:
            os.killpg(child.pid, signum)
        return module.scope_state(saved, exit_code=child.poll())
    monkeypatch.setattr(module, 'signal_scope', signal_local_scope)
    monkeypatch.setattr(module.time, 'sleep', lambda _: sleep(.01))
    return launches, children


@pytest.mark.parametrize('first_status', ['finished', 'failed', 'oom-kill'])
def test_group_join_reaps_worker_and_launches_while_peer_runs(observer, controller, monkeypatch, first_status):
    launches, children = group_batch(observer, controller, monkeypatch, first_status=first_status)
    controller.module.supervise(controller.policy)
    assert launches == ['s101-git', 's101-pf', 's102-pf', 's102-git']
    assert all(child.poll() is not None for _, child in children)
    records = [json.loads(line) for line in (controller.policy.parent / 'controller-events.jsonl').read_text().splitlines()]
    joined = next(row for row in records if row['event'] == 'worker_launched' and row['run_id'] == 's102-pf')
    assert joined['launch_reason'] == 'previous_group_run_terminal'
    assert joined['predecessor_run_id'] == 's101-pf'
    assert joined['launch_ordinal'] == 3
    assert joined['group_ordinal'] == 2
    before = list(launches)
    controller.module.supervise(controller.policy, resume=True)
    assert launches == before


def test_paused_lane_waits_without_stopping_other_group(observer, controller, monkeypatch):
    launches, _ = group_batch(observer, controller, monkeypatch, first_status='paused')
    controller.module.supervise(controller.policy)
    assert launches == ['s101-git', 's101-pf', 's102-git']
    assert not observer.hold.exists()


def test_hold_prevents_next_seed_in_both_groups(observer, controller, monkeypatch):
    launches, _ = group_batch(observer, controller, monkeypatch, hold=True)
    controller.module.supervise(controller.policy)
    assert launches == ['s101-git', 's101-pf']
    assert observer.hold.exists()


@pytest.mark.parametrize('first_status', ['failed', 'oom-kill'])
def test_unknown_business_write_holds_before_next_group_run(observer, controller, monkeypatch, first_status):
    launches, _ = group_batch(observer, controller, monkeypatch, first_status=first_status, unknown=True)
    controller.module.supervise(controller.policy)
    assert launches == ['s101-git', 's101-pf']
    assert observer.hold.exists()


def test_safe_interruption_does_not_hide_a_later_unknown_write(observer, controller, monkeypatch):
    entry = observer.manifest['runs'][0]
    root = Path(entry['output_dir']) / 'attempt-001/runtime/private/operations'
    write_json(root / 'safe.json', dict(wrapper_version=1, status='interrupted'))
    write_json(root / 'unknown.json', dict(kind='api:set_prices', status='running'))
    from saas_bench import team_checkpoint
    monkeypatch.setattr(team_checkpoint, 'recovery_operation_safe', lambda *args: True)
    assert controller.module.run_disposition(entry, dict(status='paused'), dict(result='success')) == 'hold'


def test_group_terminal_attempt_drift_refuses_restart(observer, controller, monkeypatch):
    launches, _ = group_batch(observer, controller, monkeypatch)
    controller.module.supervise(controller.policy)
    entry = observer.manifest['runs'][0]
    write_json(Path(entry['output_dir']) / 'attempt-001/attempt.json', dict(status='finished', day=29))
    with pytest.raises(ValueError, match='terminal proof'):
        controller.module.supervise(controller.policy, resume=True)
    assert len(launches) == 4


def test_explicit_resume_at_later_slot_records_the_resume_reason(observer, controller, monkeypatch):
    launches, _ = group_batch(observer, controller, monkeypatch, second_status='paused')
    controller.module.supervise(controller.policy)
    assert len(launches) == 4
    controller.module.supervise(controller.policy, resume=True)
    assert launches[-1] == 's102-pf' and len(launches) == 5
    rows = [json.loads(line) for line in (controller.policy.parent / 'controller-events.jsonl').read_text().splitlines()]
    resumed = [row for row in rows if row['event'] == 'worker_launched'][-1]
    assert resumed['mode'] == 'resume' and resumed['launch_reason'] == 'explicit_resume'
    assert resumed['predecessor_run_id'] == 's101-pf'


def test_group_freeze_and_public_driver_check_do_not_register_attempts(frozen, tmp_path):
    from saas_bench import team_batch as batch
    from saas_bench.run_state import file_hash
    base, original, _, sources = frozen
    manifest = batch.freeze_batch(tmp_path / 'formal', {seed: sources[seed] for seed in original['seeds']},
        original['pricing'], price_source=original['price_source'], public_dir=base / 'public',
        selection_rule=original['seed_selection_rule'], excluded_seeds=original['excluded_debug_seeds'],
        scheduling='group_independent')
    path = tmp_path / 'formal/manifest.json'
    assert batch.validate_manifest(path)['scheduling'] == manifest['scheduling']
    expected = {key: manifest[key] for key in ('batch_id', 'source_root', 'public_dir', 'source_commit')}
    expected.update(manifest=str(path), manifest_sha256=file_hash(path), host_logs=str(tmp_path / 'host-logs'),
        runs=[{key: row[key] for key in ('run_id', 'output_dir')} for row in manifest['runs']])
    policy = tmp_path / 'policy.json'
    write_json(policy, dict(identity=expected))
    result = subprocess.run([sys.executable, str(batch.ROOT / 'scripts/team_driver.py'),
        '--policy', str(policy), '--check'], env=dict(os.environ, PYTHONPATH=str(batch.ROOT / 'src')),
        capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['model_requests'] == 0
    assert json.loads(result.stdout)['credential_read'] is False
    assert not (path.parent / 'runs').exists()
    manifest['scheduling']['queues']['pf'].reverse()
    write_json(path, manifest)
    write_json(path.parent / 'freeze.json', dict(manifest_sha256=file_hash(path)))
    with pytest.raises(ValueError, match='scheduling queues'):
        batch.validate_manifest(path)
