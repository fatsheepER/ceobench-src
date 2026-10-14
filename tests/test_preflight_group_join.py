import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from saas_bench.run_state import write_json
from test_preflight_team_controller import controller
from test_preflight_team_observer import observer
from test_preflight_team_batch import frozen


def group_batch(observer, controller, monkeypatch, *, first_status='finished', second_status='finished', hold=False, unknown=False, partial=False, peer_partial=False, paired=False, persistent_failure=False, operator_pause=None):
    module = controller.module
    observer.manifest.update(category='formal', source_root=str(Path(__file__).resolve().parents[1]))
    observer.expected['source_root'] = observer.manifest['source_root']
    runs = [dict(run_id=f's{seed}-{group}', group=group, pair_id=f's{seed}-r1',
        output_dir=str(observer.host_logs.parent / 'runs' / f's{seed}-{group}'))
        for seed in (101, 102) for group in ('git', 'pf')]
    if paired:
        runs = runs[:2]
    observer.manifest['runs'] = runs
    if not paired:
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
    module.worker_arguments = []
    joined = controller.policy.parent / 'pf-peer-joined'

    def launch(command, **kwargs):
        if '--worker' not in command:
            return popen(command, **kwargs)
        run_id = command[command.index('--worker') + 1]
        entry = module.batch.selected(observer.manifest, run_id)
        if not paired:
            rows = [json.loads(line) for line in (controller.policy.parent / 'controller-events.jsonl').read_text().splitlines()]
            intent = rows[-1]
            assert intent['event'] == 'worker_launch_intent' and intent['run_id'] == run_id
            assert intent['launch_ordinal'] == len(launches) + 1
            assert intent['frozen_source_commit'] == observer.manifest['source_commit']
        if run_id == 's102-pf' and run_id not in launches:
            assert children[0][1].poll() is None, 'PF must join before the first Git worker exits'
        launches.append(run_id)
        module.worker_arguments.append(list(command))
        code = '''import json,sys,time
from pathlib import Path
run,out,status,hold,unknown,partial,peer_partial,model,joined,attempt,operator=sys.argv[1:]
p=Path(out)/('attempt-'+attempt)/'attempt.json';p.parent.mkdir(parents=True)
p.write_text(json.dumps(dict(status='running',execution_started=True)))
if partial=='yes' or peer_partial=='yes':
 log=p.parent/'runtime/private/ceo/model-usage.jsonl';log.parent.mkdir(parents=True)
 log.write_bytes((json.dumps(dict(event='request',call_id=run))+'\\n').encode()+b'{"event":"response"')
if run=='s101-git' and joined!='-':
 deadline=time.monotonic()+5
 while not Path(joined).exists() and time.monotonic()<deadline:time.sleep(.01)
else:time.sleep(.7 if run=='s101-git' else .03)
if peer_partial=='yes':
 with log.open('ab') as stream:stream.write((',"call_id":'+json.dumps(run)+',"usage":{"input_tokens":1,"output_tokens":1,"cached_tokens":0},"cost_usd":0,"response":{"model":'+json.dumps(model)+'}}\\n').encode())
if unknown=='yes':
 q=p.parent/'runtime/private/operations/unknown.json';q.parent.mkdir(parents=True)
 q.write_text(json.dumps(dict(kind='api:set_prices',day=0,status='running')))
if hold!='-':Path(hold).touch()
if status=='oom-kill':sys.exit(137)
if operator=='STOP':(p.parent/'STOP').touch()
p.write_text(json.dumps(dict(status=status,reason='offline_terminal_failure' if status=='failed' else status,
    execution_started=True,day=28,runtime_failure='run_evidence_fault' if operator=='STOP' else operator if operator!='-' else None)))
sys.exit(0 if status=='finished' else 2)
'''
        status = first_status if run_id == 's101-pf' and (persistent_failure or launches.count(run_id) == 1) else second_status if run_id == 's102-pf' and launches.count(run_id) == 1 else 'finished'
        child = popen([sys.executable, '-c', code, run_id, entry['output_dir'], status,
            str(observer.hold) if hold and run_id == 's101-pf' else '-',
            'yes' if unknown and run_id == 's101-pf' else 'no',
            'yes' if partial and run_id == 's101-pf' and '--resume' not in command else 'no',
            'yes' if peer_partial and run_id == 's101-git' else 'no', observer.module.round5.MODEL,
            str(joined) if not paired else '-',
            f'{len(module.batch.attempts(entry)) + 1:03}', operator_pause if run_id == 's101-pf' and operator_pause else '-'], **kwargs)
        children.append((run_id, child))
        if run_id == 's102-pf':
            joined.touch()
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
    assert launches[-2:] == ['s102-pf', 's102-git']
    assert all(child.poll() is not None for _, child in children)
    records = [json.loads(line) for line in (controller.policy.parent / 'controller-events.jsonl').read_text().splitlines()]
    joined = next(row for row in records if row['event'] == 'worker_launched' and row['run_id'] == 's102-pf')
    intent = next(row for row in records if row['event'] == 'worker_launch_intent' and row['run_id'] == 's102-pf')
    assert intent['launch_reason'] == 'previous_group_run_terminal'
    assert intent['predecessor_run_id'] == 's101-pf'
    assert joined['launch_ordinal'] == (3 if first_status == 'finished' else 4)
    before = list(launches)
    controller.module.supervise(controller.policy, resume=True)
    assert launches == before


def test_paused_lane_automatically_recovers_without_stopping_other_group(observer, controller, monkeypatch):
    launches, _ = group_batch(observer, controller, monkeypatch, first_status='paused')
    controller.module.supervise(controller.policy)
    assert launches.count('s101-pf') == 2 and launches[-2:] == ['s102-pf', 's102-git']
    assert not observer.hold.exists()


@pytest.mark.parametrize('reason', ['requested_pause', 'supervisor_cancelled', 'STOP'])
def test_deliberate_worker_stop_has_closed_receipt_and_successful_controller_exit(observer, controller, monkeypatch, reason):
    launches, children = group_batch(observer, controller, monkeypatch, first_status='paused',
        paired=True, operator_pause=reason)
    controller.module.supervise(controller.policy)
    assert launches == ['s101-git', 's101-pf']
    assert all(child.poll() is not None for _, child in children)
    terminal = controller.module.batch.read(controller.policy.parent / 'controller-terminal.json')
    stopped = terminal['s101-pf']
    assert stopped['disposition'] == 'paused' and stopped['operator_stop']
    assert stopped['children_reaped'] and stopped['unit']['scope_empty']
    result = controller.module.batch.read(controller.policy.parent / 'controller-exit.json')
    assert result['operator_stopped'] and result['children_reaped'] and not result['failed']
    rows = [json.loads(line) for line in (controller.policy.parent / 'controller-events.jsonl').read_text().splitlines()]
    assert not any(row['event'] == 'recovery_failure' for row in rows)


def test_historical_hold_does_not_stop_either_group(observer, controller, monkeypatch):
    launches, _ = group_batch(observer, controller, monkeypatch, hold=True)
    controller.module.supervise(controller.policy)
    assert 's102-pf' in launches and 's102-git' in launches


@pytest.mark.parametrize('first_status', ['failed', 'oom-kill'])
def test_unknown_business_write_is_discarded_during_recovery(observer, controller, monkeypatch, first_status):
    launches, _ = group_batch(observer, controller, monkeypatch, first_status=first_status, unknown=True)
    controller.module.supervise(controller.policy)
    assert 's102-pf' in launches and 's102-git' in launches


def test_third_failure_alarms_only_its_group_and_preserves_peer_progress(observer, controller, monkeypatch):
    launches, children = group_batch(observer, controller, monkeypatch,
                                    first_status='failed', persistent_failure=True)
    controller.module.supervise(controller.policy)
    assert launches.count('s101-pf') == 3
    assert 's102-pf' not in launches and 's102-git' in launches
    assert all(child.poll() is not None for _, child in children)
    alarm = json.loads((controller.policy.parent / 'alarm-s101-pf.json').read_text())
    assert alarm['reason'] == 'repeated_failure' and alarm['failures'] == 3
    assert not (controller.policy.parent / 'alarm-s101-git.json').exists()
    assert not observer.hold.exists()


def test_explicit_week_applies_only_to_first_launch_of_selected_run(observer, controller, monkeypatch):
    launches, children = group_batch(observer, controller, monkeypatch, first_status='failed')
    controller.module.supervise(controller.policy, run_id='s101-pf', week=4)
    assert launches == ['s101-pf', 's101-pf']
    assert all(child.poll() is not None for _, child in children)
    first, retry = controller.module.worker_arguments
    assert first[first.index('--week') + 1] == '4'
    assert '--week' not in retry and '--resume' in retry
    assert not (controller.policy.parent / 'alarm-s101-pf.json').exists()


def test_live_peer_partial_log_does_not_block_other_group_join(observer, controller, monkeypatch):
    launches, _ = group_batch(observer, controller, monkeypatch, peer_partial=True)
    controller.module.supervise(controller.policy)
    assert launches[-2:] == ['s102-pf', 's102-git']


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


def test_two_seed_group_freeze_cli_and_driver_check(tmp_path):
    from saas_bench import team_batch as batch
    from saas_bench.run_state import file_hash
    from test_preflight_team_batch import PRICES
    prices = tmp_path / 'prices.json'
    write_json(prices, PRICES)
    env = dict(os.environ, PYTHONPATH=str(batch.ROOT / 'src'), PYTHONHASHSEED='0',
        PYTHONDONTWRITEBYTECODE='1', PYTHON_DOTENV_DISABLED='1')
    command = [sys.executable, str(batch.ROOT / 'scripts/team_batch.py')]
    sources = []
    for seed in (1044, 1045):
        destination = tmp_path / f'd0-{seed}'
        subprocess.run([*command, 'seal-d0', '--seed', str(seed), '--output', str(destination),
            '--pricing', str(prices), '--price-source', PRICES['source']],
            env=env, capture_output=True, text=True, check=True, timeout=60)
        sources.extend(['--d0', f'{seed}={destination}'])
    directory = tmp_path / 'formal'
    result = subprocess.run([*command, 'freeze', *sources, '--output', str(directory),
        '--pricing', str(prices), '--price-source', PRICES['source'],
        '--excluded-debug-seed', '1043', '--seed-selection-rule', 'User selected 1044 and 1045',
        '--scheduling', 'group_independent'], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    path = directory / 'manifest.json'
    manifest = json.loads(result.stdout)
    validated = subprocess.run([*command, 'validate', '--manifest', str(path)],
        env=env, capture_output=True, text=True, check=True, timeout=60)
    assert json.loads(validated.stdout) == manifest['runs']
    assert manifest['seeds'] == [1044, 1045] and manifest['excluded_debug_seeds'] == [1043]
    assert {(row['seed'], row['repeat'], row['group']) for row in manifest['runs']} == {
        (seed, repeat, group) for seed in (1044, 1045) for repeat in (1, 2) for group in ('git', 'pf')}
    expected = {key: manifest[key] for key in ('batch_id', 'source_root', 'public_dir', 'source_commit')}
    expected.update(manifest=str(path), manifest_sha256=file_hash(path), host_logs=str(tmp_path / 'host-logs'),
        runs=[{key: row[key] for key in ('run_id', 'output_dir')} for row in manifest['runs']])
    policy = tmp_path / 'policy.json'
    write_json(policy, dict(identity=expected))
    checked = subprocess.run([sys.executable, str(batch.ROOT / 'scripts/team_driver.py'),
        '--policy', str(policy), '--check'], env=env, capture_output=True, text=True, check=True, timeout=60)
    receipt = json.loads(checked.stdout)
    assert receipt['scheduling'] == 'group_independent'
    assert receipt['model_requests'] == receipt['new_attempts'] == 0 and receipt['credential_read'] is False
    assert not (directory / 'runs').exists()
    with pytest.raises(ValueError, match='unused seeds'):
        batch.freeze_batch(tmp_path / 'excluded', {1043: tmp_path / 'absent', 1044: tmp_path / 'absent'}, PRICES,
            price_source=PRICES['source'], excluded_seeds=(1043,), selection_rule='User exclusion',
            scheduling='group_independent')
