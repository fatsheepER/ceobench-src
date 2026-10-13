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


def group_batch(observer, controller, monkeypatch, *, first_status='finished', second_status='finished', hold=False, unknown=False, partial=False, peer_partial=False, paired=False):
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
    joined = controller.policy.parent / 'pf-peer-joined'

    def launch(command, **kwargs):
        run_id = command[command.index('--worker') + 1]
        entry = module.batch.selected(observer.manifest, run_id)
        if not paired:
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
run,out,status,hold,unknown,partial,peer_partial,model,joined,attempt=sys.argv[1:]
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
p.write_text(json.dumps(dict(status=status,reason='offline_terminal_failure' if status=='failed' else status,
    execution_started=True,day=28)))
sys.exit(0 if status=='finished' else 2)
'''
        status = first_status if run_id == 's101-pf' else second_status if run_id == 's102-pf' else 'finished'
        child = popen([sys.executable, '-c', code, run_id, entry['output_dir'], status,
            str(observer.hold) if hold and run_id == 's101-pf' else '-',
            'yes' if unknown and run_id == 's101-pf' else 'no',
            'yes' if partial and run_id == 's101-pf' and '--resume' not in command else 'no',
            'yes' if peer_partial and run_id == 's101-git' else 'no', observer.module.round5.MODEL,
            str(joined) if first_status != 'paused' and not unknown and not hold and not paired else '-',
            '002' if '--resume' in command else '001'], **kwargs)
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


@pytest.mark.parametrize('unknown', [False, True])
def test_closed_oom_partial_log_settles_and_preserves_evidence(observer, controller, monkeypatch, unknown):
    launches, children = group_batch(observer, controller, monkeypatch,
        first_status='oom-kill', partial=True, unknown=unknown)
    timeout = threading.Timer(10, observer.hold.touch)
    timeout.start()
    try:
        controller.module.supervise(controller.policy)
    finally:
        timeout.cancel()
    assert all(child.poll() is not None for _, child in children)
    assert observer.hold.exists() == unknown
    if unknown:
        assert launches == ['s101-git', 's101-pf']
        terminal = controller.policy.parent / 'controller-terminal.json'
        assert not terminal.exists() or 's101-pf' not in json.loads(terminal.read_text())
        return
    assert launches == ['s101-git', 's101-pf', 's102-pf', 's102-git']
    path = Path(observer.manifest['runs'][1]['output_dir']) / 'attempt-001/runtime/private/ceo/model-usage.jsonl'
    original = path.read_bytes()
    state = json.loads((observer.output / 'state.json').read_text())
    proof = state['cursors'][str(path)]['closed_tail']
    assert Path(proof['tail_path']).read_bytes() == b'{"event":"response"'
    assert state['receipt_hashes']['request:s101-pf']
    assert json.loads((controller.policy.parent / 'controller-terminal.json').read_text())['s101-pf']['coverage'][str(path)]['terminal_fault'] == 'closed_partial_line'
    before = (observer.output / 'events.jsonl').read_bytes()
    (path.parents[3] / 'STOP').unlink()
    controller.module.supervise(controller.policy, resume=True)
    assert path.read_bytes() == original
    assert (observer.output / 'events.jsonl').read_bytes() == before
    assert len(launches) == 4
    assert (path.parents[3] / 'STOP').exists()
    assert json.loads((controller.policy.parent / 'controller-exit.json').read_text())['children_reaped']
    path.write_bytes(b'X' + original[1:])
    health = observer.poll(state=json.loads((observer.output / 'state.json').read_text()))
    assert health['status'] == 'hold' and health['checks'] == ['closed_log_changed:' + str(path)]
    assert Path(proof['tail_path']).read_bytes() == b'{"event":"response"'


def test_live_peer_partial_log_does_not_block_other_group_join(observer, controller, monkeypatch):
    launches, _ = group_batch(observer, controller, monkeypatch, peer_partial=True)
    controller.module.supervise(controller.policy)
    assert launches == ['s101-git', 's101-pf', 's102-pf', 's102-git']


def test_paired_oom_attempt_finalizes_on_disk_with_closed_tail(observer, controller, monkeypatch):
    launches, children = group_batch(observer, controller, monkeypatch,
        first_status='oom-kill', partial=True, paired=True)
    controller.module.supervise(controller.policy)
    assert launches == ['s101-git', 's101-pf']
    assert not observer.hold.exists() and all(child.poll() is not None for _, child in children)
    root = Path(observer.manifest['runs'][1]['output_dir']) / 'attempt-001'
    record = json.loads((root / 'attempt.json').read_text())
    assert record['status'] == 'failed' and record['reason'] == 'oom-kill'
    assert record['controller_finalization']['previous_status'] == 'running'
    assert record['controller_finalization']['unit']['scope_empty']
    path = root / 'runtime/private/ceo/model-usage.jsonl'
    state = json.loads((observer.output / 'state.json').read_text())
    assert state['cursors'][str(path)]['offset'] == path.stat().st_size
    assert state['cursors'][str(path)]['closed_tail']['fault_kind'] == 'permanent_capture_fault'
    assert json.loads((controller.policy.parent / 'controller-exit.json').read_text())['children_reaped']


@pytest.mark.parametrize('unknown', [False, True])
def test_paired_success_with_closed_tail_is_a_failed_capture(observer, controller, monkeypatch, unknown):
    launches, children = group_batch(observer, controller, monkeypatch, partial=True, paired=True, unknown=unknown)
    outcome = controller.module.supervise(controller.policy)
    assert launches == ['s101-git', 's101-pf']
    assert observer.hold.exists() == unknown and all(child.poll() is not None for _, child in children)
    root = Path(observer.manifest['runs'][1]['output_dir']) / 'attempt-001'
    record = json.loads((root / 'attempt.json').read_text())
    assert record['status'] == 'failed' and record['reason'] == 'run_evidence_fault'
    assert outcome[1]['status'] == 'failed'
    path = root / 'runtime/private/ceo/model-usage.jsonl'
    original = (json.dumps(dict(event='request', call_id='s101-pf')) + '\n').encode() + b'{"event":"response"'
    assert path.read_bytes() == original
    state = json.loads((observer.output / 'state.json').read_text())
    proof = state['cursors'][str(path)]['closed_tail']
    assert state['receipt_hashes']['request:s101-pf']
    assert proof['unit']['scope_empty'] and proof['unit']['exit_code'] == 0
    assert Path(proof['tail_path']).read_bytes() == b'{"event":"response"'
    assert record['controller_capture_faults'][str(path)]['terminal_fault'] == 'closed_partial_line'
    assert (root / 'STOP').exists()
    exit_record = json.loads((controller.policy.parent / 'controller-exit.json').read_text())
    assert exit_record['children_reaped'] and exit_record['failed'] and exit_record['hold'] == unknown
    assert json.loads((observer.output / 'health.json').read_text())['progress']['s101-pf']['status'] == 'failed'
    if not unknown:
        assert controller.module.supervise(controller.policy, resume=True)[0]['status'] == 'finished'
        resumed = root.parent / 'attempt-002'
        assert json.loads((resumed / 'attempt.json').read_text())['status'] == 'finished'
        assert path.read_bytes() == original and not (resumed / 'STOP').exists()
        assert not json.loads((controller.policy.parent / 'controller-exit.json').read_text())['failed']


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
    assert receipt['scheduling'] == manifest['scheduling']
    assert receipt['model_requests'] == receipt['new_attempts'] == 0 and receipt['credential_read'] is False
    assert not (directory / 'runs').exists()
    with pytest.raises(ValueError, match='unused seeds'):
        batch.freeze_batch(tmp_path / 'excluded', {1043: tmp_path / 'absent', 1044: tmp_path / 'absent'}, PRICES,
            price_source=PRICES['source'], excluded_seeds=(1043,), selection_rule='User exclusion',
            scheduling='group_independent')
