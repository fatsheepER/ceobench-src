from collections import Counter
import fcntl
import errno
import json
import os
from pathlib import Path
import tempfile

import pytest

from saas_bench import team_batch as batch
from saas_bench.run_state import tree_hash, write_json
from test_checkpoint_driver import Driver, _jsonl
from test_preflight_team_batch import PRICES
from test_round2_batch import assert_closed, cli


PROFILE = 'r5-s41-44-d56'


@pytest.fixture(scope='module')
def fifo_worlds(tmp_path_factory):
    root = Path(os.environ.get('CEOBENCH_FIFO_ARTIFACTS', tmp_path_factory.mktemp('fifo-worlds')))
    root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='sealed-', dir=root))
    sources = {seed: root / f'd0-{seed}' for seed in (41, 42, 43, 44)}
    for seed, path in sources.items():
        batch.seal_d0(seed, path, PRICES, price_source=PRICES['source'])
    return sources


@pytest.fixture
def fifo_disk(tmp_path):
    root = Path(os.environ.get('CEOBENCH_FIFO_ARTIFACTS', tmp_path))
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='fifo-', dir=root))


def freeze(worlds, root):
    manifest = batch.freeze_batch(root / 'batch', worlds, PRICES, price_source=PRICES['source'], profile=PROFILE)
    policy = Path(batch.create_policy(root / 'batch/manifest.json', root / 'control')['policy'])
    return manifest, policy


def test_eight_run_offline_launcher(fifo_worlds, fifo_disk):
    root = fifo_disk
    source, prices = root / 'd0-sources.json', root / 'prices.json'
    write_json(source, dict(d0={str(s): dict(path=str(p)) for s, p in fifo_worlds.items()}))
    write_json(prices, PRICES)
    original = {s: tree_hash(p, exact=True) for s, p in fifo_worlds.items()}
    prepared = cli('round5_main.py', '--prepare', '--profile', PROFILE, '--output', root / 'launch',
        '--d0-source-manifest', source, '--pricing', prices)
    checked = cli('round5_main.py', '--check', '--output', root / 'launch')
    assert prepared == checked
    check, = checked['batches']
    assert check['model_requests'] == check['business_operations'] == check['new_attempts'] == 0
    assert check['credential_read'] is False
    assert check['profile'] == PROFILE and check['scheduler']['slots'] == {'git': 2, 'pf': 2}
    assert check['scheduler']['max_workers'] == 4
    rows = check['assignments']
    assert [(r['seed'], r['group']) for r in rows] == [(s, g) for s in (41, 42, 43, 44) for g in ('git', 'pf')]
    assert len({r['output_dir'] for r in rows}) == 8
    assert all(r['stop_after_day'] == 56 and not Path(r['output_dir']).exists() for r in rows)
    assert check['model_configs'] == batch.round_four_models('r5-d56', PRICES)
    assert {s: tree_hash(p, exact=True) for s, p in fifo_worlds.items()} == original
    write_json(root / 'eight-run-check.json', checked)


@pytest.mark.parametrize('change', ['slots', 'total', 'seed', 'day', 'model'])
def test_strict_preflight_rejects_contract_drift(fifo_worlds, tmp_path, change):
    manifest, policy = freeze(fifo_worlds, tmp_path)
    if change == 'slots':
        manifest['scheduling']['slots']['git'] = 3
    elif change == 'total':
        manifest['scheduling']['max_workers'] = 5
    elif change == 'seed':
        manifest['seeds'][-1] = 45
    elif change == 'day':
        manifest['runs'][0]['stop_after_day'] = 49
    else:
        manifest['parameters']['model_configs']['agent']['provider'] = 'opendesign'
    path = tmp_path / 'batch/manifest.json'
    path.chmod(0o644)
    write_json(path, manifest)
    result = cli('team_driver.py', '--policy', policy, '--check', expected=1)
    assert result.returncode == 1
    assert not any(batch.attempts(r) for r in manifest['runs'])


def test_real_eight_run_fifo_isolation_d56_and_restart(fifo_worlds, fifo_disk, monkeypatch):
    from saas_bench.team_round5 import terminal_checkpoint
    root = fifo_disk
    manifest, policy = freeze(fifo_worlds, root)
    by = {(r['seed'], r['group']): r for r in manifest['runs']}
    options = dict(max_requests=150, hooks='round5_fifo_fixture',
        invalid_run=by[41, 'pf']['run_id'], wait_run=by[41, 'git']['run_id'],
        unknown_run=by[42, 'git']['run_id'],
        release_run=by[43, 'git']['run_id'], claims=str(root / 'batch/fifo-launches'))
    driver = Driver(None, root, monkeypatch, budget_seconds=1800, worker_options=options)
    driver.policies.append(policy)
    try:
        assert driver.driver.main(['--policy', str(policy), '--live']) == 3
        proofs = assert_closed(driver, policy)
        assert len(proofs) == 8
        events = _jsonl(policy.parent / 'controller-events.jsonl')
        live, peaks, order = set(), Counter(), {'git': [], 'pf': []}
        for row in events:
            if row['event'] == 'worker_launched':
                assert row['run_id'] not in live
                live.add(row['run_id'])
                group = next(r['group'] for r in manifest['runs'] if r['run_id'] == row['run_id'])
                order[group].append(row['run_id'])
                counts = Counter(next(r['group'] for r in manifest['runs'] if r['run_id'] == rid) for rid in live)
                assert len(live) <= 4 and counts['git'] <= 2 and counts['pf'] <= 2
                for group, count in counts.items():
                    peaks[group] = max(peaks[group], count)
            elif row['event'] == 'worker_reaped':
                live.remove(row['run_id'])
        assert not live and peaks == {'git': 2, 'pf': 2}
        assert order == {g: [by[s, g]['run_id'] for s in (41, 42, 43, 44)] for g in ('git', 'pf')}
        position = {(r['event'], r.get('run_id')): i for i, r in enumerate(events)}
        assert position['worker_launched', by[43, 'git']['run_id']] < position['worker_reaped', by[41, 'git']['run_id']]
        for seed, group in ((41, 'pf'), (42, 'git')):
            assert any(r['event'] == 'worker_launch_intent' and
                r.get('predecessor_run_id') == by[seed, group]['run_id'] for r in events)
        assert not (root / 'batch/SYSTEM_HOLD.json').exists()
        for entry in manifest['runs']:
            record = batch.read(batch.attempts(entry)[0] / 'attempt.json')
            if entry == by[41, 'pf']:
                assert record['hold'] and record['runtime_failure'] == 'model_attempt_limit'
            elif entry == by[42, 'git']:
                assert record['hold'] and record['runtime_failure'] in ('world_operation_failed', 'business_operation_unknown')
                assert (batch.attempts(entry)[0] / 'unknown-business-count.txt').read_text() == 'once\n'
            else:
                assert terminal_checkpoint(entry, record)['kind'] == 'observation_end'
                assert record['checkpoint_reused_at_stop']
                assert record['recovery_validation']['eligible']
        assert all(all(p['finished']['usage_matches_wire'].values()) for p in proofs)
        assert all(row['day'] < 56 for p in proofs for row in p['wire'])
        assert any(p.get('released_by_next_git_seed') for p in proofs)
        before = {p: tree_hash(p, exact=True) for r in manifest['runs'] for p in batch.attempts(r)}
        count = len(driver.children)
        assert driver.driver.main(['--policy', str(policy), '--live']) == 3
        assert len(driver.children) == count
        assert {p: tree_hash(p, exact=True) for p in before} == before
        entry = by[44, 'pf']
        record = batch.read(batch.attempts(entry)[0] / 'attempt.json')
        for changed in (dict(record, day=55), dict(record, reason='natural_end'), dict(record, checkpoint=None)):
            with pytest.raises((ValueError, TypeError)):
                terminal_checkpoint(entry, changed)
        checkpoint = Path(record['checkpoint'])
        receipt = checkpoint / 'checkpoint.json'
        content = receipt.read_bytes()
        try:
            data = json.loads(content)
            data['files']['world.nmdb'] = 'bad-checksum'
            write_json(receipt, data)
            with pytest.raises(ValueError, match='validation'):
                terminal_checkpoint(entry, record)
        finally:
            receipt.write_bytes(content)
        authority = checkpoint / 'private/workspace-publication/authority.sqlite'
        moved = authority.with_suffix('.saved')
        authority.rename(moved)
        try:
            with pytest.raises(ValueError):
                terminal_checkpoint(entry, record)
        finally:
            moved.rename(authority)
        assert terminal_checkpoint(entry, record)['recovery_eligible']
        write_json(root / 'fifo-summary.json', dict(order=order, peaks=dict(peaks), total_workers=count,
            restart_new_workers=len(driver.children)-count, attempt_preserved=True, d56_certified=6,
            isolated_holds=2, unknown_business_executions=1, local_model_only=True))
    finally:
        driver.close()


def test_permanent_quota_stops_all_slots_and_stays_latched(fifo_worlds, fifo_disk, monkeypatch):
    root = fifo_disk
    manifest, policy = freeze(fifo_worlds, root)
    entry = next(r for r in manifest['runs'] if r['group'] == 'pf')
    driver = Driver(None, root, monkeypatch, budget_seconds=600, worker_options=dict(
        max_requests=150, hooks='round5_fifo_fixture', reject_run=entry['run_id'], reject_status=429,
        reject_body={'type': 'error', 'error': {'type': 'GoUsageLimitError', 'message': 'Go usage limit exceeded'}}))
    driver.policies.append(policy)
    try:
        assert driver.driver.main(['--policy', str(policy), '--live']) == 3
        assert_closed(driver, policy)
        latch = root / 'batch/SYSTEM_HOLD.json'
        assert batch.read(latch)['reason'] == 'permanent_model_refusal'
        assert 'GoUsageLimitError' in batch.read(latch)['detail']
        assert all(not batch.attempts(r) for r in manifest['runs'] if r['seed'] in (43, 44))
        started = len(driver.children)
        assert started <= 4
        assert driver.driver.main(['--policy', str(policy), '--live']) == 3
        assert len(driver.children) == started
        write_json(root / 'quota-summary.json', dict(started_workers=started, subsequent_seeds_started=False,
            restart_new_workers=0, latch=batch.read(latch), resources_closed=True))
    finally:
        driver.close()


@pytest.mark.parametrize('fault', ['orphan', 'history'])
def test_restart_unsafe_history_never_launches(fifo_worlds, tmp_path, monkeypatch, fault):
    manifest, policy = freeze(fifo_worlds, tmp_path)
    if fault == 'orphan':
        path = tmp_path / 'batch/fifo-launches/orphan.json'
        write_json(path, dict(run_id=manifest['runs'][0]['run_id'], closed=False, unit='unconfirmed.scope'))
    else:
        path = policy.parent / 'controller-units.json'
        path.write_text('{broken')
    before = path.read_bytes()
    driver = Driver(None, tmp_path, monkeypatch, budget_seconds=120)
    try:
        assert driver.driver.main(['--policy', str(policy), '--live']) == 3
        assert driver.children == []
        assert path.read_bytes() == before
        assert not any(batch.attempts(r) for r in manifest['runs'])
        assert (tmp_path / 'batch/SYSTEM_HOLD.json').exists()
    finally:
        driver.close()


@pytest.mark.parametrize('lock_name', ['driver.lock', 'controller.lock', 'fifo-controller.lock'])
def test_nonblocking_launch_locks_prevent_duplicate_workers(fifo_worlds, tmp_path, monkeypatch, lock_name):
    _, policy = freeze(fifo_worlds, tmp_path)
    path = (tmp_path / 'batch' if lock_name == 'fifo-controller.lock' else policy.parent) / lock_name
    driver = Driver(None, tmp_path, monkeypatch, budget_seconds=120)
    try:
        with path.open('a') as locked:
            fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(BlockingIOError):
                driver.driver.main(['--policy', str(policy), '--live'])
        assert driver.children == []
        output = tmp_path / 'run-lock'
        with batch.run_lock(output, strict=True):
            with pytest.raises(RuntimeError, match='duplicate launch refused'):
                with batch.run_lock(output, strict=True):
                    pytest.fail('Duplicate run acquired the same launch lock')
    finally:
        driver.close()


def test_scope_conflict_retains_original_identity_and_never_signals_replacement(monkeypatch):
    from test_team_stop_residuals import controller_module
    from types import SimpleNamespace
    controller = controller_module()
    commands = []
    def command(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(returncode=0, stdout='LoadState=loaded\nActiveState=active\nInvocationID=replacement\nControlGroup=/replacement\n', stderr='')
    monkeypatch.setattr(controller.subprocess, 'run', command)
    monkeypatch.setattr(controller, 'unit_status', lambda *args, **kwargs: dict(unit='owned.scope'))
    saved = dict(unit='owned.scope', invocation_id='original', control_group='/original', strict_identity=True)
    for _ in range(2):
        saved = controller.scope_state(saved)
        assert saved['invocation_id'] == 'original' and saved['control_group'] == '/original'
        assert saved['scope_observation_unknown'] and not saved['scope_empty']
        with pytest.raises(RuntimeError, match='identity conflict'):
            controller.signal_scope(saved, 15)
    assert not any('kill' in command for command in commands)


def test_controller_persistence_failure_closes_workers_before_exit(fifo_worlds, fifo_disk, monkeypatch):
    root = fifo_disk
    manifest, policy = freeze(fifo_worlds, root)
    driver = Driver(None, root, monkeypatch, budget_seconds=300,
        worker_options=dict(max_requests=150, hooks='round5_fifo_fixture'))
    driver.policies.append(policy)
    persist = driver.driver.controller.write_json
    def disk_failure(path, value):
        if Path(path).name == 'controller-units.json':
            raise OSError(errno.ENOSPC, 'offline shared controller persistence fault')
        return persist(path, value)
    monkeypatch.setattr(driver.driver.controller, 'write_json', disk_failure)
    try:
        assert driver.driver.main(['--policy', str(policy), '--live']) == 2
        assert_closed(driver, policy)
        assert 0 < len(driver.children) <= 4
        assert not any(batch.attempts(r) for r in manifest['runs'] if r['seed'] in (43, 44))
        assert batch.read(root / 'batch/SYSTEM_HOLD.json')['reason'] == 'controller_persistence_failed'
    finally:
        driver.close()


def test_sqlite_backup_storage_failure_stops_all_slots(fifo_worlds, fifo_disk, monkeypatch):
    import sqlite3
    root = fifo_disk
    manifest, policy = freeze(fifo_worlds, root)
    entry = manifest['runs'][0]
    driver = Driver(None, root, monkeypatch, budget_seconds=300, worker_options=dict(
        max_requests=150, hooks='round5_fifo_fixture', sqlite_fault_run=entry['run_id'], bankrupt_day=14))
    driver.policies.append(policy)
    try:
        assert driver.driver.main(['--policy', str(policy), '--live']) == 3
        assert_closed(driver, policy)
        record = batch.read(batch.attempts(entry)[0] / 'attempt.json')
        assert record['runtime_failure'] == 'checkpoint_save_failed'
        assert record['checkpoint_error_errno'] is None
        assert record['checkpoint_error_sqlite_code'] & 0xff == sqlite3.SQLITE_READONLY
        assert record['resource_cleanup']['status'] == 'closed'
        latch = batch.read(root / 'batch/SYSTEM_HOLD.json')
        assert latch['run_id'] == entry['run_id']
        events = _jsonl(policy.parent / 'controller-events.jsonl')
        assert all(row['time'] <= latch['time'] for row in events if row['event'] == 'worker_launch_intent')
        started = len(driver.children)
        assert driver.driver.main(['--policy', str(policy), '--live']) == 3
        assert len(driver.children) == started
    finally:
        driver.close()


@pytest.mark.parametrize('when', ['week_boundary', 'midweek'])
def test_bankruptcy_requires_a_recoverable_archive(fifo_worlds, fifo_disk, monkeypatch, when):
    from saas_bench.team_round5 import terminal_checkpoint
    root = fifo_disk
    manifest, policy = freeze(fifo_worlds, root)
    entry = manifest['runs'][0]
    options = dict(bankrupt_action_run=entry['run_id']) if when == 'midweek' else dict(bankrupt_day=14)
    driver = Driver(None, root, monkeypatch, budget_seconds=300, worker_options=dict(
        max_requests=150, hooks='round5_fifo_fixture', **options))
    driver.policies.append(policy)
    try:
        _, record, proof = driver.run(policy, entry, expected=3 if when == 'midweek' else 0)
        if when == 'midweek':
            assert record['status'] == 'paused' and record['day'] == 0
            assert record['runtime_failure'] == 'checkpoint_save_failed'
            assert 'outside a recoverable handoff' in record['checkpoint_error']
            assert record['recovery_validation']['eligible']
            with pytest.raises(ValueError):
                terminal_checkpoint(entry, record)
        else:
            assert record['result']['status'] == 'natural_bankruptcy' and record['day'] == 14
            assert terminal_checkpoint(entry, record)['kind'] == 'natural_bankruptcy'
        assert not (root / 'batch/SYSTEM_HOLD.json').exists()
        assert proof['finished']['operations']
    finally:
        driver.close()


def test_unsupervised_fifo_execution_is_refused_before_an_attempt(fifo_worlds, tmp_path):
    manifest, policy = freeze(fifo_worlds, tmp_path)
    with pytest.raises(ValueError, match='supervised team_driver'):
        batch.prepare(policy.parent.parent / 'batch/manifest.json', manifest['runs'][0]['run_id'], live=True)
    assert not any(batch.attempts(r) for r in manifest['runs'])
