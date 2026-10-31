from collections import Counter
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys

import pytest

from saas_bench import team_batch as batch
from saas_bench.run_state import tree_hash, write_json
from test_preflight_team_batch import frozen
from test_checkpoint_driver import Driver, _jsonl


def cli(script, *args, expected=0):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'cli',
        str(batch.ROOT / 'scripts' / script), *map(str, args)], capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(str(batch.ROOT / part) for part in ('src', 'tests')),
                 PYTHONHASHSEED='0', PYTHON_DOTENV_DISABLED='1'), timeout=60)
    assert result.returncode == expected, result.stderr
    return json.loads(result.stdout) if expected == 0 else result


def freeze(frozen, root, profile, *, seeds=None, exploration=None, extra=(), expected=0):
    _, original, _, sources = frozen
    seeds = seeds or ([101, 102] if profile == 'r2-git-exploration' else [103, 42])
    prices = root.parent / (root.name + '-pricing.json')
    write_json(prices, original['pricing'])
    args = ['freeze', '--profile', profile, '--output', root, '--pricing', prices,
            '--price-source', original['price_source'], '--seed-selection-rule', 'offline fixture only',
            '--order-seed', '17']
    for seed in seeds:
        args.extend(['--d0', f'{seed}={sources[seed]}'])
    if exploration:
        args.extend(['--exploration-manifest', exploration])
    return cli('team_batch.py', *args, *extra, expected=expected)


@pytest.fixture
def round_two(frozen, tmp_path):
    exploration = freeze(frozen, tmp_path / 'exploration', 'r2-git-exploration',
                         extra=('--excluded-seed', '7'))
    path = tmp_path / 'exploration/manifest.json'
    paired = freeze(frozen, tmp_path / 'paired', 'r2-paired-pilot', exploration=path)
    return exploration, paired, path, tmp_path / 'paired/manifest.json'


def test_cli_freezes_exact_two_stage_assignments_and_independent_policy(round_two, tmp_path):
    exploration, paired, exp_path, paired_path = round_two
    for manifest, path, count, groups in [(exploration, exp_path, 2, {'git'}),
                                        (paired, paired_path, 4, {'git', 'pf'})]:
        rows = cli('team_batch.py', 'validate', '--manifest', path)
        assert rows == manifest['runs'] and len(rows) == count
        assert {row['group'] for row in rows} == set(manifest['prompts']) == groups
        assert {row['repeat'] for row in rows} == {1}
        assert {row['stop_after_day'] for row in rows} == {56}
        assert len({row['output_dir'] for row in rows}) == count
        assert all(manifest['batch_id'] in row['run_id'] for row in rows)
        assert all('D497' in p['system'] and 'D56' not in p['system']
                   for prompts in manifest['prompts'].values() for p in prompts.values())
        policy = cli('team_batch.py', 'policy', '--manifest', path, '--output', tmp_path / path.parent.name / 'control')
        check = cli('team_driver.py', '--policy', policy['policy'], '--check')
        save_sample('cli-' + manifest['profile'], dict(manifest=manifest, enumerated=rows, policy=policy, check=check))
        assert check['assignments'] == rows and check['profile'] == manifest['profile']
        assert check['model_requests'] == check['new_attempts'] == 0 and not check['credential_read']
        assert all(not Path(row['output_dir']).exists() for row in rows)
    assert set(exploration['seeds']).isdisjoint(paired['seeds'])
    assert set(paired['excluded_debug_seeds']) == {7, 101, 102}
    assert paired['exploration_source']['batch_id'] == exploration['batch_id']
    for seed in paired['seeds']:
        pair = [row for row in paired['runs'] if row['seed'] == seed]
        assert len(pair) == 2 and pair[0]['d0'] == pair[1]['d0'] == paired['d0'][str(seed)]['path']
    summary = cli('team_batch.py', 'summarize', '--manifest', exp_path)
    assert summary['pairs'] is None and summary['comparison_status'] == 'not_applicable_single_group'
    assert len(summary['runs']) == 2 and summary['profile'] == exploration['profile']
    summary = cli('team_batch.py', 'summarize', '--manifest', paired_path)
    assert len(summary['pairs']['pairs']) == 2 and summary['pairs']['complete_observation_pairs'] == 0


def test_policy_selection_keeps_frozen_order_and_refuses_empty_or_duplicate(round_two, tmp_path):
    _, manifest, _, path = round_two
    rows = manifest['runs']
    reverse = [arg for row in reversed(rows) for arg in ('--run-id', row['run_id'])]
    policy = cli('team_batch.py', 'policy', '--manifest', path, '--output', tmp_path / 'selected', *reverse)
    assert [row['run_id'] for row in policy['identity']['runs']] == [row['run_id'] for row in rows]
    saved = batch.read(policy['policy'])
    saved['identity']['runs'].reverse()
    write_json(policy['policy'], saved)
    assert cli('team_driver.py', '--policy', policy['policy'], '--check')['assignments'] == rows
    for group in ('git', 'pf'):
        policy = cli('team_batch.py', 'policy', '--manifest', path, '--output', tmp_path / group, '--group', group)
        check = cli('team_driver.py', '--policy', policy['policy'], '--check')
        assert check['assignments'] == [row for row in rows if row['group'] == group]
        chosen = check['assignments'][0]
        check = cli('team_driver.py', '--policy', policy['policy'], '--check', '--run-id', chosen['run_id'])
        assert check['assignments'] == [chosen]
    before = tree_hash(tmp_path / 'selected', exact=True)
    cli('team_batch.py', 'policy', '--manifest', path, '--output', tmp_path / 'selected', expected=1)
    assert tree_hash(tmp_path / 'selected', exact=True) == before
    for ids in (['missing'], [rows[0]['run_id']] * 2):
        target = tmp_path / 'invalid-policy'
        flags = [arg for run_id in ids for arg in ('--run-id', run_id)]
        cli('team_batch.py', 'policy', '--manifest', path, '--output', target, *flags, expected=1)
        assert not target.exists()


@pytest.mark.parametrize('case', ['one_seed', 'three_seeds', 'excluded', 'duplicate', 'paired_schedule',
                                  'missing_exploration', 'reused_exploration'])
def test_new_batch_rejects_incorrect_assignments_before_creating_directory(frozen, tmp_path, case):
    profile, seeds, extra, prior = 'r2-git-exploration', [101, 102], (), None
    if case == 'one_seed':
        seeds = [101]
    elif case == 'three_seeds':
        seeds = [101, 102, 103]
    elif case == 'excluded':
        extra = ('--excluded-seed', '101')
    elif case == 'duplicate':
        seeds = [101, 101]
    elif case == 'paired_schedule':
        extra = ('--scheduling', 'paired')
    else:
        profile = 'r2-paired-pilot'
        if case == 'reused_exploration':
            freeze(frozen, tmp_path / 'prior', 'r2-git-exploration')
            prior = tmp_path / 'prior/manifest.json'
    target = tmp_path / 'rejected'
    result = freeze(frozen, target, profile, seeds=seeds, extra=extra, exploration=prior,
                    expected=2 if case == 'duplicate' else 1)
    assert not target.exists() and result.stderr


def test_repeat_freeze_has_new_identity_but_same_semantic_order(frozen, tmp_path):
    first = freeze(frozen, tmp_path / 'a', 'r2-git-exploration')
    second = freeze(frozen, tmp_path / 'b', 'r2-git-exploration', seeds=[102, 101])
    assert [(r['seed'], r['repeat'], r['group']) for r in first['runs']] == [
        (r['seed'], r['repeat'], r['group']) for r in second['runs']]
    assert first['batch_id'] != second['batch_id']
    assert {r['run_id'] for r in first['runs']}.isdisjoint(r['run_id'] for r in second['runs'])


def test_paired_freeze_excludes_actual_exploration_runs_when_seed_metadata_is_stale(frozen, tmp_path):
    previous = freeze(frozen, tmp_path / 'exploration', 'r2-git-exploration')
    previous['seeds'] = [102, 103]
    source = tmp_path / 'stale-seeds.json'
    write_json(source, previous)
    target = tmp_path / 'paired'
    freeze(frozen, target, 'r2-paired-pilot', seeds=[101, 42], exploration=source, expected=1)
    assert not target.exists()


def policy_for(path, root):
    return Path(cli('team_batch.py', 'policy', '--manifest', path, '--output', root)['policy'])


def assert_closed(driver, policy):
    receipt = batch.read(policy.parent / 'controller-exit.json')
    assert not driver.expired.is_set() and receipt['children_reaped'] and receipt['memory_sampler']['closed']
    assert all(unit['scope_empty'] for unit in receipt['units'].values())
    assert all(child.poll() is not None for child, _ in driver.children)
    proofs = [batch.read(spec.with_suffix('.proof.json')) for _, spec in driver.children]
    for proof in proofs:
        assert proof['errors'] == [] and proof['model_server_closed'] and all(proof['clients_closed'])
        assert proof['closed']['live_role_threads'] == proof['closed']['sockets_remaining'] == []
        assert proof['closed']['public_requests']['active'] == []
        assert not proof['closed']['public_requests']['queued']
    return proofs


def save_sample(name, data):
    if target := os.environ.get('CEOBENCH_R229_SAMPLES'):
        write_json(Path(target) / (name + '.json'), data)


@pytest.mark.parametrize('profile', ['r2-git-exploration', 'r2-paired-pilot'])
def test_real_driver_finishes_assigned_queue_at_d56(round_two, frozen, tmp_path, monkeypatch, profile):
    exp, pair, exp_path, pair_path = round_two
    manifest, path = (exp, exp_path) if profile == exp['profile'] else (pair, pair_path)
    policy = policy_for(path, tmp_path / 'control')
    driver = Driver(frozen, tmp_path, monkeypatch, budget_seconds=600, worker_options={'max_requests': 100})
    driver.policies.append(policy)
    try:
        assert driver.driver.main(['--policy', str(policy), '--live']) == 0
        proofs = assert_closed(driver, policy)
        expected = list(range(0, 56, 7))
        records = [batch.read(batch.attempts(entry)[0] / 'attempt.json') for entry in manifest['runs']]
        assert all(record['status'] == 'finished' and record['day'] == 56 and
                   record['profile'] == profile and record['batch_id'] == manifest['batch_id'] for record in records)
        assert all(len(batch.attempts(entry)) == 1 for entry in manifest['runs'])
        for proof in proofs:
            assert proof['finished']['day'] == 56
            assert all(value == 'once\n' * 8 for value in proof['finished']['script_counts'].values())
            for role in ('ceo', 'growth', 'ops_finance'):
                assert sorted({row['day'] for row in proof['wire'] if row['role'] == role}) == expected
            assert all('D497' in row['body']['messages'][0]['content'] for row in proof['wire'])
            assert all(proof['finished']['usage_matches_wire'].values())
        events = _jsonl(policy.parent / 'controller-events.jsonl')
        launches = [row for row in events if row['event'] == 'worker_launched']
        assert Counter(row['run_id'] for row in launches) == Counter(row['run_id'] for row in manifest['runs'])
        for group in ('git', 'pf'):
            queue = [entry['run_id'] for entry in manifest['runs'] if entry['group'] == group]
            assert [row['run_id'] for row in launches if batch.selected(manifest, row['run_id'])['group'] == group] == queue
            for previous, current in zip(queue, queue[1:]):
                reaped = next(i for i, row in enumerate(events) if row['event'] == 'worker_reaped' and row['run_id'] == previous)
                launched = next(i for i, row in enumerate(events) if row['event'] == 'worker_launched' and row['run_id'] == current)
                assert reaped < launched
        summary = cli('team_batch.py', 'summarize', '--manifest', path)
        if profile == 'r2-paired-pilot':
            assert summary['pairs']['complete_observation_pairs'] == 2
            assert summary['pairs']['complete_d112_pairs'] == 0
        else:
            assert summary['pairs'] is None
        before = len(driver.children)
        assert driver.driver.main(['--policy', str(policy), '--live']) == 0
        assert len(driver.children) == before
        entry = manifest['runs'][0]
        terminal = batch.read(batch.attempts(entry)[0] / 'attempt.json')['checkpoint']
        _, record, proof = driver.run(policy, entry, checkpoint=terminal, resume=True)
        assert record['day'] == 56 and proof['wire'] == []
        save_sample(profile, dict(events=events, attempts=records, summary=summary,
            worker_days=[[dict(role=r['role'], day=r['day'], phase=r['phase']) for r in p['wire']] for p in proofs],
            script_counts=[p['finished']['script_counts'] for p in proofs], terminal_resume=record))
    finally:
        driver.close()


@pytest.mark.parametrize('stop', ['natural', 'operator'])
def test_real_driver_preserves_early_termination_and_explicit_week_restore(round_two, frozen, tmp_path, monkeypatch, stop):
    manifest, _, path, _ = round_two
    entry = manifest['runs'][0]
    policy = policy_for(path, tmp_path / 'control')
    options = {'max_requests': 100, 'bankrupt_day': 14} if stop == 'natural' else {'stop_on_day': 0}
    driver = Driver(frozen, tmp_path, monkeypatch, budget_seconds=600, worker_options=options)
    driver.policies.append(policy)
    try:
        _, record, proof = driver.run(policy, entry)
        assert record['day'] == (14 if stop == 'natural' else 0)
        if stop == 'natural':
            assert record['result']['status'] == 'natural_bankruptcy'
            assert {row['day'] for row in proof['wire']} == {0, 7}
            assert all(value == 'once\n' * 2 for value in proof['finished']['script_counts'].values())
        else:
            assert batch.read(policy.parent / 'controller-exit.json')['operator_stopped']
            assert record['status'] == 'paused'
        driver.worker_options = {'max_requests': 100}
        _, restored, resumed = driver.run(policy, entry, week=1, resume=True)
        assert restored['day'] == 56 and restored['status'] == 'finished'
        assert resumed['ready']['day'] == 0 and all(resumed['ready'][key] for key in (
            'world_equal', 'roles_equal', 'weeks_equal', 'messages_equal', 'rng_equal', 'log_prefixes_equal'))
        assert len(batch.attempts(entry)) == 2
        assert all(not batch.attempts(peer) for peer in manifest['runs'][1:])
        save_sample(stop, dict(first=record, restored=restored, proof=resumed['ready']))
    finally:
        driver.close()


def test_real_driver_holds_failed_group_once_and_finishes_peer_queue(round_two, frozen, tmp_path, monkeypatch):
    _, manifest, _, path = round_two
    failed = next(row for row in manifest['runs'] if row['group'] == 'pf')
    policy = policy_for(path, tmp_path / 'control')
    driver = Driver(frozen, tmp_path, monkeypatch, budget_seconds=600,
                    worker_options={'max_requests': 100, 'reject_run': failed['run_id']})
    driver.policies.append(policy)
    try:
        assert driver.driver.main(['--policy', str(policy), '--live']) == 3
        assert_closed(driver, policy)
        events = _jsonl(policy.parent / 'controller-events.jsonl')
        launches = [row['run_id'] for row in events if row['event'] == 'worker_launched']
        expected = [row['run_id'] for row in manifest['runs'] if row['group'] == 'git'] + [failed['run_id']]
        assert Counter(launches) == Counter(expected)
        held = batch.read(batch.attempts(failed)[0] / 'attempt.json')
        assert held['status'] == 'paused' and held['hold'] and len(batch.attempts(failed)) == 1
        assert all(batch.read(batch.attempts(row)[0] / 'attempt.json')['status'] == 'finished'
                   for row in manifest['runs'] if row['group'] == 'git')
        driver.worker_options = {'max_requests': 100}
        _, resumed, _ = driver.run(policy, failed, checkpoint=held['checkpoint'], resume=True)
        assert resumed['day'] == 56 and resumed['status'] == 'finished'
        save_sample('hold', dict(events=events, held=held, resumed=resumed))
    finally:
        driver.close()


if __name__ == '__main__':
    assert sys.argv[1] == 'cli'
    connect, get = socket.socket.connect, os.environ.get

    def no_network(channel, address):
        if isinstance(address, tuple):
            raise AssertionError('Batch preparation and --check cannot access network')
        return connect(channel, address)

    def no_credentials(key, default=None):
        if key == 'OPENCODE_API_KEY':
            raise AssertionError('Batch preparation and --check cannot read provider credentials')
        return get(key, default)

    socket.socket.connect, os.environ.get = no_network, no_credentials
    sys.argv = sys.argv[2:]
    sys.path.insert(0, str(Path(sys.argv[0]).parent))
    runpy.run_path(sys.argv[0], run_name='__main__')
