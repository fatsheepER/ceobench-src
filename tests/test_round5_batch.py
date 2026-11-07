import json
import os
import tempfile
from pathlib import Path

import pytest

from saas_bench import team_batch as batch
from saas_bench.agent_capabilities import AgentCapabilities
from saas_bench.run_state import tree_hash, write_json
from test_preflight_team_batch import PRICES
from test_round2_batch import cli
from test_round3_batch import worlds, policy_for


def freeze(worlds, root):
    return batch.freeze_batch(root, worlds, PRICES, price_source=PRICES['source'], profile='r5-d56')


@pytest.fixture
def r5_driver_root(tmp_path):
    target = Path(os.environ.get('CEOBENCH_R516_EVIDENCE', tmp_path))
    target.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='r5-driver-', dir=target))


def test_d28_missing_attempt_evidence_is_incomplete(tmp_path):
    from saas_bench.team_round5 import retained_d28
    (tmp_path / 'attempt-001').mkdir()
    result = retained_d28(dict(output_dir=str(tmp_path), category='formal'))
    assert result['status'] == 'incomplete' and result['errors']


def test_simulator_config_and_model_usage_survive_restore(worlds, tmp_path):
    import httpx
    from saas_bench.multi_agent import MultiAgentRuntime
    from test_preflight_team_flow import completion, final
    manifest = freeze(worlds, tmp_path / 'batch')
    models = manifest['parameters']['model_configs']
    requests = []
    def respond(request):
        requests.append(dict(endpoint=str(request.url), body=json.loads(request.content)))
        return completion(final('offline simulator'))
    def client(role):
        return batch.checked_client(role, 'offline', model_config=models['simulator' if role == 'simulator' else 'agent'],
            transport=httpx.MockTransport(respond))
    def world(conn, config, rng):
        return batch.simulator(conn, config, rng, client=client('simulator'))
    team = batch.make_team(manifest, manifest['runs'][-1], tmp_path / 'first', client_factory=client, simulator_factory=world)
    restored = None
    def probe(runtime):
        customer = runtime.server.simulator.customer_simulator
        customer.usage_recorder.path = runtime.root / 'private/simulator-model-usage.jsonl'
        customer.complete_text(provider='opencode', model=batch.MODEL, user='offline probe', max_tokens=99)
        sent = requests[-1]
        assert sent['endpoint'] == models['simulator']['base_url'] + 'chat/completions'
        assert sent['body']['reasoning_effort'] == 'none' and sent['body']['thinking'] == {'type': 'disabled'}
        assert sent['body']['max_tokens'] == 32768
        recorded = [json.loads(line) for line in customer.usage_recorder.path.read_text().splitlines()]
        assert next(r for r in reversed(recorded) if r['event'] == 'http_request')['body'] == sent['body']
    try:
        probe(team)
        restored = MultiAgentRuntime.restore(team.checkpoint(), tmp_path / 'restored', client_factory=client, simulator_factory=world)
        probe(restored)
        assert requests[0] == requests[1]
        assert restored.configuration['model_configs'] == models
    finally:
        if restored:
            batch.close_team(restored)
        batch.close_team(team)


def test_round_five_profile_reaches_real_runtime(worlds, tmp_path):
    manifest = freeze(worlds, tmp_path / 'batch')
    assert [(r['seed'], r['group']) for r in manifest['runs']] == [
        (seed, group) for seed in (1048, 1049) for group in ('git', 'pf')]
    assert {r['stop_after_day'] for r in manifest['runs']} == {56}
    for group in ('git', 'pf'):
        entry = next(r for r in manifest['runs'] if r['group'] == group)
        team = batch.make_team(manifest, entry, tmp_path / group)
        try:
            assert team.configuration['workspace_layout'] == 'shared-v1'
            assert team.configuration['workspace_protocol'] == ('snapshot-v1' if group == 'pf' else 'direct-v1')
            assert entry['capabilities'] == AgentCapabilities.round_three(group).to_dict()
            assert len({r.workspace for r in team.roles.values()}) == 1
            assert (getattr(team, 'workspace_publication', None) is not None) == (group == 'pf')
            assert team.configuration['model_configs'] == manifest['parameters']['model_configs']
        finally:
            batch.close_team(team)


def test_launch_prepares_four_runs_and_checks_without_requests(worlds, tmp_path):
    source = freeze(worlds, tmp_path / 'source')
    prices = tmp_path / 'prices.json'
    write_json(prices, PRICES)
    before = {seed: tree_hash(path, exact=True) for seed, path in worlds.items()}
    prepared = cli('round5_main.py', '--prepare', '--output', tmp_path / 'main',
        '--d0-source-manifest', tmp_path / 'source/manifest.json', '--pricing', prices)
    checked = cli('round5_main.py', '--check', '--output', tmp_path / 'main')
    assert prepared == checked
    assert checked['model_requests'] == checked['business_operations'] == 0
    assert checked['batches'][0]['profile'] == 'r5-d56'
    launch = batch.read(tmp_path / 'main/launch.json')
    assert launch['launcher'] == 'scripts/round5_main.py'
    assert launch['live_started'] is False
    rows = checked['batches'][0]['assignments']
    assert len(rows) == len({row['run_id'] for row in rows}) == 4
    assert all(not Path(row['output_dir']).exists() for row in rows)
    assert checked['batches'][0]['contract_check']['prompts_match']
    assert checked['batches'][0]['contract_check']['source_matches']
    assert checked['batches'][0]['contract_check']['live_verified'] is False
    assert {seed: tree_hash(path, exact=True) for seed, path in worlds.items()} == before


@pytest.mark.parametrize('change', ['endpoint', 'prompt', 'permissions', 'source', 'capabilities', 'route', 'policy', 'sdk', 'tokenizer'])
def test_check_rejects_changed_contract_without_attempts(worlds, tmp_path, change):
    from saas_bench.run_state import file_hash
    manifest = freeze(worlds, tmp_path / 'batch')
    path = tmp_path / 'batch/manifest.json'
    policy = policy_for(path, tmp_path / 'control')
    if change == 'endpoint':
        manifest['runs'][0]['stop_after_day'] = 49
    elif change == 'prompt':
        manifest['prompts']['pf']['ceo']['system'] += '\nUnapproved prompt'
    elif change == 'permissions':
        manifest['parameters']['workspace_contracts']['git']['advance_roles'] = ['growth']
    elif change == 'source':
        manifest['source_sha256'] = 'old-source'
    elif change == 'capabilities':
        manifest['runs'][0]['capabilities']['text_tools'] = True
    elif change == 'route':
        manifest['parameters']['model_configs']['agent']['base_url'] = 'https://example.invalid/v1/'
    elif change == 'sdk':
        manifest['parameters']['openai_sdk_version'] = 'old-sdk'
    elif change == 'tokenizer':
        manifest['tokenizer']['revision'] = 'wrong-tokenizer'
    path.chmod(0o644)
    write_json(path, manifest)
    freeze_path = path.parent / 'freeze.json'
    freeze_path.chmod(0o644)
    write_json(freeze_path, dict(manifest_sha256=file_hash(path)))
    controls = batch.read(policy)
    controls['identity']['manifest_sha256'] = file_hash(path)
    if change == 'policy':
        controls['identity']['runs'][0]['output_dir'] = str(tmp_path / 'wrong')
    write_json(policy, controls)
    result = cli('team_driver.py', '--policy', policy, '--check', expected=1)
    assert 'R5' in result.stderr
    assert all(not Path(r['output_dir']).exists() for r in manifest['runs'])



@pytest.mark.parametrize('copied_source', [True, False])
def test_other_checkout_cannot_certify_frozen_runtime(worlds, tmp_path, copied_source):
    import shutil
    import subprocess
    import sys
    manifest = freeze(worlds, tmp_path / 'batch')
    policy = policy_for(tmp_path / 'batch/manifest.json', tmp_path / 'control')
    checkout = tmp_path / 'different-checkout'
    shutil.copytree(batch.ROOT / 'src', checkout / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(batch.ROOT / 'scripts', checkout / 'scripts', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(batch.ROOT / 'public', checkout / 'public', ignore=shutil.ignore_patterns('__pycache__'))
    result = subprocess.run([sys.executable, str(batch.ROOT / 'tests/test_round2_batch.py'),
        'cli', str(checkout / 'scripts/team_driver.py'),
        '--policy', str(policy), '--check'], capture_output=True, text=True, timeout=90,
        env=dict(os.environ, PYTHONPATH=str((checkout if copied_source else batch.ROOT) / 'src')))
    assert result.returncode == 1 and 'executing source root' in result.stderr, result.stdout + result.stderr
    assert all(not Path(r['output_dir']).exists() for r in manifest['runs'])

def test_real_four_run_driver_d28_retention_repeated_resume_and_d56(worlds, r5_driver_root, monkeypatch):
    from collections import Counter
    from test_checkpoint_driver import Driver, _jsonl
    from test_round2_batch import assert_closed
    from saas_bench.team_round5 import retained_d28
    tmp_path = r5_driver_root
    manifest = freeze(worlds, tmp_path / 'batch')
    policy = policy_for(tmp_path / 'batch/manifest.json', tmp_path / 'control')
    driver = Driver(None, tmp_path, monkeypatch, budget_seconds=900,
        worker_options=dict(max_requests=150, hooks='round5_batch_fixture'))
    driver.policies.append(policy)
    try:
        assert driver.driver.main(['--policy', str(policy), '--live']) == 0
        proofs = assert_closed(driver, policy)
        events = _jsonl(policy.parent / 'controller-events.jsonl')
        assert Counter(r['run_id'] for r in events if r['event'] == 'worker_launched') == Counter(r['run_id'] for r in manifest['runs'])
        for entry in manifest['runs']:
            record = batch.read(batch.attempts(entry)[0] / 'attempt.json')
            assert record['status'] == 'finished' and record['day'] == 56
            assert record['checkpoint_reused_at_stop']
            inventory = retained_d28(entry)
            assert inventory['status'] == 'retained' and len(inventory['snapshots']) == 1
        for proof in proofs:
            assert all(value == 'once\n' * 8 for value in proof['counts'].values())
            assert all(value == dict(memory_max_chars=100000, max_output_tokens=32768, invalid_generation_attempts=4)
                       for value in proof['ready']['generation_policy'].values())
            assert proof['memory'] == {role: 'R5-MEMORY-' + role for role in proof['memory']}
            assert all(proof['finished']['usage_matches_wire'].values())
            assert all(row['day'] < 56 for row in proof['wire'])
            for row in proof['wire']:
                assert row['body']['model'] == batch.MODEL
                assert row['body']['reasoning_effort'] == 'high'
                assert row['body']['thinking'] == {'type': 'enabled'}
                assert row['body']['max_tokens'] == 32768
        for group in ('git', 'pf'):
            entry = next(r for r in manifest['runs'] if r['group'] == group)
            d28 = Path(retained_d28(entry)['snapshots'][0]['snapshot'])
            fingerprint = tree_hash(d28, exact=True)
            _, record, proof = driver.run(policy, entry, week=5, resume=True)
            assert record['source_checkpoint'] == str(d28)
            assert record['day'] == 56 and proof['ready']['day'] == 28
            assert all(proof['ready'][key] for key in ('world_equal', 'roles_equal', 'rng_equal', 'weeks_equal', 'messages_equal', 'log_prefixes_equal'))
            assert proof['restored_scripts'] == proof['saved_scripts']
            assert all(value == 'once\n' * 8 for value in proof['counts'].values())
            assert all(proof['finished']['usage_matches_wire'].values())
            assert tree_hash(d28, exact=True) == fingerprint
            if group == 'pf':
                assert proof['publication_exact'] and proof['past_confirmation_retained']
                assert proof['drafts']
            _, last, terminal = driver.run(policy, entry, checkpoint=record['checkpoint'], resume=True)
            assert last['day'] == 56 and terminal['wire'] == []
            assert terminal['counts'] == proof['counts']
            valid, waste = batch.run_usage(entry, PRICES)
            assert {r['call_id'] for r in valid['records']}.isdisjoint(r['call_id'] for r in waste['records'])
            assert waste['team']['complete']['total_tokens'] > 0
        summary = batch.summarize(tmp_path / 'batch/manifest.json')
        assert summary['pairs']['complete_observation_pairs'] == 2
        assert all(r['d28_retention']['status'] == 'retained' for r in summary['runs'])
    finally:
        driver.close()


@pytest.mark.parametrize('termination', ['natural', 'permanent', 'retry_stop'])
def test_r5_terminal_conditions_and_manual_recovery(worlds, r5_driver_root, monkeypatch, termination):
    from test_checkpoint_driver import Driver, _jsonl
    tmp_path = r5_driver_root
    manifest = freeze(worlds, tmp_path / 'batch')
    policy = policy_for(tmp_path / 'batch/manifest.json', tmp_path / 'control')
    entry = next(r for r in manifest['runs'] if r['group'] == 'pf')
    options = dict(max_requests=150, hooks='round5_batch_fixture')
    if termination == 'natural':
        options['bankrupt_day'] = 14
    else:
        options.update(reject_run=entry['run_id'], reject_status=429)
        if termination == 'permanent':
            options.update(record_retry_waits=True, reject_body={'type': 'error', 'error': {
                'type': 'GoUsageLimitError', 'message': 'Go usage limit exceeded'},
                'metadata': {'limitName': 'monthly'}})
        else:
            options.update(group_stop_on_retry=True, reject_body={'error': {'code': 'rate_limit_exceeded', 'message': 'temporary rate limit'}})
    driver = Driver(None, tmp_path, monkeypatch, budget_seconds=900, worker_options=options)
    driver.policies.append(policy)
    try:
        if termination == 'natural':
            attempt, record, proof = driver.run(policy, entry)
        else:
            from test_round2_batch import assert_closed
            assert driver.driver.main(['--policy', str(policy), '--live']) == (3 if termination == 'permanent' else 0)
            proofs = assert_closed(driver, policy)
            proof = next(p for p in proofs if p['ready']['configuration']['mode'] == 'pf')
            attempt = batch.attempts(entry)[0]
            record = batch.read(attempt / 'attempt.json')
            assert all(batch.read(batch.attempts(r)[0] / 'attempt.json')['status'] == 'finished'
                       for r in manifest['runs'] if r['group'] == 'git')
            assert not batch.attempts(manifest['runs'][-1])
        assert len(batch.attempts(entry)) == 1
        if termination == 'natural':
            assert all(not batch.attempts(r) for r in manifest['runs'] if r != entry)
            assert record['day'] == 14 and record['result']['status'] == 'natural_bankruptcy'
        elif termination == 'permanent':
            assert record['hold'] and record['status'] == 'paused'
            assert 'GoUsageLimitError' in record['hold_detail']
            assert len(proof['wire']) == 1 and proof['retry_waits'] == []
            assert record['recovery_validation']['eligible']
        else:
            assert batch.read(policy.parent / 'controller-exit.json')['operator_stopped']
            assert record['day'] == 0
            assert proof['group_stop_after_retry'] and proof['stop_thread_closed']
            assert proof['r5_retry_waits'][0] == 10
            assert not record.get('hold')
        if termination != 'natural':
            usage = _jsonl(attempt / 'runtime/private/ceo/model-usage.jsonl')
            request = next(r for r in usage if r['event'] == 'request')
            assert request['retry_policy'] == dict(attempts=0, delays=[10, 20, 40, 80], first_byte_seconds=600)
            assert all(r['cost_usd'] is None for r in usage if r['event'] == 'response')
        driver.worker_options = dict(max_requests=150, hooks='round5_batch_fixture')
        _, resumed, restored = driver.run(policy, entry, week=1, resume=True)
        assert resumed['status'] == 'finished' and resumed['day'] == 56
        assert restored['ready']['day'] == 0
        assert all(restored['finished']['usage_matches_wire'].values())
    finally:
        driver.close()
