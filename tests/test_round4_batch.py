from collections import Counter
import json
from pathlib import Path

import httpx
import pytest

from saas_bench import team_batch as batch
from saas_bench.agent_capabilities import AgentCapabilities
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.role_policy import ROLES
from saas_bench.run_state import tree_hash
from saas_bench.team_checkpoint import validate_snapshot, validate_runtime_data
from saas_bench.team_models import ROUND_FOUR_PROFILES
from test_preflight_team_batch import PRICES
from test_preflight_team_flow import completion, final
from test_checkpoint_driver import Driver, _jsonl
from test_round2_batch import cli, assert_closed
from test_round3_batch import worlds, policy_for, sample


def freeze(worlds, root, profile):
    seeds = (1048, 1049) if profile == 'r4-d56' else (1048,)
    return batch.freeze_batch(root, {s: worlds[s] for s in seeds}, PRICES, price_source=PRICES['source'], profile=profile)


def test_six_assignments_launch_checks_and_no_shared_attempts(worlds, tmp_path):
    original = {seed: tree_hash(path, exact=True) for seed, path in worlds.items()}
    source = freeze(worlds, tmp_path / 'source', 'r4-d56')
    prices = tmp_path / 'pricing.json'
    batch.write_json(prices, PRICES)
    prepared = cli('round4.py', '--prepare', '--output', tmp_path / 'six',
        '--d0-source-manifest', tmp_path / 'source/manifest.json', '--pricing', prices)
    checked = cli('round4.py', '--check', '--output', tmp_path / 'six')
    assert prepared == checked
    assert [len(row['assignments']) for row in checked['batches']] == [4, 1, 1]
    all_rows = [r for check in checked['batches'] for r in check['assignments']]
    assert len({r['run_id'] for r in all_rows}) == len({r['output_dir'] for r in all_rows}) == 6
    for check in checked['batches']:
        assert check['model_requests'] == check['new_attempts'] == 0 and not check['credential_read']
        assert check['model_configs']['simulator'] == checked['batches'][0]['model_configs']['simulator']
        for row in check['assignments']:
            assert row['stop_after_day'] == 56
            assert row['capabilities'] == AgentCapabilities.round_three(row['group']).to_dict()
            assert row['d0'] == str(worlds[row['seed']])
            assert not Path(row['output_dir']).exists()
        result = cli('team_batch.py', 'summarize', '--manifest', Path(check['assignments'][0]['output_dir']).parents[1] / 'manifest.json')
        assert (result['pairs'] is None) == (check['profile'] != 'r4-d56')
    assert {seed: tree_hash(path, exact=True) for seed, path in worlds.items()} == original


def assert_wire(proof, models):
    config = models['agent']
    assert proof['ready']['configuration']['model_configs'] == models
    assert all(proof['finished']['usage_matches_wire'].values())
    for row in proof['wire']:
        body = row['body']
        assert body['model'] == config['model']
        assert row['endpoint'].endswith('/responses' if config['api'] == 'responses' else '/chat/completions')
        if config['api'] == 'responses':
            assert body['reasoning']['effort'] == 'high'
            assert body['max_output_tokens'] == 32768 and body['store'] is False
            assert body['include'] == ['reasoning.encrypted_content']
            assert all(tool['strict'] is False for tool in body['tools'])
            assert 'temperature' not in body and 'reasoning_effort' not in body
            prompt = body['instructions']
        else:
            assert body['reasoning_effort'] == config['reasoning_effort']
            assert body['thinking'] == {'type': 'enabled'} and body['max_tokens'] == 32768
            prompt = body['messages'][0]['content']
        assert 'D497' in prompt and 'D56' not in prompt
    if config['api'] == 'responses':
        assert any(any(item.get('type') == 'function_call_output' for item in row['body']['input']) for row in proof['wire'])
        assert any(any(item.get('encrypted_content') == 'offline-reasoning' for item in row['body']['input']) for row in proof['wire'])


@pytest.mark.parametrize('profile', ROUND_FOUR_PROFILES)
def test_real_driver_d56_model_wire_usage_and_selected_checkpoint(worlds, tmp_path, monkeypatch, profile):
    manifest = freeze(worlds, tmp_path / 'batch', profile)
    policy = policy_for(tmp_path / 'batch/manifest.json', tmp_path / 'control')
    driver = Driver(None, tmp_path, monkeypatch, budget_seconds=900, worker_options={'max_requests': 120})
    driver.policies.append(policy)
    try:
        assert driver.driver.main(['--policy', str(policy), '--live']) == 0
        proofs = assert_closed(driver, policy)
        events = _jsonl(policy.parent / 'controller-events.jsonl')
        assert Counter(r['run_id'] for r in events if r['event'] == 'worker_launched') == Counter(r['run_id'] for r in manifest['runs'])
        for entry in manifest['runs']:
            record = batch.read(batch.attempts(entry)[0] / 'attempt.json')
            assert record['status'] == 'finished' and record['day'] == 56
            snapshot, state = validate_snapshot(record['checkpoint'])
            validate_runtime_data(snapshot, state)
            assert state['day'] == 56 and state['stage'] == 'boundary'
            assert state['configuration']['model_configs'] == manifest['parameters']['model_configs']
        for proof in proofs:
            assert_wire(proof, manifest['parameters']['model_configs'])
            assert all(value == 'once\n' * 8 for value in proof['finished']['script_counts'].values())
            for role in ROLES:
                assert sorted({row['day'] for row in proof['wire'] if row['role'] == role}) == list(range(0, 56, 7))
        entry = manifest['runs'][0]
        _, restored, proof = driver.run(policy, entry, week=2, resume=True)
        assert restored['day'] == 56 and proof['ready']['day'] == 7
        assert_wire(proof, manifest['parameters']['model_configs'])
        assert len(batch.attempts(entry)) == 2
        sample(profile, dict(manifest=manifest, result=restored, proof=proof))
    finally:
        driver.close()


@pytest.mark.parametrize('profile', ROUND_FOUR_PROFILES)
def test_simulator_is_independent_on_first_and_restored_requests(worlds, tmp_path, profile):
    manifest = freeze(worlds, tmp_path / 'batch', profile)
    models = manifest['parameters']['model_configs']
    entry = manifest['runs'][-1]
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        return completion(final('simulator unchanged'))
    def client(role):
        return batch.checked_client(role, 'offline', model_config=models['simulator' if role == 'simulator' else 'agent'], transport=httpx.MockTransport(respond))
    def world(conn, config, rng):
        return batch.simulator(conn, config, rng, client=client('simulator'))
    team = batch.make_team(manifest, entry, tmp_path / 'first', client_factory=client, simulator_factory=world)
    restored = None
    try:
        for runtime in (team,):
            assert runtime.configuration['model_configs'] == models
        def probe(runtime):
            customer = runtime.server.simulator.customer_simulator
            assert customer.model == batch.MODEL and customer.reasoning_effort == 'none'
            customer.usage_recorder.path = runtime.root / 'private/simulator-model-usage.jsonl'
            config = runtime.server.tools.config
            customer.complete_text(provider=config.social_post_llm_provider, model=config.social_post_llm_model,
                user='probe', max_tokens=99)
            assert requests[-1]['model'] == batch.MODEL
            assert requests[-1]['reasoning_effort'] == 'none' and requests[-1]['thinking'] == {'type': 'disabled'}
            assert requests[-1]['max_tokens'] == 32768
        probe(team)
        restored = MultiAgentRuntime.restore(team.checkpoint(), tmp_path / 'restored', client_factory=client, simulator_factory=world)
        probe(restored)
        assert requests[0] == requests[1]
        assert restored.configuration['model_configs'] == models
    finally:
        if restored:
            batch.close_team(restored)
        batch.close_team(team)


@pytest.mark.parametrize('stop', ['natural', 'operator', 'quota'])
def test_sol_driver_terminal_and_manual_restore(worlds, tmp_path, monkeypatch, stop):
    manifest = freeze(worlds, tmp_path / 'batch', 'r4-sol-high-pf')
    entry = manifest['runs'][0]
    policy = policy_for(tmp_path / 'batch/manifest.json', tmp_path / 'control')
    options = ({'bankrupt_day': 14} if stop == 'natural' else {'stop_on_day': 0} if stop == 'operator' else
               {'reject_run': entry['run_id'], 'reject_status': 429, 'reject_code': 'credit_balance_exhausted'})
    driver = Driver(None, tmp_path, monkeypatch, budget_seconds=600, worker_options=dict(max_requests=120, **options))
    driver.policies.append(policy)
    try:
        _, record, proof = driver.run(policy, entry, expected=3 if stop == 'quota' else 0)
        assert len(batch.attempts(entry)) == 1
        if stop == 'quota':
            assert record['status'] == 'paused' and record['hold']
            assert len(proof['wire']) == 1 and 'credit_balance_exhausted' in record['hold_detail']
        elif stop == 'operator':
            assert batch.read(policy.parent / 'controller-exit.json')['operator_stopped']
        else:
            assert record['result']['status'] == 'natural_bankruptcy' and record['day'] == 14
        driver.worker_options = {'max_requests': 120}
        _, resumed, proof = driver.run(policy, entry, week=1, resume=True)
        assert resumed['status'] == 'finished' and resumed['day'] == 56
        assert_wire(proof, manifest['parameters']['model_configs'])
        sample('sol-' + stop, dict(first=record, resumed=resumed, proof=proof))
    finally:
        driver.close()


def test_one_main_group_holds_and_peer_queue_continues(worlds, tmp_path, monkeypatch):
    manifest = freeze(worlds, tmp_path / 'batch', 'r4-d56')
    entry = next(row for row in manifest['runs'] if row['group'] == 'pf')
    policy = policy_for(tmp_path / 'batch/manifest.json', tmp_path / 'control')
    driver = Driver(None, tmp_path, monkeypatch, budget_seconds=600,
                    worker_options={'max_requests': 120, 'reject_run': entry['run_id']})
    driver.policies.append(policy)
    try:
        assert driver.driver.main(['--policy', str(policy), '--live']) == 3
        assert_closed(driver, policy)
        assert len(batch.attempts(entry)) == 1
        assert batch.read(batch.attempts(entry)[0] / 'attempt.json')['hold']
        assert all(batch.read(batch.attempts(row)[0] / 'attempt.json')['status'] == 'finished'
                   for row in manifest['runs'] if row['group'] == 'git')
        assert not batch.attempts(manifest['runs'][-1])
    finally:
        driver.close()


def test_missing_credential_holds_before_resources_and_can_prepare_from_week_one(worlds, tmp_path, monkeypatch):
    manifest = freeze(worlds, tmp_path / 'batch', 'r4-sol-high-pf')
    path, entry = tmp_path / 'batch/manifest.json', manifest['runs'][0]
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    result = batch.prepare(path, entry['run_id'], live=True)
    assert result['status'] == 'paused' and result['reason'] == 'model_configuration_missing'
    attempt = batch.attempts(entry)[0]
    assert not (attempt / 'runtime').exists()
    restored = batch.prepare(path, entry['run_id'], resume=True, week=1, live=False)
    assert restored['status'] == 'ready' and restored['attempt'] == 2
    assert (attempt / 'HOLD').exists()


def test_selected_legacy_checkpoint_keeps_legacy_clients(worlds, tmp_path, monkeypatch):
    legacy = batch.freeze_batch(tmp_path / 'legacy', worlds, PRICES,
        price_source=PRICES['source'], profile='r3-d70')
    original = next(row for row in legacy['runs'] if row['group'] == 'pf')
    first = batch.prepare(tmp_path / 'legacy/manifest.json', original['run_id'], live=False)
    assert first['status'] == 'ready'
    snapshot = Path(first['checkpoint'])
    original_hash = tree_hash(snapshot, exact=True)
    manifest = freeze(worlds, tmp_path / 'sol', 'r4-sol-high-pf')
    entry = manifest['runs'][0]
    monkeypatch.setenv('CEOBENCH_RECOVERY_CHECKPOINTS', json.dumps({entry['run_id']: str(snapshot)}))
    restored = batch.prepare(tmp_path / 'sol/manifest.json', entry['run_id'],
        resume=True, force_resume=True, live=False)
    assert restored['status'] == 'ready'
    assert restored['effective_configuration']['model'] == batch.MODEL
    assert restored['effective_configuration'].get('model_configs') is None
    _, state = validate_snapshot(restored['checkpoint'])
    assert {state['roles'][role]['client']['endpoint'] for role in ROLES} == {batch.ENDPOINT}
    assert state['simulator_usage']['client']['endpoint'] == batch.ENDPOINT
    assert tree_hash(snapshot, exact=True) == original_hash


@pytest.mark.parametrize('profile', ['legacy', *ROUND_FOUR_PROFILES])
def test_observer_consumes_real_frozen_manifest_and_model_usage(worlds, tmp_path, monkeypatch, profile):
    import importlib.util
    import shutil
    from saas_bench.model_usage import ModelUsage
    from saas_bench.team_models import round_four_models
    from test_preflight_usage import reply

    root = tmp_path / 'batch'
    if profile == 'legacy':
        manifest = batch.freeze_batch(root, worlds, PRICES, price_source=PRICES['source'],
            selection_rule='Offline observer compatibility fixture')
        assert manifest['parameters'] == batch.PARAMETERS
    else:
        manifest = freeze(worlds, root, profile)
    policy = batch.create_policy(root / 'manifest.json', tmp_path / 'control')
    expected = policy['identity']
    Path(expected['host_logs']).mkdir()  # Created by the controller before polling.
    scripts = batch.ROOT / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('r4_observer', scripts / 'team_observer.py')
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    entry = manifest['runs'][0]
    attempt = Path(entry['output_dir']) / 'attempt-001'
    batch.write_json(attempt / 'attempt.json', dict(status='running', attempt=1, day=0))
    explicit = manifest['parameters'].get('model_configs')
    models = explicit or round_four_models('r4-d56', PRICES)
    logs, clients, wire = [], [], []

    def respond(request):
        body = json.loads(request.content)
        wire.append(body)
        response = reply('responses' if request.url.path.endswith('/responses') else 'chat')
        response['model'] = body['model']
        return httpx.Response(200, json=response)

    def invoke(role, *, wrong_model=False):
        config = models['simulator' if role == 'simulator' else 'agent']
        client = batch.checked_client(role, 'observer-fixture', transport=httpx.MockTransport(respond),
            model_config=config if explicit else None)
        clients.append(client)
        path = attempt / 'runtime/private' / role / 'model-usage.jsonl'
        if path not in logs:
            logs.append(path)
        usage = ModelUsage(path, role, pricing=PRICES['rates'])
        usage.attach(client)
        kwargs = dict(model='wrong-model' if wrong_model else config['model'])
        if config['api'] == 'responses':
            kwargs.update(input='probe', reasoning={'effort': config['reasoning_effort']}, store=False,
                include=config['include'], max_output_tokens=config['max_output_tokens'])
            usage.call('responses', kwargs, lambda: client.responses.create(**kwargs))
        else:
            kwargs.update(messages=[{'role': 'user', 'content': 'probe'}],
                reasoning_effort=config['reasoning_effort'], extra_body={'thinking': {'type': config['thinking']}},
                max_tokens=config['max_output_tokens'])
            usage.call('chat', kwargs, lambda: client.chat.completions.create(**kwargs))

    state, output = {}, tmp_path / 'control/observer'
    quota = lambda: {key: dict(status='ok', percent=1) for key in ('rolling', 'weekly', 'monthly')}
    def poll():
        return observer.poll(root / 'manifest.json', output, expected, state, quota_fn=quota)
    try:
        for role in (*ROLES, 'simulator'):
            invoke(role)
        health = poll()
        assert health['checks'] == health['route_errors'] == []
        assert all(not row['transient_receipt_failures'] for row in _jsonl(output / 'events.jsonl'))
        assert health['billing_coverage'] == 'complete'
        assert state['billing_groups'][entry['run_id']]['calls'] == 4
        assert all(health['coverage'][str(path)]['offset'] == path.stat().st_size for path in logs)
        sent = [row['body'] for path in logs for row in _jsonl(path) if row['event'] == 'http_request']
        assert sent == wire
        invoke('growth', wrong_model=True)
        health = poll()
        assert health['checks'] == []
        assert [row['issue'] for row in health['route_errors']] == ['model_route_changed', 'served_model_changed']
        assert state['billing_groups'][entry['run_id']]['calls'] == 5
        # Copied inherited receipts plus a restarted observer must not double count.
        inherited = attempt.with_name('attempt-002')
        shutil.copytree(attempt, inherited)
        state = batch.read(output / 'state.json')
        health = poll()
        assert health['checks'] == health['route_errors'] == []
        assert state['billing_groups'][entry['run_id']]['calls'] == 5
        assert all(row['backlog_bytes'] == 0 for row in health['coverage'].values())
    finally:
        for client in clients:
            client.close()
    assert all(client._client.is_closed for client in clients)
