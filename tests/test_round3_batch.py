from collections import Counter
import json
import os
from pathlib import Path

import httpx
import pytest

from saas_bench import team_batch as batch
from saas_bench.agent_capabilities import AgentCapabilities
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.role_policy import ROLES
from saas_bench.run_state import tree_hash, write_json
from saas_bench.team_checkpoint import validate_runtime_data, validate_snapshot
from test_preflight_team_batch import PRICES
from test_preflight_team_flow import completion, final
from test_checkpoint_driver import Driver, _jsonl
from test_round2_batch import cli, assert_closed


@pytest.fixture(scope='module')
def worlds(tmp_path_factory):
    root = tmp_path_factory.mktemp('round3-worlds')
    sources = {}
    for seed in (1048, 1049):
        sources[seed] = root / f'd0-{seed}'
        batch.seal_d0(seed, sources[seed], PRICES, price_source=PRICES['source'])
    return sources


def freeze(worlds, root, *, seeds=(1049, 1048), extra=(), expected=0):
    prices = root.parent / (root.name + '-prices.json')
    write_json(prices, PRICES)
    args = ['freeze', '--profile', 'r3-d70', '--output', root, '--pricing', prices,
            '--price-source', PRICES['source'], '--order-seed', '17']
    for seed in seeds:
        args += ['--d0', f'{seed}={worlds.get(seed, worlds[1048])}']
    return cli('team_batch.py', *args, *extra, expected=expected)


@pytest.fixture
def round_three(worlds, tmp_path):
    manifest = freeze(worlds, tmp_path / 'batch')
    return manifest, tmp_path / 'batch/manifest.json'


def sample(name, value):
    if target := os.environ.get('CEOBENCH_R314_SAMPLES'):
        write_json(Path(target) / (name + '.json'), value)


def policy_for(path, root, *args):
    return Path(cli('team_batch.py', 'policy', '--manifest', path, '--output', root, *args)['policy'])


def test_cli_fixed_new_runs_capabilities_and_side_effect_free_check(round_three, worlds, tmp_path):
    manifest, path = round_three
    assert manifest['profile'] == 'r3-d70' and manifest['seeds'] == [1048, 1049]
    assert [(r['seed'], r['group'], r['repeat']) for r in manifest['runs']] == [
        (seed, group, 1) for seed in (1048, 1049) for group in ('git', 'pf')]
    assert {r['stop_after_day'] for r in manifest['runs']} == {70}
    assert 'exploration_source' not in manifest
    assert manifest['parameters'] == dict(batch.PARAMETERS, capabilities={
        group: AgentCapabilities.round_three(group).to_dict() for group in ('git', 'pf')})
    assert manifest['parameters']['total_days'] == 500 and manifest['parameters']['effective_end'] == 497
    assert manifest['order']['method'] == 'fixed seeds ascending then groups'
    for group in ('git', 'pf'):
        rows = [r for r in manifest['runs'] if r['group'] == group]
        assert [r['seed'] for r in rows] == [1048, 1049]
        assert manifest['scheduling']['queues'][group] == [r['run_id'] for r in rows]
        assert all(r['capabilities'] == AgentCapabilities.round_three(group).to_dict() for r in rows)
        for prompt in manifest['prompts'][group].values():
            assert 'D497' in prompt['system'] and 'D70' not in prompt['system']
            names = {t['name'] for t in prompt['tools']}
            assert not names & {'text_create', 'text_revise', 'text_retire', 'text_list', 'relation_batch'}
            assert ('escalate' in names) == (group == 'pf')
        policy = policy_for(path, tmp_path / group, '--group', group)
        selected = batch.read(policy)
        selected['identity']['runs'].reverse()
        write_json(policy, selected)
        check = cli('team_driver.py', '--policy', policy, '--check')
        assert check['assignments'] == rows and check['model_requests'] == check['new_attempts'] == 0
        assert not check['credential_read']
        one = cli('team_driver.py', '--policy', policy, '--check', '--run-id', rows[0]['run_id'])
        assert one['assignments'] == rows[:1]
    assert all(not Path(r['output_dir']).exists() for r in manifest['runs'])
    original = {seed: tree_hash(source, exact=True) for seed, source in worlds.items()}
    second = freeze(worlds, tmp_path / 'second', seeds=(1048, 1049), extra=('--order-seed', '99'))
    assert [(r['seed'], r['group']) for r in second['runs']] == [(r['seed'], r['group']) for r in manifest['runs']]
    assert not {r['run_id'] for r in second['runs']} & {r['run_id'] for r in manifest['runs']}
    assert {seed: tree_hash(source, exact=True) for seed, source in worlds.items()} == original
    sample('cli', dict(manifest=manifest, check=check))


@pytest.mark.parametrize('case', ['wrong_seed', 'one_seed', 'duplicate', 'paired', 'exploration'])
def test_bad_round_three_shape_rejected_before_creation(worlds, tmp_path, case):
    seeds, extra, code = (1048, 1049), (), 1
    if case == 'wrong_seed':
        seeds = (1048, 1050)
    elif case == 'one_seed':
        seeds = (1048,)
    elif case == 'duplicate':
        seeds, code = (1048, 1048), 2
    elif case == 'paired':
        extra = ('--scheduling', 'paired')
    else:
        extra = ('--exploration-manifest', tmp_path / 'not-required.json')
    target = tmp_path / 'rejected'
    freeze(worlds, target, seeds=seeds, extra=extra, expected=code)
    assert not target.exists()


def check_configuration(proof, entry):
    expected = entry['capabilities']
    assert proof['ready']['configuration']['capabilities'] == expected
    assert all(value == dict(memory_max_chars=100000, max_output_tokens=32768,
                             invalid_generation_attempts=4)
               for value in proof['ready']['generation_policy'].values())
    for row in proof['wire']:
        body = row['body']
        assert body['model'] == batch.MODEL and body['reasoning_effort'] == 'high'
        assert body['temperature'] == 1 and body['max_tokens'] == 32768
        assert body['thinking'] == {'type': 'enabled'}
        assert 'D497' in body['messages'][0]['content'] and 'D70' not in body['messages'][0]['content']
        names = {t['function']['name'] for t in body['tools']}
        assert not names & {'text_create', 'text_revise', 'text_retire', 'text_list', 'relation_batch'}
        assert ('escalate' in names) == (entry['group'] == 'pf')
        for tool in body['tools']:
            if tool['function']['name'] in ('bash', 'write_file', 'edit_file'):
                assert ('note' in tool['function']['parameters']['properties']) == (entry['group'] == 'pf')
    assert all(proof['finished']['usage_matches_wire'].values())


def terminal(entry, record):
    assert record['status'] == 'finished' and record['day'] == 70
    snapshot, state = validate_snapshot(record['checkpoint'])
    validate_runtime_data(snapshot, state)
    assert state['day'] == 70 and state['stage'] == 'boundary'
    assert state['configuration']['capabilities'] == entry['capabilities']
    assert all(state['roles'][role]['generation_policy']['memory_max_chars'] == 100000 for role in ROLES)


def test_real_driver_fixed_queues_d70_and_checkpoint_resume(round_three, tmp_path, monkeypatch):
    manifest, path = round_three
    policy = policy_for(path, tmp_path / 'control')
    selected = batch.read(policy)
    selected['identity']['runs'].reverse()
    write_json(policy, selected)
    driver = Driver(None, tmp_path, monkeypatch, budget_seconds=900, worker_options={'max_requests': 120})
    driver.policies.append(policy)
    try:
        assert driver.driver.main(['--policy', str(policy), '--live']) == 0
        proofs = assert_closed(driver, policy)
        events = _jsonl(policy.parent / 'controller-events.jsonl')
        launches = [row for row in events if row['event'] == 'worker_launched']
        assert Counter(row['run_id'] for row in launches) == Counter(r['run_id'] for r in manifest['runs'])
        records = []
        for entry in manifest['runs']:
            record = batch.read(batch.attempts(entry)[0] / 'attempt.json')
            records.append(record)
            terminal(entry, record)
        for (_, spec), proof in zip(driver.children, proofs):
            entry = next(r for r in manifest['runs'] if r['run_id'] in str(batch.read(spec.with_suffix('.proof.json'))['checkpoints']['ready']))
            check_configuration(proof, entry)
            assert proof['ready']['day'] == 0 and not proof['ready']['source']
            assert all(v == 'once\n' * 10 for v in proof['finished']['script_counts'].values())
            for role in ROLES:
                assert sorted({r['day'] for r in proof['wire'] if r['role'] == role}) == list(range(0, 70, 7))
            assert all(r['day'] < 70 for r in proof['wire'])
            assert any(m['kind'] == 'question' and m['status'] == 'delivered' for m in proof['finished']['messages'])
        for group in ('git', 'pf'):
            queue = [r['run_id'] for r in manifest['runs'] if r['group'] == group]
            assert [r['run_id'] for r in launches if batch.selected(manifest, r['run_id'])['group'] == group] == queue
            assert [batch.selected(manifest, rid)['seed'] for rid in queue] == [1048, 1049]
            assert next(i for i, e in enumerate(events) if e['event'] == 'worker_reaped' and e['run_id'] == queue[0]) < next(
                i for i, e in enumerate(events) if e['event'] == 'worker_launched' and e['run_id'] == queue[1])
        summary = cli('team_batch.py', 'summarize', '--manifest', path)
        assert summary['pairs']['complete_observation_pairs'] == 2
        assert all(p['observation_end'] == 70 for p in summary['pairs']['pairs'])
        count = len(driver.children)
        assert driver.driver.main(['--policy', str(policy), '--live']) == 0
        assert len(driver.children) == count
        for group in ('git', 'pf'):
            entry = next(r for r in manifest['runs'] if r['group'] == group)
            _, record, proof = driver.run(policy, entry, week=10, resume=True)
            terminal(entry, record)
            assert proof['ready']['day'] == 63
            assert all(proof['ready'][key] for key in ('world_equal', 'roles_equal', 'weeks_equal', 'messages_equal', 'rng_equal', 'log_prefixes_equal'))
            check_configuration(proof, entry)
            assert {r['day'] for r in proof['wire']} == {63}
            assert all(v == 'once\n' * 10 for v in proof['finished']['script_counts'].values())
            _, end, last = driver.run(policy, entry, checkpoint=record['checkpoint'], resume=True)
            terminal(entry, end)
            assert last['wire'] == [] and last['finished']['script_counts'] == proof['finished']['script_counts']
        sample('driver', dict(events=events, records=records, proofs=proofs, summary=summary, recovery=driver.samples))
    finally:
        driver.close()


@pytest.mark.parametrize('stop', ['natural', 'operator', 'hold'])
def test_real_driver_terminal_contracts_and_manual_restore(round_three, tmp_path, monkeypatch, stop):
    manifest, path = round_three
    entry = next(r for r in manifest['runs'] if r['group'] == 'pf')
    policy = policy_for(path, tmp_path / 'control', '--group', 'pf')
    options = {'bankrupt_day': 14} if stop == 'natural' else {'stop_on_day': 0} if stop == 'operator' else {'reject_run': entry['run_id']}
    driver = Driver(None, tmp_path, monkeypatch, budget_seconds=600, worker_options=dict(max_requests=120, **options))
    driver.policies.append(policy)
    try:
        _, record, proof = driver.run(policy, entry, expected=3 if stop == 'hold' else 0)
        assert record['day'] == (14 if stop == 'natural' else 0)
        assert len(batch.attempts(entry)) == 1
        assert all(not batch.attempts(r) for r in manifest['runs'] if r != entry)
        if stop == 'natural':
            assert record['result']['status'] == 'natural_bankruptcy'
            assert {r['day'] for r in proof['wire']} == {0, 7}
        elif stop == 'operator':
            assert batch.read(policy.parent / 'controller-exit.json')['operator_stopped']
        else:
            assert record['status'] == 'paused' and record['hold']
        driver.worker_options = {'max_requests': 120}
        _, resumed, restored = driver.run(policy, entry, week=1, resume=True)
        terminal(entry, resumed)
        check_configuration(restored, entry)
        assert restored['ready']['day'] == 0
        sample(stop, dict(first=record, resumed=resumed, proof=restored))
    finally:
        driver.close()


@pytest.mark.parametrize('group', ['git', 'pf'])
def test_simulator_model_configuration_survives_checkpoint(round_three, tmp_path, group):
    manifest, _ = round_three
    entry = next(row for row in manifest['runs'] if row['group'] == group)
    requests = []
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        return completion(final('offline simulator reply'), body.get('stream', False))
    def clients(role):
        return batch.checked_client(role, 'offline-r3', transport=httpx.MockTransport(respond))
    def simulator(conn, config, rng):
        return batch.simulator(conn, config, rng, client=clients('simulator'))
    runtime = batch.make_team(manifest, entry, tmp_path / 'source', client_factory=clients,
                              simulator_factory=simulator)
    restored = None
    try:
        def probe(team):
            config = team.server.tools.config
            assert config.total_days == 500 and team.effective_end == 497
            assert config.social_post_llm_provider == config.enterprise_llm_provider == 'opencode'
            assert config.social_post_llm_model == config.enterprise_llm_model == batch.MODEL
            customer = team.server.simulator.customer_simulator
            customer.usage_recorder.path = team.root / 'private/simulator-model-usage.jsonl'
            result = customer.complete_text(provider=config.social_post_llm_provider,
                model=config.social_post_llm_model, user='offline probe', max_tokens=100)
            assert result[0] == 'offline simulator reply'
            usage = _jsonl(customer.usage_recorder.path)
            assert any(row['event'] == 'http_request' and row['body'] == requests[-1] for row in usage)
        probe(runtime)
        snapshot = runtime.checkpoint()
        restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored', client_factory=clients,
                                              simulator_factory=simulator)
        probe(restored)
        assert len(requests) == 2
        assert requests[0] == requests[1]
        assert requests[0]['thinking'] == {'type': 'disabled'}
        assert requests[0]['reasoning_effort'] == 'none'
        assert requests[0]['model'] == batch.MODEL
        assert restored.configuration['capabilities'] == entry['capabilities']
        sample('simulator-' + group, dict(requests=requests, configuration=restored.configuration))
    finally:
        if restored:
            batch.close_team(restored)
        batch.close_team(runtime)
