"""Independent Chat configurations on direct teams and the engineering launcher."""
import copy
import json
from pathlib import Path

import pytest

from saas_bench import team_batch as batch
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.role_policy import ROLES
from saas_bench.team_models import make_chat_models, model_issue, round_four_models, validate_models, validate_profile_models
from test_chat_provider_switching import PROVIDERS, endpoint, isolated_environment
from test_preflight_team_batch import PRICES
from test_preflight_team_observer import observer
from test_round2_batch import cli
from test_team_stop_residuals import controller_module


@pytest.mark.parametrize('provider', PROVIDERS)
def test_factory_keeps_separate_defaults_and_unknown_prices(provider):
    other = next(name for name in PROVIDERS if name != provider)
    models = make_chat_models(provider, simulator_provider=other, environment={})
    assert validate_models(models) is models
    for role, selected in (('agent', provider), ('simulator', other)):
        config = models[role]
        assert (config['model'], config['base_url'], config['credential_env']) == PROVIDERS[selected]
        assert config['api'] == 'chat' and config['max_output_tokens'] == 32768
        assert config['memory_characters'] == 100000 and config['pricing']['rates'] is None
        assert config['tokenizer']['provider'] == selected
        assert config['reasoning_effort'] == ('high' if role == 'agent' else 'none')
        assert config['thinking'] == (None if selected == 'opendesign' else 'enabled' if role == 'agent' else 'disabled')
    assert models['agent'] is not models['simulator']
    custom = make_chat_models(provider, agent_model='custom-agent', simulator_model='custom-simulator',
        agent_base_url='http://127.0.0.1:10/agent/', simulator_base_url='http://127.0.0.1:11/simulator/',
        agent_reasoning_effort='none', environment={provider.upper() + '_BASE_URL': 'http://127.0.0.1:12/other/'})
    assert custom['agent']['model'] == 'custom-agent' and custom['simulator']['model'] == 'custom-simulator'
    assert custom['agent']['base_url'].endswith('/agent/') and custom['simulator']['base_url'].endswith('/simulator/')
    assert custom['agent']['reasoning_effort'] == custom['simulator']['reasoning_effort'] == 'none'
    with pytest.raises(ValueError, match='high or none'):
        make_chat_models(provider, agent_reasoning_effort='medium')


def test_frozen_models_stay_independent_from_new_defaults():
    from saas_bench.team_models import ROUND_FOUR_PROFILES
    for profile in (*ROUND_FOUR_PROFILES, 'r5-d56'):
        frozen = round_four_models(profile, PRICES)
        before = copy.deepcopy(frozen)
        validate_profile_models(profile, frozen, PRICES)
        make_chat_models('opendesign', simulator_provider='deepseek', environment={})
        assert frozen == before
        assert frozen['simulator']['provider'] == 'opencode'
        expected = 'inference/go/openai/v1/' if profile == 'r5-d56' else 'zen/go/v1/'
        assert frozen['simulator']['base_url'] == 'https://opencode.ai/' + expected
        changed = copy.deepcopy(frozen)
        changed['agent'] = make_chat_models('deepseek', environment={})['agent']
        with pytest.raises(ValueError, match='differs'):
            validate_profile_models(profile, changed, PRICES)


def test_official_simulator_audit_accepts_disabled_thinking_without_optional_effort():
    models = make_chat_models('opendesign', simulator_provider='deepseek', environment={})
    config = models['simulator']
    row = dict(event='http_request', role='simulator', endpoint=config['base_url'] + 'chat/completions',
        body=dict(model=config['model'], thinking={'type': 'disabled'}, max_tokens=config['max_output_tokens']))
    assert model_issue(row, models) is None
    row['body']['thinking']['type'] = 'enabled'
    assert model_issue(row, models) == 'simulator_thinking_changed'


@pytest.mark.parametrize('provider', PROVIDERS)
def test_factory_direct_team_all_roles_deliver_chat_contract(tmp_path, monkeypatch, endpoint,
        make_initialized_sim, make_agent_tools, provider):
    url, wire = endpoint
    models = make_chat_models(provider, simulator_provider='opendesign', agent_base_url=url + 'agent/',
        simulator_base_url=url + 'simulator/', environment={})
    conn, sim, config = make_initialized_sim()
    config.simulator_model_config = models['simulator']
    tools = make_agent_tools(conn, config)
    clients = []
    def client(role):
        value = batch.checked_client(role, 'provider-fixture', model_config=models['agent'])
        clients.append(value)
        return value
    team = None
    try:
        team = MultiAgentRuntime(tmp_path / 'team', tools, conn=conn, simulator=sim,
            client_factory=client, model_configs=models, model=models['agent']['model'],
            reasoning_effort='high', public_dir=batch.ROOT / 'public')
        for role in ROLES:
            agent = team.roles[role].agent
            assert agent.act('fixture role prompt', 0, False, {'day': 0}).tool == 'read_file'
            body = wire[-1]['body']
            assert wire[-1]['path'] == '/agent/chat/completions'
            assert body['model'] == PROVIDERS[provider][0] and body['max_tokens'] == 32768
            assert body['reasoning_effort'] == 'high'
            assert ('thinking' in body) == (provider != 'opendesign')
            assert wire[-1]['opencode_header'] == (provider == 'opencode')
            entries = [json.loads(line) for line in team.roles[role].usage.path.read_text().splitlines()]
            assert any(row['event'] == 'http_request' and row['body'] == body for row in entries)
            assert team.roles[role].usage.summary['known']['input_tokens'] == 10
    finally:
        if team:
            team.close()
        for value in clients:
            value.close()
        conn.close()
    assert all(value._client.is_closed for value in clients)


@pytest.fixture
def engineering(tmp_path):
    source = tmp_path / 'd0'
    batch.seal_d0(42, source, PRICES, price_source=PRICES['source'])
    models = make_chat_models('opendesign', simulator_provider='deepseek',
        agent_base_url='http://127.0.0.1:9/agent/', simulator_base_url='http://127.0.0.1:9/simulator/', environment={})
    manifest = batch.freeze_batch(tmp_path / 'engineering', {42: source}, PRICES,
        price_source=PRICES['source'], category='engineering', model_configs=models,
        scheduling='group_independent')
    return tmp_path / 'engineering/manifest.json', manifest, models, source


def test_public_engineering_check_and_prepare_with_independent_configs(engineering, tmp_path):
    path, manifest, models, source = engineering
    policy = Path(cli('team_batch.py', 'policy', '--manifest', path, '--output', tmp_path / 'controller')['policy'])
    checked = cli('team_driver.py', '--policy', policy, '--check')
    assert checked['model_configs'] == models
    assert checked['model_requests'] == checked['new_attempts'] == 0 and not checked['credential_read']
    entry = manifest['runs'][0]
    assert all(not Path(row['output_dir']).exists() for row in manifest['runs'])
    ready = batch.prepare(path, entry['run_id'], live=False)
    assert ready['status'] == 'ready', ready
    from saas_bench.team_checkpoint import validate_snapshot
    _, state = validate_snapshot(ready['checkpoint'])
    assert state['day'] == 0 and state['configuration']['model_configs'] == models
    for role in ROLES:
        assert state['roles'][role]['generation_policy']['max_output_tokens'] == 32768
    with pytest.raises(ValueError, match='explicit-model profile or unprofiled engineering'):
        batch.freeze_batch(tmp_path / 'invalid-formal', {41: source, 42: source}, PRICES, price_source=PRICES['source'],
            category='formal', selection_rule='fixture only', model_configs=models)


def test_controller_worker_selects_current_simulator_without_opencode_warning(engineering, tmp_path, monkeypatch, caplog):
    path, manifest, models, _ = engineering
    controller = controller_module()
    entry = manifest['runs'][0]
    expected = dict(host_logs=str(tmp_path / 'host-logs'), manifest=str(path))
    class Gate:
        def __init__(self, *args, **kwargs):
            self.expected = expected
        def manifest(self):
            return manifest
        def response(self, response):
            pass
    monkeypatch.setattr(controller, 'RequestGate', Gate)
    monkeypatch.setattr(controller, 'verify_worker_scope', lambda _: {})
    monkeypatch.delenv('OPENCODE_API_KEY')
    observed = []
    def prepare(*args, **kwargs):
        import os
        observed.append((os.environ['CEOBENCH_SIMULATOR_LLM_PROVIDER'], os.environ['CEOBENCH_SIMULATOR_LLM_MODEL']))
        assert kwargs['live'] is True
        return dict(status='finished', run_id=entry['run_id'], attempt=1, day=28)
    monkeypatch.setattr(controller.batch, 'prepare', prepare)
    import tempfile
    import os
    monkeypatch.setattr(tempfile, 'tempdir', tempfile.tempdir)
    monkeypatch.setenv('TMPDIR', os.environ['TMPDIR'])
    assert controller._worker(tmp_path / 'policy.json', entry['run_id'], resume=False, parent_pid=None,
        week=None, force_resume=False, warnings=type('Warnings', (), {'rows': []})()) == 0
    assert observed == [('deepseek', 'deepseek-flash')]
    assert not any('OPENCODE_API_KEY' in row.message for row in caplog.records)


def test_observer_does_not_query_unselected_opencode_balance(observer, monkeypatch):
    observer.manifest['parameters']['model_configs'] = make_chat_models('opendesign', simulator_provider='deepseek', environment={})
    monkeypatch.delenv('OPENCODE_API_KEY')
    def unexpected():
        raise AssertionError('Unselected OpenCode quota API was called')
    monkeypatch.setattr(observer.module.round5, 'quota_health', unexpected)
    health = observer.module.poll(observer.manifest_path, observer.output, observer.expected, observer.state)
    assert health['opencode_quota_monitor'] == 'not_applicable'
    assert health['other_provider_balance'] == 'unknown'
    assert health['quota'] is None and health['quota_error'] is None
    assert health['checks'] == [] and health['status'] == 'healthy'
    assert 'quota_last_success' not in health['state']


def test_mixed_provider_observer_still_checks_opencode_quota(observer):
    observer.manifest['parameters']['model_configs'] = make_chat_models('opendesign', simulator_provider='opencode', environment={})
    calls = []
    def quota():
        calls.append(True)
        return observer.quota()
    health = observer.module.poll(observer.manifest_path, observer.output, observer.expected, observer.state, quota_fn=quota)
    assert calls == [True] and health['opencode_quota_monitor'] == 'applicable'
    assert health['quota'] and health['other_provider_balance'] == 'unknown'


def test_real_driver_local_rejection_hold_and_selected_resume(engineering, tmp_path, monkeypatch):
    from test_checkpoint_driver import Driver
    from saas_bench.team_checkpoint import validate_snapshot
    _, _, _, source = engineering
    # The established local HTTP driver serves this exact fixture model ID.
    models = make_chat_models('opendesign', simulator_provider='deepseek', agent_model=batch.MODEL,
        agent_base_url='http://127.0.0.1:9/agent/', simulator_base_url='http://127.0.0.1:9/simulator/', environment={})
    manifest = batch.freeze_batch(tmp_path / 'driver-batch', {42: source}, PRICES,
        price_source=PRICES['source'], category='engineering', scheduling='group_independent', model_configs=models)
    policy = Path(cli('team_batch.py', 'policy', '--manifest', tmp_path / 'driver-batch/manifest.json',
        '--output', tmp_path / 'driver-control', '--group', 'git')['policy'])
    entry = next(row for row in manifest['runs'] if row['group'] == 'git')
    driver = Driver(None, tmp_path, monkeypatch, budget_seconds=240,
        worker_options={'reject_run': entry['run_id'], 'reject_status': 401, 'max_requests': 80})
    driver.policies.append(policy)
    import team_observer
    monkeypatch.setattr(driver.driver.controller, 'poll', team_observer.poll)
    try:
        _, held, proof = driver.run(policy, entry, expected=3)
        assert held['status'] == 'paused' and held['hold'] and held['checkpoint']
        assert proof['ready']['configuration']['model_configs'] == models
        health = batch.read(policy.parent / 'observer/health.json')
        assert health['opencode_quota_monitor'] == 'not_applicable' and health['quota'] is None
        assert health['other_provider_balance'] == 'unknown'
        driver.worker_options = {'max_requests': 80}
        _, completed, proof = driver.run(policy, entry, resume=True, week=1)
        assert completed['status'] == 'finished' and completed['day'] == 28
        assert proof['ready']['configuration']['model_configs'] == models
        _, state = validate_snapshot(completed['checkpoint'])
        assert state['configuration']['model_configs'] == models
        assert all(row['body']['model'] == models['agent']['model'] for row in proof['wire'])
        assert all(row['body']['stream'] is True and 'thinking' not in row['body'] for row in proof['wire'])
        assert all(proof['finished']['usage_matches_wire'].values())
        _, repeated, restored = driver.run(policy, entry, resume=True, week=2)
        assert repeated['status'] == 'finished' and repeated['day'] == 28
        assert restored['ready']['day'] == 7
        assert restored['ready']['configuration']['model_configs'] == models
        assert all(restored['finished']['usage_matches_wire'].values())
    finally:
        driver.close()
