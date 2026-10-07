import json
from copy import deepcopy

import pytest

from saas_bench.config import BenchmarkConfig, ScenarioPack, normalize_runtime_config
from saas_bench.shocks import ShockManager


def runtime_values(customer_id=123):
    return {
        'targeted_ad_spend': {'social_media': {'S1': 7.0}},
        'targeted_ops_spend': {'S1': 8.0},
        'targeted_ops_spend_by_plan': {'A': 9.0},
        'targeted_ops_spend_by_group_plan': {'S1': {'A': 10.0}},
        'targeted_ops_spend_by_customer': {customer_id: 11.0},
        'targeted_dev_spend': {'S1': 12.0},
        'ads_strength_global': .1,
        'ads_strength_by_group': {'S1': .2},
        'ads_strength_by_customer': {customer_id: .3},
        'lead_promotion_global': 1.0,
        'lead_promotion_by_group': {'S1': 2.0},
        'lead_promotion_by_channel': {'social_media': 3.0},
        'lead_promotion_by_channel_group': {'social_media': {'S1': 4.0}},
        'promotion_global': 5.0,
        'promotion_by_group': {'S1': 6.0},
        'promotion_by_customer': {customer_id: 7.0},
        'promotion_by_group_plan': {'S1': {'A': 8.0}},
    }


def test_runtime_config_restores_all_fields_in_shared_object(make_initialized_sim, make_agent_tools):
    conn, sim, config = make_initialized_sim()
    tools = make_agent_tools(conn, config)
    expected = runtime_values()
    for name, value in expected.items():
        setattr(config, name, deepcopy(value))
    sim.save_rng_states()
    for name, value in expected.items():
        setattr(config, name, {} if isinstance(value, dict) else 0.0)
    assert sim.restore_rng_states()
    assert sim.config is config is tools.config
    assert {name: getattr(config, name) for name in expected} == expected
    assert sim._get_effective_promotion(123, 'S1', 'A') == 26
    assert sim._get_lead_promotion('S1', 'social_media') == 10


@pytest.mark.parametrize('field,bad', [
    ('targeted_ad_spend', {'social_media': []}),
    ('targeted_ops_spend', []),
    ('targeted_ops_spend_by_plan', {1: 9}),
    ('targeted_ops_spend_by_group_plan', {'S1': {'A': True}}),
    ('targeted_ops_spend_by_customer', {'not-an-id': 11}),
    ('targeted_dev_spend', {'S1': -1}),
    ('ads_strength_global', 1.01),
    ('ads_strength_by_group', {'S1': float('nan')}),
    ('ads_strength_by_customer', {'123': float('inf')}),
    ('lead_promotion_global', '1'),
    ('promotion_by_customer', {'123': None}),
])
def test_runtime_config_rejects_invalid_shapes_and_values(field, bad):
    values = runtime_values()
    values[field] = bad
    with pytest.raises(ValueError, match='[Rr]untime'):
        normalize_runtime_config(values)


@pytest.mark.parametrize('invalid', ['legacy', 'missing', 'extra', 'bad_value'])
def test_bad_runtime_checkpoint_rejects_before_mutating_rng_or_config(make_initialized_sim, invalid):
    conn, sim, config = make_initialized_sim()
    sim.save_rng_states()
    states = json.loads(conn.execute("SELECT state_json FROM _rng_states WHERE name='all'").fetchone()[0])
    if invalid == 'legacy':
        states['version'] = 2
        del states['runtime_config']
    elif invalid == 'missing':
        del states['runtime_config']['targeted_ad_spend']
    elif invalid == 'extra':
        states['runtime_config']['unknown'] = {}
    else:
        states['runtime_config']['promotion_global'] = -1
    conn.execute("UPDATE _rng_states SET state_json=?", (json.dumps(states),))
    sim.rng.random(9)
    before_rng = deepcopy(sim.rng.bit_generator.state)
    config.promotion_global = 99
    with pytest.raises(ValueError):
        sim.restore_rng_states()
    assert sim.rng.bit_generator.state == before_rng
    assert config.promotion_global == 99


def test_frozen_config_normalizes_customer_keys_without_changing_manifest(tmp_path, monkeypatch):
    from dataclasses import asdict
    from saas_bench.server_entry import _session_config

    config = BenchmarkConfig()
    config.social_post_llm_provider = config.enterprise_llm_provider = 'deepseek'
    config.social_post_llm_model = config.enterprise_llm_model = 'test-model'
    for name, value in runtime_values().items():
        setattr(config, name, value)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'benchmark_config': asdict(config)}))
    before = manifest.read_bytes()
    monkeypatch.setenv('CEOBENCH_RUN_MANIFEST', str(manifest))
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'offline-only')
    monkeypatch.setenv('CEOBENCH_SIMULATOR_LLM_PROVIDER', 'deepseek')
    monkeypatch.setenv('CEOBENCH_SIMULATOR_LLM_MODEL', 'test-model')
    restored = _session_config(42, 7, 100)
    assert {name: getattr(restored, name) for name in runtime_values()} == runtime_values()
    assert manifest.read_bytes() == before


def test_every_stream_and_process_state_resume(make_initialized_sim):
    conn, sim, _ = make_initialized_sim()
    sim.shock_manager = ShockManager(conn, sim.rng, ScenarioPack(name='test', description='test'))
    streams = [sim.rng, sim._macro_rng, sim._competitor_rng, sim._competitor_post_noise_rng,
               sim._competitor_template_rng, sim._quality_rng, sim._customer_quality_noise_rng,
               sim._customer_pick_rng, sim.shock_manager.rng, *sim._group_rngs.values()]
    for rng in streams:
        rng.random(11)
    sim.current_day = 35
    sim.shutdown_mode = True
    sim._customer_quality_noise[123] = 0.91
    sim.save_rng_states()
    expected = [rng.random(7).tolist() for rng in streams]
    expected_template = sim._generate_competitor_post_template('competitor', 'minor')
    sim.current_day = 0
    sim.shutdown_mode = False
    assert sim.restore_rng_states()
    assert sim.current_day == 35 and sim.shutdown_mode
    assert sim._customer_quality_noise[123] == 0.91
    assert [rng.random(7).tolist() for rng in streams] == expected
    assert sim._generate_competitor_post_template('competitor', 'minor') == expected_template
    states = json.loads(conn.execute("SELECT state_json FROM _rng_states WHERE name='all'").fetchone()[0])
    del states['_competitor_post_noise_rng']
    conn.execute("UPDATE _rng_states SET state_json=?", (json.dumps(states),))
    with pytest.raises(ValueError, match='Incomplete'):
        sim.restore_rng_states()
