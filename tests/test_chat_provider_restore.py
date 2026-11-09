"""Explicit provider changes through the public team checkpoint restore boundary."""
import json
import shlex

import pytest
from numpy.random import default_rng

from saas_bench import team_batch as batch
from saas_bench.config import BenchmarkConfig
from saas_bench.customer_llm import CustomerSimulator
from saas_bench.database import init_database
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.run_state import tree_hash
from saas_bench.simulation import Simulator
from saas_bench.team_models import make_chat_models
from saas_bench.tools import AgentTools
from test_chat_provider_switching import endpoint, isolated_environment


RATES = dict(input=.001, output=.002, cache_read=.0001, cache_write=.001)


@pytest.mark.parametrize('explicit_identity,top_pricing', [(False, 'registered'), (True, 'registered'), (False, None)])
def test_public_restore_switches_agent_and_simulator_contracts(tmp_path, monkeypatch, endpoint, explicit_identity, top_pricing):
    url, wire = endpoint
    old = make_chat_models('opendesign', simulator_provider='deepseek', agent_model='old-agent',
        simulator_model='old-simulator', agent_base_url=url + 'old-agent/', simulator_base_url=url + 'old-simulator/', environment={})
    new = make_chat_models('deepseek', simulator_provider='opendesign', agent_model='new-agent',
        simulator_model='new-simulator', agent_base_url=url + 'new-agent/', simulator_base_url=url + 'new-simulator/', environment={})
    old['agent'].update(max_output_tokens=1111, memory_characters=3333, credential_env='OLD_AGENT_FIXTURE_KEY')
    old['simulator'].update(max_output_tokens=555, credential_env='OLD_SIMULATOR_FIXTURE_KEY')
    new['agent'].update(max_output_tokens=2222, memory_characters=4444, credential_env='NEW_AGENT_FIXTURE_KEY')
    new['simulator'].update(max_output_tokens=666, credential_env='NEW_SIMULATOR_FIXTURE_KEY')
    for name in ('OLD_AGENT_FIXTURE_KEY', 'OLD_SIMULATOR_FIXTURE_KEY', 'NEW_AGENT_FIXTURE_KEY', 'NEW_SIMULATOR_FIXTURE_KEY'):
        monkeypatch.setenv(name, name.lower())
    pricing = dict(source='offline old provider price', basis='USD/1000 tokens',
        rates={name: RATES for name in ('old-agent', 'old-simulator', 'new-agent', 'new-simulator')})
    config = BenchmarkConfig(seed=123, agent_llm_provider='opendesign', agent_llm_model='old-agent',
        agent_llm_reasoning_effort='high', social_post_llm_provider='deepseek', social_post_llm_model='old-simulator',
        enterprise_llm_provider='deepseek', enterprise_llm_model='old-simulator', simulator_model_config=old['simulator'],
        model_pricing=pricing['rates'])
    conn = init_database(':memory:')
    clients = []
    def client_factory(models):
        def client(role):
            selected = models['simulator' if role == 'simulator' else 'agent']
            value = batch.checked_client(role, 'restore-fixture', model_config=selected)
            clients.append(value)
            return value
        return client
    old_client = client_factory(old)
    sim = Simulator(conn, config, default_rng(123), customer_simulator=CustomerSimulator(old_client('simulator'), conn, config))
    sim.initialize()
    tools = AgentTools(conn, 0, tmp_path / 'host', rng=sim.rng, config=config, seed=123)
    source = MultiAgentRuntime(tmp_path / 'source', tools, conn=conn, simulator=sim, mode='git',
        client_factory=old_client, model_configs=old, model='old-agent', reasoning_effort='high',
        model_pricing=pricing if top_pricing else None, public_dir=batch.ROOT / 'public')
    restored = None
    try:
        result = source.roles['growth'].executor.execute('bash', dict(command='./novamind-operation python-c ' + shlex.quote(
            "import novamind_api as n; n.marketing.set_daily_spend(operations=123); print('fixture saved operation')")))
        assert 'fixture saved operation' in result
        agent = source.roles['growth'].agent
        assert agent.act('source dashboard', 0, False, {'day': 0}).tool == 'read_file'
        agent.record_tool_result('source fixture result')
        customer = source.server.simulator.customer_simulator
        assert customer.complete_text(provider='deepseek', model='old-simulator', user='source simulator',
            system='fixture', max_tokens=555)[0] == 'fixture final'
        customer.usage_recorder.expected_model = 'old-simulator'
        operations = [tuple(row) for row in conn.execute('SELECT * FROM business_operations ORDER BY operation_seq')]
        snapshot = source.checkpoint()
        source.close()
        conn.close()
        original = tree_hash(snapshot, exact=True)
        monkeypatch.delenv('OLD_AGENT_FIXTURE_KEY')
        monkeypatch.delenv('OLD_SIMULATOR_FIXTURE_KEY')
        new_client = client_factory(new)
        def simulator_factory(restored_conn, restored_config, rng):
            client = batch.checked_client('simulator', 'restored-fixture', model_config=restored_config.simulator_model_config)
            clients.append(client)
            customer = CustomerSimulator(client, restored_conn, restored_config)
            return Simulator(restored_conn, restored_config, rng, customer_simulator=customer)
        options = dict(model_configs=new)
        if top_pricing is None:
            options['model_pricing'] = None
        if explicit_identity:
            options.update(model='new-agent', reasoning_effort='high')
        restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored', client_factory=new_client,
            simulator_factory=simulator_factory, public_dir=batch.ROOT / 'public', **options)
        effective = restored.server.tools.config
        assert effective.agent_llm_provider == 'deepseek' and effective.agent_llm_model == 'new-agent'
        assert effective.social_post_llm_provider == effective.enterprise_llm_provider == 'opendesign'
        assert effective.social_post_llm_model == effective.enterprise_llm_model == 'new-simulator'
        assert effective.simulator_model_config == new['simulator']
        assert effective.simulator_opendesign_base_url == url + 'new-simulator/'
        assert effective.model_pricing == {}
        if top_pricing is None:
            assert restored.configuration['model_pricing'] is None
        else:
            assert restored.configuration['model_pricing']['rates'] == {}
        assert restored.configuration['model'] == 'new-agent' and restored.configuration['model_configs'] == new
        agent = restored.roles['growth'].agent
        assert agent.max_output_tokens == 2222 and agent.memory_max_chars == 4444
        assert agent.act('continued fixture result', 0, False, {'day': 0}).tool == 'read_file'
        body = wire[-1]['body']
        assert wire[-1]['path'] == '/new-agent/chat/completions'
        assert body['model'] == 'new-agent' and body['max_tokens'] == 2222
        assert body['thinking'] == {'type': 'enabled'} and body['reasoning_effort'] == 'high'
        assert any(m['role'] == 'tool' and m['content'] == 'source fixture result' for m in body['messages'])
        assert restored.roles['growth'].usage.expected_model == 'new-agent'
        customer = restored.server.simulator.customer_simulator
        assert customer.max_output_tokens == 666 and customer.model == 'new-simulator'
        assert customer.complete_text(provider=effective.social_post_llm_provider, model=effective.social_post_llm_model,
            user='restored simulator', system='fixture', max_tokens=customer.max_output_tokens)[0] == 'fixture final'
        assert wire[-1]['path'] == '/new-simulator/chat/completions'
        assert wire[-1]['body']['model'] == 'new-simulator' and wire[-1]['body']['max_tokens'] == 666
        assert wire[-1]['body']['reasoning_effort'] == 'none' and 'thinking' not in wire[-1]['body']
        assert customer.usage_recorder.expected_model == 'new-simulator'
        for usage in (restored.roles['growth'].usage, customer.usage_recorder):
            entries = [json.loads(line) for line in usage.path.read_text().splitlines()]
            sent = [row for row in entries if row['event'] == 'http_request'][-1]
            assert sent['body']['model'].startswith('new-')
            last = [row for row in entries if row['event'] == 'response'][-1]
            assert last['error'] is None and last['cost_usd'] is None
        assert [tuple(row) for row in restored.server.conn.execute('SELECT * FROM business_operations ORDER BY operation_seq')] == operations
        assert tree_hash(snapshot, exact=True) == original
        changes = json.loads((restored.root / 'recovery.json').read_text())['configuration_overrides']
        assert {'model_configs', 'model'} <= changes.keys()
        if top_pricing is not None:
            assert 'model_pricing' in changes
        for role in restored.roles:
            assert changes[role + '.max_output_tokens']['new'] == 2222
    finally:
        if restored:
            restored.close()
            restored.server.conn.close()
        if not getattr(source, '_workers_stopping', False):
            source.close()
        try:
            conn.close()
        except Exception:
            pass
        for client in clients:
            client.close()
    assert all(client._client.is_closed for client in clients)


@pytest.mark.parametrize('case', ['identity', 'contract', 'conflicting_prices'])
def test_restore_rejects_invalid_model_override_before_files_or_clients(tmp_path, case):
    models = make_chat_models('deepseek', simulator_provider='opendesign', agent_model='same-alias',
        simulator_model='same-alias', environment={})
    options = dict(model_configs=models)
    if case == 'identity':
        options['model'] = 'contradictory-model'
        expected = 'Explicit model differs'
    elif case == 'contract':
        models['agent']['api_key'] = 'forbidden-fixture-value'
        expected = 'missing or unsupported fields'
    else:
        models['agent']['pricing']['rates'] = RATES
        models['simulator']['pricing']['rates'] = dict(RATES, output=.003)
        expected = 'Conflicting prices'
    called = []
    target = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError, match=expected):
        MultiAgentRuntime.restore(tmp_path / 'unused-snapshot', target,
            client_factory=lambda role: called.append(role), **options)
    assert not target.exists() and not called
