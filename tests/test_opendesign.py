"""Offline OpenDesign contracts through the SDK, runner, simulator and usage log."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.run_test import BashAgentRunner
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from saas_bench.config import BenchmarkConfig
from saas_bench.customer_llm import CustomerSimulator
from saas_bench.database import init_database
from saas_bench.model_usage import ModelUsage
from saas_bench.payload_tokens import load_counter, tokenizer_config
from saas_bench.server_entry import _apply_simulator_llm_config, _create_simulator_openai_client
from saas_bench.team_batch import checked_client
from saas_bench.team_models import model_issue, round_four_models, validate_models, validate_profile_models


MODEL = 'deepseek-v4.1-flash-eco'
SECRET = 'offline-opendesign-private-key'


def completion(*, tool=False, reasoning_field=None, usage=True):
    message = dict(role='assistant', content='' if tool else 'hello')
    if tool:
        message['tool_calls'] = [dict(id='read1', type='function',
            function=dict(name='read_file', arguments='{"path":"MEMORY.md"}'))]
    if reasoning_field:
        message[reasoning_field] = 'fixture reasoning to preserve'
    body = dict(id='chat1', object='chat.completion', created=1, model='served-opendesign-fixture',
        choices=[dict(index=0, finish_reason='tool_calls' if tool else 'stop', message=message)])
    if usage:
        body['usage'] = dict(prompt_tokens=10, completion_tokens=2,
            prompt_tokens_details=dict(cached_tokens=3), completion_tokens_details=dict(reasoning_tokens=1))
    return body


@pytest.fixture
def endpoint():
    requests, responses = [], []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append(dict(path=self.path, headers=dict(self.headers),
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
            payload = json.dumps(responses.pop(0) if responses else completion()).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    serving = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01))
    serving.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1/', requests, responses
    finally:
        server.shutdown()
        server.server_close()
        serving.join(3)
        assert not serving.is_alive()


def model_config(url, effort='high'):
    return dict(provider='opendesign', model=MODEL, api='chat', base_url=url,
        credential_env='OPENDESIGN_API_KEY', reasoning_effort=effort, thinking=None,
        max_output_tokens=1234, memory_characters=100000, temperature=1.0,
        tokenizer=tokenizer_config('opendesign', MODEL),
        pricing=dict(source=None, basis='Live model and rates unverified', rates=None),
        contract=dict(source='https://open-design.ai/help/cloud-api/', live_verified=False))


@pytest.mark.parametrize('effort,reasoning_field', [('high', 'reasoning_content'), ('high', 'reasoning'), ('none', None)])
def test_opendesign_agent_wire_and_restored_tool_history(tmp_path, monkeypatch, endpoint, effort, reasoning_field):
    url, requests, responses = endpoint
    responses.extend([completion(tool=True, reasoning_field=reasoning_field), completion(usage=False), completion(usage=False)])
    monkeypatch.setenv('OPENDESIGN_API_KEY', SECRET)
    config = model_config(url, effort)
    clients = []
    def new_agent(name):
        client = checked_client('ceo', 'offline-opendesign', model_config=config)
        clients.append(client)
        usage = ModelUsage(tmp_path / (name + '.jsonl'), 'ceo')
        value = BashAgent(get_bash_agent_tool_descriptions(), client, model=MODEL,
            system_prompt='fixture instructions', workspace_path=tmp_path, allow_final_text=True,
            reasoning_effort=effort, model_config=config, max_output_tokens=1234, usage_recorder=usage)
        value._snapshot_path = tmp_path / 'conversation.json'
        return value
    try:
        first = new_agent('first')
        assert first.act('fixture dashboard', 0, False, {'day': 0}).tool == 'read_file'
        body = requests[0]['body']
        assert requests[0]['path'] == '/v1/chat/completions'
        assert body.get('max_tokens') == 1234
        assert body['reasoning_effort'] == effort
        assert not {'max_completion_tokens', 'thinking', 'stream', 'stream_options'} & body.keys()
        assert requests[0]['headers']['Authorization'] == 'Bearer ' + SECRET
        assert not any(name.lower().startswith('x-opencode') for name in requests[0]['headers'])
        first.record_tool_result('fixture tool result')
        first._save_conversation_snapshot(strict=True)
        restored = new_agent('restored')
        assert restored.load_conversation_snapshot(restored._snapshot_path, required=True)
        assert first.act('fixture tool result', 0, False, {'day': 0}).text == 'hello'
        assert restored.act('fixture tool result', 0, False, {'day': 0}).text == 'hello'
        assert requests[1]['body'] == requests[2]['body']
        assistant = next(m for m in requests[2]['body']['messages'] if m['role'] == 'assistant')
        if effort == 'high':
            assert assistant['reasoning_content'] == 'fixture reasoning to preserve'
        else:
            assert 'reasoning_content' not in assistant
        assert any(m['role'] == 'tool' and m['content'] == 'fixture tool result'
            and m['tool_call_id'] == 'read1' for m in requests[2]['body']['messages'])
        log = first.usage_recorder.path.read_text()
        entries = [json.loads(line) for line in log.splitlines()]
        assert SECRET not in log
        response = next(row for row in entries if row['event'] == 'response')
        assert response['response']['model'] == 'served-opendesign-fixture'
        assert response['usage']['input_tokens'] == 10 and response['usage']['cached_tokens'] == 3
        assert response['cost_usd'] is None
        assert first.usage_recorder.summary['missing_cost'] == 2
        assert first.usage_recorder.summary['missing']['input_tokens'] == 1
        assert first.usage_recorder.summary['known']['input_tokens'] == 10
        assert model_issue(response, dict(agent=config, simulator=config)) == 'served_model_changed'
    finally:
        for client in clients:
            client.close()
    assert all(client._client.is_closed for client in clients)


@pytest.fixture
def clean_provider_env(monkeypatch):
    monkeypatch.setattr('saas_bench.agents.bash_agent.run_test.load_env_file', lambda _: {})
    for name in ('OPENDESIGN_API_KEY', 'OPENDESIGN_BASE_URL', 'CEOBENCH_SIMULATOR_LLM_PROVIDER',
                 'CEOBENCH_SIMULATOR_LLM_MODEL', 'CEOBENCH_DASHBOARD_URL'):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize('custom_endpoint', [False, True])
def test_runner_selects_both_opendesign_roles(tmp_path, monkeypatch, endpoint, clean_provider_env, custom_endpoint):
    url, requests, responses = endpoint
    monkeypatch.setenv('OPENDESIGN_API_KEY', SECRET)
    runner = BashAgentRunner(provider='opendesign', model=MODEL, reasoning_effort='high',
        workspace_base=tmp_path, base_url=url if custom_endpoint else None)
    client = None
    try:
        expected = url if custom_endpoint else 'https://amr-link.open-design.ai/v1/'
        assert str(runner.client.base_url) == expected
        env = runner._server_environment()
        assert env['CEOBENCH_SIMULATOR_LLM_PROVIDER'] == 'opendesign'
        assert env['CEOBENCH_SIMULATOR_LLM_MODEL'] == MODEL
        for name in ('CEOBENCH_SIMULATOR_LLM_PROVIDER', 'CEOBENCH_SIMULATOR_LLM_MODEL', 'OPENDESIGN_BASE_URL'):
            monkeypatch.setenv(name, env[name])
        config = BenchmarkConfig()
        _apply_simulator_llm_config(config)
        client = _create_simulator_openai_client(config)
        assert client.api_key == runner.client.api_key == SECRET
        assert str(client.base_url) == expected
        assert not any(name.lower().startswith('x-opencode') for name in client.default_headers)
        assert config.simulator_opendesign_base_url == expected
        if custom_endpoint:
            runner.setup()
            manifest = json.loads((runner.workspace_dir / 'manifest.json').read_text())
            assert manifest['benchmark_config']['simulator_opendesign_base_url'] == url
            responses.append(completion(tool=True, reasoning_field='reasoning_content'))
            assert runner.agent.act('fixture dashboard', 0, False, {'day': 0}).tool == 'read_file'
            assert runner.agent.supports_responses_api is False
            assert requests[-1]['body']['reasoning_effort'] == 'high'
            assert 'max_tokens' in requests[-1]['body']
            sim = CustomerSimulator(client, init_database(':memory:'), config)
            assert sim.complete_text(provider='opendesign', model=MODEL, user='fixture user',
                system='fixture system', max_tokens=321) == ('hello', 10, 2)
            body = requests[-1]['body']
            assert requests[-1]['path'] == '/v1/chat/completions'
            assert body['model'] == MODEL and body['max_tokens'] == 321
            assert body['reasoning_effort'] == 'none' and 'thinking' not in body
            assert sim.usage_recorder.summary['missing_cost'] == 1
            sim.conn.close()
        config.enterprise_llm_provider = 'openai'
        with pytest.raises(ValueError, match='one endpoint'):
            _create_simulator_openai_client(config)
    finally:
        runner._stop_server()
        runner.client.close()
        if client:
            client.close()


def test_missing_opendesign_credentials_never_fall_back(tmp_path, monkeypatch, clean_provider_env):
    monkeypatch.setenv('OPENAI_API_KEY', 'unrelated-openai-key')
    monkeypatch.setenv('OPENCODE_API_KEY', 'unrelated-opencode-key')
    with pytest.raises(ValueError, match='No API key found for provider opendesign'):
        BashAgentRunner(provider='opendesign', model=MODEL, workspace_base=tmp_path)
    config = BenchmarkConfig(social_post_llm_provider='opendesign', enterprise_llm_provider='opendesign')
    with pytest.raises(ValueError, match='OPENDESIGN_API_KEY'):
        _apply_simulator_llm_config(config)
    with pytest.raises(ValueError, match='OPENDESIGN_API_KEY'):
        _create_simulator_openai_client(config)
    from saas_bench.run_lifecycle import RunCancelled
    with pytest.raises(RunCancelled, match='OPENDESIGN_API_KEY'):
        checked_client('ceo', 'offline-opendesign', model_config=model_config('https://amr-link.open-design.ai/v1/'))


def test_explicit_contract_audit_and_frozen_profiles():
    config = model_config('https://amr-link.open-design.ai/v1/')
    models = dict(agent=config, simulator=dict(config, reasoning_effort='none'))
    assert validate_models(models) is models
    for role in ('ceo', 'simulator'):
        body = dict(model=MODEL, max_tokens=1234, reasoning_effort='none' if role == 'simulator' else 'high')
        row = dict(event='http_request', role=role, body=body,
            endpoint=config['base_url'] + 'chat/completions')
        assert model_issue(row, models) is None
        assert model_issue(dict(row, body=dict(body, thinking={'type': 'enabled'})), models)
        assert model_issue(dict(row, body=dict(body, reasoning_effort='low')), models)
    invalid = copy.deepcopy(models)
    invalid['agent']['thinking'] = 'enabled'
    with pytest.raises(ValueError, match='OpenDesign'):
        validate_models(invalid)
    invalid = copy.deepcopy(models)
    invalid['agent']['reasoning_effort'] = 'max'
    with pytest.raises(ValueError, match='OpenDesign'):
        validate_models(invalid)
    pricing = dict(source='fixture', basis='fixture', rates={})
    for profile in ('r4-d56', 'r4-flash-max-pf', 'r4-sol-high-pf', 'r5-d56'):
        frozen = round_four_models(profile, pricing)
        validate_profile_models(profile, frozen, pricing)
        with pytest.raises(ValueError, match='differs'):
            validate_profile_models(profile, models, pricing)
    assert tokenizer_config('opendesign', MODEL)['status'] == 'unavailable'
    assert load_counter('opendesign', MODEL) is None


@pytest.mark.parametrize('explicit_override', [False, True])
def test_public_checkpoint_preserves_independent_simulator_endpoint(
        tmp_path, monkeypatch, endpoint, clean_provider_env, explicit_override):
    from saas_bench.run_state import checkpoint_directory
    url, requests, responses = endpoint
    simulator_url = url + 'simulator/'
    monkeypatch.setenv('OPENDESIGN_API_KEY', SECRET)
    monkeypatch.setenv('OPENDESIGN_BASE_URL', simulator_url)
    source = BashAgentRunner(provider='opendesign', model=MODEL, reasoning_effort='high',
        base_url=url, workspace_base=tmp_path / 'source')
    resumed = None
    try:
        source.setup()
        responses.append(completion(tool=True, reasoning_field='reasoning_content'))
        source.agent.act('fixture dashboard', 0, False, {'day': 0})
        source.agent.record_tool_result('completed fixture tool')
        source._save_checkpoint(0)
        source._stop_server()
        original = (source.workspace_dir / 'manifest.json').read_bytes()
        if explicit_override:
            expected = url + 'override/'
            monkeypatch.setenv('OPENDESIGN_BASE_URL', expected)
        else:
            expected = simulator_url
            monkeypatch.delenv('OPENDESIGN_BASE_URL')
        resumed = BashAgentRunner(continue_from=source.workspace_dir, workspace_base=tmp_path / 'recovered')
        resumed.setup()
        resumed._restore_from_checkpoint(resumed._load_checkpoint())
        assert str(resumed.client.base_url) == url
        manifest = json.loads((resumed.workspace_dir / 'manifest.json').read_text())
        assert manifest['benchmark_config']['simulator_opendesign_base_url'] == expected
        resumed._save_checkpoint(0)
        directory = checkpoint_directory(resumed.workspace_dir, resumed._load_checkpoint())
        session = json.loads((directory / 'session.json').read_text())
        assert session['benchmark_config']['simulator_opendesign_base_url'] == expected
        assert (source.workspace_dir / 'manifest.json').read_bytes() == original
        assert len(requests) == 1
    finally:
        for runner in (source, resumed):
            if runner:
                runner._stop_server()
                runner.client.close()


def test_direct_team_all_roles_use_explicit_chat_config(tmp_path, monkeypatch, endpoint, make_initialized_sim, make_agent_tools):
    from saas_bench.multi_agent import MultiAgentRuntime
    url, requests, responses = endpoint
    config = model_config(url)
    models = dict(agent=config, simulator=dict(config, reasoning_effort='none'))
    monkeypatch.setenv('OPENDESIGN_API_KEY', SECRET)
    conn, simulator, world_config = make_initialized_sim()
    world_config.simulator_model_config = models['simulator']
    tools = make_agent_tools(conn, world_config)
    clients = []
    def client_factory(role):
        client = checked_client(role, 'offline-team', model_config=config)
        clients.append(client)
        return client
    runtime = None
    try:
        runtime = MultiAgentRuntime(tmp_path / 'team', tools, conn=conn, simulator=simulator,
            client_factory=client_factory, mode='git', model=MODEL, reasoning_effort='high', model_configs=models)
        for role, value in runtime.roles.items():
            responses.append(completion(tool=True, reasoning_field='reasoning_content'))
            assert value.agent.act('fixture task for ' + role, 0, False, {'day': 0}).tool == 'read_file'
            assert not value.agent.supports_responses_api
            entries = [json.loads(line) for line in value.usage.path.read_text().splitlines()]
            sent = next(row for row in entries if row['event'] == 'http_request')
            assert sent['body'] == requests[-1]['body']
            assert sent['body']['reasoning_effort'] == 'high'
            assert sent['body']['max_tokens'] == 1234
            assert 'thinking' not in sent['body'] and 'stream' not in sent['body']
            assert requests[-1]['path'] == '/v1/chat/completions'
            assert SECRET not in value.usage.path.read_text()
            assert value.usage.summary['known']['cached_tokens'] == 3
            assert value.usage.summary['known_cost_usd'] is None
        assert len(requests) == 3
    finally:
        if runtime:
            runtime.close()
        for client in clients:
            client.close()
        conn.close()
