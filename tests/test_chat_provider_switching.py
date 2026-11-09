"""Offline provider selection through the runner, public server and checkpoints."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from saas_bench.agents.bash_agent.run_test import BashAgentRunner
from saas_bench.config import BenchmarkConfig
from saas_bench.server_entry import _apply_simulator_llm_config, _create_simulator_openai_client


PROVIDERS = {
    'deepseek': ('deepseek-flash', 'https://api.deepseek.com/', 'DEEPSEEK_API_KEY'),
    'opencode': ('deepseek-v4.1-flash', 'https://opencode.ai/inference/go/openai/v1/', 'OPENCODE_API_KEY'),
    'opendesign': ('deepseek-v4.1-flash-eco', 'https://amr-link.open-design.ai/v1/', 'OPENDESIGN_API_KEY'),
}


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr('saas_bench.agents.bash_agent.run_test.load_env_file', lambda _: {})
    monkeypatch.setattr('saas_bench.payload_tokens.load_counter', lambda *_: None)
    for name in ('CEOBENCH_SIMULATOR_LLM_PROVIDER', 'CEOBENCH_SIMULATOR_LLM_MODEL',
                 'CEOBENCH_SIMULATOR_LLM_BASE_URL', 'CEOBENCH_DASHBOARD_URL',
                 'DEEPSEEK_BASE_URL', 'OPENCODE_BASE_URL', 'OPENDESIGN_BASE_URL'):
        monkeypatch.delenv(name, raising=False)
    for provider, (_, _, key) in PROVIDERS.items():
        monkeypatch.setenv(key, 'offline-' + provider)


@pytest.mark.parametrize('provider', PROVIDERS)
def test_provider_alone_selects_own_defaults(tmp_path, provider):
    model, url, _ = PROVIDERS[provider]
    runner = BashAgentRunner(provider=provider, workspace_base=tmp_path)
    try:
        assert runner.model == model
        assert runner.base_url == url
        assert runner.reasoning_effort == 'high'
        assert runner.client.api_key == 'offline-' + provider
        assert ('x-opencode-session' in runner.client.default_headers) == (provider == 'opencode')
    finally:
        runner.client.close()


@pytest.mark.parametrize('provider', ['deepseek', 'opencode'])
def test_provider_only_resume_discards_saved_endpoint_and_model(tmp_path, monkeypatch, provider):
    source = BashAgentRunner(provider='opendesign', model='saved-custom-model',
        base_url='http://127.0.0.1:9/saved/', workspace_base=tmp_path / 'source')
    resumed = None
    try:
        source.setup()
        source._save_checkpoint(0)
        source._stop_server()
        original = (source.workspace_dir / 'manifest.json').read_bytes()
        resumed = BashAgentRunner(continue_from=source.workspace_dir, provider=provider,
            workspace_base=tmp_path / 'resumed')
        assert resumed.model == PROVIDERS[provider][0]
        assert resumed.base_url == PROVIDERS[provider][1]
        assert resumed.reasoning_effort == 'high'
        assert resumed.client.api_key == 'offline-' + provider
        assert (source.workspace_dir / 'manifest.json').read_bytes() == original
        changed = {row['field'] for row in resumed.configuration_overrides}
        assert {'provider', 'model', 'base_url'} <= changed
    finally:
        for runner in (source, resumed):
            if runner:
                runner._stop_server()
                runner.client.close()


@pytest.mark.parametrize('provider', PROVIDERS)
def test_simulator_provider_only_uses_its_model(tmp_path, monkeypatch, provider):
    monkeypatch.setenv('CEOBENCH_SIMULATOR_LLM_PROVIDER', provider)
    config = BenchmarkConfig(social_post_llm_provider='anthropic', social_post_llm_model='old-social-model',
        enterprise_llm_provider='anthropic', enterprise_llm_model='old-enterprise-model')
    _apply_simulator_llm_config(config)
    assert config.social_post_llm_model == config.enterprise_llm_model == PROVIDERS[provider][0]
    client = _create_simulator_openai_client(config)
    try:
        assert str(client.base_url) == PROVIDERS[provider][1]
        assert client.api_key == 'offline-' + provider
    finally:
        client.close()


@pytest.fixture
def endpoint():
    wire = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            wire.append(dict(path=self.path, body=body,
                opencode_header='x-opencode-session' in self.headers))
            tools = bool(body.get('tools'))
            message = dict(role='assistant', content=None if tools else 'fixture final')
            if body.get('reasoning_effort') == 'high':
                message['reasoning_content'] = 'fixture reasoning'
            if tools:
                message['tool_calls'] = [dict(id='fixture-call', type='function',
                    function=dict(name='read_file', arguments='{"path":"notes.md"}'))]
            finish = 'tool_calls' if tools else 'stop'
            usage = dict(prompt_tokens=10, completion_tokens=2, total_tokens=12,
                prompt_tokens_details=dict(cached_tokens=0), completion_tokens_details=dict(reasoning_tokens=1))
            if body.get('stream'):
                if tools:
                    message['tool_calls'][0]['index'] = 0
                chunks = [dict(id='fixture', object='chat.completion.chunk', created=1, model=body['model'],
                    choices=[dict(index=0, delta=message, finish_reason=None)]),
                    dict(id='fixture', object='chat.completion.chunk', created=1, model=body['model'],
                        choices=[dict(index=0, delta={}, finish_reason=finish)]),
                    dict(id='fixture', object='chat.completion.chunk', created=1, model=body['model'], choices=[], usage=usage)]
                payload = ''.join('data: ' + json.dumps(chunk) + '\n\n' for chunk in chunks).encode() + b'data: [DONE]\n\n'
                content_type = 'text/event-stream'
            else:
                payload = json.dumps(dict(id='fixture', object='chat.completion', created=1, model=body['model'],
                    choices=[dict(index=0, message=message, finish_reason=finish)], usage=usage)).encode()
                content_type = 'application/json'
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    serving = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01))
    serving.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/', wire
    finally:
        server.shutdown()
        server.server_close()
        serving.join(3)
        assert not serving.is_alive()


@pytest.mark.parametrize('provider,simulator_provider', [('deepseek', 'opendesign'), ('opencode', 'deepseek'), ('opendesign', 'opencode')])
def test_public_runner_wire_and_checkpoint_keep_independent_choices(
        tmp_path, monkeypatch, endpoint, provider, simulator_provider):
    from saas_bench.customer_llm import CustomerSimulator
    from saas_bench.database import init_database
    from saas_bench.run_state import checkpoint_directory
    url, wire = endpoint
    monkeypatch.setenv('CEOBENCH_SIMULATOR_LLM_PROVIDER', simulator_provider)
    monkeypatch.setenv('CEOBENCH_SIMULATOR_LLM_BASE_URL', url + 'simulator/')
    source = BashAgentRunner(provider=provider, model='selected-' + provider,
        base_url=url + 'agent/', workspace_base=tmp_path / 'source')
    resumed = None
    simulator = None
    try:
        source.setup()
        assert source.agent.act('fixture dashboard', 0, False, {'day': 0}).tool == 'read_file'
        assert source.agent.supports_responses_api is False
        source.agent.record_tool_result('fixture tool result')
        source._save_checkpoint(0)
        source._stop_server()
        manifest = json.loads((source.workspace_dir / 'manifest.json').read_text())
        source_config = manifest['benchmark_config']
        assert source_config['social_post_llm_provider'] == source_config['enterprise_llm_provider'] == simulator_provider
        assert source_config['social_post_llm_model'] == source_config['enterprise_llm_model'] == PROVIDERS[simulator_provider][0]
        assert source_config['simulator_' + simulator_provider + '_base_url'] == url + 'simulator/'
        original = (source.workspace_dir / 'manifest.json').read_bytes()
        monkeypatch.delenv('CEOBENCH_SIMULATOR_LLM_PROVIDER')
        monkeypatch.delenv('CEOBENCH_SIMULATOR_LLM_BASE_URL')
        resumed = BashAgentRunner(continue_from=source.workspace_dir, workspace_base=tmp_path / 'resumed')
        assert resumed.model == 'selected-' + provider and resumed.base_url == url + 'agent/'
        resumed.setup()
        resumed._restore_from_checkpoint(resumed._load_checkpoint())
        assert resumed.agent.act('fixture tool result', 0, False, {'day': 0}).tool == 'read_file'
        body = wire[-1]['body']
        assert wire[-1]['path'] == '/agent/chat/completions'
        assert body['model'] == 'selected-' + provider and body['max_tokens'] == 32768
        assert body['reasoning_effort'] == 'high'
        assert ('thinking' in body) == (provider != 'opendesign')
        assert wire[-1]['opencode_header'] == (provider == 'opencode')
        assert any(m['role'] == 'tool' and m['content'] == 'fixture tool result' for m in body['messages'])
        assert any(m.get('reasoning_content') == 'fixture reasoning' for m in body['messages'])
        resumed.agent.record_tool_result('completed follow-up fixture tool')
        resumed._save_checkpoint(0)
        directory = checkpoint_directory(resumed.workspace_dir, resumed._load_checkpoint())
        session = json.loads((directory / 'session.json').read_text())
        config = BenchmarkConfig(**session['benchmark_config'])
        assert config.social_post_llm_provider == simulator_provider
        assert getattr(config, 'simulator_' + simulator_provider + '_base_url') == url + 'simulator/'
        monkeypatch.setenv('CEOBENCH_SIMULATOR_USAGE_LOG', str(tmp_path / 'simulator.jsonl'))
        simulator = CustomerSimulator(_create_simulator_openai_client(config), init_database(':memory:'), config)
        assert simulator.complete_text(provider=simulator_provider, model=config.social_post_llm_model,
            user='fixture', system='fixture', max_tokens=321) == ('fixture final', 10, 2)
        assert wire[-1]['path'] == '/simulator/chat/completions'
        assert wire[-1]['body']['model'] == PROVIDERS[simulator_provider][0]
        assert wire[-1]['body']['max_tokens'] == 321
        assert wire[-1]['body'].get('reasoning_effort', 'none') == 'none'
        assert ('thinking' in wire[-1]['body']) == (simulator_provider != 'opendesign')
        for usage in (resumed.agent.usage_recorder, simulator.usage_recorder):
            entries = [json.loads(line) for line in usage.path.read_text().splitlines()]
            assert any(row['event'] == 'request' for row in entries)
            assert any(row['event'] == 'response' and row.get('error') is None for row in entries)
            assert usage.summary['known']['input_tokens'] >= 10
            assert usage.summary['missing_cost'] >= 1
        assert (source.workspace_dir / 'manifest.json').read_bytes() == original
    finally:
        if simulator:
            simulator.client.close()
            simulator.conn.close()
        for runner in (source, resumed):
            if runner:
                runner._stop_server()
                runner.client.close()


@pytest.mark.parametrize('provider', PROVIDERS)
def test_each_provider_uses_only_its_credential(tmp_path, monkeypatch, provider):
    key = PROVIDERS[provider][2]
    monkeypatch.delenv(key)
    monkeypatch.setenv('OPENAI_API_KEY', 'unrelated-fixture')
    with pytest.raises(ValueError, match='No API key found for provider ' + provider):
        BashAgentRunner(provider=provider, workspace_base=tmp_path)
    config = BenchmarkConfig(social_post_llm_provider=provider, enterprise_llm_provider=provider)
    with pytest.raises((ValueError, SystemExit)):
        _apply_simulator_llm_config(config)
    with pytest.raises(ValueError, match=key):
        _create_simulator_openai_client(config)
    explicit = BashAgentRunner(provider=provider, api_key='explicit-fixture', workspace_base=tmp_path)
    explicit.client.close()


@pytest.mark.parametrize('provider', PROVIDERS)
def test_endpoint_overrides_are_provider_scoped(tmp_path, monkeypatch, provider):
    monkeypatch.setenv('DEEPSEEK_BASE_URL', 'http://127.0.0.1:11/deepseek/')
    monkeypatch.setenv('OPENCODE_BASE_URL', 'http://127.0.0.1:12/opencode/')
    monkeypatch.setenv('OPENDESIGN_BASE_URL', 'http://127.0.0.1:13/opendesign/')
    import os
    expected = os.environ[provider.upper() + '_BASE_URL']
    runner = BashAgentRunner(provider=provider, workspace_base=tmp_path / 'environment')
    explicit = BashAgentRunner(provider=provider, base_url='http://127.0.0.1:14/explicit/',
        workspace_base=tmp_path / 'explicit')
    try:
        assert runner.base_url == expected
        assert explicit.base_url == 'http://127.0.0.1:14/explicit/'
        assert runner._server_environment()[provider.upper() + '_BASE_URL'] == expected
    finally:
        runner.client.close()
        explicit.client.close()


def test_same_provider_explicit_simulator_selection_keeps_exact_model(monkeypatch):
    monkeypatch.setenv('CEOBENCH_SIMULATOR_LLM_PROVIDER', 'opendesign')
    config = BenchmarkConfig(social_post_llm_provider='opendesign', social_post_llm_model='custom-social',
        enterprise_llm_provider='opendesign', enterprise_llm_model='custom-enterprise')
    _apply_simulator_llm_config(config)
    assert config.social_post_llm_model == 'custom-social' and config.enterprise_llm_model == 'custom-enterprise'
    monkeypatch.setenv('CEOBENCH_SIMULATOR_LLM_MODEL', 'explicit-simulator-model')
    _apply_simulator_llm_config(config)
    assert config.social_post_llm_model == config.enterprise_llm_model == 'explicit-simulator-model'


@pytest.mark.parametrize('explicit_override', [False, True])
def test_old_standalone_simulator_endpoint_is_preserved(tmp_path, monkeypatch, explicit_override):
    from dataclasses import asdict
    from saas_bench.server_entry import _session_config
    old = asdict(BenchmarkConfig(social_post_llm_provider='opencode', enterprise_llm_provider='opencode',
        social_post_llm_model='saved-social-model', enterprise_llm_model='saved-enterprise-model'))
    old.pop('simulator_opencode_base_url')
    if explicit_override:
        monkeypatch.setenv('CEOBENCH_SIMULATOR_LLM_BASE_URL', 'http://127.0.0.1:9/explicit-simulator/')
    config = _session_config(42, 500, 1000000, dict(benchmark_config=old))
    expected = 'http://127.0.0.1:9/explicit-simulator/' if explicit_override else 'https://opencode.ai/zen/go/v1/'
    assert config.simulator_opencode_base_url == expected
    assert config.social_post_llm_model == 'saved-social-model' and config.enterprise_llm_model == 'saved-enterprise-model'
    client = _create_simulator_openai_client(config)
    try:
        assert str(client.base_url) == expected
    finally:
        client.close()
    if not explicit_override:
        monkeypatch.setenv('CEOBENCH_SIMULATOR_LLM_PROVIDER', 'opencode')
        fresh = _session_config(42, 500, 1000000)
        assert fresh.simulator_opencode_base_url == PROVIDERS['opencode'][1]
