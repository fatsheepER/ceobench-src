import json
from types import SimpleNamespace

import httpx
import pytest
from anthropic import Anthropic
from openai import OpenAI

from saas_bench.agents.bash_agent.agent import BashAgent, FinalText
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from saas_bench.config import BenchmarkConfig
from saas_bench.customer_llm import CustomerSimulator
from saas_bench.database import init_database
from saas_bench.sql_evidence import SQLEvidenceStore
from test_preflight_usage import reply
from test_sql_evidence import identity


@pytest.mark.parametrize('provider,api', [('deepseek', 'chat'), ('openai', 'responses'), ('anthropic', 'messages')])
@pytest.mark.parametrize('configured', [False, True])
def test_simulator_requests_capture_explicit_output_override(tmp_path, monkeypatch, provider, api, configured):
    if configured:
        monkeypatch.setenv('CEOBENCH_MODEL_MAX_OUTPUT_TOKENS', '2048')
    else:
        monkeypatch.delenv('CEOBENCH_MODEL_MAX_OUTPUT_TOKENS', raising=False)
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=reply(api))
    http = httpx.Client(transport=httpx.MockTransport(handle))
    client = (Anthropic if api == 'messages' else OpenAI)(api_key='offline', max_retries=0, http_client=http)
    conn = init_database(':memory:')
    customer = CustomerSimulator(None if api == 'messages' else client, conn, BenchmarkConfig())
    if api == 'messages':
        customer._anthropic_client = customer.usage_recorder.attach(client)
    monkeypatch.setenv('CEOBENCH_MODEL_MAX_OUTPUT_TOKENS', '1')
    try:
        assert customer.complete_text(provider=provider, model='test-model', user='same', max_tokens=17) == ('hello', 10, 2)
        key = 'max_output_tokens' if api == 'responses' else 'max_tokens'
        assert len(requests) == 1 and requests[0][key] == (2048 if configured else 17)
        assert customer.max_output_tokens == (2048 if configured else None)
    finally:
        client.close()
        conn.close()


@pytest.mark.parametrize('api', ['chat', 'native_chat', 'responses', 'messages'])
@pytest.mark.parametrize('configured', [False, True])
def test_actual_requests_use_captured_memory_and_output_limits(tmp_path, monkeypatch, api, configured):
    if configured:
        monkeypatch.setenv('CEOBENCH_MEMORY_MAX_CHARS', '128')
        monkeypatch.setenv('CEOBENCH_MODEL_MAX_OUTPUT_TOKENS', '2048')
    else:
        monkeypatch.delenv('CEOBENCH_MEMORY_MAX_CHARS', raising=False)
        monkeypatch.delenv('CEOBENCH_MODEL_MAX_OUTPUT_TOKENS', raising=False)
    memory = 'm' * 50000 + 'DEFAULT_MEMORY_VISIBLE'
    (tmp_path / 'MEMORY.md').write_text(memory)
    requests = []
    kind = 'chat' if api == 'native_chat' else api
    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        response = reply(kind)
        if api == 'messages' and body.get('stream'):
            events = [dict(type='message_start', message=response), dict(type='message_stop')]
            payload = ''.join('event: ' + event['type'] + '\ndata: ' + json.dumps(event) + '\n\n' for event in events)
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=httpx.ByteStream(payload.encode()))
        return httpx.Response(200, json=response)
    kwargs = dict(api_key='offline', max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    client = Anthropic(**kwargs) if api == 'messages' else OpenAI(**kwargs,
        base_url='https://api.deepseek.com' if api == 'chat' else 'https://api.openai.com/v1')
    agent = BashAgent(get_bash_agent_tool_descriptions(), client, model='test-model', system_prompt='instructions',
        reasoning_effort='high' if api == 'responses' else None, allow_final_text=True, workspace_path=tmp_path)
    monkeypatch.setenv('CEOBENCH_MEMORY_MAX_CHARS', '1')
    monkeypatch.setenv('CEOBENCH_MODEL_MAX_OUTPUT_TOKENS', '1')
    try:
        assert agent.act('dashboard', 0, False, {'day': 0}) == FinalText('hello')
        assert len(requests) == 1
        body = requests[0]
        key = 'max_output_tokens' if api == 'responses' else 'max_completion_tokens' if api == 'native_chat' else 'max_tokens'
        expected = 2048 if configured else 128000 if api == 'messages' else 32768
        assert body[key] == agent.max_output_tokens == expected
        system = body['instructions'] if api == 'responses' else body['system'][0]['text'] if api == 'messages' else body['messages'][0]['content']
        assert agent.memory_max_chars == (128 if configured else 100000)
        if configured:
            assert 'm' * 128 in system and 'MEMORY.md TRUNCATED' in system
            assert 'DEFAULT_MEMORY_VISIBLE' not in system
        else:
            assert memory in system and 'MEMORY.md TRUNCATED' not in system
    finally:
        client.close()


@pytest.mark.parametrize('limit', [17, 100000, 0])
def test_memory_origins_cover_exact_configured_range(tmp_path, limit):
    from test_preflight_context import agent
    store = SQLEvidenceStore(tmp_path / 'private.sqlite', identity(capture_scope='execution'))
    value = agent(tmp_path)
    value.memory_max_chars = limit
    value.usage_recorder = SimpleNamespace(evidence_store=store)
    memory = '中' * 50000
    (tmp_path / 'MEMORY.md').write_text(' \n' + memory + '\n ')
    prompt = value._get_system_prompt_with_memory()
    assert len(prompt.origins) == 1
    item = prompt.origins[0]
    expected = memory[:limit] if limit else memory
    source = store.get_content(item['version_id'])[1].decode()
    assert source[slice(*item['source_range'])] == prompt[slice(*item['request_range'])] == expected
    assert ('MEMORY.md TRUNCATED' in prompt) == (0 < limit < len(memory))
    assert (tmp_path / 'MEMORY.md').read_text() == ' \n' + memory + '\n '
    store.assert_healthy()
