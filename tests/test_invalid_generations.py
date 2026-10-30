import json

import httpx
import pytest
from anthropic import Anthropic
from openai import OpenAI

from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from saas_bench.model_usage import ModelUsage
from saas_bench.run_lifecycle import RunCancelled
from test_preflight_usage import reply


@pytest.mark.parametrize('api', ['chat', 'responses', 'messages'])
@pytest.mark.parametrize('reason', ['no_tool', 'arguments', 'output_truncation', 'provider_refusal'])
def test_invalid_generation_limit_records_output_reason_and_feedback(tmp_path, monkeypatch, api, reason):
    monkeypatch.setenv('CEOBENCH_INVALID_GENERATION_ATTEMPTS', '1')
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        body = reply(api)
        if api == 'chat':
            message = body['choices'][0]['message']
            message['content'] = 'bad-output'
            if reason == 'arguments':
                message['tool_calls'] = [dict(id='bad', type='function', function=dict(name='read_file', arguments='[1]'))]
            elif reason == 'output_truncation':
                body['choices'][0]['finish_reason'] = 'length'
            elif reason == 'provider_refusal':
                message['refusal'] = 'provider refused bad-output'
        elif api == 'responses':
            body['output'][0]['content'][0]['text'] = 'bad-output'
            if reason == 'arguments':
                body['output'].append(dict(type='function_call', id='bad', call_id='bad', name='read_file', arguments='[1]'))
            elif reason == 'output_truncation':
                body.update(status='incomplete', incomplete_details=dict(reason='max_output_tokens'))
            elif reason == 'provider_refusal':
                body['output'][0]['content'] = [dict(type='refusal', refusal='provider refused bad-output')]
        else:
            body['content'][0]['text'] = 'bad-output'
            if reason == 'arguments':
                body['content'].append(dict(type='tool_use', id='bad', name='read_file', input=[1]))
            elif reason == 'output_truncation':
                body['stop_reason'] = 'max_tokens'
            elif reason == 'provider_refusal':
                body['stop_reason'] = 'refusal'
            events = [dict(type='message_start', message=body), dict(type='message_stop')]
            payload = ''.join('event: ' + event['type'] + '\ndata: ' + json.dumps(event) + '\n\n' for event in events)
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=httpx.ByteStream(payload.encode()))
        return httpx.Response(200, json=body)
    kwargs = dict(api_key='offline', max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    client = Anthropic(**kwargs) if api == 'messages' else OpenAI(**kwargs,
        base_url='https://api.deepseek.com' if api == 'chat' else 'https://api.openai.com/v1')
    recorder = ModelUsage(tmp_path / 'usage.jsonl', 'ceo')
    agent = BashAgent(get_bash_agent_tool_descriptions(), client, model='test-model', system_prompt='instructions',
        reasoning_effort='high' if api == 'responses' else None, usage_recorder=recorder,
        invalid_generation_attempts=2, allow_final_text=reason == 'provider_refusal', workspace_path=tmp_path)
    try:
        with pytest.raises(RunCancelled, match='after 2 model requests') as failure:
            agent.act('dashboard', 0, False, {'day': 7})
        assert failure.value.reason == 'model_attempt_limit'
        assert 'bad-output' in failure.value.detail and 'feedback' in failure.value.detail
        assert len(requests) == 2 and not agent._pending_tool_calls
        rows = [json.loads(line) for line in (tmp_path / 'invalid-generations.jsonl').read_text().splitlines()]
        assert [row['attempt'] for row in rows] == [1, 2]
        assert [row['reason'] for row in rows] == [reason, reason]
        assert all(row['feedback'] and 'bad-output' in json.dumps(row['output']) for row in rows)
        assert recorder.summary['calls'] == 2 and recorder.summary['errors'] == 0
    finally:
        client.close()


@pytest.mark.parametrize('limit', [0, 6])
def test_environment_can_extend_or_remove_invalid_generation_limit(tmp_path, monkeypatch, limit):
    monkeypatch.setenv('CEOBENCH_INVALID_GENERATION_ATTEMPTS', str(limit))
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        body = reply('chat')
        if len(requests) == 6:
            body['choices'][0]['message']['tool_calls'] = [dict(id='valid', type='function',
                function=dict(name='read_file', arguments='{"path":"MEMORY.md"}'))]
        return httpx.Response(200, json=body)
    client = OpenAI(api_key='offline', base_url='https://api.deepseek.com', max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    agent = BashAgent(get_bash_agent_tool_descriptions(), client, model='test-model', system_prompt='instructions',
                      workspace_path=tmp_path)
    try:
        action = agent.act('dashboard', 0, False, {'day': 7})
        assert action.tool == 'read_file' and action.arguments == {'path': 'MEMORY.md'}
        assert len(requests) == 6 and agent.invalid_generation_attempts == limit
    finally:
        client.close()
