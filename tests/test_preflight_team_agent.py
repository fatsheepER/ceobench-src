from concurrent.futures import ThreadPoolExecutor
import json

from anthropic import Anthropic
import httpx
from openai import OpenAI
import pytest

from saas_bench.agents.bash_agent.agent import BashAgent, FinalText
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from saas_bench.run_lifecycle import RunCancelled
from test_preflight_usage import reply
from test_preflight_team_flow import completion, final


@pytest.mark.parametrize('api', ['chat', 'responses', 'messages'])
@pytest.mark.parametrize('complete', [True, False])
def test_analyst_final_text_in_worker(tmp_path, monkeypatch, api, complete):
    monkeypatch.setattr('time.sleep', lambda _: None)
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        body = reply(api)
        if not complete:
            if api == 'chat':
                body['choices'][0]['finish_reason'] = 'length'
            elif api == 'responses':
                body['status'] = 'incomplete'
            else:
                body['stop_reason'] = 'max_tokens'
        if api != 'messages':
            return httpx.Response(200, json=body)
        events = [dict(type='message_start', message=body), dict(type='message_stop')]
        payload = ''.join('event: ' + e['type'] + '\ndata: ' + json.dumps(e) + '\n\n' for e in events)
        return httpx.Response(200, content=payload, headers={'content-type': 'text/event-stream'})
    client = (Anthropic if api == 'messages' else OpenAI)(api_key='offline', max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    agent = BashAgent(get_bash_agent_tool_descriptions(), client, model='offline',
        system_prompt='Read-only analyst. Answer in final text.', workspace_path=tmp_path,
        reasoning_effort='high' if api == 'responses' else None, allow_final_text=True)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(agent.act, 'dashboard', 0, False, {'day': 0})
            if complete:
                assert result.result() == FinalText('hello')
                assert len(requests) == 1
            else:
                with pytest.raises(RunCancelled, match='model_attempt_limit'):
                    result.result()
                assert len(requests) == 4
    finally:
        client.close()


def test_text_with_tools_remains_action(tmp_path):
    def handle(request):
        body = reply('chat')
        body['choices'][0]['message']['tool_calls'] = [dict(id='c1', type='function',
            function=dict(name='read_file', arguments='{"path":"MEMORY.md"}'))]
        return httpx.Response(200, json=body)
    with OpenAI(api_key='offline', max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        agent = BashAgent(get_bash_agent_tool_descriptions(), client, system_prompt='analyst',
            workspace_path=tmp_path, allow_final_text=True)
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(agent.act, 'dashboard', 0, False, {'day': 0}).result()
        assert result.tool == 'read_file'
        assert agent.conversation[-1].content == 'hello'
        assert agent._pending_tool_calls[0]['id'] == 'c1'


@pytest.mark.parametrize('team', [False, True])
def test_real_opencode_act_uses_team_phase_timeouts_and_preserves_single_agent(tmp_path, team):
    from types import SimpleNamespace
    from saas_bench import team_batch as batch
    timeouts = []

    def handle(request):
        timeouts.append(request.extensions['timeout'])
        return completion(final('offline advice'), stream=True)

    transport = httpx.MockTransport(handle)
    client = (batch.checked_client('growth', 'offline', transport=transport) if team else
        OpenAI(api_key='offline', base_url=batch.ENDPOINT, max_retries=0,
            http_client=httpx.Client(transport=transport)))
    try:
        agent = BashAgent(get_bash_agent_tool_descriptions(), client, model=batch.MODEL,
            system_prompt='Give business advice.', workspace_path=tmp_path, reasoning_effort='high',
            identity=SimpleNamespace(role='growth') if team else None, allow_final_text=True)
        assert agent.act('dashboard', 0, False, {'day': 0}) == FinalText('offline advice')
        assert timeouts == [dict(connect=60, read=180 if team else 60, write=60, pool=60)]
    finally:
        client.close()


@pytest.mark.parametrize('previous_success', [False, True])
def test_denied_execution_has_cancelled_status(tmp_path, previous_success):
    from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
    executor = BashAgentToolExecutor(tmp_path)
    if previous_success:
        executor.execute('write_file', {'path': 'marker', 'content': 'before'})
        assert executor.last_status == 'succeeded'
    def deny(role, operation):
        raise PermissionError('frozen')
    executor.authorize = deny
    assert executor.execute('write_file', {'path': 'marker', 'content': 'after'}).startswith('Error:')
    assert executor.last_status == 'cancelled'
    marker = tmp_path / 'marker'
    assert marker.read_text() == 'before' if previous_success else not marker.exists()
