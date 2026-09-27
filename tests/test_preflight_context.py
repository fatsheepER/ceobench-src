import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from saas_bench.agents.bash_agent.agent import BashAgent, Message
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions


def agent(tmp_path, anthropic=False):
    value = BashAgent.__new__(BashAgent)
    value.conversation = []
    value._pending_tool_calls = []
    value.current_day = 7
    value.turns_today = 2
    value.total_turns = 2
    value._last_observation = ''
    value._snapshot_path = tmp_path / 'conversation.json'
    value.workspace_path = tmp_path
    value.system_prompt = 'Original prompt'
    value.use_anthropic = anthropic
    value._skip_next_refresh = False
    value.max_turns_per_day = 0
    value._call_llm = lambda: 'action'
    return value


@pytest.mark.parametrize('shape', ['chat', 'responses', 'anthropic'])
def test_completed_tool_roundtrip_and_week_refresh(tmp_path, shape):
    first = agent(tmp_path, shape == 'anthropic')
    call = {'type': 'function_call', 'call_id': 'call1', 'name': 'bash', 'arguments': '{}'}
    first.conversation = [Message('system', 'Original prompt'),
                         Message('assistant', [call] if shape != 'chat' else '',
                                  tool_calls=[{'id': 'call1'}] if shape == 'chat' else None)]
    first._pending_tool_calls = [{'id': 'call1', 'name': 'bash'}]
    first.record_tool_result('completed')
    first._save_conversation_snapshot(strict=True)
    second = agent(tmp_path, shape == 'anthropic')
    assert second.load_conversation_snapshot(second._snapshot_path)
    before = [second._serialize_message(m) for m in second.conversation]
    assert second.act('completed', 0, False, {'day': 7}) == 'action'
    assert [second._serialize_message(m) for m in second.conversation] == before
    (tmp_path / 'MEMORY.md').write_text('remember this')
    second.act('new dashboard', 0, False, {'day': 14})
    assert all(m.role != 'assistant' for m in second.conversation)
    assert 'remember this' in second._get_system_prompt_with_memory()


def test_pending_tool_is_not_silently_replayed(tmp_path):
    first = agent(tmp_path)
    first._pending_tool_calls = [{'id': 'unknown', 'name': 'bash'}]
    first._save_conversation_snapshot()
    assert not agent(tmp_path).load_conversation_snapshot(first._snapshot_path)
    assert first.check_day_advanced('=== Week 2 Dashboard (Day 14) ===\nresult')


def test_memory_keeps_original_limit(tmp_path):
    value = agent(tmp_path)
    (tmp_path / 'MEMORY.md').write_text('a' * 40000 + 'not included')
    prompt = value._get_system_prompt_with_memory()
    assert 'a' * 40000 in prompt and 'not included' not in prompt


@pytest.mark.parametrize('capture', [False, True])
@pytest.mark.parametrize('api', ['chat', 'responses', 'messages'])
@pytest.mark.parametrize('initial_memory', ['memory A', '', None])
def test_real_sdk_context_matches_continuous_after_restore(tmp_path, api, capture, initial_memory):
    import httpx
    from openai import OpenAI
    from anthropic import Anthropic
    from test_preflight_usage import reply
    from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
    captured = []
    store = None
    if capture:
        from saas_bench.sql_evidence import SQLEvidenceStore
        from test_sql_evidence import identity, event_ids
        store = SQLEvidenceStore(tmp_path.parent / (tmp_path.name + '.sqlite'), identity(capture_scope='execution'))
    (tmp_path / 'contents.txt').write_text('contents A')
    observation = BashAgentToolExecutor(tmp_path, evidence_store=store).execute('read_file', {'path': 'contents.txt'})
    def handle(request):
        captured.append(json.loads(request.content))
        body = reply(api)
        if api == 'chat':
            body['choices'][0]['message']['tool_calls'] = [dict(id='call1', type='function',
                function=dict(name='read_file', arguments='{"path":"MEMORY.md"}'))]
            return httpx.Response(200, json=body)
        if api == 'responses':
            body['output'] = [dict(type='function_call', id='fc1', call_id='call1', name='read_file',
                                  arguments='{"path":"MEMORY.md"}', status='completed')]
            return httpx.Response(200, json=body)
        body['content'] = []
        events = [dict(type='message_start', message=body),
                  dict(type='content_block_start', index=0, content_block=dict(type='tool_use', id='call1', name='read_file', input={})),
                  dict(type='content_block_delta', index=0, delta=dict(type='input_json_delta', partial_json='{"path":"MEMORY.md"}')),
                  dict(type='content_block_stop', index=0),
                  dict(type='message_delta', delta={'stop_reason': 'tool_use', 'stop_sequence': None}, usage={'output_tokens': 2}),
                  dict(type='message_stop')]
        stream = ''.join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, content=stream, headers={'content-type': 'text/event-stream'})
    client = (Anthropic if api == 'messages' else OpenAI)(api_key='offline-only', max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    def new_agent():
        value = BashAgent(get_bash_agent_tool_descriptions(), client, system_prompt='Original instructions', workspace_path=tmp_path,
                          reasoning_effort='low' if api == 'responses' else None)
        if capture:
            value.usage_recorder.evidence_store = store
            value.usage_recorder.context_id = 'fixture-context'
        value._snapshot_path = tmp_path / 'context.json'
        return value
    if initial_memory is not None:
        (tmp_path / 'MEMORY.md').write_text(initial_memory)
    def system(body):
        if api == 'chat':
            assert body['messages'][0]['role'] == 'system'
            return body['messages'][0]['content']
        if api == 'responses':
            return body['instructions']
        return body['system'][0]['text']
    first = new_agent()
    assert first.current_day == -1
    assert first.act('dashboard', 0, False, {'day': 0}).tool == 'read_file'
    frozen = system(captured[0])
    assert frozen.startswith('Original instructions')
    assert ('memory A' in frozen) == bool(initial_memory)
    first.record_tool_result(observation)
    first._save_conversation_snapshot(strict=True)
    (tmp_path / 'MEMORY.md').write_text('memory B')
    second = new_agent()
    second.system_prompt = 'Changed base prompt must wait for the next week'
    assert second.load_conversation_snapshot(second._snapshot_path)
    first.act('contents A', 0, False, {'day': 0})
    second.act('contents A', 0, False, {'day': 0})
    assert captured[1] == captured[2]
    assert system(captured[1]) == frozen
    assert 'memory B' not in system(captured[2])
    assert 'call1' in json.dumps(captured[2]) and 'contents A' in json.dumps(captured[2])
    second.record_tool_result('contents A')
    (tmp_path / 'MEMORY.md').write_text('persistent note')
    second.act('new week', 0, False, {'day': 7})
    assert 'persistent note' in json.dumps(captured[-1])
    assert system(captured[-1]).startswith(second.system_prompt)
    assert 'call1' not in json.dumps(captured[-1])
    assert first.usage_recorder.summary['http_attempts'] == 2
    assert second.usage_recorder.summary['http_attempts'] == 2
    if capture:
        store.assert_healthy()
        model_events = [event for event in event_ids(store) if store.read_event(event)['request']['kind'] == 'model_request']
        assert len(model_events) == len(captured) == 4
        maps = [json.loads(store.get_content(event + ':occurrences')[1]) for event in model_events]
        assert maps[1] and maps[1] == [dict(item, call_id=maps[1][0]['call_id'], attempt_id=maps[1][0]['attempt_id'], send_state_event_id=maps[1][0]['send_state_event_id']) for item in maps[2]]
        assert all(item['version_id'] not in {x['version_id'] for x in maps[1]} for item in maps[-1])
        assert maps[-1]  # Actual new-week MEMORY read.
        assert 'version_id' not in first._snapshot_path.read_text()
        memory_events = [event for event in event_ids(store) if store.read_event(event)['request']['kind'] == 'memory_read']
        assert len(memory_events) == (2 if initial_memory is not None else 1)
        memory_maps = [[item for item in mapping if item['role'] == 'system'] for mapping in maps]
        assert bool(memory_maps[0]) == bool(initial_memory)
        assert [item['version_id'] for item in memory_maps[0]] == [item['version_id'] for item in memory_maps[2]]
        destination = os.environ.get('CEOBENCH_STAGE2_ARTIFACTS')
        if destination and initial_memory == 'memory A':
            output = Path(destination)
            output.mkdir(parents=True, exist_ok=True)
            (output / f'model-api-{api}.json').write_text(json.dumps({
                'evidence_kind': 'actually_executed_local_mock_transport',
                'external_provider_calls': 0, 'api': api,
                'requests_before_reset': captured[:4], 'source_occurrences': maps,
                'same_week_restored_request_equal': captured[1] == captured[2],
            }, indent=2, ensure_ascii=False))
            store.snapshot(output / f'model-api-{api}.sqlite')
    if not capture and initial_memory == 'memory A' and os.environ.get('CEOBENCH_STAGE2_ARTIFACTS'):
        output = Path(os.environ['CEOBENCH_STAGE2_ARTIFACTS'])
        output.mkdir(parents=True, exist_ok=True)
        (output / f'model-api-{api}-uncaptured.json').write_text(json.dumps({
            'evidence_kind': 'actually_executed_local_mock_transport',
            'external_provider_calls': 0, 'api': api,
            'requests_before_reset': captured[:4],
        }, indent=2, ensure_ascii=False))
    second.reset()
    assert second.current_day == -1
    second.act('reset day zero', 0, False, {'day': 0})
    assert system(captured[-1]).startswith(second.system_prompt)
    assert 'call1' not in json.dumps(captured[-1])
    client.close()


@pytest.mark.parametrize('anthropic', [False, True])
def test_legacy_snapshot_freezes_missing_system_once(tmp_path, anthropic):
    first = agent(tmp_path, anthropic)
    first.conversation = [Message('user', 'old dashboard')]
    first._observation_recorded = True
    first._save_conversation_snapshot(strict=True)
    (tmp_path / 'MEMORY.md').write_text('note at migration')
    second = agent(tmp_path, anthropic)
    assert second.load_conversation_snapshot(first._snapshot_path)
    assert second.conversation[0].role == 'system'
    assert 'note at migration' in second.conversation[0].content
    assert second.conversation[1].content == 'old dashboard'
    second._save_conversation_snapshot(strict=True)
    (tmp_path / 'MEMORY.md').write_text('later note')
    third = agent(tmp_path, anthropic)
    assert third.load_conversation_snapshot(first._snapshot_path)
    assert third.conversation[0].content == second.conversation[0].content


class Rejected(BaseException):
    pass


@pytest.mark.parametrize('effort', ['high', 'none'])
def test_deepseek_thinking_returns_reasoning_after_tool_calls_and_restore(tmp_path, monkeypatch, effort):
    import httpx
    from openai import OpenAI
    from test_preflight_usage import reply
    monkeypatch.setattr('time.sleep', lambda _: None)
    captured, rejected = [], []
    def handle(request):
        body = json.loads(request.content)
        captured.append(body)
        thinking = body['thinking']['type'] == 'enabled'
        assistants = [m for m in body['messages'] if m['role'] == 'assistant']
        # Official rule: after a tool-call turn every assistant turn must return its reasoning verbatim.
        if thinking and any(m.get('reasoning_content') != f"think {i}" for i, m in enumerate(assistants)):
            rejected.append(body)
            if len(rejected) > 1:
                # The agent feeds 400s back and retries forever; fail instead of hanging.
                raise Rejected('reasoning_content was not passed back')
            return httpx.Response(400, json={'error': {'message': 'The reasoning_content in the thinking mode '
                                                       'must be passed back to the API.', 'type': 'invalid_request_error'}})
        assert thinking or all('reasoning_content' not in m for m in assistants)
        result = reply('chat')
        message = result['choices'][0]['message']
        if thinking:
            message['reasoning_content'] = f"think {len(assistants)}"
        message['tool_calls'] = [dict(id=f'call{len(assistants)}', type='function',
                                      function=dict(name='read_file', arguments='{"path":"MEMORY.md"}'))]
        return httpx.Response(200, json=result)
    client = OpenAI(api_key='offline-only', base_url='https://api.deepseek.com', max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    def new_agent():
        value = BashAgent(get_bash_agent_tool_descriptions(), client, system_prompt='Original instructions',
                          workspace_path=tmp_path, reasoning_effort=effort)
        value._snapshot_path = tmp_path / 'context.json'
        return value
    first = new_agent()
    for _ in range(2):
        assert first.act('observation', 0, False, {'day': 0}).tool == 'read_file'
        first.record_tool_result('contents')
    first._save_conversation_snapshot(strict=True)
    second = new_agent()
    assert second.load_conversation_snapshot(second._snapshot_path)
    first.act('contents', 0, False, {'day': 0})
    second.act('contents', 0, False, {'day': 0})
    assert not rejected and captured[-1] == captured[-2]
    if effort == 'high':
        assert [m['reasoning_content'] for m in captured[-1]['messages'] if m['role'] == 'assistant'] == ['think 0', 'think 1']
        assert captured[-1]['reasoning_effort'] == 'high'
    second.record_tool_result('contents')
    second.act('new week', 0, False, {'day': 7})
    assert not rejected and all(m['role'] != 'assistant' for m in captured[-1]['messages'])
    client.close()
