import copy
import json

import httpx
import pytest
from anthropic import Anthropic, AnthropicBedrock
from openai import OpenAI

from saas_bench.config import BenchmarkConfig
from saas_bench.customer_llm import CustomerSimulator
from saas_bench.database import init_database
from saas_bench.model_usage import FIELDS, ModelUsage, cost_usd, usage_values
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions


def reply(api, usage=True):
    if api == 'chat':
        result = dict(id='chat1', object='chat.completion', created=1, model='test-model',
                      choices=[dict(index=0, finish_reason='stop', message=dict(role='assistant', content='hello'))])
        values = dict(prompt_tokens=10, completion_tokens=2, prompt_tokens_details={'cached_tokens': 3},
                      completion_tokens_details={'reasoning_tokens': 1})
    elif api == 'responses':
        result = dict(id='resp1', object='response', created_at=1, status='completed', model='test-model',
                      output=[dict(id='msg1', type='message', role='assistant', status='completed',
                                   content=[dict(type='output_text', text='hello', annotations=[])])])
        values = dict(input_tokens=10, output_tokens=2, total_tokens=12,
                      input_tokens_details={'cached_tokens': 3}, output_tokens_details={'reasoning_tokens': 1})
    else:
        result = dict(id='msg1', type='message', model='test-model', role='assistant',
                      content=[dict(type='text', text='hello')], stop_reason='end_turn', stop_sequence=None)
        values = dict(input_tokens=5, output_tokens=2, cache_read_input_tokens=3, cache_creation_input_tokens=2)
    if usage:
        result['usage'] = values
    return result


@pytest.mark.parametrize('provider,api', [('deepseek', 'chat'), ('opencode', 'chat'), ('openai', 'responses'),
                                         ('anthropic', 'messages'), ('bedrock', 'messages')])
def test_real_sdk_requests_internal_retries_and_simulator_usage(tmp_path, monkeypatch, provider, api):
    monkeypatch.setenv('CEOBENCH_SIMULATOR_USAGE_LOG', str(tmp_path / 'simulator.jsonl'))
    monkeypatch.setattr('time.sleep', lambda _: None)
    requests = []
    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(500, json={'error': {'type': 'server_error', 'message': 'try again'}})
        return httpx.Response(200, json=reply(api))
    http = httpx.Client(transport=httpx.MockTransport(handle))
    if provider == 'bedrock':
        client = AnthropicBedrock(aws_access_key='dummy', aws_secret_key='private-test-secret',
                                 aws_region='us-east-2', http_client=http, max_retries=1)
    elif provider == 'anthropic':
        client = Anthropic(api_key='private-test-secret', http_client=http, max_retries=1)
    else:
        client = OpenAI(api_key='private-test-secret', http_client=http, max_retries=1)
    sim = CustomerSimulator(client if api != 'messages' else None, init_database(':memory:'), BenchmarkConfig())
    if api == 'messages':
        setattr(sim, '_bedrock_client' if provider == 'bedrock' else '_anthropic_client', sim.usage_recorder.attach(client))
    assert sim.complete_text(provider=provider, model='test-model', user='full input', system='instructions', max_tokens=30) == ('hello', 10, 2)
    entries = [json.loads(line) for line in (tmp_path / 'simulator.jsonl').read_text().splitlines()]
    assert 'private-test-secret' not in json.dumps(entries)
    attempts = [r for r in entries if r['event'] == 'http_request']
    assert len(attempts) == 2 and [r['sdk_retry'] for r in attempts] == ['0', '1']
    assert len({r['call_id'] for r in entries}) == 1
    assert 'full input' in json.dumps(entries[0]['request'])
    assert [r['status'] for r in entries if r['event'] == 'http_response'] == [500, 200]
    assert entries[-1]['response']['usage'] == reply(api)['usage']
    assert sim.usage_recorder.summary['known']['input_tokens'] == 10
    assert sim.usage_recorder.summary['missing_cost'] == 1
    assert sim.usage_recorder.summary['http_attempts'] == 2
    assert sim.usage_recorder.summary['failed_attempts_without_usage'] == 1
    client.close()


def test_missing_usage_cache_prices_and_restored_subtotals(tmp_path):
    recorder = ModelUsage(tmp_path / 'agent.jsonl', 'agent', {'test-model': dict(input=1, output=2, cache_read=0.1, cache_write=1.5)})
    recorder.call('messages', {'model': 'test-model'}, lambda: reply('messages'), outer_attempt=1)
    assert recorder.summary['known_cost_usd'] == pytest.approx(0.0123)
    restored = ModelUsage(tmp_path / 'clone.jsonl', 'agent', recorder.pricing)
    restored.summary = copy.deepcopy(recorder.summary)
    partial = dict(reply('chat', usage=False), usage={'completion_tokens': 4})
    restored.call('chat', {}, lambda: partial, outer_attempt=2)
    restored.call('responses', {}, lambda: reply('responses', usage=False), outer_attempt=3)
    assert restored.summary['known']['input_tokens'] == 10
    assert restored.summary['known']['output_tokens'] == 6
    assert restored.summary['missing']['input_tokens'] == 2
    assert restored.summary['missing']['output_tokens'] == 1
    assert restored.summary['missing_cost'] == 2
    assert recorder.summary['calls'] == 1
    assert usage_values({}, 'chat') == dict.fromkeys(FIELDS)
    assert cost_usd(usage_values(reply('chat'), 'chat'), 'chat', None) is None
    assert usage_values({'usage': {'prompt_tokens': 8, 'prompt_cache_hit_tokens': 3}}, 'chat')['cached_tokens'] == 3


def test_go_cache_fields_and_sourced_time_bounded_prices(tmp_path):
    from datetime import datetime, timezone
    from saas_bench.model_usage import load_pricing
    response = reply('chat')
    response['usage']['prompt_tokens_details']['cache_write_tokens'] = 0
    usage = usage_values(response, 'chat')
    assert usage['cache_creation_tokens'] == 0
    response['usage']['prompt_tokens_details']['cache_creation_input_tokens'] = 2
    assert usage_values(response, 'chat')['cache_creation_tokens'] == 2
    rates = dict(input=.00015, output=.0006, cache_read=.000003,
                 valid_from='2026-09-22T10:00:00+00:00', valid_until='2026-09-23T01:00:00+00:00')
    path = tmp_path / 'pricing.json'
    path.write_text(json.dumps(dict(source='https://opencode.ai/docs/go/', basis='subscription quota, USD/1k', rates={'test-model': rates})))
    assert load_pricing(path)['rates']['test-model'] == rates
    assert cost_usd(usage, 'chat', rates, datetime(2026, 9, 22, 11, tzinfo=timezone.utc)) == pytest.approx(.000002259)
    assert cost_usd(usage, 'chat', rates, datetime(2026, 9, 23, 1, tzinfo=timezone.utc)) is None
    for invalid in (-1, float('inf'), True):
        rates['input'] = invalid
        path.write_text(json.dumps(dict(source='documented', basis='USD/1k', rates={'test-model': rates})))
        with pytest.raises(ValueError, match='finite nonnegative'):
            load_pricing(path)


@pytest.mark.parametrize('capture', [False, True])
def test_connection_retry_and_interrupted_anthropic_stream(tmp_path, monkeypatch, capture):
    monkeypatch.setattr('time.sleep', lambda _: None)
    attempts = []
    class Interrupted(httpx.SyncByteStream):
        def __iter__(self):
            message = reply('messages')
            message['content'] = []
            yield ('event: message_start\ndata: ' + json.dumps({'type': 'message_start', 'message': message}) + '\n\n').encode()
            raise httpx.ReadError('offline interruption')
    def handle(request):
        attempts.append(request)
        if len(attempts) == 1:
            raise httpx.ConnectError('offline connection error')
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=Interrupted())
    from saas_bench.sql_evidence import SQLEvidenceStore
    from test_sql_evidence import identity, event_ids
    store = SQLEvidenceStore(tmp_path / 'private.sqlite', identity(capture_scope='execution')) if capture else None
    recorder = ModelUsage(tmp_path / 'agent.jsonl', 'agent', evidence_store=store)
    client = recorder.attach(Anthropic(api_key='test-secret', max_retries=1,
                                      http_client=httpx.Client(transport=httpx.MockTransport(handle))))
    request = dict(model='test-model', messages=[{'role': 'user', 'content': 'hello'}], max_tokens=30)
    def invoke():
        with client.messages.stream(**request) as stream:
            return stream.get_final_message()
    with pytest.raises(httpx.ReadError):
        recorder.call('messages', request, invoke, outer_attempt=1)
    entries = [json.loads(s) for s in recorder.path.read_text().splitlines()]
    assert len([r for r in entries if r['event'] == 'http_request']) == 2
    assert any(r['event'] == 'http_error' and r['error'] == 'ConnectError' for r in entries)
    partial = next(r for r in entries if r['event'] == 'http_response')
    assert 'message_start' in partial['body'] and partial['error'] == 'ReadError'
    assert recorder.summary['errors'] == 1
    assert recorder.summary['known'] == dict.fromkeys(FIELDS)
    assert recorder.summary['missing'] == dict.fromkeys(FIELDS, 1)
    if store:
        store.assert_healthy()
        events = [store.read_event(event) for event in event_ids(store)]
        assert len(events) == 2
        assert events[0]['result']['send_state'] == 'unknown'
        assert events[1]['result']['response_error'] == 'ReadError'
    client.close()


@pytest.mark.parametrize('status', [503, 200])
def test_agent_outer_retry_records_full_request_and_missing_usage(tmp_path, monkeypatch, status):
    from saas_bench.agents.bash_agent.agent import BashAgent
    monkeypatch.setattr('time.sleep', lambda _: None)
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(status, json={'error': {'message': 'offline retry', 'type': 'server_error'}})
        body = reply('chat', usage=False)
        body['choices'][0]['message']['tool_calls'] = [dict(id='call1', type='function',
            function=dict(name='read_file', arguments='{"path":"MEMORY.md"}'))]
        return httpx.Response(200, json=body)
    client = OpenAI(api_key='test-secret', max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    recorder = ModelUsage(tmp_path / 'agent.jsonl', 'agent')
    agent = BashAgent(get_bash_agent_tool_descriptions(), client, system_prompt='original instructions', workspace_path=tmp_path, usage_recorder=recorder)
    assert agent.act('dashboard', 0, False, {'day': 7}).tool == 'read_file'
    entries = [json.loads(s) for s in recorder.path.read_text().splitlines()]
    calls = [r for r in entries if r['event'] == 'request']
    assert [r['outer_attempt'] for r in calls] == [1, 2]
    assert all('original instructions' in json.dumps(r['request']['messages']) for r in calls)
    assert agent.last_input_tokens is None
    assert recorder.summary['missing']['input_tokens'] == 2
    assert recorder.summary['known']['input_tokens'] is None
    assert recorder.summary['errors'] == 1
    assert recorder.summary['failed_http_attempts'] == 1
    client.close()


@pytest.mark.parametrize('finish_reason', ['tool_calls', 'length'])
@pytest.mark.parametrize('reasoning_fields', [
    {'reasoning_content': 'think '},
    {'reasoning': 'think '},
    {'reasoning_content': None, 'reasoning': 'think '},
    {'reasoning_content': '', 'reasoning': 'think '},
    {'reasoning_content': 'think ', 'reasoning': 'ignored '},
], ids=['original', 'alias', 'null', 'empty', 'precedence'])
@pytest.mark.parametrize('effort', ['high', 'none'])
def test_go_stream_preserves_reasoning_tools_usage_and_terminal_receipt(tmp_path, finish_reason, reasoning_fields, effort):
    from saas_bench.agents.bash_agent.agent import BashAgent
    requests, reasoning = [], []
    def handle(request):
        requests.append(json.loads(request.content))
        deltas = [
            dict(role='assistant', content='', **reasoning_fields, tool_calls=[dict(index=0, id='call1', type='function', function=dict(name='read_file', arguments='{"path":'))]),
            dict(tool_calls=[dict(index=0, id=None, type=None, function=None)]),
            dict(tool_calls=[dict(index=0, id=None, type=None, function=dict(name=None, arguments=None))]),
            dict(**{key: 'again' if value else value for key, value in reasoning_fields.items()}, tool_calls=[dict(index=0, id=None, type=None, function=dict(name=None, arguments='"MEMORY.md"}'))]),
            {},
        ]
        frames = [dict(id='chat1', object='chat.completion.chunk', created=1, model='test-model', choices=[dict(index=0, delta=delta, finish_reason=finish_reason if i == len(deltas) - 1 else None)]) for i, delta in enumerate(deltas)]
        frames.append(dict(id='chat1', object='chat.completion.chunk', created=1, model='test-model', choices=[], usage=dict(reply('chat')['usage'], total_tokens=12)))
        data = ''.join('data: ' + json.dumps(frame) + '\n\n' for frame in frames) + 'data: [DONE]\n\n'
        return httpx.Response(200, stream=httpx.ByteStream(data.encode()), headers={'content-type': 'text/event-stream'})
    client = OpenAI(api_key='offline', base_url='https://opencode.ai/zen/go/v1/', max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    recorder = ModelUsage(tmp_path / 'agent.jsonl', 'agent', {'test-model': dict(input=1, output=2, cache_read=.1)})
    agent = BashAgent(get_bash_agent_tool_descriptions(), client, model='test-model', reasoning_effort=effort, system_prompt='instructions', workspace_path=tmp_path, usage_recorder=recorder,
                      tool_result_callback=lambda *args: reasoning.append(args))
    action = agent.act('dashboard', 0, False, {'day': 7})
    assert action.tool == 'read_file' and action.arguments == {'path': 'MEMORY.md'}
    assert agent.conversation[-1].reasoning_content == ('think again' if effort == 'high' else None)
    assert reasoning == [(1, 7, '_reasoning', {}, 'think again')]
    assert requests[0]['stream'] and requests[0]['stream_options'] == {'include_usage': True}
    assert recorder.summary['known']['input_tokens'] == 10 and recorder.summary['known']['cached_tokens'] == 3
    assert recorder.summary['known_cost_usd'] == pytest.approx(.0113)
    assert recorder.summary['failed_http_attempts'] == recorder.summary['missing_cost'] == 0
    entries = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    http_response = next(row for row in entries if row['event'] == 'http_response')
    assert http_response['error'] is None
    assert '"type": null' in http_response['body'] and '"function": null' in http_response['body']
    agent.record_tool_result('contents')
    assert agent.act('contents', 0, False, {'day': 7}).tool == 'read_file'
    assistant = next(message for message in requests[1]['messages'] if message['role'] == 'assistant')
    assert 'reasoning' not in assistant
    if effort == 'high':
        assert assistant['reasoning_content'] == 'think again'
    else:
        assert 'reasoning_content' not in assistant
    client.close()


@pytest.mark.parametrize('fault', ['missing_done', 'missing_finish', 'missing_both', 'missing_id',
                                  'missing_type', 'missing_name', 'missing_function', 'invalid_json',
                                  'non_object_arguments', 'empty_arguments'])
def test_go_incomplete_stream_retries_without_exposing_partial_tools(tmp_path, monkeypatch, fault):
    from saas_bench.agents.bash_agent.agent import BashAgent
    monkeypatch.setattr('time.sleep', lambda _: None)
    requests, responses, payloads = [], [], []
    def handle(request):
        requests.append(json.loads(request.content))
        first = len(requests) == 1
        done = not first or fault not in ('missing_done', 'missing_both')
        finished = not first or fault not in ('missing_finish', 'missing_both')
        base = dict(id='chat1', object='chat.completion.chunk', created=1, model='test-model')
        tool = dict(index=0, id='call1', type='function', function=dict(name='read_file',
                    arguments=json.dumps({'path': 'PARTIAL.md' if first else 'MEMORY.md'})))
        if first:
            if fault in ('missing_id', 'missing_type', 'missing_function'):
                tool[fault.removeprefix('missing_')] = None
            elif fault == 'missing_name':
                tool['function']['name'] = None
            elif fault in ('invalid_json', 'non_object_arguments', 'empty_arguments'):
                tool['function']['arguments'] = {'invalid_json': '{', 'non_object_arguments': '[]',
                                                 'empty_arguments': ''}[fault]
        frames = [dict(base, choices=[dict(index=0, delta=dict(role='assistant', tool_calls=[
            tool]), finish_reason=None)]),
            dict(base, choices=[dict(index=0, delta={}, finish_reason='tool_calls' if finished else None)]),
            dict(base, choices=[], usage=dict(reply('chat')['usage'], total_tokens=12))]
        payload = ''.join('data: ' + json.dumps(frame) + '\n\n' for frame in frames)
        if done:
            payload += 'data: [DONE]\n\n'
        payloads.append(payload)
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=httpx.ByteStream(payload.encode()))
    client = OpenAI(api_key='offline', base_url='https://opencode.ai/zen/go/v1/', max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    recorder = ModelUsage(tmp_path / 'agent.jsonl', 'agent', {'test-model': dict(input=1, output=2, cache_read=.1)})
    agent = BashAgent(get_bash_agent_tool_descriptions(), client, model='test-model', system_prompt='instructions',
                      workspace_path=tmp_path, usage_recorder=recorder, response_callback=lambda **kwargs: responses.append(kwargs))
    action = agent.act('dashboard', 0, False, {'day': 7})
    assert action.tool == 'read_file' and action.arguments == {'path': 'MEMORY.md'}
    assert len(requests) == 2 and requests[0] == requests[1]
    assert len(responses) == 1 and len(agent.conversation) == 3
    assert recorder.summary['errors'] == recorder.summary['missing_cost'] == 1
    assert recorder.summary['known']['input_tokens'] == 10
    assert recorder.summary['missing']['input_tokens'] == 1
    entries = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    logical = [row for row in entries if row['event'] == 'response']
    assert logical[0]['error'] == 'APIConnectionError' and logical[0]['response'] is None
    assert logical[0]['cost_usd'] is None and logical[0]['usage'] == dict.fromkeys(FIELDS)
    assert logical[1]['error'] is None
    first_http = next(row for row in entries if row['event'] == 'http_response')
    assert first_http['body'] == payloads[0]
    done = fault not in ('missing_done', 'missing_both')
    assert first_http['error'] == (None if done else 'stream_closed_before_completion')
    client.close()


@pytest.mark.parametrize('provider,api', [('deepseek', 'chat'), ('opencode', 'chat'), ('openai', 'responses'),
                                         ('anthropic', 'messages'), ('bedrock', 'messages')])
def test_http200_error_body_is_a_failed_model_call(tmp_path, monkeypatch, provider, api):
    from saas_bench.sql_evidence import SQLEvidenceStore
    from test_sql_evidence import identity, event_ids
    monkeypatch.setenv('CEOBENCH_SIMULATOR_USAGE_LOG', str(tmp_path / 'simulator.jsonl'))
    body = {'error': {'message': 'We were unable to start processing your request within the 900-second '
                                'timeout limit. Please try again later.'}}
    http = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body)))
    if provider == 'bedrock':
        client = AnthropicBedrock(aws_access_key='offline', aws_secret_key='offline', aws_region='us-east-2',
                                 http_client=http, max_retries=0)
    else:
        client = (Anthropic if api == 'messages' else OpenAI)(api_key='offline', http_client=http, max_retries=0)
    sim = CustomerSimulator(client if api != 'messages' else None, init_database(':memory:'), BenchmarkConfig())
    if api == 'messages':
        setattr(sim, '_bedrock_client' if provider == 'bedrock' else '_anthropic_client', sim.usage_recorder.attach(client))
    store = SQLEvidenceStore(tmp_path / 'private.sqlite', identity(capture_scope='execution'))
    sim.usage_recorder.evidence_store = store
    try:
        with pytest.raises(ValueError, match='900-second timeout'):
            sim.complete_text(provider=provider, model='test-model', user='offline replay', max_tokens=30)
        summary = sim.usage_recorder.summary
        assert summary['calls'] == summary['errors'] == summary['http_attempts'] == summary['failed_http_attempts'] == 1
        assert summary['failed_attempts_without_usage'] == summary['missing_cost'] == 1
        assert summary['known'] == dict.fromkeys(FIELDS) and summary['known_cost_usd'] is None
        entries = [json.loads(line) for line in (tmp_path / 'simulator.jsonl').read_text().splitlines()]
        http_response = next(e for e in entries if e['event'] == 'http_response')
        assert http_response['status'] == 200 and json.loads(http_response['body']) == body
        assert '900-second timeout' in http_response['error']
        assert entries[-1]['response']['error'] == body['error'] and entries[-1]['error']
        captured = store.read_event(event_ids(store)[0])['result']
        assert captured['status'] == 'failed' and captured['http_status'] == 200
        assert captured['send_state'] == 'response_received'
        store.assert_healthy()
    finally:
        client.close()
        sim.conn.close()


@pytest.mark.parametrize('api,field', [('chat', 'choices'), ('responses', 'output'), ('messages', 'content')])
def test_malformed_response_is_recorded_before_returning(tmp_path, api, field):
    recorder = ModelUsage(tmp_path / 'usage.jsonl', 'agent')
    malformed = reply(api)
    malformed[field] = None
    with pytest.raises(ValueError, match=field):
        recorder.call(api, {}, lambda: malformed)
    assert recorder.summary['errors'] == 1
    assert recorder.summary['known']['input_tokens'] == 10
    assert json.loads(recorder.path.read_text().splitlines()[-1])['response'] == malformed


def test_agent_rejects_empty_tools_before_any_request():
    from saas_bench.agents.bash_agent.agent import BashAgent
    with pytest.raises(ValueError, match='requires tools'):
        BashAgent([], object())


def test_usage_log_pairs_responses_by_request_day_and_retains_unknowns(tmp_path):
    from saas_bench.model_usage import summarize_usage_log
    events = []
    def request(call, day):
        events.append(dict(event='request', call_id=call, day=day))
    def response(call, body, error=None, cost=None):
        events.append(dict(event='response', call_id=call, api='chat', response=body,
                           usage=usage_values(body, 'chat'), error=error, cost_usd=cost))
    request('prefix', 21)
    response('prefix', reply('chat'), cost=.1)
    request('no-tool', 28)
    request('next-week', 35)
    response('next-week', reply('chat'), cost=.3)
    response('no-tool', reply('chat'), cost=.2)
    request('tool', 28)
    tool = reply('chat')
    tool['choices'][0]['message']['tool_calls'] = [dict(id='tool1', type='function',
        function=dict(name='read_file', arguments='{}'))]
    tool['usage']['prompt_tokens'] = 20
    response('tool', tool, cost=.4)
    request('failed', 28)
    response('failed', None, error='APITimeoutError')
    request('unreturned', 28)
    log = tmp_path / 'agent_requests.jsonl'
    log.write_text(''.join(json.dumps(row) + '\n' for row in events))
    usage = summarize_usage_log(log, start_day=28, end_day=35)
    assert usage['calls'] == 3 and usage['errors'] == 1
    assert usage['known']['input_tokens'] == 30
    assert usage['known']['output_tokens'] == 4
    assert usage['known']['cached_tokens'] == 6
    assert usage['known']['reasoning_tokens'] == 2
    assert usage['missing']['input_tokens'] == 1
    assert usage['known']['cache_creation_tokens'] is None
    assert usage['missing']['cache_creation_tokens'] == 3
    assert usage['known_cost_usd'] == pytest.approx(.6) and usage['missing_cost'] == 1
    assert usage['unreturned_requests'] == 1
    all_usage = summarize_usage_log(log)
    assert all_usage['calls'] == 5 and all_usage['known']['input_tokens'] == 50


def test_usage_delta_excludes_prior_calls_and_preserves_unknown_or_zero_usage():
    from saas_bench.model_usage import usage_delta
    recorder = ModelUsage(None, 'agent')
    recorder.call('chat', {}, lambda: reply('chat'))
    previous = copy.deepcopy(recorder.summary)
    recorder.call('chat', {}, lambda: reply('chat', usage=False))
    delta = usage_delta(previous, recorder.summary)
    assert delta['calls'] == 1 and delta['known'] == dict.fromkeys(FIELDS)
    assert delta['missing'] == dict.fromkeys(FIELDS, 1)
    previous = copy.deepcopy(recorder.summary)
    assert usage_delta(previous, recorder.summary) == dict(calls=0, known=dict.fromkeys(FIELDS, 0),
                                                          missing=dict.fromkeys(FIELDS, 0))
    zero = reply('chat')
    zero['usage'] = dict(prompt_tokens=0, completion_tokens=0, prompt_tokens_details={'cached_tokens': 0},
                         completion_tokens_details={'reasoning_tokens': 0}, cache_creation_input_tokens=0)
    recorder.call('chat', {}, lambda: zero)
    delta = usage_delta(previous, recorder.summary)
    assert delta['calls'] == 1 and delta['known'] == delta['missing'] == dict.fromkeys(FIELDS, 0)
