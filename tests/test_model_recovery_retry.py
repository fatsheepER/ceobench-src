import json
import threading

import httpx
import pytest
from anthropic import Anthropic, AnthropicBedrock
from openai import OpenAI

from saas_bench.config import BenchmarkConfig
from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from saas_bench.customer_llm import CustomerSimulator
from saas_bench.database import init_database
from saas_bench.llm import AgentLLM, CostTracker
from saas_bench.model_usage import ModelUsage
from saas_bench.run_lifecycle import RunCancelled
from test_preflight_usage import reply


@pytest.mark.parametrize('provider,api', [('opencode', 'chat'), ('deepseek', 'chat'),
    ('openai', 'responses'), ('anthropic', 'messages'), ('bedrock', 'messages')])
def test_customer_retries_one_unchanged_request_and_retains_attempts(tmp_path, monkeypatch, provider, api):
    requests, waits = [], []
    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) < 3:
            raise httpx.ConnectError('temporary disconnect', request=request)
        return httpx.Response(200, json=reply(api))
    http = httpx.Client(transport=httpx.MockTransport(handle))
    if provider == 'bedrock':
        client = AnthropicBedrock(aws_access_key='offline', aws_secret_key='offline',
            aws_region='us-east-2', http_client=http, max_retries=0)
    else:
        client = (Anthropic if api == 'messages' else OpenAI)(api_key='offline', http_client=http, max_retries=0)
    monkeypatch.setenv('CEOBENCH_SIMULATOR_USAGE_LOG', str(tmp_path / 'usage.jsonl'))
    conn = init_database(':memory:')
    customer = CustomerSimulator(client if api != 'messages' else None, conn, BenchmarkConfig())
    if api == 'messages':
        setattr(customer, '_bedrock_client' if provider == 'bedrock' else '_anthropic_client',
                customer.usage_recorder.attach(client))
    monkeypatch.setattr(customer.usage_recorder.lifecycle, 'sleep', waits.append)
    try:
        assert customer.complete_text(provider=provider, model='test-model', user='same user input',
            system='same system input', max_tokens=30) == ('hello', 10, 2)
        assert waits == [5, 10]
        assert requests[0] == requests[1] == requests[2]
        rows = [json.loads(line) for line in (tmp_path / 'usage.jsonl').read_text().splitlines()]
        assert len({row['call_id'] for row in rows}) == 1
        assert [row['event'] for row in rows] == ['request', 'http_request', 'http_error',
            'http_request', 'http_error', 'http_request', 'http_response', 'response']
        assert customer.usage_recorder.summary['calls'] == 1
        assert customer.usage_recorder.summary['http_attempts'] == 3
        assert customer.usage_recorder.summary['failed_http_attempts'] == 2
        assert rows[-1]['error'] is None
    finally:
        client.close()
        conn.close()


def test_shared_deadline_includes_all_backoff_and_does_not_reset(tmp_path, monkeypatch):
    clock, calls, waits = [100.0], [], []
    recorder = ModelUsage(tmp_path / 'usage.jsonl', 'ceo')
    monkeypatch.setattr('saas_bench.model_usage.MODEL_REQUEST_SECONDS', 45)
    monkeypatch.setattr('saas_bench.model_usage.time.monotonic', lambda: clock[0])
    def wait(seconds):
        waits.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(recorder.lifecycle, 'sleep', wait)
    def invoke():
        calls.append(clock[0])
        raise httpx.ConnectError('offline')
    with pytest.raises(RunCancelled, match='model_request_deadline'):
        recorder.call('chat', {'model': 'test-model'}, invoke, day=21)
    assert calls == [100, 105, 115, 135]
    assert waits == [5, 10, 20, 10]
    assert recorder.summary['calls'] == recorder.summary['errors'] == 1
    assert recorder.summary['known']['input_tokens'] is None


def test_deadline_includes_call_lock_wait(monkeypatch):
    clock = [100.0]
    recorder = ModelUsage(None, 'ceo')
    monkeypatch.setattr('saas_bench.model_usage.MODEL_REQUEST_SECONDS', .25)
    monkeypatch.setattr('saas_bench.model_usage.time.monotonic', lambda: clock[0])
    class Locked:
        def acquire(self, timeout):
            clock[0] += timeout
            return False
    recorder.call_lock = Locked()
    with pytest.raises(RunCancelled, match='model_request_deadline'):
        recorder.call('chat', {}, lambda: pytest.fail('request must not start'))
    assert clock[0] == pytest.approx(100.25)
    assert recorder.summary['calls'] == 0


@pytest.mark.parametrize('role', ['ceo', 'growth', 'ops_finance', 'simulator'])
def test_all_roles_share_900_second_exponential_outage_budget(monkeypatch, role):
    from saas_bench.model_usage import MODEL_REQUEST_SECONDS
    assert MODEL_REQUEST_SECONDS == 900
    clock, waits, calls = [100.0], [], []
    recorder = ModelUsage(None, role)
    monkeypatch.setattr('saas_bench.model_usage.time.monotonic', lambda: clock[0])
    def wait(seconds):
        waits.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(recorder.lifecycle, 'sleep', wait)
    def invoke():
        calls.append(clock[0])
        raise httpx.ConnectError('offline')
    with pytest.raises(RunCancelled, match='model_request_deadline'):
        recorder.call('chat', {}, invoke)
    assert waits == [5, 10, 20, 40, 80, 160, 180, 180, 180, 45]
    assert calls == [100, 105, 115, 135, 175, 255, 415, 595, 775, 955]
    assert clock[0] == 1000
    assert recorder.summary['calls'] == recorder.summary['errors'] == 1


def test_cancellation_during_backoff_prevents_another_attempt(monkeypatch):
    recorder = ModelUsage(None, 'ceo')
    calls = []
    monkeypatch.setattr(recorder.lifecycle, 'sleep', lambda _: recorder.lifecycle.cancel('supervisor_cancelled'))
    def invoke():
        calls.append(1)
        raise httpx.ConnectError('offline')
    with pytest.raises(RunCancelled, match='supervisor_cancelled'):
        recorder.call('chat', {}, invoke)
    assert calls == [1]


def test_remaining_deadline_caps_sdk_phase_timeouts_and_permanent_errors(tmp_path, monkeypatch):
    timeouts = []
    def handle(request):
        timeouts.append(request.extensions['timeout'])
        return httpx.Response(401, json={'error': {'message': 'unauthorized'}})
    monkeypatch.setattr('saas_bench.model_usage.MODEL_REQUEST_SECONDS', 2)
    recorder = ModelUsage(tmp_path / 'usage.jsonl', 'ceo')
    client = recorder.attach(OpenAI(api_key='offline', max_retries=2,
        http_client=httpx.Client(transport=httpx.MockTransport(handle))))
    request = dict(model='test-model', messages=[{'role': 'user', 'content': 'unchanged'}])
    try:
        with pytest.raises(Exception) as error:
            recorder.call('chat', request, lambda: client.chat.completions.create(**request))
        assert error.value.status_code == 401
        assert client.max_retries == 2
        assert len(timeouts) == 1
        assert all(0 < value <= 2 for value in timeouts[0].values())
    finally:
        client.close()


@pytest.mark.parametrize('status', [408, 409, 429, 500, 503])
def test_retryable_http_status_keeps_one_logical_call(monkeypatch, status):
    attempts = []
    def handle(request):
        attempts.append(request)
        return (httpx.Response(status, json={'error': {'message': 'temporary'}})
                if len(attempts) == 1 else httpx.Response(200, json=reply('chat')))
    recorder = ModelUsage(None, 'ceo')
    monkeypatch.setattr(recorder.lifecycle, 'sleep', lambda _: None)
    client = recorder.attach(OpenAI(api_key='offline', max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle))))
    request = dict(model='test-model', messages=[{'role': 'user', 'content': 'same'}])
    try:
        response = recorder.call('chat', request, lambda: client.chat.completions.create(**request))
        assert response.choices[0].message.content == 'hello'
        assert recorder.summary['calls'] == 1
        assert recorder.summary['http_attempts'] == 2
        assert recorder.summary['failed_http_attempts'] == 1
    finally:
        client.close()


def test_threaded_agent_uses_shared_deadline(monkeypatch):
    clock, failures, requests = [100.0], [], []
    monkeypatch.setattr('saas_bench.model_usage.MODEL_REQUEST_SECONDS', 15)
    monkeypatch.setattr('saas_bench.model_usage.time.monotonic', lambda: clock[0])
    def handle(request):
        requests.append(request)
        raise httpx.ConnectError('offline', request=request)
    client = OpenAI(api_key='offline', max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    agent = BashAgent(get_bash_agent_tool_descriptions(), client, model='test-model', system_prompt='same')
    def wait(seconds):
        clock[0] += seconds
    monkeypatch.setattr(agent.usage_recorder.lifecycle, 'sleep', wait)
    request = dict(model='test-model', messages=[{'role': 'user', 'content': 'same'}])
    def invoke():
        try:
            agent._request_model('chat', request, lambda: client.chat.completions.create(**request))
        except BaseException as error:
            failures.append(error)
    worker = threading.Thread(target=invoke)
    try:
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert len(requests) == 2
        assert len(failures) == 1 and isinstance(failures[0], RunCancelled)
        assert str(failures[0]) == 'model_request_deadline'
        assert agent.usage_recorder.summary['calls'] == 1
    finally:
        client.close()


def test_legacy_agent_live_path_uses_shared_retry(monkeypatch):
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            raise httpx.ConnectError('temporary disconnect', request=request)
        return httpx.Response(200, json=reply('responses'))
    client = OpenAI(api_key='offline', max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    conn = init_database(':memory:')
    config = BenchmarkConfig(agent_llm_model='test-model')
    agent = AgentLLM(client, conn, config, CostTracker(conn, config))
    monkeypatch.setattr(agent.usage_recorder.lifecycle, 'sleep', lambda _: None)
    try:
        response, _ = agent.get_action({'day': 0, 'cash': 1000}, [], 0)
        assert response.output_text == 'hello'
        assert requests[0] == requests[1]
        assert agent.usage_recorder.summary['calls'] == 1
        assert agent.usage_recorder.summary['http_attempts'] == 2
    finally:
        client.close()
        conn.close()
