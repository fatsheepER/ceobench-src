import errno
import json
import socket
import ssl
import threading

import httpx
import pytest
from anthropic import Anthropic, AnthropicBedrock
from openai import OpenAI, APIConnectionError

from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from saas_bench.config import BenchmarkConfig
from saas_bench.customer_llm import CustomerSimulator
from saas_bench.database import init_database
from saas_bench.llm import AgentLLM, CostTracker
from saas_bench.model_usage import (ModelUsage, _ATTEMPT, _Stream, check_model_call, is_network_error,
                                   physical_usage_summary, summarize_usage_log, usage_delta)
from saas_bench.run_lifecycle import RunCancelled
from test_preflight_usage import reply


@pytest.mark.parametrize('physical_http', [False, True])
def test_physical_reporting_counts_only_receipted_retry_failures(tmp_path, monkeypatch, physical_http):
    recorder = ModelUsage(tmp_path / 'usage.jsonl', 'ceo')
    monkeypatch.setattr(recorder.lifecycle, 'sleep', lambda _: None)
    previous = json.loads(json.dumps(recorder.summary))
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(503, json={'error': {'message': 'temporary'}}) if len(requests) == 1 else httpx.Response(200, json=reply('chat'))
    client = recorder.attach(OpenAI(api_key='offline', max_retries=2,
        http_client=httpx.Client(transport=httpx.MockTransport(handle))))
    request = dict(model='test-model', messages=[dict(role='user', content='same')])
    invocations = []
    def invoke():
        invocations.append(1)
        if physical_http:
            return client.chat.completions.create(**request)
        if len(invocations) == 1:
            raise httpx.ConnectError('local invocation without an HTTP request')
        return reply('chat')
    try:
        recorder.call('chat', request, invoke, day=7)
    finally:
        client.close()
    rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    failed_receipts = [row for row in rows if row['event'] == 'http_response' and row['status'] == 503]
    assert len(invocations) == 2 and recorder.summary['calls'] == 1
    view = physical_usage_summary(recorder.summary)
    delta = usage_delta(previous, recorder.summary)
    logged = summarize_usage_log(recorder.path, start_day=7, end_day=14)
    assert len(failed_receipts) == int(physical_http)
    assert view['calls'] == delta['calls'] == logged['calls'] == 1 + len(failed_receipts)
    assert view['missing']['input_tokens'] == delta['missing']['input_tokens'] == logged['missing']['input_tokens'] == len(failed_receipts)
    assert view['known']['input_tokens'] == logged['known']['input_tokens'] == 10
    assert view['known']['cache_creation_tokens'] is None
    assert view['known_cost_usd'] is logged['known_cost_usd'] is None
    assert view['missing_cost'] == logged['missing_cost'] == view['calls']


@pytest.mark.parametrize('provider,api', [('opencode', 'chat'), ('deepseek', 'chat'),
    ('openai', 'responses'), ('anthropic', 'messages'), ('bedrock', 'messages')])
def test_customer_network_retries_preserve_request_and_sdk_policy(tmp_path, monkeypatch, provider, api):
    requests, waits = [], []
    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) < 5:
            raise httpx.ConnectError('temporary disconnect', request=request)
        return httpx.Response(200, json=reply(api))
    http = httpx.Client(transport=httpx.MockTransport(handle))
    client = (AnthropicBedrock(aws_access_key='offline', aws_secret_key='offline', aws_region='us-east-2',
        http_client=http, max_retries=2) if provider == 'bedrock' else
        (Anthropic if api == 'messages' else OpenAI)(api_key='offline', http_client=http, max_retries=2))
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
        assert len(requests) == 5 and all(request == requests[0] for request in requests)
        assert waits == [10, 20, 40, 80] and client.max_retries == 2
        rows = [json.loads(line) for line in (tmp_path / 'usage.jsonl').read_text().splitlines()]
        assert len({row['call_id'] for row in rows}) == 1
        assert customer.usage_recorder.summary['calls'] == 1
        assert customer.usage_recorder.summary['http_attempts'] == 5
        assert customer.usage_recorder.summary['failed_http_attempts'] == 4
        assert rows[-1]['error'] is None and rows[-1]['model_attempts'] == 5
    finally:
        client.close()
        conn.close()


@pytest.mark.parametrize('role', ['ceo', 'growth', 'ops_finance', 'simulator'])
@pytest.mark.parametrize('error_type', [httpx.ConnectError, httpx.ReadTimeout, ssl.SSLError,
                                       socket.gaierror, ConnectionResetError])
def test_all_roles_retry_network_errors_exactly_five_times(monkeypatch, role, error_type):
    recorder, waits, requests = ModelUsage(None, role), [], []
    monkeypatch.setattr(recorder.lifecycle, 'sleep', waits.append)
    def handle(request):
        requests.append(request)
        raise error_type('offline')
    client = recorder.attach(OpenAI(api_key='offline', max_retries=2,
        http_client=httpx.Client(transport=httpx.MockTransport(handle))))
    request = dict(model='test-model', messages=[{'role': 'user', 'content': 'same'}])
    try:
        with pytest.raises(RunCancelled, match='model_request_failed'):
            recorder.call('chat', request, lambda: client.chat.completions.create(**request))
        assert len(requests) == 5 and waits == [10, 20, 40, 80]
        assert recorder.summary['calls'] == recorder.summary['errors'] == 1
        assert client.max_retries == 2 and recorder.lifecycle.reason is None
        assert len(client._client.event_hooks['request']) == 1
    finally:
        client.close()


@pytest.mark.parametrize('cause', [ValueError('invalid'), httpx.LocalProtocolError('invalid headers'),
    httpx.UnsupportedProtocol('unsupported URL'), RuntimeError('local processing'),
    OSError(errno.ENOSPC, 'disk full')])
def test_generic_sdk_connection_error_does_not_hide_a_nonnetwork_cause(cause):
    wrapped = APIConnectionError(request=httpx.Request('POST', 'https://offline.invalid'))
    wrapped.__cause__ = cause
    assert not is_network_error(wrapped)


@pytest.mark.parametrize('status', [400, 401, 403, 408, 409, 429, 500, 503])
def test_http_errors_retry_temporary_rejections_and_preserve_cause(monkeypatch, status):
    requests, waits = [], []
    recorder = ModelUsage(None, 'ceo')
    monkeypatch.setattr(recorder.lifecycle, 'sleep', waits.append)
    def handle(request):
        requests.append(request)
        return httpx.Response(status, json={'error': {'message': 'provider rejection'}})
    client = recorder.attach(OpenAI(api_key='offline', max_retries=2,
        http_client=httpx.Client(transport=httpx.MockTransport(handle))))
    request = dict(model='test-model', messages=[{'role': 'user', 'content': 'same'}])
    try:
        with pytest.raises(RunCancelled, match='model_request_failed') as error:
            recorder.call('chat', request, lambda: client.chat.completions.create(**request))
        assert error.value.__cause__.status_code == status
        retryable = status in (408, 409, 429) or status >= 500
        assert len(requests) == (5 if retryable else 1) and client.max_retries == 2
        assert waits == ([10, 20, 40, 80] if retryable else [])
        assert recorder.summary['failed_http_attempts'] == len(requests)
        assert requests[0].extensions['timeout']['read'] >= 600
    finally:
        client.close()


def test_each_silent_attempt_timeout_wakes_own_socket_without_cancelling_peer(monkeypatch):
    recorder, waits, calls = ModelUsage(None, 'ceo'), [], []
    monkeypatch.setattr('saas_bench.model_usage.MODEL_ATTEMPT_SECONDS', .01)
    monkeypatch.setattr(recorder.lifecycle, 'sleep', waits.append)
    def invoke():
        calls.append(1)
        channel, peer = socket.socketpair()
        with channel, peer:
            _ATTEMPT.get()._sockets.add(channel)
            assert channel.recv(1) == b''
    channel, peer = socket.socketpair()
    with channel, peer:
        recorder.lifecycle._sockets.add(channel)
        with pytest.raises(RunCancelled, match='model_request_failed') as error:
            recorder.call('chat', {}, invoke)
        assert isinstance(error.value.__cause__, TimeoutError)
        assert calls == [1] * 5 and waits == [10, 20, 40, 80]
        assert recorder.lifecycle.reason is None
        channel.sendall(b'peer continues')
        assert peer.recv(32) == b'peer continues'


def test_first_response_at_599_seconds_allows_stream_beyond_600(monkeypatch):
    clock, received = [100.0], []
    monkeypatch.setattr('saas_bench.model_usage.time.monotonic', lambda: clock[0])
    recorder = ModelUsage(None, 'simulator')
    class ActiveStream(httpx.SyncByteStream):
        def __iter__(self):
            clock[0] += 599
            yield b'first reply'
            clock[0] += 1200
            yield b'continuing reply'
    def invoke():
        stream = _Stream(ActiveStream(), lambda body, error: received.append((body, error)),
                         check=check_model_call)
        assert list(stream) == [b'first reply', b'continuing reply']
        assert _ATTEMPT.get().received_reply
        return reply('chat')
    assert recorder.call('chat', {}, invoke)['model'] == 'test-model'
    assert clock[0] == 1899 and received == [(b'first replycontinuing reply', None)]


@pytest.mark.parametrize('phase', ['first_body', 'reused_body', 'reused_headers'])
def test_sdk_watchdog_owns_response_and_reused_connection(monkeypatch, phase):
    clock, requests, waits, expired = [100.0], [], [], []
    monkeypatch.setattr('saas_bench.model_usage.time.monotonic', lambda: clock[0])
    class ManualTimer:
        ident = None
        def __init__(self, seconds, callback):
            assert seconds == 600
            self.callback, self.cancelled = callback, False
        def start(self):
            pass
        def cancel(self):
            self.cancelled = True
        def fire(self):
            assert not self.cancelled
            self.callback()
    monkeypatch.setattr('saas_bench.model_usage.threading.Timer', ManualTimer)
    recorder, peer_recorder = ModelUsage(None, 'ceo'), ModelUsage(None, 'growth')
    monkeypatch.setattr(recorder.lifecycle, 'sleep', waits.append)
    warm = phase != 'first_body'
    channel, remote = socket.socketpair()
    peer_channel, peer_remote = socket.socketpair()
    class NetworkStream:
        def get_extra_info(self, name):
            assert name == 'socket'
            return channel
    def silent_read():
        attempt = _ATTEMPT.get()
        assert channel in attempt._sockets
        assert peer_channel not in attempt._sockets
        clock[0] = attempt.deadline
        attempt.timer.fire()
        assert channel.recv(1) == b''
        expired.append(attempt.reason)
    class SilentBody(httpx.SyncByteStream):
        def __iter__(self):
            silent_read()
            yield b''
    def handle(request):
        requests.append(request)
        if warm and len(requests) == 1:
            return httpx.Response(200, stream=httpx.ByteStream(json.dumps(reply('chat')).encode()),
                extensions={'network_stream': NetworkStream()})
        if len(requests) > int(warm) + 1:
            raise httpx.ConnectError('closed pooled connection', request=request)
        clock[0] += 599
        if phase == 'reused_headers':
            silent_read()
            raise httpx.ReadError('headers never arrived', request=request)
        return httpx.Response(200, stream=SilentBody(), extensions={'network_stream': NetworkStream()})
    client = recorder.attach(OpenAI(api_key='offline', max_retries=2,
        http_client=httpx.Client(transport=httpx.MockTransport(handle))))
    request = dict(model='test-model', messages=[{'role': 'user', 'content': 'same'}])
    with channel, remote, peer_channel, peer_remote:
        peer_recorder.lifecycle._sockets.add(peer_channel)
        try:
            if warm:
                assert recorder.call('chat', request, lambda: client.chat.completions.create(**request)).model == 'test-model'
                assert channel in client._client._ceobench_model_sockets
            with pytest.raises(RunCancelled, match='model_request_failed'):
                recorder.call('chat', request, lambda: client.chat.completions.create(**request))
            assert len(requests) == int(warm) + 5 and waits == [10, 20, 40, 80]
            assert expired == ['model_attempt_timeout']
            assert recorder.lifecycle.reason is None and peer_recorder.lifecycle.reason is None
            peer_channel.sendall(b'peer continues')
            assert peer_remote.recv(32) == b'peer continues'
        finally:
            client.close()


def test_operator_cancellation_during_backoff_prevents_another_request(monkeypatch):
    recorder, calls = ModelUsage(None, 'ceo'), []
    monkeypatch.setattr(recorder.lifecycle, 'sleep', lambda _: recorder.lifecycle.cancel('supervisor_cancelled'))
    def invoke():
        calls.append(1)
        raise httpx.ConnectError('offline')
    with pytest.raises(RunCancelled, match='supervisor_cancelled'):
        recorder.call('chat', {}, invoke)
    assert calls == [1]


def test_threaded_agent_retries_five_network_requests(monkeypatch):
    failures, requests = [], []
    def handle(request):
        requests.append(request)
        raise httpx.ConnectError('offline', request=request)
    client = OpenAI(api_key='offline', max_retries=2,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    agent = BashAgent(get_bash_agent_tool_descriptions(), client, model='test-model', system_prompt='same')
    monkeypatch.setattr(agent.usage_recorder.lifecycle, 'sleep', lambda _: None)
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
        assert not worker.is_alive() and len(requests) == 5
        assert len(failures) == 1 and failures[0].reason == 'model_request_failed'
        assert 'ConnectError: offline' in str(failures[0])
    finally:
        client.close()


def test_legacy_agent_live_path_uses_shared_network_retry(monkeypatch):
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            raise httpx.ConnectError('temporary disconnect', request=request)
        return httpx.Response(200, json=reply('responses'))
    client = OpenAI(api_key='offline', max_retries=2,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    conn = init_database(':memory:')
    config = BenchmarkConfig(agent_llm_model='test-model')
    agent = AgentLLM(client, conn, config, CostTracker(conn, config))
    monkeypatch.setattr(agent.usage_recorder.lifecycle, 'sleep', lambda _: None)
    try:
        response, _ = agent.get_action({'day': 0, 'cash': 1000}, [], 0)
        assert response.output_text == 'hello' and requests[0] == requests[1]
        assert agent.usage_recorder.summary['calls'] == 1
        assert agent.usage_recorder.summary['http_attempts'] == 2
    finally:
        client.close()
        conn.close()
