"""OpenDesign gateway shapes through the real SDK, HTTP transport and ModelUsage."""
import copy
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading

import pytest

from saas_bench.agents.bash_agent.agent import BashAgent, FinalText
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from saas_bench.environment import Action
from saas_bench.model_usage import ModelUsage
from saas_bench.run_lifecycle import RunCancelled
from saas_bench.team_batch import checked_client
from test_opendesign import MODEL, SECRET, completion, model_config, stream_chunks


REASONING = 'The user wants me to call lookup_metric with key "probe_value".'


@dataclass
class GatewayReply:
    body: dict
    chunks: list | None = None
    done: bool = True
    disconnect: bool = False


@pytest.fixture
def gateway():
    requests, replies, payloads = [], [], []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            reply = replies.pop(0) if replies else GatewayReply(completion())
            if body.get('stream'):
                chunks = reply.chunks if reply.chunks is not None else stream_chunks(reply.body)
                payload = ''.join('data: ' + (chunk if isinstance(chunk, str) else json.dumps(chunk))
                    + '\n\n' for chunk in chunks)
                if reply.done:
                    payload += 'data: [DONE]\n\n'
                content_type = 'text/event-stream'
            else:
                payload = json.dumps(reply.body)
                content_type = 'application/json'
            payloads.append(payload)
            encoded = payload.encode()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(encoded) + (100 if reply.disconnect else 0)))
            self.end_headers()
            self.wfile.write(encoded)
            self.wfile.flush()
            if reply.disconnect:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    serving = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01))
    serving.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1/', requests, replies, payloads
    finally:
        server.shutdown()
        server.server_close()
        serving.join(3)
        assert not serving.is_alive()


@pytest.fixture
def agent_factory(tmp_path, monkeypatch, gateway):
    monkeypatch.setenv('OPENDESIGN_API_KEY', SECRET)
    url, _, _, _ = gateway
    clients = []
    def create(name='agent', tools=None):
        config = model_config(url)
        client = checked_client('ceo', 'offline-opendesign-transport', model_config=config)
        clients.append(client)
        agent = BashAgent(tools or get_bash_agent_tool_descriptions(), client, model=MODEL,
            system_prompt='fixture instructions', workspace_path=tmp_path,
            allow_final_text=True, reasoning_effort='high', model_config=config,
            usage_recorder=ModelUsage(tmp_path / (name + '.jsonl'), 'ceo'))
        agent._snapshot_path = tmp_path / 'conversation.json'
        return agent
    try:
        yield create
    finally:
        for client in clients:
            client.close()
        assert all(client._client.is_closed for client in clients)


def metric_completion():
    body = completion(tool=True)
    message = body['choices'][0]['message']
    message['content'] = "I'll look up that metric for you."
    message['reasoning'] = REASONING
    message['tool_calls'][0]['function'] = dict(name='lookup_metric', arguments='{"key": "probe_value"}')
    return body


def rows(agent, event):
    return [row for line in agent.usage_recorder.path.read_text().splitlines()
        if (row := json.loads(line))['event'] == event]


def test_captured_split_choice_uses_complete_stream_and_restores_exact_reasoning(
        tmp_path, gateway, agent_factory):
    # Representative public fields of the 2026-11-09 AMR capture: commentary in
    # choice 0 without finish_reason, actual tool call in choice 1. SSE uses index 0.
    _, requests, replies, payloads = gateway
    streamed = metric_completion()
    split = copy.deepcopy(streamed)
    message = split['choices'][0]['message']
    split['choices'] = [dict(index=0, message=dict(role='assistant', content=message['content'])),
        dict(index=1, finish_reason='tool_calls', message=dict(role='assistant', tool_calls=message['tool_calls']))]
    replies.append(GatewayReply(split, chunks=stream_chunks(streamed)))
    tools = [dict(name='lookup_metric', description='Return an isolated test metric.',
        parameters=dict(type='object', properties={'key': {'type': 'string'}}, required=['key']))]
    first = agent_factory('first', tools)
    action = first.act('Read probe_value, then state its exact value.', 0, False, {'day': 0})
    assert action == Action(tool='lookup_metric', arguments={'key': 'probe_value'})
    assert len(requests) == 1
    assert requests[0]['stream'] and requests[0]['stream_options'] == {'include_usage': True}
    response, = rows(first, 'response')
    assert response['error'] is None and response['usage']['input_tokens'] == 10
    assert response['response']['choices'][0]['message']['reasoning'] == REASONING
    assert [row['body'] for row in rows(first, 'http_response')] == payloads
    assert payloads[0].endswith('data: [DONE]\n\n')

    first.record_tool_result('73')
    first._save_conversation_snapshot(strict=True)
    restored = agent_factory('restored', tools)
    assert restored.load_conversation_snapshot(restored._snapshot_path, required=True)
    final = completion()
    final['choices'][0]['message']['content'] = 'probe_value is 73.'
    replies.extend([GatewayReply(final), GatewayReply(final)])
    assert first.act('73', 0, False, {'day': 0}) == FinalText('probe_value is 73.')
    assert restored.act('73', 0, False, {'day': 0}) == FinalText('probe_value is 73.')
    assert requests[1] == requests[2]
    assistant = next(m for m in requests[2]['messages'] if m['role'] == 'assistant')
    assert assistant['reasoning_content'] == REASONING
    assert any(m['role'] == 'tool' and m['tool_call_id'] == 'read1' and m['content'] == '73'
        for m in requests[2]['messages'])
    assert rows(restored, 'http_request')[0]['body'] == requests[2]


@pytest.mark.parametrize('tool', [False, True])
@pytest.mark.parametrize('fault', ['length', 'content_filter', 'refusal', 'missing_finish', 'inconsistent_finish'])
def test_unsuccessful_terminal_never_accepts_text_or_executes_tool_batch(
        gateway, agent_factory, monkeypatch, tool, fault):
    _, requests, replies, payloads = gateway
    bad = metric_completion() if tool else completion()
    choice = bad['choices'][0]
    if fault == 'refusal':
        choice['message']['refusal'] = 'Provider refused this request.'
    elif fault == 'inconsistent_finish':
        choice['finish_reason'] = 'stop' if tool else 'tool_calls'
    else:
        choice['finish_reason'] = None if fault == 'missing_finish' else fault
    good = completion(tool=tool)
    replies.extend([GatewayReply(bad), GatewayReply(good)])
    agent = agent_factory()
    waits, callbacks = [], []
    monkeypatch.setattr(agent.usage_recorder.lifecycle, 'sleep', waits.append)
    agent.response_callback = lambda **kwargs: callbacks.append(kwargs)
    result = agent.act('fixture task', 0, False, {'day': 0})
    assert result == (Action(tool='read_file', arguments={'path': 'MEMORY.md'}) if tool else FinalText('hello'))
    assert len(requests) == 2 and waits == [] and len(callbacks) == 1
    assert not any(m['role'] == 'assistant' for m in requests[1]['messages'])
    logical = rows(agent, 'response')
    assert [row['error'] for row in logical] == ['InvalidModelResponse', None]
    assert logical[0]['usage']['input_tokens'] == 10
    assert logical[0]['response']['usage'] == bad['usage']
    assert agent.usage_recorder.summary['known']['input_tokens'] == 20
    assert agent.usage_recorder.summary['known_cost_usd'] is None
    assert [row['body'] for row in rows(agent, 'http_response')] == payloads
    invalid = json.loads(agent.usage_recorder.path.with_name('invalid-generations.jsonl').read_text())
    assert invalid['reason'] in ('stream_schema', 'completion_status')
    assert 'previous response was invalid' in requests[1]['messages'][-1]['content']


@pytest.mark.parametrize('fault', ['missing_done', 'disconnect'])
def test_incomplete_transport_retries_same_request_before_returning_action(
        gateway, agent_factory, monkeypatch, fault):
    _, requests, replies, payloads = gateway
    bad = metric_completion()
    replies.extend([GatewayReply(bad, done=False, disconnect=fault == 'disconnect'),
        GatewayReply(completion(tool=True))])
    agent = agent_factory()
    waits = []
    monkeypatch.setattr(agent.usage_recorder.lifecycle, 'sleep', waits.append)
    assert agent.act('fixture task', 0, False, {'day': 0}) == Action(
        tool='read_file', arguments={'path': 'MEMORY.md'})
    assert len(requests) == 2 and requests[0] == requests[1] and waits == [10]
    logical, = rows(agent, 'response')
    assert logical['error'] is None and logical['model_attempts'] == 2
    assert logical['retried_http_attempts'] == 1
    assert agent.usage_recorder.summary['failed_http_attempts'] == 1
    assert not agent.usage_recorder.path.with_name('invalid-generations.jsonl').exists()
    assert [row['body'] for row in rows(agent, 'http_response')] == payloads


@pytest.mark.parametrize('fault', ['malformed_event', 'missing_finish', 'unsupported_finish', 'multiple_choices'])
def test_completed_malformed_stream_reaches_finite_hold_and_next_call_continues(
        gateway, agent_factory, monkeypatch, fault):
    _, requests, replies, _ = gateway
    bad = metric_completion()
    if fault == 'missing_finish':
        bad['choices'][0].pop('finish_reason')
    elif fault == 'unsupported_finish':
        bad['choices'][0]['finish_reason'] = 'unexpected'
    elif fault == 'multiple_choices':
        extra = copy.deepcopy(bad['choices'][0])
        extra['index'] = 1
        bad['choices'].append(extra)
    chunks = stream_chunks(bad)
    if fault == 'malformed_event':
        chunks.insert(1, '{broken')
    replies.extend(GatewayReply(bad, chunks=chunks) for _ in range(4))
    agent = agent_factory()
    waits = []
    monkeypatch.setattr(agent.usage_recorder.lifecycle, 'sleep', waits.append)
    with pytest.raises(RunCancelled, match='model_attempt_limit'):
        agent.act('fixture task', 0, False, {'day': 0})
    assert len(requests) == 4 and waits == [] and not agent._pending_tool_calls
    assert not any(m.role == 'assistant' for m in agent.conversation)
    logical = rows(agent, 'response')
    assert len(logical) == 4
    assert all(row['error'] == 'InvalidModelResponse' and row['model_attempts'] == 1 for row in logical)
    assert all(row['usage']['input_tokens'] == 10 for row in logical)
    assert agent.act('fixture task', 0, False, {'day': 0}) == FinalText('hello')
    assert len(requests) == 5


def test_explicit_official_deepseek_config_keeps_contract_at_custom_endpoint(tmp_path, monkeypatch, gateway):
    url, requests, replies, _ = gateway
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'offline-official-key')
    config = dict(model_config(url), provider='deepseek', model='deepseek-flash',
        credential_env='DEEPSEEK_API_KEY', thinking='enabled')
    client = checked_client('ceo', 'offline-official-transport', model_config=config)
    try:
        replies.append(GatewayReply(completion(tool=True, reasoning_field='reasoning_content')))
        agent = BashAgent(get_bash_agent_tool_descriptions(), client, model=config['model'],
            workspace_path=tmp_path, reasoning_effort='high', model_config=config, max_output_tokens=1234,
            usage_recorder=ModelUsage(tmp_path / 'official.jsonl', 'ceo'))
        assert agent.act('fixture task', 0, False, {'day': 0}) == Action(
            tool='read_file', arguments={'path': 'MEMORY.md'})
        body, = requests
        assert body['model'] == 'deepseek-flash' and body['max_tokens'] == 1234
        assert body['reasoning_effort'] == 'high' and body['thinking'] == {'type': 'enabled'}
        assert not {'stream', 'stream_options', 'max_completion_tokens'} & body.keys()
        assert rows(agent, 'http_request')[0]['body'] == body
    finally:
        client.close()
    assert client._client.is_closed
