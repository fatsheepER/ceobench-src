"""OpenDesign simulator terminal proof through local HTTP and the public SDK."""
import copy
import json

import pytest

from saas_bench.config import BenchmarkConfig
from saas_bench.customer_llm import CustomerSimulator
from saas_bench.database import init_database
from saas_bench.run_lifecycle import RunCancelled
from saas_bench.team_batch import checked_client
from test_opendesign import MODEL, SECRET, completion, endpoint, model_config


@pytest.fixture
def simulator_factory(tmp_path, monkeypatch, endpoint):
    url, _, _ = endpoint
    monkeypatch.setenv('OPENDESIGN_API_KEY', SECRET)
    monkeypatch.setenv('CEOBENCH_INVALID_GENERATION_ATTEMPTS', '4')
    clients, connections = [], []
    def create(provider='opendesign'):
        config = BenchmarkConfig(simulator_model_config=dict(model_config(url, 'none'), provider=provider))
        client = checked_client('simulator', 'offline-terminal-proof', model_config=config.simulator_model_config)
        clients.append(client)
        conn = init_database(':memory:')
        connections.append(conn)
        simulator = CustomerSimulator(client, conn, config)
        simulator.usage_recorder.path = tmp_path / (provider + '.jsonl')
        waits = []
        monkeypatch.setattr(simulator.usage_recorder.lifecycle, 'sleep', waits.append)
        return simulator, waits
    try:
        yield create
    finally:
        for client in clients:
            client.close()
        for conn in connections:
            conn.close()
        assert all(client._client.is_closed for client in clients)


def invalid_completion(fault, usage):
    body = completion(tool=fault == 'tools', usage=usage)
    choice = body['choices'][0]
    choice['finish_reason'] = 'stop'
    choice['message']['content'] = 'partial customer reply'
    if fault in ('length', 'content_filter'):
        choice['finish_reason'] = fault
    elif fault == 'missing_finish':
        choice.pop('finish_reason')
    elif fault == 'refusal':
        choice['message']['refusal'] = 'Provider refused the request.'
    elif fault == 'multiple_choices':
        extra = copy.deepcopy(choice)
        extra['index'] = 1
        body['choices'].append(extra)
    elif fault in ('empty', 'blank'):
        choice['message']['content'] = '' if fault == 'empty' else ' \n\t '
    return body


def rows(simulator, event):
    return [row for line in simulator.usage_recorder.path.read_text().splitlines()
        if (row := json.loads(line))['event'] == event]


FAULTS = ['length', 'content_filter', 'missing_finish', 'refusal', 'multiple_choices', 'empty', 'blank', 'tools']


@pytest.mark.parametrize('usage', [False, True])
@pytest.mark.parametrize('fault', FAULTS)
def test_simulator_rejects_unsuccessful_terminal_before_counting_success(
        endpoint, simulator_factory, fault, usage):
    _, requests, responses = endpoint
    bad = invalid_completion(fault, usage)
    responses.extend([bad, completion(usage=usage)])
    simulator, waits = simulator_factory()
    result = simulator.complete_text(provider='opendesign', model=MODEL, system='fixture customer',
        user='Write the complete customer reply.', max_tokens=321)
    assert result == ('hello', 10 if usage else None, 2 if usage else None)
    assert len(requests) == 2 and waits == []
    first, corrected = [row['body'] for row in requests]
    assert first['max_tokens'] == 1234 and first['reasoning_effort'] == 'none'
    assert not {'stream', 'stream_options', 'thinking'} & first.keys()
    assert corrected['messages'][:-1] == first['messages']
    assert 'previous response was invalid' in corrected['messages'][-1]['content']
    logical = rows(simulator, 'response')
    assert [row['error'] for row in logical] == ['InvalidModelResponse', None]
    assert logical[0]['response'] == bad
    assert [row['model_attempts'] for row in logical] == [1, 1]
    assert all(row['usage']['input_tokens'] == (10 if usage else None) for row in logical)
    assert all(row['cost_usd'] is None for row in logical)
    assert [json.loads(row['body']) for row in rows(simulator, 'http_response')] == [bad, completion(usage=usage)]
    assert [row['body'] for row in rows(simulator, 'http_request')] == [first, corrected]
    summary = simulator.usage_recorder.summary
    assert summary['known']['input_tokens'] == (20 if usage else None)
    assert summary['known_cost_usd'] is None and summary['missing_cost'] == 2
    assert SECRET not in simulator.usage_recorder.path.read_text()


@pytest.mark.parametrize('usage', [False, True])
@pytest.mark.parametrize('fault', FAULTS)
def test_simulator_four_invalid_terminals_hold_and_next_call_starts_fresh(
        endpoint, simulator_factory, fault, usage):
    _, requests, responses = endpoint
    bad = invalid_completion(fault, usage)
    responses.extend([bad] * 4 + [completion(usage=usage)])
    simulator, waits = simulator_factory()
    with pytest.raises(RunCancelled, match='model_attempt_limit'):
        simulator.complete_text(provider='opendesign', model=MODEL, user='original task', max_tokens=321)
    assert len(requests) == 4 and waits == []
    logical = rows(simulator, 'response')
    assert len(logical) == 4
    assert all(row['error'] == 'InvalidModelResponse' and row['model_attempts'] == 1 for row in logical)
    assert all(row['response'] == bad and row['usage']['input_tokens'] == (10 if usage else None) for row in logical)
    assert all(row['cost_usd'] is None for row in logical)
    assert simulator.complete_text(provider='opendesign', model=MODEL, user='new task', max_tokens=321) == (
        'hello', 10 if usage else None, 2 if usage else None)
    assert len(requests) == 5 and waits == []
    assert requests[-1]['body']['messages'] == [dict(role='user', content='new task')]
    assert [row['error'] for row in rows(simulator, 'response')] == ['InvalidModelResponse'] * 4 + [None]
    summary = simulator.usage_recorder.summary
    assert summary['known']['input_tokens'] == (50 if usage else None)
    assert summary['known_cost_usd'] is None and summary['missing_cost'] == 5


@pytest.mark.parametrize('provider', ['deepseek', 'opencode'])
def test_other_chat_simulators_keep_existing_terminal_behavior(endpoint, simulator_factory, provider):
    _, requests, responses = endpoint
    responses.append(invalid_completion('length', True))
    simulator, waits = simulator_factory(provider)
    assert simulator.complete_text(provider=provider, model=MODEL, user='fixture task', max_tokens=321) == (
        'partial customer reply', 10, 2)
    assert len(requests) == 1 and waits == []
    logical, = rows(simulator, 'response')
    assert logical['error'] is None


@pytest.mark.parametrize('usage', [False, True])
def test_http_200_provider_error_holds_once_and_next_independent_call_continues(
        endpoint, simulator_factory, usage):
    _, requests, responses = endpoint
    error = {'error': {'code': 'insufficient_balance', 'type': 'insufficient_balance',
        'message': '[code=insufficient_balance] insufficient wallet balance'}}
    if usage:
        error['usage'] = completion()['usage']
    responses.extend([error, completion(usage=usage)])
    simulator, waits = simulator_factory()
    with pytest.raises(RunCancelled, match='model_request_failed') as failure:
        simulator.complete_text(provider='opendesign', model=MODEL, user='original task', max_tokens=321)
    assert 'ValueError: Provider error:' in failure.value.detail
    assert 'insufficient_balance' in failure.value.detail
    assert len(requests) == 1 and waits == []
    logical, = rows(simulator, 'response')
    assert logical['error'] == 'RunCancelled' and logical['model_attempts'] == 1
    assert logical['response'] == error and logical['usage']['input_tokens'] == (10 if usage else None)
    assert logical['cost_usd'] is None and 'insufficient_balance' in logical['last_model_error']
    http, = rows(simulator, 'http_response')
    assert http['status'] == 200 and json.loads(http['body']) == error
    assert 'insufficient_balance' in http['error']
    assert simulator.complete_text(provider='opendesign', model=MODEL, user='new task', max_tokens=321) == (
        'hello', 10 if usage else None, 2 if usage else None)
    assert len(requests) == 2 and waits == []
    assert requests[-1]['body']['messages'] == [dict(role='user', content='new task')]
    assert [row['error'] for row in rows(simulator, 'response')] == ['RunCancelled', None]
    assert simulator.usage_recorder.summary['known']['input_tokens'] == (20 if usage else None)
    assert simulator.usage_recorder.summary['known_cost_usd'] is None
