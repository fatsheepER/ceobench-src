"""Round-four wire contracts, independent simulator and permanent quota failures."""
import json
import httpx
import pytest
from openai import OpenAI

from saas_bench.model_usage import ModelUsage
from saas_bench.run_lifecycle import RunCancelled
from test_preflight_usage import reply
from test_preflight_team_batch import frozen


@pytest.mark.parametrize('code', ['credit_balance_exhausted', 'insufficient_quota', 'billing_hard_limit_reached',
                                  'GoUsageLimitError', 'insufficient_balance'])
def test_permanent_quota_holds_without_retry_and_next_call_continues(tmp_path, monkeypatch, code):
    requests, waits = [], []
    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            # Actual AMR wallet refusal: https://github.com/nexu-io/open-design/issues/5246
            body = ({'error': {'code': code, 'message': '[code=insufficient_balance] insufficient wallet balance',
                              'type': code}} if code == 'insufficient_balance' else
                    {'type': 'error', 'error': {'type': code, 'message': 'Go usage limit exceeded'},
                     'metadata': {'limitName': 'monthly'}} if code == 'GoUsageLimitError' else
                    {'error': {'code': code, 'type': code, 'message': 'quota exhausted'}})
            return httpx.Response(429, json=body)
        return httpx.Response(200, json=reply('responses'))
    client = OpenAI(api_key='offline-secret', max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    recorder = ModelUsage(tmp_path / 'usage.jsonl', 'ceo')
    recorder.attach(client)
    monkeypatch.setattr(recorder.lifecycle, 'sleep', waits.append)
    try:
        with pytest.raises(RunCancelled, match='model_request_failed'):
            recorder.call('responses', {'model': 'test-model'}, lambda: client.responses.create(model='test-model', input='test'))
        assert len(requests) == 1 and waits == []
        assert recorder.call('responses', {'model': 'test-model'}, lambda: client.responses.create(model='test-model', input='next')).output_text == 'hello'
        assert 'offline-secret' not in recorder.path.read_text()
    finally:
        client.close()


@pytest.mark.parametrize('code', ['rate_limit_exceeded', None])
def test_transient_429_keeps_existing_retry(tmp_path, monkeypatch, code):
    requests, waits = [], []
    def respond(request):
        requests.append(request)
        return (httpx.Response(429, json={'error': {'code': code, 'message': 'try later'}}) if len(requests) == 1
                else httpx.Response(200, json=reply('responses', usage=False)))
    client = OpenAI(api_key='offline', max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    recorder = ModelUsage(tmp_path / 'usage.jsonl', 'ceo')
    recorder.attach(client)
    monkeypatch.setattr(recorder.lifecycle, 'sleep', waits.append)
    try:
        recorder.call('responses', {'model': 'test-model'}, lambda: client.responses.create(model='test-model', input='next'))
        assert len(requests) == 2 and waits == [10]
        assert all(value is None for value in recorder.summary['known'].values())
        assert recorder.summary['known_cost_usd'] is None
    finally:
        client.close()


@pytest.mark.parametrize('profile', ['r4-d56', 'r4-flash-max-pf', 'r4-sol-high-pf'])
def test_frozen_contract_and_monitor_classification(profile):
    from saas_bench.team_models import round_four_models, model_issue, validate_profile_models
    from test_preflight_team_batch import PRICES
    import copy
    models = round_four_models(profile, PRICES)
    validate_profile_models(profile, models, PRICES)
    for role in ('ceo', 'growth', 'ops_finance', 'simulator'):
        config = models['simulator' if role == 'simulator' else 'agent']
        body = {'model': config['model']}
        if config['api'] == 'responses':
            body.update(reasoning={'effort': 'high'}, store=False, include=config['include'], max_output_tokens=32768)
        else:
            body.update(reasoning_effort=config['reasoning_effort'], thinking={'type': config['thinking']}, max_tokens=32768)
        row = dict(event='http_request', role=role, body=body,
            endpoint=config['base_url'] + ('responses' if config['api'] == 'responses' else 'chat/completions'))
        assert model_issue(row, models) is None
        assert model_issue(dict(row, body=dict(body, model='wrong-model')), models) == 'model_route_changed'
    wrong = copy.deepcopy(models)
    wrong['simulator']['model'] = 'gpt-6.1-sol'
    with pytest.raises(ValueError, match='simulator model'):
        validate_profile_models(profile, wrong, PRICES)
    wrong = copy.deepcopy(models)
    wrong['agent']['api_key'] = 'not-permitted'
    with pytest.raises(ValueError, match='credential values'):
        validate_profile_models(profile, wrong, PRICES)


@pytest.mark.parametrize('fault', ['disconnect', 'cancel'])
def test_sol_public_agent_real_http_disconnect_and_cancel(tmp_path, monkeypatch, fault):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import socket
    import threading
    from saas_bench import team_batch as batch
    from saas_bench.team_models import round_four_models
    from saas_bench.agents.bash_agent.agent import BashAgent, FinalText
    from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
    from test_preflight_team_batch import PRICES
    started, release = threading.Event(), threading.Event()
    requests, results, waits = [], [], []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            started.set()
            if fault == 'disconnect' and len(requests) == 1:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            if fault == 'cancel':
                release.wait(5)
            result = reply('responses', usage=False)
            result['model'] = 'gpt-6.1-sol'
            payload = json.dumps(result).encode()
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    serving = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01))
    serving.start()
    config = round_four_models('r4-sol-high-pf', PRICES)['agent']
    config.update(base_url=f'http://127.0.0.1:{server.server_port}/v1/', credential_env='R4_OFFLINE_KEY')
    monkeypatch.setenv('R4_OFFLINE_KEY', 'offline-private-value')
    client = batch.checked_client('ceo', 'offline-test', model_config=config)
    usage = ModelUsage(tmp_path / 'usage.jsonl', 'ceo')
    monkeypatch.setattr(usage.lifecycle, 'sleep', waits.append)
    agent = BashAgent(get_bash_agent_tool_descriptions(), client, model=config['model'], reasoning_effort='high',
        model_config=config, allow_final_text=True, usage_recorder=usage, workspace_path=tmp_path)
    def run():
        try:
            results.append(agent.act('task', 0, False, {'day': 0}))
        except BaseException as exc:
            results.append(exc)
    worker = threading.Thread(target=run)
    try:
        worker.start()
        assert started.wait(3)
        if fault == 'cancel':
            usage.lifecycle.cancel('supervisor_cancelled')
        worker.join(3)
        assert not worker.is_alive()
        if fault == 'cancel':
            assert isinstance(results[0], RunCancelled) and len(requests) == 1 and waits == []
            assert not release.is_set()  # client exits while server still owns its work
        else:
            assert results == [FinalText('hello')] and len(requests) == 2 and waits == [10]
            assert requests[0] == requests[1]
            assert all(value is None for value in usage.summary['known'].values())
        assert requests[-1]['model'] == config['model'] and requests[-1]['reasoning']['effort'] == 'high'
        assert all(tool['strict'] is False for tool in requests[-1]['tools'])
        assert 'offline-private-value' not in usage.path.read_text()
    finally:
        release.set()
        usage.lifecycle.cancel('test_cleanup')
        worker.join(3)
        client.close()
        server.shutdown()
        server.server_close()
        serving.join(3)
    assert client._client.is_closed and not serving.is_alive()


@pytest.mark.parametrize('group', ['git', 'pf'])
@pytest.mark.parametrize('code', ['GoUsageLimitError', 'insufficient_balance'])
def test_permanent_quota_real_driver_holds_once_and_manual_resume_continues(frozen, tmp_path, monkeypatch, group, code):
    from pathlib import Path
    from saas_bench import team_batch as batch
    from saas_bench.team_checkpoint import validate_snapshot
    from saas_bench.team_recovery import latest_checkpoint
    from test_checkpoint_driver import Driver, _finished, _jsonl

    monkeypatch.delenv('CEOBENCH_MODEL_REQUEST_ATTEMPTS', raising=False)
    monkeypatch.delenv('CEOBENCH_MODEL_RETRY_DELAYS', raising=False)
    monkeypatch.delenv('CEOBENCH_MODEL_ATTEMPT_SECONDS', raising=False)
    body = ({'error': {'code': code, 'message': '[code=insufficient_balance] insufficient wallet balance',
                      'type': code}} if code == 'insufficient_balance' else
            {'type': 'error', 'error': {'type': code, 'message': 'Go usage limit exceeded'},
             'metadata': {'limitName': 'monthly'}})
    driver = Driver(frozen, tmp_path, monkeypatch, budget_seconds=120)
    policy, entry = driver.policy('quota', group)
    driver.worker_options = dict(reject_run=entry['run_id'], reject_status=429, reject_body=body,
                                 record_retry_waits=True, max_requests=1)
    try:
        attempt, record, proof = driver.run(policy, entry, expected=3)
        assert record['status'] == 'paused' and record['hold']
        assert record['runtime_failure'] == 'model_request_failed'
        assert code in record['hold_detail']
        assert len(proof['wire']) == 1 and proof['retry_waits'] == []
        assert record['recovery_validation']['eligible']
        checkpoint = Path(proof['checkpoints']['ready'])
        _, state = validate_snapshot(checkpoint)
        assert state['day'] == 0 and state['stage'] == 'boundary'
        assert latest_checkpoint(entry, week=1) == checkpoint
        hold = batch.read(attempt / 'HOLD')
        assert hold['manual_resume_required'] and hold['reason'] == 'model_request_failed'
        terminal = batch.read(policy.parent / 'controller-terminal.json')[entry['run_id']]
        assert terminal['disposition'] == 'hold' and terminal['manual_resume_required']
        controller = batch.read(policy.parent / 'controller-exit.json')
        assert controller['hold'] and not controller['auto_resume'] and not controller['operator_stopped']
        events = _jsonl(policy.parent / 'controller-events.jsonl')
        assert len([row for row in events if row['event'] == 'worker_launched']) == 1
        usage = _jsonl(attempt / 'runtime/private/ceo/model-usage.jsonl')
        logical, = [row for row in usage if row['event'] == 'response']
        request, = [row for row in usage if row['event'] == 'request']
        http, = [row for row in usage if row['event'] == 'http_response']
        assert request['retry_policy'] == dict(attempts=0, delays=[10, 20, 40, 80], first_byte_seconds=600)
        assert http['status'] == 429 and json.loads(http['body']) == body
        assert logical['model_attempts'] == 1 and logical['retried_http_attempts'] == 0
        assert logical['cost_usd'] is None and all(value is None for value in logical['usage'].values())
        assert code in logical['last_model_error']
        driver.samples[-1].update(attempt=record, quota_http=http, logical_response=logical,
                                  retry_policy=request['retry_policy'], hold=hold)
        driver.worker_options = {}
        resumed_attempt, resumed, restored = driver.run(policy, entry, week=1, resume=True)
        driver.samples[-1]['attempt'] = resumed
        _finished(resumed, restored, source=checkpoint)
        assert resumed_attempt != attempt and len(batch.attempts(entry)) == 2
    finally:
        driver.close()
