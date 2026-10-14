import json
from datetime import datetime, timedelta, timezone

import httpx
from openai import APIConnectionError, OpenAI
import pytest

from saas_bench.agents.bash_agent.agent import BashAgent, FinalText
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from saas_bench.model_usage import ModelUsage, billing_assessment, summarize_usage_log
from saas_bench.run_lifecycle import RunCancelled
from saas_bench.team_usage import aggregate
from test_preflight_team_model_monitor import load_monitor
from test_preflight_team_flow import completion, final
from test_preflight_team_observer import append, events, observer, request


STAMP = '2026-10-13T23:50:58.684551+00:00'
RATES = dict(input=.00015, output=.0006, cache_read=.000003, cache_write=0,
             peak_schedule='weekday_01_04_06_10_utc', peak_multiplier=2)
PRICES = dict(source='frozen test tariff', basis='USD per 1000 tokens',
              rates={'deepseek-v4.1-flash': RATES})


def response(call, *, missing=(), role='ceo', model='deepseek-v4.1-flash', error=None):
    usage = dict(input_tokens=60459, output_tokens=2614, cached_tokens=0,
                 cache_creation_tokens=0, reasoning_tokens=None)
    for field in missing:
        if field in usage:
            usage[field] = None
    return dict(event='response', call_id=call, timestamp=STAMP, role=role, api='chat',
                response={'model': model}, usage=usage, error=error, pricing=RATES,
                cost_usd=None if 'cost_usd' in missing else .01063725)


def logical_request(call, role='ceo'):
    return dict(event='request', call_id=call, role=role, api='chat', day=84,
                timestamp=STAMP, request={'model': 'deepseek-v4.1-flash'})


def test_monitor_script_d84_interval_and_required_tokens(tmp_path, monkeypatch):
    monitor = load_monitor(monkeypatch)
    path = tmp_path / 'engineering/runs/pf/attempt-001/runtime/private/ceo/model-usage.jsonl'
    path.parent.mkdir(parents=True)
    d84 = response('d84', missing=('cached_tokens', 'cache_creation_tokens', 'cost_usd'))
    append(path, logical_request('d84'), d84)
    state = {}
    notices = monitor.poll(tmp_path, state)
    assert not (tmp_path / 'engineering/HOLD').exists()
    assert notices[0]['event'] == 'billing_fields_missing'
    assert notices[0]['missing_fields'] == ['cached_tokens', 'cache_creation_tokens', 'cost_usd']
    assert notices[0]['cost']['kind'] == 'interval'
    assert notices[0]['cost']['lower_usd'] == pytest.approx(.001749777)
    assert notices[0]['cost']['upper_usd'] == pytest.approx(.01063725)
    resumed = path.parents[4] / 'attempt-002/runtime/private/ceo/model-usage.jsonl'
    resumed.parent.mkdir(parents=True)
    resumed.write_bytes(path.read_bytes())
    saved = json.loads(json.dumps(state))
    assert monitor.poll(tmp_path, saved) == []
    assert saved['billing_groups']['pf']['calls'] == 1
    assert saved['billing_groups']['pf']['missing_calls'] == 1
    append(resumed, response('missing-output', missing=('output_tokens',)))
    assert monitor.poll(tmp_path, saved)[0]['issue'] == 'missing_usage'
    assert (tmp_path / 'engineering/HOLD').exists()


def test_observer_main_billing_warning_persists_without_hold(observer, monkeypatch, capsys):
    path = observer.log(role='pf')
    raw = dict(model=observer.module.round5.MODEL,
               usage=dict(prompt_tokens=60459, completion_tokens=2614))
    append(path, logical_request('d84'), request(observer),
           dict(event='http_response', call_id='d84', attempt_id='attempt', status=200,
                body=json.dumps(raw)),
           response('d84', missing=('cached_tokens', 'cache_creation_tokens', 'cost_usd')))
    policy = observer.output.parent / 'policy.json'
    policy.write_text(json.dumps(dict(identity=observer.expected)))
    original = observer.module.poll
    monkeypatch.setattr(observer.module, 'poll',
                        lambda *args, **kwargs: original(*args, quota_fn=observer.quota, **kwargs))
    observer.module.main(['--policy', str(policy), '--once'])
    assert json.loads(capsys.readouterr().out)['status'] == 'healthy'
    assert not observer.hold.exists()
    health = json.loads((observer.output / 'health.json').read_text())
    assert health['billing_warnings'][0]['missing_fields'] == [
        'cached_tokens', 'cache_creation_tokens', 'cost_usd']
    assert health['state']['billing_groups']['pf']['calls'] == 1
    assert events(observer)[0]['billing_warnings'][0]['cost']['kind'] == 'interval'
    observer.module.main(['--policy', str(policy), '--once'])
    assert json.loads(capsys.readouterr().out)['status'] == 'healthy'
    assert len(events(observer)) == 1


def test_group_warning_strict_five_percent_twenty_streak_and_roles(observer):
    path = observer.log(role='pf')
    for ordinal in range(380):
        append(path, logical_request(f'known-{ordinal}'), response(f'known-{ordinal}'))
    assert observer.poll()['status'] == 'healthy'
    for ordinal in range(19):
        missing = response(f'missing-{ordinal}', missing=('cost_usd',), role='growth')
        missing['timestamp'] = f'2026-10-13T23:51:{ordinal:02d}+00:00'
        append(path, logical_request(f'missing-{ordinal}'),
               missing)
    assert observer.poll()['billing_warnings'] == []
    missing = response('missing-19', missing=('cost_usd',))
    missing['timestamp'] = '2026-10-13T23:51:19+00:00'
    append(path, logical_request('missing-19'), missing)
    health = observer.poll()
    warning = health['billing_warnings'][0]
    assert warning['triggers'] == ['consecutive_missing_20']
    assert warning['calls'] == 400 and warning['missing_calls'] == 20
    assert health['status'] == 'healthy'
    missing = response('missing-20', missing=('cost_usd',))
    missing['timestamp'] = '2026-10-13T23:51:20+00:00'
    append(path, logical_request('missing-20'), missing)
    warning = observer.poll()['billing_warnings'][0]
    assert warning['triggers'] == ['missing_over_five_percent', 'consecutive_missing_20']
    git = observer.log(role='git')
    append(git, logical_request('git-known'), response('git-known'))
    assert observer.poll()['state']['billing_groups']['git']['missing_calls'] == 0


def test_streak_follows_response_time_across_role_logs(observer):
    ceo = observer.log(role='pf')
    growth = ceo.parent.parent / 'growth/model-usage.jsonl'
    growth.parent.mkdir()
    start = datetime(2026, 10, 13, 23, tzinfo=timezone.utc)
    for ordinal in range(400):
        row = response(f'prefix-{ordinal}')
        row['timestamp'] = (start + timedelta(seconds=ordinal)).isoformat()
        append(ceo, row)
    for ordinal in range(20):
        missing = response(f'missing-{ordinal}', missing=('cost_usd',))
        missing['timestamp'] = (start + timedelta(seconds=400 + ordinal * 2)).isoformat()
        known = response(f'between-{ordinal}', role='growth')
        known['timestamp'] = (start + timedelta(seconds=401 + ordinal * 2)).isoformat()
        append(ceo, missing)
        append(growth, known)
    health = observer.poll()
    assert health['status'] == 'healthy'
    assert health['billing_warnings'] == []
    assert health['state']['billing_groups']['pf']['calls'] == 440
    assert health['state']['billing_groups']['pf']['consecutive_missing'] == 0


def test_old_observer_state_backfills_with_existing_budget_and_no_duplicate_alerts(observer, monkeypatch, capsys):
    path = observer.log(role='pf')
    for ordinal in range(380):
        append(path, logical_request(f'known-{ordinal}'), response(f'known-{ordinal}'))
    for ordinal in range(19):
        missing = response(f'missing-{ordinal}', missing=('cost_usd',))
        missing['timestamp'] = f'2026-10-13T23:51:{ordinal:02d}+00:00'
        append(path, logical_request(f'missing-{ordinal}'), missing)
    observer.poll()
    legacy = json.loads(json.dumps(observer.state))
    for field in ('billing_groups', 'billing_requests', 'billing_pending', 'billing_coverage'):
        legacy.pop(field, None)
    missing = response('missing-19', missing=('cost_usd',))
    missing['timestamp'] = '2026-10-13T23:51:19+00:00'
    append(path, logical_request('missing-19'), missing)
    partial = observer.module.poll(observer.manifest_path, observer.output, observer.expected, legacy,
                                   quota_fn=observer.quota, max_seconds=0)
    assert partial['status'] == 'pending'
    assert partial['billing_coverage'] == 'rebuilding' and partial['billing_warnings'] == []
    policy = observer.output.parent / 'policy.json'
    policy.write_text(json.dumps(dict(identity=observer.expected)))
    original = observer.module.poll
    monkeypatch.setattr(observer.module, 'poll',
                        lambda *args, **kwargs: original(*args, quota_fn=observer.quota, **kwargs))
    observer.module.main(['--policy', str(policy), '--once'])
    assert json.loads(capsys.readouterr().out)['status'] == 'healthy'
    health = json.loads((observer.output / 'health.json').read_text())
    assert health['billing_coverage'] == 'complete'
    assert health['state']['billing_groups']['pf']['calls'] == 400
    assert health['billing_warnings'][0]['triggers'] == ['consecutive_missing_20']
    assert len(events(observer)) == 1
    observer.module.main(['--policy', str(policy), '--once'])
    assert json.loads(capsys.readouterr().out)['status'] == 'healthy'
    assert len(events(observer)) == 1


def test_old_independent_monitor_state_backfills_without_duplicate_alerts(tmp_path, monkeypatch):
    monitor = load_monitor(monkeypatch)
    path = tmp_path / 'engineering/runs/pf/attempt-001/runtime/private/ceo/model-usage.jsonl'
    path.parent.mkdir(parents=True)
    for ordinal in range(20):
        append(path, logical_request(f'known-{ordinal}'), response(f'known-{ordinal}'))
    state = {}
    assert monitor.poll(tmp_path, state) == []
    for field in ('billing_groups', 'billing_requests', 'billing_pending', 'billing_coverage'):
        state.pop(field, None)
    missing = response('new', missing=('cost_usd',))
    missing['timestamp'] = '2026-10-13T23:51:00+00:00'
    append(path, logical_request('new'), missing)
    assert monitor.poll(tmp_path, state) == []
    assert state['billing_groups']['pf']['calls'] == 21
    assert state['billing_groups']['pf']['missing_calls'] == 1
    assert state['billing_coverage'] == 'complete'
    assert not (tmp_path / 'engineering/HOLD').exists()


def test_round5_shared_billing_recorder_deduplicates_copied_prefix(tmp_path, monkeypatch):
    monitor = load_monitor(monkeypatch)
    state = {}
    monitor.record_billing(logical_request('copied'), state, 'pf', tmp_path / 'first')
    row = response('copied', missing=('cost_usd',))
    changed = monitor.record_billing(row, state, 'pf', tmp_path / 'first')
    assert monitor.billing_warnings(state, [changed])[0]['missing_fields'] == ['cost_usd']
    restored = json.loads(json.dumps(state))
    assert monitor.record_billing(row, restored, 'pf', tmp_path / 'resumed') is None
    assert monitor.billing_warnings(restored, []) == []
    assert restored['billing_groups']['pf']['calls'] == 1
    assert restored['billing_groups']['pf']['receipts']['copied']['path'] == str(tmp_path / 'first')


def test_old_cursor_truncation_guard_survives_partial_billing_rebuild(observer, monkeypatch):
    path = observer.log(role='pf')
    for ordinal in range(30):
        append(path, logical_request(f'known-{ordinal}'), response(f'known-{ordinal}'))
    observer.poll()
    legacy = json.loads(json.dumps(observer.state))
    for field in ('billing_groups', 'billing_requests', 'billing_pending', 'billing_coverage'):
        legacy.pop(field, None)
    original = observer.module.round5.read_increment
    calls = []
    clock = [0]
    monkeypatch.setattr(observer.module.time, 'monotonic', lambda: clock[0])

    def reader(*args, **kwargs):
        result = original(*args, **dict(kwargs, max_bytes=1000))
        calls.append(result[1])
        clock[0] = 2
        return result

    monkeypatch.setattr(observer.module.round5, 'read_increment', reader)
    health = observer.module.poll(observer.manifest_path, observer.output, observer.expected, legacy,
                                   quota_fn=observer.quota, max_seconds=1)
    assert health['status'] == 'pending'
    cursor = health['state']['cursors'][str(path)]
    assert 0 < cursor['offset'] < cursor['billing_previous_offset'] == path.stat().st_size
    restarted = json.loads(json.dumps(health['state']))
    path.write_bytes(path.read_bytes()[:cursor['offset']])
    monkeypatch.setattr(observer.module.round5, 'read_increment', original)
    failed = observer.module.poll(observer.manifest_path, observer.output, observer.expected, restarted,
                                  quota_fn=observer.quota, max_seconds=1)
    assert failed['status'] == 'hold'
    assert failed['checks'] == ['log_truncated:' + str(path)]


@pytest.mark.parametrize('change,issue', [
    ({'missing': ('input_tokens',)}, 'missing_usage'),
    ({'missing': ('output_tokens',)}, 'missing_usage'),
    ({'missing': ('cached_tokens', 'cost_usd'), 'model': 'changed'}, 'served_model_changed'),
    ({'error': 'ValueError'}, 'model_call_error'),
])
def test_observer_non_network_final_response_fault_still_holds(observer, change, issue):
    path = observer.log()
    append(path, response('fault', **change))
    health = observer.poll()
    assert health['status'] == 'hold' and observer.hold.exists()
    assert health['route_errors'][0]['issue'] == issue


def test_model_usage_harness_pause_is_visible_and_provider_value_error_still_holds(observer):
    path = observer.log(role='pf')
    mode = ['pause']
    reply = dict(id='offline', object='chat.completion', created=1, model=observer.module.round5.MODEL,
                 choices=[dict(index=0, message=dict(role='assistant', content='ok'), finish_reason='stop')],
                 usage=dict(prompt_tokens=20, completion_tokens=10,
                            prompt_tokens_details=dict(cached_tokens=0, cache_write_tokens=0)),
                 error={'type': 'provider_failure', 'message': 'Failed response'})

    def respond(request):
        if mode[0] == 'pause':
            raise RunCancelled('requested_pause')
        return httpx.Response(200, json=reply)

    client = OpenAI(api_key='offline', base_url=observer.module.round5.ENDPOINT,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)), max_retries=0)
    recorder = ModelUsage(path, 'ceo', PRICES['rates'])
    recorder.attach(client)
    kwargs = dict(model=reply['model'], messages=[], reasoning_effort='high',
                  extra_body=dict(thinking={'type': 'enabled'}))
    try:
        with pytest.raises(RunCancelled, match='requested_pause'):
            recorder.call('chat', kwargs, lambda: client.chat.completions.create(**kwargs), day=0)
        receipt = json.loads(path.read_text().splitlines()[-1])
        assert receipt['error'] == 'RunCancelled' and receipt['response'] is None
        health = observer.poll()
        assert health['status'] == 'healthy' and not observer.hold.exists()
        assert health['route_errors'] == []
        issues = [row['issue'] for event in events(observer) for row in event['transient_receipt_failures']]
        assert issues == ['transport_error', 'model_call_interrupted']
        mode[0] = 'provider_error'
        with pytest.raises(ValueError, match='Provider error'):
            recorder.call('chat', kwargs, lambda: client.chat.completions.create(**kwargs), day=0)
        health = observer.poll()
        assert health['status'] == 'hold' and observer.hold.exists()
        assert health['route_errors'][0]['issue'] == 'model_call_error'
    finally:
        client.close()


@pytest.mark.parametrize('recover', [True, False])
def test_real_harness_transport_retry_only_holds_after_attempt_limit(observer, monkeypatch, recover):
    path = observer.log(role='pf')
    requests, waits = [], []

    def respond(request):
        requests.append(request)
        if not recover or len(requests) <= 3:
            raise httpx.ConnectError('temporary outage', request=request)
        return completion(final('recovered advice'), stream=True)

    def sleep(seconds):
        waits.append(seconds)
        health = observer.poll()
        assert health['status'] == 'healthy' and not observer.hold.exists()
        assert health['route_errors'] == []

    monkeypatch.setattr('time.sleep', sleep)
    client = OpenAI(api_key='offline', base_url=observer.module.round5.ENDPOINT, max_retries=2,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    recorder = ModelUsage(path, 'ceo', PRICES['rates'])
    agent = BashAgent(get_bash_agent_tool_descriptions(), client, model=observer.module.round5.MODEL,
        system_prompt='Give business advice.', workspace_path=path.parent, reasoning_effort='high',
        usage_recorder=recorder, allow_final_text=True)
    try:
        if recover:
            assert agent.act('dashboard', 0, False, {'day': 0}) == FinalText('recovered advice')
        else:
            with pytest.raises(RunCancelled, match='model_attempt_limit'):
                agent.act('dashboard', 0, False, {'day': 0})
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        logical = [row for row in rows if row['event'] == 'request']
        assert [row['outer_attempt'] for row in logical] == ([1, 2] if recover else [1, 2, 3, 4])
        assert len(requests) == (4 if recover else 12)
        assert [wait for wait in waits if wait >= 10] == ([10] if recover else [10, 20, 40])
        assert {row['error'] for row in rows if row['event'] == 'response' and row.get('error')} == {'APIConnectionError'}
        if recover:
            health = observer.poll()
            assert health['status'] == 'healthy' and not observer.hold.exists()
            assert health['route_errors'] == []
            assert observer.poll(state={})['status'] == 'healthy'
        else:
            assert observer.poll()['status'] == 'healthy' and not observer.hold.exists()
            (path.parents[3] / 'attempt.json').write_text(json.dumps(dict(status='failed',
                attempt='001', day=0, reason='failed', runtime_failure='model_attempt_limit',
                runtime_phase='ceo')))
            health = observer.poll()
            assert health['status'] == 'hold' and observer.hold.exists()
            assert health['checks'] == ['model_attempt_limit:pf']
            assert health['progress']['pf']['runtime_failure'] == 'model_attempt_limit'
        issues = [row['issue'] for event in events(observer) for row in event['transient_receipt_failures']]
        assert issues and set(issues) == {'transport_error'}
    finally:
        client.close()


@pytest.mark.parametrize('error', [httpx.UnsupportedProtocol, httpx.LocalProtocolError, ValueError])
def test_sdk_connection_wrapper_preserves_immediate_configuration_hold(observer, error):
    path = observer.log()

    def respond(request):
        raise error('invalid local configuration')

    client = OpenAI(api_key='offline', base_url=observer.module.round5.ENDPOINT, max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    recorder = ModelUsage(path, 'ceo', PRICES['rates'])
    recorder.attach(client)
    kwargs = dict(model=observer.module.round5.MODEL, messages=[], reasoning_effort='high',
                  extra_body=dict(thinking={'type': 'enabled'}))
    try:
        with pytest.raises(APIConnectionError):
            recorder.call('chat', kwargs, lambda: client.chat.completions.create(**kwargs), day=0)
        assert json.loads(path.read_text().splitlines()[-1])['error'] == 'APIConnectionError'
        health = observer.poll()
        assert health['status'] == 'hold' and observer.hold.exists()
        assert [row['issue'] for row in health['route_errors']] == ['model_call_error']
    finally:
        client.close()


def test_sdk_and_log_summaries_keep_raw_null_and_cost_categories(tmp_path):
    reply = dict(id='offline', object='chat.completion', created=1, model='deepseek-v4.1-flash',
                 choices=[dict(index=0, message=dict(role='assistant', content='success'),
                               finish_reason='stop')],
                 usage=dict(prompt_tokens=60459, completion_tokens=2614))
    client = OpenAI(api_key='offline', http_client=httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=reply))), max_retries=0)
    path = tmp_path / 'model-usage.jsonl'
    recorder = ModelUsage(path, 'ceo', PRICES['rates'])
    recorder.attach(client)
    try:
        result = recorder.call('chat', {'model': reply['model']},
            lambda: client.chat.completions.create(model=reply['model'], messages=[]), day=84)
        assert result.choices[0].message.content == 'success'
    finally:
        client.close()
    receipt = json.loads(path.read_text().splitlines()[-1])
    assert receipt['usage']['cached_tokens'] is receipt['cost_usd'] is None
    assert recorder.summary['costs']['interval']['calls'] == 1
    assert recorder.summary['costs']['interval']['lower_usd'] is not None
    assert summarize_usage_log(path)['costs'] == recorder.summary['costs']
    rows = [logical_request('exact'), response('exact'),
            logical_request('recompute'), response('recompute', missing=('cost_usd',)),
            logical_request('interval'), response('interval', missing=('cached_tokens', 'cost_usd'))]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    before = path.read_bytes()
    summary = summarize_usage_log(path)
    assert summary['costs']['exact'] == dict(calls=1, subtotal_usd=pytest.approx(.01063725))
    assert summary['costs']['recomputed'] == dict(calls=1, subtotal_usd=pytest.approx(.01063725))
    assert summary['costs']['interval']['calls'] == 1
    assert summary['costs']['missing_field_calls'] == 2
    assert path.read_bytes() == before


def test_team_accounting_uses_frozen_uniform_cost_not_recorded_cost(tmp_path):
    rows = [logical_request('exact'), dict(response('exact'), cost_usd=.4),
            logical_request('recompute'), response('recompute', missing=('cost_usd',)),
            logical_request('interval'), response('interval', missing=('cached_tokens', 'cost_usd'))]
    path = tmp_path / 'usage.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    result = aggregate([path], PRICES)
    costs = result['team']['costs']
    assert costs['exact']['subtotal_usd'] == pytest.approx(.01063725)
    assert costs['recomputed']['subtotal_usd'] == pytest.approx(.01063725)
    assert costs['interval']['lower_usd'] == pytest.approx(.001749777)
    assert result['records'][2]['metrics']['cached_tokens'] is None
    assert result['records'][1]['metrics']['recorded_cost_usd'] is None
    assert result['records'][2]['cost']['kind'] == 'interval'
    assert result['roles']['ceo']['costs'] == costs


@pytest.mark.parametrize('missing,kind,lower,upper', [
    (('cost_usd',), 'recomputed', None, None),
    (('cached_tokens', 'cost_usd'), 'interval', .001749777, .01063725),
    (('cache_creation_tokens',), 'exact', None, None),
])
def test_cost_assessment_at_original_time_before_peak(missing, kind, lower, upper):
    row = response('boundary', missing=missing)
    assessment = billing_assessment(row['usage'], 'chat', RATES,
        at=datetime(2026, 10, 14, 0, 59, 59, tzinfo=timezone.utc), recorded_cost=row['cost_usd'])
    assert assessment['kind'] == kind
    assert assessment['usd'] == (pytest.approx(.01063725) if kind != 'interval' else None)
    assert assessment['lower_usd'] == (pytest.approx(lower) if lower is not None else None)
    assert assessment['upper_usd'] == (pytest.approx(upper) if upper is not None else None)


def test_log_recompute_uses_request_time_when_response_crosses_peak(tmp_path):
    req = dict(logical_request('boundary'), timestamp='2026-10-14T00:59:59+00:00')
    reply = dict(response('boundary', missing=('cost_usd',)), timestamp='2026-10-14T01:00:01+00:00')
    path = tmp_path / 'boundary.jsonl'
    path.write_text(json.dumps(req) + '\n' + json.dumps(reply) + '\n')
    assert summarize_usage_log(path)['costs']['recomputed']['subtotal_usd'] == pytest.approx(.01063725)
    assert aggregate([path], PRICES)['team']['costs']['recomputed']['subtotal_usd'] == pytest.approx(.01063725)


@pytest.mark.parametrize('rates,at', [
    (None, datetime.fromisoformat(STAMP)),
    (RATES, None),
    (dict(RATES, valid_until='2026-10-13T23:00:00+00:00'), datetime.fromisoformat(STAMP)),
    (dict(input=1), datetime.fromisoformat(STAMP)),
])
def test_missing_frozen_price_or_time_keeps_cost_unknown(rates, at):
    row = response('unknown', missing=('cost_usd',))
    assert billing_assessment(row['usage'], 'chat', rates, at=at) == dict(
        kind='unknown', usd=None, lower_usd=None, upper_usd=None, missing_fields=['cost_usd'])


@pytest.mark.parametrize('read,lower,upper', [(10, .095, .14), (None, .014, .154)])
def test_messages_missing_cache_write_has_feasible_endpoints(read, lower, upper):
    usage = dict(input_tokens=100, output_tokens=2, cached_tokens=read,
                 cache_creation_tokens=None, reasoning_tokens=None)
    rates = dict(input=1, output=2, cache_read=.1, cache_write=1.5)
    result = billing_assessment(usage, 'messages', rates)
    assert result['kind'] == 'interval'
    assert result['lower_usd'] == pytest.approx(lower)
    assert result['upper_usd'] == pytest.approx(upper)
    assert usage['cache_creation_tokens'] is None


def test_legacy_model_usage_summary_names_unclassified_prefix(tmp_path):
    rates = dict(input=.00015, output=.0006, cache_read=.000003, cache_write=0)
    recorder = ModelUsage(tmp_path / 'legacy.jsonl', 'ceo', {'deepseek-v4.1-flash': rates})
    reply = dict(model='deepseek-v4.1-flash', choices=[dict(message={'role': 'assistant', 'content': 'ok'})],
                 usage=dict(prompt_tokens=10, completion_tokens=2,
                            prompt_tokens_details=dict(cached_tokens=3, cache_write_tokens=0)))
    recorder.call('chat', {}, lambda: reply, day=84)
    recorder.summary.pop('costs')
    recorder.call('chat', {}, lambda: reply, day=84)
    assert recorder.summary['calls'] == 2
    assert recorder.summary['costs']['unclassified_prior_calls'] == 1
    assert recorder.summary['costs']['exact']['calls'] == 1
    assert recorder.summary['costs']['exact']['subtotal_usd'] == pytest.approx(.000002259)
    assert summarize_usage_log(recorder.path)['costs']['unclassified_prior_calls'] == 0
    assert summarize_usage_log(recorder.path)['costs']['exact']['calls'] == 2
