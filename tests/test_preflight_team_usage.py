import copy
import json
import tracemalloc
from types import SimpleNamespace

import httpx
from openai import OpenAI
import pytest

from saas_bench.database import init_database, save_predictions
from saas_bench.model_usage import ModelUsage, usage_values
from saas_bench.team_results import summarize_pairs, summarize_run
from saas_bench.team_usage import aggregate


PRICES = dict(source='offline frozen provider table', basis='USD per 1000 tokens',
    rates={'test-model': dict(input=1, output=2, cache_read=.1, cache_write=1.5)})
STAMP = '2026-10-07T00:00:00+00:00'


def body(input_tokens=10, output_tokens=2, cached=3, reasoning=1, *, api='chat', bill=None):
    if api == 'chat':
        usage = dict(prompt_tokens=input_tokens, completion_tokens=output_tokens,
            prompt_tokens_details={'cached_tokens': cached, 'cache_write_tokens': 0},
            completion_tokens_details={'reasoning_tokens': reasoning}, total_tokens=9999)
    elif api == 'messages':
        usage = dict(input_tokens=input_tokens, output_tokens=output_tokens,
            cache_read_input_tokens=cached, cache_creation_input_tokens=2,
            output_tokens_details={'reasoning_tokens': reasoning})
    else:
        usage = dict(input_tokens=input_tokens, output_tokens=output_tokens,
            input_tokens_details={'cached_tokens': cached, 'cache_write_tokens': 0},
            output_tokens_details={'reasoning_tokens': reasoning})
    if bill is not None:
        usage['cost'] = bill
    return dict(model='test-model', usage=usage, choices=[dict(index=0,
        message=dict(role='assistant', content='offline'), finish_reason='stop')],
        id='response', object='chat.completion', created=1)


def receipts(call, role, result=None, *, api='chat', day=0, attempts=(), cost=None, returned=True):
    common = dict(call_id=call, role=role, timestamp=STAMP)
    rows = [dict(common, event='request', api=api, day=day, request={'model': 'test-model'})]
    for retry, (attempt, response, error, status) in enumerate(attempts):
        rows.append(dict(common, event='http_request', attempt_id=attempt, body={'model': 'test-model'}, sdk_retry=str(retry)))
        if status is None:
            rows.append(dict(common, event='http_error', attempt_id=attempt, error=error))
        else:
            rows.append(dict(common, event='http_response', attempt_id=attempt,
                body=json.dumps(response), error=error, status=status))
    if returned:
        rows.append(dict(common, event='response', api=api, response=result,
            usage=usage_values(result, api), error='failed' if result is None else None, cost_usd=cost))
    return rows


def log(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


@pytest.mark.parametrize('tail', [b'{"event":', b'{"event":"response"',
    b'{"reply":"\xe4\xb8', b'{"event":"request","flag":tru'])
def test_truncated_final_receipt_preserves_known_usage_and_pending_tail(tmp_path, tail):
    path = log(tmp_path / 'model-usage.jsonl', receipts('known', 'ceo', body(), cost=.1))
    prefix = path.read_bytes()
    path.write_bytes(prefix + tail)
    result = aggregate([path], PRICES)
    assert result['team']['known']['input_tokens'] == 10
    assert result['team']['known']['output_tokens'] == 2
    assert result['team']['known']['total_tokens'] == 12
    assert all(value is None for value in result['team']['complete'].values())
    assert all(value is None for value in result['all_resources']['complete'].values())
    assert result['missing_receipts'] == [dict(path=str(path.resolve()), line=3,
        offset=len(prefix), bytes=len(tail), category='formal', reason='unterminated_log_tail')]
    assert path.read_bytes() == prefix + tail
    assert not list(tmp_path.glob('*.fault.json'))


@pytest.mark.parametrize('broken', [b'{bad}\n', b'{bad}', b'{"event":\n',
    b'{"reply":"\xff"}\n', b'{"event":\n{"event":"response"}\n'])
def test_complete_or_midfile_malformed_receipt_still_fails(tmp_path, broken):
    path = log(tmp_path / 'model-usage.jsonl', receipts('known', 'ceo', body(), cost=.1))
    path.write_bytes(path.read_bytes() + broken)
    with pytest.raises((json.JSONDecodeError, UnicodeDecodeError)):
        aggregate([path], PRICES)


def test_completed_pending_log_next_read_restores_complete_totals(tmp_path):
    rows = receipts('known', 'ceo', body(), cost=.1)
    next_rows = receipts('next', 'growth', body(20, 4), cost=.2)
    raw = ''.join(json.dumps(row) + '\n' for row in rows + next_rows).encode()
    path = tmp_path / 'model-usage.jsonl'
    path.write_bytes(raw[:-5])
    assert aggregate([path], PRICES)['missing_receipts']
    path.write_bytes(raw)
    result = aggregate([path], PRICES)
    assert result['team']['complete']['input_tokens'] == 30
    assert result['team']['complete']['output_tokens'] == 6
    assert result['team']['complete']['total_tokens'] == 36
    assert result['missing_receipts'] == []


@pytest.mark.parametrize('resource', ['simulator', 'shared_d0', 'abandoned'])
def test_explicit_resource_tail_keeps_confirmed_primary_totals(tmp_path, resource):
    formal = log(tmp_path / 'formal.jsonl', receipts('formal', 'ceo', body(), cost=.1))
    resource_log = log(tmp_path / f'{resource}.jsonl',
        receipts(resource, 'simulator' if resource != 'abandoned' else 'growth', body(20, 4), cost=.2))
    paths = dict(formal=[formal], **{resource: [resource_log]})
    confirmed = aggregate(paths, PRICES)
    original = resource_log.read_bytes() + b'{"reply":"\xe4\xb8'
    resource_log.write_bytes(original)
    result = aggregate(paths, PRICES)
    assert result['team'] == confirmed['team']
    assert result['roles'] == confirmed['roles']
    assert result['resources']['formal'] == confirmed['resources']['formal']
    assert result['resources'][resource]['known'] == confirmed['resources'][resource]['known']
    assert result['resources'][resource]['known']['total_tokens'] == 24
    for summary in (result['resources'][resource], result['all_resources'], result['failed_attempt_usage']):
        assert all(value is None for value in summary['complete'].values())
    assert result['all_resources']['known'] == confirmed['all_resources']['known']
    assert result['missing_receipts'][0]['category'] == resource
    assert resource_log.read_bytes() == original


@pytest.mark.parametrize('resource,primary', [('formal', 'engineering'), ('engineering', 'formal')])
def test_unrouted_role_tail_only_invalidates_source_category_and_simulator(tmp_path, resource, primary):
    paths = {category: [log(tmp_path / f'{category}.jsonl',
        receipts(category, 'simulator' if category == 'simulator' else 'ceo', body(), cost=.1))]
        for category in ('formal', 'engineering', 'simulator')}
    confirmed = aggregate(paths, PRICES, category=primary)
    resource_log = paths[resource][0]
    original = resource_log.read_bytes() + b'{"event":'
    resource_log.write_bytes(original)
    result = aggregate(paths, PRICES, category=primary)
    assert result['team'] == confirmed['team']
    assert result['roles'] == confirmed['roles']
    assert result['resources'][primary] == confirmed['resources'][primary]
    for affected in (resource, 'simulator'):
        assert result['resources'][affected]['known'] == confirmed['resources'][affected]['known']
        assert all(value is None for value in result['resources'][affected]['complete'].values())
    assert resource_log.read_bytes() == original


def test_manually_recomputed_team_and_all_resource_receipts(tmp_path):
    failed, success = body(5, 1, 2, 1, bill=.01), body(bill=.02)
    rows = receipts('retry', 'ceo', success, attempts=[('retry-1', failed, 'provider_error', 503),
        ('retry-2', success, None, 200)], cost=.3)
    rows += receipts('regenerated', 'ceo', body(7, 3, 1, 2, bill=.03), cost=.4)
    rows += receipts('analyst', 'growth', body(5, 4, 3, 2, api='messages', bill=.04), api='messages', cost=.5)
    rows += receipts('forecast', 'ops_finance', body(20, 5, 4, 3, api='responses', bill=.05), api='responses', cost=.6)
    rows += receipts('partial', 'growth', dict(model='test-model', usage={'completion_tokens': 6}))
    rows += receipts('no-response', 'ops_finance', returned=False)
    rows += receipts('missing', 'ceo', {'model': 'test-model'})
    original = log(tmp_path / 'original.jsonl', rows)
    resumed = log(tmp_path / 'resumed.jsonl', rows + receipts('followup', 'growth', body(2, 1, 0, 0, bill=.06), day=7, cost=.7))
    simulator = log(tmp_path / 'simulator.jsonl', receipts('sim', 'simulator', body(30, 5, 0, 1)))
    d0 = log(tmp_path / 'd0.jsonl', receipts('shared', 'simulator', body(40, 6, 0, 2)))
    engineering = log(tmp_path / 'engineering.jsonl', receipts('eng', 'ceo', body(50, 7, 0, 3)))
    abandoned = log(tmp_path / 'abandoned.jsonl', receipts('discarded', 'growth', body(60, 8, 0, 4)))
    result = aggregate(dict(formal=[original, resumed, simulator], shared_d0=[d0, d0],
        engineering=[engineering], abandoned=[abandoned]), PRICES)
    team = result['team']
    assert team['billed_units'] == 9 and result['calls'] == 12 and result['attempts'] == 2
    assert team['known']['input_tokens'] == 54
    assert team['known']['output_tokens'] == 22
    assert team['known']['total_tokens'] == 70
    assert team['known']['cached_tokens'] == 13
    assert team['known']['cache_creation_tokens'] == 2
    assert team['known']['reasoning_tokens'] == 9
    assert team['known']['uniform_cost_usd'] == pytest.approx(.0753)
    assert team['known']['raw_billed_cost_usd'] == pytest.approx(.21)
    assert team['known']['recorded_cost_usd'] == pytest.approx(2.5)
    assert team['missing']['input_tokens'] == team['missing']['total_tokens'] == 3
    assert team['missing']['output_tokens'] == 2
    assert team['missing']['uniform_cost_usd'] == team['missing']['raw_billed_cost_usd'] == 3
    assert team['complete']['input_tokens'] is None and team['complete']['uniform_cost_usd'] is None
    assert team['unreturned_requests'] == team['no_response_requests'] == 1
    assert result['failed_attempt_usage']['known']['input_tokens'] == 5
    assert result['failed_attempt_usage']['known']['uniform_cost_usd'] == pytest.approx(.0052)
    assert result['duplicates'] == len(rows) + 2
    for field in team['known']:
        assert sum(role['known'][field] or 0 for role in result['roles'].values()) == pytest.approx(team['known'][field])
    assert {key: value['known']['input_tokens'] for key, value in result['resources'].items()} == dict(
        formal=54, simulator=30, shared_d0=40, engineering=50, abandoned=60)
    assert result['all_resources']['known']['input_tokens'] == 234
    engineering_trial = aggregate([engineering, simulator], PRICES, category='engineering')
    assert engineering_trial['team']['known']['input_tokens'] == 50
    assert engineering_trial['resources']['engineering']['known']['input_tokens'] == 50
    assert engineering_trial['resources']['simulator']['known']['input_tokens'] == 30
    suffix = aggregate([original, resumed], PRICES, start_day=7, end_day=14)
    assert suffix['team']['complete']['input_tokens'] == 2
    assert suffix['team']['complete']['total_tokens'] == 3
    assert suffix['team']['complete']['uniform_cost_usd'] == pytest.approx(.004)


def test_real_sdk_http_retry_retains_known_failed_usage_once(tmp_path, monkeypatch):
    from saas_bench.model_usage import physical_usage_summary, summarize_usage_log
    monkeypatch.setattr('time.sleep', lambda _: None)
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(503 if len(requests) == 1 else 200,
            json=body(5, 1, 2, 1, bill=.01) if len(requests) == 1 else body(bill=.02))
    recorder = ModelUsage(tmp_path / 'ceo.jsonl', 'ceo', PRICES['rates'])
    monkeypatch.setattr(recorder.lifecycle, 'sleep', lambda seconds: None)
    client = recorder.attach(OpenAI(api_key='offline', max_retries=1,
        http_client=httpx.Client(transport=httpx.MockTransport(handle))))
    request = dict(model='test-model', messages=[dict(role='user', content='offline')])
    try:
        assert recorder.call('chat', request, lambda: client.chat.completions.create(**request), day=0).model == 'test-model'
    finally:
        client.close()
    result = aggregate([recorder.path], PRICES)
    assert len(requests) == result['attempts'] == result['team']['billed_units'] == 2
    assert result['calls'] == 1
    assert result['team']['complete']['input_tokens'] == 15
    assert result['team']['complete']['output_tokens'] == 3
    assert result['team']['complete']['total_tokens'] == 18
    assert result['team']['complete']['uniform_cost_usd'] == pytest.approx(.0165)
    assert result['team']['complete']['raw_billed_cost_usd'] == pytest.approx(.03)
    assert result['failed_attempt_usage']['complete']['input_tokens'] == 5
    for report in (physical_usage_summary(recorder.summary), summarize_usage_log(recorder.path)):
        assert report['calls'] == 2 and report['errors'] == 1
        assert report['known']['input_tokens'] == 15 and report['known']['output_tokens'] == 3
        assert report['missing']['input_tokens'] == 0


def test_model_mismatch_preserves_raw_usage_and_continues(tmp_path, monkeypatch, caplog):
    from saas_bench.run_lifecycle import RunCancelled
    recorder = ModelUsage(tmp_path / 'model.jsonl', 'ceo', PRICES['rates'], expected_model='frozen-model')
    waits = []
    monkeypatch.setattr(recorder.lifecycle, 'sleep', waits.append)
    assert recorder.call('chat', {'model': 'frozen-model'}, lambda: body(), day=0)['model'] == 'test-model'
    assert 'Returned model differs from frozen configuration' in caplog.text
    assert waits == []
    row = json.loads(recorder.path.read_text().splitlines()[-1])
    assert row['response']['model'] == 'test-model' and row['usage']['input_tokens'] == 10
    assert row['error'] is None and recorder.summary['errors'] == 0
    assert row['model_attempts'] == 1 and row['last_model_error'] is None
    assert row['cost_usd'] == pytest.approx(.0113)
    recorder.expected_model = 'test-model'
    assert recorder.call('chat', {}, lambda: body(), day=7)['model'] == 'test-model'
    with pytest.raises(RunCancelled, match='model_request_failed'):
        recorder.call('chat', {}, lambda: None, day=14)
    assert json.loads(recorder.path.read_text().splitlines()[-1])['response'] is None


def test_terminal_http_error_keeps_reported_usage_in_logical_and_physical_reports(tmp_path):
    from saas_bench.model_usage import physical_usage_summary, summarize_usage_log
    from saas_bench.run_lifecycle import RunCancelled
    recorder = ModelUsage(tmp_path / 'ceo.jsonl', 'ceo', PRICES['rates'], model_request_attempts=1)
    client = recorder.attach(OpenAI(api_key='offline', max_retries=2,
        http_client=httpx.Client(transport=httpx.MockTransport(
            lambda _: httpx.Response(503, json=body(5, 1, 2, 1, bill=.01))))))
    request = dict(model='test-model', messages=[dict(role='user', content='offline')])
    try:
        with pytest.raises(RunCancelled, match='after 1 attempts'):
            recorder.call('chat', request, lambda: client.chat.completions.create(**request), day=0)
    finally:
        client.close()
    for report in (recorder.summary, physical_usage_summary(recorder.summary), summarize_usage_log(recorder.path)):
        assert report['calls'] == report['errors'] == 1
        assert report['known']['input_tokens'] == 5 and report['known']['output_tokens'] == 1
        assert report['missing']['input_tokens'] == 0
    assert aggregate([recorder.path], PRICES)['team']['complete']['input_tokens'] == 5


def test_shared_recorder_threads_keep_failed_usage_local_to_each_call(tmp_path):
    import threading
    from saas_bench.model_usage import physical_usage_summary
    from saas_bench.run_lifecycle import RunCancelled
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    recorder = ModelUsage(tmp_path / 'ceo.jsonl', 'simulator', PRICES['rates'], model_request_attempts=1)
    outcomes = {}
    def handle(request):
        owner = json.loads(request.content)['messages'][0]['content']
        if owner == 'first':
            entered.set()
            assert release.wait(3)
            return httpx.Response(503, json=body(5, 1, 2, 1))
        return httpx.Response(200, json=body())
    client = recorder.attach(OpenAI(api_key='offline', max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle))))
    def call(owner):
        request = dict(model='test-model', messages=[dict(role='user', content=owner)])
        try:
            recorder.call('chat', request, lambda: client.chat.completions.create(**request), day=0, owner=owner)
        except RunCancelled as exc:
            outcomes[owner] = exc.reason
    def second():
        second_started.set()
        def fail_without_http():
            raise ValueError('local failure without HTTP')
        try:
            recorder.call('chat', {}, fail_without_http, day=0, owner='generic')
        except RunCancelled as exc:
            outcomes['generic'] = exc.reason
        call('second')
    first, other = threading.Thread(target=lambda: call('first')), threading.Thread(target=second)
    try:
        first.start()
        assert entered.wait(3)
        other.start()
        assert second_started.wait(3)
        release.set()
        first.join(3)
        other.join(3)
        assert not first.is_alive() and not other.is_alive()
        assert outcomes == dict(first='model_request_failed', generic='model_request_failed')
        rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
        owners = {row['call_id']: row['owner'] for row in rows if row['event'] == 'request'}
        replies = {owners[row['call_id']]: row['usage'] for row in rows if row['event'] == 'response'}
        assert replies['first']['input_tokens'] == 5
        assert replies['generic']['input_tokens'] is None
        assert replies['second']['input_tokens'] == 10
        report = physical_usage_summary(recorder.summary)
        assert report['calls'] == 3 and report['known']['input_tokens'] == 15
        assert report['missing']['input_tokens'] == 1
    finally:
        release.set()
        for thread in (first, other):
            if thread.ident is not None:
                thread.join(3)
        client.close()


def test_interleaved_recovery_paths_do_not_choose_final_attempt_by_file_order(tmp_path):
    failed, success = body(5, 1, 2, 1, bill=.01), body(bill=.02)
    rows = receipts('retry', 'ceo', success, attempts=[('retry-1', failed, 'provider_error', 503),
        ('retry-2', success, None, 200)], cost=.3)
    last = log(tmp_path / 'last.jsonl', [rows[0], *rows[3:]])
    first = log(tmp_path / 'first.jsonl', rows[1:3])
    for paths in ([last, first], [first, last]):
        result = aggregate(paths, PRICES)
        assert result['team']['complete']['input_tokens'] == 15
        assert result['team']['complete']['uniform_cost_usd'] == pytest.approx(.0165)
        assert result['failed_attempt_usage']['complete']['input_tokens'] == 5
        assert result['failed_attempt_usage']['known']['recorded_cost_usd'] is None


@pytest.mark.parametrize('outcome', ['success', 'failed', 'unreturned'])
def test_transport_failure_points_to_its_physical_error_receipt(tmp_path, outcome):
    attempts = [('transport-failed', None, 'ConnectError', None)]
    if outcome == 'success':
        attempts.append(('retry-success', body(), None, 200))
    rows = receipts('transport', 'growth', body() if outcome == 'success' else None,
                    attempts=attempts, returned=outcome != 'unreturned')
    path = log(tmp_path / 'transport.jsonl', rows)
    copied = log(tmp_path / 'copied.jsonl', rows)
    result = aggregate([path, copied], PRICES)
    failed = next(row for row in result['records'] if row['attempt_id'] == 'transport-failed')
    assert failed['failed_http_attempt']
    assert failed['response_receipt'] == dict(path=str(path.resolve()), line=3)
    receipt = json.loads(path.read_text().splitlines()[failed['response_receipt']['line'] - 1])
    assert receipt['event'] == 'http_error' and receipt['attempt_id'] == failed['attempt_id']
    assert result['duplicates'] == len(rows)
    if outcome == 'success':
        success = next(row for row in result['records'] if row['attempt_id'] == 'retry-success')
        assert success['response_receipt'] == dict(path=str(path.resolve()), line=5)


def test_partial_stream_usage_and_no_http_return_remain_visible(tmp_path):
    raw = ('event: message_start\ndata: ' + json.dumps({'type': 'message_start',
        'message': {'model': 'test-model', 'usage': {'input_tokens': 5,
            'cache_read_input_tokens': 3, 'cache_creation_input_tokens': 2}}}) + '\n\n' +
        'event: message_delta\ndata: ' + json.dumps({'type': 'message_delta', 'usage': {'output_tokens': 4}}) + '\n\n')
    rows = receipts('interrupted', 'growth', None, api='messages')
    rows[1:1] = [dict(event='http_request', call_id='interrupted', attempt_id='stream', role='growth', timestamp=STAMP),
        dict(event='http_response', call_id='interrupted', attempt_id='stream', role='growth',
            status=200, body=raw, error='ReadError', timestamp=STAMP),
        dict(event='http_error', call_id='interrupted', attempt_id='stream', role='growth', error='ReadError', timestamp=STAMP)]
    rows += receipts('pending', 'ceo', returned=False)
    rows.append(dict(event='http_request', call_id='pending', attempt_id='pending-attempt', role='ceo', timestamp=STAMP))
    result = aggregate([log(tmp_path / 'stream.jsonl', rows)], PRICES)
    assert result['team']['known']['input_tokens'] == 10
    assert result['team']['known']['output_tokens'] == 4
    assert result['team']['known']['total_tokens'] == 14
    assert result['failed_attempt_usage']['known']['uniform_cost_usd'] == pytest.approx(.0163)
    assert result['team']['no_response_requests'] == 2
    assert result['team']['unreturned_requests'] == 1
    assert result['team']['missing']['raw_billed_cost_usd'] == 2
    assert any(gap['reason'] == 'unreturned_http_request' for gap in result['missing_receipts'])
    stream = next(row for row in result['records'] if row['attempt_id'] == 'stream')
    assert stream['response_receipt'] == dict(path=str((tmp_path / 'stream.jsonl').resolve()), line=3)


def test_conflicting_copied_receipts_and_invalid_counts_are_rejected(tmp_path):
    rows = receipts('same', 'ceo', body())
    first = log(tmp_path / 'first.jsonl', rows)
    changed = copy.deepcopy(rows)
    changed[-1]['usage']['input_tokens'] = 11
    second = log(tmp_path / 'second.jsonl', changed)
    with pytest.raises(ValueError, match='Conflicting immutable'):
        aggregate([first, second], PRICES)
    with pytest.raises(ValueError, match='Conflicting immutable'):
        aggregate(dict(formal=[first], engineering=[first]), PRICES)
    changed = copy.deepcopy(rows)
    changed[-1]['usage']['cached_tokens'] = -1
    with pytest.raises(ValueError, match='finite nonnegative'):
        aggregate([log(tmp_path / 'invalid.jsonl', changed)], PRICES)
    changed[0].pop('call_id')
    with pytest.raises(ValueError, match='Immutable'):
        aggregate([log(tmp_path / 'identity.jsonl', changed)], PRICES)


@pytest.mark.parametrize('numeric_equivalent', [False, True])
def test_aggregate_encoding_different_duplicates_keep_full_equality(tmp_path, numeric_equivalent):
    rows = receipts('same', 'ceo', body(), attempts=[('same-http', body(), None, 200)], cost=.1)
    rows[0]['request']['messages'] = [dict(role='user', content='离线')]
    first = log(tmp_path / 'first.jsonl', rows)
    copied = copy.deepcopy(rows)
    if numeric_equivalent:
        copied[0]['day'] = 0.0
        copied[2]['status'] = 200.0
        copied[-1]['response']['created'] = 1.0
        copied[-1]['usage']['input_tokens'] = 10.0
    second = tmp_path / 'encoded.jsonl'
    second.write_text(''.join(json.dumps(dict(reversed(list(row.items()))),
        ensure_ascii=False, separators=(',', ':')) + '\n' for row in copied))
    expected = aggregate([first], PRICES)
    result = aggregate([first, second], PRICES)
    assert result == dict(expected, duplicates=len(rows))
    assert result['records'][0]['response_receipt'] == dict(path=str(first.resolve()), line=3)


@pytest.mark.parametrize('event', ['request', 'http_request', 'http_response', 'response'])
def test_aggregate_changed_discarded_body_still_conflicts(tmp_path, event):
    rows = receipts('same', 'ceo', body(), attempts=[('same-http', body(), None, 200)], cost=.1)
    first = log(tmp_path / 'first.jsonl', rows)
    changed = copy.deepcopy(rows)
    row = next(row for row in changed if row['event'] == event)
    if event == 'http_response':
        raw = json.loads(row['body'])
        raw['choices'][0]['message']['content'] = 'changed without changing usage'
        row['body'] = json.dumps(raw)
    elif event == 'response':
        row['response']['choices'][0]['message']['content'] = 'changed without changing usage'
    else:
        row['request' if event == 'request' else 'body']['messages'] = [dict(content='changed')]
    second = log(tmp_path / 'changed.jsonl', changed)
    with pytest.raises(ValueError, match='Conflicting immutable'):
        aggregate([first, second], PRICES)


def test_aggregate_memory_does_not_retain_payload_history(tmp_path):
    path = tmp_path / 'large.jsonl'
    payload = 'x' * (256 * 1024)
    with path.open('w') as stream:
        for ordinal in range(48):
            response = body()
            response['choices'][0]['message']['content'] = payload
            rows = receipts(f'call-{ordinal}', 'ceo', response,
                attempts=[(f'http-{ordinal}', response, None, 200)], cost=.1)
            rows[0]['request']['messages'] = [dict(role='user', content=payload)]
            rows[1]['body']['messages'] = [dict(role='user', content=payload)]
            for row in rows:
                stream.write(json.dumps(row) + '\n')
    assert path.stat().st_size > 48 * 1024**2
    tracemalloc.start()
    try:
        result = aggregate([path], PRICES)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result['calls'] == result['attempts'] == 48
    assert result['team']['complete']['input_tokens'] == 480
    assert result['team']['complete']['output_tokens'] == 96
    assert result['duplicates'] == 0 and result['missing_receipts'] == []
    assert result['records'][-1]['request_receipt'] == dict(path=str(path.resolve()), line=189)
    assert result['records'][-1]['response_receipt'] == dict(path=str(path.resolve()), line=191)
    print(f'48 MiB payload history: traced aggregation peak={peak} bytes')
    assert peak < 12 * 1024**2, f'Aggregation retained payload history: traced peak={peak} bytes'


@pytest.mark.parametrize('physical', [False, True])
@pytest.mark.parametrize('source,key', [('row', 'raw_cost_usd'), ('row', 'billed_cost_usd'),
    ('body', 'cost_usd'), ('body', 'cost'), ('usage', 'cost_usd'), ('usage', 'cost')])
def test_aggregate_retains_raw_billing_sources(tmp_path, physical, source, key):
    response = body()
    rows = receipts('bill', 'ceo', response,
        attempts=[('bill-http', response, None, 200)] if physical else (), cost=.4)
    row = next(row for row in rows if row['event'] == ('http_response' if physical else 'response'))
    raw = json.loads(row['body']) if physical else row['response']
    target = row if source == 'row' else raw if source == 'body' else raw['usage']
    target[key] = 1.25
    if physical:
        row['body'] = json.dumps(raw)
    result = aggregate([log(tmp_path / 'bill.jsonl', rows)], PRICES)
    assert result['team']['complete']['raw_billed_cost_usd'] == 1.25
    assert result['team']['complete']['recorded_cost_usd'] == .4
    assert result['team']['complete']['uniform_cost_usd'] == pytest.approx(.0113)


@pytest.mark.parametrize('response', [None, {}])
def test_aggregate_empty_response_retains_receive_state(tmp_path, response):
    result = aggregate([log(tmp_path / 'empty.jsonl', receipts('empty', 'ceo', response))], PRICES)
    assert result['records'][0]['unreturned'] is False
    assert result['records'][0]['no_response'] is (response is None)
    assert result['team']['complete']['input_tokens'] is None


def test_aggregate_encoding_difference_detects_changed_original_receipt(tmp_path):
    rows = receipts('same', 'ceo', body(), attempts=[('same-http', body(), None, 200)])
    first = log(tmp_path / 'first.jsonl', rows)
    copied = tmp_path / 'encoded.jsonl'
    copied.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))
    def files():
        yield first
        first.write_text(first.read_text().replace('offline', 'changed', 1))
        yield copied
    with pytest.raises(ValueError, match='Original immutable model receipt changed'):
        aggregate(files(), PRICES)


def test_frozen_model_time_prices_and_zero_values(tmp_path):
    rows = receipts('zero', 'ceo', body(0, 0, 0, 0))
    result = aggregate([log(tmp_path / 'zero.jsonl', rows)], PRICES)
    assert result['team']['complete']['input_tokens'] == result['team']['complete']['uniform_cost_usd'] == 0
    assert result['team']['complete']['raw_billed_cost_usd'] is None
    prices = copy.deepcopy(PRICES)
    prices['rates']['test-model'].update(valid_from='2026-10-06T00:00:00+00:00', valid_until=STAMP)
    assert aggregate([tmp_path / 'zero.jsonl'], prices)['team']['complete']['uniform_cost_usd'] is None
    for row in rows:
        row.pop('timestamp')
    assert aggregate([log(tmp_path / 'no-time.jsonl', rows)], prices)['team']['complete']['uniform_cost_usd'] is None
    unknown_model = receipts('unknown-model', 'growth', dict(body(), model='new-model'))
    result = aggregate([log(tmp_path / 'model.jsonl', unknown_model)], PRICES)
    assert result['team']['complete']['input_tokens'] == 10
    assert result['team']['complete']['uniform_cost_usd'] is None
    engineering = aggregate([tmp_path / 'model.jsonl'], PRICES, category='engineering')
    assert engineering['team']['complete']['input_tokens'] == 10
    assert engineering['roles']['growth']['complete']['input_tokens'] == 10
    assert engineering['resources']['engineering']['complete']['input_tokens'] == 10
    assert engineering['resources']['formal']['known']['input_tokens'] is None


def test_result_cash_scoring_terminal_statuses_and_pair_missingness(tmp_path):
    conn = init_database(':memory:')
    conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES (0,'initial_funding',100,'offline')")
    conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES (112,'operations',-10,'advance')")
    conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES (112,'research_project',-40,'new week')")
    save_predictions(conn, 84, {28: {'cash': dict(point=90, lower=85, upper=95)},
        84: {'cash': dict(point=100, lower=90, upper=110)}}, 0)
    usage = aggregate([log(tmp_path / 'usage.jsonl', receipts('run', 'ceo', body()))], PRICES)
    args = dict(run_id='git-run', group='git', seed=11, repeat=1, day=112,
        reason='observation_end', usage=usage,
        last_week_result=SimpleNamespace(day=112, cash=90, mrr=42))
    git = summarize_run(conn, **args)
    assert git['status'] == 'reached_d112'
    assert git['cash_at_advance'] == 90
    assert git['business_metrics_day'] == 112 and git['mrr'] == 42
    assert git['predictions']['four_week_mean_interval_score'] == 10
    assert git['unexpired_predictions'][0]['target_day'] == 168
    assert git['unexpired_predictions'][0]['status'] == 'pending'
    pf = summarize_run(conn, **dict(args, run_id='pf-run', group='pf'))
    result = summarize_pairs([git, pf])
    assert result['complete_d112_pairs'] == 1
    assert result['pairs'][0]['comparison']['total_tokens']['pf_minus_git'] == 0
    assert result['pairs'][0]['comparison']['total_tokens']['pf_over_git'] == 1
    for reason, status in [('paused', 'paused'), ('failed', 'technical_failure')]:
        run = summarize_run(conn, **dict(args, reason=reason, day=28,
            last_week_result=SimpleNamespace(day=28, cash=100, mrr=30)))
        assert run['status'] == status and len(run['unexpired_predictions']) == 2
    conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES (14,'operations',-200,'bankruptcy')")
    bankrupt = summarize_run(conn, **dict(args, run_id='bankrupt', seed=12, day=14, reason='natural_end',
        last_week_result=SimpleNamespace(day=14, cash=-100, mrr=0)))
    assert bankrupt['status'] == 'natural_bankruptcy'
    assert bankrupt['predictions']['status_counts']['unmatured_bankruptcy'] == 2
    partner = dict(pf, seed=12)
    result = summarize_pairs([bankrupt, partner])
    assert result['complete_d112_pairs'] == 0
    assert result['pairs'][0]['common_completed_week_end'] == 14
    zero = aggregate([log(tmp_path / 'zero.jsonl', receipts('zero', 'ceo', body(0, 0, 0, 0)))], PRICES)
    missing = aggregate([log(tmp_path / 'missing.jsonl', receipts('missing', 'ceo', {}))], PRICES)
    result = summarize_pairs([dict(git, usage=zero), pf])
    ratio = result['pairs'][0]['comparison']['input_tokens']
    assert ratio['pf_minus_git'] == 10 and ratio['pf_over_git'] is None
    assert ratio['ratio_reason'] == 'zero_git_denominator'
    result = summarize_pairs([dict(git, usage=missing), pf])
    ratio = result['pairs'][0]['comparison']['input_tokens']
    assert ratio['pf_minus_git'] is ratio['pf_over_git'] is None
    assert ratio['missing_groups'] == ['git']
    assert summarize_pairs([git])['pairs'][0]['missing_groups'] == ['pf']
    not_started = dict(run_id='missing-pf', group='pf', seed=11, repeat=1, status='not_started')
    result = summarize_pairs([git, not_started])
    assert result['status_counts']['not_started'] == 1
    assert result['pairs'][0]['comparison']['total_tokens']['missing_groups'] == ['pf']
    assert result['pairs'][0]['comparison']['total_tokens']['pf_over_git'] is None
    failed = dict(not_started, status='technical_failure', usage=None)
    assert summarize_pairs([git, failed])['pairs'][0]['statuses']['pf'] == 'technical_failure'
    with pytest.raises(ValueError, match='exactly one'):
        summarize_pairs([git, git])
    conn.close()


def test_result_uses_last_complete_week_after_partial_advance():
    conn = init_database(':memory:')
    conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES (0,'initial_funding',100,'initial')")
    conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES (7,'operations',-10,'complete week')")
    conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES (8,'operations',-30,'partial advance')")
    args = dict(run_id='pf-run', group='pf', seed=11, repeat=1, day=8, reason='paused', usage={})
    result = summarize_run(conn, **args, last_week_result=SimpleNamespace(day=7, cash=90, mrr=42))
    assert result['day'] == 8
    assert result['business_metrics_day'] == 7
    assert result['cash_at_advance'] == 90
    assert result['mrr'] == 42
    missing = summarize_run(conn, **args)
    assert missing['business_metrics_day'] is None
    assert missing['cash_at_advance'] is missing['mrr'] is None
    conn.close()
