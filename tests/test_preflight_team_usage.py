import copy
import json

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


def test_real_sdk_retry_counts_known_failed_http_usage_once(tmp_path, monkeypatch):
    monkeypatch.setattr('time.sleep', lambda _: None)
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(503 if len(requests) == 1 else 200,
            json=body(5, 1, 2, 1, bill=.01) if len(requests) == 1 else body(bill=.02))
    recorder = ModelUsage(tmp_path / 'ceo.jsonl', 'ceo', PRICES['rates'])
    client = recorder.attach(OpenAI(api_key='offline', max_retries=1,
        http_client=httpx.Client(transport=httpx.MockTransport(handle))))
    request = dict(model='test-model', messages=[dict(role='user', content='offline')])
    try:
        recorder.call('chat', request, lambda: client.chat.completions.create(**request), day=0)
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


def test_returned_model_mismatch_is_rejected_after_raw_receipt(tmp_path):
    recorder = ModelUsage(tmp_path / 'model.jsonl', 'ceo', PRICES['rates'], expected_model='frozen-model')
    with pytest.raises(ValueError, match='frozen configuration'):
        recorder.call('chat', {'model': 'frozen-model'}, lambda: body(), day=0)
    row = json.loads(recorder.path.read_text().splitlines()[-1])
    assert row['response']['model'] == 'test-model' and row['usage']['input_tokens'] == 10
    assert row['error'] == 'ValueError' and recorder.summary['errors'] == 1
    assert row['cost_usd'] == pytest.approx(.0113)
    recorder.expected_model = 'test-model'
    assert recorder.call('chat', {}, lambda: body(), day=7)['model'] == 'test-model'
    with pytest.raises(ValueError, match='object'):
        recorder.call('chat', {}, lambda: None, day=14)
    assert json.loads(recorder.path.read_text().splitlines()[-1])['response'] is None


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
        reason='observation_end', usage=usage)
    git = summarize_run(conn, **args)
    assert git['status'] == 'reached_d112'
    assert git['cash_at_advance'] == 90
    assert git['predictions']['four_week_mean_interval_score'] == 10
    assert git['unexpired_predictions'][0]['target_day'] == 168
    assert git['unexpired_predictions'][0]['status'] == 'pending'
    pf = summarize_run(conn, **dict(args, run_id='pf-run', group='pf'))
    result = summarize_pairs([git, pf])
    assert result['complete_d112_pairs'] == 1
    assert result['pairs'][0]['comparison']['total_tokens']['pf_minus_git'] == 0
    assert result['pairs'][0]['comparison']['total_tokens']['pf_over_git'] == 1
    for reason, status in [('paused', 'paused'), ('failed', 'technical_failure')]:
        run = summarize_run(conn, **dict(args, reason=reason, day=28))
        assert run['status'] == status and len(run['unexpired_predictions']) == 2
    conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES (14,'operations',-200,'bankruptcy')")
    bankrupt = summarize_run(conn, **dict(args, run_id='bankrupt', seed=12, day=14, reason='natural_end'))
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
