import json

import httpx
import pytest

from saas_bench import model_usage
from saas_bench.role_policy import ROLES
from saas_bench.team_usage import aggregate
from test_preflight_team_batch import PRICES
from test_preflight_team_flow import ADVANCE, calls, final, make_team


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def register_source(team, role):
    runtime = team.roles[role]
    runtime.executor.execute('write_file', dict(path='facts.txt', content='original evidence\n'))
    commit = team._snapshot_history(runtime, 'test: initial weekly evidence')
    registered = runtime.executor.execute('text_create', dict(text='Current weekly evidence',
        objects=[dict(kind='metric', id='cash')], references=[dict(cite=f'facts.txt@{commit[:7]}')],
        applies='0-', reason='offline weekly check fixture'))
    assert registered.startswith('Registered r1.1'), registered
    runtime.executor.execute('write_file', dict(path='facts.txt', content='changed evidence\n'))


@pytest.mark.parametrize('has_summary', [False, True])
@pytest.mark.parametrize('retry', [False, True])
def test_git_weekly_checks_have_independent_results_and_actual_model_delivery(
        monkeypatch, has_summary, retry):
    attempts = {}

    def respond(team, role, body):
        key = role, team.server.tools.current_day
        attempts[key] = attempts.get(key, 0) + 1
        if retry and key == ('growth', 0) and attempts[key] == 1:
            raise httpx.ConnectError('offline model connection interrupted')
        return calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final('Evidence checked.')

    monkeypatch.setattr(model_usage, 'MODEL_RETRY_DELAYS', (0, 0, 0, 0))
    with make_team('git', respond, 'git-weekly-receipts') as team:
        if has_summary:
            for role in ROLES:
                register_source(team, role)
        assert team.run(stop_after_day=14).day == 14
        for role, runtime in team.roles.items():
            audit = [row for row in rows(runtime.audit.path) if row['kind'] == 'git_weekly_check']
            started = [row for row in audit if row['status'] == 'started']
            completed = [row for row in audit if row['status'] == 'succeeded']
            assert len(started) == len(completed) == 2
            assert {row['day'] for row in completed} == {0, 7}
            assert {row['check_id'] for row in completed} == {row['event_id'] for row in started}
            usage = rows(runtime.usage.path)
            assembled = [row for row in rows(runtime.audit.path) if row['kind'] == 'git_weekly_check_delivery']
            assert len(assembled) == 2
            assert aggregate([runtime.usage.path], PRICES)['calls'] == 2
            by_call = {}
            for row in usage:
                if row['event'] == 'request':
                    by_call[row['call_id']] = row['day']
                if row['event'] not in ('http_request', 'http_response', 'http_error'):
                    continue
                receipt = row['weekly_check']
                result = next(item for item in completed if item['day'] == by_call[row['call_id']])
                assert receipt['check_id'] == result['check_id']
                assert receipt['result_event_id'] == result['event_id']
                assert receipt['day'] == result['day']
                assert receipt['outcome'] == ('summary_ready' if has_summary else 'not_applicable')
                assert receipt['delivery_state'] == ({'http_request': 'send_attempted',
                    'http_response': 'response_received', 'http_error': 'send_unknown'}[row['event']]
                    if has_summary else 'not_applicable')
                if row['event'] == 'http_request':
                    text = '\n'.join(message['content'] for message in row['body']['messages']
                                     if isinstance(message.get('content'), str))
                    assert receipt['check_id'] not in text and result['event_id'] not in text
                    if has_summary:
                        assert result['summary'] in text
            expected = 3 if role == 'growth' and retry else 2
            assert sum(row['event'] == 'http_request' for row in usage) == expected
            assert len({row['check_id'] for row in completed}) == 2


def test_git_weekly_check_failure_has_receipt_without_model_send(monkeypatch):
    with make_team('git', lambda *args: final('checked'), 'git-weekly-failure') as team:
        runtime = team.roles['ops_finance']

        def fail():
            raise ValueError('offline invalid registration file')

        monkeypatch.setattr(runtime.registry, '_load', fail)
        assert team.run(stop_after_day=7).reason == 'failed'
        audit = [row for row in rows(runtime.audit.path) if row['kind'] == 'git_weekly_check']
        assert [row['status'] for row in audit] == ['started', 'failed']
        assert audit[1]['check_id'] == audit[0]['event_id']
        assert audit[1]['error'] == 'ValueError'
        assert audit[1]['delivery_state'] == 'not_assembled'
        assert not any(row['event'] == 'http_request' for row in rows(runtime.usage.path))
        assert not any(row['role'] == 'ops_finance' for row in team.test_wire)


def test_git_weekly_check_does_not_claim_delivery_without_summary_in_payload():
    with make_team('git', lambda *args: final('checked'), 'git-weekly-omitted') as team:
        register_source(team, 'growth')
        runtime = team.roles['growth']
        team._observation('growth', 'public dashboard', '')
        request = dict(model=runtime.agent.model, messages=[dict(role='user', content='Unrelated input.')])
        runtime.agent._request_model('chat', request,
            lambda: runtime.agent.client.chat.completions.create(**request))
        deliveries = [row['weekly_check'] for row in rows(runtime.usage.path)
                      if row['event'] in ('http_request', 'http_response')]
        assert len(deliveries) == 2
        assert all(receipt['delivery_state'] == 'not_in_request' for receipt in deliveries)


def test_git_weekly_check_receipt_survives_answers_ready_checkpoint(tmp_path):
    from saas_bench.multi_agent import MultiAgentRuntime
    from test_preflight_team_checkpoint import clients, dispose, fast_simulator, make_team as checkpoint_team
    team = checkpoint_team(tmp_path / 'continuous', 'git')
    forks = []

    def save(runtime, boundary):
        if boundary == 'answers_ready' and not forks:
            snapshot = runtime.checkpoint()
            forks.append(MultiAgentRuntime.restore(snapshot, tmp_path / 'restored',
                client_factory=clients(), simulator_factory=fast_simulator))

    try:
        for role in ROLES:
            register_source(team, role)
        team.checkpoint_callback = save
        assert team.run(stop_after_day=7).reason == 'observation_end', team.failure
        restored = forks[0]
        original = team.roles['ceo'].usage.weekly_check
        assert restored.roles['ceo'].usage.weekly_check == original
        assert restored.run(stop_after_day=7).reason == 'observation_end', restored.failure
        requests = [row for row in rows(restored.roles['ceo'].usage.path) if row['event'] == 'http_request']
        assert requests
        assert all(row['weekly_check']['check_id'] == original['receipt']['check_id'] for row in requests)
        assert all(row['weekly_check']['delivery_state'] == 'send_attempted' for row in requests)
    finally:
        for runtime in forks:
            dispose(runtime)
        dispose(team)
