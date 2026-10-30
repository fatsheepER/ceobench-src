from collections import Counter
import json
import os
from pathlib import Path
import shlex
import threading

import pytest

from saas_bench.role_policy import ROLES
from test_preflight_team_flow import ADVANCE, calls, final, make_team, raw, register


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_opening_question_precedes_scripts_and_closing_receives_fresh_full_reports(mode):
    counts, sessions = Counter(), {}
    barriers = {day: threading.Barrier(2) for day in (0, 7)}
    reports = {role: role + '-FULL-REPORT\n' + 'exact supporting evidence\n' * 100 + role + '-END'
               for role in ROLES[1:]}

    def respond(team, role, body):
        day = team.server.tools.current_day
        assert team.test_wire[0]['role'] == 'ceo', 'The first weekly model request must be CEO opening'
        counts[role, day] += 1
        n = counts[role, day]
        session = team.roles[role].identity.session_id
        assert sessions.setdefault((role, day), session) == session
        if day == 7:
            assert session != sessions[role, 0]
        messages = body['messages']
        rendered = json.dumps(messages)
        own_memory = role + '-MEMORY-END'
        assert own_memory in messages[0]['content']
        for peer in ROLES:
            if peer != role:
                assert peer + '-MEMORY-END' not in messages[0]['content']
        if role == 'ceo':
            if n <= 2:
                assert not any((runtime.workspace / 'script-days.txt').exists() and
                    (runtime.workspace / 'script-days.txt').read_text().endswith(str(day) + '\n')
                    for runtime in team.roles.values())
                assert f'Day {day}' in rendered
                if n == 1:
                    status, denied = raw(team, 'ceo', '/next-week', {})
                    assert status == 403, denied
                    status, changed = raw(team, 'ceo', '/call',
                        dict(tool='set_prices', args=dict(A=31 + day)))
                    assert status == 200 and changed['success'], changed
                    return calls(('ask_analyst', dict(role='growth', message='OPENING-QUESTION')))
                assert 'OPENING-ANSWER' in rendered
                status, denied = raw(team, 'ceo', '/next-week', {})
                assert status == 403, denied
                return final(f'CEO-DIRECTIONS-D{day}: preserve cash and investigate capacity.')
            assert n in (3, 4)
            assert f'CEO-DIRECTIONS-D{day}' in rendered
            for analyst in ROLES[1:]:
                assert sum(str(message.get('content', '')).count(reports[analyst]) for message in messages) == 1
            assert sum(str(message.get('content', '')).count('PRIVATE-SCRIPT-ceo') for message in messages) == 1
            user = [message['content'] for message in messages if message['role'] == 'user'][-1]
            assert 'capacity_tier=2' in user
            assert f'price_A={31 + day}' in user
            if n == 3:
                return calls(('write_file', dict(path='closing.txt', content='received exact reports')))
            return calls(('bash', dict(command=ADVANCE)))
        if role == 'growth' and n == 1:
            assert 'OPENING-QUESTION' in messages[-1]['content']
            assert not (team.roles[role].workspace / 'script-days.txt').exists() or day == 7
            return final('OPENING-ANSWER: use the current baseline.')
        formal = 2 if role == 'growth' else 1
        if n == formal:
            assert f'CEO-DIRECTIONS-D{day}' in messages[-1]['content']
            assert 'PRIVATE-SCRIPT-' + role in messages[-1]['content']
            assert counts['ceo', day] == 2
            barriers[day].wait(timeout=10)
            if role == 'ops_finance':
                command = 'python -c ' + shlex.quote(
                    "import novamind_api as n; print(n.infrastructure.set_capacity_tier(2))")
                return calls(('bash', dict(command=command)))
        return final(reports[role])

    with make_team(mode, respond, 'ceo-weekly-opening', opening_responder=respond) as team:
        def dashboard(day, result):
            row = team.server.conn.execute('SELECT price_A, capacity_tier FROM config_history ORDER BY day DESC LIMIT 1').fetchone()
            return f'=== Week {day // 7 + 1} Dashboard (Day {day}) ===\nprice_A={row[0]} capacity_tier={row[1]}'
        team.server.dashboard_callback = dashboard
        for role, runtime in team.roles.items():
            assert runtime.agent.memory_max_chars == 100000
            (runtime.workspace / 'MEMORY.md').write_text('memory\n' * 7000 + role + '-MEMORY-END')
            register(team, role, 'weekly.py',
                "import novamind_api as nm\nwith open('script-days.txt','a') as stream: stream.write(str(nm.vars.current_day)+'\\n')\n"
                + 'print(' + repr('PRIVATE-SCRIPT-' + role) + ')')
        outcome = team.run(stop_after_day=14)
        if target := os.environ.get('CEOBENCH_OPENING_EVIDENCE'):
            evidence = Path(target)
            evidence.mkdir(parents=True, exist_ok=True)
            (evidence / (mode + '-flow.json')).write_text(json.dumps(dict(
                requests=team.test_wire, timeline=team.test_timeline, failure=team.failure,
                failure_detail=getattr(team, 'failure_detail', None)), indent=2))
        assert outcome.reason == 'observation_end', (team.failure, getattr(team, 'failure_detail', None))
        for role, runtime in team.roles.items():
            assert (runtime.workspace / 'script-days.txt').read_text() == '0\n7\n'
            sent = [row for row in rows(runtime.usage.path) if row['event'] == 'http_request']
            actual = [row['body'] for row in team.test_wire if row['role'] == role]
            assert [row['body'] for row in sent] == actual
            assert len(sent) == (8 if role == 'ceo' else 4)
            per_week = len(sent) // 2
            first, second = sent[:per_week], sent[per_week:]
            assert len({row['context_id'] for row in first}) == len({row['context_id'] for row in second}) == 1
            assert first[0]['context_id'] != second[0]['context_id']
            for week in (first, second):
                for before, after in zip(week, week[1:]):
                    assert after['message_ids'][:len(before['message_ids'])] == before['message_ids']
            assert not set(first[-1]['message_ids']) & set(second[0]['message_ids'])


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('endpoint', ['observation', 'D56', 'natural', 'bankruptcy'])
def test_endpoint_checked_before_opening_or_registered_scripts(mode, endpoint):
    def unexpected(*args):
        raise AssertionError('No model may run past the endpoint')

    with make_team(mode, unexpected, 'opening-endpoint', total_days=56 if endpoint == 'natural' else 500) as team:
        for role in ROLES:
            register(team, role, 'must-not-run.py', "open('unexpected.txt','w').write('ran')")
        day = 56 if endpoint in ('D56', 'natural') else 7
        team.server.tools.current_day = day
        if endpoint == 'bankruptcy':
            team.server.conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES(?, 'operations', -1e12, 'bankruptcy')", (day,))
            team.server.conn.commit()
        stop = 56 if endpoint != 'observation' else 7
        outcome = team.run(stop_after_day=stop)
        assert outcome.reason == ('observation_end' if endpoint in ('observation', 'D56') else 'natural_end')
        assert not team.test_wire
        assert all(not (runtime.workspace / 'unexpected.txt').exists() for runtime in team.roles.values())


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_closing_still_requires_weekly_advance(mode):
    counts = Counter()

    def respond(team, role, body):
        counts[role] += 1
        if role == 'ceo':
            return final('Opening directions.' if counts[role] == 1 else 'Closing without advance.')
        return final(role + ' report')

    with make_team(mode, respond, 'closing-needs-advance', opening_responder=respond) as team:
        outcome = team.run(stop_after_day=7)
        assert outcome.reason == 'paused'
        assert team.failure == 'model_attempt_limit'
        assert counts == dict(ceo=5, growth=1, ops_finance=1)
        assert team.server.tools.current_day == 0


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('failed_phase', ['ceo_opening', 'ceo'])
def test_checkpoint_after_model_failure_does_not_duplicate_accepted_input(tmp_path, mode, failed_phase):
    import httpx
    from openai import OpenAI
    from saas_bench.multi_agent import MultiAgentRuntime
    from saas_bench.team_prompts import CEO_OPENING, CEO_CLOSING
    from test_preflight_team_checkpoint import dispose, fast_simulator
    from test_preflight_team_flow import completion

    def answer(role, body):
        if role != 'ceo':
            return final(role + '-RECOVERY-REPORT')
        if any(CEO_CLOSING in str(message.get('content', '')) for message in body['messages']):
            return calls(('bash', dict(command=ADVANCE)))
        return final('RECOVERY-DIRECTIONS')

    def fail(team, role, body):
        if role == 'ceo' and team.phase == failed_phase:
            return httpx.Response(400, json=dict(error=dict(message='injected rejected request', type='invalid_request')))
        return answer(role, body)

    wire = []
    def clients(role):
        def respond(request):
            body = json.loads(request.content)
            wire.append(dict(role=role, body=body))
            return completion(answer(role, body), body.get('stream', False))
        return OpenAI(api_key='OFFLINE-ONLY', base_url='https://opencode.ai/zen/go/v1', max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    with make_team(mode, fail, 'accepted-input-recovery', opening_responder=fail) as original:
        original.server.simulator = fast_simulator(original.server.conn, original.server.tools.config,
                                                   original.server.tools.rng)
        for role in ROLES:
            register(original, role, 'weekly.py',
                "import novamind_api as nm\nwith open('script-days.txt','a') as stream: stream.write(str(nm.vars.current_day)+'\\n')\n")
        assert original.run(stop_after_day=7).reason == 'paused'
        assert original.failure == 'model_request_failed'
        snapshot = original.checkpoint()
        resumed = MultiAgentRuntime.restore(snapshot, tmp_path / 'resumed', client_factory=clients,
                                             simulator_factory=fast_simulator)
        try:
            assert resumed.run(stop_after_day=7).reason == 'observation_end', resumed.failure
            closing = [row['body']['messages'] for row in wire if row['role'] == 'ceo'][-1]
            for prompt in (CEO_OPENING, CEO_CLOSING):
                assert sum(prompt in str(message.get('content', '')) for message in closing) == 1
            for role in ROLES[1:]:
                assert sum(str(message.get('content', '')).count(role + '-RECOVERY-REPORT') for message in closing) == 1
            assert all(message.status == 'delivered' for message in resumed.messages)
            assert all((runtime.workspace / 'script-days.txt').read_text() == '0\n' for runtime in resumed.roles.values())
        finally:
            dispose(resumed)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_batch_wrapper_and_opening_ask_and_answers_ready_checkpoint_recovery(tmp_path, mode):
    import httpx
    from numpy.random import default_rng
    from openai import OpenAI
    from saas_bench import team_batch as batch
    from saas_bench.config import BenchmarkConfig
    from saas_bench.database import init_database
    from saas_bench.multi_agent import MultiAgentRuntime
    from saas_bench.tools import AgentTools
    from test_preflight_team_checkpoint import PRICING, dispose, fast_simulator
    from test_preflight_team_flow import completion

    wire = []

    def clients(role):
        def respond(request):
            body = json.loads(request.content)
            wire.append(dict(role=role, body=body))
            messages = body['messages']
            if role == 'ceo':
                opened = any(message['role'] == 'assistant' and message.get('content') == 'OPENING-DIRECTIONS'
                             for message in messages)
                if opened:
                    message = calls(('bash', dict(command=ADVANCE)))
                elif any(message.get('name') == 'ask_analyst' for message in messages):
                    message = final('OPENING-DIRECTIONS')
                else:
                    message = calls(('ask_analyst', dict(role='growth', message='EARLY-QUESTION')))
            else:
                message = final('EARLY-ANSWER' if 'Team request |' in messages[-1]['content']
                                and '\nEARLY-QUESTION\n' in messages[-1]['content']
                                else role + '-EXACT-FORMAL-REPORT')
            return completion(message, body.get('stream', False))
        return OpenAI(api_key='OFFLINE-ONLY', base_url='https://opencode.ai/zen/go/v1', max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    directory = tmp_path / 'run' / 'attempt-001'
    directory.mkdir(parents=True)
    conn = init_database(':memory:')
    config = BenchmarkConfig(seed=123)
    simulator = fast_simulator(conn, config, default_rng(123))
    simulator.initialize()
    tools = AgentTools(conn, 0, directory / 'host', rng=simulator.rng, config=config, seed=123)
    original = MultiAgentRuntime(directory / 'runtime', tools, simulator=simulator, conn=conn,
        client_factory=clients, mode=mode, model_pricing=PRICING)
    entry = dict(run_id='opening-' + mode, output_dir=str(directory.parent), group=mode,
                 seed=123, repeat=1, stop_after_day=14, category='formal')
    record = dict(attempt=1, status='ready', checkpoint=None)
    samples = {}
    try:
        for role in ROLES:
            register(original, role, 'weekly.py',
                "import novamind_api as nm\nwith open('script-days.txt','a') as stream: stream.write(str(nm.vars.current_day)+'\\n')\n")
        result = batch._execute(original, dict(pricing=PRICING), entry, directory, record, tmp_path / 'STOP')
        assert result['status'] == 'finished', result.get('runtime_failure')
        assert result['reason'] == 'observation_end'
        opening_day7 = next(row['body']['messages'][-1]['content'] for row in wire if row['role'] == 'ceo'
            and len(row['body']['messages']) == 2 and 'Day 7' in row['body']['messages'][-1]['content'])
        assert "This Week's Metrics" in opening_day7
        assert 'Cash:' in opening_day7 and 'Current Config' in opening_day7
        assert not (directory / 'HOLD').exists()
        snapshots = [(path.parent, json.loads(path.read_text()))
                     for path in (original.root / 'checkpoints').glob('*/team.json')]
        for stage in ('ceo_opening', 'ceo'):
            snapshot, state = next((path, state) for path, state in snapshots if state['stage'] == stage and state['day'] == 0)
            assert state['week']['ceo_started'] == (stage == 'ceo_opening')
            if stage == 'ceo':
                formal = [message for message in state['messages'] if message['text'] != 'EARLY-QUESTION']
                assert len(formal) == 2 and all(message['status'] == 'completed' for message in formal)
            resumed = MultiAgentRuntime.restore(snapshot, tmp_path / ('restored-' + stage),
                client_factory=clients, simulator_factory=fast_simulator)
            try:
                before = len(wire)
                outcome = resumed.run(stop_after_day=7)
                assert outcome.reason == 'observation_end', (resumed.failure, getattr(resumed, 'failure_detail', None))
                sent = wire[before:]
                assert all(resumed.roles[role].identity.session_id == state['roles'][role]['identity']['session_id']
                           for role in ROLES)
                assert all((runtime.workspace / 'script-days.txt').read_text() == '0\n'
                           for runtime in resumed.roles.values())
                assert len(resumed.messages) == 3
                assert all(message.status == 'delivered' for message in resumed.messages)
                ceo_requests = [row['body'] for row in sent if row['role'] == 'ceo']
                closing = ceo_requests[-1]['messages']
                assert sum(message.get('content') == 'OPENING-DIRECTIONS' for message in closing) == 1
                for role in ROLES[1:]:
                    assert sum(str(message.get('content', '')).count(role + '-EXACT-FORMAL-REPORT')
                               for message in closing) == 1
                if stage == 'ceo':
                    assert all(row['role'] == 'ceo' for row in sent)
                samples[stage] = dict(stage=state['stage'], week=state['week'], requests=sent,
                    sessions={role: runtime.identity.session_id for role, runtime in resumed.roles.items()},
                    final_stage=resumed._stage)
            finally:
                dispose(resumed)
        if target := os.environ.get('CEOBENCH_OPENING_EVIDENCE'):
            evidence = Path(target)
            evidence.mkdir(parents=True, exist_ok=True)
            (evidence / (mode + '-recovery.json')).write_text(json.dumps(samples, indent=2))
            (evidence / (mode + '-attempt.json')).write_text(json.dumps(result, indent=2))
    finally:
        dispose(original)
