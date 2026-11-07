from collections import Counter
import json
import os
from pathlib import Path

import httpx
from openai import OpenAI
import pytest

from saas_bench.agent_capabilities import AgentCapabilities
from saas_bench.database import save_predictions
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.role_policy import ROLES
from test_preflight_team_checkpoint import dispose, fast_simulator
from test_preflight_team_flow import ADVANCE, calls, completion, final, make_team, raw


PLANS = ('plans/commercial.md', 'plans/service.md', 'plans/cash.md', 'plans/decisions.md')


def options(mode):
    return dict(workspace_layout='shared-v1', capabilities=AgentCapabilities.round_three(mode),
                workspace_protocol='snapshot-v1' if mode == 'pf' else 'direct-v1')


def evidence(name, value):
    if target := os.environ.get('CEOBENCH_R5_PROMPT_EVIDENCE'):
        path = Path(target) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def requests(runtime, cutoff=0):
    return [json.loads(line)['body'] for line in runtime.usage.path.read_text().splitlines()[cutoff:]
            if json.loads(line)['event'] == 'http_request']


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_shared_plans_are_empty_templates_readable_in_actual_inputs(mode):
    with make_team(mode, lambda *args: calls(('bash', dict(command='cat ' + ' '.join(PLANS)))),
                   'r5-plan-templates', **options(mode)) as team:
        for role, runtime in team.roles.items():
            action = runtime.agent.act('Read the shared plan templates.', 0, False, {'day': 0})
            output = runtime.executor.execute(action.tool, action.arguments)
            runtime.agent.act(output, 0, False, {'day': 0})
            body = requests(runtime)[-1]
            evidence(mode + '-' + role + '-templates.json', body)
            for path in PLANS:
                template = runtime.workspace / path
                assert template.is_file(), path
                content = template.read_text()
                assert content in body['messages'][-1]['content']
                assert not any(char.isdigit() for char in content)
                assert path in body['messages'][0]['content']
                assert path in (runtime.workspace / 'docs/workspace-history.md').read_text()


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_plain_messages_first_restored_and_all_role_questions_reach_model_usage(tmp_path, mode):
    counts = Counter()
    wires = []
    holder = {}
    reports = {'growth': 'GROWTH-MESSAGE\n' + '自由消息 exact spacing  \n' * 80 + 'GROWTH-END',
               'ops_finance': 'See plans/cash.md. OPS-MESSAGE-END'}

    def answer(team, role, body):
        day = team.server.tools.current_day
        counts[day, role, 'requests'] += 1
        assert counts[day, role, 'requests'] <= 20, (role, day, team.phase, body['messages'][-1])
        if team._workers[role].question is not None:
            return final('ANSWER-' + role)
        if role == 'ceo':
            if team.phase == 'ceo_opening':
                counts[day, 'opening'] += 1
                if counts[day, 'opening'] == 1:
                    return calls(*[('ask_analyst', dict(role=peer, message='OPENING-QUESTION-' + peer))
                                   for peer in ROLES[1:]])
                return final('Coordinate current plans and complete weekly analysis.')
            return calls(('bash', dict(command=ADVANCE)))
        counts[day, role] += 1
        if counts[day, role] == 1:
            return calls(('ask_agent', dict(role='ceo', message='WEEKLY-QUESTION-' + role)))
        if counts[day, role] == 2 and role == 'ops_finance':
            return calls(('write_file', dict(path='plans/cash.md', content='BASELINE-FILE-ONLY\n' +
                '7: 123456 120000 130000\n28: 145678 140000 150000\n84: 167890 160000 170000\n182: 189012 180000 190000\n')))
        return final(reports[role])

    def client(role):
        def respond(request):
            body = json.loads(request.content)
            wires.append(dict(role=role, body=body))
            return completion(answer(holder['team'], role, body), body.get('stream', False))
        return OpenAI(api_key='OFFLINE-ONLY', base_url='https://opencode.ai/zen/go/v1', max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    def check(team, wire, phase, cutoffs=None):
        for role, runtime in team.roles.items():
            sent = requests(runtime, (cutoffs or {}).get(role, 0))
            actual = [r['body'] for r in wire if r['role'] == role]
            assert sent == actual and sent
            evidence(mode + '-' + phase + '-' + role + '.json', sent)
            system = sent[0]['messages'][0]['content']
            for path in PLANS:
                assert path in system
            assert 'Only memory/' + role + '.md is auto-loaded' in system
            for stale in ('full handoffs', 'full analyst handoffs', 'In your handoff, distinguish',
                          'Include data coverage and units', 'Report actions taken, returned results'):
                assert stale not in json.dumps(sent)
            assert ('pf workspace' in system) == (mode == 'pf')
            assert 'text_create' not in system
            assert any('Team request |' in str(m.get('content', '')) for b in sent for m in b['messages'])
            if role != 'ceo':
                assert any('Choose its content, length and format.' in str(m.get('content', ''))
                           for b in sent for m in b['messages'])
        closing = [r['body'] for r in wire if r['role'] == 'ceo' and
                   any('OPS-MESSAGE-END' in str(m.get('content', '')) for m in r['body']['messages'])][-1]
        for report in reports.values():
            assert sum(str(m.get('content', '')).count(report) for m in closing['messages']) == 1
        rendered = json.dumps(closing)
        assert 'BASELINE-FILE-ONLY' not in rendered and '123456' not in rendered
        for message in team.messages:
            if message.kind != 'weekly_task':
                continue
            handoff = team._handoff(message)
            assert handoff.split('\n\n', 1)[1] == message.reply == reports[message.receiver]
            assert 'Snapshot commit: ' in handoff
            assert team._git(team.roles[message.receiver], 'rev-parse', message.handoff_commit)

    with make_team(mode, answer, 'r5-input-flow', opening_responder=answer, **options(mode)) as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        team.server.simulator.initialize(resume=True)
        assert team.run(stop_after_day=7).reason == 'observation_end', getattr(team, 'failure_detail', team.failure)
        check(team, team.test_wire, 'first')
        checkpoint = team.checkpoint()
        restored = MultiAgentRuntime.restore(checkpoint, tmp_path / 'restored', client_factory=client,
                                             simulator_factory=fast_simulator)
        holder['team'] = restored
        try:
            assert restored.server.simulator.current_day == 7
            cutoffs = {role: len(runtime.usage.path.read_text().splitlines()) if runtime.usage.path.exists() else 0
                       for role, runtime in restored.roles.items()}
            assert restored.run(stop_after_day=14).reason == 'observation_end', getattr(restored, 'failure_detail', restored.failure)
            check(restored, wires, 'restored', cutoffs)
        finally:
            dispose(restored)


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('day', [7, 28, 56])
def test_due_forecasts_same_numbers_for_ceo_ops_and_no_endpoint_requests(mode, day):
    def answer(team, role, body):
        if role == 'ceo':
            return final('Review forecasts.') if team.phase == 'ceo_opening' else calls(('bash', dict(command=ADVANCE)))
        return final('Weekly message.')

    with make_team(mode, answer, 'r5-due-forecasts', opening_responder=answer, **options(mode)) as team:
        conn = team.server.conn
        team.server.tools.current_day = day
        conn.execute("DELETE FROM ledger")
        conn.executemany('INSERT INTO ledger(day,category,amount,note) VALUES(?,?,?,?)',
            [(0, 'initial_funding', 1000000, 'initial'), (day, 'operations', -100, 'settled'),
             (day, 'market_research', -200, 'after advance'), (day + 1, 'operations', -900, 'future')])
        for horizon in (7, 28):
            if day >= horizon:
                save_predictions(conn, day - horizon, {horizon: {'cash': dict(point=900000 + horizon, lower=800000, upper=1100000)}}, 1)
        save_predictions(conn, day - 7, {28: {'cash': dict(point=777777, lower=1, upper=2000000)}}, 2)
        save_predictions(conn, 0, {84: {'cash': dict(point=888888, lower=1, upper=2000000)}}, 3)
        conn.commit()
        for role in ROLES:
            status, result = raw(team, role, '/daily-scripts', dict(name='scripts/endpoint.py',
                content="import novamind_api\nprint('REGISTERED-BATCH-RAN')"))
            assert status == 200 and result['success']
        outcome = team.run(stop_after_day=56 if day == 56 else day + 7)
        assert outcome.reason == 'observation_end', getattr(team, 'failure_detail', team.failure)
        if day == 56:
            assert not team.test_wire
            assert all(not team.server.role_script_results[role] for role in ROLES)
            return
        summary = {}
        for role in ROLES:
            sent = requests(team.roles[role])
            evidence(mode + '-due-' + str(day) + '-' + role + '.json', sent)
            first = sent[0]['messages'][-1]['content']
            if role == 'growth':
                assert 'Due CEO cash forecasts' not in first
                continue
            assert 'Due CEO cash forecasts' in first
            section = first.split('Due CEO cash forecasts', 1)[1].split('\n\n', 1)[0]
            summary[role] = section
            assert '999900' in section and '999700' not in section
            assert '900007' in section
            assert ('900028' in section) == (day >= 28)
            assert '777777' not in section and '888888' not in section
        assert summary['ceo'] == summary['ops_finance']


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('boundary', ['ask_completed', 'answers_ready'])
def test_due_review_survives_opening_and_handoff_checkpoint_without_duplicate(tmp_path, mode, boundary):
    checkpoints, wire, counts = [], [], Counter()
    holder = {}

    def answer(team, role, body):
        counts[id(team), role] += 1
        assert counts[id(team), role] < 15
        if team._workers[role].question:
            return final('OPENING-OPS-ANSWER')
        if role == 'ceo':
            if team.phase == 'ceo_opening':
                if 'OPENING-OPS-ANSWER' not in json.dumps(body['messages']):
                    return calls(('ask_analyst', dict(role='ops_finance', message='Review the due forecasts.')))
                return final('Review the cash plan.')
            return calls(('bash', dict(command=ADVANCE)))
        return final('Ordinary weekly message.')

    def client(role):
        def respond(request):
            body = json.loads(request.content)
            wire.append(dict(role=role, body=body))
            return completion(answer(holder['team'], role, body), body.get('stream', False))
        return OpenAI(api_key='OFFLINE-ONLY', base_url='https://opencode.ai/zen/go/v1', max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    with make_team(mode, answer, 'r5-due-recovery', opening_responder=answer, **options(mode)) as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        team.server.simulator.initialize(resume=True)
        team.server.tools.current_day = team.server.simulator.current_day = 28
        for submit, horizon in ((0, 28), (21, 7)):
            save_predictions(team.server.conn, submit, {horizon: {'cash': dict(point=1234567, lower=1200000, upper=1300000)}}, 1)
        team.server.conn.commit()

        def save(runtime, name):
            if name == boundary and not checkpoints:
                checkpoints.append((runtime.checkpoint(), runtime._week['forecast_review']))
        team.checkpoint_callback = save
        assert team.run(stop_after_day=35).reason == 'observation_end', getattr(team, 'failure_detail', team.failure)
        assert len(checkpoints) == 1
        selected, summary = checkpoints[0]
        restored = MultiAgentRuntime.restore(selected, tmp_path / 'restored', client_factory=client,
                                             simulator_factory=fast_simulator)
        holder['team'] = restored
        try:
            cutoffs = {role: len(runtime.usage.path.read_text().splitlines()) if runtime.usage.path.exists() else 0
                       for role, runtime in restored.roles.items()}
            assert restored._week['forecast_review'] == summary
            assert restored.run(stop_after_day=35).reason == 'observation_end', getattr(restored, 'failure_detail', restored.failure)
            for role, runtime in restored.roles.items():
                sent = requests(runtime, cutoffs[role])
                assert sent == [row['body'] for row in wire if row['role'] == role]
                for body in sent:
                    count = sum(str(message.get('content', '')).count(summary) for message in body['messages'])
                    assert count == (0 if role == 'growth' else 1)
                evidence(mode + '-due-recovery-' + boundary + '-' + role + '.json', sent)
            for role in ('ceo', 'ops_finance'):
                assert sum(str(message.content).count(summary) for message in restored.roles[role].agent.conversation) == 1
            assert all(row.status in ('delivered', 'completed') for row in restored.messages)
        finally:
            dispose(restored)
