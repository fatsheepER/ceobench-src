from contextlib import closing
import json

import pytest

from saas_bench.role_policy import ROLES
from test_preflight_team_flow import ADVANCE, calls, final, make_team, register


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def register_changed_finding(team, role, mode):
    runtime = team.roles[role]
    written = runtime.executor.execute('write_file', dict(path='facts.txt', content='original evidence\n'))
    assert not written.startswith('Error:'), written
    if mode == 'git':
        commit = team._snapshot_history(runtime, 'test: original finding evidence')
        cite = 'facts.txt@' + commit[:7]
    else:
        cite = 'facts.txt@v1'
    result = runtime.executor.execute('text_create', dict(text='OLD-FINDING-' + role,
        objects=[dict(kind='metric', id='cash')], references=[dict(cite=cite)],
        applies='0-', reason='retained historical finding'))
    assert result.startswith(f'Registered {role}:r1.1'), result
    changed = runtime.executor.execute('write_file', dict(path='facts.txt', content='changed evidence\n'))
    assert not changed.startswith('Error:'), changed


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_new_week_delivers_dashboard_scripts_and_full_handoffs_without_checks(mode):
    reports = {role: role + '-FULL-REPORT\n' + 'supporting evidence\n' * 100 + role + '-REPORT-END'
               for role in ROLES[1:]}

    def respond(team, role, body):
        if role == 'ceo':
            return calls(('bash', dict(command=ADVANCE)))
        return final(reports[role])

    with make_team(mode, respond, 'week-start-no-checks') as team:
        for role in ROLES:
            register_changed_finding(team, role, mode)
            register(team, role, 'report.py', 'print(' + repr('PRIVATE-SCRIPT-' + role) + ')')
        outcome = team.run(stop_after_day=14)
        assert outcome.reason == 'observation_end', team.failure
        assert len(team.test_wire) == 6
        for role, runtime in team.roles.items():
            usage = rows(runtime.usage.path)
            sent = [row for row in usage if row['event'] == 'http_request']
            actual = [row['body'] for row in team.test_wire if row['role'] == role]
            assert [row['body'] for row in sent] == actual
            assert len(sent) == 2
            for row, day in zip(sent, (0, 7)):
                user = next(message['content'] for message in row['body']['messages']
                            if message['role'] == 'user')
                assert f'=== Week {day // 7 + 1} Dashboard (Day {day}) ===' in user
                assert 'PRIVATE-SCRIPT-' + role in user
                assert 'OLD-FINDING-' not in user
                assert 'Check of your registered texts' not in user
                if role == 'ceo':
                    for analyst in ROLES[1:]:
                        assert reports[analyst] in user
                        request = next(message for message in team.messages
                                       if message.receiver == analyst and message.day == day)
                        assert team._handoff(request) in user
            assert all('weekly_check' not in row for row in usage)
            audit = rows(runtime.audit.path)
            assert not any(row['kind'] in ('git_weekly_check', 'git_weekly_check_delivery') for row in audit)
            executions = [row for row in audit if row['kind'] == 'registered_script_execution']
            assert len(executions) == 2
            assert all(row['status'] == 'succeeded' for row in executions)
            assert 'OLD-FINDING-' + role in runtime.registry.path.read_text()
            if runtime.store:
                with closing(runtime.store.connect()) as conn:
                    kinds = [json.loads(row[0])['kind'] for row in conn.execute('SELECT request FROM requests')]
                assert not any(kind in ('pf_weekly_check', 'pf_refresh', 'pf_dependencies') for kind in kinds)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_answers_ready_checkpoint_keeps_observation_without_check_state(tmp_path, mode):
    from saas_bench.multi_agent import MultiAgentRuntime
    from test_preflight_team_checkpoint import clients, dispose, fast_simulator, make_team as checkpoint_team

    original = checkpoint_team(tmp_path / 'continuous', mode)
    snapshots = []
    restored = None

    def save(runtime, boundary):
        if boundary == 'answers_ready' and not snapshots:
            snapshots.append(runtime.checkpoint())

    try:
        for role in ROLES:
            register_changed_finding(original, role, mode)
        original.checkpoint_callback = save
        assert original.run(stop_after_day=7).reason == 'observation_end', original.failure
        state = json.loads((snapshots[0] / 'team.json').read_text())
        assert all('weekly_check' not in role for role in state['roles'].values())
        assert 'OLD-FINDING-' not in state['week']['observation']
        assert 'Check of your registered texts' not in state['week']['observation']
        restored = MultiAgentRuntime.restore(snapshots[0], tmp_path / 'restored',
            client_factory=clients(), simulator_factory=fast_simulator)
        initial_requests = list(restored.messages)
        assert len(initial_requests) == 2
        for request in initial_requests:
            assert restored._handoff(request) in restored._week['observation']
        assert restored.run(stop_after_day=7).reason == 'observation_end', restored.failure
        sent = [row for row in rows(restored.roles['ceo'].usage.path) if row['event'] == 'http_request']
        assert sent
        for row in sent:
            assert 'weekly_check' not in row
            user = next(message['content'] for message in row['body']['messages'] if message['role'] == 'user')
            assert 'Dashboard' in user
            assert 'OLD-FINDING-' not in user
            assert 'Check of your registered texts' not in user
            for request in initial_requests:
                assert restored._handoff(request) in user
        for role in ROLES:
            assert (restored.roles[role].workspace / 'script-days.txt').read_text() == '0\n'
    finally:
        if restored is not None:
            dispose(restored)
        dispose(original)
