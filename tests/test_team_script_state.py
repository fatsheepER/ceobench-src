from dataclasses import asdict
from collections import Counter
import json
import os
from pathlib import Path

import pytest

from saas_bench.role_policy import ROLES
from test_preflight_team_checkpoint import fast_simulator
from test_preflight_team_flow import ADVANCE, calls, final, make_team


def register_cli(team, role, name, code):
    executor = team.roles[role].executor
    assert 'Error:' not in executor.execute('write_file', dict(path=name, content=code))
    output = executor.execute('bash', dict(command='./novamind-operation register-daily-script ' + name))
    assert 'registered' in output.lower(), output
    assert 'Error:' not in executor.execute('write_file', dict(path=name, content="raise AssertionError('edited source ran')"))


def save_evidence(team, label, **facts):
    if target := os.environ.get('CEOBENCH_SCRIPT_STATE_EVIDENCE'):
        root = Path(target)
        root.mkdir(parents=True, exist_ok=True)
        usage = {role: [json.loads(line) for line in runtime.usage.path.read_text().splitlines()]
                 if runtime.usage.path.exists() else [] for role, runtime in team.roles.items()}
        (root / (team.configuration['mode'] + '-' + label + '.json')).write_text(json.dumps(dict(
            wire=team.test_wire, usage=usage, messages=[asdict(message) for message in team.messages],
            scripts=team.server.role_script_results, stage=team._stage, failure=team.failure, **facts), indent=2))


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('role', ['ceo', 'growth'])
def test_partial_script_batch_receipt_failure_does_not_become_new_week_boundary(mode, role, monkeypatch):
    from saas_bench import run_state
    def responder(team, sender, body):
        return calls(('bash', dict(command=ADVANCE))) if sender == 'ceo' else final(sender + ' report')

    with make_team(mode, responder, 'partial-script-state') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        register_cli(team, role, 'first.py',
            "from pathlib import Path\nimport novamind_api as n\n"
            "with open('executions.txt', 'a') as out: out.write('first\\n')\n"
            "n.infrastructure.set_capacity_tier(2)\nprint('FIRST-COMPLETED', flush=True)\n")
        register_cli(team, role, 'second.py', "print('PARTIAL-SECOND', flush=True)\nraise ValueError('ordinary script error')\n")
        before = team.checkpoint()
        failed_receipts = []
        write_json = run_state.write_json
        def fail_second_receipt(path, data):
            if Path(path).name == 'execution.json' and data['name'] == 'second.py':
                failed_receipts.append(Path(path))
                raise OSError('second script receipt could not be saved')
            return write_json(path, data)
        monkeypatch.setattr(run_state, 'write_json', fail_second_receipt)
        outcome = team.run(stop_after_day=7)
        records = team.server.role_script_results[role]
        save_evidence(team, 'partial-' + role, outcome=asdict(outcome), prior_checkpoint=str(before))
        assert (outcome.day, outcome.reason) == (0, 'failed'), team.failure
        assert 'second script receipt could not be saved' in team.failure
        assert [record['classification'] for record in records] == ['succeeded']
        assert all(record['facts']['boundary_closed'] for record in records)
        assert (team.roles[role].workspace / 'executions.txt').read_text() == 'first\n'
        receipt, = failed_receipts
        assert not receipt.exists()
        assert (receipt.parent / 'stdout.bin').read_text() == 'PARTIAL-SECOND\n'
        assert not any(executor.preserved_process for executor in team.server.role_executors.values())
        with pytest.raises(RuntimeError, match='recoverable handoff'):
            team.checkpoint()
        assert json.loads((team.root / 'checkpoint.json').read_text())['snapshot'] == str(before)
        assert team._stage == ('ceo_scripts' if role == 'ceo' else 'analysis')
        progress = team.roles[role].week
        assert progress.scripts_status == 'interrupted'
        assert progress.weekly_status != 'completed'
        assert json.loads((team.root / 'private' / role / 'week-state.json').read_text()) == progress.snapshot()


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_role_week_axes_survive_opening_questions_repeated_wakeups_and_new_week(mode):
    turns, sessions, seen = Counter(), {}, []

    def respond(team, role, body):
        day = team.server.tools.current_day
        progress = team.roles[role].week
        assert progress.initialized and progress.day == day
        assert progress.session_id == team.roles[role].identity.session_id
        assert sessions.setdefault((role, day), progress.session_id) == progress.session_id
        if day == 7:
            assert progress.session_id != sessions[role, 0]
        seen.append(dict(role=role, stage=team.phase, progress=progress.snapshot()))
        rendered = json.dumps(body['messages'])
        turns[role, day] += 1
        if role == 'ceo':
            n = turns[role, day]
            assert progress.weekly_status == 'in_progress'
            if team.phase == 'ceo_opening':
                assert progress.scripts_status == 'pending'
                assert 'OWNER-ceo' not in rendered
                if n < 3:
                    return calls(('ask_analyst', dict(role='growth', message='EARLY-' + str(n))))
                assert all(team.roles[peer].week.weekly_status == 'pending' for peer in ROLES[1:])
                return final('DIRECTIONS-' + str(day))
            assert all(team.roles[peer].week.weekly_status == 'completed' for peer in ROLES[1:])
            for peer in ROLES:
                assert team.roles[peer].week.scripts_status == 'succeeded'
                assert team._automatic_scripts(peer) == team.roles[peer].week.scripts_output
            if n < 6:
                return calls(('ask_analyst', dict(role='growth', message='LATE-' + str(n))))
            return calls(('bash', dict(command=ADVANCE)))
        question = team._workers[role].question
        if question:
            assert question.text in rendered
            early = question.text.startswith('EARLY-')
            assert progress.weekly_status == ('pending' if early else 'completed')
            assert progress.scripts_status == ('pending' if early else 'succeeded')
            return final('ANSWER-' + question.text)
        assert progress.scripts_status == 'succeeded'
        assert progress.weekly_status == 'in_progress'
        assert progress.weekly_request_id
        assert 'OWNER-' + role in rendered
        assert all('OWNER-' + peer not in rendered for peer in ROLES if peer != role)
        assert 'DENIED-PRICE' in rendered
        return final(role + ' formal report')

    with make_team(mode, respond, 'role-week-axes', opening_responder=respond) as team:
        for role in ROLES:
            code = ("import novamind_api as n\n"
                    "with open('auto-days.txt', 'a') as out: out.write(str(n.vars.current_day) + '\\n')\n"
                    "n.infrastructure.set_capacity_tier(2)\n" + f"print('OWNER-{role}')\n")
            if role != 'ceo':
                code += ("try: n.pricing.set_prices(A=999)\n"
                         "except n.NovaMindAPIError: print('DENIED-PRICE')\n"
                         "else: raise AssertionError('author permission lost')\n")
            register_cli(team, role, 'weekly.py', code)
        outcome = team.run(stop_after_day=14)
        save_evidence(team, 'axes', seen=seen, outcome=asdict(outcome))
        assert outcome.reason == 'observation_end', team.failure
        for role, runtime in team.roles.items():
            assert runtime.week.weekly_status == 'completed'
            assert (runtime.workspace / 'auto-days.txt').read_text() == '0\n7\n'
            records = [json.loads(line) for line in runtime.audit.path.read_text().splitlines()]
            assert len([record for record in records if record['kind'] == 'registered_script_execution']) == 2
            usage = [json.loads(line) for line in runtime.usage.path.read_text().splitlines()]
            sent = [record['body'] for record in usage if record['event'] == 'http_request']
            assert sent == [record['body'] for record in team.test_wire if record['role'] == role]
        assert len([message for message in team.messages if message.kind == 'weekly_task']) == 4
        assert len([message for message in team.messages if message.kind == 'question']) == 8
        assert all(message.status == 'delivered' for message in team.messages)


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('failure', ['nonzero', 'timeout'])
def test_script_error_is_terminal_once_across_later_questions(mode, failure):
    turns = Counter()
    def respond(team, role, body):
        turns[role] += 1
        state = team.roles[role].week
        assert state.scripts_status == 'script_error'
        rendered = json.dumps(body['messages'])
        assert 'SCRIPT-PARTIAL-' + role in rendered
        assert ('timed out' if failure == 'timeout' else 'RuntimeError: expected script error') in rendered
        assert all('SCRIPT-PARTIAL-' + peer not in rendered for peer in ROLES if peer != role)
        assert team._automatic_scripts(role) == state.scripts_output
        if role == 'ceo':
            if turns[role] == 1:
                return calls(('ask_analyst', dict(role='growth', message='Check the partial output once more.')))
            return calls(('bash', dict(command=ADVANCE)))
        return final(role + ' repaired advice')

    with make_team(mode, respond, 'script-known-error') as team:
        for role in ROLES:
            code = ("with open('script-count.txt', 'a') as out: out.write('once\\n')\n"
                    + f"print('SCRIPT-PARTIAL-{role}', flush=True)\n")
            code += "import time\ntime.sleep(60)\n" if failure == 'timeout' else "raise RuntimeError('expected script error')\n"
            register_cli(team, role, 'error.py', code)
            if failure == 'timeout':
                team.server.role_executors[role].bash_timeout = .4
        outcome = team.run(stop_after_day=7)
        save_evidence(team, 'known-error', outcome=asdict(outcome))
        assert outcome.reason == 'observation_end', team.failure
        assert turns == Counter(ceo=2, growth=2, ops_finance=1)
        for runtime in team.roles.values():
            assert runtime.week.scripts_status == 'script_error'
            assert runtime.week.weekly_status == 'completed'
            assert (runtime.workspace / 'script-count.txt').read_text() == 'once\n'


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('boundary', ['ask_completed', 'answers_ready'])
def test_existing_checkpoint_restores_role_axes_and_script_output_once(tmp_path, mode, boundary):
    import httpx
    from openai import OpenAI
    from saas_bench.execution_capture import CapturedText
    from saas_bench.multi_agent import MultiAgentRuntime
    from saas_bench.team_prompts import CEO_CLOSING
    from test_preflight_team_checkpoint import dispose
    from test_preflight_team_flow import completion

    snapshots, restored_wire = [], []

    def response(role, body):
        if role != 'ceo':
            return final(role + '-REPORT')
        messages = body['messages']
        if any(CEO_CLOSING in str(message.get('content', '')) for message in messages):
            return calls(('bash', dict(command=ADVANCE)))
        if not any(message.get('name') == 'ask_analyst' for message in messages):
            return calls(('ask_analyst', dict(role='growth', message='PRE-OPENING-CHECK')))
        return final('RECOVERY-DIRECTIONS')

    def clients(role):
        def respond(request):
            body = json.loads(request.content)
            restored_wire.append(dict(role=role, body=body))
            return completion(response(role, body), body.get('stream', False))
        return OpenAI(api_key='OFFLINE-ONLY', base_url='https://opencode.ai/zen/go/v1', max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    def respond(team, role, body):
        return response(role, body)

    with make_team(mode, respond, 'script-state-resume', opening_responder=respond) as original:
        original.server.simulator = fast_simulator(original.server.conn, original.server.tools.config, original.server.tools.rng)
        for role in ROLES:
            register_cli(original, role, 'once.py', "with open('count.txt', 'a') as out: out.write('once\\n')\n"
                + f"print('SAVED-OUTPUT-{role}')\n")

        def save(runtime, kind):
            if kind == boundary:
                snapshots.append(runtime.checkpoint())

        original.checkpoint_callback = save
        assert original.run(stop_after_day=7).reason == 'observation_end', original.failure
        assert len(snapshots) == 1
        saved = json.loads((snapshots[0] / 'team.json').read_text())
        assert set(saved['role_weeks']) == set(ROLES)
        for role, state in saved['role_weeks'].items():
            assert state['scripts_status'] == ('pending' if boundary == 'ask_completed' else 'succeeded')
            assert state['initialized'] == (boundary == 'answers_ready' or role in ('ceo', 'growth'))
            assert state['weekly_status'] == ('in_progress' if role == 'ceo' else
                                             'pending' if boundary == 'ask_completed' else 'completed')
        resumed = MultiAgentRuntime.restore(snapshots[0], tmp_path / 'resumed', client_factory=clients,
                                             simulator_factory=fast_simulator)
        try:
            assert {role: runtime.week.snapshot() for role, runtime in resumed.roles.items()} == saved['role_weeks']
            if mode == 'pf' and boundary == 'answers_ready':
                for runtime in resumed.roles.values():
                    assert isinstance(runtime.week.scripts_output, CapturedText)
                    assert runtime.week.scripts_output.origins
            assert resumed.run(stop_after_day=7).reason == 'observation_end', resumed.failure
            for role, runtime in resumed.roles.items():
                assert (runtime.workspace / 'count.txt').read_text() == 'once\n'
                assert runtime.week.weekly_status == 'completed'
                sent = [row['body']['messages'] for row in restored_wire if row['role'] == role]
                if sent:
                    assert sum(str(message.get('content', '')).count('SAVED-OUTPUT-' + role)
                               for message in sent[-1]) == 1
                assert runtime.week.session_id == saved['role_weeks'][role]['session_id']
            if boundary == 'answers_ready':
                assert [row['role'] for row in restored_wire] == ['ceo']
            if target := os.environ.get('CEOBENCH_SCRIPT_STATE_EVIDENCE'):
                (Path(target) / (mode + '-restore-' + boundary + '.json')).write_text(json.dumps(dict(
                    saved=saved['role_weeks'], wire=restored_wire,
                    restored={role: runtime.week.snapshot() for role, runtime in resumed.roles.items()},
                    sources=saved['sources']), indent=2))
        finally:
            dispose(resumed)


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('phase', ['ceo_opening', 'ceo'])
def test_input_state_persistence_failure_keeps_acceptance_before_resume(tmp_path, mode, phase, monkeypatch):
    import httpx
    from openai import OpenAI
    from saas_bench import multi_agent
    from saas_bench.team_prompts import CEO_CLOSING, CEO_OPENING
    from test_preflight_team_checkpoint import dispose
    from test_preflight_team_flow import completion

    prompt = CEO_OPENING if phase == 'ceo_opening' else CEO_CLOSING
    wire, failed = [], []

    def answer(role, body):
        if role != 'ceo':
            return final(role + '-HANDOFF')
        if any(CEO_CLOSING in str(message.get('content', '')) for message in body['messages']):
            return calls(('bash', dict(command=ADVANCE)))
        return final('DIRECTIONS')

    def respond(team, role, body):
        return answer(role, body)

    def clients(role):
        def call(request):
            body = json.loads(request.content)
            wire.append(dict(role=role, body=body))
            return completion(answer(role, body), body.get('stream', False))
        return OpenAI(api_key='OFFLINE-ONLY', base_url='https://opencode.ai/zen/go/v1', max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(call)))

    with make_team(mode, respond, 'input-state-fault', opening_responder=respond) as original:
        original.server.simulator = fast_simulator(original.server.conn, original.server.tools.config, original.server.tools.rng)
        write = multi_agent.write_json

        def fail_after_accept(path, data):
            if (not failed and Path(path) == original.root / 'private/ceo/week-state.json'
                    and original._stage == phase and any(prompt in str(message.content)
                        for message in original.roles['ceo'].agent.conversation)):
                failed.append(True)
                raise OSError('injected role state persistence failure')
            return write(path, data)

        monkeypatch.setattr(multi_agent, 'write_json', fail_after_accept)
        assert original.run(stop_after_day=7).reason == 'failed'
        assert failed and 'injected role state' in original.failure
        snapshot = original.checkpoint()
        resumed = multi_agent.MultiAgentRuntime.restore(snapshot, tmp_path / 'resumed', client_factory=clients,
                                                        simulator_factory=fast_simulator)
        try:
            assert resumed.run(stop_after_day=7).reason == 'observation_end', resumed.failure
            closing = [row['body']['messages'] for row in wire if row['role'] == 'ceo'][-1]
            for text in (CEO_OPENING, CEO_CLOSING):
                assert sum(text in str(message.get('content', '')) for message in closing) == 1
            if target := os.environ.get('CEOBENCH_SCRIPT_STATE_EVIDENCE'):
                root = Path(target)
                root.mkdir(parents=True, exist_ok=True)
                (root / (mode + '-input-fault-' + phase + '.json')).write_text(json.dumps(dict(
                    wire=wire, snapshot=json.loads((snapshot / 'team.json').read_text())['role_weeks']), indent=2))
        finally:
            dispose(resumed)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_empty_batch_does_not_run_late_registration_until_next_week(mode):
    turns = Counter()
    code = "import novamind_api as n\nwith open('late-days.txt', 'a') as out: out.write(str(n.vars.current_day) + '\\n')\nprint('LATE-OUTPUT')\n"

    def respond(team, role, body):
        day = team.server.tools.current_day
        turns[role, day] += 1
        state = team.roles[role].week
        assert state.scripts_status == 'succeeded'
        if role == 'ceo':
            if turns[role, day] == 1:
                return calls(('ask_agent', dict(role='growth', message='LATE-REGISTRATION-CHECK')))
            return calls(('bash', dict(command=ADVANCE)))
        if role == 'growth':
            if day == 0:
                assert team._automatic_scripts(role) == ''
                assert not (team.roles[role].workspace / 'late-days.txt').exists()
                if turns[role, day] == 1:
                    return calls(('write_file', dict(path='late.py', content=code)),
                                 ('bash', dict(command='./novamind-operation register-daily-script late.py')))
            else:
                assert (team.roles[role].workspace / 'late-days.txt').read_text() == '7\n'
                assert 'LATE-OUTPUT' in json.dumps(body['messages'])
        return final(role + ' report')

    with make_team(mode, respond, 'empty-script-batch') as team:
        outcome = team.run(stop_after_day=14)
        save_evidence(team, 'empty-batch', outcome=asdict(outcome))
        assert outcome.reason == 'observation_end', team.failure
        assert (team.roles['growth'].workspace / 'late-days.txt').read_text() == '7\n'
