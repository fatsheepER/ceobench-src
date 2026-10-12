import json

import pytest

from saas_bench.role_policy import ROLES
from test_preflight_team_checkpoint import fast_simulator
from test_preflight_team_flow import ADVANCE, calls, final, make_team
from test_text_registry import declaration


def script(role):
    body = ("import json\nimport novamind_api as nm\n"
            "day = nm.vars.current_day\n"
            f"print(f'AUTO-{role}-D{{day}}')\n"
            "with open('auto-days.txt', 'a') as stream: stream.write(str(day) + '\\n')\n"
            "with open('auto-result.json', 'w') as stream: json.dump({'day': day}, stream)\n")
    if role == 'ceo':
        return body + "nm.pricing.set_prices(A=31)\n"
    return body + ("try: nm.pricing.set_prices(A=900)\n"
                   "except nm.NovaMindAPIError: print('WRITE-DENIED')\n"
                   "else: raise AssertionError('analyst changed the world')\n")


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_deployed_cli_snapshot_runs_next_week_and_reaches_owner_input(mode):
    counts, saved = {}, []

    def responder(team, role, body):
        day = team.server.tools.current_day
        key = (role, day)
        turn = counts[key] = counts.get(key, 0) + 1
        if day == 7:
            rendered = json.dumps(body['messages'], ensure_ascii=False)
            assert f'AUTO-{role}-D7' in rendered
            assert all(f'AUTO-{peer}-D7' not in rendered for peer in ROLES if peer != role)
            if role != 'ceo':
                assert 'WRITE-DENIED' in rendered
            assert (team.roles[role].workspace / 'auto-days.txt').read_text() == '7\n'
        if day == 0 and turn == 1:
            return calls(
                ('write_file', dict(path='weekly.py', content=script(role))),
                ('bash', dict(command='./novamind-operation register-daily-script weekly.py && '
                                      './novamind-operation list-daily-scripts')),
                ('write_file', dict(path='weekly.py', content="raise AssertionError('edited source ran')\n")))
        if role != 'ceo':
            if mode == 'pf' and day == 7:
                if turn == 1:
                    return calls(('read_file', dict(path='auto-result.json')))
                if turn == 2:
                    reference = dict(evidence={'path': 'auto-result.json'}, purpose='current',
                                     select={'path': '/day'},
                                     predicate={'type': 'threshold', 'op': '>=', 'value': 7})
                    return calls(('text_create', declaration(references=[reference])))
                if turn == 3:
                    return calls(('bash', dict(command='pf depend r1 --detail && pf log auto-result.json')))
                if turn == 4:
                    assert 'not recomputed' in rendered
                    assert 'cited observation day 7' in rendered
                    assert 'checked day 7' in rendered
                    assert 'auto-result.json@v1 · day 7' in rendered
            followup = 5 if mode == 'pf' else 2
            if day == 7 and turn == followup:
                return calls(('bash', dict(command='./novamind-operation remove-daily-script weekly.py && '
                                                  'test -f weekly.py && ./novamind-operation list-daily-scripts')))
            return final(f'{role} handoff at day {day}')
        if (day == 0 and turn == 2) or (day == 7 and turn == 1):
            return calls(*(('ask_analyst', dict(role=peer, message='Check your advice again.'))
                           for peer in ROLES[1:]))
        prefix = ('./novamind-operation remove-daily-script weekly.py && '
                  'test -f weekly.py && ./novamind-operation list-daily-scripts && ') if day == 7 else ''
        return calls(('bash', dict(command=prefix + ADVANCE)))

    with make_team(mode, responder, 'script-cli') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        def checkpoint(runtime, boundary):
            if boundary == 'week_boundary' and runtime.server.tools.current_day == 7:
                saved.append(runtime.checkpoint())
        team.checkpoint_callback = checkpoint
        outcome = team.run(stop_after_day=14)
        assert outcome.reason == 'observation_end', team.failure
        assert len(saved) == 1
        state = json.loads((saved[0] / 'team.json').read_text())
        assert state['scripts'] == {role: {'weekly.py': script(role)} for role in ROLES}
        price = team.server.conn.execute('SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0]
        assert price == 31
        for role in ROLES:
            assert (team.roles[role].workspace / 'auto-days.txt').read_text() == '7\n'
            assert team.server.get_daily_scripts(role) == {}
            assert (team.roles[role].workspace / 'weekly.py').read_text() == "raise AssertionError('edited source ran')\n"
            audit = [json.loads(line) for line in team.roles[role].audit.path.read_text().splitlines()]
            executions = [row for row in audit if row['kind'] == 'registered_script_execution']
            assert len(executions) == 1 and executions[0]['status'] == 'succeeded'
            assert any(item['role'] == role and item['day'] == 7 and
                       f'AUTO-{role}-D7' in json.dumps(item['body']['messages']) for item in team.test_wire)
