import json
import re

import pytest

from test_preflight_team_flow import ADVANCE, calls, final, make_team, raw


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_script_receipts_expose_name_size_and_change_without_checksum(mode):
    with make_team(mode, lambda *args: final('done'), 'public-script-receipt') as team:
        executor = team.roles['growth'].executor
        code = "print('business observation')\n"
        executor.execute('write_file', dict(path='observe.py', content=code))
        command = './novamind-operation register-daily-script observe.py'
        first = executor.execute('bash', dict(command=command))
        assert '"registered": true' in first and '"changed": true' in first
        assert 'sha256' not in first
        same = raw(team, 'growth', '/daily-scripts', dict(name='observe.py', content=code))[1]
        assert same['data'] == dict(name='observe.py', registered=True, changed=False, size=len(code))
        changed = raw(team, 'growth', '/daily-scripts', dict(name='observe.py', content=code + '# revised\n'))[1]
        assert changed['data']['changed'] is True
        listed = executor.execute('bash', dict(command='./novamind-operation list-daily-scripts'))
        assert '"name": "observe.py"' in listed and 'sha256' not in listed
        assert team.server.get_daily_scripts('growth')['observe.py'] == code + '# revised\n'
        for command in ('register-daily-script --help', 'list-daily-scripts --help'):
            help_text = executor.execute('bash', dict(command='./novamind-operation ' + command))
            assert 'sha256' not in help_text


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_team_messages_publish_short_requests_without_world_identity(mode):
    asked = False

    def respond(team, role, body):
        nonlocal asked
        if role == 'ceo':
            return calls(('bash', dict(command=ADVANCE)))
        if role == 'growth' and not asked:
            asked = True
            return calls(('ask_agent', dict(role='ops_finance', message='Confirm capacity cost.')))
        return final('Capacity costs were reviewed.')

    with make_team(mode, respond, 'public-team-receipt') as team:
        assert team.run(stop_after_day=7).reason == 'observation_end', team.failure
        public_messages = '\n'.join(str(m.get('content', '')) for entry in team.test_wire
                                    for m in entry['body']['messages'])
        for message in team.messages:
            assert message.request_id not in public_messages
            assert message.world_state_id not in public_messages
        assert re.search(r'request_id=growth:request[1-9][0-9]*', public_messages)
        commits = re.findall(r'Snapshot commit: (\S+)', public_messages)
        assert commits and all(re.fullmatch(r'[0-9a-f]{7,12}', c) for c in commits)
        for role, runtime in team.roles.items():
            usage = [json.loads(line)['body'] for line in runtime.usage.path.read_text().splitlines()
                     if json.loads(line)['event'] == 'http_request']
            assert usage == [entry['body'] for entry in team.test_wire if entry['role'] == role]
