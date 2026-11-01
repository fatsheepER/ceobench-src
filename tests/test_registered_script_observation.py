import json
from pathlib import Path
import shlex

import httpx
from openai import OpenAI
import pytest

from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.role_policy import ROLES
from test_preflight_team_checkpoint import dispose, fast_simulator
from test_preflight_team_flow import ADVANCE, PREDICTIONS, calls, completion, final, make_team, raw, register
from test_responsibility_authority import ACTIONS, READS, world


DIRECTIONS = 'Hold Plan B at tier 3 and S1 development spend at 5000. Capacity may increase to 2.'


def protected_settings(team):
    tier = team.server.conn.execute('SELECT tier_B FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0]
    return tier, team.server.tools.config.targeted_dev_spend


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_stale_automatic_writes_cannot_precede_current_directions_or_replay(tmp_path, mode):
    seen, checkpoints = [], []
    code = '''import novamind_api as n
with open('automatic-count.txt', 'a') as out: out.write('once\\n')
for tool, args in [(n.pricing.set_model_tiers, {'B': 2}),
                   (n.analytics.set_targeted_dev_spend, {'targeted_spend': {'S1': 1500}})]:
    try:
        tool(**args)
    except n.NovaMindAPIError as error:
        print('AUTOMATIC-DENIED', error)
    else:
        print('STALE-WRITE-APPLIED')
print('AUTOMATIC-OBSERVATION', n.query('SELECT tier_B FROM config_history ORDER BY day DESC LIMIT 1'))
'''

    def respond(team, role, body):
        messages = body['messages']
        if role == 'growth':
            rendered = json.dumps(messages)
            assert DIRECTIONS in rendered
            assert 'AUTOMATIC-DENIED' in rendered
            assert 'observation-only' in rendered
            assert 'STALE-WRITE-APPLIED' not in rendered
            assert protected_settings(team) == (3, {'S1': 5000})
            if not seen:
                seen.append(True)
                command = 'python -c ' + shlex.quote('import novamind_api as n; print(n.infrastructure.set_capacity_tier(2))')
                return calls(('bash', dict(command=command)))
            return final('Held both settings. Increased capacity to 2 after reading the current CEO directions.')
        if role == 'ceo':
            assert protected_settings(team) == (3, {'S1': 5000})
            return calls(('bash', dict(command=ADVANCE)))
        return final('Checked the current plan.')

    with make_team(mode, respond, 'automatic-observation', opening_responder=lambda *args: final(DIRECTIONS)) as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        assert raw(team, 'ceo', '/call', dict(tool='set_model_tiers', args={'B': 3}))[1]['success']
        assert raw(team, 'ceo', '/call', dict(tool='set_targeted_dev_spend', args={'targeted_spend': {'S1': 5000}}))[1]['success']
        checksum = register(team, 'growth', 'old-policy.py', code)

        def checkpoint(runtime, boundary):
            if boundary == 'answers_ready':
                checkpoints.append(runtime.checkpoint())

        team.checkpoint_callback = checkpoint
        outcome = team.run(stop_after_day=7)
        assert outcome.reason == 'observation_end', team.failure
        assert seen and len(checkpoints) == 1
        assert protected_settings(team) == (3, {'S1': 5000})
        assert team.server.conn.execute('SELECT capacity_tier FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] == 2
        assert not team.server._operation_failed
        record, = team.server.role_script_results['growth']
        assert record['sha256'] == checksum
        assert record['classification'] == 'succeeded'
        assert record['facts']['boundary_closed'] is True
        assert (Path(record['evidence']) / 'code.py').read_text() == code
        assert 'AUTOMATIC-DENIED' in (Path(record['evidence']) / 'stdout.bin').read_text()
        operations = [json.loads(path.read_text()) for path in (team.root / 'private/operations').glob('*.json')]
        assert {row['kind'] for row in operations if row['role'] == 'growth'} == {'api:set_capacity_tier'}
        for role in ROLES:
            usage = [json.loads(line) for line in team.roles[role].usage.path.read_text().splitlines()]
            sent = [row['body'] for row in usage if row['event'] == 'http_request']
            assert sent == [row['body'] for row in team.test_wire if row['role'] == role]
        growth_input = [row['body']['messages'] for row in team.test_wire if row['role'] == 'growth'][0]
        assert 'AUTOMATIC-DENIED' in json.dumps(growth_input)
        assert DIRECTIONS in json.dumps(growth_input)
        saved = checkpoints[0]

    restored_wire = []

    def clients(role):
        def answer(request):
            body = json.loads(request.content)
            restored_wire.append(dict(role=role, body=body))
            assert role == 'ceo'
            return completion(calls(('bash', dict(command=ADVANCE))), body.get('stream', False))
        return OpenAI(api_key='OFFLINE-ONLY', base_url='https://opencode.ai/zen/go/v1', max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(answer)))

    restored = MultiAgentRuntime.restore(saved, tmp_path / 'resumed', client_factory=clients,
                                        simulator_factory=fast_simulator)
    try:
        assert restored.run(stop_after_day=7).reason == 'observation_end', restored.failure
        assert restored_wire and {row['role'] for row in restored_wire} == {'ceo'}
        assert (restored.roles['growth'].workspace / 'automatic-count.txt').read_text() == 'once\n'
        assert restored.roles['growth'].week.scripts_status == 'succeeded'
        assert protected_settings(restored) == (3, {'S1': 5000})
    finally:
        dispose(restored)


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('role', ROLES)
def test_registered_listener_rejects_all_mutation_routes_without_poisoning_owner(mode, role):
    program = '''import http.client, json, os, socket
from novamind_api import _client
ACTIONS = ACTION_DATA
READS = READ_DATA

def request(path, body=None, method=None):
    connection = http.client.HTTPConnection('localhost')
    connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.sock.connect(os.environ['NOVAMIND_API_SOCKET'])
    connection.request(method or ('POST' if body is not None else 'GET'), path,
                       json.dumps(body) if body is not None else None,
                       {'Content-Type': 'application/json', 'X-Initiator': 'model'})
    response = connection.getresponse()
    result = response.status, json.load(response)
    connection.close()
    return result

for name, (args, module, allowed) in ACTIONS.items():
    status, result = request('/call', {'tool': name, 'args': args, 'role': 'ceo', 'initiator': 'model'})
    assert status == (200 if name in READS else 403), (name, status, result)
    if name not in READS:
        assert 'observation-only' in result['error'], (name, result)
    assert request('/query', {'sql': 'SELECT 1 AS continued'})[0] == 200
for path, body, method in [
    ('/next-week', {'predictions': PREDICTION_DATA, 'rationale': 'unattended'}, 'POST'),
    ('/daily-scripts', {'name': 'new.py', 'content': "print('new')"}, 'POST'),
    ('/daily-scripts', {'name': 'matrix.py'}, 'DELETE')]:
    status, result = request(path, body, method)
    assert status == 403 and 'observation-only' in result['error'], (path, status, result)
for path in ('/daily-scripts', '/vars', '/dashboard', '/game-status', '/health'):
    assert request(path)[0] == 200, path
with open('observation.txt', 'w') as out: out.write('local calculation retained')
print('OBSERVATION-MATRIX-COMPLETE')
'''.replace('ACTION_DATA', repr(ACTIONS)).replace('READ_DATA', repr(READS)).replace('PREDICTION_DATA', repr(PREDICTIONS))
    with make_team(mode, lambda *args: final('unused'), 'automatic-routes-' + role) as team:
        register(team, role, 'matrix.py', program)
        team._set_phase('ceo' if role == 'ceo' else 'analysis')
        before = world(team)
        scripts = team.server.get_daily_scripts(role)
        epoch = team.server._request_mutations
        output = team._script_output(role)
        assert 'OBSERVATION-MATRIX-COMPLETE' in output, output
        assert world(team) == before
        assert team.server._request_mutations == epoch
        assert team.server.get_daily_scripts(role) == scripts
        assert not team.server._operation_failed
        assert not team.server.role_executors[role].preserved_process
        assert (team.roles[role].workspace / 'observation.txt').read_text() == 'local calculation retained'
        assert raw(team, role, '/query', dict(sql='SELECT 1 AS continued'))[0] == 200
        changed = raw(team, role, '/call', dict(tool='set_capacity_tier', args=dict(tier=2)))
        assert changed[0] == 200 and changed[1]['success'], changed
        assert team.server.conn.execute('SELECT capacity_tier FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] == 2
        operations = [json.loads(path.read_text()) for path in (team.root / 'private/operations').glob('*.json')]
        assert {row['kind'] for row in operations if row['kind'].startswith('api:')} == {'api:set_capacity_tier'}
        assert all(row['status'] == 'completed' for row in operations)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_restored_prompt_revokes_legacy_script_writes_without_replay(tmp_path, mode):
    old_grants = ('Registered scripts\nhave the same permissions as their owner.',
                  "Scripts have their owner's business permissions.",
                  'Registration snapshots the source file under its filename. Re-register after edits.')
    checkpoints, wire = [], []

    def respond(team, role, body):
        return calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final(role + ' report')

    with make_team(mode, respond, 'automatic-legacy-prompt') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        for role in ROLES:
            register(team, role, 'once.py', "with open('once.txt', 'a') as out: out.write('once\\n')\nprint('OBSERVED-ONCE')\n")

        def save(runtime, boundary):
            if boundary != 'answers_ready':
                return
            for name, owner in runtime.roles.items():
                legacy = '\nUNRELATED-SAVED-PROMPT-' + name + '\n' + '\n'.join(old_grants)
                owner.agent.system_prompt += legacy
                for message in owner.agent.conversation:
                    if message.role == 'system':
                        message.content += legacy + '\nPRESERVED-MEMORY-' + name
            checkpoints.append(runtime.checkpoint())

        team.checkpoint_callback = save
        assert team.run(stop_after_day=7).reason == 'observation_end', team.failure
        assert len(checkpoints) == 1
        selected = checkpoints[0]

    def clients(role):
        def answer(request):
            body = json.loads(request.content)
            wire.append(dict(role=role, body=body))
            assert role == 'ceo'
            system, = [message['content'] for message in body['messages'] if message['role'] == 'system']
            assert all(grant not in system for grant in old_grants)
            assert 'observation-only' in system
            assert 'UNRELATED-SAVED-PROMPT-ceo' in system
            assert 'PRESERVED-MEMORY-ceo' in system
            return completion(calls(('bash', dict(command=ADVANCE))), body.get('stream', False))
        return OpenAI(api_key='OFFLINE-ONLY', base_url='https://opencode.ai/zen/go/v1', max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(answer)))

    resumed = MultiAgentRuntime.restore(selected, tmp_path / 'resumed', client_factory=clients,
                                       simulator_factory=fast_simulator)
    try:
        for role, owner in resumed.roles.items():
            assert all(grant not in owner.agent.system_prompt for grant in old_grants)
            assert 'observation-only' in owner.agent.system_prompt
            assert 'UNRELATED-SAVED-PROMPT-' + role in owner.agent.system_prompt
        assert resumed.run(stop_after_day=7).reason == 'observation_end', resumed.failure
        assert wire and {row['role'] for row in wire} == {'ceo'}
        for role, owner in resumed.roles.items():
            assert (owner.workspace / 'once.txt').read_text() == 'once\n'
            assert owner.week.scripts_status == 'succeeded'
        usage = [json.loads(line) for line in resumed.roles['ceo'].usage.path.read_text().splitlines()]
        sent = [row['body'] for row in usage if row['event'] == 'http_request']
        assert sent[-len(wire):] == [row['body'] for row in wire]
    finally:
        dispose(resumed)
