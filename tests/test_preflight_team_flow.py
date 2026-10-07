import copy
from contextlib import contextmanager
import hashlib
import http.client
import json
import os
from pathlib import Path
import shlex
import socket
import sqlite3
import tempfile
import threading
import time
from types import SimpleNamespace
import uuid

import httpx
from numpy.random import default_rng
from openai import OpenAI
import pytest

from saas_bench.config import BenchmarkConfig
from saas_bench.database import init_database
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.role_policy import ROLES
from saas_bench.run_lifecycle import WorkerLifecycle
from saas_bench.simulation import Simulator
from saas_bench.tools import AgentTools


PREDICTIONS = {f'cash_{h}wk': dict(point=1000000, lower=0, upper=2000000)
               for h in (1, 4, 12, 26)}
ADVANCE = './novamind-operation next-week "offline weekly rationale" ' + '1000000 0 2000000 ' * 4


def calls(*actions):
    return dict(role='assistant', content='', tool_calls=[
        dict(id=uuid.uuid4().hex, type='function',
             function=dict(name=name, arguments=json.dumps(args))) for name, args in actions])


def final(text):
    return dict(role='assistant', content=text)


def completion(message, stream=False):
    common = dict(id=uuid.uuid4().hex, created=1, model='deepseek-v4.1-flash')
    usage = dict(prompt_tokens=20, completion_tokens=10, total_tokens=30,
                 prompt_tokens_details=dict(cached_tokens=0),
                 completion_tokens_details=dict(reasoning_tokens=1))
    reason = 'tool_calls' if message.get('tool_calls') else 'stop'
    if not stream:
        return httpx.Response(200, json=dict(common, object='chat.completion', usage=usage,
            choices=[dict(index=0, finish_reason=reason, message=message)]))
    delta = dict(message)
    if 'tool_calls' in delta:
        delta['tool_calls'] = [dict(c, index=i) for i, c in enumerate(delta['tool_calls'])]
    chunks = [dict(common, object='chat.completion.chunk',
                   choices=[dict(index=0, delta=delta, finish_reason=None)]),
              dict(common, object='chat.completion.chunk', usage=usage,
                   choices=[dict(index=0, delta={}, finish_reason=reason)])]
    body = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n'
    return httpx.Response(200, content=body.encode(), headers={'content-type': 'text/event-stream'})


@contextmanager
def make_team(mode, responder, label):
    base = Path(os.environ.get('CEOBENCH_TEAM_ARTIFACTS', tempfile.mkdtemp(prefix='team-flow-')))
    base.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f'{label}-{mode}-', dir=base))
    conn = init_database(root / 'world.db')
    config = BenchmarkConfig(seed=123)
    Simulator(conn, config, default_rng(123)).initialize()
    conn.close()
    conn = sqlite3.connect(root / 'world.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    tools = AgentTools(conn, 0, root / 'host', config=config, rng=default_rng(123))
    wire, timeline, holder = [], [], {}
    lock = threading.Lock()

    def event(kind, **facts):
        with lock:
            timeline.append(dict(kind=kind, at=time.monotonic(), day=tools.current_day, **facts))

    def client(role):
        def respond(req):
            body = json.loads(req.content)
            with lock:
                wire.append(dict(role=role, day=tools.current_day, body=body))
            event('model_start', role=role)
            result = responder(holder['team'], role, body)
            event('model_end', role=role)
            if isinstance(result, httpx.Response):
                return result
            return completion(result, body.get('stream', False))
        return OpenAI(api_key='OFFLINE-ONLY', base_url='https://opencode.ai/zen/go/v1',
                      max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    def step():
        event('settlement', target=tools.current_day + 7)
        return SimpleNamespace(day=tools.current_day + 7)

    def dashboard(day, result):
        price = conn.execute('SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0]
        return f'=== Week {day // 7 + 1} Dashboard (Day {day}) ===\nPUBLIC price_A={price}'

    team = MultiAgentRuntime(root / 'run', tools, conn=conn, simulator=SimpleNamespace(step_week=step),
                            client_factory=client, mode=mode, dashboard_callback=dashboard)
    holder['team'] = team
    team.test_event, team.test_wire, team.test_timeline = event, wire, timeline
    team.test_root = root
    try:
        yield team
    finally:
        (root / 'provider-requests.json').write_text(json.dumps(wire, indent=2))
        (root / 'timeline.json').write_text(json.dumps(timeline, indent=2))
        for runtime in team.roles.values():
            runtime.agent.client.close()
        team.close()
        conn.close()


def raw(team, role, path, body=None, method=None):
    connection = http.client.HTTPConnection('localhost')
    connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.sock.connect(str(team.server.role_sockets[role]))
    connection.request(method or ('POST' if body is not None else 'GET'), path,
                       json.dumps(body) if body is not None else None,
                       {'content-type': 'application/json'})
    response = connection.getresponse()
    result = response.status, json.load(response)
    connection.close()
    return result


def snapshot(team):
    files = {str(p.relative_to(team.roles['ceo'].workspace)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in team.roles['ceo'].workspace.rglob('*') if p.is_file()}
    return (team.server.conn.serialize(), team.server.tools.current_day,
            copy.deepcopy(team.server.tools.rng.bit_generator.state), team.server.world_state_id, files)


def register(team, role, name, code):
    path = team.roles[role].workspace / name
    path.write_text(code)
    status, body = raw(team, role, '/daily-scripts', dict(name=name, content=code))
    assert status == 200 and body['success'], body
    path.write_text("raise AssertionError('edited file must not execute')")
    return hashlib.sha256(code.encode()).hexdigest()


def probe_ceo_frozen(team):
    before = snapshot(team)
    for tool, args in [('set_prices', {'A': 900}), ('research_market', {}),
                       ('research_group', {'group_id': 'S1', 'target_level': 2}),
                       ('start_research_project', {'tier': 1})]:
        assert raw(team, 'ceo', '/call', dict(tool=tool, args=args))[0] == 403
        with pytest.raises(PermissionError):
            team.server.execute_tool(tool, args, role='ceo')
    assert raw(team, 'ceo', '/next-week', dict(predictions=PREDICTIONS, rationale='forbidden'))[0] == 403
    assert raw(team, 'ceo', '/daily-scripts', dict(name='evil.py', content='print(1)'))[0] == 403
    assert raw(team, 'ceo', '/daily-scripts', dict(name='same.py'), 'DELETE')[0] == 403
    for executor in (team.roles['ceo'].executor, team.server.role_executors['ceo']):
        for name, args in [('write_file', dict(path='marker.txt', content='changed')),
                           ('edit_file', dict(path='marker.txt', old_string='stable', new_string='changed')),
                           ('bash', dict(command='printf changed > marker.txt')),
                           ('text_create', dict(text='forbidden'))]:
            result = executor.execute(name, args)
            assert result.startswith('Error:') or 'frozen' in result.lower(), result
    assert snapshot(team) == before
    team.test_event('freeze_verified')


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_two_weeks_parallel_reports_followups_scripts_and_reset(mode):
    initial_barriers = {day: threading.Barrier(2) for day in (0, 7)}
    ask_barriers = {day: threading.Barrier(2) for day in (0, 7)}
    counters, completed, sessions, script_hashes = {}, {}, {}, {}
    lock = threading.Lock()

    def respond(team, role, body):
        day = team.server.tools.current_day
        with lock:
            key = (role, day)
            n = counters[key] = counters.get(key, 0) + 1
        messages = body['messages']
        system = messages[0]['content']
        assert '500' in system and '497' in system
        assert 'D112' not in system and 'MEMORY_OVERFLOW' not in system
        memory = (team.roles[role].workspace / 'MEMORY.md').read_text().strip()
        start = system.index(f'KEEP-{role}\n')
        assert system[start:start + 40000] == memory[:40000]
        assert system[start + 40000:].startswith('\n\n--- MEMORY.md TRUNCATED ---')
        assert all(f'KEEP-{peer}\n' not in system for peer in ROLES if peer != role)
        assert body['reasoning_effort'] == 'high' and body['thinking']['type'] == 'enabled'
        sessions.setdefault(key, team.roles[role].identity.session_id)
        rendered = json.dumps(messages)
        assert f'PRIVATE-SCRIPT-{role}' in rendered
        for peer in ROLES:
            if peer != role:
                assert f'PRIVATE-SCRIPT-{peer}' not in rendered
        if day == 7:
            assert 'REPORT-growth-D0' not in rendered and 'REPORT-ops_finance-D0' not in rendered
            assert 'QUESTION-D0' not in rendered
        if role != 'ceo':
            if n == 1:
                assert 'REQUIRED' not in system or 'advance to the next week' not in system.lower()
                initial_barriers[day].wait(timeout=10)
                team.test_event('analyst_initial_overlap', role=role)
                assert counters.get(('ceo', day), 0) == 0
                if role == 'growth':
                    probe_ceo_frozen(team)
                return calls(('bash', dict(command='./novamind-operation python-c ' + shlex.quote(
                    "import novamind_api as nm; print(nm.query('SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1')['rows'])"))),
                    ('text_list', {}))
            if n == 2:
                assert any('31' in m['content'] for m in messages if m['role'] == 'tool' and m.get('name') == 'bash')
                assert 'r1' in messages[-1]['content'] and not messages[-1]['content'].startswith('Error:')
                with lock:
                    completed[key] = True
                return final(f'REPORT-{role}-D{day}. Free prose, no required fields.')
            question = messages[-1]['content']
            assert isinstance(question, str) and f'QUESTION-D{day}' in question
            if n == 3:
                ask_barriers[day].wait(timeout=10)
                team.test_event('analyst_ask_overlap', role=role)
                if role == 'growth':
                    probe_ceo_frozen(team)
            if role == 'growth' and n == 4:
                assert f'REPLY-growth-D{day}-3' in rendered
            return final(f'REPLY-{role}-D{day}-{n}. Answer to {question}')
        assert completed.get(('growth', day)) and completed.get(('ops_finance', day))
        assert f'REPORT-growth-D{day}' in rendered and f'REPORT-ops_finance-D{day}' in rendered
        if n == 1:
            return calls(('ask_analyst', dict(role='growth', message=f'QUESTION-D{day} first')),
                         ('ask_analyst', dict(role='ops_finance', message=f'QUESTION-D{day} other')),
                         ('ask_analyst', dict(role='growth', message=f'QUESTION-D{day} second')))
        results = [m for m in messages if m['role'] == 'tool' and m.get('name') == 'ask_analyst']
        assert len(results) == 3 and len({m['tool_call_id'] for m in results}) == 3
        assert sum(f'REPLY-growth-D{day}-3' in m['content'] for m in results) == 1
        assert sum(f'REPLY-growth-D{day}-4' in m['content'] for m in results) == 1
        return calls(('bash', dict(command=ADVANCE)),
                     ('write_file', dict(path='should-not-exist', content='old week')))

    with make_team(mode, respond, 'two-weeks') as team:
        for role, runtime in team.roles.items():
            (runtime.workspace / 'MEMORY.md').write_text(f'KEEP-{role}\n' + 'm' * 40000 + 'MEMORY_OVERFLOW')
            record = runtime.executor.execute('text_create', dict(text=f'pre-session-{role}',
                objects=[dict(kind='plan', id='A')], references=[dict(cite='unknown: offline')],
                applies='0-', reason='offline session ownership'))
            assert record.startswith('Registered r1.1'), record
        (team.roles['ceo'].workspace / 'marker.txt').write_text('stable')
        ceo_code = "import novamind_api as nm\nnm.pricing.set_prices(A=31)\nprint('PRIVATE-SCRIPT-ceo')\n"
        script_hashes['ceo'] = register(team, 'ceo', 'same.py', ceo_code)
        for role in ROLES[1:]:
            foreign = str(team.roles['ceo'].workspace / 'marker.txt')
            code = ("import novamind_api as nm\n"
                    "from novamind_api import _client\n"
                    "print(nm.query('SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1')['rows'])\n"
                    "try: nm.pricing.set_prices(A=900)\n"
                    "except _client.NovaMindAPIError: print('WRITE-DENIED')\n"
                    "else: raise AssertionError('world changed')\n"
                    f"try: open({foreign!r}, 'w').write('bad')\n"
                    "except OSError: print('FILE-DENIED')\n"
                    "else: raise AssertionError('foreign file changed')\n"
                    f"print('PRIVATE-SCRIPT-{role}')\n")
            script_hashes[role] = register(team, role, 'same.py', code)
        team.run(stop_after_day=14)
        assert team.server.tools.current_day == 14
        assert not (team.roles['ceo'].workspace / 'should-not-exist').exists()
        assert (team.roles['ceo'].workspace / 'marker.txt').read_text() == 'stable'
        assert all(sessions[(role, 0)] != sessions[(role, 7)] for role in ROLES)
        for day in (0, 7):
            assert counters[('ceo', day)] == 2
            assert counters[('growth', day)] == 4
            assert counters[('ops_finance', day)] == 3
        assert all(w['day'] in (0, 7) for w in team.test_wire)
        journal = [json.loads(line) for line in (team.root / 'private' / 'messages.jsonl').read_text().splitlines()]
        delivered = [m for m in journal if m['status'] == 'delivered']
        assert len(delivered) == 10
        assert len({m['request_id'] for m in delivered}) == 10
        assert all(m['world_state_id'] and m['sender_session_id'] and m['receiver_session_id'] for m in delivered)
        assert all(m['reply'] and m['text'] and m['sender'] and m['receiver'] for m in delivered)
        for role, runtime in team.roles.items():
            records = [json.loads(line) for line in runtime.audit.path.read_text().splitlines()]
            scripts = [r for r in records if r['kind'] == 'registered_script_execution']
            assert len(scripts) == 2 and all(r['role'] == role for r in scripts)
            assert all(r['sha256'] == script_hashes[role] for r in scripts)
            assert len({r['session_id'] for r in scripts}) == 2
        (team.test_root / 'expected-scripts.json').write_text(json.dumps(script_hashes, indent=2))


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('stop', [28, 112])
def test_observation_stop_precedes_next_scripts_and_models(mode, stop):
    decision_days = {role: [] for role in ROLES}

    def respond(team, role, body):
        day = team.server.tools.current_day
        decision_days[role].append(day)
        assert '500' in body['messages'][0]['content'] and '497' in body['messages'][0]['content']
        assert f'D{stop} liquidation' not in json.dumps(body)
        if role != 'ceo':
            return final(f'Advice from {role} at D{day}')
        return calls(('bash', dict(command=ADVANCE + '; ' + ADVANCE +
                                  '; ./novamind-operation python-c "import novamind_api as nm; nm.pricing.set_prices(A=900)"')),
                     ('write_file', dict(path='after-advance', content='must cancel')))

    with make_team(mode, respond, f'stop-{stop}') as team:
        for role in ROLES:
            code = (
                "import novamind_api as nm\n"
                "with open('script-days.txt','a') as stream: stream.write(str(nm.vars.current_day)+'\\n')\n")
            if role == 'ceo':
                code += "nm.pricing.set_prices(A=31)\n"
            register(team, role, 'same.py', code)
        team.run(stop_after_day=stop)
        expected = list(range(0, stop, 7))
        assert team.server.tools.current_day == stop
        assert decision_days == dict.fromkeys(ROLES, expected)
        assert len(expected) == (4 if stop == 28 else 16)
        for runtime in team.roles.values():
            assert list(map(int, (runtime.workspace / 'script-days.txt').read_text().splitlines())) == expected
        assert not (team.roles['ceo'].workspace / 'after-advance').exists()
        price = team.server.conn.execute('SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0]
        assert price == 31
        assert len([e for e in team.test_timeline if e['kind'] == 'settlement']) == len(expected)
        count = len(team.test_wire)
        team.run(stop_after_day=stop)
        assert len(team.test_wire) == count


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_hold_without_weekly_turn_limit(mode):
    counters = dict.fromkeys(ROLES, 0)

    def respond(team, role, body):
        counters[role] += 1
        if role != 'ceo':
            return final('Free text complete')
        if counters[role] <= 102:
            return calls(('write_file', dict(path='last-turn.txt', content=str(counters[role]))))
        return calls(('bash', dict(command=ADVANCE)))

    with make_team(mode, respond, 'unlimited') as team:
        team.run(stop_after_day=7)
        assert counters == dict(ceo=103, growth=1, ops_finance=1)
        assert (team.roles['ceo'].workspace / 'last-turn.txt').read_text() == '102'
    with make_team(mode, respond, 'held') as team:
        hold = team.test_root / 'HOLD'
        hold.touch()
        team.run(stop_after_day=7, hold=hold)
        assert not team.test_wire and team.server.tools.current_day == 0


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_pause_during_analyst_wait_stops_before_ceo(mode):
    entered = threading.Barrier(2)

    def respond(team, role, body):
        assert role != 'ceo'
        entered.wait(timeout=10)
        team.request_pause()
        return final('Answer received while pause was requested')

    with make_team(mode, respond, 'pause') as team:
        team.run(stop_after_day=7)
        assert team.server.tools.current_day == 0
        assert {w['role'] for w in team.test_wire} == set(ROLES[1:])
        probe_ceo_frozen(team)


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('terminal', ['bankrupt', 'configured_end'])
def test_natural_end_checked_before_scripts_and_calls(mode, terminal):
    def no_call(*args):
        raise AssertionError('terminal world requested a model')

    with make_team(mode, no_call, terminal) as team:
        register(team, 'ceo', 'same.py', "open('terminal-script-ran', 'w').write('bad')")
        if terminal == 'bankrupt':
            team.server.conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES(0,'initial_funding',-2000000,'offline bankruptcy')")
            team.server.conn.commit()
        else:
            team.server.tools.set_current_day(497)
        outcome = team.run(stop_after_day=112)
        assert outcome.reason == 'natural_end' and not team.test_wire
        assert not (team.roles['ceo'].workspace / 'terminal-script-ran').exists()


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('trigger', ['ceo_script', 'ceo_tool'])
def test_bankruptcy_during_week_stops_remaining_scripts_and_tools(mode, trigger, monkeypatch):
    ceo_calls = 0

    def respond(team, role, body):
        nonlocal ceo_calls
        assert trigger == 'ceo_tool'
        if role != 'ceo':
            return final('Advice before bankruptcy')
        ceo_calls += 1
        assert ceo_calls == 1, 'CEO called again after bankruptcy'
        return calls(('bash', dict(command='./novamind-operation python-c ' + shlex.quote(
                         'import novamind_api as nm; nm.pricing.set_prices(A=99)'))),
                     ('write_file', dict(path='after-bankruptcy', content='must cancel')))

    with make_team(mode, respond, f'bankrupt-{trigger}') as team:
        original = team.server.tools.set_prices

        def bankrupting_price(args):
            result = original(args)
            if result.success and args.get('A') == 99:
                team.server.conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES(0,'initial_funding',-2000000,'injected bankruptcy')")
                team.server.conn.commit()
            return result

        monkeypatch.setattr(team.server.tools, 'set_prices', bankrupting_price)
        if trigger == 'ceo_script':
            register(team, 'ceo', 'first.py', 'import novamind_api as nm; nm.pricing.set_prices(A=99)')
            register(team, 'ceo', 'second.py', "open('after-bankruptcy','w').write('must cancel')")
        outcome = team.run(stop_after_day=112)
        assert outcome.reason == 'natural_end' and outcome.day == 0
        assert not (team.roles['ceo'].workspace / 'after-bankruptcy').exists()
        if trigger == 'ceo_script':
            assert not team.test_wire
            records = [json.loads(line) for line in team.roles['ceo'].audit.path.read_text().splitlines()]
            executions = [r for r in records if r['kind'] == 'registered_script_execution']
            assert not any(r['name'] == 'second.py' and r['status'] == 'succeeded' for r in executions)
        else:
            assert len([w for w in team.test_wire if w['role'] == 'ceo']) == 1
        for executor in (team.roles['ceo'].executor, team.server.role_executors['ceo']):
            assert executor.execute('write_file', dict(path='late', content='denied')).startswith('Error:')
            assert executor.last_status != 'succeeded'


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_freeze_waits_for_inflight_ceo_write_before_shared_dashboard(mode, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    threads, errors, expected_world = [], [], []

    def respond(team, role, body):
        assert 'PUBLIC price_A=37' in json.dumps(body['messages'])
        if role != 'ceo':
            assert team.server.world_state_id == expected_world[0]
            return final('Advice from completed CEO write')
        return calls(('bash', dict(command=ADVANCE)))

    with make_team(mode, respond, 'freeze-inflight') as team:
        executor = team.roles['ceo'].executor
        execute = executor._exec_bash

        def blocked_bash(args):
            entered.set()
            assert release.wait(timeout=10)
            result = execute(args)
            expected_world.append(team.server.world_state_id)
            return result

        monkeypatch.setattr(executor, '_exec_bash', blocked_bash)
        scripts, set_phase = team._script_output, team._set_phase

        def run_write():
            try:
                result = executor.execute('bash', dict(command='./novamind-operation python-c ' + shlex.quote(
                    'import novamind_api as nm; nm.pricing.set_prices(A=37)')))
                assert executor.last_status == 'succeeded', result
            except BaseException as exc:
                errors.append(exc)

        def start_write(role):
            result = scripts(role)
            if role == 'ceo':
                thread = threading.Thread(target=run_write)
                threads.append(thread)
                thread.start()
                assert entered.wait(timeout=10)
            return result

        def freeze(phase):
            if phase == 'analysis':
                def finish_write():
                    time.sleep(.1)
                    if team.phase != 'ceo_scripts':
                        errors.append(AssertionError('freeze crossed an active CEO executor'))
                    release.set()
                thread = threading.Thread(target=finish_write)
                threads.append(thread)
                thread.start()
            return set_phase(phase)

        monkeypatch.setattr(team, '_script_output', start_write)
        monkeypatch.setattr(team, '_set_phase', freeze)
        outcome = team.run(stop_after_day=7)
        release.set()
        for thread in threads:
            thread.join(timeout=10)
            assert not thread.is_alive()
        assert not errors, errors
        assert outcome.reason == 'observation_end'


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_http_retry_attempt_limit_and_hold_during_retry(mode, monkeypatch):
    waits = []

    def immediate_sleep(lifecycle, seconds):
        waits.append(seconds)
        lifecycle.check(retry=True)

    monkeypatch.setattr(WorkerLifecycle, 'sleep', immediate_sleep)
    counts = dict.fromkeys(ROLES, 0)

    def unavailable(team, role, body):
        assert role != 'ceo'
        counts[role] += 1
        return httpx.Response(503, json=dict(error=dict(message='offline retry probe', type='server_error')))

    with make_team(mode, unavailable, 'retry-limit') as team:
        outcome = team.run(stop_after_day=7)
        assert outcome.reason in ('failed', 'paused')
        assert max(counts.values()) == 4 and all(n <= 4 for n in counts.values())
        assert counts['ceo'] == 0 and team.server.tools.current_day == 0
        assert waits and max(waits) <= 180
        probe_ceo_frozen(team)

    def hold_on_error(team, role, body):
        (team.test_root / 'HOLD').touch()
        return unavailable(team, role, body)

    before = dict(counts)
    with make_team(mode, hold_on_error, 'retry-hold') as team:
        outcome = team.run(stop_after_day=7, hold=team.test_root / 'HOLD')
        assert outcome.reason == 'paused'
        assert len(team.test_wire) <= 2 and team.server.tools.current_day == 0
        assert counts['ceo'] == before['ceo']
