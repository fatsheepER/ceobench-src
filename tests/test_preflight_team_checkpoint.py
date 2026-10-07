from dataclasses import MISSING, fields
import hashlib
import json
from pathlib import Path
import re
import shlex
import sqlite3
import threading

import httpx
from numpy.random import default_rng
from openai import OpenAI
import pytest

from saas_bench.config import BenchmarkConfig
from saas_bench.database import init_database
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.role_policy import ROLES
from saas_bench.run_state import write_json
from saas_bench.simulation import DayResult, Simulator
from saas_bench.tools import AgentTools
from test_preflight_team_flow import ADVANCE, calls, completion, final, raw


PRICING = dict(source='offline fixed table', basis='USD per 1000 tokens',
    rates={'deepseek-v4.1-flash': dict(input=.1, output=.2, cache_read=.01)})


def fast_simulator(conn, config, rng):
    simulator = Simulator(conn, config, rng)
    def step():
        simulator.current_day += 7
        amount = int(simulator.rng.integers(100, 1000))
        conn.execute('INSERT INTO ledger(day,category,amount,note) VALUES(?,?,?,?)',
            (simulator.current_day, 'ad_revenue', amount, 'deterministic settlement'))
        conn.commit()
        required = {field.name: 0 for field in fields(DayResult) if field.default is MISSING
            and field.default_factory is MISSING}
        return DayResult(**dict(required, day=simulator.current_day))
    simulator.step_week = step
    return simulator


def clients():
    def create(role):
        def respond(request):
            body = json.loads(request.content)
            messages = body['messages']
            if role != 'ceo':
                question = messages[-1]['content']
                return completion(final(role + ' answer ' + question), body.get('stream', False))
            asks = [message for message in messages if message.get('name') == 'ask_analyst']
            if not asks:
                message = calls(('ask_analyst', dict(role='growth', message='Check the existing advice again.')))
            else:
                message = calls(('write_file', dict(path='final.txt', content='same final decision')),
                    ('bash', dict(command=ADVANCE)))
            return completion(message, body.get('stream', False))
        return OpenAI(api_key='OFFLINE-ONLY', base_url='https://opencode.ai/zen/go/v1', max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    return create


def make_team(root, mode):
    root.mkdir()
    conn = init_database(':memory:')
    config = BenchmarkConfig(seed=123)
    simulator = fast_simulator(conn, config, default_rng(123))
    simulator.initialize()
    tools = AgentTools(conn, 0, root / 'host', rng=simulator.rng, config=config, seed=123)
    runtime = MultiAgentRuntime(root / 'run', tools, simulator=simulator, conn=conn,
        client_factory=clients(), mode=mode, model_pricing=PRICING)
    for role in ROLES:
        code = "import novamind_api as nm\nwith open('script-days.txt','a') as stream: stream.write(str(nm.vars.current_day)+'\\n')\n"
        if role == 'ceo':
            code += "nm.market.research_market()\n"
        status, result = raw(runtime, role, '/daily-scripts', dict(name='same.py', content=code))
        assert status == 200 and result['success'], result
        (runtime.roles[role].workspace / 'MEMORY.md').write_text(role + ' memory')
    return runtime


def dispose(runtime):
    for role in runtime.roles.values():
        role.agent.client.close()
    runtime.close()
    runtime.server.conn.close()


def behavior(runtime):
    runtime.server.simulator.save_rng_states()
    dump = '\n'.join(runtime.server.conn.iterdump())
    (runtime.root / 'private' / 'final-world.sql').write_text(dump)
    tables = {}
    for (table,) in runtime.server.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_stat%'"):
        columns = [row[1] for row in runtime.server.conn.execute(f'PRAGMA table_info("{table}")')
            if table != 'predictions' or row[1] != 'submitted_at']
        selected = ','.join(f'"{name}"' for name in columns)
        tables[table] = dict(columns=columns, rows=[tuple(row) for row in
            runtime.server.conn.execute(f'SELECT {selected} FROM "{table}"')])
    business = json.dumps(tables, sort_keys=True)
    (runtime.root / 'private' / 'final-world.json').write_text(business)
    root = str(runtime.root)
    state = {}
    for role, item in runtime.roles.items():
        conversation = [dict(role=message.role, content=message.content,
            name=message.name, reasoning_content=message.reasoning_content)
            for message in item.agent.conversation]
        conversation = json.loads(json.dumps(conversation).replace(root, '<ROOT>'))
        for message in conversation:
            if isinstance(message['content'], str):
                message['content'] = re.sub(r'(\[git: MEMORY\.md last committed in (?:week-\d+ \()?)([0-9a-f]{7})',
                    lambda match: match[1] + 'tree:' + runtime._git(item, 'rev-parse', match[2] + '^{tree}'),
                    message['content'])
        state[role] = dict(conversation=conversation,
            files={str(path.relative_to(item.workspace)): path.read_text() for path in item.workspace.glob('*.txt')},
            memory=(item.workspace / 'MEMORY.md').read_text(),
            history=re.sub(r'Analyst handoff [0-9a-f]{7}', 'Analyst handoff <ID>', runtime._git(item, 'log', '--format=%s')),
            trees=runtime._git(item, 'log', '--format=%T'), usage=item.usage.summary)
    ledger = [tuple(row) for row in runtime.server.conn.execute('SELECT day,category,amount,note FROM ledger ORDER BY rowid')]
    state['world'] = dict(day=runtime.server.tools.current_day, ledger=ledger,
        database_sha256=hashlib.sha256(business.encode()).hexdigest(),
        rng=runtime.server.simulator.rng.bit_generator.state,
        tools_rng=runtime.server.tools.rng.bit_generator.state)
    state['messages'] = [(message.sender, message.receiver, message.text, message.day,
        message.status, message.reply) for message in runtime.messages]
    return state


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('boundary', ['answers_ready', 'ask_completed', 'week_boundary'])
def test_continuous_and_recovered_team_match_all_boundaries(tmp_path, mode, boundary):
    continuous = make_team(tmp_path / 'continuous', mode)
    snapshots = []
    def save(runtime, current):
        if current == boundary and not snapshots:
            snapshots.append(runtime.checkpoint())
    continuous.checkpoint_callback = save
    try:
        outcome = continuous.run(stop_after_day=14)
        assert outcome.reason == 'observation_end', continuous.failure
        expected = behavior(continuous)
        snapshot = snapshots[0]
        checksum = hashlib.sha256((snapshot / 'world.nmdb').read_bytes()).hexdigest()
        restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored',
            client_factory=clients(), simulator_factory=fast_simulator)
        try:
            saved = json.loads((snapshot / 'team.json').read_text())
            assert all(restored.roles[role].identity.session_id == saved['roles'][role]['identity']['session_id']
                for role in ROLES)
            assert restored.run(stop_after_day=14).reason == 'observation_end', restored.failure
            actual = behavior(restored)
            write_json(tmp_path / 'comparison.json', dict(mode=mode, boundary=boundary,
                continuous=expected, recovered=actual, equal=actual == expected))
            assert actual == expected
            for role in ROLES:
                assert (restored.roles[role].workspace / 'script-days.txt').read_text() == '0\n7\n'
                assert restored.roles[role].identity.session_id != saved['roles'][role]['identity']['session_id']
            journal = [json.loads(line) for line in restored.message_path.read_text().splitlines()]
            delivered = [row['request_id'] for row in journal if row['status'] == 'delivered']
            assert len(delivered) == len(set(delivered)) == 6
            assert checksum == hashlib.sha256((snapshot / 'world.nmdb').read_bytes()).hexdigest()
            assert restored.configuration['model_pricing'] == PRICING
            assert all(role.usage.pricing == PRICING['rates'] for role in restored.roles.values())
        finally:
            dispose(restored)
    finally:
        dispose(continuous)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_independent_clones_execute_unstarted_scripts_and_preserve_attempts(tmp_path, mode):
    original = make_team(tmp_path / 'source', mode)
    snapshot = original.checkpoint()
    clones = [MultiAgentRuntime.restore(snapshot, tmp_path / name,
        client_factory=clients(), simulator_factory=fast_simulator) for name in ('one', 'two')]
    try:
        clones[0].roles['ceo'].executor.execute('write_file', dict(path='only-one', content='one'))
        assert not (clones[1].roles['ceo'].workspace / 'only-one').exists()
        assert not (original.roles['ceo'].workspace / 'only-one').exists()
        for clone in clones:
            assert clone.run(stop_after_day=7).reason == 'observation_end', clone.failure
            assert (clone.roles['ceo'].workspace / 'script-days.txt').read_text() == '0\n'
        assert len(list(original.root.glob('checkpoints/*/checkpoint.json'))) == 1
        assert all((clone.root / 'recovery.json').exists() for clone in clones)
    finally:
        for clone in clones:
            dispose(clone)
        dispose(original)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_unknown_business_write_refuses_checkpoint_and_old_safe_restore(tmp_path, mode, monkeypatch):
    runtime = make_team(tmp_path / 'unknown', mode)
    snapshot = runtime.checkpoint()
    original = runtime.server.tools.set_prices
    def unknown(args):
        original(args)
        raise ConnectionError('write happened; response outcome unknown')
    monkeypatch.setattr(runtime.server.tools, 'set_prices', unknown)
    try:
        status, result = raw(runtime, 'ceo', '/call', dict(tool='set_prices', args={'A': 29}))
        assert status == 500, result
        with pytest.raises(RuntimeError, match='unknown'):
            runtime.checkpoint()
        with pytest.raises(RuntimeError, match='unknown'):
            MultiAgentRuntime.restore(snapshot, tmp_path / 'forbidden',
                client_factory=clients(), simulator_factory=fast_simulator)
        assert list((runtime.root / 'private' / 'operations').glob('*.json'))
    finally:
        dispose(runtime)


def test_checkpoint_guards_and_frozen_configuration(tmp_path, monkeypatch):
    runtime = make_team(tmp_path / 'guards', 'git')
    try:
        runtime.roles['ceo'].agent._pending_tool_calls = [dict(id='unknown-tool')]
        with pytest.raises(RuntimeError, match='Unresolved'):
            runtime.checkpoint()
        runtime.roles['ceo'].agent._pending_tool_calls = []
        runtime.server._advance_lock.acquire()
        try:
            with pytest.raises(RuntimeError, match='settlement'):
                runtime.checkpoint()
        finally:
            runtime.server._advance_lock.release()
        runtime.server._sql_active = 1
        try:
            with monkeypatch.context() as patch:
                patch.setattr(runtime.server._sql_lock, 'wait_for', lambda predicate, timeout: False)
                with pytest.raises(RuntimeError, match='query'):
                    runtime.checkpoint()
            assert not runtime.server._sql_paused
        finally:
            runtime.server._sql_active = 0
        snapshot = runtime.checkpoint()
        with pytest.raises(ValueError, match='configuration drift'):
            MultiAgentRuntime.restore(snapshot, tmp_path / 'wrong',
                client_factory=clients(), model='different-model')
        with pytest.raises(FileExistsError):
            runtime.checkpoint(snapshot)
        with pytest.raises(ValueError, match='client configuration drift'):
            MultiAgentRuntime.restore(snapshot, tmp_path / 'changed-client',
                client_factory=lambda role: OpenAI(api_key='OFFLINE', max_retries=1))
        assert snapshot.exists()
    finally:
        dispose(runtime)


def test_checkpoint_refuses_active_model_request(tmp_path):
    runtime = make_team(tmp_path / 'model-active', 'git')
    started, release = threading.Event(), threading.Event()
    def response():
        started.set()
        assert release.wait(10)
        return dict(model='deepseek-v4.1-flash', choices=[dict(message=dict(role='assistant', content='done'))])
    thread = threading.Thread(target=lambda: runtime.roles['growth'].usage.call(
        'chat', {'model': 'deepseek-v4.1-flash'}, response, day=0))
    try:
        thread.start()
        assert started.wait(10)
        with pytest.raises(RuntimeError, match='Model request'):
            runtime.checkpoint()
        release.set()
        thread.join(10)
        assert not thread.is_alive()
        assert runtime.checkpoint().exists()
    finally:
        release.set()
        thread.join(10)
        dispose(runtime)


def test_relocated_prompt_preserves_memory_source_spans():
    from saas_bench.execution_capture import CapturedText, origin
    from saas_bench.team_checkpoint import _relocate_prompt
    original, destination = '/old/run', '/longer/new/run'
    memory = 'Saved path /old/run/notes.txt.'
    prefix = 'Workspace /old/run\nMemory\n'
    text = CapturedText(prefix + memory + '\nReadable /old/run',
        [origin('memory@v1', memory, target=len(prefix))], pf_read={'reader': 'ceo'})
    relocated = _relocate_prompt(text, original, destination)
    assert relocated == 'Workspace /longer/new/run\nMemory\n' + memory + '\nReadable /longer/new/run'
    assert isinstance(relocated, CapturedText) and relocated.pf_read == text.pf_read
    a, b = relocated.origins[0]['request_range']
    assert relocated[a:b] == memory
    assert relocated.origins[0]['source_range'] == [0, len(memory)]
