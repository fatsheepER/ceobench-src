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
from saas_bench.execution_capture import text_sources
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.team_checkpoint import validate_snapshot
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


def make_team(root, mode, **options):
    root.mkdir()
    conn = init_database(':memory:')
    config = BenchmarkConfig(seed=123)
    simulator = fast_simulator(conn, config, default_rng(123))
    simulator.initialize()
    tools = AgentTools(conn, 0, root / 'host', rng=simulator.rng, config=config, seed=123)
    runtime = MultiAgentRuntime(root / 'run', tools, simulator=simulator, conn=conn,
        client_factory=clients(), mode=mode, model_pricing=PRICING, **options)
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
    worlds = list(dict.fromkeys(message.world_state_id for message in runtime.messages))
    header = re.compile(r'Analyst handoff \| role=(\w+) \| day=(\d+) \| world_state_id=(\w+) \| request_id=(\w+)\n'
                        r'Workspace: ([^\n]+)\nSnapshot commit: ([0-9a-f]+)\n')

    def normalize_handoff(match):
        role, day, world, request_id, workspace, commit = match.groups()
        index, request = next((i, r) for i, r in enumerate(runtime.messages) if r.request_id == request_id)
        assert (role, int(day), world, commit) == (request.receiver, request.day, request.world_state_id, request.handoff_commit)
        assert workspace == '<ROOT>/roles/' + role
        tree = runtime._git(runtime.roles[role], 'rev-parse', commit + '^{tree}')
        return (f'Analyst handoff | role={role} | day={day} | world_state_id=state-{worlds.index(world)} | request_id=request-{index}\n'
                f'Workspace: {workspace}\nSnapshot commit: tree-{tree}\n')

    for role, item in runtime.roles.items():
        conversation = [dict(role=message.role, content=message.content,
            name=message.name, reasoning_content=message.reasoning_content)
            for message in item.agent.conversation]
        conversation = json.loads(json.dumps(conversation).replace(root, '<ROOT>'))
        for message in conversation:
            if isinstance(message['content'], str):
                message['content'] = header.sub(normalize_handoff, message['content'])
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
    snapshots, forks = [], []
    def save(runtime, current):
        if current == boundary and not snapshots:
            snapshots.append(runtime.checkpoint())
            # Fork while the source is still at this handoff, before it advances.
            forks.append(MultiAgentRuntime.restore(snapshots[0], tmp_path / 'restored',
                client_factory=clients(), simulator_factory=fast_simulator))
    continuous.checkpoint_callback = save
    try:
        outcome = continuous.run(stop_after_day=14)
        assert outcome.reason == 'observation_end', continuous.failure
        expected = behavior(continuous)
        snapshot = snapshots[0]
        checksum = hashlib.sha256((snapshot / 'world.nmdb').read_bytes()).hexdigest()
        restored = forks[0]
        try:
            saved = json.loads((snapshot / 'team.json').read_text())
            assert all(restored.roles[role].identity.session_id == saved['roles'][role]['identity']['session_id']
                for role in ROLES)
            relocated_sources = text_sources(dict(week=restored._week, dashboard=restored.server._last_dashboard,
                script_results=restored.server.role_script_results))
            assert len(relocated_sources) == len(saved['sources'])
            for old, new in zip(saved['sources'], relocated_sources):
                assert old['pointer'] == new['pointer']
                assert len(old['origins']) == len(new['origins'])
                for before, after in zip(old['origins'], new['origins']):
                    assert {k: v for k, v in before.items() if k != 'request_range'} == {
                        k: v for k, v in after.items() if k != 'request_range'}
                    a, b = before['request_range']
                    c, d = after['request_range']
                    assert old['text'][a:b] == new['text'][c:d]
            if boundary != 'week_boundary':
                for request in restored.messages:
                    assert restored._handoff(request) in restored._week['observation'] or any(
                        restored._handoff(request) in m.content for m in restored.roles['ceo'].agent.conversation
                        if isinstance(m.content, str))
            if mode == 'pf' and boundary == 'answers_ready':
                assert any(s.get('pf_read_spans') for s in saved['sources'])
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
            forks.clear()
    finally:
        for fork in forks:
            dispose(fork)
        dispose(continuous)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_checkpoint_restores_bash_files_with_host_runtime_names(tmp_path, mode):
    original = make_team(tmp_path / 'source', mode)
    restored = None
    try:
        for role in original.roles.values():
            role.executor.execute('bash', dict(command="printf 'role data' > saved.pid; printf 'role port' > .server.port"))
            code = "from pathlib import Path; import py_compile; p=Path('weekly_rules.py'); p.write_text('PRICE = 29\\n'); py_compile.compile(str(p), cfile='weekly_rules.pyc', doraise=True); p.unlink(); Path('__pycache__').mkdir(exist_ok=True); Path('__pycache__/artifact.bin').write_bytes(b'role artifact')"
            role.executor.execute('bash', dict(command='python -c ' + shlex.quote(code)))
            assert (role.workspace / 'saved.pid').read_text() == 'role data'
        snapshot = original.checkpoint()
        restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored',
            client_factory=clients(), simulator_factory=fast_simulator)
        for role in restored.roles.values():
            assert (role.workspace / 'saved.pid').read_text() == 'role data'
            assert (role.workspace / '.server.port').read_text() == 'role port'
            assert role.executor.execute('bash', dict(command='python -c "import weekly_rules; print(weekly_rules.PRICE)"')).strip() == '29'
            assert (role.workspace / '__pycache__/artifact.bin').read_bytes() == b'role artifact'
        assert restored.run(stop_after_day=7).reason == 'observation_end', restored.failure
    finally:
        if restored:
            dispose(restored)
        dispose(original)


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('filename', ['weekly_rules.pyc', '__pycache__/artifact.bin'])
def test_checkpoint_rolls_back_later_bytecode_or_cache_writes(tmp_path, mode, filename):
    runtime = make_team(tmp_path / 'bytecode-write', mode)
    restored = None
    try:
        snapshot = runtime.checkpoint()
        runtime.roles['growth'].executor.execute('bash', dict(
            command='mkdir -p __pycache__; printf changed > ' + shlex.quote(filename)))
        restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored',
            client_factory=clients(), simulator_factory=fast_simulator)
        assert not (restored.roles['growth'].workspace / filename).exists()
        assert (runtime.roles['growth'].workspace / filename).read_bytes() == b'changed'
    finally:
        if restored:
            dispose(restored)
        dispose(runtime)


def test_legacy_role_checksum_warns_about_unchecked_bytecode(tmp_path, caplog):
    from saas_bench.run_state import tree_hash, write_json
    runtime = make_team(tmp_path / 'legacy-role-hash', 'git')
    restored = rejected = None
    try:
        snapshot = runtime.checkpoint()
        receipt = json.loads((snapshot / 'checkpoint.json').read_text())
        receipt.pop('roles_hash_policy', None)
        write_json(snapshot / 'checkpoint.json', receipt)
        restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'legacy-restored',
            client_factory=clients(), simulator_factory=fast_simulator)
        path = runtime.roles['growth'].workspace / 'weekly_rules.pyc'
        path.write_bytes(b'uncheckpointed')
        rejected = MultiAgentRuntime.restore(snapshot, tmp_path / 'legacy-rollback',
            client_factory=clients(), simulator_factory=fast_simulator)
        assert not (rejected.roles['growth'].workspace / 'weekly_rules.pyc').exists()
        assert path.read_bytes() == b'uncheckpointed'
        (snapshot / 'roles/growth/weekly_rules.pyc').write_bytes(b'unchecked')
        assert tree_hash(snapshot / 'roles') == receipt['roles_sha256']
        validate_snapshot(snapshot)
        assert 'checksum mismatch for roles' in caplog.text
        receipt['roles_hash_policy'] = 'unknown'
        write_json(snapshot / 'checkpoint.json', receipt)
        validate_snapshot(snapshot)
        assert 'Unknown role checkpoint hash policy' in caplog.text
    finally:
        if restored:
            dispose(restored)
        if rejected:
            dispose(rejected)
        dispose(runtime)


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
def test_unknown_business_write_rejects_new_snapshot_and_restores_old_checkpoint(tmp_path, mode, monkeypatch):
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
        restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored',
            client_factory=clients(), simulator_factory=fast_simulator)
        try:
            assert restored.server.conn.execute('SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] != 29
            assert restored.server.tools.current_day == 0
        finally:
            dispose(restored)
        assert list((runtime.root / 'private' / 'operations').glob('*.json'))
    finally:
        dispose(runtime)


def test_restore_records_execution_revision_without_approval(tmp_path, monkeypatch):
    from saas_bench import team_checkpoint
    runtime = make_team(tmp_path / 'source', 'git')
    restored = None
    try:
        snapshot = runtime.checkpoint()
        checksum = team_checkpoint.tree_hash
        package = Path(team_checkpoint.__file__).parent
        monkeypatch.setattr(team_checkpoint, 'tree_hash', lambda path, **options:
            'changed-source' if Path(path) == package else checksum(path, **options))
        monkeypatch.setattr(team_checkpoint, 'artifact_hashes', lambda path: {'revision': 'changed-public'})
        before = {path: path.read_bytes() for path in snapshot.rglob('*') if path.is_file()}
        restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored',
            client_factory=clients(), simulator_factory=fast_simulator)
        receipt = json.loads((restored.root / 'recovery.json').read_text())
        assert receipt['source_revision_changed'] and receipt['public_build_changed']
        assert restored.server.tools.current_day == 0
        assert {path: path.read_bytes() for path in before} == before
        assert not (snapshot.parent / '.retention-pins.json').exists()
    finally:
        if restored:
            dispose(restored)
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
        changed = MultiAgentRuntime.restore(snapshot, tmp_path / 'wrong',
            client_factory=clients(), model='different-model')
        try:
            assert changed.roles['ceo'].agent.model == 'different-model'
            assert changed.server.tools.config.social_post_llm_model == 'different-model'
            assert changed.server.simulator.config is changed.server.tools.config
        finally:
            dispose(changed)
        with pytest.raises(FileExistsError):
            runtime.checkpoint(snapshot)
        changed = MultiAgentRuntime.restore(snapshot, tmp_path / 'changed-client',
            client_factory=lambda role: OpenAI(api_key='OFFLINE', max_retries=1))
        try:
            assert changed.roles['ceo'].agent.client.max_retries == 1
            assert changed.server.tools.current_day == runtime.server.tools.current_day
        finally:
            dispose(changed)
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
    only_read = CapturedText(prefix + memory, pf_read_spans=[dict(id='read',
        request_range=[len(prefix), len(prefix) + len(memory)])])
    moved = _relocate_prompt(only_read, original, destination)
    a, b = moved.pf_read_spans[0]['request_range']
    assert moved[a:b] == memory and a == len(prefix.replace(original, destination))
    nested = _relocate_prompt([dict(type='tool_result', tool_use_id='ask', content=text)], original, destination)
    assert nested[0]['content'] == relocated
    assert nested[0]['content'].origins == relocated.origins
    git_reply = 'Workspace: /old/run/roles/growth\nEvidence contains /old/run/notes.txt'
    moved_git = _relocate_prompt(git_reply, 'Workspace: /old/run/roles/', 'Workspace: /new/run/roles/')
    assert moved_git == 'Workspace: /new/run/roles/growth\nEvidence contains /old/run/notes.txt'


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_bash_policy_checkpoint_roundtrip_and_save_drift(tmp_path, mode):
    runtime = make_team(tmp_path / 'thresholds', mode, analyst_bash_timeout=2, analyst_bash_warn_after=.05)
    restored = None
    try:
        snapshot = runtime.checkpoint()
        restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored-thresholds',
            client_factory=clients(), simulator_factory=fast_simulator,
            analyst_bash_timeout=2, analyst_bash_warn_after=.05)
        for name, role in restored.roles.items():
            expected = (1200, None) if name == 'ceo' else (2, .05)
            assert (role.executor.bash_timeout, role.executor.bash_warn_after) == expected
            automatic = restored.server.role_executors[name]
            assert (automatic.bash_timeout, automatic.bash_warn_after) == (1200, None)
            description = next(tool['description'] for tool in role.agent.tool_descriptions if tool['name'] == 'bash')
            assert ('0.05 seconds' in description) == (name != 'ceo')
        assert '[Framework timing notice]' in restored.roles['growth'].executor.execute('bash', dict(command='sleep .08; echo 42'))
        changed = MultiAgentRuntime.restore(snapshot, tmp_path / 'drift', client_factory=clients(), analyst_bash_warn_after=.1)
        try:
            assert changed.roles['growth'].executor.bash_warn_after == .1
        finally:
            dispose(changed)
        runtime.roles['growth'].executor.bash_timeout = 3
        drift = runtime.checkpoint()
        assert validate_snapshot(drift)[1]['roles']['growth']['executor']['bash_timeout'] == 3
    finally:
        if restored:
            dispose(restored)
        dispose(runtime)


def test_legacy_checkpoint_preserves_custom_hard_limit_and_absent_soft(tmp_path, monkeypatch):
    from saas_bench import team_checkpoint
    runtime = make_team(tmp_path / 'legacy', 'git')
    restored = None
    try:
        runtime.configuration.pop('analyst_bash_timeout')
        runtime.configuration.pop('analyst_bash_warn_after')
        runtime.roles['growth'].executor.bash_timeout = 17
        runtime.server.role_executors['ops_finance'].bash_timeout = 29
        snapshot = runtime.checkpoint()
        validate = team_checkpoint.validate_snapshot
        def legacy(source):
            directory, state = validate(source)
            for saved in state['roles'].values():
                saved['executor'].pop('bash_warn_after')
                saved['script_executor'].pop('bash_warn_after')
            return directory, state
        monkeypatch.setattr(team_checkpoint, 'validate_snapshot', legacy)
        restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored-legacy',
            client_factory=clients(), simulator_factory=fast_simulator)
        assert restored.roles['growth'].executor.bash_timeout == 17
        assert restored.server.role_executors['ops_finance'].bash_timeout == 29
        assert all(role.executor.bash_warn_after is None for role in restored.roles.values())
        assert '[Framework timing notice]' not in restored.roles['growth'].executor.execute('bash', dict(command='echo 42'))
        for name, settings in [('new-policy', dict(analyst_bash_timeout=480, analyst_bash_warn_after=120)),
                               ('new-soft', dict(analyst_bash_warn_after=1))]:
            changed = MultiAgentRuntime.restore(snapshot, tmp_path / name, client_factory=clients(), **settings)
            try:
                assert changed.roles['growth'].executor.bash_timeout == settings.get('analyst_bash_timeout', 17)
                assert changed.roles['growth'].executor.bash_warn_after == settings['analyst_bash_warn_after']
            finally:
                dispose(changed)
    finally:
        if restored:
            dispose(restored)
        dispose(runtime)


def test_checkpoint_preserves_saved_executor_policy_disagreement(tmp_path, monkeypatch, caplog):
    from saas_bench import team_checkpoint
    runtime = make_team(tmp_path / 'mismatch', 'git', analyst_bash_timeout=480, analyst_bash_warn_after=120)
    try:
        snapshot = runtime.checkpoint()
        validate = team_checkpoint.validate_snapshot
        def changed(source):
            directory, state = validate(source)
            state['roles']['growth']['executor']['bash_warn_after'] = 60
            return directory, state
        monkeypatch.setattr(team_checkpoint, 'validate_snapshot', changed)
        restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored', client_factory=clients())
        try:
            assert restored.roles['growth'].executor.bash_warn_after == 60
            assert 'Bash policy drift' in caplog.text
        finally:
            dispose(restored)
    finally:
        dispose(runtime)


def test_explicit_recovery_config_reaches_requests_executors_and_simulator(tmp_path):
    runtime = make_team(tmp_path / 'override-source', 'git', total_days=42,
                        analyst_bash_timeout=2, analyst_bash_warn_after=.05)
    restored = None
    requests = []
    try:
        snapshot = runtime.checkpoint()
        original = (snapshot / 'team.json').read_bytes()
        factory = clients()
        def create(role):
            client = factory(role)
            client._client.event_hooks['request'].append(lambda request: requests.append(json.loads(request.content)))
            return client
        restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'override-restored',
            client_factory=create, simulator_factory=fast_simulator,
            model='override-model', reasoning_effort='low', total_days=56,
            ceo_bash_timeout=19, analyst_bash_timeout=7, analyst_bash_warn_after=.1)
        assert restored.total_days == restored.server.tools.config.total_days == 56
        assert restored.server.tools.config is restored.server.simulator.config
        assert restored.server.tools.config.social_post_llm_model == 'override-model'
        assert restored.roles['ceo'].executor.bash_timeout == 19
        for name in ('growth', 'ops_finance'):
            assert restored.roles[name].executor.bash_timeout == 7
            assert restored.roles[name].executor.bash_warn_after == .1
        assert restored.run(stop_after_day=7).reason == 'observation_end'
        assert requests and all(body['model'] == 'override-model' for body in requests)
        assert all('maximize final cash over about 56 days' in body['messages'][0]['content']
                   for body in requests)
        changes = json.loads((restored.root / 'recovery.json').read_text())['configuration_overrides']
        assert changes['total_days'] == dict(old=42, new=56, effective_at='new_attempt')
        assert (snapshot / 'team.json').read_bytes() == original
    finally:
        if restored:
            dispose(restored)
        dispose(runtime)


def test_recovery_retains_numeric_settings_unless_explicitly_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv('CEOBENCH_MODEL_REQUEST_ATTEMPTS', '9')
    monkeypatch.setenv('CEOBENCH_BASH_OUTPUT_MAX_CHARS', '45678')
    runtime = make_team(tmp_path / 'numeric-source', 'git')
    restored = changed = None
    try:
        snapshot = runtime.checkpoint()
        monkeypatch.delenv('CEOBENCH_MODEL_REQUEST_ATTEMPTS')
        monkeypatch.delenv('CEOBENCH_BASH_OUTPUT_MAX_CHARS')
        restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'numeric-restored',
            client_factory=clients(), simulator_factory=fast_simulator)
        assert all(role.usage.model_request_attempts == 9 for role in restored.roles.values())
        assert restored.roles['ceo'].executor.runtime_environment['CEOBENCH_BASH_OUTPUT_MAX_CHARS'] == '45678'
        import os
        assert 'CEOBENCH_MODEL_REQUEST_ATTEMPTS' not in os.environ
        assert 'CEOBENCH_BASH_OUTPUT_MAX_CHARS' not in os.environ
        monkeypatch.setenv('CEOBENCH_MODEL_REQUEST_ATTEMPTS', '7')
        changed = MultiAgentRuntime.restore(snapshot, tmp_path / 'numeric-changed',
            client_factory=clients(), simulator_factory=fast_simulator)
        assert all(role.usage.model_request_attempts == 7 for role in changed.roles.values())
        assert all(role.usage.model_request_attempts == 9 for role in restored.roles.values())
        saved = json.loads((changed.root / 'recovery.json').read_text())['runtime_environment_overrides']
        assert saved['CEOBENCH_MODEL_REQUEST_ATTEMPTS']['old'] == '9'
        assert saved['CEOBENCH_MODEL_REQUEST_ATTEMPTS']['new'] == '7'
    finally:
        for team in (changed, restored, runtime):
            if team:
                dispose(team)
