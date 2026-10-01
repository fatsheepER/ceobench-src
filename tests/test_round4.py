"""Round-four regressions: real shell programs and all provider tool batches, offline."""
import json
from pathlib import Path

import httpx
import pytest

from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from test_text_registry import workspace, captured, send
from test_preflight_integration import offline_runner, packed_public
from test_public_sql import server


def test_all_28_recorded_pf_programs_use_bash(workspace, tmp_path):
    store, _, executor = captured(workspace, tmp_path)
    for name in ['MEMORY.md', 'w15_h.py.out', 'w27_state.py.out', 'w27_deep.py.out',
                 *[f'w{n}_act.py.out' for n in [24, 25, 26, 27, 28, 30]]]:
        executor.execute('write_file', dict(path=name, content='saved receipt\n', note='producer'))
    executor.execute('write_file', dict(path='MEMORY.md', content='updated memory\n'))
    cases = json.loads((Path(__file__).parent / 'fixtures/round4_pf_commands.json').read_text())
    calls = []
    for case in cases:
        output = executor.execute('bash', case['arguments'])
        assert output.pf_calls, case
        for call in output.pf_calls:
            assert call['note'] == case['arguments'].get('note')
            event = store.read_event(call['event_id'])
            assert event['result']['exit_code'] == call['exit_code']
        calls.extend(output.pf_calls)
    assert len(cases) == 28 and len(calls) == 39
    assert not store.fault


def test_shell_redirection_errors_cwd_notes_and_log_ownership(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    executor.execute('bash', dict(command="mkdir -p 'business/sessions' sessions; echo hidden > sessions/a.log; "
                                 "echo keep > business/sessions/a.log; printf 'one\\ntwo\\n' > 'business/a b.txt'"))
    assert store.latest_version('sessions/a.log', 'file_bytes') == (None, None)
    assert store.latest_version('business/sessions/a.log', 'file_bytes')
    output = executor.execute('bash', dict(command="cd business; pf show 'a b.txt' > saved.txt; "
        "pf show missing 2>failure.txt | cat; echo status:$?; pf log 'a b.txt' | grep '^file'", note='reader'))
    assert 'status:0' in output and 'file business/a b.txt:' in output
    assert [c['exit_code'] for c in output.pf_calls] == [0, 1, 0]
    assert (workspace / 'business/failure.txt').read_text().startswith('Error:')
    send(store, output)
    assert not any(o['version_id'] == store.latest_version('business/saved.txt', 'file_bytes')[0]
                   for o in output.origins)
    untouched = executor.execute('bash', dict(command="cat <<'EOF'\npf show does-not-exist\nEOF"))
    assert untouched.pf_calls == [] and 'pf show does-not-exist' in untouched
    assert executor.execute('bash', dict(command='pf')).pf_calls[0]['exit_code'] == 0
    assert executor.execute('bash', dict(command='pf log')).pf_calls[0]['exit_code'] == 2


@pytest.mark.parametrize('api', ['chat', 'responses', 'messages'])
def test_provider_batches_return_each_id_once_and_restore(tmp_path, api):
    from openai import OpenAI
    from anthropic import Anthropic
    from test_preflight_usage import reply
    calls = [('first', 'write_file', {'path': 'a', 'content': 'a'}),
             ('second', 'read_file', {'path': 'missing'}),
             ('third', 'read_file', {'path': 'a'})]
    def handle(request):
        body = reply(api)
        if api == 'chat':
            body['choices'][0]['message']['tool_calls'] = [dict(id=i, type='function',
                function=dict(name=n, arguments=json.dumps(a))) for i, n, a in calls]
            return httpx.Response(200, json=body)
        if api == 'responses':
            body['output'] = [dict(type='function_call', id='fc_'+i, call_id=i, name=n,
                                  arguments=json.dumps(a), status='completed') for i, n, a in calls]
            return httpx.Response(200, json=body)
        body['content'] = []
        events = [dict(type='message_start', message=body)]
        for index, (i, n, a) in enumerate(calls):
            events.extend([dict(type='content_block_start', index=index,
                                content_block=dict(type='tool_use', id=i, name=n, input={})),
                           dict(type='content_block_delta', index=index,
                                delta=dict(type='input_json_delta', partial_json=json.dumps(a))),
                           dict(type='content_block_stop', index=index)])
        events.extend([dict(type='message_delta', delta={'stop_reason': 'tool_use', 'stop_sequence': None},
                            usage={'output_tokens': 2}), dict(type='message_stop')])
        return httpx.Response(200, content=''.join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events),
                              headers={'content-type': 'text/event-stream'})
    client = (Anthropic if api == 'messages' else OpenAI)(api_key='offline-only', max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    agent = BashAgent(get_bash_agent_tool_descriptions(), client, workspace_path=tmp_path,
                     reasoning_effort='low' if api == 'responses' else None)
    action = agent.act('dashboard', 0, False, {'day': 0})
    for i, n, a in calls:
        assert action.tool == n and action.arguments == a
        agent.record_tool_result('Error: missing' if i == 'second' else i, call_id=i)
        action = agent.next_tool_action()
    assert action is None
    if api == 'messages':
        assert [b['tool_use_id'] for b in agent.conversation[-1].content] == [c[0] for c in calls]
    else:
        assert [m.tool_call_id for m in agent.conversation if m.role == 'tool'] == [c[0] for c in calls]
    agent._snapshot_path = tmp_path / 'context.json'
    agent._save_conversation_snapshot(strict=True)
    assert agent.load_conversation_snapshot(agent._snapshot_path)
    client.close()


def test_runner_cancels_after_week_advance_only(offline_runner, monkeypatch):
    from openai import OpenAI
    from test_preflight_usage import reply
    runner = offline_runner(text_registration='pf', stop_after_day=7)
    def handle(request):
        body = reply('chat')
        actions = [('read_file', {'path': 'missing'}), ('write_file', {'path': 'before', 'content': 'yes'}),
                   ('bash', {'command': "./novamind-operation next-week 'offline test'" + ' 100000 -100000 1000000' * 4}),
                   ('write_file', {'path': 'after', 'content': 'no'})]
        body['choices'][0]['message']['tool_calls'] = [dict(id=str(i), type='function',
            function=dict(name=n, arguments=json.dumps(a))) for i, (n, a) in enumerate(actions)]
        return httpx.Response(200, json=body)
    runner.client.close()
    runner.client = OpenAI(api_key='offline-only', base_url='https://api.deepseek.com', max_retries=0,
                          http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    runner.agent.client = runner.agent.usage_recorder.attach(runner.client)
    monkeypatch.setattr(runner, 'setup', lambda: None)
    result = runner.run(verbose=False)
    assert result['outcome'] == 'stopped'
    assert (runner.agent.workspace_path / 'before').read_text() == 'yes'
    assert not (runner.agent.workspace_path / 'after').exists()
    results = [m for m in runner.agent.conversation if m.role == 'tool']
    assert [m.tool_call_id for m in results] == ['0', '1', '2', '3']
    assert results[-1].content.startswith('Cancelled:')


def test_checkpoint_freeze_is_immutable_and_recovery_preserves_original(offline_runner, monkeypatch, tmp_path):
    import threading
    from saas_bench import db_protection
    from saas_bench.run_state import recover_run, checkpoint_directory, tree_hash
    runner = offline_runner(text_registration='pf')
    runner._execute_tool('write_file', {'path': 'MEMORY.md', 'content': 'frozen'})
    runner._save_checkpoint(0)
    pointer = (runner.workspace_dir / 'checkpoint.json').read_bytes()
    entered, finish = threading.Event(), threading.Event()
    encrypt = db_protection.encrypt_plain_atomic
    def delayed(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return encrypt(*args, **kwargs)
    monkeypatch.setattr(db_protection, 'encrypt_plain_atomic', delayed)
    runner._save_checkpoint(0, wait=False)
    assert entered.wait(2)
    try:
        assert (runner.workspace_dir / 'checkpoint.json').read_bytes() == pointer
        runner._begin_operation('write_file', 0)
        runner._execute_tool('write_file', {'path': 'MEMORY.md', 'content': 'after freeze'})
    finally:
        finish.set()
    runner._wait_checkpoint()
    checkpoint = json.loads((runner.workspace_dir / 'checkpoint.json').read_text())
    directory = checkpoint_directory(runner.workspace_dir, checkpoint)
    assert (directory / 'agent_workspace/MEMORY.md').read_text() == 'frozen'
    with pytest.raises(ValueError, match='recover_run.py'):
        runner._load_checkpoint()
    runner._stop_server()
    before = tree_hash(runner.workspace_dir)
    with pytest.raises(ValueError, match='outside the original run'):
        recover_run(runner.workspace_dir, runner.workspace_dir / 'recovery')
    recovered = offline_runner(recover_run(runner.workspace_dir, tmp_path / 'recovery'))
    assert recovered.agent_workspace.joinpath('MEMORY.md').read_text() == 'frozen'
    recovered._execute_tool('write_file', {'path': 'new', 'content': 'new attempt'})
    recovered._weekly_record(recovered._get_game_status())
    usage = json.loads((recovered.workspace_dir / 'usage_summary.json').read_text())
    assert usage['attempt_usage']['agent']['calls'] == 0
    assert tree_hash(runner.workspace_dir) == before
    assert recovered.sql_evidence_config['branch_id'] != runner.sql_evidence_config['branch_id']


@pytest.mark.parametrize('end_day', [35, 42])
def test_checkpoint_cadence_keeps_weekly_metrics_without_duplicate_world_saves(offline_runner, monkeypatch, end_day):
    from test_stage5_prep import fake_weeks
    runner = offline_runner(text_registration='pf')
    runner.total_days = end_day
    world = runner.agent_workspace / 'sessions' / runner._session_id / 'world.nmdb'
    before = world.read_bytes()
    requests = fake_weeks(runner, monkeypatch)
    assert runner.run(verbose=False)['outcome'] == 'completed'
    assert len(requests) == end_day // 7
    checkpoints = [json.loads(p.read_text()) for p in (runner.workspace_dir / 'checkpoints').glob('*/checkpoint.json')]
    assert sorted(c['day'] for c in checkpoints) == sorted({0, 35, end_day})
    assert world.read_bytes() == before
    assert [json.loads(line)['day'] for line in (runner.workspace_dir / 'weekly.jsonl').read_text().splitlines()] == list(range(7, end_day+1, 7))


def test_pending_review_keeps_conditions_live_and_manual_scope_matters(workspace, tmp_path, server):
    from test_pf_stale import chain, forward
    from test_text_registry import call, declaration
    store, registry, executor = chain(workspace, tmp_path, server)
    source = store.load_state('declaration:r1.1')['references'][0]['version_id']
    handle = registry.resolver.handle(source)
    call(registry, 'revise', record='r1', reason='add an explicit condition', references=[
        dict(evidence={'version': handle}, purpose='current'),
        dict(evidence={'version': handle}, purpose='current', select={'col': 'amount'},
             predicate={'type': 'threshold', 'op': '>=', 'value': 40})])
    query = executor.pf_queries
    refreshed = []
    original = query.refresh
    query.refresh = lambda versions, parent: (refreshed.append(versions) or original(versions, parent))
    server.conn.execute('UPDATE ledger SET amount=39'); server.conn.commit()
    first = query.weekly_check(7)
    assert 'PREDICATE FAILS' in first and 'Pending review:' in first
    pending = store.load_state('pf_review')['pending']
    root = next(v for v in pending if query.records[v]['version'] == 'r1.2')
    server.conn.execute('UPDATE ledger SET amount=42'); server.conn.commit()
    second = query.weekly_check(14)
    assert 'PREDICATE FAILS' not in second
    assert store.load_state('pf_review')['pending'][root]['last_day'] == 7
    assert len(refreshed) == 2 and all(len(v) == 1 for v in refreshed)
    # A complete manual check of this root can clear it after the source recovers.
    forward(executor, 'r1', depth=1)
    assert root not in store.load_state('pf_review')['pending']
    query._index(None)
    warm = {v: list(e) for v, e in query.outgoing.items() if e}
    query._index(None)
    assert query.index_stats['new_versions'] == 0
    from saas_bench.pf_queries import PFQueries
    cold = PFQueries(registry)
    cold._index(None)
    assert {v: list(e) for v, e in cold.outgoing.items() if e} == warm


def test_each_refresh_source_gets_its_own_timeout(workspace, tmp_path, server, monkeypatch):
    from saas_bench.pf_refresh import refresh
    from saas_bench.public_sql import execute_query
    from test_pf_stale import record_sql
    store, _, _ = captured(workspace, tmp_path)
    server.sql_evidence = store
    slow_sql = 'WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x) SELECT sum(n) FROM x'
    slow = record_sql(store, slow_sql, {'success': True, 'columns': ['n'], 'rows': [{'n': 0}], 'row_count': 1})
    fast = record_sql(store, 'SELECT amount FROM ledger', execute_query(server, 'SELECT amount FROM ledger'))
    parent = store.begin_event('pf_weekly_check', {'day': 0})
    monkeypatch.setattr(server, 'QUERY_TIMEOUT_SECONDS', .025)
    result = refresh(server, [slow, fast], parent)
    receipts = [store.read_event(store.get_content(result[v])[0]['created_by_event'])['result'] for v in (slow, fast)]
    assert [r['http_status'] for r in receipts] == [504, 200]
    assert all(r['attempted'] for r in receipts)
    store.complete(parent)


def test_retry_backoff_and_manual_retry_do_not_block_other_sources(workspace, tmp_path, server):
    from test_pf_stale import chain, forward
    from saas_bench.public_sql import PUBLIC_POLICY_VERSION
    from saas_bench.sql_evidence import encoded
    store, registry, executor = chain(workspace, tmp_path, server, select={'col': 'amount'})
    query, attempts = executor.pf_queries, []
    day = [7]
    registry.sim_day = lambda: day[0]
    def timeout(versions, parent):
        attempts.append(day[0])
        output = {}
        for version in versions:
            event = store.begin(encoded({'sql': 'SELECT amount FROM ledger'}), PUBLIC_POLICY_VERSION, parent)
            store.finish(event, 504, '', encoded({'success': False, 'error': 'refresh_timed_out'}),
                         dict(day=day[0], attempted=True))
            store.delivered(event, 'internal')
            output[version] = event + ':public_response'
        return output
    query.refresh = timeout
    for current in [7, 14, 21, 28, 35]:
        day[0] = current
        query.weekly_check(current)
    assert attempts == [7, 14, 28]
    forward(executor, 'r1')
    assert attempts == [7, 14, 28, 35]


def test_failed_background_checkpoint_keeps_the_previous_pointer(offline_runner, monkeypatch):
    from saas_bench import db_protection
    from saas_bench.run_state import checkpoint_directory
    runner = offline_runner()
    runner._save_checkpoint(0)
    pointer = (runner.workspace_dir / 'checkpoint.json').read_bytes()
    def failed(*args, **kwargs):
        raise OSError('injected encryption failure')
    monkeypatch.setattr(db_protection, 'encrypt_plain_atomic', failed)
    runner._save_checkpoint(0, wait=False)
    with pytest.raises(OSError, match='encryption failure'):
        runner._wait_checkpoint()
    assert (runner.workspace_dir / 'checkpoint.json').read_bytes() == pointer
    assert checkpoint_directory(runner.workspace_dir, json.loads(pointer)).is_dir()
