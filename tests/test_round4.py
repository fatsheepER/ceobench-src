"""Round-four regressions: real shell programs and all provider tool batches, offline."""
import json
from pathlib import Path
import re

import httpx
import pytest

from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from test_text_registry import workspace, captured, send
from test_preflight_integration import offline_runner, packed_public
from test_public_sql import server


@pytest.mark.parametrize('command', [
    './novamind-operation query "SELECT * FROM config_overrides ORDER BY day DESC LIMIT 40"; pf log settings.py | head -30',
    'pf log settings.py; ./novamind-operation python report.py',
    './novamind-operation query "SELECT * FROM config_overrides" | head -c 80; pf log settings.py',
])
def test_mixed_pf_output_can_be_cited_and_traced(offline_runner, command):
    from test_pf_queries import query
    runner = offline_runner(text_registration='pf')
    runner._execute_tool('write_file', dict(path='settings.py', content='# settings\n'))
    runner._execute_tool('write_file', dict(path='report.py', content=(
        'import novamind_api as nm\nprint(nm.query("SELECT * FROM config_overrides"))\n'
        'raise RuntimeError("after printing")\n')))
    output = runner._execute_tool('bash', dict(command=command))
    assert 'columns' in output and output.pf_calls
    match = re.search(r'\[pf: (cmd\d+@v\d+)', output)
    assert match, output
    handle = match.group(1)
    send(runner.evidence_store, output)
    receipt = runner._execute_tool('text_create', dict(text='Saved investment history',
        objects=[dict(kind='customer_group', id='S1')], applies='0-', reason='history',
        references=[dict(cite=handle)]))
    assert receipt.startswith('Registered r1.1'), receipt
    shown = runner._execute_tool('bash', dict(command=f'pf show {handle} --full'))
    assert shown.split('\n', 1)[1] == output
    graph = query(runner.tool_executor, 'pf_dependencies', target={'record': 'r1'},
                  include_execution=True, depth=4)
    assert any('config_overrides' in (row['target'] or {}).get('sql', '') for row in graph['items'])
    retrieved = runner._execute_tool('bash', dict(command='pf depend r1 --detail'))
    assert not re.search(r'\[pf: cmd\d+@v\d+', retrieved)
    version = retrieved.origins[0]['version_id']
    assert runner.evidence_store.get_content(version)[0]['pf_retrieval']
    runner.tool_executor.pf_queries._index(None)
    assert version not in runner.tool_executor.pf_queries.nodes
    send(runner.evidence_store, retrieved)
    with pytest.raises(ValueError, match='Only public captured evidence'):
        resolver = runner.tool_executor.text_registry.resolver
        resolver.resolve({'version': resolver.handle(version)}, {})


def test_failed_citation_keeps_its_identity_and_revision_history(offline_runner):
    runner = offline_runner(text_registration='pf')
    output = runner._execute_tool('bash', dict(command='./novamind-operation python-c \'\n'
        'import novamind_api as nm\nprint(nm.query("SELECT name FROM sqlite_master"))\n\'',
        note='List DB tables'))
    assert '[exit code: 1]' in output
    handle = re.search(r'\[pf: (cmd\d+@v\d+)', output).group(1)
    send(runner.evidence_store, output)
    for i in (1, 2):
        receipt = runner._execute_tool('text_create', dict(text='Development history verified',
            objects=[dict(kind='customer_group', id='S1')], applies='0-', reason='reproduce mismatch',
            references=[dict(cite=handle, note='config_overrides history')]))
        assert receipt.startswith(f'Registered r{i}.1'), receipt
        assert 'failed' in receipt and 'sqlite_master' in receipt and 'exit code: 1' in receipt
        assert 'List DB tables' in receipt
    listed = json.loads(runner._execute_tool('text_list', {'limit': 1, 'raw': True}))
    assert listed['next_after'] == 1 and len(listed['records']) == 1
    assert listed['records'][0]['status'] == 'active'
    assert listed['records'][0]['references'][0]['evidence']['version'] == handle
    more = json.loads(runner._execute_tool('text_list', {'after': listed['next_after'], 'raw': True}))
    assert [r['version'] for r in more['records']] == ['r2.1']
    history = runner._execute_tool('bash', dict(command='./novamind-operation query '
        '"SELECT * FROM config_overrides ORDER BY day DESC LIMIT 40"; pf log MEMORY.md'))
    send(runner.evidence_store, history)
    correct = re.search(r'\[pf: (cmd\d+@v\d+)', history).group(1)
    revised = runner._execute_tool('text_revise', dict(record='r1', text='Use the actual investment history',
        reason='The earlier citation was a rejected table lookup', references=[dict(cite=correct)]))
    assert revised.startswith('Revised r1.2') and 'config_overrides' in revised
    listed = json.loads(runner._execute_tool('text_list', {'raw': True}))
    assert [record['version'] for record in listed['records']] == ['r1.2', 'r2.1']
    records = json.loads(runner.tool_executor.text_registry.path.read_text())['records']['r1']
    assert records[0]['references'][0]['evidence']['version'] == handle
    assert records[1]['references'][0]['evidence']['version'] == correct


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
    # The entire tool batch came from one recorded model request, including the
    # write that precedes the context reset caused by next-week.
    from contextlib import closing
    with closing(runner.evidence_store.connect()) as conn:
        facts = [json.loads(row[0]) for row in conn.execute("SELECT request FROM requests WHERE "
            "json_extract(request, '$.request.path') IN ('before','missing') AND "
            "json_extract(request, '$.model_request_event') IS NOT NULL")]
    assert len(facts) == 2
    assert len({f['model_request_event'] for f in facts}) == 1
    request = runner.evidence_store.read_event(facts[0]['model_request_event'])
    assert request['result']['send_state'] == 'response_received'
    assert all(f['model_context_id'] == request['request']['request']['context_id'] for f in facts)


def test_checkpoint_freeze_is_immutable_and_recovery_preserves_original(offline_runner, monkeypatch, tmp_path, caplog):
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
    assert runner._load_checkpoint() == checkpoint
    assert 'Discarding post-checkpoint operation outcomes' in caplog.text
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


def test_recovery_before_first_fork_checkpoint_preserves_group(offline_runner, tmp_path):
    from saas_bench.run_state import clone_sql_run, recover_run, tree_hash, write_json
    from test_preflight_integration import advance
    prefix = offline_runner(text_registration='prefix')
    advance(prefix)
    prefix._save_checkpoint(7)
    for mode in ('git', 'pf'):
        name = mode
        source = clone_sql_run(prefix.workspace_dir, tmp_path / name, name,
                               text_registration=mode)
        before = tree_hash(source)
        recovered_path = recover_run(source, tmp_path / (name + '-recovery'))
        recovered = offline_runner(recovered_path)
        assert recovered.text_registration == mode
        assert bool(recovered.evidence_store) is (mode == 'pf')
        assert tree_hash(source) == before
        # A second interruption before publication still inherits the prefix generation.
        again = offline_runner(recover_run(recovered_path, tmp_path / (name + '-again')))
        assert again.text_registration == mode
        assert bool(again.evidence_store) is (mode == 'pf')
        manifest = json.loads((source / 'manifest.json').read_text())
        manifest['configuration']['seed'] += 1
        write_json(source / 'manifest.json', manifest)
        revised = recover_run(source, tmp_path / (name + '-revised'))
        assert (revised / 'checkpoint.json').exists()


def test_saved_survey_dates_reach_log_and_full_read(workspace, tmp_path):
    from saas_bench.execution_capture import finish_http
    from saas_bench.sql_evidence import encoded
    from pf_fixtures import cite
    store, registry, executor = captured(workspace, tmp_path)
    day = [7]
    registry.sim_day = lambda: day[0]
    dashboard = store.begin_event('dashboard_generation', {'day': 7})
    store.complete(dashboard)
    def acquire(parent=None):
        event = store.begin_event('public_http', dict(method='POST', path='/call',
            parsed=dict(tool='get_group_insights', args={'group_id': 'S1'})), parent=parent)
        finish_http(store, event, 200, '', encoded(dict(success=True,
                    data=dict(group_id='S1', snapshot_day=0))), dict(day=day[0]))
        return event + ':public_response'
    version = acquire()
    handle = registry.resolver.handle(version)
    cite(store, registry, executor, version)
    day[0] = 308
    output = executor.execute('bash', dict(command=f'pf log {handle}'))
    assert 'survey day 0' in output
    assert 'acquired day' in output and 'research_group' in output
    header = json.loads(executor.execute('pf_read', dict(target={'version': handle})).split('\n')[0])
    assert header['target']['snapshot_day'] == 0
    assert header['target']['day'] == 7


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
