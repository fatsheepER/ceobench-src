from contextlib import closing
import json
import os
from pathlib import Path
import re
import shutil

import httpx
from openai import OpenAI
import pytest

from saas_bench.agents.bash_agent.agent import BashAgent, Message
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from saas_bench.config import BenchmarkConfig
from saas_bench.model_usage import ModelUsage
from saas_bench.novamind_api import _client
from saas_bench.pf_queries import PFQueries
from saas_bench.role_policy import AgentIdentity
from saas_bench.text_registry import TextRegistry
from test_execution_capture import captured as http_captured, records
from test_preflight_usage import reply
from test_public_sql import server
from test_sql_evidence import identity, settled
from test_text_registry import captured, declaration, workspace


def artifact(name, value):
    if directory := os.environ.get('CEOBENCH_RECEIPT_ARTIFACTS'):
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        test = re.sub(r'[^a-zA-Z0-9_-]', '_', os.environ.get('PYTEST_CURRENT_TEST', 'receipt'))
        (target / f'{test}-{name}').write_text(value if isinstance(value, str) else
            json.dumps(value, ensure_ascii=False, indent=2))


def model_receives(store, workspace, output, tool='bash'):
    requests = []

    def respond(request):
        requests.append(json.loads(request.read()))
        response = reply('chat')
        response['choices'][0]['message']['tool_calls'] = [dict(id='next', type='function',
            function=dict(name='bash', arguments='{"command":"true"}'))]
        return httpx.Response(200, json=response)

    usage = ModelUsage(None, 'agent', evidence_store=store)
    with OpenAI(api_key='offline', base_url='https://api.deepseek.com', max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        agent = BashAgent(get_bash_agent_tool_descriptions(True, True), client, model='test-model',
            workspace_path=workspace, usage_recorder=usage, text_registration=True, pf=True)
        agent.current_day = 7
        call = dict(id='result', name=tool, arguments={})
        agent.conversation = [Message(role='assistant', content='', tool_calls=[dict(
            id='result', type='function', function=dict(name=tool, arguments='{}'))])]
        agent._pending_tool_calls = [call]
        agent.record_tool_result(output)
        assert agent.act('', 0, False, {'day': 7}).tool == 'bash'
    assert len(requests) == 1
    wire = next(message['content'] for message in requests[0]['messages'] if message['role'] == 'tool')
    assert wire == output
    artifact(f'model-input-{tool}-{len(output)}.json', requests[0])
    artifact(f'model-event-{tool}-{len(output)}.json', {'event_id': usage.last_request_event})
    return wire


@pytest.mark.parametrize('mode', ['git', 'prefix', 'pf'])
def test_text_list_compact_raw_pagination_and_model_coverage(workspace, tmp_path, mode):
    store, registry, executor = captured(workspace, tmp_path, mode)
    actor = AgentIdentity('test-run', 'world-1', 'ceo', 'ceo-agent', 'ceo-session')
    registry.identity = actor
    long_text = 'Saved first sentence.\n' + 'HIDDEN_BODY ' * 80
    first = executor.execute('text_create', declaration(text=long_text,
        applies_at={'start_day': 7, 'end_day': 13}))
    assert not first.startswith('Error:'), first
    second = executor.execute('text_create', declaration(text='Second saved text.',
        applies_at={'unknown': 'pending date'}))
    assert not second.startswith('Error:'), second
    saved = registry.path.read_bytes()
    listing = executor.execute('text_list', {'limit': 1})
    artifact('text-list.txt', listing)
    assert 'ceo:r1.1' in listing and 'active' in listing and '7-13' in listing
    assert 'Saved first sentence.' in listing and 'HIDDEN_BODY' not in listing
    assert all(field not in listing for field in ('agent_id', 'session_id', 'world_id', 'run_id'))
    assert '"after":1' in listing and '"raw":true' in listing
    wire = model_receives(store, workspace, listing, 'text_list')
    assert 'saved' in wire.lower() and 'checked' not in wire.lower()
    second = executor.execute('text_list', {'after': 1, 'limit': 1})
    assert 'ceo:r2.1' in second and 'unknown: pending date' in second
    assert 'No active texts' in executor.execute('text_list', {'after': 2})
    raw = executor.execute('text_list', {'raw': True, 'limit': 1})
    artifact('text-list-raw.json', raw)
    data = json.loads(raw)
    assert data['records'][0]['text'] == long_text
    assert data['shareable_references'] == {'r1.1': 'ceo:r1.1'}
    assert data['next_after'] == 1 and registry.path.read_bytes() == saved
    if mode != 'git':
        version = store.load_state('declaration:r1.1')['version_id']
        assert not any(o['version_id'] == version for o in getattr(listing, 'origins', []))
        assert any(o['version_id'] == version for o in raw.origins)
    model_receives(store, workspace, raw, 'text_list')
    store.assert_healthy()


def test_log_commands_are_separate_and_shell_quoted(workspace, tmp_path):
    _, _, executor = captured(workspace, tmp_path)
    for value in ('before\n', 'after\n'):
        executor.execute('write_file', {'path': 'my plan.txt', 'content': value})
    output = executor.execute('bash', {'command': "pf log 'my plan.txt'"})
    artifact('log.txt', output)
    assert "pf blame 'my plan.txt'\n" in output
    assert "Before changing this source: pf rdepend 'my plan.txt@v2' --detail." in output
    commands = [line for line in output.splitlines() if line.startswith('pf ')]
    assert len(commands) == 3
    for command in commands:
        assert not executor.execute('bash', {'command': command}).startswith('Error:')


def test_multi_capture_inventory_reaches_all_results_and_frozen_pages(http_captured, monkeypatch):
    api, store, executor = http_captured
    registry = TextRegistry(executor.workspace_path, 'pf', store, sim_day=lambda: 7)
    executor.text_registry = registry
    executor.pf_queries = PFQueries(registry)
    shutil.copy(Path(__file__).parents[1] / 'public/novamind-operation', executor.workspace_path)
    monkeypatch.setitem(api.runtime_environment, 'CEOBENCH_SQL_ROW_LIMIT', '2')
    code = '''import json
from pathlib import Path
import novamind_api as n
from novamind_api import _client
print(json.dumps(n.query('SELECT amount FROM ledger WHERE 0')))
print(json.dumps(n.query('SELECT 1 AS n UNION ALL SELECT 2 UNION ALL SELECT 3')))
for _ in range(2):
    print(n.vars.current_day)
try:
    _client.call('not_a_tool')
except _client.NovaMindAPIError as exc:
    print('Rejected:', exc)
for i in range(23):
    Path(f'part-{i}.txt').write_text(f'saved {i}')
'''
    (executor.workspace_path / 'make.py').write_text(code)
    output = executor.execute('bash', {'command': './novamind-operation python make.py'})
    artifact('multi-output.txt', output)
    assert '2 SQL' in output and '3 API' in output
    command = re.search(r'captures: (pf more c[0-9]+)', output)
    assert command, output
    wire = model_receives(store, executor.workspace_path, output)
    assert '"rows": []' in wire and 'truncated' in wire and 'Rejected: Unknown tool' in wire
    page_text = executor.execute('bash', {'command': command[1]})
    page = executor.pf_queries.last_answer
    artifact('captures-first.json', page)
    assert 'Captured in' in page_text and 'not a dependency claim' in page_text
    assert page['next_cursor'] and page['remaining'] > 0
    items = list(page['items'])
    executor.execute('write_file', {'path': 'later.txt', 'content': 'outside original execution'})
    while page['next_cursor']:
        page_text = executor.execute('bash', {'command': 'pf more ' + page['next_cursor']})
        page = executor.pf_queries.last_answer
        artifact(f'captures-remaining-{page["remaining"]}.json', page)
        items += page['items']
    assert len(items) == page['total']
    assert {item['path'] for item in items if item.get('path')} == {f'part-{i}.txt' for i in range(23)}
    queries = [item for item in items if item.get('sql')]
    assert len(queries) == 2 and {item['rows'] for item in queries} == {0, 2}
    assert any(item.get('truncated') for item in queries)
    responses = [item for item in items if item['layer'] == 'server_public_response']
    assert len(responses) == 5 and any(item['status'] == 'failed' for item in responses)
    before = len([r for r in records(store) if r['request']['kind'] in ('sql_query', 'public_http')])
    for item in items:
        read = executor.execute('bash', {'command': 'pf show ' + item['version'] + ' --full'})
        assert not read.startswith('Error:'), read
    assert len([r for r in records(store) if r['request']['kind'] in ('sql_query', 'public_http')]) == before
    model_receives(store, executor.workspace_path, page_text)
    settled(api)
    store.assert_healthy()


def test_capacity_change_survives_public_sdk(server, make_agent_tools, monkeypatch):
    server.tools = make_agent_tools(server.conn, BenchmarkConfig(), day=7)
    server.conn.execute('''INSERT INTO config_history(day,price_A,price_B,price_C,tier_A,tier_B,tier_C,
        spend_advertising,spend_operations,spend_development,capacity_tier)
        VALUES(0,10,20,30,1,2,3,0,0,0,4)''')
    server.conn.commit()
    server.start()
    monkeypatch.setenv('NOVAMIND_API_PORT', str(server.port))
    result = _client.call('set_capacity_tier', {'tier': 3})
    artifact('capacity.json', result)
    assert result['tier'] == 3
    assert result['changes'] == {'capacity_tier': {'before': 4, 'after': 3}}
    assert server.conn.execute('SELECT capacity_tier FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] == 3
    assert _client.call('set_capacity_tier', {'tier': 3})['changes'] == {}


def test_sdk_reports_http_refusal_reason(server, monkeypatch):
    server.start()
    monkeypatch.setenv('NOVAMIND_API_PORT', str(server.port))
    monkeypatch.setattr(server, 'execute_tool', lambda *args, **kwargs: (_ for _ in ()).throw(
        PermissionError('This role cannot write prices')))
    with pytest.raises(_client.NovaMindAPIError, match='^This role cannot write prices$'):
        _client.call('set_prices', {'A': 29})


def test_read_file_reports_requested_scope_and_empty_result(workspace, tmp_path):
    _, _, executor = captured(workspace, tmp_path)
    (workspace / 'lines.txt').write_text('one\ntwo\nthree')
    result = executor.execute('read_file', {'path': 'lines.txt', 'offset': 2, 'limit': 1})
    assert 'lines 2-2 of 3' in result and 'offset=3' in result
    assert 'one' not in result and 'two' in result and 'three' not in result
    assert 'past end of file' in executor.execute('read_file', {'path': 'lines.txt', 'offset': 8})


def test_repeated_command_keeps_each_footer_bound_to_its_execution(http_captured):
    api, store, executor = http_captured
    executor.text_registry = TextRegistry(executor.workspace_path, 'pf', store, sim_day=lambda: 7)
    executor.pf_queries = PFQueries(executor.text_registry)
    shutil.copy(Path(__file__).parents[1] / 'public/novamind-operation', executor.workspace_path)
    script = executor.workspace_path / 'same.py'
    script.write_text("from pathlib import Path\nimport novamind_api as n\nn.query('SELECT ' + Path('selector').read_text() + ' AS n')\nprint('same stdout')\n")
    outputs, cursors = [], []
    for value in ('1', '2'):
        (executor.workspace_path / 'selector').write_text(value)
        output = executor.execute('bash', {'command': './novamind-operation python same.py'})
        outputs.append(output)
        cursors.append(re.search(r'captures: (pf more c\d+)', output)[1])
    artifact('repeated-outputs.json', outputs)
    assert outputs[0].split('|')[0] == outputs[1].split('|')[0]
    for i, command in enumerate(cursors, 1):
        result = executor.execute('bash', {'command': command})
        artifact(f'repeated-execution-{i}.txt', result)
        assert [item['sql'] for item in executor.pf_queries.last_answer['items'] if 'sql' in item] == [f'SELECT {i} AS n']
        assert 'this execution' in result
    bash = [row for row in records(store) if row['request']['kind'] == 'bash'
            and row['request']['request']['command'] == './novamind-operation python same.py']
    version = bash[0]['request']['event_id'] + ':tool_return'
    handle = executor.text_registry.resolver.handle(version)
    latest = executor.execute('bash', {'command': f'pf show {handle} --captures'})
    assert 'latest captured execution' in latest
    assert [item['sql'] for item in executor.pf_queries.last_answer['items'] if 'sql' in item] == ['SELECT 2 AS n']
    artifact('capture-event-ids.json', [row['request']['event_id'] for row in bash])
    settled(api)
    store.assert_healthy()


@pytest.mark.parametrize('text,visible,hidden', [
    ('First ' + 'word ' * 100 + '. Second.', 'First', 'Second'),
    ('第一句。第二句不能默认展开。', '第一句。', '第二句'),
])
def test_text_excerpt_bounds_and_sentence_breaks(workspace, tmp_path, text, visible, hidden):
    _, registry, executor = captured(workspace, tmp_path)
    assert executor.execute('text_create', declaration(text=text)).startswith('Registered')
    result = executor.execute('text_list', {})
    assert visible in result and hidden not in result and '... …' not in result
    assert json.loads(executor.execute('text_list', {'raw': True}))['records'][0]['text'] == text


@pytest.mark.parametrize('name', [
    'set_targeted_ad_spend', 'set_targeted_dev_spend', 'set_targeted_ops_spend',
    'set_ads_strength', 'set_lead_promotion', 'set_promotion',
])
def test_targeted_changes_are_actual_leaves_and_same_writes_still_execute(server, make_agent_tools, monkeypatch, name):
    from copy import deepcopy
    from test_runtime_config_tools import SETTINGS
    server.tools = make_agent_tools(server.conn, BenchmarkConfig(), day=7)
    server.start()
    monkeypatch.setenv('NOVAMIND_API_PORT', str(server.port))
    initial = deepcopy(SETTINGS[name])
    result = _client.call(name, initial)
    assert result['changes'] and all(value['before'] != value['after'] for value in result['changes'].values())
    before = server.conn.execute('SELECT COUNT(*) FROM config_overrides').fetchone()[0]
    assert _client.call(name, initial)['changes'] == {}
    assert server.conn.execute('SELECT COUNT(*) FROM config_overrides').fetchone()[0] == before + 1
    scope = next(key for key, value in initial.items() if isinstance(value, dict))
    cleared = _client.call(name, {scope: {}})
    artifact(name + '-clear.json', cleared)
    assert cleared['changes'] and all(value['after'] is None for value in cleared['changes'].values())
    expected = initial[scope]
    assert len(cleared['changes']) == sum(len(v) if isinstance(v, dict) else 1 for v in expected.values())
    assert all(not isinstance(value['before'], dict) for value in cleared['changes'].values())
    for other, value in initial.items():
        if other != scope:
            assert cleared['global' if other.startswith('global_') else other] == value


@pytest.mark.parametrize('name,args,expected', [
    ('set_prices', {'A': 15}, {'price_A': {'before': 10, 'after': 15}}),
    ('set_model_tiers', {'B': 3}, {'tier_B': {'before': 2, 'after': 3}}),
    ('set_daily_spend', {'operations': 40}, {'spend_operations': {'before': 0, 'after': 40}}),
    ('set_usage_quotas', {'A': 50, 'B': 20, 'C': 30}, {'quota_A': {'before': 10, 'after': 50}}),
])
def test_plan_setting_changes_survive_public_sdk(server, make_agent_tools, monkeypatch, name, args, expected):
    server.tools = make_agent_tools(server.conn, BenchmarkConfig(), day=7)
    server.conn.execute('''INSERT INTO config_history(day,price_A,price_B,price_C,tier_A,tier_B,tier_C,
        spend_advertising,spend_operations,spend_development,capacity_tier,quota_A,quota_B,quota_C)
        VALUES(0,10,20,30,1,2,3,0,0,0,4,10,20,30)''')
    server.conn.commit()
    server.start()
    monkeypatch.setenv('NOVAMIND_API_PORT', str(server.port))
    result = _client.call(name, args)
    artifact(name + '.json', result)
    assert result['changes'] == expected
    assert _client.call(name, args)['changes'] == {}


def test_capture_cursor_rechecks_role_visibility():
    from test_preflight_team_flow import final, make_team
    with make_team('pf', lambda *args: final('done'), 'capture-cursor-access') as team:
        private = team.roles['ops_finance']
        output = private.executor.execute('bash', {'command': 'echo secret > private.txt; echo done'})
        cursor = re.search(r'captures: pf more (c\d+)', output)[1]
        saved = private.store.load_state('pf_cursors')[cursor]
        growth = team.roles['growth']
        growth.store.save_state('pf_cursors', {'c1': saved})
        denied = growth.executor.execute('bash', {'command': 'pf more c1'})
        assert 'inaccessible' in denied.lower() and 'private.txt' not in denied


def test_business_batch_partial_failure_reaches_model_with_actual_changes(http_captured, make_agent_tools):
    api, store, executor = http_captured
    api.tools = make_agent_tools(api.conn, BenchmarkConfig(), day=7)
    api.conn.execute('''INSERT INTO config_history(day,price_A,price_B,price_C,tier_A,tier_B,tier_C,
        spend_advertising,spend_operations,spend_development,capacity_tier)
        VALUES(0,10,20,30,1,2,3,0,0,0,4)''')
    api.conn.commit()
    executor.text_registry = TextRegistry(executor.workspace_path, 'pf', store, sim_day=lambda: 7)
    executor.pf_queries = PFQueries(executor.text_registry)
    shutil.copy(Path(__file__).parents[1] / 'public/novamind-operation', executor.workspace_path)
    (executor.workspace_path / 'batch.py').write_text('''import json
from novamind_api import infrastructure, _client
for tier in (3, 4, 99):
    try:
        print(json.dumps(infrastructure.set_capacity_tier(tier)))
    except _client.NovaMindAPIError as exc:
        print('Rejected:', exc)
''')
    output = executor.execute('bash', {'command': './novamind-operation python batch.py'})
    artifact('partial-business-batch.txt', output)
    wire = model_receives(store, executor.workspace_path, output)
    assert '"before": 4, "after": 3' in wire and '"before": 3, "after": 4' in wire
    assert 'Rejected: Capacity tier must be 0-7' in wire
    result = executor.execute('bash', {'command': re.search(r'captures: (pf more c\d+)', output)[1]})
    artifact('partial-business-captures.txt', result)
    assert 'Partial failure' in result
    calls = [item for item in executor.pf_queries.last_answer['items'] if item['layer'] == 'server_public_response']
    assert [item['status'] for item in calls] == ['succeeded', 'succeeded', 'failed']
    assert api.conn.execute('SELECT capacity_tier FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] == 4
    model_receives(store, executor.workspace_path, result)
    settled(api)
    store.assert_healthy()


@pytest.mark.parametrize('case', ['failed_sql', 'rejected_sql', 'partial_api'])
def test_capture_summary_counts_business_calls_not_parent_failures(http_captured, monkeypatch, case):
    from saas_bench.tools import ToolResult
    api, store, executor = http_captured
    executor.text_registry = TextRegistry(executor.workspace_path, 'pf', store, sim_day=lambda: 7)
    executor.pf_queries = PFQueries(executor.text_registry)
    shutil.copy(Path(__file__).parents[1] / 'public/novamind-operation', executor.workspace_path)
    if case in ('failed_sql', 'rejected_sql'):
        sql = 'DELETE FROM ledger' if case == 'failed_sql' else 'SELECT * FROM group_insight_snapshots'
        code = f'import novamind_api as n\nn.query({sql!r})\n'
        expected = {'failed' if case == 'failed_sql' else 'rejected': 1}
    else:
        monkeypatch.setattr(api, 'execute_tool', lambda *args, **kwargs: ToolResult(True, 'batch result',
            dict(results=[dict(customer_id=42, success=True), dict(customer_id=43, error='rejected')])) )
        code = "from novamind_api import _client\nprint(_client.call('reject_enterprise_deal', {'deals': []}))\n"
        expected = {'partially_succeeded': 1}
    (executor.workspace_path / 'failure.py').write_text(code)
    output = executor.execute('bash', {'command': './novamind-operation python failure.py'})
    page_text = executor.execute('bash', {'command': re.search(r'captures: (pf more c\d+)', output)[1]})
    artifact(case + '-output.txt', output)
    artifact(case + '-captures.txt', page_text)
    artifact(case + '-page.json', executor.pf_queries.last_answer)
    model_receives(store, executor.workspace_path, page_text)
    page = executor.pf_queries.last_answer
    assert page['call_statuses'] == expected
    assert next(iter(expected)) + '=1' in page_text
    assert sum(page['call_statuses'].values()) == 1
    if case in ('failed_sql', 'rejected_sql'):
        assert page['root']['status'] == 'failed' and page['script_statuses'] == {'failed': 1}
    settled(api)
    store.assert_healthy()


def test_capture_summary_preserves_unknown_calls_without_fabricated_handles(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    parent = store.begin_event('bash', {'command': 'unknown fixture'})
    child = store.begin_event('cli_python', {'script': 'unknown.py'}, parent=parent)
    call = store.begin_event('public_http', {'method': 'POST', 'path': '/call',
        'parsed': {'tool': 'set_capacity_tier', 'args': {'tier': 3}}}, parent=child)
    store.complete(call, 'result_unknown')
    store.complete(child, 'failed', completion='unfinished')
    version = store.version(parent, 'tool_return', 'Partial output', layer='tool_return')
    store.complete(parent, 'failed')
    from types import SimpleNamespace
    output = executor.pf_queries.decorate(SimpleNamespace(event=parent, facts={}, origins=[]), 'Partial output', {})
    artifact('missing-response-footer.txt', output)
    command = re.search(r'captures: (pf more c\d+)', output)
    assert command, output
    page_text = executor.execute('bash', {'command': command[1]})
    artifact('missing-response-captures.txt', page_text)
    artifact('missing-response-page.json', executor.pf_queries.last_answer)
    page = executor.pf_queries.last_answer
    assert page['call_statuses'] == {'result_unknown': 1}
    assert page['script_statuses'] == {'unfinished': 1}
    missing, = [item for item in page['items'] if item.get('missing_response')]
    assert missing['status'] == 'result_unknown' and 'version' not in missing
    assert 'no saved public response' in page_text and 'result_unknown=1' in page_text


def test_killed_public_script_keeps_file_and_unfinished_status(http_captured, monkeypatch):
    api, store, executor = http_captured
    executor.text_registry = TextRegistry(executor.workspace_path, 'pf', store, sim_day=lambda: 7)
    executor.pf_queries = PFQueries(executor.text_registry)
    shutil.copy(Path(__file__).parents[1] / 'public/novamind-operation', executor.workspace_path)
    monkeypatch.setattr(executor, '_bwrap', lambda: None)
    executor.bash_timeout, executor.bash_warn_after = 2, .1
    (executor.workspace_path / 'timeout.py').write_text('''import os, time
from pathlib import Path
Path('side.txt').write_text('saved before timeout')
Path('pid.txt').write_text(str(os.getpid()))
time.sleep(30)
''')
    output = executor.execute('bash', {'command': './novamind-operation python timeout.py'})
    artifact('timeout-output.txt', output)
    assert 'Task terminated' in output
    pid = int((executor.workspace_path / 'pid.txt').read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    bash = next(row for row in records(store) if row['request']['kind'] == 'bash')
    assert bash['result']['process_boundary']['descendants_closed'] is True
    command = re.search(r'captures: (pf more c\d+)', output)[1]
    executor.bash_timeout = 10
    result = executor.execute('bash', {'command': command})
    artifact('timeout-captures.txt', result)
    page = executor.pf_queries.last_answer
    assert page['root']['status'] == 'timed_out'
    assert page['script_statuses'] == {'unfinished': 1}
    assert page['call_statuses'] == {}
    assert 'unfinished=1' in result
    assert {item['path'] for item in page['items'] if item.get('path')} == {'side.txt', 'pid.txt'}
    assert 'saved before timeout' in executor.execute('bash', {'command': 'pf show side.txt@v1 --full'})
    model_receives(store, executor.workspace_path, result)
    settled(api)
    store.assert_healthy()


def test_blank_sql_call_is_counted_as_sql_in_bash_footer(http_captured):
    api, store, executor = http_captured
    executor.text_registry = TextRegistry(executor.workspace_path, 'pf', store, sim_day=lambda: 7)
    executor.pf_queries = PFQueries(executor.text_registry)
    shutil.copy(Path(__file__).parents[1] / 'public/novamind-operation', executor.workspace_path)
    (executor.workspace_path / 'blank.py').write_text('''import novamind_api as n
from novamind_api import _client
try:
    n.query('')
except _client.NovaMindAPIError as exc:
    print('Rejected:', exc)
print(n.vars.current_day)
''')
    output = executor.execute('bash', {'command': './novamind-operation python blank.py'})
    artifact('blank-sql-footer.txt', output)
    assert '1 SQL, 1 API' in output
    model_receives(store, executor.workspace_path, output)
    page_text = executor.execute('bash', {'command': re.search(r'captures: (pf more c\d+)', output)[1]})
    assert executor.pf_queries.last_answer['call_statuses'] == {'rejected': 1, 'succeeded': 1}
    assert 'Partial failure' in page_text
    settled(api)
    store.assert_healthy()


def test_text_unknown_window_is_bounded_single_line_with_complete_raw(workspace, tmp_path):
    _, _, executor = captured(workspace, tmp_path)
    reason = 'Pending date\n' + 'LONG_UNKNOWN_REASON ' * 70 + '\nHidden continuation'
    assert executor.execute('text_create', declaration(applies_at={'unknown': reason})).startswith('Registered')
    output = executor.execute('text_list', {})
    artifact('unknown-window-list.txt', output)
    assert len(output.splitlines()) == 3
    assert len(output.splitlines()[1]) < 300
    assert 'Hidden continuation' not in output
    raw = executor.execute('text_list', {'raw': True})
    assert json.loads(raw)['records'][0]['applies_at'] == {'unknown': reason}
