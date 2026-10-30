from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import sqlite3
import threading
import time

import pytest

from saas_bench.run_lifecycle import RunCancelled
from test_preflight_team_flow import ADVANCE, calls, final, make_team, raw


def query(team, sql, role='ops_finance'):
    status, result = raw(team, role, '/query', {'sql': sql})
    assert status == 200 and result['success'], result
    assert not result.get('truncated'), result
    return result['rows']


def records(team):
    return query(team, 'SELECT * FROM business_operations ORDER BY operation_seq')


def effects(team):
    return query(team, 'SELECT * FROM business_effects ORDER BY effect_seq,item_index')


def named_values(value, name):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == name:
                yield item
            yield from named_values(item, name)
    elif isinstance(value, list):
        for item in value:
            yield from named_values(item, name)


def wait_until(predicate):
    deadline = time.monotonic() + 10
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.01)
    assert predicate()


def sample(name, value):
    if destination := os.environ.get('CEOBENCH_R225_SAMPLES'):
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_effect_order_follows_serialized_writes_not_request_entry(mode, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    with make_team(mode, lambda *args: final('done'), 'ledger-order') as team:
        team._set_phase('analysis')
        execute = team.server.execute_tool

        def delayed(name, args, **kwargs):
            if kwargs.get('role') == 'growth' and name == 'set_daily_spend':
                entered.set()
                assert release.wait(10)
            return execute(name, args, **kwargs)

        monkeypatch.setattr(team.server, 'execute_tool', delayed)
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(raw, team, 'growth', '/call',
                    {'tool': 'set_daily_spend', 'args': {'operations': 111}, 'actor_id': 'ceo'})
                assert entered.wait(10)
                status, result = raw(team, 'ops_finance', '/call',
                    {'tool': 'set_daily_spend', 'args': {'operations': 222}, 'actor_id': 'ceo'})
                assert status == 200 and result['success'], result
                release.set()
                status, result = first.result(timeout=10)
                assert status == 200 and result['success'], result
        finally:
            release.set()
        operations, changed = records(team), effects(team)
        assert [r['actor'] for r in operations] == ['ops_finance', 'growth']
        assert all(r['initiator'] == 'model' and r['outcome'] == 'succeeded' for r in operations)
        refs = [r['operation_ref'] for r in operations]
        first_effects = [r for r in changed if r['operation_ref'] == refs[0]]
        last_effects = [r for r in changed if r['operation_ref'] == refs[1]]
        assert first_effects and last_effects
        assert max(r['effect_seq'] for r in first_effects) < min(r['effect_seq'] for r in last_effects)
        assert 222 in list(named_values([json.loads(r['after_json']) for r in first_effects], 'spend_operations'))
        assert 222 in list(named_values([json.loads(r['before_json']) for r in last_effects], 'spend_operations'))
        assert 111 in list(named_values([json.loads(r['after_json']) for r in last_effects], 'spend_operations'))
        assert query(team, 'SELECT spend_operations FROM config_history ORDER BY day DESC LIMIT 1')[0]['spend_operations'] == 111
        sample('effect-order-' + mode, {'operations': operations, 'effects': changed})


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_permission_refusal_and_business_rejection_do_not_create_effects(mode):
    with make_team(mode, lambda *args: final('done'), 'ledger-rejections') as team:
        team._set_phase('analysis')
        assert raw(team, 'growth', '/call', {'tool': 'set_prices', 'args': {'A': 99}})[0] == 403
        assert records(team) == [] and effects(team) == []
        status, response = raw(team, 'growth', '/call', {'tool': 'set_capacity_tier', 'args': {'tier': 999}})
        assert status == 200 and response['success'] is False
        rejected = records(team)
        assert len(rejected) == 1 and rejected[0]['outcome'] == 'rejected'
        assert effects(team) == []
        status, response = raw(team, 'growth', '/call', {'tool': 'set_capacity_tier', 'args': {'tier': 2}})
        assert status == 200 and response['success'] is True
        assert [r['outcome'] for r in records(team)] == ['rejected', 'succeeded']
        assert not team.server._operation_failed


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_script_source_survives_client_exit_and_late_dispatch(mode, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    with make_team(mode, lambda *args: final('done'), 'ledger-script-late') as team:
        team._set_phase('analysis')
        execute = team.server.execute_tool

        def delayed(name, args, **kwargs):
            if name == 'set_capacity_tier' and args.get('tier') == 3:
                entered.set()
                assert release.wait(10)
            return execute(name, args, **kwargs)

        monkeypatch.setattr(team.server, 'execute_tool', delayed)
        program = '''import http.client,json,os,socket
c=http.client.HTTPConnection('localhost')
c.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
c.sock.connect(os.environ['NOVAMIND_API_SOCKET'])
c.request('POST','/call',json.dumps({'tool':'set_capacity_tier','args':{'tier':3},'role':'ceo','initiator':'model'}),{'Content-Type':'application/json','X-Initiator':'model'})
c.close()
print('SCRIPT-CLIENT-EXITED')
'''
        assert raw(team, 'growth', '/daily-scripts', {'name': 'late.py', 'content': program})[0] == 200
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                running = pool.submit(team.server._run_daily_scripts_internal, 'growth')
                assert entered.wait(10)
                output = running.result(timeout=10)
                assert 'SCRIPT-CLIENT-EXITED' in output['late.py']
                assert team.server.role_executors['growth'].registered_execution is False
                release.set()
        finally:
            release.set()
        wait_until(lambda: not team.server.public_request_state()['active'])
        assert raw(team, 'growth', '/call', {'tool': 'set_capacity_tier', 'args': {'tier': 4}, 'initiator': 'registered_script'})[0] == 200
        rows = records(team)
        assert [(r['actor'], r['initiator'], r['outcome']) for r in rows] == [
            ('growth', 'registered_script', 'succeeded'), ('growth', 'model', 'succeeded')]
        assert query(team, 'SELECT capacity_tier FROM config_history ORDER BY day DESC LIMIT 1')[0]['capacity_tier'] == 4
        sample('script-late-' + mode, {'operations': rows, 'effects': effects(team), 'script_output': str(output['late.py'])})


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_committed_research_survives_dirty_failure_without_finalizer_commit(mode, monkeypatch):
    with make_team(mode, lambda *args: final('done'), 'ledger-dirty-research') as team:
        team._set_phase('ceo')
        tools = team.server.tools
        tools.config.market_research_discover_prob = 1
        before = {r['group_id']: r['info_level'] for r in team.server.conn.execute('SELECT group_id,info_level FROM group_info_levels')}

        def dirty_failure(*args, **kwargs):
            tools.conn.execute("INSERT INTO notifications(day,type,message) VALUES(0,'event_alert','PENDING-ONLY')")
            raise RuntimeError('PRIVATE-LEDGER-FAULT-CANARY')

        monkeypatch.setattr(tools, 'get_group_insights', dirty_failure)
        status, response = raw(team, 'ceo', '/call', {'tool': 'research_market', 'args': {}})
        assert status == 500 and response['success'] is False
        wait_until(lambda: not team.server.public_request_state()['active'])
        assert team.server._operation_failed and team.server.conn.in_transaction
        assert team.server.conn.execute("SELECT COUNT(*) FROM notifications WHERE message='PENDING-ONLY'").fetchone()[0] == 1
        with sqlite3.connect(team.test_root / 'world.db') as durable:
            durable.row_factory = sqlite3.Row
            assert durable.execute("SELECT COUNT(*) FROM notifications WHERE message='PENDING-ONLY'").fetchone()[0] == 0
            charge = durable.execute("SELECT amount FROM ledger WHERE category='market_research'").fetchall()
            assert [r[0] for r in charge] == [-tools.config.discovery_cost_level_1]
            discovered = [r['group_id'] for r in durable.execute('SELECT group_id,info_level FROM group_info_levels') if r['info_level'] > before[r['group_id']]]
            assert len(discovered) == 1
            operations = [dict(r) for r in durable.execute('SELECT * FROM business_operations ORDER BY operation_seq')]
            committed = [dict(r) for r in durable.execute('SELECT * FROM business_effects ORDER BY effect_seq,item_index')]
        assert len(operations) == 1 and operations[0]['outcome'] in ('unfinished', 'unknown')
        assert committed
        assert -tools.config.discovery_cost_level_1 in list(named_values([json.loads(r['after_json']) for r in committed], 'amount'))
        assert discovered[0] in json.dumps(committed)
        assert 'PENDING-ONLY' not in json.dumps(committed)
        assert 'PRIVATE-LEDGER-FAULT-CANARY' not in json.dumps(operations)
        with pytest.raises(RunCancelled):
            team._check_model_send()
        with pytest.raises((RuntimeError, RunCancelled)):
            team.checkpoint()
        sample('dirty-research-' + mode, {'operations': operations, 'effects': committed, 'discovered': discovered, 'http': response})


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_public_ledger_query_is_delivered_to_requesting_model_only(mode):
    count = {'ceo': 0, 'growth': 0, 'ops_finance': 0}
    marker = 'LEDGER-WITNESS-ONLY-GROWTH'
    code = "import json; from novamind_api import marketing,query; marketing.set_daily_spend(operations=321); print('" + marker + "',json.dumps(query('SELECT actor,initiator,tool,outcome,operation_ref FROM business_operations ORDER BY operation_seq')['rows']))"

    def respond(team, role, body):
        count[role] += 1
        if role == 'growth' and count[role] == 1:
            return calls(('bash', {'command': './novamind-operation python-c ' + shlex.quote(code)}))
        if role == 'ceo':
            return calls(('bash', {'command': ADVANCE}))
        return final('No further work.')

    with make_team(mode, respond, 'ledger-model-input', total_days=7) as team:
        outcome = team.run(stop_after_day=7)
        assert outcome.reason == "natural_end", (outcome, team.failure)
        rows = records(team)
        operation = next(r for r in rows if r['tool'] == 'set_daily_spend')
        assert operation['actor'] == 'growth' and operation['initiator'] == 'model'
        observed = {}
        for role in count:
            inputs = [r['body'] for r in team.test_wire if r['role'] == role]
            observed[role] = inputs
            tool_text = '\n'.join(str(m.get('content', '')) for body in inputs for m in body['messages'] if m['role'] == 'tool')
            if role == 'growth':
                assert marker in tool_text and operation['operation_ref'] in tool_text
                assert 'set_daily_spend' in tool_text and 'succeeded' in tool_text
            else:
                assert marker not in tool_text
        usage = [json.loads(line) for line in (team.root / 'private/growth/model-usage.jsonl').read_text().splitlines()]
        assert any(r.get('event') == 'http_request' and marker in json.dumps(r.get('body', {})) for r in usage)
        sample('model-input-' + mode, {'operations': rows, 'provider': observed, 'usage': usage})


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_all_public_mutators_and_composite_paging(mode):
    from test_responsibility_authority import ACTIONS, READS, prepare, assert_business, assert_role_business
    with make_team(mode, lambda *args: final('done'), 'ledger-mutators') as team:
        prepare(team)
        team._set_phase('ceo')
        initial_price = query(team, 'SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1')[0]['price_A']
        program = '''import importlib,json
from novamind_api import query
for name,(args,module,roles) in ACTIONS.items():
    result=getattr(importlib.import_module('novamind_api.'+module),name)(**args)
    print(name,json.dumps(result))
print('ALL-MUTATORS-COMPLETE')
'''.replace('ACTIONS', repr(ACTIONS))
        output = team.roles['ceo'].executor.execute('bash', {'command': './novamind-operation python-c ' + shlex.quote(program)})
        assert 'ALL-MUTATORS-COMPLETE' in output and '[exit code:' not in output, output
        assert 'op:' not in output
        assert_business(team)
        assert_role_business(team, 'ceo', initial_price)
        rows, changed = records(team), effects(team)
        expected = set(ACTIONS) - READS
        assert len(rows) == len(expected) == 17
        assert {r['tool'] for r in rows} == expected
        assert all((r['actor'], r['initiator'], r['outcome']) == ('ceo', 'model', 'succeeded') for r in rows)
        assert set(r['operation_ref'] for r in changed) == set(r['operation_ref'] for r in rows)
        for r in rows:
            actual = json.loads(r['parameters_json'])
            expected_args = ACTIONS[r['tool']][0]
            assert {key: actual[key] for key in expected_args} == expected_args
            assert all(value is None for key, value in actual.items() if key not in expected_args)
            result = json.loads(r['result_json'])
            assert result['success'] is True
        serialized = json.dumps(rows + changed)
        for private in ('relationship', 'next_reply_day', 'current_offer_price', '_internal_status', 'insight_snapshot'):
            assert private not in serialized
        by_tool = {r['tool']: r for r in rows}
        for name in ('set_prices', 'set_capacity_tier', 'set_promotion', 'set_targeted_ops_spend'):
            status, response = raw(team, 'ceo', '/call', {'tool': name, 'args': ACTIONS[name][0]})
            assert status == 200 and response['success'], response
            last = records(team)[-1]
            assert last['operation_ref'] != by_tool[name]['operation_ref']
            repeated = [e for e in effects(team) if e['operation_ref'] == last['operation_ref']]
            assert repeated and all(e['before_json'] == e['after_json'] for e in repeated)
        status, result = raw(team, 'ceo', '/call', {'tool': 'send_enterprise_deal', 'args': {'deals': [[900001, [['A', 7]]], [999999, [['A', 8]]]]}})
        assert status == 200
        partial = records(team)[-1]
        assert partial['outcome'] == 'partial'
        item_results = json.loads(partial['result_json'])['data']['results']
        assert len(item_results) == 2 and any(item.get('error') for item in item_results)
        high = query(team, 'SELECT MAX(effect_seq) AS n FROM business_effects')[0]['n']
        cursor, paged = (0, -1), []
        while True:
            page = query(team, 'SELECT * FROM business_effects WHERE effect_seq <= %d AND (effect_seq > %d OR (effect_seq = %d AND item_index > %d)) ORDER BY effect_seq,item_index LIMIT 1' % (high, cursor[0], cursor[0], cursor[1]))
            if not page:
                break
            paged += page
            cursor = page[0]['effect_seq'], page[0]['item_index']
        assert paged == effects(team)
        assert len({r['effect_ref'] for r in paged}) == len(paged)
        assert any(r['item_index'] > 0 for r in paged)
        sample('all-mutators-' + mode, {'operations': records(team), 'effects': paged, 'default_receipts': str(output)})
