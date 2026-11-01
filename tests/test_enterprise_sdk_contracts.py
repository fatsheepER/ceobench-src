from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

from numpy.random import default_rng
import pytest

from saas_bench.config import BenchmarkConfig
from saas_bench.database import create_enterprise_thread
from saas_bench.simulation import Simulator
from test_public_sql import server


ROOT = Path(__file__).resolve().parents[1]
SCOPES = {
    'by_group': ('targeted_ops_spend', {'E1': 10}, {'S1': 20}),
    'by_plan': ('targeted_ops_spend_by_plan', {'A': 30}, {'B': 40}),
    'by_group_plan': ('targeted_ops_spend_by_group_plan', {'E1': {'A': 50}}, {'S1': {'B': 60}}),
    'by_customer': ('targeted_ops_spend_by_customer', {'312': 70}, {'88': 80}),
}


@pytest.fixture
def sdk(server, make_agent_tools):
    server.tools = make_agent_tools(server.conn, BenchmarkConfig(), day=0)
    server.start()

    def call(module, name, *args, **kwargs):
        program = '''import importlib,json,sys
from pathlib import Path
module,name,args,kwargs=json.loads(sys.argv[1])
m=importlib.import_module('novamind_api.'+module)
assert Path(m.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve())
try:
    result=getattr(m,name)(*args,**kwargs)
except Exception as exc:
    result={'error_type':type(exc).__name__,'error':str(exc)}
print(json.dumps(result))
'''
        sdk_dir = ROOT / 'public/docs'
        env = dict(os.environ, PYTHONPATH=str(sdk_dir), NOVAMIND_API_PORT=str(server.port))
        env.pop('NOVAMIND_API_SOCKET', None)
        run = subprocess.run([sys.executable, '-c', program,
                              json.dumps([module, name, args, kwargs]), str(sdk_dir)],
                             cwd=server.tools.workspace_path, env=env,
                             capture_output=True, text=True, timeout=15)
        assert run.returncode == 0, run.stderr
        return json.loads(run.stdout)

    server.sdk_call = call
    return server



def assert_enterprise_result(api, result, expected, tool, outcome='succeeded'):
    assert result['results'] == expected
    receipt = result['operation']
    operation = api.conn.execute(
        'SELECT tool,outcome,operation_seq,result_json FROM business_operations WHERE operation_ref=?',
        (receipt['operation_ref'],)).fetchone()
    assert operation is not None
    assert (operation['tool'], operation['outcome']) == (tool, outcome)
    assert receipt['outcome'] == outcome and receipt['operation_seq'] == operation['operation_seq']
    assert json.loads(operation['result_json'])['data']['results'] == expected
    effects = [row[0] for row in api.conn.execute(
        'SELECT DISTINCT effect_seq FROM business_effects WHERE operation_ref=? ORDER BY effect_seq',
        (receipt['operation_ref'],))]
    assert receipt['effect_seqs'] == effects and effects


def ops_state(api):
    return {scope: deepcopy(getattr(api.tools.config, field))
            for scope, (field, _, _) in SCOPES.items()}


@pytest.mark.parametrize('scope', SCOPES)
def test_generated_ops_sdk_scope_replacement_unchanged_and_clear(sdk, scope):
    call = lambda **kwargs: sdk.sdk_call('analytics', 'set_targeted_ops_spend', **kwargs)
    seed = {key: initial for key, (_, initial, _) in SCOPES.items()}
    for key in [scope, *(key for key in seed if key != scope)]:
        initial = seed[key]
        result = call(**{key: initial})
        assert result.get(key) == initial, result
    before = ops_state(sdk)
    replacement = SCOPES[scope][2]
    replaced = call(**{scope: replacement})
    stored = {int(key): value for key, value in replacement.items()} if scope == 'by_customer' else replacement
    expected = dict(before, **{scope: stored})
    assert replaced[scope] == replacement
    assert ops_state(sdk) == expected
    for kwargs in ({}, {scope: None}, {'targeted_spend': None}):
        unchanged = call(**kwargs)
        assert unchanged['changes'] == {}, unchanged
        assert ops_state(sdk) == expected
    cleared = call(**{scope: {}})
    assert cleared[scope] == {}
    assert ops_state(sdk) == dict(expected, **{scope: {}})
    assert cleared['total_extra_per_day'] == sum(
        sum(sum(v.values()) if isinstance(v, dict) else v for v in values.values())
        for values in ops_state(sdk).values())


def test_generated_ops_sdk_legacy_alias_and_atomic_failure(sdk):
    call = lambda *args, **kwargs: sdk.sdk_call('analytics', 'set_targeted_ops_spend', *args, **kwargs)
    assert call({'E1': 123})['by_group'] == {'E1': 123}
    before = ops_state(sdk)
    overrides = [dict(row) for row in sdk.conn.execute('SELECT * FROM config_overrides')]
    for kwargs in (
        {'targeted_spend': {}, 'by_group': {}},
        {'by_group': {'S1': 10}, 'by_plan': {'B': 20},
         'by_group_plan': {'E1': {'A': 30}}, 'by_customer': {'bad-id': 40}},
    ):
        failed = call(**kwargs)
        assert failed['error_type'] == 'NovaMindAPIError', failed
        assert ops_state(sdk) == before
        assert [dict(row) for row in sdk.conn.execute('SELECT * FROM config_overrides')] == overrides
    assert call({})['by_group'] == {}
    assert sdk.tools.config.targeted_ops_spend == {}


def test_generated_dev_sdk_replaces_group_map_and_clears(sdk):
    call = lambda *args, **kwargs: sdk.sdk_call('analytics', 'set_targeted_dev_spend', *args, **kwargs)
    assert call({'E1': 100, 'S1': 200})['targeted_spend'] == {'E1': 100, 'S1': 200}
    assert call({'E2': 300})['targeted_spend'] == {'E2': 300}
    assert sdk.tools.config.targeted_dev_spend == {'E2': 300}
    assert call(by_plan={'A': 1})['error_type'] == 'TypeError'
    assert sdk.tools.config.targeted_dev_spend == {'E2': 300}
    assert call({})['targeted_spend'] == {}
    assert sdk.tools.config.targeted_dev_spend == {}


def customer(api, cid, status='subscribed', thread_type=None):
    api.conn.execute('''INSERT INTO customers
        (customer_id,customer_type,group_id,created_day,steepness_left,steepness_right,
         c_max,usage_demand,quality_sensitivity,price_sensitivity,willingness_to_pay,
         usage_scale,patience,seat_count,email,reply_delay_mean,reply_delay_std)
        VALUES (?,'large','E1',0,1,1,100,1,1,1,1,1,1,20,?,1,0)''',
        (cid, f'customer{cid}@example.com'))
    if status is not None:
        api.conn.execute('''INSERT INTO subscriptions
            (customer_id,plan,listed_price,effective_price,start_day,status,billing_day_mod30,end_day)
            VALUES (?,'A',10,10,0,?,0,?)''', (cid, status, 0 if status == 'cancelled' else None))
    thread = create_enterprise_thread(api.conn, cid, thread_type, 0) if thread_type else None
    api.conn.commit()
    return thread


@pytest.mark.parametrize('thread_type', ['renegotiation', 'renewal', 'churn_prevention', 'plan_change'])
def test_generated_reject_sdk_closes_thread_preserves_subscriber(sdk, thread_type):
    tid, mid = customer(sdk, 312, thread_type=thread_type)
    assert mid != 312
    before = dict(sdk.conn.execute('SELECT * FROM subscriptions WHERE customer_id=312').fetchone())
    wrong = sdk.sdk_call('enterprise', 'reject_enterprise_deal', [{'customer_id': mid}])
    assert wrong['error_type'] == 'NovaMindAPIError'
    rejected = sdk.sdk_call('enterprise', 'reject_enterprise_deal', [{'customer_id': 312}])
    assert_enterprise_result(sdk, rejected, [{'customer_id': 312, 'success': True}], 'reject_enterprise_deal')
    latest = sdk.conn.execute('SELECT * FROM enterprise_turns WHERE thread_id=? ORDER BY message_id DESC', (tid,)).fetchone()
    assert latest['closed'] == 1 and latest['close_reason'] == 'agent_rejected'
    assert dict(sdk.conn.execute('SELECT * FROM subscriptions WHERE customer_id=312').fetchone()) == before
    requoted = sdk.sdk_call('enterprise', 'send_enterprise_deal', [[312, [['A', 9]]]])
    assert_enterprise_result(sdk, requoted, [{'customer_id': 312, 'success': True}], 'send_enterprise_deal')
    new_thread = sdk.conn.execute('SELECT thread_id,thread_type FROM enterprise_turns WHERE customer_id=312 ORDER BY message_id DESC').fetchone()
    assert new_thread['thread_id'] != tid and new_thread['thread_type'] == 'renegotiation'


def test_generated_reject_sdk_loses_pending_lead(sdk):
    tid, mid = customer(sdk, 312, status='lead', thread_type='new_lead')
    assert mid != 312
    result = sdk.sdk_call('enterprise', 'reject_enterprise_deal', [{'customer_id': 312}])
    assert_enterprise_result(sdk, result, [{'customer_id': 312, 'success': True}], 'reject_enterprise_deal')
    assert tuple(sdk.conn.execute('SELECT status,end_day FROM subscriptions WHERE customer_id=312').fetchone()) == ('lost', 0)
    assert sdk.conn.execute('SELECT close_reason FROM enterprise_turns WHERE thread_id=? ORDER BY message_id DESC', (tid,)).fetchone()[0] == 'agent_rejected'


@pytest.mark.parametrize('inactive_status,ended', [('cancelled', True), ('cancelled', False), ('subscribed', True), (None, False)])
def test_generated_quote_sdk_requires_active_subscriber_without_open_thread(sdk, inactive_status, ended):
    customer(sdk, 312)
    customer(sdk, 88, status=inactive_status)
    sdk.conn.execute('UPDATE subscriptions SET end_day=? WHERE customer_id=88', (0 if ended else None,))
    sdk.conn.commit()
    sent = sdk.sdk_call('enterprise', 'send_enterprise_deal', [[312, [['A', 9]]]])
    assert_enterprise_result(sdk, sent, [{'customer_id': 312, 'success': True}], 'send_enterprise_deal')
    latest = sdk.conn.execute('SELECT * FROM enterprise_turns WHERE customer_id=312 ORDER BY message_id DESC').fetchone()
    assert latest['thread_type'] == 'renegotiation' and latest['sender'] == 'agent'
    assert json.loads(latest['offer_json']) == [{'plan': 'A', 'price_per_seat': 9, 'contract_months': 1}]
    failed = sdk.sdk_call('enterprise', 'send_enterprise_deal', [[88, [['A', 9]]]])
    assert failed['error_type'] == 'NovaMindAPIError' and 'no active subscription' in failed['error']
    assert sdk.conn.execute('SELECT count(*) FROM enterprise_turns WHERE customer_id=88').fetchone()[0] == 0


def test_generated_quote_sdk_replies_using_customer_id(sdk):
    tid, mid = customer(sdk, 312, status='lead', thread_type='new_lead')
    assert mid != 312
    wrong = sdk.sdk_call('enterprise', 'send_enterprise_deal', [[mid, [['A', 9]]]])
    assert wrong['error_type'] == 'NovaMindAPIError'
    sent = sdk.sdk_call('enterprise', 'send_enterprise_deal', [[312, [['A', 9], ['B', 12]]]])
    assert_enterprise_result(sdk, sent, [{'customer_id': 312, 'success': True}], 'send_enterprise_deal')
    latest = sdk.conn.execute('SELECT * FROM enterprise_turns WHERE thread_id=? ORDER BY message_id DESC', (tid,)).fetchone()
    assert latest['sender'] == 'agent' and latest['turn_number'] == 1
    assert all(offer['contract_months'] == 1 for offer in json.loads(latest['offer_json']))


@pytest.mark.parametrize('tool', ['send_enterprise_deal', 'reject_enterprise_deal'])
def test_generated_enterprise_sdk_returns_success_and_error_in_mixed_batch(sdk, tool):
    tid, _ = customer(sdk, 312, thread_type='churn_prevention')
    customer(sdk, 88, status='cancelled')
    if tool == 'send_enterprise_deal':
        deals = [[312, [['A', 9]]], [88, [['A', 9]]]]
        error = 'no subscription'
    else:
        deals = [{'customer_id': 312}, {'customer_id': 88}]
        error = 'no active thread'
    result = sdk.sdk_call('enterprise', tool, deals)
    assert_enterprise_result(sdk, result, [
        {'customer_id': 312, 'success': True}, {'customer_id': 88, 'error': error}], tool, 'partial')
    latest = sdk.conn.execute('SELECT * FROM enterprise_turns WHERE thread_id=? ORDER BY message_id DESC', (tid,)).fetchone()
    assert latest['sender'] == 'agent' and latest['turn_number'] == 1
    assert latest['closed'] == int(tool == 'reject_enterprise_deal')
    assert sdk.conn.execute('SELECT status FROM subscriptions WHERE customer_id=312').fetchone()[0] == 'subscribed'
    assert sdk.conn.execute('SELECT status FROM subscriptions WHERE customer_id=88').fetchone()[0] == 'cancelled'
    assert sdk.conn.execute('SELECT count(*) FROM enterprise_turns WHERE customer_id=88').fetchone()[0] == 0


def test_simulator_customer_response_timeout_day6_day7(sdk):
    for cid, status, kind in [(312, 'lead', 'new_lead'), (88, 'subscribed', 'churn_prevention')]:
        customer(sdk, cid, status=status, thread_type=kind)
    sim = Simulator(sdk.conn, sdk.tools.config, default_rng(123))
    sim._cached_q_shared_per_plan = {'A': .5}
    sim._group_sub_counts = {'E1': 1}
    for day in (6, 7):
        sim.current_day = day
        sim._process_agent_response_timeouts({}, [])
        states = {row['customer_id']: (row['status'], row['end_day'])
                  for row in sdk.conn.execute('SELECT customer_id,status,end_day FROM subscriptions')}
        if day == 6:
            assert states == {312: ('lead', None), 88: ('subscribed', None)}
            assert all(row[0] is None for row in sdk.conn.execute('SELECT _internal_status FROM enterprise_turns'))
        else:
            assert states == {312: ('lost', 7), 88: ('cancelled', 7)}
            assert {row[0] for row in sdk.conn.execute('SELECT _internal_status FROM enterprise_turns')} == {'timeout'}


def test_generated_enterprise_docs_match_http_contracts(tmp_path):
    docs = {doc['name']: doc for doc in json.loads((ROOT / 'public/docs/api/enterprise.json').read_text())}
    quote = docs['send_enterprise_deal']
    assert '7 days' in quote['description']
    assert 'active enterprise subscription' in quote['description']
    assert 'message_id' in quote['description'] and 'customer_id' in quote['description']
    assert 'status (str)' not in quote['output_schema']['results']
    assert 'NovaMindAPIError' in quote['output_schema']['_note']
    assert 'at least one' in quote['output_schema']['_note']
    for deal in quote['example_call']['arguments']['deals']:
        assert all(len(offer) == 2 for offer in deal[1])
    reject = docs['reject_enterprise_deal']
    assert 'subscription unchanged' in reject['impact']
    assert 'thread_type (str)' not in reject['output_schema']['results']
    assert 'NovaMindAPIError' in reject['output_schema']['_note']
    assert 'at least one' in reject['output_schema']['_note']
    table = json.loads((ROOT / 'public/docs/tables/enterprise_turns.json').read_text())
    assert 'send_enterprise_deal/reject_enterprise_deal' not in table['columns']['message_id']
    assert 'customer_id' in table['columns']['customer_id']
    from scripts.generate_public_docs import render_simulator_instructions, render_tables_reference
    for name, render in [('simulator-instructions.md', render_simulator_instructions),
                         ('tables-reference.md', render_tables_reference)]:
        generated = tmp_path / name
        render(generated)
        assert generated.read_text() == (ROOT / 'public/docs' / name).read_text()


def test_generated_targeted_docs_explain_replacement():
    docs = {doc['name']: doc for doc in json.loads((ROOT / 'public/docs/api/analytics.json').read_text())}
    ops = docs['set_targeted_ops_spend']
    assert 'replaces' in ops['parameters']['by_group']
    dev = docs['set_targeted_dev_spend']
    assert 'replaces' in dev['parameters']['targeted_spend']['description']
    assert '{}' in dev['parameters']['targeted_spend']['description']
    assert set(dev['inputSchema']['properties']) == {'targeted_spend'}
