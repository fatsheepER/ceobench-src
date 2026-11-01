import json
import shlex
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from test_business_ledger_public import effects, records, wait_until
from test_preflight_team_flow import ADVANCE, PREDICTIONS, calls, final, make_team, raw


def assert_operation(team, receipt, outcome):
    ref = receipt['operation_ref']
    operation = next(row for row in records(team) if row['operation_ref'] == ref)
    assert receipt['operation_seq'] == operation['operation_seq']
    assert receipt['outcome'] == operation['outcome'] == outcome
    assert receipt['effect_seqs'] == sorted({row['effect_seq'] for row in effects(team)
                                           if row['operation_ref'] == ref})
    assert 'operation' not in json.loads(operation['result_json'])
    assert 'operation' not in (json.loads(operation['result_json']).get('data') or {})
    return operation


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_public_receipts_distinguish_same_value_rejection_and_partial(mode):
    from test_responsibility_authority import prepare
    with make_team(mode, lambda *args: final('done'), 'relation-operation-http') as team:
        prepare(team)
        team._set_phase('analysis')
        status, denied = raw(team, 'growth', '/call', {'tool': 'set_prices', 'args': {'A': 99}})
        assert status == 403 and 'operation' not in denied
        assert records(team) == []
        refs = []
        for tier, outcome in ((3, 'succeeded'), (3, 'succeeded'), (99, 'rejected')):
            status, response = raw(team, 'growth', '/call', {'tool': 'set_capacity_tier', 'args': {'tier': tier}})
            assert status == 200
            operation = assert_operation(team, response['operation'], outcome)
            refs.append(operation['operation_ref'])
            if len(refs) == 2:
                assert response['data']['changes'] == {}
        assert len(set(refs)) == 3
        team._set_phase('ceo')
        status, response = raw(team, 'ceo', '/call', {'tool': 'send_enterprise_deal',
            'args': {'deals': [[900001, [['A', 7]]], [999999, [['A', 8]]]]}})
        assert status == 200 and response['success']
        assert_operation(team, response['operation'], 'partial')
        assert any(item.get('error') for item in response['data']['results'])
        from saas_bench.business_ledger import read_operation
        bound = read_operation(team.server.conn, response['operation']['operation_ref'])
        assert bound['operation']['operation_ref'] == response['operation']['operation_ref']
        assert bound['effects'] == [row for row in effects(team)
                                    if row['operation_ref'] == response['operation']['operation_ref']]
        assert read_operation(team.server.conn, "op:missing' OR 1=1 --") is None


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_sdk_operation_receipts_enter_actual_model_request_without_ledger_query(mode):
    count = {'ceo': 0, 'growth': 0, 'ops_finance': 0}
    program = '''import json
from novamind_api import infrastructure, _client
for tier in (3, 3, 99):
    try:
        print('SDK-RESULT', json.dumps(infrastructure.set_capacity_tier(tier)))
    except _client.NovaMindAPIError as exc:
        print('SDK-REJECTED', str(exc))
        print('SDK-REJECTED-RECORD', json.dumps(exc.operation))
'''

    def respond(team, role, body):
        count[role] += 1
        if role == 'growth' and count[role] == 1:
            return calls(('bash', {'command': './novamind-operation python-c ' + shlex.quote(program)}))
        if role == 'ceo':
            return calls(('bash', {'command': ADVANCE}))
        return final('No further action.')

    with make_team(mode, respond, 'relation-operation-model', total_days=7) as team:
        assert team.run(stop_after_day=7).reason == 'natural_end', team.failure
        operations = [row for row in records(team) if row['tool'] == 'set_capacity_tier']
        assert [row['outcome'] for row in operations] == ['succeeded', 'succeeded', 'rejected']
        requests = [item['body'] for item in team.test_wire if item['role'] == 'growth']
        text = '\n'.join(str(message.get('content', '')) for body in requests
                         for message in body['messages'] if message['role'] == 'tool')
        assert 'SDK-REJECTED' in text and '"changes": {}' in text
        for operation in operations:
            assert operation['operation_ref'] in text
        usage = [json.loads(line) for line in (team.root / 'private/growth/model-usage.jsonl').read_text().splitlines()]
        assert any(item.get('event') == 'http_request' and all(operation['operation_ref'] in json.dumps(item.get('body', {}))
                   for operation in operations) for item in usage)
        assert not team.server._operation_failed


@pytest.mark.parametrize('mode', ('git', 'pf'))
@pytest.mark.parametrize('gateway', ('cli', 'sdk'))
def test_week_advance_public_receipt_keeps_canonical_terminal_result(mode, gateway):
    with make_team(mode, lambda *args: final('done'), 'relation-operation-advance') as team:
        team._set_phase('ceo')
        command = ADVANCE if gateway == 'cli' else './novamind-operation python-c ' + shlex.quote(
            'import json; from novamind_api._client import next_week; print(json.dumps(next_week(' +
            repr(PREDICTIONS) + ", 'receipt test')))"
        )
        output = team.roles['ceo'].executor.execute('bash', {'command': command})
        operation = next(row for row in records(team) if row['tool'] == 'next_week')
        assert operation['operation_ref'] in output and 'succeeded' in output, output
        assert set(json.loads(operation['result_json'])) == {'success', 'day', 'advance'}
        assert team.server.tools.current_day == 7


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_rejected_prediction_receipt_is_discoverable_and_retry_continues(mode):
    with make_team(mode, lambda *args: final('done'), 'relation-operation-rejected-advance') as team:
        team._set_phase('ceo')
        conn = team.server.conn
        conn.execute("CREATE TEMP TRIGGER reject_prediction_receipt BEFORE INSERT ON business_effects WHEN json_extract(NEW.target_json,'$.table')='predictions' BEGIN SELECT RAISE(ABORT,'prediction effect fault'); END")
        status, response = raw(team, 'ceo', '/next-week', {'predictions': PREDICTIONS, 'rationale': 'fault test'})
        assert status == 200 and response['error'] == 'prediction_save_failed'
        assert_operation(team, response['operation'], 'rejected')
        conn.execute('DROP TRIGGER reject_prediction_receipt')
        status, response = raw(team, 'ceo', '/next-week', {'predictions': PREDICTIONS, 'rationale': 'retry after rollback'})
        assert status == 200 and response['success']
        assert_operation(team, response['operation'], 'succeeded')
        assert not team.server._operation_failed


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_timeout_receipt_does_not_wait_for_late_operation_or_claim_terminal_state(mode):
    entered, release = threading.Event(), threading.Event()
    with make_team(mode, lambda *args: final('done'), 'relation-operation-late') as team:
        team._set_phase('ceo')
        team.server.STEP_WEEK_TIMEOUT = .05
        original = team.server.simulator.step_week

        def step():
            entered.set()
            assert release.wait(10)
            return original()

        team.server.simulator.step_week = step
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(raw, team, 'ceo', '/next-week', {'predictions': PREDICTIONS, 'rationale': 'late receipt'})
            try:
                assert entered.wait(5)
                status, response = pending.result(timeout=5)
                assert status == 200 and response['error'] == 'step_week_timeout'
                receipt = response['operation']
                assert receipt['outcome'] == 'unfinished' and receipt['effect_seqs'] is None
            finally:
                release.set()
        wait_until(lambda: team.server._advance_outcome is not None)
        operation = dict(team.server.conn.execute('SELECT * FROM business_operations WHERE operation_ref=?',
                         (receipt['operation_ref'],)).fetchone())
        assert operation['outcome'] == 'succeeded'
        assert receipt['outcome'] == 'unfinished'


def test_operation_metadata_does_not_change_duplicate_finalization():
    from saas_bench.business_ledger import BusinessOperation
    from saas_bench.database import init_database
    from saas_bench.tools import ToolResult
    conn = init_database(':memory:')
    try:
        operation = BusinessOperation(conn, actor='ceo', initiator='model', tool='set_capacity_tier', day=0, parameters={})
        result = ToolResult(True, 'unchanged', {'changes': {}})
        operation.finish(result)
        result.operation = operation.receipt()
        before = conn.serialize()
        operation.finish(result)
        assert conn.serialize() == before
        assert result.to_json()['operation'] == operation.receipt()
        result.data = {'changes': {'capacity_tier': {'before': 1, 'after': 2}}}
        with pytest.raises(RuntimeError, match='Conflicting business operation finalization'):
            operation.finish(result)
        assert conn.serialize() == before
    finally:
        conn.close()
