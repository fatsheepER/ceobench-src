import http.client
import json
import socket
import sqlite3
import threading

from numpy.random import default_rng
import pytest

from saas_bench.business_ledger import CommitConnection
from saas_bench.run_lifecycle import RunCancelled
from saas_bench.simulation import Simulator
from test_preflight_team_flow import ADVANCE, PREDICTIONS, final, make_team, raw
from test_business_ledger_public import effects, query, records, sample, wait_until


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_real_simulator_commits_predictions_and_day_progress(mode):
    with make_team(mode, lambda *args: final('done'), 'ledger-real-simulator') as team:
        team._set_phase('ceo')
        sim = Simulator(team.server.conn, team.server.tools.config, default_rng(123))
        sim.initialize(resume=True)
        sim.restore_rng_states()
        team.server.simulator = sim
        output = team.roles['ceo'].executor.execute('bash', {'command': ADVANCE})
        assert 'Day 7' in output and '[exit code:' not in output, output
        rows, changed = records(team), effects(team)
        assert len(rows) == 1 and rows[0]['tool'] == 'next_week' and rows[0]['outcome'] == 'succeeded'
        assert set(json.loads(rows[0]['result_json'])) == {'success', 'day', 'advance'}
        predictions = [r for r in changed if json.loads(r['target_json']).get('table') == 'predictions']
        progress = [r for r in changed if r['kind'] == 'advance_progress']
        assert len(predictions) == 4 and len({r['effect_seq'] for r in predictions}) == 1
        assert progress and max(r['effect_seq'] for r in predictions) < min(r['effect_seq'] for r in progress)
        assert {r['day'] for r in progress} == set(range(1, 8))
        assert json.loads(progress[-1]['after_json']) == {'reached_day': 7, 'last_completed_day': 7}
        assert sim.conn is team.server.conn and team.server.tools.conn is team.server.conn
        assert query(team, 'SELECT MAX(day) AS day FROM ledger')[0]['day'] == 7
        wait_until(lambda: not team.server.public_request_state()['active'])
        sample('real-simulator-' + mode, {'operations': rows, 'effects': changed, 'receipt': str(output)})


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_prediction_ledger_failure_rolls_back_and_next_public_call_continues(mode):
    with make_team(mode, lambda *args: final('done'), 'ledger-prediction-fault') as team:
        team._set_phase('ceo')
        conn = team.server.conn
        conn.execute("CREATE TEMP TRIGGER deny_prediction_effect BEFORE INSERT ON business_effects WHEN json_extract(NEW.target_json,'$.table')='predictions' BEGIN SELECT RAISE(ABORT,'prediction effect fault'); END")
        status, result = raw(team, 'ceo', '/next-week', {'predictions': PREDICTIONS, 'rationale': 'fault test'})
        assert status == 200 and result['error'] == 'prediction_save_failed', result
        assert team.server.tools.current_day == 0
        assert not conn.in_transaction and not team.server._operation_failed
        assert query(team, 'SELECT COUNT(*) AS n FROM predictions')[0]['n'] == 0
        assert effects(team) == [] and records(team)[0]['outcome'] == 'rejected'
        conn.execute('DROP TRIGGER deny_prediction_effect')
        status, result = raw(team, 'ceo', '/next-week', {'predictions': PREDICTIONS, 'rationale': 'retry after rollback'})
        assert status == 200 and result['success'] and result['day'] == 7, result
        assert [r['outcome'] for r in records(team)] == ['rejected', 'succeeded']


@pytest.mark.parametrize('mode', ('git', 'pf'))
@pytest.mark.parametrize('after_commit', (False, True))
def test_public_runtime_commit_fault_keeps_sql_memory_and_hold_consistent(mode, after_commit, monkeypatch):
    with make_team(mode, lambda *args: final('done'), 'ledger-runtime-fault') as team:
        team._set_phase('analysis')
        conn, tools = team.server.conn, team.server.tools
        old = tools.config.promotion_global
        original = CommitConnection._commit
        def fail(adapter):
            if adapter.operation.tool == 'set_promotion':
                if after_commit:
                    original(adapter)
                raise RuntimeError('PRIVATE-COMMIT-CANARY')
            return original(adapter)
        monkeypatch.setattr(CommitConnection, '_commit', fail)
        status, response = raw(team, 'growth', '/call', {'tool': 'set_promotion', 'args': {'global_promotion': .4}})
        assert status == 500 and response['success'] is False
        wait_until(lambda: not team.server.public_request_state()['active'])
        assert team.server._operation_failed and not conn.in_transaction
        assert tools.config.promotion_global == (.4 if after_commit else old)
        assert conn.execute('SELECT COUNT(*) FROM config_overrides').fetchone()[0] == int(after_commit)
        assert conn.execute('SELECT COUNT(*) FROM business_effects').fetchone()[0] == int(after_commit)
        row = dict(conn.execute('SELECT * FROM business_operations').fetchone())
        assert row['outcome'] == 'unknown' and 'PRIVATE-COMMIT-CANARY' not in row['result_json']
        status, blocked = raw(team, 'growth', '/query', {'sql': 'SELECT 1 AS continued'})
        assert status == 503 and not blocked['success']
        with pytest.raises(RunCancelled):
            team._check_model_send()
        with pytest.raises((RuntimeError, RunCancelled)):
            team.checkpoint()
        sample('runtime-fault-' + mode + '-' + str(after_commit), {'operation': row, 'committed': after_commit})


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_client_exit_preserves_original_advance_until_late_commits_settle(mode):
    started, release = threading.Event(), threading.Event()
    with make_team(mode, lambda *args: final('done'), 'ledger-late-advance') as team:
        from types import SimpleNamespace
        team._set_phase('ceo')
        sim = SimpleNamespace(conn=team.server.conn, current_day=0)
        def step():
            sim.current_day = 1
            sim.conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES(1,'compute',-1,'first committed')")
            sim.conn.commit()
            started.set()
            assert release.wait(10)
            sim.current_day = 7
            sim.conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES(7,'compute',-2,'late committed')")
            sim.conn.commit()
            return SimpleNamespace(day=7)
        sim.step_week = step
        team.server.simulator = sim
        connection = http.client.HTTPConnection('localhost')
        connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        from saas_bench.runtime_temp import connect_unix
        connect_unix(connection.sock, team.server.role_sockets['ceo'])
        try:
            connection.request('POST', '/next-week', json.dumps({'predictions': PREDICTIONS, 'rationale': 'late commit test'}), {'Content-Type': 'application/json'})
            assert started.wait(10)
            connection.close()
            with sqlite3.connect(team.test_root / 'world.db') as durable:
                admitted = durable.execute('SELECT operation_ref,outcome FROM business_operations').fetchall()
                assert len(admitted) == 1 and admitted[0][1] == 'unfinished'
                assert durable.execute("SELECT COUNT(*) FROM business_effects WHERE kind='advance_progress'").fetchone()[0] == 1
        finally:
            release.set()
            connection.close()
        wait_until(lambda: team.server._advance_outcome is not None and not team.server.public_request_state()['active'])
        row = dict(team.server.conn.execute('SELECT * FROM business_operations').fetchone())
        changed = [dict(r) for r in team.server.conn.execute('SELECT * FROM business_effects ORDER BY effect_seq,item_index')]
        assert row['operation_ref'] == admitted[0][0] and row['outcome'] == 'succeeded'
        assert row['actor'] == 'ceo' and row['initiator'] == 'model'
        assert [r['day'] for r in changed if r['kind'] == 'advance_progress'] == [1, 7]
        assert sim.conn is team.server.conn and team.server.tools.conn is team.server.conn
        assert team.server._advance_outcome['status'] == 'completed'
        sample('late-advance-' + mode, {'operation': row, 'effects': changed, 'advance': team.server._advance_outcome})


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_customer_model_cost_commit_records_its_business_segment(mode):
    import httpx
    from openai import OpenAI
    from saas_bench.customer_llm import CustomerSimulator
    from test_preflight_team_flow import completion
    with make_team(mode, lambda *args: final('done'), 'ledger-customer-commit') as team:
        team._set_phase('ceo')
        client = OpenAI(api_key='OFFLINE-ONLY', base_url='https://opencode.ai/zen/go/v1',
            max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(
                lambda request: completion(final('Offline customer response.')))))
        customer = CustomerSimulator(client, team.server.conn, team.server.tools.config)
        customer.usage_recorder.path = team.test_root / 'simulator-usage.jsonl'
        sim = Simulator(team.server.conn, team.server.tools.config, default_rng(123), customer_simulator=customer)
        team.server.simulator = sim
        def step():
            sim.current_day = 1
            sim.conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES(1,'compute',-10,'cost-helper-commit')")
            text, incoming, outgoing = customer.complete_text(provider='opencode', model='deepseek-v4.1-flash', user='Offline scenario', max_tokens=50)
            assert text == 'Offline customer response.'
            customer._log_cost(1, 'offline-probe', incoming, outgoing)
            sim.conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES(1,'compute',-20,'dirty-cost-tail')")
            raise RuntimeError('PRIVATE-CUSTOMER-COMMIT-FAULT')
        sim.step_week = step
        try:
            status, result = raw(team, 'ceo', '/next-week', {'predictions': PREDICTIONS, 'rationale': 'cost commit fault'})
            assert status == 200 and not result['success']
            wait_until(lambda: not team.server.public_request_state()['active'])
            assert team.server._operation_failed and team.server.conn.in_transaction
            with sqlite3.connect(team.test_root / 'world.db') as durable:
                durable.row_factory = sqlite3.Row
                assert durable.execute("SELECT amount FROM ledger WHERE note='cost-helper-commit'").fetchone()[0] == -10
                assert durable.execute("SELECT COUNT(*) FROM ledger WHERE note='dirty-cost-tail'").fetchone()[0] == 0
                assert durable.execute("SELECT COUNT(*) FROM api_costs WHERE purpose='offline-probe'").fetchone()[0] == 1
                changed = [dict(r) for r in durable.execute("SELECT * FROM business_effects WHERE kind='advance_progress'")]
            assert len(changed) == 1
            assert json.loads(changed[0]['after_json'])['reached_day'] == 1
            assert customer.conn is team.server.conn and sim.conn is team.server.conn
            usage = [json.loads(line) for line in customer.usage_recorder.path.read_text().splitlines()]
            assert any(r['event'] == 'http_request' for r in usage)
            sample('customer-cost-' + mode, {'effects': changed, 'usage': usage})
        finally:
            client.close()


def test_terminal_result_is_idempotent_and_rejects_conflicts():
    from saas_bench.business_ledger import BusinessOperation
    from saas_bench.database import init_database
    conn = init_database(':memory:')
    try:
        operation = BusinessOperation(conn, actor='ceo', initiator='model', tool='next_week', day=0, parameters={})
        terminal = {'success': True, 'day': 7, 'advance': {'status': 'completed'}}
        operation.finish(terminal)
        before = conn.serialize()
        operation.finish({'advance': {'status': 'completed'}, 'day': 7, 'success': True})
        assert conn.serialize() == before
        with pytest.raises(RuntimeError, match='Conflicting business operation finalization'):
            operation.finish(dict(terminal, day=14))
        assert conn.serialize() == before
        mapped = BusinessOperation(conn, actor='ceo', initiator='model', tool='set_targeted_ops_spend', day=0, parameters={})
        result = {'success': True, 'data': {1: (1, 2)}}
        mapped.finish(result)
        frozen = conn.serialize()
        mapped.finish(result)
        assert conn.serialize() == frozen
    finally:
        conn.close()


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_prediction_finalization_holds_world_lock_against_other_public_writes(mode, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from saas_bench.business_ledger import BusinessOperation
    entered, release, attempted = threading.Event(), threading.Event(), threading.Event()
    with make_team(mode, lambda *args: final('done'), 'ledger-finalize-lock') as team:
        team._set_phase('ceo')
        original_finish = BusinessOperation.finish
        original_execute = team.server.execute_tool
        def fail_predictions(*args):
            raise OSError('known prediction failure')
        def paused_finish(operation, result=None, **kwargs):
            if isinstance(result, dict) and result.get('error') == 'prediction_save_failed':
                entered.set()
                assert release.wait(10)
            return original_finish(operation, result, **kwargs)
        def entering_write(*args, **kwargs):
            attempted.set()
            return original_execute(*args, **kwargs)
        monkeypatch.setattr('saas_bench.database.save_predictions', fail_predictions)
        monkeypatch.setattr(BusinessOperation, 'finish', paused_finish)
        monkeypatch.setattr(team.server, 'execute_tool', entering_write)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(raw, team, 'ceo', '/next-week', {'predictions': PREDICTIONS, 'rationale': 'known rollback'})
            try:
                assert entered.wait(10)
                second = pool.submit(raw, team, 'ceo', '/call', {'tool': 'set_capacity_tier', 'args': {'tier': 2}})
                assert attempted.wait(10)
                with pytest.raises(TimeoutError):
                    second.result(timeout=.2)
            finally:
                release.set()
            assert first.result(timeout=10)[1]['error'] == 'prediction_save_failed'
            assert second.result(timeout=10)[1]['success']
        assert [r['outcome'] for r in records(team)] == ['rejected', 'succeeded']
        assert len({r['operation_ref'] for r in effects(team)}) == 1
