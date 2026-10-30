from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import socket
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from saas_bench import team_batch as batch
from saas_bench.run_lifecycle import RunCancelled
from test_preflight_team_batch import changed_manifest, frozen
from test_preflight_team_flow import ADVANCE, PREDICTIONS, calls, completion, final, make_team, raw


def wait_until(predicate):
    deadline = time.monotonic() + 3
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.01)
    assert predicate()


def operations(team):
    return {path: json.loads(path.read_text()) for path in (team.root / 'private/operations').glob('*.json')}


def completed_day(team, day):
    team.server.conn.execute('CREATE TABLE IF NOT EXISTS _rng_states(name TEXT PRIMARY KEY,state_json TEXT NOT NULL)')
    team.server.conn.execute("INSERT OR REPLACE INTO _rng_states(name,state_json) VALUES ('all',?)",
        (json.dumps({'_sim_state': {'current_day': day}}),))
    team.server.conn.commit()


@pytest.mark.parametrize('nested', [False, True])
def test_slow_advance_warns_past_ordinary_bash_deadline_and_next_command_keeps_limit(nested, caplog):
    with make_team('git', lambda *args: final('unused'), 'advance-warning-only') as team:
        team._set_phase('ceo')
        executor = team.roles['ceo'].executor
        executor.bash_timeout = .5
        executor.advance_warn_after = .05
        team.server.advance_warn_after = .05

        def step():
            time.sleep(.8)
            return SimpleNamespace(day=7)

        team.server.simulator.step_week = step
        command = ADVANCE
        if nested:
            (executor.workspace_path / 'advance.sh').write_text(ADVANCE)
            command = 'bash advance.sh'
        result = executor.execute('bash', dict(command=command))
        assert 'Day 7' in result
        assert executor._bash_facts['timed_out'] is False
        assert executor._bash_facts['process_boundary']['descendants_closed'] is True
        assert 'Slow week advancement still running' in caplog.text
        assert team.server.STEP_WEEK_TIMEOUT is None
        team._set_phase('ceo')
        result = executor.execute('bash', dict(command='sleep 1'))
        assert 'timed out after 0.5 seconds' in result
        assert executor._bash_facts['process_boundary']['descendants_closed'] is True


@pytest.mark.parametrize('reached,last_completed,progression', [(0, 0, 'not_advanced'), (3, 2, 'partial')])
@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_public_advance_failure_has_durable_terminal_progress(mode, reached, last_completed, progression):
    with make_team(mode, lambda *args: final('unused'), 'advance-failed-terminal') as team:
        team._set_phase('ceo')
        sim = team.server.simulator
        sim.current_day = 0
        completed_day(team, 0)

        def fail():
            sim.current_day = reached
            completed_day(team, last_completed)
            raise RuntimeError('injected advancement failure')

        sim.step_week = fail
        status, result = raw(team, 'ceo', '/next-week', dict(predictions=PREDICTIONS, rationale='offline failure'))
        assert status == 200
        assert result['success'] is False and result['error'] == 'step_week_failed'
        expected = dict(status='failed', old_day=0, target_day=7, reached_day=reached,
            last_completed_day=last_completed, progression=progression,
            error='RuntimeError: injected advancement failure')
        assert result['advance'] == expected
        rows = operations(team)
        assert len(rows) == 1
        path, receipt = next(iter(rows.items()))
        assert receipt['status'] == 'failed' and receipt['advance'] == expected
        if mode == 'pf':
            from test_sql_evidence import event_ids
            store = team.roles['ceo'].store
            public = [store.read_event(event) for event in event_ids(store)
                if store.read_event(event)['request']['kind'] == 'public_http']
            assert len(public) == 1
            assert public[0]['result']['status'] == 'failed'
        original = path.read_bytes()
        team._finish_operation(path)
        team._finish_operation(path, advance_outcome=expected)
        assert path.read_bytes() == original
        with pytest.raises(RuntimeError, match='Conflicting operation finalization'):
            team._finish_operation(path, advance_outcome=dict(expected, status='completed'))
        with pytest.raises(RunCancelled, match='world_operation_failed'):
            team._check_model_send()
        assert team.server.tools.current_day == 0


def test_public_advance_admission_failure_records_no_progress():
    with make_team('git', lambda *args: final('unused'), 'advance-admission-failure') as team:
        team._set_phase('ceo')
        team.server.simulator.current_day = 3
        team.server._advance_lock.acquire()
        try:
            status, result = raw(team, 'ceo', '/next-week', dict(predictions=PREDICTIONS, rationale='offline contention'))
        finally:
            team.server._advance_lock.release()
        assert status == 200 and result['error'] == 'advance_week_failed'
        receipt = next(iter(operations(team).values()))
        assert receipt['status'] == 'failed'
        assert receipt['advance']['progression'] == 'not_advanced'
        assert receipt['advance']['reached_day'] == receipt['advance']['last_completed_day'] == 0


def test_post_step_dashboard_failure_keeps_complete_advance_and_failed_wrapper():
    from saas_bench.agents.bash_agent.tools import NextDayTimeoutError
    with make_team('git', lambda *args: final('unused'), 'advance-dashboard-failure') as team:
        team._set_phase('ceo')
        sim = team.server.simulator
        sim.current_day = 0

        def step():
            sim.current_day = 7
            completed_day(team, 7)
            return SimpleNamespace(day=7, cash=100, mrr=42)

        def dashboard(*args):
            raise RuntimeError('injected dashboard failure')

        sim.step_week = step
        team.server.dashboard_callback = dashboard
        executor = team.roles['ceo'].executor
        wrapper_path = team._begin_operation('ceo_tool:bash')
        with pytest.raises(NextDayTimeoutError):
            executor.execute('bash', dict(command=ADVANCE))
        with pytest.raises(RunCancelled, match='world_operation_failed'):
            team._finish_operation(wrapper_path, executor)
        rows = list(operations(team).values())
        advance = next(row for row in rows if row['kind'] == 'api:advance')
        wrapper = next(row for row in rows if row['kind'] == 'ceo_tool:bash')
        assert advance['status'] == 'completed' and advance['advance']['progression'] == 'complete'
        assert advance['advance']['reached_day'] == advance['advance']['last_completed_day'] == 7
        assert wrapper['status'] == 'failed'
        assert team.server._last_day_result.mrr == 42


@pytest.mark.parametrize('late_failure', [False, True])
@pytest.mark.parametrize('client_disconnect', [False, True])
def test_public_timeout_or_disconnect_gets_late_advance_terminal(late_failure, client_disconnect):
    entered, release = threading.Event(), threading.Event()
    with make_team('git', lambda *args: final('unused'), 'advance-late-terminal') as team:
        team._set_phase('ceo')
        sim = team.server.simulator
        sim.current_day = 0
        completed_day(team, 0)
        team.server.STEP_WEEK_TIMEOUT = .1

        def step():
            sim.current_day = 1
            entered.set()
            assert release.wait(3)
            if late_failure:
                sim.current_day = 3
                completed_day(team, 2)
                raise RuntimeError('injected late failure')
            sim.current_day = 7
            completed_day(team, 7)
            return SimpleNamespace(day=7)

        sim.step_week = step
        try:
            if client_disconnect:
                from saas_bench.runtime_temp import connect_unix
                connection = http.client.HTTPConnection('localhost', timeout=.03)
                connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                connection.sock.settimeout(.03)
                connect_unix(connection.sock, team.server.role_sockets['ceo'])
                connection.request('POST', '/next-week', json.dumps(dict(predictions=PREDICTIONS, rationale='offline disconnect')),
                    {'content-type': 'application/json'})
                assert entered.wait(2)
                with pytest.raises(TimeoutError):
                    connection.getresponse()
                connection.close()
                wait_until(lambda: not team.server.public_request_state()['active'])
            else:
                status, result = raw(team, 'ceo', '/next-week', dict(predictions=PREDICTIONS, rationale='offline timeout'))
                assert status == 200 and result['error'] == 'step_week_timeout'
            assert entered.is_set()
            rows = operations(team)
            assert len(rows) == 1
            path = next(iter(rows))
            assert rows[path]['status'] == 'running'
            release.set()
            wait_until(lambda: json.loads(path.read_text())['status'] != 'running')
            receipt = json.loads(path.read_text())
            assert receipt['status'] == ('failed' if late_failure else 'completed')
            assert receipt['advance']['reached_day'] == (3 if late_failure else 7)
            assert receipt['advance']['last_completed_day'] == (2 if late_failure else 7)
            assert receipt['advance']['progression'] == ('partial' if late_failure else 'complete')
            with pytest.raises(RunCancelled, match='world_operation_failed'):
                team._check_model_send()
        finally:
            release.set()


@pytest.mark.parametrize('late_failure', [False, True])
@pytest.mark.parametrize('cancel_before', [False, True])
def test_cli_disconnect_then_runtime_close_settles_after_http_finishes(late_failure, cancel_before, recwarn):
    from saas_bench.agents.bash_agent.tools import NextDayTimeoutError
    entered, release = threading.Event(), threading.Event()
    with make_team('git', lambda *args: final('unused'), 'advance-close-before-settle') as team:
        team._set_phase('ceo')
        sim = team.server.simulator
        sim.current_day = 0
        completed_day(team, 0)
        team.server.STEP_WEEK_TIMEOUT = 30

        def step():
            sim.current_day = 1
            entered.set()
            assert release.wait(10)
            sim.current_day = 3 if late_failure else 7
            completed_day(team, 2 if late_failure else 7)
            if late_failure:
                raise RuntimeError('injected failure after runtime close')
            return SimpleNamespace(day=7)

        sim.step_week = step
        executor = team.roles['ceo'].executor
        executor.bash_timeout = .2
        executor.advance_timeout = .2
        hook_errors, hook_threads = [], []
        hook_finished = threading.Event()
        finish_request = team.server.request_finished_callback

        def finished_request():
            hook_threads.append(threading.current_thread())
            try:
                finish_request()
            except BaseException as exc:
                hook_errors.append(type(exc).__name__ + ': ' + str(exc))
                raise
            finally:
                hook_finished.set()

        team.server.request_finished_callback = finished_request
        wrapper_path = team._begin_operation('ceo_tool:bash')
        try:
            with pytest.raises(NextDayTimeoutError):
                executor.execute('bash', dict(command=ADVANCE))
            assert entered.is_set()
            with pytest.raises(RunCancelled, match='business_operation_pending'):
                team._finish_operation(wrapper_path, executor)
            assert json.loads(wrapper_path.read_text())['status'] == 'running'
            team.close()
            assert not release.is_set() and team.server.public_request_state()['active']
            if cancel_before:
                team.request_pause()
            release.set()
            wait_until(lambda: json.loads(wrapper_path.read_text())['status'] == 'failed')
            wait_until(lambda: not team.server.public_request_state()['active'])
            assert hook_finished.wait(3)
            for thread in hook_threads:
                thread.join(1)
                assert not thread.is_alive()
            assert hook_errors == []
            assert not any(issubclass(warning.category, pytest.PytestUnhandledThreadExceptionWarning) for warning in recwarn)
            rows = list(operations(team).values())
            advance = next(row for row in rows if row['kind'] == 'api:advance')
            wrapper = next(row for row in rows if row['kind'] == 'ceo_tool:bash')
            assert advance['status'] == ('failed' if late_failure else 'completed')
            assert team.server._advance_outcome == advance['advance']
            assert wrapper['status'] == 'failed'
            assert wrapper['finalization']['process_boundary']['descendants_closed'] is True
            assert team.server._step_day_timed_out is False
            assert executor.preserved_process is None
            with pytest.raises(RunCancelled, match='requested_pause' if cancel_before else 'world_operation_failed'):
                team._check_model_send()
        finally:
            release.set()


def test_advance_settlement_during_listener_close_keeps_terminal_consistent(monkeypatch):
    from saas_bench.agents.bash_agent.tools import NextDayTimeoutError
    entered, release_step, listener_closed, release_close = (threading.Event() for _ in range(4))
    callback_entered, callback_failed = threading.Event(), threading.Event()
    errors = []
    with make_team('git', lambda *args: final('unused'), 'advance-during-listener-close') as team:
        team._set_phase('ceo')
        sim = team.server.simulator
        sim.current_day = 0
        completed_day(team, 0)
        team.server.STEP_WEEK_TIMEOUT = 30

        def step():
            entered.set()
            assert release_step.wait(10)
            sim.current_day = 7
            completed_day(team, 7)
            return SimpleNamespace(day=7)

        first = team.server._role_httpds[0]
        close_listener = first.server_close
        settle_pending = team._settle_pending_operations

        def close():
            close_listener()
            listener_closed.set()
            assert release_close.wait(10)

        def settle():
            callback_entered.set()
            try:
                return settle_pending()
            except BaseException as exc:
                errors.append(type(exc).__name__ + ': ' + str(exc))
                callback_failed.set()
                raise

        monkeypatch.setattr(first, 'server_close', close)
        monkeypatch.setattr(team, '_settle_pending_operations', settle)
        team.server.request_finished_callback = settle
        sim.step_week = step
        executor = team.roles['ceo'].executor
        executor.bash_timeout = .2
        executor.advance_timeout = .2
        wrapper_path = team._begin_operation('ceo_tool:bash')
        try:
            with pytest.raises(NextDayTimeoutError):
                executor.execute('bash', dict(command=ADVANCE))
            assert entered.is_set()
            with pytest.raises(RunCancelled, match='business_operation_pending'):
                team._finish_operation(wrapper_path, executor)
            with ThreadPoolExecutor(max_workers=1) as pool:
                closed = pool.submit(team.close)
                assert listener_closed.wait(3)
                release_step.set()
                assert callback_entered.wait(3)
                callback_failed.wait(.3)
                release_close.set()
                closed.result(timeout=5)
            wait_until(lambda: not team.server.public_request_state()['active'])
            assert errors == []
            wait_until(lambda: json.loads(wrapper_path.read_text())['status'] == 'failed')
            advance = next(row for row in operations(team).values() if row['kind'] == 'api:advance')
            assert advance['status'] == 'completed'
            assert team.server._advance_outcome == advance['advance']
        finally:
            release_step.set()
            release_close.set()


def test_cli_failure_holds_once_and_manual_checkpoint_resume_preserves_failed_attempt(frozen, tmp_path):
    def stop_after_four_weeks(manifest):
        for run in manifest['runs']:
            run['stop_after_day'] = 28

    path = changed_manifest(frozen, tmp_path, stop_after_four_weeks)
    entry = next(row for row in batch.read(path)['runs'] if row['group'] == 'git')
    sent = []

    def client(role):
        def respond(request):
            sent.append(role)
            body = json.loads(request.content)
            message = calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final('Offline analyst answer.')
            return completion(message, body.get('stream', False))
        return batch.checked_client(role, 'advance-terminal', transport=httpx.MockTransport(respond))

    def failing_simulator(conn, config, rng):
        sim = batch.simulator(conn, config, rng)

        def fail():
            sim.current_day += 1
            sim.save_rng_states()
            sim.current_day += 1
            conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES (?,'operations',-1,'partial fault')", (sim.current_day,))
            conn.commit()
            raise RuntimeError('injected partial CLI advance')

        sim.step_week = fail
        return sim

    first = batch.prepare(path, entry['run_id'], live=True, client_factory=client, simulator_factory=failing_simulator)
    directory = batch.attempts(entry)[0]
    assert first['status'] == 'paused' and first['runtime_failure'] == 'world_operation_failed'
    assert first['hold'] and 'unknown' not in first['hold_detail'].lower()
    assert sent.count('ceo') == 1
    receipts = [json.loads(row.read_text()) for row in (directory / 'runtime/private/operations').glob('*.json')]
    advance = next(row for row in receipts if row['kind'] == 'api:advance')
    wrapper = next(row for row in receipts if row['kind'] == 'ceo_tool:bash')
    assert advance['status'] == wrapper['status'] == 'failed'
    assert advance['advance']['reached_day'] == 2 and advance['advance']['last_completed_day'] == 1
    assert wrapper['finalization']['process_boundary']['descendants_closed'] is True
    usage = [json.loads(row) for row in (directory / 'runtime/private/ceo/model-usage.jsonl').read_text().splitlines()]
    assert sum(row['event'] == 'request' for row in usage) == 1
    original = {row: row.read_bytes() for row in directory.rglob('*') if row.is_file()}
    resumed = batch.prepare(path, entry['run_id'], resume=True, live=True, week=1, force_resume=True,
        client_factory=client, simulator_factory=batch.simulator)
    assert resumed['attempt'] == 2 and resumed['status'] == 'finished' and resumed['day'] == 28
    assert resumed['checkpoint_rollback']['day'] == 0
    assert {row: row.read_bytes() for row in original} == original
