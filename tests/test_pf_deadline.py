"""PF host calls share one deadline and leave unresolved work behind a stop gate."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
from types import SimpleNamespace
import urllib.error

import pytest

from saas_bench.agents.bash_agent.tools import ProcessBoundaryError
from saas_bench.pf_deadline import Budget, budget_scope, check, current_budget
from saas_bench.pf_refresh import refresh
from saas_bench.run_lifecycle import RunCancelled
from test_execution_capture import captured, records
from test_pf_stale import record_sql
from test_public_sql import server
from test_text_registry import workspace


def test_pf_model_output_discloses_formatting_null_limit(workspace, tmp_path, server):
    from test_pf_stale import chain, forward
    store, registry, executor = chain(workspace, tmp_path, server,
        sql="SELECT printf('%1048577s','x') AS n")
    answer = forward(executor)
    leaf = answer['items'][-1]
    assert leaf['check']['predicate_result'] == 'cannot_check'
    assert leaf['check']['affected'] is True
    assert 'SQLite formatting function returned NULL' in leaf['check']['reason']
    text = executor.execute('bash', {'command': 'pf depend r1'})
    assert '1 MiB value limit' in text
    assert 'SQLite formatting function returned NULL' in text


def test_uncooperative_pf_cleanup_is_bounded_and_stops_next_tool(captured):
    _, store, executor = captured
    entered, release = threading.Event(), threading.Event()

    def blocked(operation, arguments):
        assert operation == 'pf_dependencies'
        entered.set()
        assert release.wait(5), 'PF release missing'
        return 'PF returned after its client was killed'

    executor.pf_queries = SimpleNamespace(execute=blocked, decorate=lambda capture, result, after: result)
    executor.bash_warn_after, executor.bash_timeout = .1, .3
    with ThreadPoolExecutor(max_workers=1) as pool:
        started = time.monotonic()
        running = pool.submit(executor.execute, 'bash', {'command': 'pf depend r1'})
        service = None
        try:
            assert entered.wait(2)
            with pytest.raises(ProcessBoundaryError, match='PF host request remains active'):
                running.result(timeout=2)
            elapsed = time.monotonic() - started
            assert elapsed < 1.6
            service = executor._pf_service
            assert service.thread.is_alive()
            bash, child = records(store)
            assert bash['result']['boundary_closed'] is True
            assert bash['result']['status'] == 'result_unknown'
            assert bash['result']['pf_cleanup'] == dict(closed=False,
                unfinished_event_ids=[child['request']['event_id']])
            assert child['result']['status'] == 'result_unknown'
            assert child['result']['completed_at'] is None
            with pytest.raises(ProcessBoundaryError, match='PF host request remains active'):
                executor.execute('bash', {'command': 'touch must-not-run'})
            assert not (executor.workspace_path / 'must-not-run').exists()
            print(json.dumps(dict(case='pf_host_uncooperative', bounded_return=True,
                boundary_closed=True, pf_closed=False, next_tool_blocked=True)))
        finally:
            release.set()
            if service is None:
                service = getattr(executor, '_pf_service', None)
            if service:
                service.thread.join(timeout=2)
                assert not service.thread.is_alive()
                service.close()
    child = records(store)[1]
    assert child['result']['status'] == 'timed_out'
    with pytest.raises(RuntimeError, match='capture failed'):
        store.assert_healthy()


def test_cooperative_pf_deadline_closes_host_and_allows_next_tool(captured):
    _, store, executor = captured
    entered = threading.Event()

    def polling(operation, arguments):
        entered.set()
        assert current_budget() is not None
        while True:
            check()
            time.sleep(.005)

    executor.pf_queries = SimpleNamespace(execute=polling, decorate=lambda capture, result, after: result)
    executor.bash_warn_after, executor.bash_timeout = .1, .3
    result = executor.execute('bash', {'command': 'pf depend r1'})
    assert entered.is_set()
    assert 'timed out' in result or 'PF call time limit exceeded' in result
    assert executor._pf_service is None
    bash, child = records(store)
    assert bash['result']['pf_cleanup'] == dict(closed=True, unfinished_event_ids=[])
    assert child['result']['status'] == 'timed_out'
    store.assert_healthy()
    assert executor.execute('bash', {'command': 'echo next-tool'}) == 'next-tool\n'
    print(json.dumps(dict(case='pf_host_cooperative', pf_closed=True,
                         child_status='timed_out', next_tool='next-tool')))


def test_refresh_sources_share_deadline_and_preserve_completed_reads(captured, monkeypatch):
    api, store, _ = captured
    from saas_bench import pf_refresh
    api.QUERY_TIMEOUT_SECONDS = .15
    sqls = ['SELECT 42 AS n',
            'WITH RECURSIVE t(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM t WHERE n<100000000) SELECT sum(n) AS n FROM t',
            'SELECT 3 AS n']
    versions = [record_sql(store, sql, dict(columns=['n'], rows=[{'n': 42}])) for sql in sqls]
    parent = store.begin_event('pf_dependencies', {'target': {'record': 'r1'}})
    deadlines = []
    execute = pf_refresh.execute_snapshot

    def observed(conn, sql, deadline, execution):
        deadlines.append(deadline)
        return execute(conn, sql, deadline, execution)

    monkeypatch.setattr(pf_refresh, 'execute_snapshot', observed)
    before = api.conn.serialize()
    result = refresh(api, versions, parent)
    store.complete(parent)
    assert api.conn.serialize() == before
    assert len(deadlines) == 2 and deadlines[0] == deadlines[1]
    children = [store.read_event(result[version].split(':')[0]) for version in versions]
    assert [row['result']['status'] for row in children] == ['succeeded', 'timed_out', 'timed_out']
    assert [row['result']['attempted'] for row in children] == [True, True, False]
    assert json.loads(store.get_content(result[versions[0]])[1])['rows'] == [{'n': 42}]
    store.assert_healthy()
    print(json.dumps(dict(case='pf_refresh_total_deadline', shared_deadline=True,
        statuses=['succeeded', 'timed_out', 'timed_out'], attempted=[True, True, False])))


def test_refresh_lock_wait_observes_cancellation(captured):
    api, store, _ = captured
    version = record_sql(store, 'SELECT 42 AS n', dict(columns=['n'], rows=[{'n': 42}]))
    parent = store.begin_event('pf_dependencies', {'target': {'record': 'r1'}})
    cancelled = threading.Event()
    budget = Budget(time.monotonic() + 5, cancelled=cancelled)
    locked, release = threading.Event(), threading.Event()

    def hold_lock():
        with api._lock:
            locked.set()
            assert release.wait(3)

    def run():
        with budget_scope(budget=budget):
            return refresh(api, [version], parent)

    with ThreadPoolExecutor(max_workers=2) as pool:
        holding = pool.submit(hold_lock)
        try:
            assert locked.wait(1)
            running = pool.submit(run)
            time.sleep(.03)
            cancelled.set()
            with pytest.raises(TimeoutError, match='cancelled'):
                running.result(timeout=.5)
        finally:
            release.set()
            holding.result(timeout=1)
    store.complete(parent, 'timed_out')
    children = [row for row in records(store) if row['request']['kind'] == 'sql_query'
                and row['request']['parent_event_id'] == parent]
    assert len(children) == 1
    assert children[0]['result']['status'] == 'timed_out'
    assert children[0]['result']['attempted'] is False
    store.assert_healthy()


def test_transport_uses_remaining_pf_budget():
    from saas_bench.agents.bash_agent.run_test import BashAgentRunner
    posted = []
    runner = object.__new__(BashAgentRunner)
    runner._http_post = lambda path, data, timeout: posted.append((path, data, timeout)) or {'versions': {'old': 'new'}}
    with budget_scope(timeout=.5):
        assert runner._pf_refresh(['old'], 'parent') == {'old': 'new'}
    path, data, timeout = posted[0]
    assert path == '/pf-refresh'
    assert data == dict(versions=['old'], parent='parent', remaining_seconds=timeout)
    assert 0 < timeout <= .5


def test_real_refresh_transport_propagates_deadline(captured):
    from saas_bench.agents.bash_agent.run_test import BashAgentRunner
    api, store, _ = captured
    api.checkpoint_token = 'private-fixture-token'
    runner = object.__new__(BashAgentRunner)
    runner._server_port, runner._checkpoint_token = api.port, api.checkpoint_token
    sql = 'WITH RECURSIVE t(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM t WHERE n<100000000) SELECT sum(n) AS n FROM t'
    version = record_sql(store, sql, dict(columns=['n'], rows=[{'n': 42}]))
    parent = store.begin_event('pf_dependencies', {})
    with pytest.raises((TimeoutError, urllib.error.HTTPError)):
        with budget_scope(timeout=.1):
            runner._pf_refresh([version], parent)
    deadline = time.monotonic() + 2
    while True:
        children = [row for row in records(store) if row['request']['parent_event_id'] == parent]
        if children and children[0]['result']['completed_at'] and children[0]['delivery']['send_state'] == 'internal':
            break
        assert time.monotonic() < deadline, 'PF refresh did not record its timeout'
        time.sleep(.01)
    assert len(children) == 1
    assert children[0]['result']['status'] == 'timed_out'
    assert children[0]['result']['refresh_error'] == 'TimeoutError'
    store.complete(parent, 'timed_out')
    store.assert_healthy()


def test_refresh_large_sql_value_is_explicitly_rejected(captured):
    api, store, _ = captured
    sql = "SELECT replace(printf('%1000000s','x'),' ','xxx') AS n"
    versions = [record_sql(store, 'SELECT 42 AS n', dict(columns=['n'], rows=[{'n': 42}])),
                record_sql(store, sql, {})]
    parent = store.begin_event('pf_dependencies', {})
    result = refresh(api, versions, parent)
    children = [store.read_event(result[version].split(':')[0]) for version in versions]
    assert [row['result']['http_status'] for row in children] == [200, 413]
    assert children[1]['result']['permanent_error'] is True
    assert children[1]['result']['error_kind'] == 'query'
    assert json.loads(store.get_content(result[versions[1]])[1]) == dict(
        success=False, error='refresh_result_too_large')
    assert json.loads(store.get_content(result[versions[0]])[1])['rows'] == [{'n': 42}]
    store.complete(parent)
    store.assert_healthy()


@pytest.mark.parametrize('sql,warning', [("SELECT printf('%2000000000s','x') AS n", True),
                                         ("SELECT printf('') AS n", False),
                                         ("SELECT printf('%k','x') AS n", True)])
def test_refresh_sql_formatting_null_keeps_warning(captured, sql, warning):
    from saas_bench.public_sql import FORMATTING_NULL_WARNING
    api, store, _ = captured
    version = record_sql(store, sql, {})
    parent = store.begin_event('pf_dependencies', {})
    result = refresh(api, [version], parent)
    child = store.read_event(result[version].split(':')[0])
    assert child['result']['http_status'] == 200
    body = json.loads(store.get_content(result[version])[1])
    assert body['rows'] == [{'n': None}]
    if warning:
        assert body['warning'] == FORMATTING_NULL_WARNING
    else:
        assert 'warning' not in body
    store.complete(parent)
    store.assert_healthy()


def test_refresh_format_warning_does_not_leak_from_failed_source(captured):
    api, store, _ = captured
    sql = "SELECT printf('%2000000000s','x') AS a, replace(printf('%1000000s','x'),' ','xxx') AS b"
    versions = [record_sql(store, sql, {}), record_sql(store, 'SELECT 42 AS n', {})]
    parent = store.begin_event('pf_dependencies', {})
    result = refresh(api, versions, parent)
    children = [store.read_event(result[version].split(':')[0]) for version in versions]
    assert [row['result']['http_status'] for row in children] == [413, 200]
    body = json.loads(store.get_content(result[versions[1]])[1])
    assert body['rows'] == [{'n': 42}]
    assert 'warning' not in body
    store.complete(parent)
    store.assert_healthy()


def test_refresh_serialized_response_has_a_byte_limit(captured, monkeypatch):
    api, store, _ = captured
    monkeypatch.setattr('saas_bench.pf_refresh.RESPONSE_LIMIT_BYTES', 64)
    version = record_sql(store, 'SELECT 42 AS n', {})
    parent = store.begin_event('pf_dependencies', {})
    result = refresh(api, [version], parent)
    child = store.read_event(result[version].split(':')[0])
    assert child['result']['http_status'] == 413
    assert child['result']['refresh_error'] == 'QueryResultTooLarge'
    assert child['result']['refresh_error_detail'] == 'PF refresh response exceeded 64 bytes. Narrow the source.'
    assert json.loads(store.get_content(result[version])[1]) == dict(
        success=False, error='refresh_result_too_large')
    store.complete(parent)
    store.assert_healthy()


def test_lifecycle_cancellation_is_visible_in_sql_progress():
    def stop():
        raise RunCancelled('worker stopped')
    budget = Budget(time.monotonic() + 5, check=stop)
    assert budget.expired()
    assert budget.cancelled.is_set()
    with pytest.raises(TimeoutError, match='cancelled'):
        budget.remaining()
