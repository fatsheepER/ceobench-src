"""PF host calls share one deadline and leave unresolved work behind a stop gate."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
from types import SimpleNamespace

import pytest

from saas_bench.agents.bash_agent.tools import ProcessBoundaryError
from saas_bench.pf_deadline import Budget, check, current_budget
from saas_bench.run_lifecycle import RunCancelled
from test_execution_capture import captured, records
from test_public_sql import server

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
                running.result(timeout=4)
            elapsed = time.monotonic() - started
            assert elapsed < 4
            assert not release.is_set()
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


def test_lifecycle_cancellation_is_visible_in_sql_progress():
    def stop():
        raise RunCancelled('worker stopped')
    budget = Budget(time.monotonic() + 5, check=stop)
    assert budget.expired()
    assert budget.cancelled.is_set()
    with pytest.raises(TimeoutError, match='cancelled'):
        budget.remaining()
