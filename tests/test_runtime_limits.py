import argparse
import json
import resource
import subprocess
import time

import pytest

from saas_bench.process_boundary import Boundary, command_memory_limit
from saas_bench.runtime_options import add_runtime_arguments, runtime_environment
from test_public_sql import server


@pytest.mark.parametrize('value', ['0', 'max', 'infinity', 'unlimited'])
def test_bash_memory_limit_can_be_disabled(monkeypatch, value):
    monkeypatch.setenv('CEOBENCH_BASH_MEMORY_MAX_BYTES', value)
    assert command_memory_limit() == 0
    boundary = Boundary("python -c 'import resource,json; print(json.dumps(resource.getrlimit(resource.RLIMIT_AS)))'",
                        verified=True, memory_limit_bytes=0)
    try:
        process = subprocess.Popen(['bash', '-c', boundary.command], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = boundary.communicate(process, 5)
        assert process.returncode == 0, stderr
        assert json.loads(stdout) == list(resource.getrlimit(resource.RLIMIT_AS))
        assert boundary.record['memory_limit_bytes'] == 0
        assert boundary.record['boundary_closed'] is True
    finally:
        boundary.close()


def test_launch_runtime_options_export_only_explicit_values():
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    assert runtime_environment(parser.parse_args([])) == {}
    args = parser.parse_args(['--bash-memory-max', 'max', '--model-request-attempts', 'unlimited',
                              '--model-retry-delays', '1,2,3', '--advance-timeout', '9000'])
    assert runtime_environment(args) == {
        'CEOBENCH_BASH_MEMORY_MAX_BYTES': '0',
        'CEOBENCH_MODEL_REQUEST_ATTEMPTS': '0',
        'CEOBENCH_MODEL_RETRY_DELAYS': '1,2,3',
        'CEOBENCH_ADVANCE_TIMEOUT_SECONDS': '9000.0',
    }


def test_runtime_environment_snapshot_keeps_defaults_and_explicit_values(monkeypatch):
    from saas_bench.runtime_options import DEFAULTS, OPTIONS, runtime_environment_snapshot
    assert OPTIONS.keys() == DEFAULTS.keys()
    for env, _, _ in OPTIONS.values():
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv('CEOBENCH_CEO_BASH_TIMEOUT_SECONDS', '2400')
    monkeypatch.setenv('CEOBENCH_SQL_ROW_LIMIT', '17')
    saved = runtime_environment_snapshot()
    assert saved['CEOBENCH_CEO_BASH_TIMEOUT_SECONDS'] == '2400.0'
    assert saved['CEOBENCH_ANALYST_BASH_TIMEOUT_SECONDS'] == '480.0'
    assert saved['CEOBENCH_SQL_ROW_LIMIT'] == '17'
    assert saved['CEOBENCH_MODEL_RETRY_DELAYS'] == '10,20,40,80'
    assert 'CEOBENCH_MODEL_MAX_OUTPUT_TOKENS' not in saved


def test_public_sql_configured_result_scope_keeps_readonly_policy(server, monkeypatch):
    from saas_bench.public_sql import QueryDenied, execute_query
    sql = 'SELECT 1 AS n UNION ALL SELECT 2 UNION ALL SELECT 3'
    monkeypatch.setitem(server.runtime_environment, 'CEOBENCH_SQL_ROW_LIMIT', '2')
    result = execute_query(server, sql)
    assert result['rows'] == [{'n': 1}, {'n': 2}]
    assert result['truncated'] is True and 'exceeded 2 rows' in result['warning']
    monkeypatch.setitem(server.runtime_environment, 'CEOBENCH_SQL_ROW_LIMIT', '0')
    assert execute_query(server, sql)['rows'] == [{'n': 1}, {'n': 2}, {'n': 3}]
    with pytest.raises(QueryDenied):
        execute_query(server, 'DELETE FROM main.ledger')
    assert execute_query(server, 'SELECT amount FROM ledger')['rows'] == [{'amount': 42.0}]


def test_public_sql_configured_column_limit_reaches_reader(server, monkeypatch):
    import sqlite3
    from saas_bench.public_sql import execute_query
    sql = 'SELECT ' + ','.join(f'1 AS n{index}' for index in range(129))
    monkeypatch.setitem(server.runtime_environment, 'CEOBENCH_SQL_COLUMN_LIMIT', '128')
    with pytest.raises(sqlite3.OperationalError, match='too many columns'):
        execute_query(server, sql)
    monkeypatch.setitem(server.runtime_environment, 'CEOBENCH_SQL_COLUMN_LIMIT', '512')
    assert len(execute_query(server, sql)['columns']) == 129


def test_pf_deadline_can_be_relaxed_without_losing_cancel(monkeypatch):
    from saas_bench.pf_deadline import budget_scope
    monkeypatch.setenv('CEOBENCH_PF_TIMEOUT_SECONDS', '1800')
    with budget_scope() as budget:
        assert 1799 < budget.deadline - time.monotonic() <= 1800
    monkeypatch.setenv('CEOBENCH_PF_TIMEOUT_SECONDS', '0')
    with pytest.raises(TimeoutError, match='cancelled'):
        with budget_scope() as budget:
            assert budget.deadline == float('inf')
            budget.cancelled.set()
            budget.check()
    with pytest.raises(TimeoutError, match='time limit'):
        with budget_scope(timeout=0):
            pytest.fail('An exhausted caller budget must stay exhausted')


def test_executor_limits_and_child_environment_keep_their_launch_snapshot(tmp_path, monkeypatch):
    from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
    monkeypatch.setenv('CEOBENCH_BASH_OUTPUT_MAX_CHARS', '3')
    monkeypatch.setenv('CEOBENCH_PUBLIC_PYTHON_TIMEOUT_SECONDS', '40')
    first = BashAgentToolExecutor(tmp_path)
    monkeypatch.setenv('CEOBENCH_BASH_OUTPUT_MAX_CHARS', '100')
    monkeypatch.setenv('CEOBENCH_PUBLIC_PYTHON_TIMEOUT_SECONDS', '80')
    second = BashAgentToolExecutor(tmp_path)
    assert first.extra_env['CEOBENCH_PUBLIC_PYTHON_TIMEOUT_SECONDS'] == '40.0'
    assert second.extra_env['CEOBENCH_PUBLIC_PYTHON_TIMEOUT_SECONDS'] == '80.0'
    assert 'output truncated' in first.execute('bash', {'command': 'printf abcdef'})
    assert second.execute('bash', {'command': 'printf abcdef'}) == 'abcdef'


def test_truncated_bash_search_and_glob_keep_readable_full_outputs(tmp_path, monkeypatch):
    from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
    executor = BashAgentToolExecutor(tmp_path)
    monkeypatch.setitem(executor.runtime_environment, 'CEOBENCH_BASH_OUTPUT_MAX_CHARS', '12')
    result = executor.execute('bash', {'command': "printf 'abcdefghijklmnopqrstuvwxyz'"})
    assert 'output truncated' in result
    path = result.split('full output: ', 1)[1].split(']', 1)[0]
    assert (tmp_path / path).read_text() == 'abcdefghijklmnopqrstuvwxyz'
    monkeypatch.setitem(executor.runtime_environment, 'CEOBENCH_SEARCH_MATCH_LIMIT', '2')
    (tmp_path / 'matches.txt').write_text('match1\nmatch2\nmatch3\n')
    result = executor.execute('search_files', {'pattern': 'match', 'path': 'matches.txt'})
    assert 'Search truncated' in result and 'matches.txt:3:' not in result
    path = result.split('Full searched output: ', 1)[1].split(']', 1)[0]
    assert 'matches.txt:3: match3' in (tmp_path / path).read_text()
    monkeypatch.setitem(executor.runtime_environment, 'CEOBENCH_GLOB_LIMIT', '2')
    for index in range(3):
        (tmp_path / f'candidate-{index}').touch()
    result = executor.execute('glob_files', {'pattern': 'candidate-*'})
    assert 'Glob truncated' in result
    path = result.split('Full output: ', 1)[1].split(']', 1)[0]
    assert (tmp_path / path).read_text().splitlines() == ['candidate-0', 'candidate-1', 'candidate-2']


def test_public_python_timeout_is_configurable(monkeypatch, tmp_path):
    from saas_bench import _public_cli
    monkeypatch.setattr(_public_cli, '_base_dir', lambda: tmp_path)
    monkeypatch.setenv('CEOBENCH_PUBLIC_PYTHON_TIMEOUT_SECONDS', '.05')
    with pytest.raises(subprocess.TimeoutExpired):
        _public_cli._execute_python('__env__', 1, 'import time; time.sleep(1)')
    monkeypatch.setenv('CEOBENCH_PUBLIC_PYTHON_TIMEOUT_SECONDS', '0')
    with pytest.raises(SystemExit) as result:
        _public_cli._execute_python('__env__', 1, 'print("completed")')
    assert result.value.code == 0


def test_public_client_aligns_only_known_operation_deadlines(monkeypatch):
    import io
    from saas_bench import _public_cli
    waits = []
    def opened(request, *, timeout):
        waits.append((request.full_url.rsplit('/', 1)[-1], timeout))
        return io.BytesIO(b'{}')
    monkeypatch.setattr(_public_cli, 'urlopen', opened)
    monkeypatch.setenv('CEOBENCH_ADVANCE_TIMEOUT_SECONDS', '120')
    monkeypatch.setenv('CEOBENCH_SQL_TIMEOUT_SECONDS', '600')
    monkeypatch.setenv('CEOBENCH_SQL_RESPONSE_TIMEOUT_SECONDS', '120')
    monkeypatch.setenv('CEOBENCH_COMPLETION_WAIT_SECONDS', '30')
    for path in ('/next-week', '/query', '/daily-scripts'):
        assert _public_cli._api_call(1, 'POST', path) == {}
    assert waits == [('next-week', 150), ('query', 750), ('daily-scripts', 1800)]
