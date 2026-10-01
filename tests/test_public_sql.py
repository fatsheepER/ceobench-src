"""Public SQL policy tests use synthetic worlds; no provider calls."""
import hashlib
import json
import shutil
from pathlib import Path
import sqlite3
import sys
import threading
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from saas_bench.api_server import NovaMindAPIServer
from saas_bench.database import init_database
from saas_bench.public_sql import (
    PUBLIC_COLUMNS, PUBLIC_POLICY_VERSION, QueryDenied, SnapshotUnavailable,
    execute_query, install_authorizer, query_snapshot,
)


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr('saas_bench.api_server._ORACLE_MODE', False)
    initial = init_database(tmp_path / 'world.db')
    initial.close()
    conn = sqlite3.connect(tmp_path / 'world.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES(0,'operations',42,NULL)")
    conn.execute("INSERT INTO group_insight_snapshots VALUES('S1',20,123,0.25,456)")
    conn.commit()
    tools = SimpleNamespace(workspace_path=tmp_path / 'workspace', current_day=0)
    tools.workspace_path.mkdir()
    tools.set_current_day = lambda day: setattr(tools, 'current_day', day)
    api = NovaMindAPIServer(tools, conn=conn)
    yield api
    api.stop()
    conn.close()


DENIED = [
    "UPDATE ledger SET amount=99", "-- comment\nDELETE FROM ledger",
    "WITH c AS (SELECT 1) UPDATE ledger SET amount=99 RETURNING amount",
    "WITH c AS (SELECT 1) DELETE FROM ledger",
    "WITH c AS (SELECT 1) INSERT INTO ledger(day,category,amount) VALUES(0,'x',1)",
    "CREATE TABLE leak(x)", "CREATE TEMP TABLE leak(x)", "DROP TABLE ledger",
    "ATTACH ':memory:' AS leak", "DETACH main", "PRAGMA query_only=OFF",
    "PRAGMA table_info(customers)", "SELECT * FROM pragma_table_info('customers')",
    "SELECT * FROM sqlite_master", "SELECT * FROM sqlite_temp_master",
    "SELECT * FROM group_insight_snapshots", "SELECT count(*) FROM group_insight_snapshots",
    "SELECT l.id FROM ledger l JOIN group_insight_snapshots s ON l.day=s.snapshot_day",
    "WITH ledger AS (SELECT * FROM main.group_insight_snapshots) SELECT * FROM ledger",
    "SELECT (SELECT snapshot_c_max FROM group_insight_snapshots) FROM ledger",
    "SELECT actual_completion_day FROM research_projects",
    "SELECT actual_completion_day AS DoNe_On FROM research_projects",
    "SELECT project_id FROM research_projects WHERE actual_completion_day=19",
    "SELECT project_id FROM research_projects ORDER BY actual_completion_day",
    "SELECT sum(actual_completion_day) FROM research_projects",
    "SELECT p.project_id FROM research_projects p JOIN ledger l ON p.actual_completion_day=l.day",
    "SELECT first_billing_done FROM subscriptions",
    "SELECT seat_count FROM customers", "SELECT effect_by_group FROM agent_social_media_posts",
    "SELECT views_by_group FROM agent_social_media_posts",
    "SELECT reasoning_by_group FROM agent_social_media_posts",
    "SELECT actual_completion_day AS x FROM main.research_projects",
    "SELECT * FROM main.research_projects", "SELECT rowid FROM main.daily_usage",
    "SELECT load_extension('missing')", "SELECT sqlite_version()",
    "BEGIN", "SAVEPOINT evil", "VACUUM", "REINDEX",
    "SELECT 1; DELETE FROM ledger",
]


@pytest.mark.parametrize('sql', DENIED)
def test_denies_entire_query_without_world_changes(server, sql):
    before = server.conn.serialize()
    with pytest.raises((QueryDenied, sqlite3.Error)):
        execute_query(server, sql)
    assert server.conn.serialize() == before
    assert not server.conn.in_transaction


def test_policy_and_every_public_projection(server):
    assert PUBLIC_POLICY_VERSION == 'public-sql-v1'
    assert len(PUBLIC_COLUMNS) == 19
    assert sum(map(len, PUBLIC_COLUMNS.values())) == 145
    # Any membership/order change requires an explicit policy review and fixture update.
    assert hashlib.sha256(json.dumps(PUBLIC_COLUMNS, sort_keys=True).encode()).hexdigest() == 'e845b1e89b2793c55cc7003adc26bdc6c28478cd241a9d1f6bed6f304352fb93'
    for table, columns in PUBLIC_COLUMNS.items():
        result = execute_query(server, f'SELECT * FROM "{table}" LIMIT 0')
        original_order = [row[1] for row in server.conn.execute(f'PRAGMA table_info("{table}")') if row[1] in columns]
        assert result['columns'] == original_order
        assert result['rows'] == []
        execute_query(server, f'SELECT count(*) FROM "{table}"')
    result = execute_query(server, 'SELECT s.seat_count FROM subscriptions s JOIN customers c USING(customer_id)')
    assert result['columns'] == ['seat_count']
    assert execute_query(server, "SELECT amount AS actual_completion_day FROM ledger")['rows'] == [{'actual_completion_day': 42.0}]


def test_legal_sql_results_and_limits(server):
    query = '''WITH x AS (SELECT amount,note FROM ledger UNION ALL SELECT amount,note FROM ledger)
               SELECT amount,note,row_number() OVER(ORDER BY amount) AS n FROM x ORDER BY n DESC'''
    assert execute_query(server, query)['rows'] == [dict(amount=42.0,note=None,n=2),dict(amount=42.0,note=None,n=1)]
    assert execute_query(server, "SELECT 'pragma sqlite_master' AS label")['rows'][0]['label'] == 'pragma sqlite_master'
    result = execute_query(server, 'WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x WHERE n<5001) SELECT n FROM x')
    assert result['row_count'] == 5000 and result['truncated']
    assert result['rows'][0] == {'n': 1} and result['rows'][-1] == {'n': 5000}
    assert execute_query(server, "SELECT json_extract('{\"n\":4}', '$.n') AS n, round(avg(amount)) AS a FROM ledger")['rows'] == [{'n': 4, 'a': 42.0}]


def test_fresh_snapshot_cleanup_and_readonly_independently(server):
    paths = []
    for amount in (43,44):
        server.conn.execute('UPDATE ledger SET amount=?', (amount,)); server.conn.commit()
        with query_snapshot(server, time.monotonic()+5) as (conn, metadata):
            # Remove the authorizer to verify the independent read-only protection.
            conn.set_authorizer(None)
            path = Path(conn.execute('PRAGMA database_list').fetchone()[2]); paths.append(path)
            assert path.exists() and not path.is_relative_to(server.script_workspace)
            assert metadata['day'] == 0
            assert conn.execute('SELECT amount FROM ledger').fetchone()[0] == amount
            conn.execute('PRAGMA query_only=OFF')
            with pytest.raises(sqlite3.OperationalError, match='readonly'):
                conn.execute('UPDATE main.ledger SET amount=99')
        assert path.exists()  # Same-world queries reuse this immutable snapshot.
        with query_snapshot(server, time.monotonic()+5) as (_, again):
            assert again['snapshot_reused'] and again['snapshot_ref'] == metadata['snapshot_ref']
    assert paths[0] != paths[1]
    assert not paths[0].exists()
    # Authorizer independently rejects writes on an otherwise writable connection.
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE ledger(amount)')
    install_authorizer(conn)
    with pytest.raises(sqlite3.DatabaseError): conn.execute('INSERT INTO ledger VALUES(1)')
    conn.close()


def test_timeout_transaction_and_unknown_world(server, monkeypatch):
    monkeypatch.setattr(server, 'QUERY_TIMEOUT_SECONDS', 0.02)
    with pytest.raises(TimeoutError):
        execute_query(server, 'WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x) SELECT sum(n) FROM x')
    assert execute_query(server, 'SELECT amount FROM ledger')['row_count'] == 1
    server.conn.execute('UPDATE ledger SET amount=45')
    with pytest.raises(SnapshotUnavailable): execute_query(server, 'SELECT 1')
    assert server.conn.in_transaction  # No implicit commit or rollback.
    server.conn.rollback()
    server._operation_failed = True
    with pytest.raises(SnapshotUnavailable): execute_query(server, 'SELECT 1')


@pytest.mark.parametrize('tool,args', [
    ('set_targeted_ad_spend', {'targeted_spend': {'social_media': {'S1': 10}}}),
    ('set_targeted_ops_spend', {'targeted_spend': {'S1': 10}}),
    ('set_targeted_dev_spend', {'targeted_spend': {'S1': 10}}),
    ('set_ads_strength', {'global_strength': 0.1}),
    ('set_lead_promotion', {'global_promotion': 1}),
    ('set_promotion', {'global_promotion': 1}),
])
def test_config_override_success_allows_immediate_public_query(server, tool, args):
    from saas_bench.tools import AgentTools
    server.tools = AgentTools(server.conn, 0, server.script_workspace)
    before = execute_query(server, 'SELECT amount FROM ledger')['rows']
    assert server.execute_tool(tool, args).success
    assert execute_query(server, 'SELECT amount FROM ledger')['rows'] == before
    assert not server.conn.in_transaction
    # A successful configuration action must persist its history independently
    # of the next weekly advance or another tool's incidental commit.
    db_path = server.conn.execute('PRAGMA database_list').fetchone()[2]
    with sqlite3.connect(db_path) as independent:
        assert independent.execute('SELECT tool_name FROM config_overrides').fetchall() == [(tool,)]


def request(server, body):
    req = urllib.request.Request(f'http://127.0.0.1:{server.port}/query', json.dumps(body).encode(), {'Content-Type':'application/json'})
    try: response = urllib.request.urlopen(req, timeout=5)
    except urllib.error.HTTPError as exc: response = exc
    with response: return response.status, json.load(response)


def test_http_sdk_and_weekly_script(server, monkeypatch):
    from saas_bench.novamind_api import _client
    server.start()
    monkeypatch.setenv('NOVAMIND_API_PORT', str(server.port))
    assert request(server, {'sql': 'SELECT * FROM ledger'})[0] == 200
    for sql in ('WITH c AS (SELECT 1) UPDATE ledger SET amount=99', 'SELECT * FROM group_insight_snapshots',
                'SELECT actual_completion_day AS done_on FROM research_projects'):
        status, body = request(server, {'sql': sql})
        assert status >= 400 and body['success'] is False
        with pytest.raises(_client.NovaMindAPIError): _client.query(sql)
    for body in ({}, {'sql': 1}, {'sql': None}, [], {'sql':' '}):
        assert request(server, body)[0] == 400
    assert _client.query('SELECT * FROM ledger')['rows'][0]['amount'] == 42
    shutil.copytree(Path(_client.__file__).parent, server.tools.workspace_path / 'docs' / 'novamind_api')
    server.set_daily_scripts({'query': "import novamind_api as nm\nprint(nm.query('SELECT amount FROM ledger'))"})
    assert '42' in server._run_daily_scripts_internal()['query']


def test_oracle_is_readonly_and_forbidden_in_formal(server, monkeypatch):
    server.oracle_mode = True
    assert execute_query(server, 'SELECT * FROM group_insight_snapshots')['row_count'] == 1
    assert execute_query(server, 'SELECT name FROM sqlite_master')['row_count'] > 0
    with pytest.raises(QueryDenied): execute_query(server, 'UPDATE ledger SET amount=99')
    monkeypatch.setattr('saas_bench.api_server._ORACLE_MODE', True)
    with pytest.raises(ValueError, match='oracle'):
        NovaMindAPIServer(server.tools, require_sandbox=True)
    monkeypatch.setenv('CEOBENCH_RUN_KIND', 'formal')
    with pytest.raises(ValueError, match='oracle'): NovaMindAPIServer(server.tools)


def test_lock_wait_is_bounded_and_snapshot_day_is_atomic(server, monkeypatch):
    entered, finish = threading.Event(), threading.Event()
    def change_world():
        with server._lock:
            entered.set(); finish.wait(2)
            server.conn.execute('UPDATE ledger SET day=7'); server.conn.commit()
            server.tools.set_current_day(7)
    worker = threading.Thread(target=change_world); worker.start(); assert entered.wait(2)
    monkeypatch.setattr(server, 'QUERY_TIMEOUT_SECONDS', 0.02)
    try:
        with pytest.raises(TimeoutError): execute_query(server, 'SELECT 1')
    finally: finish.set(); worker.join(2)
    with query_snapshot(server, time.monotonic()+5) as (conn, metadata):
        assert metadata['day'] == conn.execute('SELECT day FROM ledger').fetchone()[0] == 7


def test_queries_preserve_every_simulator_random_stream(make_initialized_sim, make_agent_tools, tmp_path):
    from saas_bench.config import ScenarioPack
    from saas_bench.shocks import ShockManager
    conn, sim, config = make_initialized_sim()
    sim.shock_manager = ShockManager(conn, sim.rng, ScenarioPack(name='test', description='test'))
    tools = make_agent_tools(conn, config)
    api = NovaMindAPIServer(tools, simulator=sim, conn=conn)
    sim.save_rng_states()
    before = conn.serialize()
    streams = [sim.rng, sim._macro_rng, sim._competitor_rng, sim._competitor_post_noise_rng,
               sim._competitor_template_rng, sim._quality_rng, sim._customer_quality_noise_rng,
               sim._customer_pick_rng, sim.shock_manager.rng, *sim._group_rngs.values()]
    states = json.dumps([r.bit_generator.state for r in streams], sort_keys=True)
    execute_query(api, 'SELECT * FROM subscriptions LIMIT 2')
    for sql in DENIED:
        with pytest.raises((QueryDenied, sqlite3.Error)): execute_query(api, sql)
    assert conn.serialize() == before
    assert json.dumps([r.bit_generator.state for r in streams], sort_keys=True) == states
    assert sim.current_day == tools.current_day == 0
    conn.close()


def test_cleanup_on_query_and_backup_failure(server, monkeypatch, tmp_path):
    import saas_bench.public_sql as policy
    import tempfile
    real_temp = tempfile.TemporaryDirectory
    directories = []
    def temporary(**kwargs):
        result = real_temp(dir=tmp_path, **kwargs)
        directories.append(Path(result.name))
        return result
    monkeypatch.setattr(policy.tempfile, 'TemporaryDirectory', temporary)
    with pytest.raises((QueryDenied, sqlite3.Error)):
        execute_query(server, 'SELECT * FROM group_insight_snapshots')
    source = server.conn
    def failed_backup(target, **kwargs):
        target.execute('CREATE TABLE partial(x)')
        raise OSError('injected backup failure')
    server.conn = SimpleNamespace(in_transaction=False, backup=failed_backup,
                                  total_changes=source.total_changes, execute=source.execute)
    try:
        with pytest.raises(OSError, match='backup failure'): execute_query(server, 'SELECT 1')
    finally: server.conn = source
    assert directories[0].exists() and not directories[1].exists()
    assert execute_query(server, 'SELECT 1')['success']
    server.stop()
    assert all(not d.exists() for d in directories)


def test_serialization_timeout_returns_504(server, monkeypatch):
    server.start()
    monkeypatch.setattr(server, 'QUERY_RESPONSE_TIMEOUT_SECONDS', 0)
    assert request(server, {'sql': 'SELECT 1'})[0] == 504
    monkeypatch.setattr(server, 'QUERY_RESPONSE_TIMEOUT_SECONDS', 30)
    assert request(server, {'sql': 'SELECT 1'})[0] == 200


def test_week_publishes_shocks_database_and_day_together(server):
    from concurrent.futures import ThreadPoolExecutor
    entered, query_started, finish = threading.Event(), threading.Event(), threading.Event()
    def shock(day):
        assert server._lock._is_owned()
        if day == 1:
            server.conn.execute('UPDATE ledger SET amount=43'); server.conn.commit()
            entered.set()
            assert finish.wait(3)
        return []
    def step():
        assert server._lock._is_owned()
        server.conn.execute('UPDATE ledger SET day=7'); server.conn.commit()
        return SimpleNamespace(day=7)
    server.shock_manager = SimpleNamespace(check_and_generate_shocks=shock, get_inbox_items=lambda _: [])
    server.simulator = SimpleNamespace(step_week=step)
    server.dashboard_callback = lambda *_: 'dashboard'
    def query():
        query_started.set()
        with query_snapshot(server, time.monotonic()+5) as (conn, metadata):
            return metadata['day'], tuple(conn.execute('SELECT day,amount FROM ledger').fetchone())
    with ThreadPoolExecutor(max_workers=2) as pool:
        advance = pool.submit(server.advance_week)
        assert entered.wait(3)
        reading = pool.submit(query)
        try:
            assert query_started.wait(3)
            assert not reading.done()
        finally: finish.set()
        assert advance.result(timeout=5)['success']
        assert reading.result(timeout=5) == (7, (7,43.0))


def test_quoted_hidden_column_is_not_a_string_constant(server):
    with pytest.raises(sqlite3.OperationalError, match='no such column'):
        execute_query(server, 'SELECT "actual_completion_day" FROM research_projects')


def test_send_timeout_does_not_send_a_second_response(monkeypatch):
    from saas_bench.api_server import _APIHandler
    statuses = []
    def timeout(_): raise TimeoutError('injected stalled client')
    handler = SimpleNamespace(
        server=SimpleNamespace(_api_server=SimpleNamespace(QUERY_RESPONSE_TIMEOUT_SECONDS=30)),
        connection=SimpleNamespace(settimeout=lambda _: None),
        send_response=statuses.append, send_header=lambda *_: None,
        end_headers=lambda: None, wfile=SimpleNamespace(write=timeout), close_connection=False,
        _capture_response=lambda *_: None, _capture_delivery=lambda *_: None)
    _APIHandler._send_query_json(handler, {'success':True})
    assert statuses == [200] and handler.close_connection
    # A closed log pipe must not turn a completed response into a second HTTP error.
    monkeypatch.setattr('builtins.print', lambda *args, **kwargs: timeout(None))
    _APIHandler._send_query_json(handler, {'success': True})
    assert statuses == [200, 200]


@pytest.mark.skipif(sys.platform != 'linux', reason='Actual snapshot isolation requires Linux bubblewrap')
def test_formal_sandbox_cannot_read_query_snapshot(server):
    import shlex
    from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
    with query_snapshot(server, time.monotonic()+5) as (conn, _):
        conn.set_authorizer(None)
        snapshot = Path(conn.execute('PRAGMA database_list').fetchone()[2])
        assert snapshot.exists()
        executor = BashAgentToolExecutor(server.script_workspace, require_sandbox=True)
        executor.verify_sandbox()
        output = executor.execute('bash', {'command': 'test ! -r ' + shlex.quote(str(snapshot)) + ' && echo snapshot-inaccessible'})
        assert output.strip() == 'snapshot-inaccessible'
