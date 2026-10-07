"""Public SQL policy tests use synthetic worlds; no provider calls."""
from contextlib import closing
import hashlib
import json
import shutil
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from saas_bench.api_server import NovaMindAPIServer
from saas_bench.database import SharedMemoryConnection, init_database
from saas_bench.public_sql import (
    PUBLIC_COLUMNS, PUBLIC_POLICY_VERSION, QueryDenied, SnapshotUnavailable,
    execute_query, execute_snapshot, install_authorizer, query_snapshot,
)


@pytest.fixture(params=['file', 'memory'])
def server(tmp_path, monkeypatch, request):
    monkeypatch.setattr('saas_bench.api_server._ORACLE_MODE', False)
    if request.param == 'memory':
        conn = init_database(':memory:')
        conn.execute('PRAGMA foreign_keys=OFF')
    else:
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


def test_cohort_payment_join_fits_bounded_sql_work(server):
    server.conn.executemany('''INSERT INTO subscriptions
        (customer_id,plan,listed_price,effective_price,start_day,status,billing_day_mod30)
        VALUES (?,'A',10,10,0,'subscribed',0)''', ((i,) for i in range(1, 1001)))
    server.conn.executemany('INSERT INTO ledger(day,category,amount,note) VALUES (?,?,?,?)',
        ((day, 'subscription_payment', 10 + day / 15, f'Subscription payment from customer {i}')
         for day in (0, 30, 60, 90, 120) for i in range(1, 1001)))
    server.conn.commit()
    sql = '''SELECT s.start_day,l.day AS payday,ROUND(AVG(l.amount),2) AS avg_amt,COUNT(*) AS n
        FROM subscriptions s JOIN ledger l
          ON l.note=('Subscription payment from customer ' || s.customer_id)
        WHERE s.plan='A' AND s.start_day BETWEEN 0 AND 7
          AND l.day IN (s.start_day,s.start_day+30)
        GROUP BY s.start_day,l.day ORDER BY s.start_day,l.day'''
    deadline = time.monotonic() + 5
    with query_snapshot(server, deadline) as (conn, metadata):
        instructions = 0
        def limit_work():
            nonlocal instructions
            instructions += 1000
            return instructions > 100_000
        conn.set_progress_handler(limit_work, 1000)
        result = execute_snapshot(conn, sql, deadline, metadata)
    assert result['rows'] == [dict(start_day=0, payday=0, avg_amt=10.0, n=1000),
                              dict(start_day=0, payday=30, avg_amt=12.0, n=1000)]


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


def test_fresh_reader_cleanup_and_readonly_independently(server):
    references = set()
    for amount in (43, 44):
        server.conn.execute('UPDATE ledger SET amount=?', (amount,))
        server.conn.commit()
        with query_snapshot(server, time.monotonic()+5) as (conn, metadata):
            assert conn is not server.conn
            assert metadata['day'] == 0 and metadata['snapshot_bytes'] == 0
            assert not metadata['snapshot_reused']
            references.add(metadata['snapshot_ref'])
            assert conn.execute('SELECT amount FROM ledger').fetchone()[0] == amount
            conn.set_authorizer(None)
            with pytest.raises(sqlite3.OperationalError, match='readonly'):
                conn.execute('UPDATE main.ledger SET amount=99')
            conn.execute('PRAGMA query_only=OFF')
            install_authorizer(conn)
            with pytest.raises(sqlite3.DatabaseError, match='authorized'):
                conn.execute('UPDATE main.ledger SET amount=99')
            if not isinstance(server.conn, SharedMemoryConnection):
                conn.set_authorizer(None)
                with pytest.raises(sqlite3.OperationalError, match='readonly'):
                    conn.execute('UPDATE main.ledger SET amount=99')
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            conn.execute('SELECT 1')
        with query_snapshot(server, time.monotonic()+5) as (_, again):
            assert not again['snapshot_reused']
            references.add(again['snapshot_ref'])
    assert len(references) == 4


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
    db_path = (server.conn.query_uri if isinstance(server.conn, SharedMemoryConnection) else
               server.conn.execute('PRAGMA database_list').fetchone()[2])
    with closing(sqlite3.connect(db_path, uri=True)) as independent:
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


@pytest.mark.parametrize('failure', ['query', 'schema', 'open', 'close'])
def test_reader_cleanup_and_unlock_on_failure(server, monkeypatch, failure):
    import saas_bench.public_sql as policy
    readers = []
    connect = sqlite3.connect
    class Reader(sqlite3.Connection):
        def close(self):
            assert server._lock._is_owned()
            super().close()
            if failure == 'close':
                raise OSError('injected close failure')
    def open_reader(*args, **kwargs):
        assert server._lock._is_owned()
        if failure == 'open':
            raise OSError('injected open failure')
        conn = connect(*args, **kwargs, factory=Reader)
        readers.append(conn)
        return conn
    monkeypatch.setattr(policy.sqlite3, 'connect', open_reader)
    if failure == 'schema':
        server.conn.execute('DROP TABLE research_projects')
        server.conn.commit()
    with pytest.raises((QueryDenied, SnapshotUnavailable, sqlite3.Error, OSError)):
        execute_query(server, 'SELECT * FROM group_insight_snapshots' if failure == 'query' else 'SELECT 1')
    assert not server._lock._is_owned()
    for conn in readers:
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            conn.execute('SELECT 1')
    with server._lock:
        server.conn.execute('UPDATE ledger SET amount=43')
        server.conn.commit()


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


@pytest.mark.skipif(sys.platform != 'linux', reason='Actual isolation requires Linux bubblewrap')
def test_formal_sandbox_cannot_read_world(server):
    import shlex
    from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
    with query_snapshot(server, time.monotonic()+5) as (conn, _):
        conn.set_authorizer(None)
        filename = conn.execute('PRAGMA database_list').fetchone()[2]
        if filename:
            command = 'test ! -r ' + shlex.quote(filename) + ' && echo world-inaccessible'
        else:
            probe = ('import sqlite3; c=sqlite3.connect(' + repr(server.conn.query_uri) +
                     ',uri=True); assert not c.execute("SELECT name FROM sqlite_master").fetchall(); '
                     'print("world-inaccessible")')
            command = 'python -c ' + shlex.quote(probe)
        executor = BashAgentToolExecutor(server.script_workspace, require_sandbox=True)
        executor.verify_sandbox()
        output = executor.execute('bash', {'command': command})
        assert output.strip() == 'world-inaccessible'


def test_world_connection_policy_and_temp_state_are_untouched(server):
    world = server.conn
    world.execute('CREATE TEMP TABLE ledger(amount)')
    world.execute('INSERT INTO temp.ledger VALUES(99)')
    world.commit()
    world.create_function('private_udf', 0, lambda: 73)
    world.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML, True)
    world.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DDL, True)
    factory = lambda cursor, row: tuple(row)
    world.row_factory = factory
    before = world.serialize(), world.serialize(name='temp')
    for sql in ('SELECT amount FROM ledger', 'SELECT private_udf()'):
        if 'private_udf' in sql:
            with pytest.raises(sqlite3.OperationalError, match='no such function'):
                execute_query(server, sql)
        else:
            assert execute_query(server, sql)['rows'] == [{'amount': 42.0}]
    assert (world.serialize(), world.serialize(name='temp')) == before
    assert world.row_factory is factory
    assert world.getconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML)
    assert world.getconfig(sqlite3.SQLITE_DBCONFIG_DQS_DDL)
    assert world.execute('SELECT amount,private_udf(),"missing" FROM ledger').fetchone() == (99, 73, 'missing')
    assert world.execute('PRAGMA query_only').fetchone() == (0,)


def test_world_lock_covers_fetch_and_reader_close(server, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    entered = threading.Event()
    def writer():
        entered.set()
        with server._lock:
            server.conn.execute('UPDATE ledger SET amount=43')
            server.conn.commit()
    connect = sqlite3.connect
    class Reader(sqlite3.Connection):
        def close(self):
            assert server._lock._is_owned()
            super().close()
    monkeypatch.setattr(sqlite3, 'connect', lambda *args, **kwargs: connect(*args, **kwargs, factory=Reader))
    with ThreadPoolExecutor(max_workers=1) as pool:
        with query_snapshot(server, time.monotonic()+5) as (conn, _):
            cursor = conn.execute('SELECT amount FROM ledger')
            writing = pool.submit(writer)
            assert entered.wait(2)
            assert not writing.done()
            assert cursor.fetchone()[0] == 42
        writing.result(timeout=2)
    assert execute_query(server, 'SELECT amount FROM ledger')['rows'] == [{'amount': 43.0}]


@pytest.mark.parametrize('kind', ['private_memory', 'workspace_file', 'workspace_symlink'])
def test_unsupported_worlds_fail_explicitly(server, tmp_path, monkeypatch, kind):
    if kind == 'private_memory':
        conn = sqlite3.connect(':memory:')
        message = 'named shared-memory'
    else:
        path = server.script_workspace / 'world.db'
        if kind == 'workspace_symlink':
            link = tmp_path / 'world-link.db'
            link.symlink_to(path)
            path = link
        conn = init_database(path)
        message = 'outside the agent workspace'
    with closing(conn):
        monkeypatch.setattr(server, 'conn', conn)
        with pytest.raises(SnapshotUnavailable, match=message):
            execute_query(server, 'SELECT 1')
        assert not server._lock._is_owned()


def test_encrypted_loads_have_independent_named_worlds(tmp_path, monkeypatch):
    from saas_bench.db_protection import load_session_db, save_session_db
    monkeypatch.setattr('saas_bench.db_protection._get_key', lambda: 'public-sql-test-key')
    path = tmp_path / 'world.nmdb'
    with closing(init_database(':memory:')) as initial:
        initial.execute("INSERT INTO ledger(day,category,amount) VALUES(0,'operations',42)")
        initial.commit()
        save_session_db(initial, path)
        with closing(init_database(':memory:')) as other:
            assert initial.query_uri != other.query_uri
            assert other.execute('SELECT count(*) FROM ledger').fetchone()[0] == 0
    original = path.read_bytes()
    with closing(load_session_db(path)) as first, closing(load_session_db(path)) as second:
        assert isinstance(first, SharedMemoryConnection) and first.query_uri != second.query_uri
        first.execute('UPDATE ledger SET amount=43')
        first.commit()
        api = NovaMindAPIServer(SimpleNamespace(workspace_path=tmp_path / 'agent', current_day=0), conn=first)
        assert execute_query(api, 'SELECT amount FROM ledger')['rows'] == [{'amount': 43.0}]
        assert second.execute('SELECT amount FROM ledger').fetchone()[0] == 42
        with closing(load_session_db(path, in_memory=False)) as encrypted:
            assert encrypted.execute('SELECT amount FROM ledger').fetchone()[0] == 42
    assert path.read_bytes() == original
    assert not list(tmp_path.glob('*.plain.tmp'))


@pytest.mark.parametrize('failure', ['export', 'destination', 'backup', 'pragma', 'analyze'])
def test_load_failure_closes_connections_and_removes_plaintext(tmp_path, monkeypatch, failure):
    import saas_bench.db_protection as protection
    monkeypatch.setattr(protection, '_get_key', lambda: 'test-key')
    readers, destinations = [], []
    connect = sqlite3.connect
    class Source(sqlite3.Connection):
        def backup(self, target, **kwargs):
            if failure == 'backup':
                raise RuntimeError('injected backup failure')
            return super().backup(target, **kwargs)
    class Destination(sqlite3.Connection):
        def execute(self, sql, *args):
            if (failure == 'pragma' and sql.startswith('PRAGMA')) or (failure == 'analyze' and sql == 'ANALYZE'):
                raise RuntimeError('injected configuration failure')
            return super().execute(sql, *args)
    def export(source, path, key):
        with closing(connect(path)) as conn:
            conn.execute('CREATE TABLE example(n)')
        if failure == 'export':
            raise RuntimeError('injected export failure')
    def open_source(*args, **kwargs):
        conn = connect(*args, **kwargs, factory=Source)
        readers.append(conn)
        return conn
    def open_destination():
        if failure == 'destination':
            raise RuntimeError('injected connection failure')
        conn = connect(':memory:', factory=Destination)
        destinations.append(conn)
        return conn
    monkeypatch.setattr(protection, '_export_encrypted_to_plain', export)
    monkeypatch.setattr(protection.sqlite3, 'connect', open_source)
    monkeypatch.setattr(protection, 'connect_shared_memory', open_destination)
    with pytest.raises(RuntimeError, match='injected'):
        protection.load_session_db(tmp_path / 'world.nmdb')
    assert not list(tmp_path.glob('*.plain.tmp'))
    assert len(readers) == (failure != 'export')
    assert len(destinations) == (failure not in ('export', 'destination'))
    for conn in readers + destinations:
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            conn.execute('SELECT 1')


@pytest.mark.skipif(sys.platform != 'linux', reason='Uses Linux process file-size limit')
def test_http_query_needs_no_world_copy(tmp_path):
    probe = '''
import json, resource, signal
from pathlib import Path
from types import SimpleNamespace
from saas_bench.api_server import NovaMindAPIServer
from saas_bench.database import init_database
from test_public_sql import request
world = init_database(':memory:')
api = NovaMindAPIServer(SimpleNamespace(workspace_path=Path.cwd(), current_day=0), conn=world)
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (4096, 4096))
api.start()
try:
    for _ in range(2):
        status, body = request(api, {'sql': 'SELECT 1 AS n'})
        assert status == 200, (status, body)
        assert body['rows'] == [{'n': 1}], body
    assert world.execute('SELECT 1').fetchone()[0] == 1
finally:
    api.stop()
    world.close()
'''
    import os
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('failure,stage,kind,code', [
    ('setup_io', 'setup', 'service', sqlite3.SQLITE_IOERR_WRITE),
    ('execute_io', 'execute', 'service', sqlite3.SQLITE_IOERR_WRITE),
    ('setup_syntax', 'setup', 'service', sqlite3.SQLITE_ERROR),
    ('setup_unexpected', 'setup', 'service', None),
    ('syntax', 'execute', 'query', sqlite3.SQLITE_ERROR),
    ('denied', 'execute', 'denied', sqlite3.SQLITE_AUTH),
    ('readonly', 'execute', 'denied', sqlite3.SQLITE_READONLY),
    ('readonly_lock', 'execute', 'service', sqlite3.SQLITE_READONLY_CANTLOCK),
    ('timeout', 'execute', 'timeout', sqlite3.SQLITE_INTERRUPT),
    ('unavailable', 'setup', 'service', None),
])
def test_http_failure_diagnostics_preserve_extended_codes(server, monkeypatch, capfd,
                                                        failure, stage, kind, code):
    connect = sqlite3.connect
    class Reader(sqlite3.Connection):
        def execute(self, sql, *args):
            if ((failure in ('execute_io', 'readonly', 'readonly_lock') and sql == 'SELECT 1 AS n') or
                    (failure == 'setup_syntax' and sql.startswith('PRAGMA main.table_info'))):
                error = sqlite3.OperationalError('injected SQL failure')
                error.sqlite_errorcode = code
                error.sqlite_errorname = {sqlite3.SQLITE_IOERR_WRITE: 'SQLITE_IOERR_WRITE',
                                          sqlite3.SQLITE_READONLY: 'SQLITE_READONLY',
                                          sqlite3.SQLITE_READONLY_CANTLOCK: 'SQLITE_READONLY_CANTLOCK'}.get(code, 'SQLITE_ERROR')
                raise error
            return super().execute(sql, *args)
    def open_reader(*args, **kwargs):
        if failure == 'setup_io':
            error = sqlite3.OperationalError('disk I/O error')
            error.sqlite_errorcode, error.sqlite_errorname = sqlite3.SQLITE_IOERR_WRITE, 'SQLITE_IOERR_WRITE'
            raise error
        if failure == 'setup_unexpected':
            raise OSError('injected reader failure')
        return connect(*args, **kwargs, factory=Reader)
    monkeypatch.setattr(sqlite3, 'connect', open_reader)
    server._operation_failed = failure == 'unavailable'
    server.QUERY_TIMEOUT_SECONDS = .02 if failure == 'timeout' else 5
    sql = {'syntax': 'SELECT FROM ledger', 'denied': 'SELECT * FROM group_insight_snapshots',
           'timeout': 'WITH RECURSIVE x(n) AS (VALUES(1) UNION ALL SELECT n+1 FROM x) SELECT sum(n) FROM x'}.get(failure, 'SELECT 1 AS n')
    server.start()
    status, body = request(server, {'sql': sql})
    assert status >= 400 and not body['success']
    assert 'sqlite_errorcode' not in body and 'SQLITE_IOERR_WRITE' not in json.dumps(body)
    diagnostics = [json.loads(line.removeprefix('[public_sql] '))
                   for line in capfd.readouterr().err.splitlines() if line.startswith('[public_sql] ')]
    record, = diagnostics
    assert record['failure_stage'] == stage and record['error_kind'] == kind
    assert record['sqlite_errorcode'] == code
    assert record['permanent_error'] is (kind in ('query', 'denied'))
    if code == sqlite3.SQLITE_IOERR_WRITE:
        assert record['sqlite_errorname'] == 'SQLITE_IOERR_WRITE'
    if code == sqlite3.SQLITE_READONLY_CANTLOCK:
        assert status == 500 and record['sqlite_errorname'] == 'SQLITE_READONLY_CANTLOCK'


@pytest.mark.parametrize('failure', ['setup', 'setup_io', 'execute'])
def test_pf_refresh_service_failure_has_private_diagnostics(server, tmp_path, monkeypatch, capfd, failure):
    from saas_bench.pf_refresh import refresh
    from saas_bench.sql_evidence import FORMAT, SQLEvidenceStore
    from test_pf_stale import record_sql
    store = SQLEvidenceStore(tmp_path / 'evidence.sqlite', dict(format=FORMAT, run_id='test', branch_id='test',
                                                            data_source_id='test'))
    server.sql_evidence = store
    version = record_sql(store, 'SELECT 1 AS n', dict(success=True, columns=['n'], rows=[{'n': 1}], row_count=1))
    parent = store.begin_event('pf_weekly_check', {'day': 0})
    if failure == 'setup':
        server._operation_failed = True
    else:
        connect = sqlite3.connect
        class Reader(sqlite3.Connection):
            def execute(self, sql, *args):
                if sql == 'SELECT 1 AS n':
                    error = sqlite3.OperationalError('disk I/O error')
                    error.sqlite_errorcode, error.sqlite_errorname = sqlite3.SQLITE_IOERR_WRITE, 'SQLITE_IOERR_WRITE'
                    raise error
                return super().execute(sql, *args)
        def open_reader(*args, **kwargs):
            if failure == 'setup_io' and kwargs.get('uri'):
                error = sqlite3.OperationalError('disk I/O error')
                error.sqlite_errorcode, error.sqlite_errorname = sqlite3.SQLITE_IOERR_WRITE, 'SQLITE_IOERR_WRITE'
                raise error
            return connect(*args, **kwargs, factory=Reader)
        monkeypatch.setattr(sqlite3, 'connect', open_reader)
    if failure == 'setup_io':
        with pytest.raises(sqlite3.OperationalError, match='disk I/O error'):
            refresh(server, [version], parent)
    else:
        result = refresh(server, [version], parent)
        meta, raw = store.get_content(result[version])
        assert 'sqlite_errorcode' not in json.loads(raw)
        captured = store.read_event(meta['created_by_event'])['result']
        assert captured['error_kind'] == 'service' and captured['failure_stage'] == failure
    diagnostics = [json.loads(line.removeprefix('[public_sql] '))
                   for line in capfd.readouterr().err.splitlines() if line.startswith('[public_sql] ')]
    failed, = [row for row in diagnostics if row.get('error_kind') == 'service']
    if failure == 'setup_io':
        assert failed['failure_stage'] == 'setup' and failed['sqlite_errorcode'] == sqlite3.SQLITE_IOERR_WRITE
    if failure == 'execute':
        assert failed['sqlite_errorcode'] == captured['sqlite_errorcode'] == sqlite3.SQLITE_IOERR_WRITE
        assert failed['sqlite_errorname'] == captured['sqlite_errorname'] == 'SQLITE_IOERR_WRITE'
