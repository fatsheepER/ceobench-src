"""Read-only SQL snapshots shared by public queries and future evidence replay."""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import uuid

from .database import TABLE_DOCS


PUBLIC_POLICY_VERSION = 'public-sql-v1'
PUBLIC_COLUMNS = {table: tuple(doc['columns']) for table, doc in TABLE_DOCS.items()}
ROW_LIMIT = 5000
# Built-ins only: no extension loading, connection metadata, or application UDFs.
FUNCTIONS = frozenset('''
abs coalesce ifnull nullif iif if max min round sign typeof unicode char
length octet_length lower upper trim ltrim rtrim substr substring instr replace
concat concat_ws format printf quote hex unhex like glob likelihood likely unlikely
count sum avg total group_concat string_agg
row_number rank dense_rank percent_rank cume_dist ntile lag lead first_value last_value nth_value
date time datetime julianday unixepoch strftime timediff current_date current_time current_timestamp
ceil ceiling floor trunc sqrt pow power exp ln log log10 log2 mod pi
acos acosh asin asinh atan atan2 atanh cos cosh sin sinh tan tanh degrees radians
json json_array json_array_length json_extract json_object json_type json_valid
json_quote json_group_array json_group_object json_insert json_replace json_set
json_remove json_patch json_error_position -> ->>
'''.split())


class QueryDenied(Exception):
    pass


class SnapshotUnavailable(Exception):
    pass


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Query time limit exceeded')


def install_authorizer(conn, *, oracle=False):
    """Install a fail-closed policy on a fresh connection with no application UDFs."""
    denied = []
    def authorize(action, table, column, database, source):
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION and column in FUNCTIONS:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            if oracle and database in ('main', 'temp', None):
                return sqlite3.SQLITE_OK
            # SQLite reports database=None for some column-free COUNT(*) reads.
            if table in PUBLIC_COLUMNS and (
                (column == '' and database in ('main', 'temp', None)) or
                (database in ('main', 'temp') and column in PUBLIC_COLUMNS[table])
            ):
                return sqlite3.SQLITE_OK
        denied.append(action)
        return sqlite3.SQLITE_DENY

    conn.set_authorizer(authorize)
    return denied


@contextmanager
def query_snapshot(server, deadline, metadata=None):
    """Copy the live world under its lock; never query the asynchronous disk save."""
    started = time.monotonic()
    metadata = metadata if metadata is not None else {}
    metadata.update(snapshot_ref=uuid.uuid4().hex, public_policy_version=PUBLIC_POLICY_VERSION,
                    oracle=server.oracle_mode, snapshot_status='failed')
    try:
        if not server._lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise TimeoutError('Query lock wait exceeded time limit')
        try:
            metadata['lock_seconds'] = time.monotonic() - started
            check_deadline(deadline)
            if server.conn is None or server._operation_failed or server._step_day_timed_out:
                raise SnapshotUnavailable('World state unavailable for querying')
            if server.conn.in_transaction:
                raise SnapshotUnavailable('World has an unfinished transaction')
            # total_changes includes writes followed by rollback; data_version also catches
            # commits from other connections. Schema changes and policy changes invalidate too.
            key = (server.conn, server.conn.total_changes,
                   server.conn.execute('PRAGMA data_version').fetchone()[0],
                   server.conn.execute('PRAGMA schema_version').fetchone()[0],
                   server.tools.current_day, PUBLIC_POLICY_VERSION, server.oracle_mode)
            snapshot = getattr(server, '_query_snapshot', None)
            reused = snapshot is not None and snapshot['key'] == key
            before = time.monotonic()
            if not reused:
                directory = tempfile.TemporaryDirectory(prefix='novamind-query-')
                path = Path(directory.name).resolve() / 'snapshot.db'
                if path.is_relative_to(Path(server.script_workspace).resolve()):
                    directory.cleanup()
                    raise SnapshotUnavailable('Query snapshots must be outside the agent workspace')
                target = sqlite3.connect(path)
                try:
                    server.conn.backup(target, pages=256,
                                       progress=lambda *_: check_deadline(deadline), sleep=0.01)
                except BaseException:
                    target.close()
                    directory.cleanup()
                    raise
                finally:
                    target.close()
                snapshot = dict(key=key, directory=directory, path=path, ref=uuid.uuid4().hex)
                server._query_snapshot = snapshot
            # Local reference keeps a retired snapshot alive until its last reader closes.
            path = snapshot['path']
            metadata.update(day=server.tools.current_day, snapshot_ref=snapshot['ref'],
                            snapshot_reused=reused, snapshot_seconds=time.monotonic() - before,
                            snapshot_bytes=path.stat().st_size,
                            world_revision=list(key[1:5]))
        finally:
            server._lock.release()
        conn = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True,
                               timeout=max(0, deadline - time.monotonic()))
        try:
            conn.row_factory = sqlite3.Row
            # Unknown quoted column names must fail, not become string literals.
            conn.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML, False)
            conn.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DDL, False)
            if not server.oracle_mode:
                for table, columns in PUBLIC_COLUMNS.items():
                    # Preserve SELECT * column order from the original schema.
                    schema = [row[1] for row in conn.execute(f'PRAGMA main.table_info("{table}")')]
                    if not set(columns).issubset(schema):
                        raise SnapshotUnavailable('World schema is missing public columns')
                    names = ', '.join('"' + c + '"' for c in schema if c in columns)
                    conn.execute(f'CREATE TEMP VIEW "{table}" AS SELECT {names} FROM main."{table}"')
            conn.execute('PRAGMA query_only=ON')
            denied = install_authorizer(conn, oracle=server.oracle_mode)
            conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            check_deadline(deadline)
            before = time.monotonic()
            try:
                yield conn, metadata
                metadata['snapshot_status'] = 'success'
            except sqlite3.Error as exc:
                if denied:
                    raise QueryDenied('Query is not allowed by the read-only SQL policy. Read docs/tables/ for public columns.') from exc
                raise
            finally:
                metadata['sql_seconds'] = time.monotonic() - before
        finally:
            conn.close()
    finally:
        metadata['total_seconds'] = time.monotonic() - started
        # Server diagnostics only; never add snapshot paths or metadata to public results.
        try:
            print('[public_sql] ' + json.dumps(metadata), file=sys.stderr, flush=True)
        except OSError:
            pass  # A closed diagnostic stream must not override the query result.


def execute_query(server, sql, *, metadata=None):
    metadata = metadata if metadata is not None else {}
    metadata['executed_sql'] = None
    deadline = time.monotonic() + server.QUERY_TIMEOUT_SECONDS
    try:
        with query_snapshot(server, deadline, metadata) as (conn, _):
            return execute_snapshot(conn, sql, deadline, metadata)
    except sqlite3.Error as exc:
        code = getattr(exc, 'sqlite_errorcode', None)
        if code in (sqlite3.SQLITE_AUTH, sqlite3.SQLITE_READONLY):
            raise QueryDenied('Query is not allowed by the read-only SQL policy. Read docs/tables/ for public columns.') from exc
        if code == sqlite3.SQLITE_INTERRUPT:
            raise TimeoutError('Query time limit exceeded') from exc
        raise


def execute_snapshot(conn, sql, deadline, metadata):
    """Execute one original query on an already authorized snapshot."""
    check_deadline(deadline)
    metadata.update(attempted_sql=sql, executed_sql=None)
    cursor = conn.execute(sql)
    metadata['executed_sql'] = sql
    columns = [desc[0] for desc in cursor.description] if cursor.description else []
    raw = cursor.fetchmany(ROW_LIMIT + 1)
    metadata['losses'] = ['blob_coercion'] if any(
        isinstance(value, bytes) for row in raw[:ROW_LIMIT] for value in row) else []
    check_deadline(deadline)
    result = {'success': True, 'columns': columns,
              'rows': [dict(row) for row in raw[:ROW_LIMIT]],
              'row_count': min(len(raw), ROW_LIMIT)}
    if len(raw) > ROW_LIMIT:
        result['truncated'] = True
        result['warning'] = (
            f'Result exceeded {ROW_LIMIT} rows and was truncated. '
            'Add a LIMIT clause to your query, or use COUNT/GROUP BY to '
            'aggregate results instead of fetching all rows.'
        )
    return result
