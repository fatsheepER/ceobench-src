from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import sys
import time
import uuid

from .database import TABLE_DOCS, SharedMemoryConnection


PUBLIC_POLICY_VERSION = 'public-sql-v1'
PUBLIC_COLUMNS = {table: tuple(doc['columns']) for table, doc in TABLE_DOCS.items()}
ROW_LIMIT = 5000
CELL_LIMIT_BYTES = 1024 * 1024
RESULT_LIMIT_BYTES = 8 * 1024 * 1024
RESPONSE_LIMIT_BYTES = 16 * 1024 * 1024
FORMATTING_NULL_LOSS = 'sqlite_formatting_null_requires_review'
FORMATTING_NULL_WARNING = (
    'A SQLite formatting function returned NULL. SQLite returns NULL for empty or invalid '
    'formats and when formatting exceeds the 1 MiB value limit. Check the format and narrow '
    'the value before using this result.'
)
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


class QueryResultTooLarge(Exception):
    pass


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Query time limit exceeded')


def record_query_failure(metadata, exc, stage, *, log=False):
    cause = exc
    while cause.__cause__ is not None:
        cause = cause.__cause__
    code = getattr(cause, 'sqlite_errorcode', None)
    base = code & 0xff if code is not None else None
    kind = 'service'
    if stage == 'execute':
        if isinstance(exc, QueryDenied) or base == sqlite3.SQLITE_AUTH or code == sqlite3.SQLITE_READONLY:
            kind = 'denied'
        elif isinstance(exc, TimeoutError) or base == sqlite3.SQLITE_INTERRUPT:
            kind = 'timeout'
        elif (base in (sqlite3.SQLITE_ERROR, sqlite3.SQLITE_TOOBIG)
              or isinstance(exc, (sqlite3.ProgrammingError, QueryResultTooLarge))):
            kind = 'query'
    metadata.update(snapshot_status='failed', error_type=type(exc).__name__, failure_stage=stage,
                    error_kind=kind, permanent_error=kind in ('query', 'denied'),
                    sqlite_errorcode=code, sqlite_errorname=getattr(cause, 'sqlite_errorname', None))
    if log:
        log_query(metadata)


def log_query(metadata):
    try:
        print('[public_sql] ' + json.dumps(metadata), file=sys.stderr, flush=True)
    except OSError:
        pass  # A closed diagnostic stream must not override the query result.


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
    """Read the live world on an independent connection while holding its lock."""
    started = time.monotonic()
    metadata = metadata if metadata is not None else {}
    metadata.update(snapshot_ref=uuid.uuid4().hex, public_policy_version=PUBLIC_POLICY_VERSION,
                    oracle=server.oracle_mode, snapshot_status='failed')
    stage = 'lock'
    try:
        # ponytail: serialize queries and world writes; use frozen readers if contention matters.
        if not server._lock.acquire(timeout=max(0, deadline - time.monotonic())):
            raise TimeoutError('Query lock wait exceeded time limit')
        try:
            stage = 'setup'
            metadata['lock_seconds'] = time.monotonic() - started
            check_deadline(deadline)
            if server.conn is None or server._operation_failed or server._step_day_timed_out:
                raise SnapshotUnavailable('World state unavailable for querying')
            if server.conn.in_transaction:
                raise SnapshotUnavailable('World has an unfinished transaction')
            before = time.monotonic()
            revision = [server.conn.total_changes,
                        server.conn.execute('PRAGMA data_version').fetchone()[0],
                        server.conn.execute('PRAGMA schema_version').fetchone()[0],
                        server.tools.current_day]
            check_deadline(deadline)
            if isinstance(server.conn, SharedMemoryConnection):
                uri = server.conn.query_uri
            elif isinstance(server.conn, sqlite3.Connection):
                filename = next(row[2] for row in server.conn.execute('PRAGMA database_list')
                                if row[1] == 'main')
                if not filename:
                    raise SnapshotUnavailable('World requires a named shared-memory connection')
                path = Path(filename).resolve()
                if path.is_relative_to(Path(server.script_workspace).resolve()):
                    raise SnapshotUnavailable('World database must be outside the agent workspace')
                uri = path.as_uri() + '?mode=ro'
            else:
                raise SnapshotUnavailable('World requires a standard SQLite connection')
            metadata.update(day=server.tools.current_day, snapshot_reused=False,
                            snapshot_bytes=0, world_revision=revision)
            conn = sqlite3.connect(uri, uri=True,
                                   timeout=max(0, deadline - time.monotonic()))
            formatter = None
            try:
                formatter = sqlite3.connect(':memory:')
                conn.row_factory = sqlite3.Row
                conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, CELL_LIMIT_BYTES)
                conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, CELL_LIMIT_BYTES)
                conn.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 128)
                formatter.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, CELL_LIMIT_BYTES)
                def bounded_format(*values):
                    if time.monotonic() >= deadline:
                        metadata['_format_timed_out'] = True
                        raise TimeoutError('Query time limit exceeded')
                    result = formatter.execute('SELECT printf(' + ','.join('?' for _ in values) + ')', values).fetchone()[0]
                    # SQLite also returns NULL on overflow, without SQLITE_TOOBIG.
                    if result is None and values and values[0] not in (None, '', b''):
                        metadata['_format_returned_null'] = True
                    return result
                for name in ('printf', 'format'):
                    conn.create_function(name, -1, bounded_format, deterministic=True)
                # Unknown quoted column names must fail, not become string literals.
                conn.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DML, False)
                conn.setconfig(sqlite3.SQLITE_DBCONFIG_DQS_DDL, False)
                if not server.oracle_mode:
                    for table, columns in PUBLIC_COLUMNS.items():
                        check_deadline(deadline)
                        schema = [row[1] for row in conn.execute(f'PRAGMA main.table_info("{table}")')]
                        if not set(columns).issubset(schema):
                            raise SnapshotUnavailable('World schema is missing public columns')
                        names = ', '.join('"' + c + '"' for c in schema if c in columns)
                        conn.execute(f'CREATE TEMP VIEW "{table}" AS SELECT {names} FROM main."{table}"')
                conn.execute('PRAGMA query_only=ON')
                denied = install_authorizer(conn, oracle=server.oracle_mode)
                conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
                check_deadline(deadline)
                metadata['snapshot_seconds'] = time.monotonic() - before
                before = time.monotonic()
                try:
                    stage = 'execute'
                    yield conn, metadata
                    metadata['snapshot_status'] = 'success'
                except sqlite3.Error as exc:
                    if metadata.pop('_format_timed_out', False):
                        raise TimeoutError('Query time limit exceeded') from exc
                    if denied:
                        raise QueryDenied('Query is not allowed by the read-only SQL policy. Read docs/tables/ for public columns.') from exc
                    raise
                finally:
                    metadata['sql_seconds'] = time.monotonic() - before
            finally:
                try:
                    try:
                        if formatter is not None:
                            formatter.close()
                    finally:
                        conn.close()
                except Exception:
                    stage = 'cleanup'
                    raise
        finally:
            server._lock.release()
    except Exception as exc:
        record_query_failure(metadata, exc, stage)
        raise
    finally:
        metadata['total_seconds'] = time.monotonic() - started
        log_query(metadata)


def execute_query(server, sql, *, metadata=None):
    metadata = metadata if metadata is not None else {}
    metadata['executed_sql'] = None
    deadline = metadata.pop('_request_deadline', time.monotonic() + server.QUERY_TIMEOUT_SECONDS)
    try:
        with query_snapshot(server, deadline, metadata) as (conn, _):
            return execute_snapshot(conn, sql, deadline, metadata)
    except sqlite3.Error as exc:
        code = getattr(exc, 'sqlite_errorcode', None) or 0
        if code & 0xff == sqlite3.SQLITE_AUTH or code == sqlite3.SQLITE_READONLY:
            raise QueryDenied('Query is not allowed by the read-only SQL policy. Read docs/tables/ for public columns.') from exc
        if code & 0xff == sqlite3.SQLITE_INTERRUPT:
            raise TimeoutError('Query time limit exceeded') from exc
        if code & 0xff == sqlite3.SQLITE_TOOBIG:
            raise QueryResultTooLarge(
                f'SQL value or row exceeded {CELL_LIMIT_BYTES} bytes. Narrow or aggregate the query.') from exc
        raise


def execute_snapshot(conn, sql, deadline, metadata):
    """Execute one original query on an already authorized snapshot."""
    check_deadline(deadline)
    metadata.update(attempted_sql=sql, executed_sql=None)
    cursor = conn.execute(sql)
    metadata['executed_sql'] = sql
    columns = [desc[0] for desc in cursor.description] if cursor.description else []
    raw, size = [], 0
    for row in cursor:
        check_deadline(deadline)
        for chunk in json.JSONEncoder(default=str).iterencode(dict(row)):
            check_deadline(deadline)
            size += len(chunk.encode())
            if size > RESULT_LIMIT_BYTES:
                raise QueryResultTooLarge(
                    f'SQL result exceeded {RESULT_LIMIT_BYTES} bytes. Narrow or aggregate the query.')
        raw.append(row)
        if len(raw) > ROW_LIMIT:
            break
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
    if metadata.pop('_format_returned_null', False):
        metadata['losses'].append(FORMATTING_NULL_LOSS)
        result['warning'] = (result.get('warning', '') + ' ' + FORMATTING_NULL_WARNING).strip()
    return result
