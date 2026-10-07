"""Harness-only replay of captured public reads. Never executes agent code."""
from contextlib import nullcontext
import json
import sqlite3
import time

from .execution_capture import READ_TOOLS, finish_http
from .public_sql import (PUBLIC_POLICY_VERSION, QueryDenied, SnapshotUnavailable,
                         check_deadline, execute_snapshot, install_authorizer, query_snapshot,
                         record_query_failure)


def replayable(event):
    request = event['request'].get('request') or {}
    return bool(event.get('query_definition') or
                (request.get('method') == 'GET' and request.get('path') == '/vars') or
                (request.get('method') == 'POST' and request.get('path') == '/call' and
                 (request.get('parsed') or {}).get('tool') in READ_TOOLS))


def refresh(server, versions, parent, *, store=None):
    store = store or server.sql_evidence
    if not store or server.oracle_mode:
        raise ValueError('Public evidence capture is required')
    if store.read_event(parent)['request']['kind'] not in ('pf_dependencies', 'pf_weekly_check'):
        raise ValueError('Refresh requires a forward query or weekly check event')
    sources = {}
    for version in dict.fromkeys(versions):
        meta, _ = store.public_content(version)
        event = store.read_event(meta['created_by_event'], evidence=True)
        definition = event['query_definition']
        if meta['layer'] != 'server_public_response' or not replayable(event):
            raise ValueError('Only captured SQL and approved public reads can be refreshed')
        if definition and (definition[4] != PUBLIC_POLICY_VERSION or definition[6] is not None):
            raise ValueError('Unsupported captured query policy or parameters')
        sources[version] = event
    events, result = {}, {}
    for version, source in sources.items():
        definition = source['query_definition']
        events[version] = (store.begin(json.dumps({'sql': definition[5]}).encode(), PUBLIC_POLICY_VERSION, parent)
                           if definition else store.begin_event('public_http', source['request']['request'], parent=parent))

    def save(version, status, raw, execution):
        event = events[version]
        if sources[version]['query_definition']:
            store.finish(event, status, '', raw, execution)
            store.delivered(event, 'internal')
        else:
            finish_http(store, event, status, '', raw, execution)
        result[version] = event + ':public_response'

    deadline = time.monotonic() + server.QUERY_TIMEOUT_SECONDS
    # ponytail: hold the world lock for this synchronous check; use frozen SDK readers
    # if measured business-action contention warrants releasing it earlier.
    locked = False
    snapshot = {}
    has_sql = any(e['query_definition'] for e in sources.values())
    stage = 'lock'
    try:
        locked = server._lock.acquire(timeout=max(0, deadline - time.monotonic()))
        if not locked:
            raise TimeoutError('Refresh lock wait exceeded time limit')
        stage = 'setup'
        check_deadline(deadline)
        if server._operation_failed or server._step_day_timed_out or server.conn.in_transaction:
            raise SnapshotUnavailable('World state unavailable for refresh')
        context = (query_snapshot(server, deadline, snapshot) if has_sql
                   else nullcontext((None, {'day': server.tools.current_day})))
        with context as (conn, snapshot):
            for version, source in sources.items():
                started = time.monotonic()
                deadline = started + server.QUERY_TIMEOUT_SECONDS
                definition = source['query_definition']
                request = source['request'].get('request')
                execution = dict(snapshot, snapshot_status='success', refresh_of=version, attempted=True)
                status = 200
                try:
                    check_deadline(deadline)
                    if definition:
                        # Reset the denial record for every statement in this shared snapshot.
                        denied = install_authorizer(conn)
                        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
                        try:
                            body = execute_snapshot(conn, definition[5], deadline, execution)
                        except sqlite3.Error as exc:
                            if denied:
                                raise QueryDenied('Query is not allowed by the read-only SQL policy') from exc
                            if (getattr(exc, 'sqlite_errorcode', None) or 0) & 0xff == sqlite3.SQLITE_INTERRUPT:
                                raise TimeoutError('Query time limit exceeded') from exc
                            raise
                        from .api_server import _get_enum_hint_for_query
                        hint = _get_enum_hint_for_query(definition[5], body['rows'])
                        if hint:
                            body['hint'] = hint
                    elif request['path'] == '/vars':
                        body = {'current_day': server.tools.current_day}
                    else:
                        parsed = request['parsed']
                        server.conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
                        try:
                            body = server.execute_tool(parsed['tool'], parsed.get('args', {})).to_json()
                        finally:
                            server.conn.set_progress_handler(None, 0)
                    check_deadline(deadline)
                    chunks = []
                    for chunk in json.JSONEncoder(default=str).iterencode(body):
                        check_deadline(deadline)
                        chunks.append(chunk)
                    raw = ''.join(chunks).encode()
                    check_deadline(deadline)
                except Exception as exc:
                    status = 403 if isinstance(exc, QueryDenied) else 504 if isinstance(exc, TimeoutError) else 500
                    # Detailed private diagnostics never become a public error string.
                    execution['refresh_error'] = type(exc).__name__
                    execution['refresh_error_detail'] = str(exc)
                    if definition:
                        record_query_failure(execution, exc, 'execute', log=True)
                    raw = json.dumps(dict(success=False, error='refresh_denied' if status == 403 else
                                          'refresh_timed_out' if status == 504 else 'refresh_failed')).encode()
                execution['refresh_seconds'] = time.monotonic() - started
                save(version, status, raw, execution)
        return result
    except (TimeoutError, SnapshotUnavailable) as exc:
        if has_sql and 'error_kind' not in snapshot:
            record_query_failure(snapshot, exc, stage, log=True)
        for version in sources:
            if version not in result:
                save(version, 504 if isinstance(exc, TimeoutError) else 503,
                     b'{"success":false,"error":"snapshot_unavailable"}',
                     dict(snapshot, refresh_of=version, refresh_error=type(exc).__name__, attempted=False))
        return result
    except Exception as exc:
        if has_sql and 'error_kind' not in snapshot:
            record_query_failure(snapshot, exc, stage, log=True)
        # A capture failure must stop collection; a failed read is recorded above.
        store.fail(exc)
        raise
    finally:
        if locked:
            server._lock.release()
