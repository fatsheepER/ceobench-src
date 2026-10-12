"""HTTP JSON-RPC server for NovaMind API.

Bridges the novamind_api Python library (running in a subprocess) to the
AgentTools instance (running in the main runner process). Communication
is via HTTP on localhost with a random OS-assigned port.
"""

import json
import math
import os
import re
import sqlite3
import socket
import select
import sys
import threading
import time
import traceback
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

# Oracle debug reads expose internal data through the same read-only SQL policy.
# Formal runs reject this startup setting.
_ORACLE_MODE: bool = os.environ.get("ORACLE_MODE") == "1"

READ_ONLY_TASK_TOOLS = frozenset(('get_social_posts', 'get_cost_info', 'get_market_overview',
    'get_group_insights', 'list_research_projects', 'list_all_tables', 'describe_tables',
    'get_tool_documentation', 'list_daily_calculations', 'list_scripts'))

from .role_policy import CALL_POLICY, ROLES, require_call, require_advance, require_role
from .tools import AgentTools, ToolResult
from .database import TABLE_DOCS
from .environment import build_weekly_dashboard


from .public_sql import (PUBLIC_COLUMNS, RESPONSE_LIMIT_BYTES, QueryDenied,
                         QueryResultTooLarge, SnapshotUnavailable, execute_query)


class RequestRejected(ValueError):
    pass


class RequestDeadlineExceeded(TimeoutError):
    pass

_TABLE_COLUMNS = {table: list(columns) for table, columns in PUBLIC_COLUMNS.items()}

# Build column→valid_values mapping for enum hint messages.
# Parses TABLE_DOCS column descriptions for patterns like "'val1', 'val2', 'val3'"
_COLUMN_ENUM_VALUES: Dict[str, Dict[str, List[str]]] = {}  # table -> {col -> [values]}
for _tname, _tinfo in TABLE_DOCS.items():
    for _col, _desc in _tinfo.get('columns', {}).items():
        # Skip descriptions with "e.g." — those are examples, not exhaustive enums
        if 'e.g.' in _desc.lower():
            continue
        # Extract quoted enum values from descriptions like "TEXT — 'lead', 'subscribed', 'cancelled', 'lost'"
        _vals = re.findall(r"'([^']+)'", _desc)
        if len(_vals) >= 2:  # Only treat as enum if 2+ values found
            _COLUMN_ENUM_VALUES.setdefault(_tname, {})[_col] = _vals


def _get_enum_hint_for_query(sql: str, rows: List[Dict]) -> Optional[str]:
    """If a query returned 0 rows and uses string comparisons on enum columns,
    return a hint about valid values. Returns None if no hint is applicable."""
    if rows:  # Only hint on empty results
        return None

    sql_lower = sql.lower()

    # Find table aliases: "FROM tablename alias" or "JOIN tablename alias" or "tablename AS alias"
    alias_map: Dict[str, str] = {}  # alias -> table_name
    for table_name in _COLUMN_ENUM_VALUES:
        # Match: tablename alias (no AS), tablename AS alias
        for m in re.finditer(
            r'\b' + re.escape(table_name) + r'\s+(?:as\s+)?(\w+)',
            sql_lower
        ):
            alias = m.group(1)
            # Skip SQL keywords that might follow table name
            if alias not in ('on', 'where', 'set', 'join', 'inner', 'left', 'right',
                             'outer', 'cross', 'group', 'order', 'having', 'limit',
                             'union', 'except', 'intersect', 'and', 'or', 'not',
                             'select', 'from', 'as', 'natural', 'using'):
                alias_map[alias] = table_name
        # Also match bare table name (no alias)
        if re.search(r'\b' + re.escape(table_name) + r'\b', sql_lower):
            alias_map[table_name] = table_name

    if not alias_map:
        return None

    # Find string comparisons: col = "val", col = 'val', alias.col = "val", alias.col = 'val'
    hints = []
    for m in re.finditer(r"(\w+)\.(\w+)\s*=\s*[\"']([^\"']+)[\"']", sql_lower):
        prefix, col, val = m.group(1), m.group(2), m.group(3)
        table = alias_map.get(prefix)
        if table and table in _COLUMN_ENUM_VALUES:
            enum_vals = _COLUMN_ENUM_VALUES[table].get(col)
            if enum_vals and val not in enum_vals:
                hints.append(
                    f"'{val}' is not a valid value for {table}.{col}. "
                    f"Valid values: {', '.join(repr(v) for v in enum_vals)}"
                )

    # Also match unqualified: col = "val"
    for m in re.finditer(r"(?<!\.)(\w+)\s*=\s*[\"']([^\"']+)[\"']", sql_lower):
        col, val = m.group(1), m.group(2)
        # Check if this is a prefix.col pattern (already handled above)
        start = m.start()
        if start > 0 and sql_lower[start - 1] == '.':
            continue
        # Find which tables in the query have this column with enum values
        for alias, table in alias_map.items():
            if table in _COLUMN_ENUM_VALUES:
                enum_vals = _COLUMN_ENUM_VALUES[table].get(col)
                if enum_vals and val not in enum_vals:
                    hints.append(
                        f"'{val}' is not a valid value for {table}.{col}. "
                        f"Valid values: {', '.join(repr(v) for v in enum_vals)}"
                    )

    # Deduplicate
    seen = set()
    unique_hints = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            unique_hints.append(h)

    if unique_hints:
        return "Note: " + "; ".join(unique_hints)
    return None


def _get_helpful_query_error(error: Exception, sql: str) -> str:
    """Generate a helpful error message for SQL errors, including column hints."""
    err_str = str(error).lower()

    if 'no such column' in err_str:
        match = re.search(r'no such column: ([\w.]+)', str(error))
        if match:
            bad_col = match.group(1)
            # Find tables referenced in the query
            sql_lower = sql.lower()
            matched_tables = {}
            for table_name, cols in _TABLE_COLUMNS.items():
                if re.search(r'\b' + re.escape(table_name) + r'\b', sql_lower):
                    matched_tables[table_name] = cols
            if matched_tables:
                hints = []
                for tname, cols in matched_tables.items():
                    hints.append(f"  {tname}: {', '.join(cols)}")
                return (
                    f"no such column: {bad_col}. "
                    f"Valid columns for tables in your query:\n"
                    + "\n".join(hints)
                )
            return f"no such column: {bad_col}. Use describe_tables() or read docs/tables/ to check column names."

    if 'no such table' in err_str:
        match = re.search(r'no such table: (\w+)', str(error))
        if match:
            bad_table = match.group(1)
            valid = sorted(_TABLE_COLUMNS.keys())
            return f"no such table: {bad_table}. Valid tables: {', '.join(valid)}"

    if 'ambiguous column name' in err_str:
        match = re.search(r'ambiguous column name: (\w+)', str(error))
        if match:
            col = match.group(1)
            # Find which tables have this column
            tables_with_col = [t for t, cols in _TABLE_COLUMNS.items() if col in cols]
            return (
                f"ambiguous column name: {col}. "
                f"This column exists in: {', '.join(tables_with_col)}. "
                f"Use table aliases (e.g. t.{col}) to disambiguate."
            )

    return str(error)


from .execution_capture import public_handler


class _APIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the NovaMind API server."""

    @property
    def role(self):
        return self.server._role

    @property
    def evidence(self):
        return self.server._api_server.evidence_for(self.role)

    def handle_one_request(self):
        api = self.server._api_server
        with api._request_lock:
            api._active_requests[id(self.connection)] = dict(read_only=False, method=None, path=None)
        for name in ('_request_deadline', '_request_body', '_request_body_complete'):
            self.__dict__.pop(name, None)
        self._start_public_request()
        timer = threading.Timer(max(0, self._request_deadline - time.monotonic()),
                                self._expire_request_input)
        timer.daemon = True
        self._request_input_timer = timer
        timer.start()
        try:
            return super().handle_one_request()
        finally:
            timer.cancel()

    def parse_request(self):
        parsed = super().parse_request()
        if parsed:
            api = self.server._api_server
            read_only = (self.command == 'POST' and self.path in ('/query', '/_capture') or
                self.command == 'GET' and self.path in ('/health', '/game-status', '/daily-scripts', '/vars'))
            with api._request_lock:
                api._active_requests[id(self.connection)] = dict(
                    read_only=read_only, method=self.command, path=self.path)
                if not read_only and self.path != '/call':
                    api._request_mutations += 1
        return parsed

    def _expire_request_input(self):
        try:
            self.connection.shutdown(socket.SHUT_RD)
        except OSError:
            pass

    def _start_public_request(self):
        if not hasattr(self, '_request_deadline'):
            self._request_deadline = time.monotonic() + self.server._api_server.QUERY_TIMEOUT_SECONDS
            self._request_body = bytearray()

    def _read_request_bytes(self):
        if getattr(self, '_request_body_complete', False):
            return self._request_body
        try:
            length = int(self.headers.get('Content-Length', 0))
        except (ValueError, TypeError) as exc:
            raise RequestRejected('Invalid Content-Length') from exc
        limit = 64 * 1024 * 1024 if self.path == '/_capture' else 2 * 1024 * 1024
        if not 0 <= length <= limit or self.headers.get('Transfer-Encoding'):
            raise RequestRejected(f'Request body must have Content-Length between 0 and {limit} bytes')
        while len(self._request_body) < length:
            remaining = self._request_deadline - time.monotonic()
            if remaining <= 0:
                raise RequestDeadlineExceeded('Request input exceeded its time limit')
            self.connection.settimeout(remaining)
            try:
                part = self.rfile.read1(min(65536, length - len(self._request_body)))
            except TimeoutError as exc:
                raise RequestDeadlineExceeded('Request input exceeded its time limit') from exc
            if not part:
                if time.monotonic() >= self._request_deadline:
                    raise RequestDeadlineExceeded('Request input exceeded its time limit')
                raise RequestRejected('Incomplete request body')
            self._request_body.extend(part)
        if time.monotonic() >= self._request_deadline:
            raise RequestDeadlineExceeded('Request input exceeded its time limit')
        self._request_body = bytes(self._request_body)
        self._request_body_complete = True
        if timer := getattr(self, '_request_input_timer', None):
            timer.cancel()
        return self._request_body

    def _admit_sql(self):
        api = self.server._api_server
        with api._sql_lock:
            while api._sql_paused:
                store = self.evidence
                if store:
                    try:
                        if store.parent(self.headers.get('X-Capture-Context')):
                            break
                    except ValueError:
                        pass
                remaining = self._request_deadline - time.monotonic()
                if remaining <= 0:
                    raise RequestDeadlineExceeded('Request admission exceeded its time limit')
                api._sql_lock.wait(timeout=remaining)
            api._sql_active += 1

    def _reject_request(self, exc):
        self.close_connection = True
        self._sql_execution = dict(permanent_error=isinstance(exc, RequestRejected),
                                   failure_stage='request', error_kind='timeout' if
                                   isinstance(exc, RequestDeadlineExceeded) else 'query')
        self._sql_event, self._sql_capture_failed = None, False
        self._generic_capture = self.path != '/query'
        store = self.evidence
        if store and self.path != '/_capture':
            try:
                parent = store.parent(self.headers.get('X-Capture-Context'), active=False)
                if self.path == '/query':
                    from .public_sql import PUBLIC_POLICY_VERSION
                    self._sql_event = store.begin(bytes(self._request_body), PUBLIC_POLICY_VERSION, parent=parent)
                else:
                    from .execution_capture import capture_http
                    self._sql_event = capture_http(self, bytes(self._request_body), active_parent=False)
                if client := self.headers.get('X-Capture-Call'):
                    store.bind_client(self._sql_event, client)
            except (ValueError, KeyError):
                pass
            except Exception as error:
                self._sql_capture_failed = True
                store.fail(error)
        self.connection.settimeout(self.server._api_server.QUERY_RESPONSE_TIMEOUT_SECONDS)
        try:
            self._send_json({'success': False, 'error': str(exc)},
                            504 if isinstance(exc, RequestDeadlineExceeded) else 400)
        except OSError:
            pass

    def admit(self):
        api = self.server._api_server
        controls = ('/checkpoint', '/pf-refresh', '/run-metrics')
        if api.team_mode and ((self.role is None and self.path not in controls) or
                              (self.role is not None and self.path in controls)):
            self._send_json({'success': False, 'error': 'Host-bound access required'}, 403)
            return False
        return True

    # Suppress default logging to stderr
    def log_message(self, format, *args):
        pass

    @public_handler
    def do_POST(self):
        try:
            if (self.server._api_server.read_only_task and
                    self.path not in ('/call', '/query', '/checkpoint', '/pf-refresh', '/run-metrics')):
                self._send_json({'success': False, 'error': 'Decision preparation cannot change the world'}, 403)
                return
            if self.path == '/call':
                self._handle_call()
            elif self.path == '/next-week':
                self._handle_next_week()
            elif self.path == '/query':
                self._handle_query()
            elif self.path == '/daily-scripts':
                self._handle_daily_scripts_post()
            elif self.path in ('/checkpoint', '/pf-refresh', '/run-metrics'):
                import secrets
                expected_token = self.server._api_server.checkpoint_token
                supplied_token = self.headers.get('X-Harness-Token', '')
                if not expected_token or not secrets.compare_digest(supplied_token, expected_token):
                    self._send_json({'error': 'Harness access required'}, 403)
                    return
                body = self._read_body()
                if self.path == '/run-metrics':
                    api = self.server._api_server
                    self._send_json({'day': api.tools.current_day,
                                     'usage': api.simulator.customer_simulator.usage_recorder.summary})
                    return
                if self.path == '/pf-refresh':
                    if (set(body) not in ({'versions', 'parent'}, {'versions', 'parent', 'remaining_seconds'})
                            or not isinstance(body.get('parent'), str) or
                            not isinstance(body['versions'], list) or
                            not all(isinstance(v, str) for v in body['versions']) or
                            ('remaining_seconds' in body and
                             (type(body['remaining_seconds']) not in (int, float) or
                              not math.isfinite(body['remaining_seconds']) or body['remaining_seconds'] <= 0))):
                        self._send_json({'error': 'versions and parent are required'}, 400)
                        return
                    from .pf_refresh import refresh
                    from .pf_deadline import budget_scope
                    try:
                        with budget_scope(timeout=min(body.get('remaining_seconds', self.server._api_server.QUERY_TIMEOUT_SECONDS),
                                                      self.server._api_server.QUERY_TIMEOUT_SECONDS)):
                            result = refresh(self.server._api_server, body['versions'], body['parent'])
                    except (ValueError, KeyError, TimeoutError, SnapshotUnavailable) as exc:
                        self._send_json({'error': type(exc).__name__}, 400)
                        return
                    self._send_json({'versions': result})
                    return
                if set(body) != {'expected_day'} or not isinstance(body['expected_day'], int):
                    self._send_json({'success': False, 'error': 'expected_day is required; no other fields allowed'}, 400)
                    return
                self._send_json(self.server._api_server.checkpoint(body['expected_day']))
            else:
                self._send_json({"error": f"Unknown endpoint: {self.path}"}, 404)
        except (RequestRejected, RequestDeadlineExceeded):
            raise
        except Exception as exc:
            self._send_internal_error(exc, op=f"POST {self.path}")

    @public_handler
    def do_GET(self):
        try:
            if self.path == '/vars':
                self._handle_vars()
            elif self.path == '/health':
                evidence = self.evidence
                self._send_json({"status": "capture_failed" if evidence and
                                 (evidence.fault or evidence.fault_path.exists()) else "ok"})
            elif self.path == '/daily-scripts':
                self._handle_daily_scripts_get()
            elif self.path == '/dashboard':
                self._handle_dashboard_get()
            elif self.path == '/game-status':
                self._handle_game_status()
            else:
                self._send_json({"error": f"Unknown endpoint: {self.path}"}, 404)
        except Exception as exc:
            self._send_internal_error(exc, op=f"GET {self.path}")

    @public_handler
    def do_DELETE(self):
        try:
            if self.server._api_server.read_only_task:
                self._send_json({'success': False, 'error': 'Decision preparation cannot change the world'}, 403)
                return
            if self.path == '/daily-scripts':
                self._handle_daily_scripts_delete()
            else:
                self._send_json({"error": f"Unknown endpoint: {self.path}"}, 404)
        except Exception as exc:
            self._send_internal_error(exc, op=f"DELETE {self.path}")

    def _send_internal_error(self, exc: BaseException, *, op: str, status: int = 500) -> None:
        """Centralized 5xx response.

        The full traceback is written to the server log (stderr) so operators
        can debug, but the agent only ever sees a stable
        ``{"error": "internal_error", "request_id": ...}`` payload — no
        exception type, no message, no traceback, no source paths. This is
        the choke point that prevents engine internals from leaking into
        the agent's tool-result stream the way they did before.
        """
        request_id = uuid.uuid4().hex[:12]
        try:
            tb = traceback.format_exc()
        except Exception:
            tb = "<traceback unavailable>"
        try:
            print(
                f"[api_server] internal_error op={op} request_id={request_id}\n{tb}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass
        if self.path == '/query' and getattr(self, '_query_response_started', False):
            self.close_connection = True
            return
        try:
            self._send_json(
                {
                    "success": False,
                    "error": "internal_error",
                    "request_id": request_id,
                    "data": None,
                },
                status,
            )
        except Exception:
            # Connection may already be torn down; nothing useful to do.
            pass

    def _read_body(self) -> Dict:
        body = self._read_request_bytes()
        if getattr(self, '_control_capture', False):
            self._control_request = body.hex()
        return json.loads(body) if body else {}

    def _send_json(self, data: Dict, status: int = 200):
        if self.path == '/query':
            return self._send_query_json(data, status)
        if getattr(self, '_generic_capture', False):
            from .execution_capture import text_sources
            self._sql_execution['public_fields'] = text_sources(data)
            if self.path == '/next-week' and self.server._api_server._step_day_timed_out:
                self._sql_execution['world_outcome_unknown'] = True
        response = json.dumps(data, default=str).encode()
        self._capture_response(status, response)
        try:
            if self.path == '/query':
                self._query_response_started = True
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        except OSError:
            self._capture_delivery('failed')
            raise
        else:
            self._capture_delivery('sent')

    def _capture_sql(self, method, *args):
        store = self.evidence
        if store is None or getattr(self, '_sql_capture_failed', False):
            return
        try:
            if method == 'finish' and getattr(self, '_generic_capture', False):
                from .execution_capture import finish_http
                return finish_http(store, *args)
            return getattr(store, method)(*args)
        except Exception as exc:
            self._sql_capture_failed = True
            store.fail(exc)

    def _capture_response(self, status, response):
        self._audit_status = status
        if getattr(self, '_control_capture', False):
            self._control_record = dict(path=self.path, method=self.command, status=status, body_hex=response.hex(), request_hex=getattr(self, '_control_request', ''))
        if getattr(self, '_sql_event', None):
            self._sql_response = status, response

    def end_headers(self):
        if hasattr(self, '_control_record'):
            self._control_record['headers'] = (b''.join(self._headers_buffer) + b'\r\n').decode('latin-1')
        if getattr(self, '_sql_event', None) and getattr(self, '_sql_response', None):
            status, body = self._sql_response
            self._sql_response = None
            # Includes BaseHTTPRequestHandler's actual Server/Date/status line too.
            headers = (b''.join(self._headers_buffer) + b'\r\n').decode('latin-1')
            self._capture_sql('finish', self._sql_event, status, headers, body, self._sql_execution)
        super().end_headers()

    def _capture_delivery(self, state):
        if hasattr(self, '_control_record'):
            self._control_record['send_state'] = state
        if getattr(self, '_sql_event', None):
            self._capture_sql('delivered', self._sql_event, state)

    def _handle_call(self):
        """Handle a tool call: POST /call {"tool": "...", "args": {...}}."""
        try:
            body = self._read_body()
            tool_name = body.get('tool', '')
            self._audit_tool = tool_name
            args = body.get('args', {})

            server: NovaMindAPIServer = self.server._api_server
            with server._request_lock:
                read_only = CALL_POLICY.get(tool_name) == 'read'
                server._active_requests[id(self.connection)]['read_only'] = read_only
                if not read_only:
                    server._request_mutations += 1
            if server.read_only_task and tool_name not in READ_ONLY_TASK_TOOLS:
                self._send_json({'success': False, 'error': 'Decision preparation cannot change the world'}, 403)
                return
            result = server.execute_tool(tool_name, args, role=self.role)

            if isinstance(result, ToolResult):
                self._send_json(result.to_json())
            else:
                # Fallback for non-ToolResult returns
                self._send_json({"success": True, "data": {"output": str(result)}, "message": str(result)})
        except (RequestRejected, RequestDeadlineExceeded):
            raise
        except PermissionError as e:
            self._send_json({"success": False, "error": str(e)}, 403)
        except Exception as e:
            self._send_internal_error(e, op="call")

    def _handle_next_week(self):
        """Handle next-week advancement: POST /next-week.

        Body must contain:
        - ``rationale`` (string, non-empty): the agent's strategic reasoning
          for this week's actions. Replaces the old standalone log_rationale
          tool. Stored via event_logger as tool_name='log_rationale'.
        - ``predictions`` with one entry per horizon
          (``cash_1wk``, ``cash_4wk``, ``cash_12wk``, ``cash_26wk`` for
          +7/+28/+84/+182 days). Each entry must be an object with three
          numeric fields: ``point``, ``lower``, ``upper`` — the agent's point
          estimate plus the 95% CI lower and upper bounds.

        Missing/empty rationale, missing prediction keys, non-numeric values,
        or ``lower > upper`` / ``point`` outside ``[lower, upper]`` return 400.
        """
        try:
            server: NovaMindAPIServer = self.server._api_server
            try:
                require_advance(self.role)
                with server._lock:
                    server.authorize_team(self.role, 'advance')
            except PermissionError as e:
                self._send_json({"success": False, "error": str(e)}, 403)
                return
            body = self._read_body() or {}
            rationale = body.get("rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                self._send_json({
                    "success": False,
                    "error": "Missing 'rationale'. Required: a non-empty string capturing your strategic reasoning for this week's actions.",
                }, 400)
                return
            preds_raw = body.get("predictions")
            horizon_map = {"cash_1wk": 7, "cash_4wk": 28, "cash_12wk": 84, "cash_26wk": 182}
            required_keys = ", ".join(horizon_map.keys())
            if not isinstance(preds_raw, dict):
                self._send_json({
                    "success": False,
                    "error": f"Missing 'predictions' object. Required keys: {required_keys}. Each must be an object {{point, lower, upper}} (95% CI bounds in dollars).",
                }, 400)
                return

            parsed = {}
            for key, horizon in horizon_map.items():
                if key not in preds_raw:
                    self._send_json({
                        "success": False,
                        "error": f"Missing prediction '{key}'. Required keys: {required_keys}.",
                    }, 400)
                    return
                entry = preds_raw[key]
                if not isinstance(entry, dict):
                    self._send_json({
                        "success": False,
                        "error": f"Prediction '{key}' must be an object with fields 'point', 'lower', 'upper' (got {type(entry).__name__}).",
                    }, 400)
                    return
                try:
                    point = float(entry["point"])
                    lower = float(entry["lower"])
                    upper = float(entry["upper"])
                except KeyError as ke:
                    self._send_json({
                        "success": False,
                        "error": f"Prediction '{key}' missing field {ke}. Required: point, lower, upper.",
                    }, 400)
                    return
                except (TypeError, ValueError):
                    self._send_json({
                        "success": False,
                        "error": f"Prediction '{key}' fields point/lower/upper must all be numbers, got {entry!r}.",
                    }, 400)
                    return
                if not all(math.isfinite(v) for v in (point, lower, upper)):
                    self._send_json({'success': False, 'error': f"Prediction '{key}' must contain finite numbers."}, 400)
                    return
                if lower > upper:
                    self._send_json({
                        "success": False,
                        "error": f"Prediction '{key}': lower ({lower}) must be <= upper ({upper}).",
                    }, 400)
                    return
                if point < lower or point > upper:
                    self._send_json({
                        "success": False,
                        "error": f"Prediction '{key}': point ({point}) must satisfy lower <= point <= upper (got [{lower}, {upper}]).",
                    }, 400)
                    return
                parsed[horizon] = {"cash": {"point": point, "lower": lower, "upper": upper}}

            result = server.advance_week(predictions=parsed, rationale=rationale, role=self.role)
            self._send_json(result)
        except (RequestRejected, RequestDeadlineExceeded):
            raise
        except PermissionError as e:
            self._send_json({'success': False, 'error': str(e)}, 403)
        except Exception as e:
            self._send_internal_error(e, op="next-week")

    def _handle_query(self):
        api = self.server._api_server
        if self.evidence and self.evidence.execution_capture:
            self._query_response_started = False
            return self._handle_query_request()
        self._admit_sql()
        try:
            self._query_response_started = False
            self._sql_event = None
            self._sql_response = None
            self._sql_capture_failed = False
            self._sql_execution = {}
            self._handle_query_request()
        except (RequestRejected, RequestDeadlineExceeded) as exc:
            self._reject_request(exc)
        finally:
            self._sql_event = None
            with api._sql_lock:
                api._sql_active -= 1
                api._sql_lock.notify_all()

    def _handle_query_request(self):
        sql = ''
        try:
            raw = self._read_request_bytes()
            from .public_sql import PUBLIC_POLICY_VERSION
            if not (self.evidence and self.evidence.execution_capture):
                self._sql_event = self._capture_sql('begin', raw, PUBLIC_POLICY_VERSION)
            body = json.loads(raw) if raw else {}
            if not isinstance(body, dict) or not isinstance(body.get('sql'), str) or not body['sql'].strip():
                self._send_json({'success': False, 'error': 'sql must be a non-empty string'}, 400)
                return
            sql = body['sql']
            self._sql_execution['_request_deadline'] = self._request_deadline
            response = execute_query(self.server._api_server, sql, metadata=self._sql_execution)
            enum_hint = _get_enum_hint_for_query(sql, response['rows'])
            if enum_hint:
                response['hint'] = enum_hint
            self._send_query_json(response)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({'success': False, 'error': 'Invalid JSON body'}, 400)
        except (RequestRejected, RequestDeadlineExceeded):
            raise
        except QueryResultTooLarge as exc:
            self._sql_execution['permanent_error'] = True
            self._send_json({'success': False, 'error': str(exc)}, 413)
        except QueryDenied as exc:
            self._sql_execution.setdefault('permanent_error', True)
            self._send_json({'success': False, 'error': str(exc)}, 403)
        except SnapshotUnavailable as exc:
            self._send_json({'success': False, 'error': str(exc)}, 503)
        except TimeoutError:
            if not self._query_response_started:
                self._send_json({'success': False, 'error': 'Query exceeded its time limit. Narrow the query or try again when the world is idle.'}, 504)
        except sqlite3.Error as exc:
            self._sql_execution['permanent_error'] = self._sql_execution.get('permanent_error',
                (getattr(exc, 'sqlite_errorcode', None) or 0) & 0xff == sqlite3.SQLITE_ERROR)
            self._send_json({'success': False, 'error': _get_helpful_query_error(exc, sql)}, 500)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            if not self._query_response_started:
                self._send_internal_error(exc, op='query')

    def _send_query_json(self, data, status=200):
        """Bound serialization and socket writes separately from SQL execution."""
        import time
        from .public_sql import check_deadline
        started = time.monotonic()
        chunks = []
        size = 0
        response_budget = self.server._api_server.QUERY_RESPONSE_TIMEOUT_SECONDS
        deadline = started + (max(.05, response_budget) if status >= 400 else response_budget)
        for chunk in json.JSONEncoder(default=str).iterencode(data):
            check_deadline(deadline)
            payload = chunk.encode()
            size += len(payload)
            if size > RESPONSE_LIMIT_BYTES:
                raise QueryResultTooLarge(f'SQL response exceeded {RESPONSE_LIMIT_BYTES} bytes. Narrow the query.')
            chunks.append(payload)
        response = b''.join(chunks)
        check_deadline(deadline)
        capture_started = time.monotonic()
        self._capture_response(status, response)
        # Capture has its own accounting, never spend SQL/response execution budgets on it.
        deadline += time.monotonic() - capture_started
        serialized = time.monotonic()
        self.connection.settimeout(max(0.001, deadline - serialized))
        try:
            self._query_response_started = True
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            self._capture_delivery('sent')
        except TimeoutError:
            # Headers have already been sent; never append a second HTTP response.
            self.close_connection = True
            self._capture_delivery('failed')
        except OSError:
            self._capture_delivery('failed')
            raise
        finally:
            try:
                print('[public_sql_response] ' + json.dumps({
                    'serialization_seconds': serialized - started,
                    'send_seconds': time.monotonic() - serialized,
                }), file=sys.stderr, flush=True)
            except OSError:
                pass

    def _handle_daily_scripts_post(self):
        """Register a daily script snapshot: POST /daily-scripts {"name": "x.py", "content": "..."}."""
        try:
            body = self._read_body()
            name = body.get('name', '')
            content = body.get('content', '')
            if not isinstance(name, str) or not name or not isinstance(content, str):
                self._send_json({"success": False, "error": "name and content must be strings; name cannot be empty"}, 400)
                return
            server: NovaMindAPIServer = self.server._api_server
            with server._lock:
                scripts = server.get_daily_scripts(self.role)
                scripts[name] = content
                server.set_daily_scripts(scripts, registration=name, role=self.role)
            import hashlib
            self._send_json({"success": True, "data": {"name": name, "registered": True,
                "sha256": hashlib.sha256(content.encode()).hexdigest()}})
        except PermissionError as e:
            self._send_json({'success': False, 'error': str(e)}, 403)
        except Exception as e:
            self._send_internal_error(e, op="daily-scripts:post")

    def _handle_daily_scripts_get(self):
        """List registered daily scripts: GET /daily-scripts."""
        server: NovaMindAPIServer = self.server._api_server
        with server._lock:
            import hashlib
            scripts = [{"name": n, "size": len(c), "sha256": hashlib.sha256(c.encode()).hexdigest()}
                       for n, c in server.get_daily_scripts(self.role).items()]
        self._send_json({"success": True, "data": {"scripts": scripts}})

    def _handle_daily_scripts_delete(self):
        """Remove a daily script: DELETE /daily-scripts {"name": "x.py"}."""
        try:
            body = self._read_body()
            name = body.get('name', '')
            server: NovaMindAPIServer = self.server._api_server
            with server._lock:
                server.authorize_team(self.role, 'scripts')
                if name not in server.get_daily_scripts(self.role):
                    self._send_json({"success": False, "error": f"Script not found: {name}"}, 404)
                    return
                scripts = server.get_daily_scripts(self.role)
                del scripts[name]
                server.set_daily_scripts(scripts, role=self.role)
            self._send_json({"success": True, "data": {"removed": name}})
        except PermissionError as e:
            self._send_json({'success': False, 'error': str(e)}, 403)
        except Exception as e:
            self._send_internal_error(e, op="daily-scripts:delete")

    def _handle_vars(self):
        """Handle variable queries: GET /vars."""
        server: NovaMindAPIServer = self.server._api_server
        self._send_json({
            "current_day": server.tools.current_day,
        })

    def _handle_dashboard_get(self):
        """Return current dashboard: GET /dashboard.

        Returns the last built dashboard (from advance_week), or builds
        a fresh one for the current day if none exists yet.
        """
        server: NovaMindAPIServer = self.server._api_server
        dashboard = server._last_dashboard
        if not dashboard and server.conn:
            day = server.tools.current_day
            dashboard = build_weekly_dashboard(server.conn, day)
            from .execution_capture import dashboard_version
            dashboard = dashboard_version(server, dashboard, day, role=self.role)
        if server.team_mode:
            from .execution_capture import dashboard_version
            dashboard = dashboard_version(server, str(dashboard), server.tools.current_day, role=self.role)
        self._send_json({
            "dashboard": dashboard or f"=== Day {server.tools.current_day} ===\n(No data)",
            "day": server.tools.current_day,
        })

    def _handle_game_status(self):
        """Return simulation state for harness: GET /game-status.

        Returns day, cash, subscriber count, and timeout flag.
        """
        from .database import get_cash, get_active_subscriber_count
        server: NovaMindAPIServer = self.server._api_server
        cash = 0
        subs = 0
        if server.conn and not (server._advance_lock.locked() or server._operation_failed or server._step_day_timed_out):
            cash = get_cash(server.conn)
            subs = get_active_subscriber_count(server.conn)
        self._send_json({
            "day": server.tools.current_day,
            "cash": cash,
            "subscribers": subs,
            "timed_out": server._step_day_timed_out,
            "operation_failed": server._operation_failed,
            "operation_in_progress": server._advance_lock.locked(),
        })


# Map tool names to AgentTools methods + argument extraction
_TOOL_DISPATCH = {
    'set_prices': lambda tools, args: tools.set_prices({k: v for k, v in args.items() if v is not None}),
    'set_model_tiers': lambda tools, args: tools.set_model_tiers({k: v for k, v in args.items() if v is not None}),
    'set_daily_spend': lambda tools, args: tools.set_daily_spend({k: v for k, v in args.items() if v is not None}),
    'set_targeted_ad_spend': lambda tools, args: tools.set_targeted_ad_spend(args.get('targeted_spend', args)),
    'set_capacity_tier': lambda tools, args: tools.set_capacity_tier(args.get('tier', args.get('capacity_tier', 0))),
    'set_usage_quotas': lambda tools, args: tools.set_usage_quotas(args),
    'send_enterprise_deal': lambda tools, args: tools.send_enterprise_deal(deals=args.get('deals', [])),
    'reject_enterprise_deal': lambda tools, args: tools.reject_enterprise_deal(deals=args.get('deals', [])),
    'get_social_posts': lambda tools, args: tools.get_social_posts(args.get('days', 7), args.get('limit', 50)),
    'post_social_media': lambda tools, args: tools.post_social_media(args.get('content', ''), args.get('reply_to_post_id')),
    'get_cost_info': lambda tools, args: tools.get_cost_info(),
    'start_research_project': lambda tools, args: tools.start_research_project(args.get('tier', args.get('project_id', ''))),
    'list_research_projects': lambda tools, args: tools.list_research_projects(),
    'research_market': lambda tools, args: tools.research_market(),
    'research_group': lambda tools, args: tools.research_group(args.get('group_id', ''), args.get('target_level')),
    'get_market_overview': lambda tools, args: tools.get_market_overview(),
    'get_group_insights': lambda tools, args: tools.get_group_insights(args.get('group_id', '')),
    'set_targeted_ops_spend': lambda tools, args: tools.set_targeted_ops_spend(
        targeted_spend=args.get('targeted_spend', None if any(key in args for key in
            ('by_group', 'by_plan', 'by_group_plan', 'by_customer')) else args),
        by_group=args.get('by_group'),
        by_plan=args.get('by_plan'),
        by_group_plan=args.get('by_group_plan'),
        by_customer=args.get('by_customer'),
    ),
    'set_targeted_dev_spend': lambda tools, args: tools.set_targeted_dev_spend(args.get('targeted_spend', args)),
    'set_ads_strength': lambda tools, args: tools.set_ads_strength(
        global_strength=args.get('global_strength'),
        by_group=args.get('by_group'),
        by_customer=args.get('by_customer'),
    ),
    'set_lead_promotion': lambda tools, args: tools.set_lead_promotion(
        global_promotion=args.get('global_promotion'),
        by_group=args.get('by_group'),
        by_channel=args.get('by_channel'),
        by_channel_group=args.get('by_channel_group'),
    ),
    'set_promotion': lambda tools, args: tools.set_promotion(
        global_promotion=args.get('global_promotion'),
        by_group=args.get('by_group'),
        by_customer=args.get('by_customer'),
        by_group_plan=args.get('by_group_plan'),
    ),
}


assert set(CALL_POLICY) == set(_TOOL_DISPATCH), 'Every public tool needs an explicit role policy'


class _TrackedHTTPServer(ThreadingHTTPServer):
    def get_request(self):
        api = self._api_server
        # Account for accepted sockets before a handler thread can be delayed.
        with api._request_lock:
            request, address = super().get_request()
            api._active_requests[id(request)] = dict(read_only=False, method=None, path=None)
        return request, address

    def shutdown_request(self, request):
        try:
            super().shutdown_request(request)
        finally:
            api = self._api_server
            with api._request_lock:
                api._active_requests.pop(id(request), None)
                api._request_lock.notify_all()


def registered_script_classification(facts, error=None):
    import errno
    boundary = facts.get('process_boundary') or {}
    resource_errno = {errno.ENOSPC, errno.ENOMEM, errno.EMFILE, errno.ENFILE, errno.EAGAIN, errno.EDQUOT}
    if (facts.get('error_errno') in resource_errno or boundary.get('spawn_errno') in resource_errno or
            facts.get('timed_out') or facts.get('cancelled') or
            facts.get('exit_code') == -9 or boundary.get('exit_code') == -9):
        return 'transient_infrastructure'
    if facts.get('process_started') is False or facts.get('target_started') is not True and not error:
        return 'transient_infrastructure'
    if facts.get('boundary_closed') is not True or error:
        return 'unknown'
    if facts.get('target_started') is not True:
        return 'transient_infrastructure'
    if facts.get('exit_code') == 0:
        return 'succeeded'
    return 'script_error' if facts.get('exit_code') is not None else 'unknown'


class _UnixHTTPServer(_TrackedHTTPServer):
    address_family = socket.AF_UNIX

    def server_bind(self):
        from .runtime_temp import bind_unix
        bind_unix(self.socket, self.server_address)
        self.server_name, self.server_port = 'localhost', 0


class NovaMindAPIServer:
    """HTTP API server wrapping AgentTools for subprocess communication.

    Usage:
        server = NovaMindAPIServer(tools, simulator, conn)
        server.start()  # Starts in background thread
        port = server.port  # OS-assigned port
        ...
        server.stop()
    """

    def __init__(self, tools: AgentTools, simulator=None, conn=None,
                 day_callback=None, dashboard_callback=None,
                 shock_manager=None, event_logger=None, script_workspace=None,
                 require_sandbox=False, sql_evidence=None, checkpoint_token=None,
                 role_sockets=None, role_stores=None, role_executors=None, role_audits=None):
        """Initialize the API server.

        Args:
            tools: AgentTools instance to dispatch calls to
            simulator: Simulator instance for next-week advancement
            conn: Database connection for dashboard building
            day_callback: Optional callback(day, dashboard) called after advancing a day
            dashboard_callback: Optional callback(day) -> dashboard string
            shock_manager: Optional ShockManager for generating shocks each day
            event_logger: Optional EventLogger for logging events
        """
        self.oracle_mode = _ORACLE_MODE
        self.read_only_task = os.environ.get('CEOBENCH_READ_ONLY_TASK') == '1'
        self.role_sockets = dict(role_sockets or {})
        self.team_mode = bool(self.role_sockets)
        self.team_phase = None
        self.script_check = None
        if self.team_mode and set(self.role_sockets) != set(ROLES):
            raise ValueError('Team requires exactly three role listeners')
        self.role_stores = role_stores or {}
        self.role_audits = role_audits or {}
        self.role_executors = role_executors or {}
        self._role_httpds = []
        self.world_state_id = uuid.uuid4().hex
        self.sql_evidence = sql_evidence
        if sql_evidence is not None:
            if self.oracle_mode:
                raise ValueError('Oracle cannot write public SQL evidence')
            from pathlib import Path
            if sql_evidence.path.is_relative_to(Path(script_workspace or tools.workspace_path).resolve()):
                raise ValueError('SQL evidence must be outside the agent workspace')
            sql_evidence.assert_healthy()
        if self.oracle_mode and (require_sandbox or os.environ.get('CEOBENCH_RUN_KIND') == 'formal'):
            raise ValueError('Formal runs cannot enable oracle mode')
        self.tools = tools
        self.simulator = simulator
        self.conn = conn
        self.day_callback = day_callback
        self.dashboard_callback = dashboard_callback
        self.shock_manager = shock_manager
        self.event_logger = event_logger
        self._httpd: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.port: int = 0
        self._lock = threading.RLock()
        self._sql_lock = threading.Condition()
        self._request_lock = threading.Condition()
        self._active_requests = {}
        self._request_mutations = 0
        self._sql_active = 0
        self._sql_paused = False
        self._advance_lock = threading.Lock()
        self._operation_failed = False
        self.checkpoint_callback = None
        self.checkpoint_token = checkpoint_token
        self._last_dashboard: str = ""
        self._last_day_result = None
        self._daily_scripts: Dict[str, str] = {}  # name -> content snapshot
        self._role_scripts = {role: {} for role in ROLES}
        self._role_scripts["ceo"] = self._daily_scripts
        self.script_workspace = script_workspace or tools.workspace_path
        self.require_sandbox = require_sandbox
        self.last_script_results = []
        self.role_script_results = {role: [] for role in ROLES}
        if conn is not None:
            conn.execute('CREATE TABLE IF NOT EXISTS _registered_scripts (position INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, content TEXT NOT NULL, sha256 TEXT NOT NULL)')
            saved_scripts = conn.execute('SELECT name, content, sha256 FROM _registered_scripts ORDER BY position').fetchall()
            import hashlib
            for name, content, checksum in saved_scripts:
                if hashlib.sha256(content.encode()).hexdigest() != checksum:
                    raise ValueError('Registered script checksum mismatch: ' + name)
                self._daily_scripts[name] = content
        if self.team_mode and conn is not None:
            conn.execute('CREATE TABLE IF NOT EXISTS _team_scripts (role TEXT NOT NULL, position INTEGER NOT NULL, name TEXT NOT NULL, content TEXT NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY(role,name))')
            for role, name, content, checksum in conn.execute('SELECT role,name,content,sha256 FROM _team_scripts ORDER BY position'):
                require_role(role)
                if hashlib.sha256(content.encode()).hexdigest() != checksum:
                    raise ValueError('Registered script checksum mismatch: ' + name)
                self._role_scripts[role][name] = content
        self._step_day_timed_out: bool = False  # Set when step_day exceeds timeout

    def start(self):
        """Start the HTTP server in a background thread."""
        self._httpd = _TrackedHTTPServer(('127.0.0.1', 0), _APIHandler)
        self._httpd._api_server = self
        self._httpd._role = None if self.team_mode else "ceo"
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        for role, path in self.role_sockets.items():
            from pathlib import Path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            httpd = _UnixHTTPServer(str(path), _APIHandler)
            httpd._api_server, httpd._role = self, role
            self._role_httpds.append(httpd)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def stop(self):
        """Stop the HTTP server."""
        for httpd in self._role_httpds:
            httpd.shutdown()
            httpd.server_close()
            os.unlink(httpd.server_address)
        self._role_httpds.clear()
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def public_request_state(self):
        with self._request_lock:
            listeners = [server.socket for server in [self._httpd, *self._role_httpds] if server]
            return dict(active=list(self._active_requests.values()),
                queued=bool(listeners and select.select(listeners, [], [], 0)[0]),
                mutation_epoch=self._request_mutations)

    def evidence_for(self, role):
        return self.role_stores.get(role) if self.team_mode else self.sql_evidence

    def authorize_team(self, role, operation):
        if not self.team_mode or self.team_phase is None:
            return
        require_role(role)
        phase = self.team_phase
        if operation == 'read':
            return
        allowed = ('ceo_scripts', 'ceo') if role == 'ceo' else ('analysis', 'asking')
        if phase not in allowed or (operation == 'advance' and phase != 'ceo'):
            raise PermissionError(f'{role} {operation} is frozen during {phase}')

    def public_dashboard(self):
        with self._lock:
            day = self.tools.current_day
            if self.dashboard_callback:
                dashboard = self.dashboard_callback(day, self._last_day_result)
            elif self.conn:
                dashboard = build_weekly_dashboard(self.conn, day, self._last_day_result)
            else:
                dashboard = f'=== Day {day} Dashboard ===\n(No dashboard data available)'
            from .execution_capture import dashboard_version
            self._last_dashboard = dashboard_version(self, dashboard, day)
            return self._last_dashboard

    def execute_tool(self, tool_name: str, args: Dict[str, Any], *, role='ceo') -> Any:
        """Execute a tool call with thread safety."""
        require_call(role, tool_name)
        with self._lock:
            self.authorize_team(role, CALL_POLICY.get(tool_name, 'write'))
            dispatch_fn = _TOOL_DISPATCH.get(tool_name)
            if dispatch_fn is None:
                return ToolResult(False, f"Unknown tool: {tool_name}")
            result = dispatch_fn(self.tools, args)
            if CALL_POLICY[tool_name] != 'read' and result.success:
                self.world_state_id = uuid.uuid4().hex
                if self.team_mode and self.team_phase is not None and self.conn is not None:
                    from .database import get_cash
                    if get_cash(self.conn) < 0:
                        self.team_phase = 'boundary'
            return result

    # Maximum allowed time for step_week before auto-quit (seconds)
    STEP_WEEK_TIMEOUT = 4200  # 7× longer than old per-day timeout

    # Lock wait, reader setup and SQL share this deadline. The world lock stays
    # held until the separate query connection closes.
    QUERY_TIMEOUT_SECONDS = 120
    QUERY_RESPONSE_TIMEOUT_SECONDS = 30

    def checkpoint(self, expected_day):
        if not self._advance_lock.acquire(blocking=False):
            raise RuntimeError('Cannot checkpoint an in-flight week')
        try:
            with self._sql_lock:
                self._sql_paused = True
                try:
                    if self.sql_evidence and self.sql_evidence.execution_capture:
                        self.sql_evidence.save_state('admission_paused', True)
                    if not self._sql_lock.wait_for(lambda: self._sql_active == 0, timeout=180):
                        raise TimeoutError('SQL capture did not finish before checkpoint')
                    if self.sql_evidence is not None and self.sql_evidence.execution_capture:
                        import time
                        deadline = time.monotonic() + 180
                        while True:
                            self.sql_evidence.assert_healthy(quiescent=False)
                            try:
                                self.sql_evidence.assert_healthy()
                                if self._sql_active == 0:
                                    break
                            except RuntimeError as exc:
                                if (not str(exc).startswith('Unconfirmed') or
                                        str(exc).startswith('Unconfirmed execution outcome')):
                                    raise
                            if time.monotonic() >= deadline:
                                raise TimeoutError('Execution capture did not finish before checkpoint')
                            self._sql_lock.wait(timeout=0.05)
                    # No query can hold the world lock while this checkpoint owns it.
                    with self._lock:
                        if self.sql_evidence is not None:
                            self.sql_evidence.assert_healthy()
                        if self._operation_failed or self._step_day_timed_out:
                            raise RuntimeError('Operation outcome unknown; checkpoint refused')
                        if expected_day != self.tools.current_day:
                            raise ValueError('Checkpoint day mismatch')
                        if self.checkpoint_callback is None:
                            raise RuntimeError('Harness checkpoint directory is not configured')
                        return self.checkpoint_callback()
                finally:
                    try:
                        if self.sql_evidence and self.sql_evidence.execution_capture:
                            self.sql_evidence.save_state('admission_paused', False)
                    finally:
                        self._sql_paused = False
                        self._sql_lock.notify_all()
        finally:
            self._advance_lock.release()

    def advance_week(self, predictions=None, rationale=None, *, role="ceo"):
        require_advance(role)
        with self._lock:
            self.authorize_team(role, 'advance')
        if not self._advance_lock.acquire(blocking=False):
            raise RuntimeError('Week advancement already in progress')
        try:
            if self._operation_failed or self._step_day_timed_out:
                raise RuntimeError('Previous operation outcome unknown; continuation refused')
            return self._advance_week(predictions, rationale)
        except PermissionError:
            raise
        except Exception:
            self._operation_failed = True
            raise
        finally:
            self._advance_lock.release()

    def _advance_week(self, predictions: Optional[Dict[int, Dict[str, float]]] = None,
                     rationale: Optional[str] = None) -> Dict[str, Any]:
        """Advance the simulator by one week (7 days) and return the dashboard.

        Enforces a hard timeout (STEP_WEEK_TIMEOUT seconds) on step_week().
        If exceeded, returns an error so the runner can save checkpoint and exit.

        If a shock_manager is configured, shocks are checked before step_week
        and inbox items are included in the dashboard.

        ``predictions`` (optional): maps horizon_days -> {metric: value}. Saved
        to the ``predictions`` table before advancing. Used by the prediction
        benchmark component.

        ``rationale`` (optional at the Python level, required at the HTTP layer
        — see _handle_next_week): the agent's strategic reasoning for this
        week's actions. Logged via event_logger with tool_name='log_rationale'
        for analysis (preserves the old standalone log_rationale storage shape).
        """
        import time as _time
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

        with self._lock:
            self.authorize_team('ceo', 'advance')
            if self.simulator is None:
                return {"success": False, "error": "No simulator configured"}
            old_day = self.tools.current_day

        # Log rationale BEFORE stepping so it's attributed to the day the
        # decisions were made, not the post-step day.
        if rationale and self.event_logger:
            try:
                self.event_logger.log_agent_action(
                    tool_name='log_rationale',
                    arguments={'rationale': rationale},
                    result={'logged': True},
                    success=True,
                )
            except Exception:
                import traceback
                print(traceback.format_exc(), file=sys.stderr, flush=True)

        # Persist predictions before stepping the world (so submit_day reflects
        # the day the prediction was made, not the post-step day).
        if predictions and self.conn is None:
            return {'success': False, 'error': 'prediction_save_failed', 'day': old_day}
        if predictions and self.conn is not None:
            from saas_bench.database import save_predictions as _save_predictions
            _pred_exc_tb = None
            try:
                with self._lock:
                    self.conn.execute('SAVEPOINT week_predictions')
                    try:
                        _save_predictions(self.conn, old_day, predictions, _time.time())
                        self.conn.execute('RELEASE week_predictions')
                        self.conn.commit()
                    except Exception:
                        if self.conn.in_transaction:
                            self.conn.rollback()
                        raise
            except Exception:
                import traceback
                _pred_exc_tb = traceback.format_exc()
            if _pred_exc_tb is not None:
                # Log AFTER releasing the lock so a buffered stderr write can't
                # back-pressure the lock holder. (run 27c000a5 d105 hang.)
                print(_pred_exc_tb, file=sys.stderr, flush=True)
                return {'success': False, 'error': 'prediction_save_failed', 'day': old_day}

        # Run step_week in a worker thread so we can enforce a timeout.
        _step_start = _time.monotonic()

        def _do_step():
            with self._lock:
                try:
                    # Publish the new world and day together, including shocks.
                    if self.shock_manager:
                        for d in range(old_day + 1, old_day + 8):
                            new_shocks = self.shock_manager.check_and_generate_shocks(d)
                            if self.event_logger:
                                for shock in new_shocks:
                                    self.event_logger.log_shock(shock.shock_type, shock.details)
                    result = self.simulator.step_week()
                    self._last_day_result = result
                    self.tools.set_current_day(result.day)
                    self.world_state_id = uuid.uuid4().hex
                    if self.team_mode and self.team_phase is not None:
                        self.team_phase = 'boundary'
                    return result
                except Exception:
                    # Readers must see the failure before this lock is released.
                    self._operation_failed = True
                    raise

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_do_step)
        try:
            week_result = future.result(timeout=self.STEP_WEEK_TIMEOUT)
        except FuturesTimeoutError:
            elapsed = _time.monotonic() - _step_start
            self._last_step_elapsed = elapsed
            self._step_day_timed_out = True
            executor.shutdown(wait=False, cancel_futures=True)
            return {
                "success": False,
                "error": "step_week_timeout",
                "elapsed": elapsed,
                "message": f"step_week exceeded {self.STEP_WEEK_TIMEOUT}s timeout ({elapsed:.1f}s elapsed). Save checkpoint and exit.",
            }
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        executor.shutdown(wait=False)

        self._last_step_elapsed = _time.monotonic() - _step_start
        new_day = week_result.day

        # Build inbox items from shocks + enterprise threads (covering the whole week)
        inbox = []
        if self.shock_manager:
            inbox.extend(self.shock_manager.get_inbox_items(new_day))
        if self.conn:
            from saas_bench.environment import get_thread_inbox_items
            week_start = old_day + 1
            inbox.extend(get_thread_inbox_items(self.conn, new_day, week_start_day=week_start))

        # Build dashboard OUTSIDE the lock so weekly scripts can call back
        # to the API server (e.g., nm.query()) without deadlocking.
        calc_outputs = {} if self.team_mode else self._run_daily_scripts_internal()
        if self.dashboard_callback:
            dashboard = self.dashboard_callback(new_day, week_result)
        elif self.conn:
            # Run weekly scripts if available
            dashboard = build_weekly_dashboard(self.conn, new_day, week_result, calc_outputs, inbox)
        else:
            week = (new_day + 6) // 7
            dashboard = f"=== Week {week} Dashboard (Day {new_day}) ===\n(No dashboard data available)"

        from .execution_capture import CapturedText, dashboard_version
        if predictions:
            receipt = []
            for horizon, metrics in predictions.items():
                if 'cash' not in metrics:
                    continue
                cash = metrics['cash']
                values = cash if isinstance(cash, dict) else {'point': cash}
                amounts = ', '.join(f"{field}=USD {float(values[field])}"
                    for field in ('point', 'lower', 'upper') if values.get(field) is not None)
                receipt.append(f"  +{horizon} days: {amounts}")
            if receipt:
                receipt = f"\n\nSubmitted cash forecasts at D{old_day} (USD; horizons from submission):\n" + '\n'.join(receipt)
                dashboard = CapturedText(dashboard + receipt, getattr(dashboard, 'origins', []))

        dashboard = dashboard_version(self, dashboard, new_day)
        with self._lock:
            self._last_dashboard = dashboard

        if self.day_callback:
            self.day_callback(new_day, dashboard)

        return {
            "success": True,
            "day": new_day,
            "dashboard": dashboard,
        }

    @property
    def last_dashboard(self) -> str:
        return self._last_dashboard

    def get_daily_scripts(self, role="ceo") -> Dict[str, str]:
        """Get all registered daily script snapshots (name -> content)."""
        with self._lock:
            require_role(role)
            return dict(self._role_scripts[role])

    def _run_daily_scripts_internal(self, role="ceo"):
        require_role(role)
        import hashlib
        import shlex
        from pathlib import Path
        from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
        workspace = Path(self.script_workspace)
        if self.team_mode and self.role_executors.get(role) is None:
            raise RuntimeError('Registered scripts require their owner-bound executor')
        executor = self.role_executors.get(role) or BashAgentToolExecutor(workspace, bash_timeout=300,
            evidence_store=self.sql_evidence if self.sql_evidence and self.sql_evidence.execution_capture else None,
            require_sandbox=self.require_sandbox,
            env={'NOVAMIND_API_PORT': str(self.port), 'PYTHONHASHSEED': '0'})
        root = executor.guest_root
        executor.extra_env['PYTHONPATH'] = os.pathsep.join((root + '/docs', root))
        results = {}
        records = []
        with self._lock:
            self.role_script_results[role] = records
            if not self.team_mode:
                self.last_script_results = records
        executor.extra_env['PYTHONDONTWRITEBYTECODE'] = '1'
        for name, code in self.get_daily_scripts(role).items():
            import uuid
            from .execution_capture import ExecutionCapture, CURRENT_EVENT
            from .run_state import write_json
            capture = ExecutionCapture(executor.evidence_store) if executor.evidence_store else None
            token = None
            output, error = '', None
            executor.last_status = 'not_started'
            executor._bash_facts = dict(process_started=False, target_started=False, boundary_closed=True)
            executor._raw_streams = (b'', b'')
            executor._bash_error = None
            executor.registered_execution = True
            audit = self.role_audits.get(role)
            evidence_root = audit.path.parent if audit else workspace / 'sessions' / 'registered-script-evidence'
            evidence = evidence_root / 'scripts' / uuid.uuid4().hex
            try:
                if self.team_mode and self.team_phase is not None and self.script_check:
                    self.script_check()
                if capture:
                    versions = capture.store.load_state('script_versions') or {}
                    parent_token = CURRENT_EVENT.set(None) if self.team_mode else None
                    try:
                        capture.begin('registered_script_execution', {'name': name, 'registered_version': versions.get(name),
                                                                      'day': self.tools.current_day})
                    finally:
                        if parent_token is not None:
                            CURRENT_EVENT.reset(parent_token)
                    capture.blob('code', code, 'executed_code', derived_from=versions.get(name))
                    token = CURRENT_EVENT.set(capture.event)
                command = shlex.quote(executor.python) + ' -c ' + shlex.quote(code)
                output = executor.execute('bash', {'command': command})
                if capture and capture.event:
                    child = output.origins[0]['version_id'].rsplit(':', 1)[0] if getattr(output, 'origins', []) else None
                    child_record = capture.safe(capture.store.read_event, child) if child else None
                    status = child_record['result']['status'] if child_record else 'result_unknown'
                    if child_record and not getattr(output, 'pf_read', None):
                        from .agents.bash_agent.tools import read_identity
                        from .execution_capture import CapturedText
                        key = capture.safe(read_identity, capture.store, child, executor.workspace_path,
                                           'bash', {'command': command}, executor.guest_root)
                        read = capture.blob('read_full', output, 'pf_read_full',
                            target=output.origins[0]['version_id'], key=key, mode='content',
                            read_range=[0, len(output)], complete=not child_record['result'].get('output_truncated'),
                            force_full=False, read_kind='tool_call')
                        if read:
                            output = CapturedText(output, output.origins, {'id': read})
                    capture.safe(capture.store.complete, capture.event, status, child_event_id=child)
            except BaseException as exc:
                error = exc
                from .run_lifecycle import RunCancelled
                if isinstance(exc, RunCancelled) and executor._bash_facts.get('boundary_closed') is True:
                    executor.last_status = 'cancelled'
                    executor._bash_facts['cancelled'] = str(exc)
                if capture and capture.event:
                    status = 'cancelled' if executor.last_status == 'cancelled' else 'result_unknown'
                    capture.safe(capture.store.complete, capture.event, status, error=type(exc).__name__)
                raise
            finally:
                if token is not None:
                    CURRENT_EVENT.reset(token)
                executor.registered_execution = False
                facts = dict(executor._bash_facts)
                failure = error or executor._bash_error
                if failure:
                    facts.update(error_type=type(failure).__name__, error_errno=getattr(failure, 'errno', None))
                raw_stdout, raw_stderr = executor._raw_streams
                classification = registered_script_classification(facts, failure)
                if executor.evidence_store and executor.evidence_store.fault:
                    classification = 'unknown'
                    facts['capture_fault'] = executor.evidence_store.fault
                record = dict(name=name, role=role, day=getattr(self.tools, 'current_day', None),
                    sha256=hashlib.sha256(code.encode()).hexdigest(), output=str(output),
                    status=executor.last_status, classification=classification, facts=facts,
                    error=str(failure) if failure else None, evidence=str(evidence))
                evidence.mkdir(parents=True)
                (evidence / 'stdout.bin').write_bytes(raw_stdout)
                (evidence / 'stderr.bin').write_bytes(raw_stderr)
                (evidence / 'code.py').write_text(code)
                write_json(evidence / 'execution.json', record)
                records.append(record)
                if audit:
                    audit.record('registered_script_execution', **record)
            results[name] = output
            if self.team_mode and classification in ('transient_infrastructure', 'unknown'):
                break
        if executor.evidence_store:
            executor.evidence_store.assert_healthy(quiescent=False)
        return results

    def set_daily_scripts(self, scripts: Dict[str, str], registration=None, *, role="ceo"):
        require_role(role)
        """Restore daily scripts from checkpoint."""
        with self._lock:
            self.authorize_team(role, 'scripts')
            import hashlib
            if not isinstance(scripts, dict) or any(not isinstance(n, str) or not n or not isinstance(c, str)
                                                    for n, c in scripts.items()):
                raise ValueError('Invalid registered script snapshots')
            if self.team_mode and role != "ceo" and self.conn is not None:
                with self.conn:
                    self.conn.execute('DELETE FROM _team_scripts WHERE role=?', (role,))
                    self.conn.executemany('INSERT INTO _team_scripts VALUES (?,?,?,?,?)',
                        [(role, i, n, c, hashlib.sha256(c.encode()).hexdigest())
                         for i, (n, c) in enumerate(scripts.items())])
            elif self.conn is not None:
                self.conn.execute('SAVEPOINT registered_scripts')
                try:
                    self.conn.execute('DELETE FROM _registered_scripts')
                    self.conn.executemany('INSERT INTO _registered_scripts VALUES (?, ?, ?, ?)',
                        [(i, n, c, hashlib.sha256(c.encode()).hexdigest())
                         for i, (n, c) in enumerate(scripts.items())])
                    self.conn.execute('RELEASE registered_scripts')
                except Exception:
                    self.conn.execute('ROLLBACK TO registered_scripts')
                    self.conn.execute('RELEASE registered_scripts')
                    raise
            before = self._role_scripts[role]
            self._role_scripts[role] = dict(scripts)
            if role == "ceo":
                self._daily_scripts = self._role_scripts[role]
            if audit := self.role_audits.get(role):
                audit.record('script_registration', names=list(scripts))
            from .execution_capture import registered_scripts
            try:
                registered_scripts(self, scripts, before, registration, role=role)
            except Exception as exc:
                self.evidence_for(role).fail(exc)
