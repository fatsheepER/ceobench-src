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
import sys
import threading
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

from .tools import AgentTools, ToolResult
from .database import TABLE_DOCS
from .environment import build_weekly_dashboard


from .public_sql import PUBLIC_COLUMNS, QueryDenied, SnapshotUnavailable, execute_query

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
                    if (set(body) != {'versions', 'parent'} or not isinstance(body['parent'], str) or
                            not isinstance(body['versions'], list) or
                            not all(isinstance(v, str) for v in body['versions'])):
                        self._send_json({'error': 'versions and parent are required'}, 400)
                        return
                    from .pf_refresh import refresh
                    try:
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
        except Exception as exc:
            self._send_internal_error(exc, op=f"POST {self.path}")

    @public_handler
    def do_GET(self):
        try:
            if self.path == '/vars':
                self._handle_vars()
            elif self.path == '/health':
                evidence = self.server._api_server.sql_evidence
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
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        if getattr(self, '_control_capture', False):
            self._control_request = body.hex()
        return json.loads(body) if body else {}

    def _send_json(self, data: Dict, status: int = 200):
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
        store = self.server._api_server.sql_evidence
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
            args = body.get('args', {})

            server: NovaMindAPIServer = self.server._api_server
            if server.read_only_task and tool_name not in READ_ONLY_TASK_TOOLS:
                self._send_json({'success': False, 'error': 'Decision preparation cannot change the world'}, 403)
                return
            result = server.execute_tool(tool_name, args)

            if isinstance(result, ToolResult):
                self._send_json(result.to_json())
            else:
                # Fallback for non-ToolResult returns
                self._send_json({"success": True, "data": {"output": str(result)}, "message": str(result)})
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

            result = server.advance_week(predictions=parsed, rationale=rationale)
            self._send_json(result)
        except Exception as e:
            self._send_internal_error(e, op="next-week")

    def _handle_query(self):
        api = self.server._api_server
        if api.sql_evidence and api.sql_evidence.execution_capture:
            self._query_response_started = False
            return self._handle_query_request()
        with api._sql_lock:
            while api._sql_paused:
                api._sql_lock.wait()
            api._sql_active += 1
        try:
            self._query_response_started = False
            self._sql_event = None
            self._sql_response = None
            self._sql_capture_failed = False
            self._sql_execution = {}
            self._handle_query_request()
        finally:
            self._sql_event = None
            with api._sql_lock:
                api._sql_active -= 1
                api._sql_lock.notify_all()

    def _handle_query_request(self):
        sql = ''
        try:
            raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
            from .public_sql import PUBLIC_POLICY_VERSION
            if not (self.server._api_server.sql_evidence and self.server._api_server.sql_evidence.execution_capture):
                self._sql_event = self._capture_sql('begin', raw, PUBLIC_POLICY_VERSION)
            body = json.loads(raw) if raw else {}
            if not isinstance(body, dict) or not isinstance(body.get('sql'), str) or not body['sql'].strip():
                self._send_json({'success': False, 'error': 'sql must be a non-empty string'}, 400)
                return
            sql = body['sql']
            response = execute_query(self.server._api_server, sql, metadata=self._sql_execution)
            enum_hint = _get_enum_hint_for_query(sql, response['rows'])
            if enum_hint:
                response['hint'] = enum_hint
            self._send_query_json(response)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({'success': False, 'error': 'Invalid JSON body'}, 400)
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

    def _send_query_json(self, data):
        """Bound serialization and socket writes separately from SQL execution."""
        import time
        from .public_sql import check_deadline
        started = time.monotonic()
        chunks = []
        deadline = started + self.server._api_server.QUERY_RESPONSE_TIMEOUT_SECONDS
        for chunk in json.JSONEncoder(default=str).iterencode(data):
            check_deadline(deadline)
            chunks.append(chunk.encode())
        response = b''.join(chunks)
        check_deadline(deadline)
        capture_started = time.monotonic()
        self._capture_response(200, response)
        # Capture has its own accounting, never spend SQL/response execution budgets on it.
        deadline += time.monotonic() - capture_started
        serialized = time.monotonic()
        self.connection.settimeout(max(0.001, deadline - serialized))
        try:
            self._query_response_started = True
            self.send_response(200)
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
                scripts = server.get_daily_scripts()
                scripts[name] = content
                server.set_daily_scripts(scripts, registration=name)
            self._send_json({"success": True, "data": {"name": name, "registered": True}})
        except Exception as e:
            self._send_internal_error(e, op="daily-scripts:post")

    def _handle_daily_scripts_get(self):
        """List registered daily scripts: GET /daily-scripts."""
        server: NovaMindAPIServer = self.server._api_server
        with server._lock:
            scripts = [{"name": n, "size": len(c)} for n, c in server._daily_scripts.items()]
        self._send_json({"success": True, "data": {"scripts": scripts}})

    def _handle_daily_scripts_delete(self):
        """Remove a daily script: DELETE /daily-scripts {"name": "x.py"}."""
        try:
            body = self._read_body()
            name = body.get('name', '')
            server: NovaMindAPIServer = self.server._api_server
            with server._lock:
                if name not in server._daily_scripts:
                    self._send_json({"success": False, "error": f"Script not found: {name}"}, 404)
                    return
                scripts = server.get_daily_scripts()
                del scripts[name]
                server.set_daily_scripts(scripts)
            self._send_json({"success": True, "data": {"removed": name}})
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
            dashboard = dashboard_version(server, dashboard, day)
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
    'set_targeted_ops_spend': lambda tools, args: tools.set_targeted_ops_spend(args.get('targeted_spend', args)),
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
                 require_sandbox=False, sql_evidence=None, checkpoint_token=None):
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
        self._sql_active = 0
        self._sql_paused = False
        self._advance_lock = threading.Lock()
        self._operation_failed = False
        self.checkpoint_callback = None
        self.checkpoint_token = checkpoint_token
        self._last_dashboard: str = ""
        self._last_day_result = None
        self._daily_scripts: Dict[str, str] = {}  # name -> content snapshot
        self.script_workspace = script_workspace or tools.workspace_path
        self.require_sandbox = require_sandbox
        self.last_script_results = []
        if conn is not None:
            conn.execute('CREATE TABLE IF NOT EXISTS _registered_scripts (position INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, content TEXT NOT NULL, sha256 TEXT NOT NULL)')
            saved_scripts = conn.execute('SELECT name, content, sha256 FROM _registered_scripts ORDER BY position').fetchall()
            import hashlib
            for name, content, checksum in saved_scripts:
                if hashlib.sha256(content.encode()).hexdigest() != checksum:
                    raise ValueError('Registered script checksum mismatch: ' + name)
                self._daily_scripts[name] = content
        self._step_day_timed_out: bool = False  # Set when step_day exceeds timeout

    def start(self):
        """Start the HTTP server in a background thread."""
        self._httpd = ThreadingHTTPServer(('127.0.0.1', 0), _APIHandler)
        self._httpd._api_server = self
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the HTTP server."""
        if self._httpd:
            self._httpd.shutdown()
            self._httpd = None

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool call with thread safety."""
        with self._lock:
            dispatch_fn = _TOOL_DISPATCH.get(tool_name)
            if dispatch_fn is None:
                return ToolResult(False, f"Unknown tool: {tool_name}")
            return dispatch_fn(self.tools, args)

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

    def advance_week(self, predictions=None, rationale=None):
        if not self._advance_lock.acquire(blocking=False):
            raise RuntimeError('Week advancement already in progress')
        try:
            if self._operation_failed or self._step_day_timed_out:
                raise RuntimeError('Previous operation outcome unknown; continuation refused')
            return self._advance_week(predictions, rationale)
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
        calc_outputs = self._run_daily_scripts_internal()
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

    def get_daily_scripts(self) -> Dict[str, str]:
        """Get all registered daily script snapshots (name -> content)."""
        with self._lock:
            return dict(self._daily_scripts)

    def _run_daily_scripts_internal(self):
        import hashlib
        import shlex
        from pathlib import Path
        from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
        workspace = Path(self.script_workspace)
        executor = BashAgentToolExecutor(workspace, bash_timeout=300,
            evidence_store=self.sql_evidence if self.sql_evidence and self.sql_evidence.execution_capture else None,
            require_sandbox=self.require_sandbox,
            env={'NOVAMIND_API_PORT': str(self.port), 'PYTHONHASHSEED': '0'})
        root = executor.guest_root
        executor.extra_env['PYTHONPATH'] = os.pathsep.join((root + '/docs', root))
        results = {}
        self.last_script_results = []
        for name, code in self.get_daily_scripts().items():
            from .execution_capture import ExecutionCapture, CURRENT_EVENT
            capture = ExecutionCapture(executor.evidence_store) if executor.evidence_store else None
            token = None
            if capture:
                versions = capture.store.load_state('script_versions') or {}
                capture.begin('registered_script_execution', {'name': name, 'registered_version': versions.get(name)})
                capture.blob('code', code, 'executed_code', derived_from=versions.get(name))
                token = CURRENT_EVENT.set(capture.event)
            try:
                output = executor.execute('bash', {'command': shlex.quote(executor.python) + ' -c ' + shlex.quote(code)})
            except BaseException as exc:
                if capture and capture.event:
                    capture.safe(capture.store.complete, capture.event, 'result_unknown', error=type(exc).__name__)
                raise
            else:
                if capture and capture.event:
                    child = output.origins[0]['version_id'].rsplit(':', 1)[0] if getattr(output, 'origins', []) else None
                    child_record = capture.safe(capture.store.read_event, child) if child else None
                    status = child_record['result']['status'] if child_record else 'result_unknown'
                    capture.safe(capture.store.complete, capture.event, status, child_event_id=child)
            finally:
                if token is not None:
                    CURRENT_EVENT.reset(token)
            results[name] = output
            self.last_script_results.append(dict(name=name,
                sha256=hashlib.sha256(code.encode()).hexdigest(), output=output))
        return results

    def set_daily_scripts(self, scripts: Dict[str, str], registration=None):
        """Restore daily scripts from checkpoint."""
        with self._lock:
            import hashlib
            if not isinstance(scripts, dict) or any(not isinstance(n, str) or not n or not isinstance(c, str)
                                                    for n, c in scripts.items()):
                raise ValueError('Invalid registered script snapshots')
            if self.conn is not None:
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
            before = self._daily_scripts
            self._daily_scripts = dict(scripts)
            from .execution_capture import registered_scripts
            try:
                registered_scripts(self, scripts, before, registration)
            except Exception as exc:
                self.sql_evidence.fail(exc)
