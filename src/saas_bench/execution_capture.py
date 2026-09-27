"""Host-only execution capture. Public clients never import this module."""
from contextvars import ContextVar
from contextlib import closing
import fnmatch
import io
import json
import os
from pathlib import Path
import stat
import time
import uuid

from .sql_evidence import digest, encoded, now

CURRENT_EVENT = ContextVar('capture_event', default=None)
EXCLUSIONS = ['sessions/*/world.nmdb', 'sessions/*/world.nmdb-*',
              'sessions/*/*.plain.tmp*', 'sessions/*/*.nmdb.tmp*']
OBJECT_FIELDS = dict(project_id='research_project', customer_id='customer', group_id='customer_group',
                     thread_id='enterprise_thread', post_id='social_post', agent_post_id='agent_social_post',
                     reply_to_post_id='social_post', discovered_group_id='customer_group',
                     plan='plan', channel='ad_channel')
READ_TOOLS = frozenset({'get_social_posts', 'get_cost_info', 'list_research_projects',
                        'get_market_overview', 'get_group_insights'})
_EXCLUSION_PARTS = [tuple(pattern.split('/')) for pattern in EXCLUSIONS]


def excluded(rel):
    """Right-anchored, per-segment glob, exactly as PurePosixPath(rel).match(pattern)."""
    parts = rel.split('/')
    return any(len(parts) >= len(pattern) and
               all(fnmatch.fnmatchcase(part, glob) for part, glob in zip(parts[-len(pattern):], pattern))
               for pattern in _EXCLUSION_PARTS)


class CapturedText(str):
    def __new__(cls, text, origins=(), pf_read=None):
        value = super().__new__(cls, text)
        value.origins = list(origins)
        value.pf_read = pf_read
        return value


def origin(version, text, start=0, end=None, target=0):
    end = len(text) if end is None else end
    return dict(version_id=version, source_range=[start, end],
                request_range=[target, target + end - start], full_source=start == 0 and end == len(text))


def decoded(raw):
    # Matches read_text()/Popen(text=True), including universal newlines.
    with io.TextIOWrapper(io.BytesIO(raw)) as stream:
        return stream.read()


class ExecutionCapture:
    def __init__(self, store):
        self.store = store
        self.event = None
        self.slots = 0
        self.origins = []
        self.facts = {}

    def safe(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            self.store.fail(exc)
            return None

    def begin(self, kind, args, parent=None, **facts):
        self.slots, self.origins, self.facts = 0, [], {}
        self.event = self.safe(self.store.begin_event, kind, args, parent=parent or CURRENT_EVENT.get(), **facts)
        return self.event

    def blob(self, slot, value, layer, **metadata):
        if not self.event:
            return None
        return self.safe(self.store.version, self.event, slot, value, layer=layer, **metadata)

    def file(self, path, raw, text=None, phase='read'):
        self.slots += 1
        raw_version = self.blob(f'file_{self.slots}_{phase}', raw, 'file_bytes', object_id=str(path))
        if text is None:
            return raw_version
        version = self.blob(f'file_{self.slots}_text', text, 'file_text', derived_from=raw_version,
                            transformation='decode_universal_newlines')
        return version

    def snapshot(self, workspace, phase):
        started = time.monotonic()
        with self.store.batch():
            items = self._snapshot(workspace, phase)
        self.facts[phase + '_scan_seconds'] = time.monotonic() - started
        return items

    def _snapshot(self, workspace, phase):
        items = {}
        reused = 0
        root = str(Path(workspace).resolve())
        # Every file is read and hashed on each boundary (no mtime shortcut); only bytes
        # that differ from the newest visible version of the path get a new version.
        for directory, dirs, files in os.walk(root, followlinks=False):
            prefix = '' if directory == root else directory[len(root) + 1:] + '/'
            for name in sorted(dirs + files):
                path = directory + '/' + name
                rel = prefix + name
                if excluded(rel):
                    items[rel] = {'type': 'excluded', 'reason': 'simulator_private_state'}
                    continue
                info = os.lstat(path)
                item = dict(mode=info.st_mode, size=info.st_size)
                if stat.S_ISLNK(info.st_mode):
                    item.update(type='symlink', target=os.readlink(path))
                elif stat.S_ISREG(info.st_mode):
                    # O_NOFOLLOW prevents a replacement symlink escaping the workspace.
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    with os.fdopen(fd, 'rb') as stream:
                        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                            raise RuntimeError('File changed type during capture: ' + rel)
                        raw = stream.read()
                        after = os.fstat(stream.fileno())
                    if (info.st_ino, info.st_size, info.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                        raise RuntimeError('File changed during boundary capture: ' + rel)
                    sha = digest(raw)
                    version, latest_sha = self.store.latest_version(rel, 'file_bytes') if self.event else (None, None)
                    if version and latest_sha == sha:
                        reused += 1
                    else:
                        version = self.file(rel, raw, phase=phase)
                    item.update(type='file', sha256=sha, version=version)
                else:
                    item['type'] = 'directory' if stat.S_ISDIR(info.st_mode) else 'special'
                items[rel] = item
        self.blob('workspace_' + phase, encoded(items), 'workspace_boundary', exclusions=EXCLUSIONS,
                  reused_file_versions=reused)
        return items

    def finish(self, text, status='succeeded', read_key=None, read_complete=True, **facts):
        version = self.blob('tool_return', text, 'tool_return', segments=self.origins) if text is not None else None
        pf_read = getattr(text, 'pf_read', None)
        if read_key and version:
            # The exact tool return is the read target; repeated identical calls in one
            # request context can then be delivered as DELTA or UNCHANGED.
            read = self.blob('read_full', text, 'pf_read_full', target=version, key=read_key, mode='content',
                             read_range=[0, len(text)], complete=read_complete, force_full=False,
                             read_kind='tool_call')
            pf_read = {'id': read} if read else pf_read
        self.facts.update(facts)
        if self.event:
            self.safe(self.store.complete, self.event, status, **self.facts)
        origins = [origin(version, text)] if version else []
        return CapturedText(text, origins + self.origins, pf_read) if text is not None else None


def capture_http(handler, raw):
    store = handler.server._api_server.sql_evidence
    token = handler.headers.get('X-Capture-Context')
    parent = store.parent(token) if token else None
    try:
        request = json.loads(raw) if raw else {}
    except (ValueError, UnicodeError):
        request = None
    return store.begin_event('public_http', dict(method=handler.command, path=handler.path,
                             body_hex=raw.hex(), parsed=request), parent=parent, requires_delivery=True, admitted=True,
                             parent_reason=None if parent else 'unpropagated_context')


def finish_http(store, event, status, headers, body, execution):
    try:
        value = json.loads(body)
    except (ValueError, UnicodeError):
        value = {}
    request = store.read_event(event)['request']['request']
    tool = (request.get('parsed') or {}).get('tool') if isinstance(request.get('parsed'), dict) else None
    classification = 'read' if request['method'] == 'GET' or tool in READ_TOOLS else 'write_receipt'
    objects, dates, outcomes = [], [], []
    scalar_objects = dict(OBJECT_FIELDS)
    keyed_objects = dict(by_group='customer_group', by_customer='customer', by_plan='plan',
                         by_channel='ad_channel', model_tiers='model_tier', capacity_tiers='capacity_tier')
    nested_objects = dict(by_group_plan=('customer_group', 'plan'), by_channel_group=('ad_channel', 'customer_group'))
    if tool == 'set_targeted_ad_spend':
        nested_objects['targeted_spend'] = ('ad_channel', 'customer_group')
    elif tool in ('set_targeted_ops_spend', 'set_targeted_dev_spend'):
        keyed_objects['targeted_spend'] = 'customer_group'
    if tool in ('set_prices', 'set_model_tiers', 'set_usage_quotas'):
        keyed_objects.update(updated='plan', current='plan')
    tier_kind = 'research_tier' if tool in ('start_research_project', 'list_research_projects') else 'capacity_tier' if tool == 'set_capacity_tier' else None
    if tier_kind:
        scalar_objects['tier'] = tier_kind
    def fields(data, pointer='', basis='public_response_field'):
        if isinstance(data, dict):
            for key, val in data.items():
                path = pointer + '/' + key.replace('~', '~0').replace('/', '~1')
                if key in scalar_objects and type(val) in (str, int):
                    objects.append(dict(kind=scalar_objects[key], value=val, path=path, basis=basis))
                if key in keyed_objects and isinstance(val, dict):
                    objects.extend(dict(kind=keyed_objects[key], value=k, path=path + '/' + str(k), basis=basis + '_dictionary_key') for k in val)
                if key in nested_objects and isinstance(val, dict):
                    outer, inner = nested_objects[key]
                    for k, children in val.items():
                        objects.append(dict(kind=outer, value=k, path=path + '/' + str(k), basis=basis + '_dictionary_key'))
                        if isinstance(children, dict):
                            objects.extend(dict(kind=inner, value=c, path=path + '/' + str(k) + '/' + str(c), basis=basis + '_dictionary_key') for c in children)
                if key in ('day', 'current_day', 'start_day', 'started_day', 'end_day', 'expected_completion_day', 'snapshot_day', 'data_day', 'next_reply_day') and type(val) is int:
                    dates.append(dict(field=key, value=val, path=path, basis=basis))
                if key == 'success' and type(val) is bool and pointer and basis == 'public_response_field':
                    outcomes.append(dict(path=path, success=val))
                if key == 'error' and val and pointer.startswith('/data/results/'):
                    outcomes.append(dict(path=path, success=False))
                fields(val, path, basis)
        elif isinstance(data, list):
            for i, val in enumerate(data):
                fields(val, pointer + '/' + str(i), basis)
    fields(value)
    fields(request.get('parsed'), '/request', 'public_request_target')
    store.version(event, 'public_response', body, layer='server_public_response',
                  source_truncated=bool(value.get('truncated')), objects=objects, public_fields=execution.get('public_fields', []),
                  content_time={'status': 'declared', 'fields': dates} if dates else {'status': 'unknown', 'reason': 'not_declared'})
    outcome = ('result_unknown' if execution.get('world_outcome_unknown') else
               'succeeded' if status < 400 and value.get('success', True) else 'failed')
    if outcome == 'succeeded' and any(not item['success'] for item in outcomes):
        outcome = 'partially_succeeded' if any(item['success'] for item in outcomes) else 'failed'
    store.complete(event, outcome,
                   http_status=status, headers=headers, classification=classification,
                   public_success=value.get('success'), item_outcomes=outcomes, receive_state='unknown',
                   **{k: execution[k] for k in ('refresh_of', 'day', 'snapshot_ref', 'refresh_error') if k in execution})


def public_handler(method):
    """Admit complete public operations, allowing active executions to finish callbacks."""
    from functools import wraps
    @wraps(method)
    def observed(handler):
        api = handler.server._api_server
        store = api.sql_evidence
        if handler.path == '/_capture':
            return receive_client(handler)
        if not store or not store.execution_capture:
            return method(handler)
        if handler.path in ('/health', '/game-status', '/checkpoint', '/pf-refresh'):
            handler._control_capture = True
            try:
                return method(handler)
            finally:
                if hasattr(handler, '_control_record'):
                    try:
                        with api._lock:
                            with store.path.with_suffix('.controls.jsonl').open('a') as stream:
                                handler._control_record['recorded_at'] = now()
                                stream.write(json.dumps(handler._control_record) + '\n')
                                stream.flush()
                                os.fsync(stream.fileno())
                    except Exception as exc:
                        store.fail(exc)
        if store.fault or store.fault_path.exists():
            handler._send_json({'success': False, 'error': 'Execution capture failed; branch stopped'}, 503)
            return
        with api._sql_lock:
            while api._sql_paused:
                try:
                    if store.parent(handler.headers.get('X-Capture-Context')):
                        break
                except ValueError:
                    pass
                api._sql_lock.wait()
            api._sql_active += 1
        handler._sql_event, handler._sql_response = None, None
        handler._sql_execution, handler._sql_capture_failed = {}, False
        handler._generic_capture = handler.path != '/query'
        token = None
        try:
            raw = handler.rfile.read(int(handler.headers.get('Content-Length', 0)))
            handler.rfile = io.BytesIO(raw)
            try:
                if handler.path == '/query':
                    from .public_sql import PUBLIC_POLICY_VERSION
                    handler._sql_event = store.begin(raw, PUBLIC_POLICY_VERSION,
                        parent=store.parent(handler.headers.get('X-Capture-Context')))
                else:
                    handler._sql_event = capture_http(handler, raw)
                client = handler.headers.get('X-Capture-Call')
                if client:
                    store.bind_client(handler._sql_event, client)
                token = CURRENT_EVENT.set(handler._sql_event)
            except Exception as exc:
                handler._sql_capture_failed = True
                store.fail(exc)
            return method(handler)
        finally:
            if token is not None:
                CURRENT_EVENT.reset(token)
            with api._sql_lock:
                api._sql_active -= 1
                api._sql_lock.notify_all()
    return observed


def receive_client(handler):
    store = handler.server._api_server.sql_evidence
    if not store or not store.execution_capture or handler.command != 'POST':
        handler._send_json({'error': 'Unknown endpoint'}, 404)
        return
    try:
        length = int(handler.headers.get('Content-Length', 0))
        if not 0 < length <= 64 * 1024 * 1024:
            raise ValueError('Invalid capture payload length')
        body = json.loads(handler.rfile.read(length))
        if set(body) != {'context', 'call', 'record'}:
            raise ValueError('Invalid capture record')
        parent = store.parent(body['context'])
        if parent is None or not isinstance(body['record'], dict):
            raise ValueError('Capture requires an active execution')
        record = body['record']
        if record.get('kind') == 'python_start':
            if set(record) != {'kind', 'code', 'source'} or not all(isinstance(record[k], str) for k in ('code', 'source')):
                raise ValueError('Invalid Python start record')
            child = store.begin_event('cli_python', {'source': record['source']}, parent=parent)
            store.version(child, 'code', record['code'], layer='executed_code')
            store.context(child, body['call'])
            handler._send_json({'accepted': True})
            return
        if record.get('kind') == 'python_end':
            if set(record) != {'kind', 'stdout', 'stderr', 'exit_code', 'error'}:
                raise ValueError('Invalid Python end record')
            child = store.parent(body['call'])
            if store.read_event(child)['request']['parent_event_id'] != parent:
                raise ValueError('Python execution parent mismatch')
            for name in ('stdout', 'stderr'):
                raw = bytes.fromhex(record[name])
                store.version(child, name + '_bytes', raw, layer=name + '_bytes')
                try:
                    store.version(child, name, decoded(raw), layer=name, derived_from=child + ':' + name + '_bytes')
                except UnicodeError:
                    pass  # Raw bytes survive a client-visible decode failure.
            status = 'result_unknown' if record['exit_code'] is None else 'failed' if record['exit_code'] else 'succeeded'
            store.complete(child, status, exit_code=record['exit_code'], error=record['error'])
            if status == 'result_unknown':
                store.fail('Python child outcome unknown')
            handler._send_json({'accepted': True})
            return
        if not set(body['record']) <= {'body', 'parsed', 'returned', 'error', 'state', 'projection', 'transformations'}:
            raise ValueError('Unsupported capture fields')
        with closing(store.connect()) as conn:
            row = conn.execute('SELECT r.request FROM client_calls c JOIN requests r USING(event_id) WHERE c.token=?', (body['call'],)).fetchone()
            if not row or json.loads(row[0]).get('parent_event_id') != parent:
                raise ValueError('Client call does not belong to this execution')
        store.received(body['call'], body['record'])
        handler._send_json({'accepted': True})
    except Exception as exc:
        store.fail(exc)
        handler._send_json({'accepted': False}, 400)


def slice_origins(origins, start, end, target=0):
    result = []
    for item in origins:
        a, b = item['request_range']
        left, right = max(start, a), min(end, b)
        if left < right:
            source = item['source_range'][0] + left - a
            result.append(dict(item, source_range=[source, source + right - left],
                               request_range=[target + left - start, target + right - start],
                               full_source=item['full_source'] and left == a and right == b))
    return result


def query_projection(capture, stdout):
    """Attribute an unchanged, complete standard CLI projection to its SQL result.

    Arbitrary Python prints, pipes, and concatenated output remain unassigned.
    Both the observed CLI print and its enclosing stdout must match exactly and
    use the same OS pipe. Redirecting the query and echoing equal text cannot
    establish this relation.
    """
    store = capture.store
    with closing(store.connect()) as conn:
        rows = conn.execute('''SELECT v.version_id, r.event_id FROM versions v
            JOIN requests r USING(event_id)
            WHERE json_extract(v.metadata, '$.layer') = 'program_projection'
            AND json_extract(r.request, '$.parent_event_id') = ? AND r.query_id IS NOT NULL''',
            (capture.event,)).fetchall()
    matches = []
    for row in rows:
        projection = json.loads(store.get_content(row['version_id'])[1])
        prints = [part['text'] for part in projection if part['stream'] == 'stdout'
                  and part.get('sink') is not None and part['sink'] == capture.facts.get('stdout_sink')]
        if not prints or ''.join(prints) != stdout:
            continue
        public = row['event_id'] + ':public_response'
        meta, raw = store.get_content(public)
        body = json.loads(raw)
        try:
            projected = json.loads(stdout)
        except ValueError:
            continue
        expected = {k: body.get(k) for k in ('columns', 'rows', 'row_count')}
        if body.get('success') and projected == expected:
            matches.append((public, meta))
    if len(matches) != 1:
        return []
    public, meta = matches[0]
    version = capture.blob('query_projection', stdout, 'query_model_projection',
                           object_id=meta['object_id'], derived_from=public,
                           source_truncated=meta['source_truncated'], transformation='standard_cli_query_projection')
    return [origin(version, stdout)] if version else []


def text_sources(value, pointer=''):
    """Walk known string identities; never search for matching text."""
    result = []
    if isinstance(value, CapturedText):
        result.append(dict(pointer=pointer, text=str(value), origins=value.origins))
        if value.pf_read is not None:
            result[-1]['pf_read'] = value.pf_read
    elif isinstance(value, dict):
        for key, item in value.items():
            result.extend(text_sources(item, pointer + '/' + str(key).replace('~', '~0').replace('/', '~1')))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            result.extend(text_sources(item, pointer + '/' + str(i)))
    return result


def at_pointer(value, pointer):
    for key in pointer.split('/')[1:]:
        key = key.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def restore_sources(value, records):
    for record in records:
        pointer = record['pointer']
        parent, key = pointer.rsplit('/', 1)
        target = at_pointer(value, parent)
        key = int(key) if isinstance(target, list) else key.replace('~1', '/').replace('~0', '~')
        if target[key] != record['text']:
            raise ValueError('Private source state differs from conversation snapshot')
        target[key] = CapturedText(target[key], record['origins'], record.get('pf_read'))


def model_request(store, raw, sources, call_id, attempt_id, context_id):
    event = store.begin_event('model_request', dict(call_id=call_id, attempt_id=attempt_id, context_id=context_id))
    # One transaction for the wire, occurrences and every read of this request.
    with store.batch():
        _record_model_request(store, event, raw, sources, call_id, attempt_id, context_id)
    return event


def _record_model_request(store, event, raw, sources, call_id, attempt_id, context_id):
    store.version(event, 'wire', raw, layer='model_request_wire')
    body = json.loads(raw)
    occurrences = []
    for source in sources:
        text = at_pointer(body, source['pointer'])
        parts = source['pointer'].split('/')
        role = 'system' if parts[1] in ('instructions', 'system') else (body[parts[1]][int(parts[2])].get('role', 'tool') if parts[1] in ('messages', 'input') else 'unknown')
        if text != source['text']:
            raise ValueError('SDK serialization changed a source-bearing field')
        for item in source['origins']:
            metadata, content = store.get_content(item['version_id'])
            original = content.decode('utf-8')
            a, b = item['source_range']
            c, d = item['request_range']
            if not (0 <= a <= b <= len(original) and 0 <= c <= d <= len(text)) or original[a:b] != text[c:d]:
                raise ValueError('Source occurrence range does not match final request')
            occurrences.append(dict(item, reader='ceo', role=role, context_id=context_id, call_id=call_id,
                                    attempt_id=attempt_id, json_pointer=source['pointer'],
                                    send_state_event_id=event))
    store.version(event, 'occurrences', encoded(occurrences), layer='model_source_occurrences')
    if any(source.get('pf_read') for source in sources):
        from .pf_read import record_request
        record_request(store, event, body, sources, context_id)


def registered_scripts(api, scripts, before, registration=None):
    store = api.sql_evidence
    if not store or not store.execution_capture:
        return
    capture = ExecutionCapture(store)
    changed = [name for name in scripts if name == registration or scripts[name] != before.get(name)]
    removed = sorted(before.keys() - scripts.keys())
    capture.begin('script_registration', dict(names=changed, removed=removed))
    versions = store.load_state('script_versions') or {}
    for name in changed:
        versions[name] = capture.blob('script_' + digest(name.encode()), scripts[name],
                                     'registered_script', object_id='registered_script:' + name)
    for name in removed:
        versions.pop(name, None)
    capture.safe(store.save_state, 'script_versions', versions)
    if capture.event:
        capture.safe(store.complete, capture.event, outputs=list(versions.values()))


def dashboard_version(api, text, day):
    store = api.sql_evidence
    if not store or not store.execution_capture:
        return text
    capture = ExecutionCapture(store)
    capture.begin('dashboard_generation', {'day': day})
    version = capture.blob('dashboard', str(text), 'dashboard', object_id='dashboard',
                          segments=getattr(text, 'origins', []),
                          content_time={'status': 'declared', 'day': day, 'basis': 'dashboard_generation'})
    if capture.event:
        capture.safe(store.complete, capture.event)
    sources = ([origin(version, text)] if version else []) + getattr(text, 'origins', [])
    capture.safe(store.save_state, 'dashboard', dict(version=version, day=day, origins=sources, sha256=digest(text.encode())))
    return CapturedText(text, sources)
