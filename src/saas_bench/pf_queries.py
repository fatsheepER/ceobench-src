"""PF-only historical queries, rebuilt from immutable capture and declaration records."""
from collections import defaultdict, deque
from contextlib import closing
import difflib
import json
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, ValidationError, model_validator

from .execution_capture import CapturedText, CURRENT_EVENT, OBJECT_FIELDS, origin
from . import pf_render
from .registration_evidence import HANDLES
from .registration_schema import BusinessObject, Input, Text
from .sql_evidence import encoded


class Target(Input):
    path: Text | None = None
    sql: Text | None = None
    record: Annotated[str, Field(pattern=r'^r[1-9][0-9]*(\.[1-9][0-9]*)?$')] | None = None
    version: Annotated[str, Field(pattern=r'^v[1-9][0-9]*$')] | None = None

    @model_validator(mode='after')
    def captured_target(self):
        if sum(v is not None for v in (self.path, self.sql, self.record, self.version)) != 1:
            raise ValueError('Specify exactly one captured path, SQL, registered text, or version handle')
        if self.path is not None:
            path = PurePosixPath(self.path)
            if path.is_absolute() or '..' in path.parts or path.as_posix() != self.path or '\x00' in self.path:
                raise ValueError('Path must be normalized and workspace-relative')
        return self


class Page(Input):
    cursor: Annotated[str, Field(pattern=r'^c[1-9][0-9]*$')] | None = None
    limit: Annotated[int, Field(ge=1, le=100)] = 20
    detail: bool = False


class Search(Page):
    object: BusinessObject | None = None


class Trace(Page):
    target: Target | None = None
    depth: Annotated[int, Field(ge=1, le=10)] = 2
    include_execution: bool = False


class Dependents(Trace):
    current_only: bool = True


class Dependencies(Trace):
    purpose: Literal['current', 'historical_only'] = 'current'


class Read(Page):
    target: Target | None = None
    mode: Literal['history', 'content', 'diff'] = 'content'
    baseline: Target | None = None
    full: bool = False


MODELS = dict(pf_search=Search, pf_dependencies=Dependencies, pf_dependents=Dependents, pf_read=Read)


def tool_definitions():
    descriptions = {
        'pf_search': 'Saved evidence and registered texts about one business object, e.g. kind=research_project '
                     'id=t10_2 or kind=customer_group id=S2, newest first, one line each. detail=true adds when '
                     'you saw each item and where the object appears. Read contents with pf_read.',
        'pf_dependencies': 'What a registered text or file relies on, one line per cited source. With purpose=current '
                           '(default) cited queries and reads are rerun and files compared; each line says unchanged, '
                           'changed (with the differing rows or values) or whether a predicate holds. Use before '
                           'acting on an earlier plan or conclusion. detail=true lists every underlying query, read '
                           'and file. purpose=historical_only only lists history. Advisory.',
        'pf_dependents': 'Registered texts that cite a version, e.g. forecasts and plans that still rely on an '
                         'assumption you just corrected. detail=true adds scripts, files and outputs derived from it.',
        'pf_read': 'Read a saved version (mode=content), list its versions newest first (history), or diff two '
                   'versions of the same object (diff), including results you never saved as files. full=true '
                   'returns full text.',
    }
    return [dict(name=name, description=descriptions[name] + ' Continue a page with only its cursor.',
                 parameters=model.model_json_schema()) for name, model in MODELS.items()]


PF_PROMPT = '''

PF saves the query results, command outputs and workspace file versions from your
tool calls, and answers questions about them across weeks. The weekly check reruns
the queries and reads your active texts cite and compares the files they cite; a
reference with select and a threshold predicate is checked against that condition
instead of reporting any change. Checks are advisory; you decide what to do.
- pf_dependencies: before acting on an earlier plan or conclusion, see what it
  cites and what changed. Example: r4 "keep B at $99 while S2 new B
  subscriptions stay >= 770" cites a query with that threshold; the check reruns
  the query and reports whether the threshold still holds.
- pf_dependents: after correcting an assumption, find the texts that cite it.
- pf_search: saved items about one business object, newest first.
- pf_read: read or diff saved versions, including outputs you never saved.
Results give one line per item; "detail": true adds the underlying sources.
Repeated reads may come back compact. UNCHANGED means identical to the named base;
DELTA lists [start,end,text] replacements (zero-based character offsets into the
base, applied together). The base is the previous result of the same call, or of
the same script (previous_output_of:<script>). Use pf_read with full=true, or repeat
the call once, for full text.
'''


RELATION = dict(reference='cites', same_execution='computed from', returned_range='printed from',
                public_field_range='field from', executed_by='run by', derived_from='derived from',
                observed_edit_input='edited from')
# Layers that count as a checked source of a derived output; stdout copies and code do not.
SOURCE_LAYERS = {'server_public_response', 'file_bytes', 'registered_text', 'dashboard'}


def _depth(row):
    """Edges from the queried root have depth 1; a missing target adds no node to the path."""
    return len(row['path']) - (1 if row['edge']['target'] else 0)


def _weekly(row):
    """Direct current references the week-start check covers; explicit unknowns are not evidence."""
    evidence = row['edge'].get('reference', {}).get('evidence', {})
    return _depth(row) == 1 and not row['historical_only'] and 'unknown' not in evidence


def _about(content_time):
    """Declared content days as 'day 7' or 'days 7-14'."""
    if content_time.get('status') != 'declared':
        return None
    days = [content_time[k] for k in ('day', 'start_day', 'end_day') if type(content_time.get(k)) is int]
    days += [f['value'] for f in content_time.get('fields', []) if type(f.get('value')) is int]
    if not days:
        return None
    low, high = min(days), max(days)
    return f'day {low}' if low == high else f'days {low}-{high}'


# Private request wires, declarations, comparison blobs and workspace manifests
# are never addressable through the agent's historical reader.
PUBLIC_LAYERS = {'file_bytes', 'file_text', 'server_public_response', 'registered_text',
                 'registered_script', 'executed_code', 'stdout', 'stderr', 'dashboard',
                 'query_model_projection', 'tool_return'}


def evidence_key(version, meta, query, result, request):
    if meta['layer'] == 'server_public_response' and query:
        return ('query', encoded([query[1], *query[3:]]).decode())
    if meta['layer'] == 'server_public_response' and result.get('classification') == 'read':
        return ('public_read', encoded([request['method'], request['path'], request.get('parsed')]).decode())
    if meta['layer'] == 'dashboard':
        return ('dashboard', 'latest')
    return (meta['layer'], meta.get('object_id') or version)


def noise(path):
    """Interpreter caches and CLI session logs are captured but never shown as handles."""
    parts = path.split('/')
    return '__pycache__' in parts or parts[0] == 'sessions'


class PFQueries:
    def __init__(self, registry, *, stale_checks=True, refresh=None):
        if registry.mode != 'pf':
            raise ValueError('PF queries require PF mode')
        self.store, self.resolver = registry.store, registry.resolver
        self.workspace = registry.workspace
        self.stale_checks, self.refresh = stale_checks, refresh

    def decorate(self, capture, text, after):
        kind = self.store.read_event(capture.event)['request']['kind']
        if kind == 'read_file':
            for item in capture.origins:
                meta, _ = self.resolver.content(item['version_id'])
                if meta['layer'] == 'file_text':
                    return str(text) + '\n[' + self.resolver.handle(meta['derived_from']) + ']'
        if kind != 'bash':
            return text
        with closing(self.store.connect()) as conn:
            queries = conn.execute('''WITH RECURSIVE children(event_id) AS (
                SELECT ? UNION SELECT r.event_id FROM requests r JOIN children c
                ON json_extract(r.request, '$.parent_event_id')=c.event_id)
                SELECT v.version_id FROM versions v JOIN requests r USING(event_id)
                WHERE r.event_id IN children AND r.query_id IS NOT NULL
                AND json_extract(v.metadata, '$.layer')='server_public_response'
                ORDER BY v.rowid''', (capture.event,)).fetchall()
        entries = [('q', None, row[0]) for row in queries]
        entries += [('写', path, after[path]['version']) for path in capture.facts.get('changed_paths', [])
                    if after and after.get(path, {}).get('version') and not noise(path)]
        if not entries:
            return text
        # Only displayed entries receive handles: [输出: v43 | q: v40 v41 | 写: forecast.json v42].
        # 输出 names this exact tool return (saved by capture.finish under a fixed id), so the
        # printed result of a script can be cited with its queries traced as upstream.
        groups = {'输出': [self.resolver.handle(capture.event + ':tool_return')]}
        for group, path, version in entries[:8]:
            groups.setdefault(group, []).append((path + ' ' if path else '') + self.resolver.handle(version))
        displayed = [group + ': ' + ' '.join(items) for group, items in groups.items()]
        if len(entries) > 8:
            displayed.append(f'另有 {len(entries) - 8} 项')
        return str(text) + '\n[' + ' | '.join(displayed) + ']'

    def execute(self, operation, args):
        """Agent-facing result: compact text, one line per item."""
        page = self.last_answer = self.answer(operation, args)  # structured form kept for audits
        if isinstance(page, str):
            return page  # pf_read content and diff keep their header line and exact body
        if operation == 'pf_search':
            wanted = self.values['object']
            return pf_render.render_list(page, f"pf_search {wanted['kind']}={wanted['id']}", wanted)
        if operation == 'pf_read':
            return pf_render.render_list(page, 'Versions of ' + page['root']['what'])
        if operation == 'pf_dependents':
            return pf_render.render_dependents(page)
        return pf_render.render_dependencies(page)

    def answer(self, operation, args):
        """Structured page for one query; pf_read content and diff return their delivered text."""
        try:
            values = MODELS[operation].model_validate(args).model_dump(exclude_none=True)
        except ValidationError as exc:
            raise ValueError('; '.join('.'.join(map(str, e['loc'])) + ': ' + e['msg']
                                      for e in exc.errors(include_input=False))) from exc
        offset, cutoff = 0, None
        self.check_state = None
        self.cursor_name = 'pf_cursors'
        self.cursors = self.store.load_state(self.cursor_name) or {}
        if 'cursor' in values:
            if set(args) != {'cursor'}:
                raise ValueError('Continue with cursor only, or omit cursor to start a new query')
            saved = self.cursors.get(values['cursor'])
            if not saved or saved['operation'] != operation:
                raise ValueError('Unknown query cursor; start a new query')
            values, offset, cutoff = saved['values'], saved['offset'], saved['cutoff']
            self.check_state = saved.get('check_state')
        self._index(cutoff)
        self.operation, self.values = operation, values
        if operation == 'pf_search':
            if 'object' not in values:
                raise ValueError('object is required')
            wanted = values['object']
            matches = [v for v in self.nodes if not self._rerun(v) and any(
                       o['kind'] == wanted['kind'] and str(o['id']) == wanted['id'] for o in self._objects(v))]
            return self._page(matches[::-1], offset, self._describe)
        if 'target' not in values:
            raise ValueError('target is required')
        target = self._target(values['target'])
        if operation == 'pf_read':
            mode = values['mode']
            if (mode == 'diff') != ('baseline' in values):
                raise ValueError('baseline is required only for diff')
            if values['full'] and mode != 'content':
                raise ValueError('full is supported only for content reads')
            if values['detail'] and mode != 'history':
                raise ValueError('detail is supported only for history')
            if mode == 'history':
                key = self._key(target)
                listed = [v for v in self.nodes if self._key(v) == key and not self._rerun(v)][::-1]
                return self._page(listed, offset, self._describe, root=self._describe(listed[0] if listed else target))
            return self._read(target, offset)
        # Design 3.4: every query view on the way reruns, including a derived file's
        # upstream on the capture layer, so current-purpose tracing follows execution.
        current = operation == 'pf_dependencies' and values['purpose'] == 'current'
        self.follow_execution = values['include_execution'] or current
        checking = current and self.stale_checks
        if self.check_state:
            rows = self.store.load_state(self.check_state)
        else:
            rows = self._trace(target, operation == 'pf_dependents')
            if checking:
                event = CURRENT_EVENT.get()
                own_event = event is None
                if own_event:
                    event = self.store.begin_event('pf_dependencies', values)
                token = CURRENT_EVENT.set(event)
                try:
                    self._check(rows)
                    self.check_state = 'pf_check:' + event
                    self.store.save_state(self.check_state, rows)
                    if own_event:
                        self.store.complete(event)
                finally:
                    CURRENT_EVENT.reset(token)
        describe = lambda r, all_rows=rows: self._top(r, all_rows) if _depth(r) == 1 else self._describe_edge(r)
        if not values['detail']:
            # Compact: one line per declared reference (or referrer); deeper rows only summarized.
            if operation == 'pf_dependencies':
                rows, describe = [r for r in rows if _depth(r) == 1], lambda r, all_rows=rows: self._top(r, all_rows)
            else:
                seen, endpoints = set(), []
                for row in rows:
                    source = row['edge']['source']
                    if source in self.records and not row['traversal_only'] and source not in seen:
                        seen.add(source)
                        endpoints.append(row)
                rows = endpoints
        return self._page(rows, offset, describe, root=self._describe(target),
                          stale_check='performed' if checking else 'not_performed')

    def weekly_check(self, day):
        """Week-start digest: rerun what active registered texts cite, list only what changed."""
        if not self.stale_checks:
            return None
        self._index(None)
        self.operation = 'pf_dependencies'
        self.values = dict(depth=3, purpose='current', include_execution=False, detail=False, limit=100)
        self.follow_execution = True
        active = sorted((v for v, record in self.records.items()
                         if record['status'] == 'active' and self.latest[self._key(v)] == v),
                        key=lambda v: int(self.records[v]['id'][1:]), reverse=True)
        traced = {}
        for version in active:
            rows = self._trace(version, False)
            if any(_weekly(r) for r in rows):
                traced[version] = rows
        if not traced:
            return None
        event = self.store.begin_event('pf_weekly_check', {'day': day})
        token = CURRENT_EVENT.set(event)
        try:
            self._check([row for rows in traced.values() for row in rows])
            entries = []
            for version, rows in traced.items():
                tops = [self._top(r, rows) for r in rows if _weekly(r)]
                entries.append(dict(text=self._describe(version), total=len(tops), changed=[
                    t for t in tops if t['check']['affected'] or t['check']['predicate_result'] in ('fails', 'cannot_check')]))
            text = pf_render.render_weekly(entries, day, pf=True)
            version = self.store.version(event, 'weekly_check', text, layer='weekly_check')
            self.store.complete(event)
        finally:
            CURRENT_EVENT.reset(token)
        return CapturedText(text, [origin(version, text)])

    def _check(self, rows):
        from .pf_stale import StaleCheck
        return StaleCheck(self).run(rows)

    def _top(self, row, rows):
        """A direct dependency, with a count of the underlying sources that changed."""
        result = self._describe_edge(row)
        check = row.get('check')
        if not check or row['historical_only']:
            return result
        below = [r for r in rows if _depth(r) > 1 and r['path'][:2] == row['path'] and r.get('check')
                 and r['edge']['target'] in self.nodes
                 and self.nodes[r['edge']['target']]['meta']['layer'] in SOURCE_LAYERS]
        failed = [r for r in below if r['check']['predicate_result'] == 'fails']
        changed = [r for r in below if r['check']['version_changed'] and r['check']['predicate_result'] in
                   ('not_declared', 'not_checked')]
        unknown = [r for r in below if r['check']['predicate_result'] == 'cannot_check']
        parts = []
        for rows_, head in ((failed, 'PREDICATE FAILS on'), (changed, 'changed:')):
            if rows_:
                example = self._describe_edge(rows_[0])
                parts.append(f"{len(rows_)} of {len(below)} underlying sources {head} "
                             f"{example['target']['what']}: {example['check'].get('summary', 'changed')}")
        if unknown:
            reasons = ', '.join(sorted({str(r['check']['reason']) for r in unknown}))
            parts.append(f'{len(unknown)} could not be checked ({reasons})')
        if parts:
            result['check']['sources'] = '; '.join(parts)
        return result

    def _index(self, cutoff):
        # ponytail: rebuild O(versions + edges) per query/page; persist an index if
        # measured history-query latency becomes material in longer experiments.
        self.events, self.nodes, self.records = {}, {}, {}
        self.outgoing, self.incoming = defaultdict(list), defaultdict(list)
        self.latest, self.reads, self.event_day = {}, defaultdict(list), {}
        day = 0
        with closing(self.store.connect()) as conn:
            conn.execute('BEGIN')
            self.cutoff = cutoff if cutoff is not None else conn.execute('SELECT coalesce(max(rowid),0) FROM versions').fetchone()[0]
            for row in conn.execute('''SELECT r.event_id,r.request,s.record,q.definition FROM requests r
                    LEFT JOIN results s USING(event_id) LEFT JOIN queries q ON r.query_id=q.id
                    ORDER BY r.rowid'''):
                try:
                    self.store._visible(conn, row['event_id'])
                except KeyError:
                    continue
                request = json.loads(row['request'])
                # The server records each weekly dashboard with its simulated day; later
                # events happen on that day. Agents reason in simulated days, not clock time.
                if request['kind'] == 'dashboard_generation':
                    day = (request.get('request') or {}).get('day', day)
                self.event_day[row['event_id']] = day
                self.events[row['event_id']] = dict(request=request,
                    result=json.loads(row['record']) if row['record'] else {},
                    query=json.loads(row['definition']) if row['definition'] else None)
            versions = conn.execute('SELECT rowid,* FROM versions WHERE rowid<=? ORDER BY rowid', (self.cutoff,)).fetchall()
        private = []
        for row in versions:
            if row['event_id'] not in self.events:
                continue
            meta = json.loads(row['metadata'])
            event = self.events[row['event_id']]
            if meta['layer'] in PUBLIC_LAYERS and not event['request']['kind'].startswith('pf_'):
                self.nodes[row['version_id']] = dict(row, meta=meta)
                if meta['layer'] == 'registered_text':
                    record = json.loads(self.resolver.content(row['version_id'])[1])
                    self.records[row['version_id']] = record
                if meta['layer'] == 'server_public_response' and event['query']:
                    body = json.loads(self.resolver.content(row['version_id'])[1])
                    meta['objects'] = [dict(kind=OBJECT_FIELDS[key], value=value,
                        basis='public_result_column', path=f'/rows/{i}/{key}')
                        for i, result in enumerate(body.get('rows', [])) for key, value in result.items()
                        if key in OBJECT_FIELDS and type(value) in (str, int)]
                self.latest[self._key(row['version_id'])] = row['version_id']
            elif meta['layer'] in ('agent_declaration', 'model_source_occurrences', 'model_reconstructions', 'workspace_boundary'):
                private.append((row['version_id'], row['event_id'], meta['layer']))
        files, execution_outputs, execution_inputs = {}, defaultdict(list), defaultdict(list)
        codes = defaultdict(list)
        for version, row in self.nodes.items():
            if row['meta']['layer'] == 'executed_code':
                codes[row['event_id']].append(version)
        for version, row in self.nodes.items():
            meta, event = row['meta'], self.events[row['event_id']]
            self._edge(version, meta.get('derived_from'), 'derived_from')
            for segment in meta.get('segments', []):
                self._edge(version, segment['version_id'], 'returned_range',
                           source_range=segment['source_range'], output_range=segment['request_range'])
            for field in meta.get('public_fields', []):
                for segment in field['origins']:
                    self._edge(version, segment['version_id'], 'public_field_range',
                               source_range=segment['source_range'], field=field['pointer'])
            if meta['layer'] == 'file_bytes':
                key = meta['object_id']
                previous = files.get(key)
                if previous and self.nodes[previous]['content_hash'] == row['content_hash']:
                    self._edge(version, previous, 'same_content_observation')
                else:
                    files[key] = version
            if meta['layer'] in ('executed_code', 'server_public_response'):
                parent = row['event_id']
                seen = set()
                while parent in self.events and parent not in seen:
                    seen.add(parent)
                    execution_inputs[parent].append(version)
                    if meta['layer'] == 'server_public_response':
                        for code in codes[parent]:
                            self._edge(version, code, 'executed_by')
                    parent = self.events[parent]['request'].get('parent_event_id')
            if meta['layer'] == 'file_bytes' and event['request']['kind'] == 'edit_file' and version.endswith('_read'):
                execution_inputs[row['event_id']].append(version)
        for version, event_id, layer in private:
            value = json.loads(self.resolver.content(version)[1])
            if layer == 'agent_declaration' and value['version_id'] in self.records:
                source = value['version_id']
                refs = self.records[source]['references']
                for i, ref in enumerate(refs):
                    binding = value['references'][i] if i < len(value['references']) else {}
                    resolved = binding.get('status') == 'resolved' and binding.get('git_content_matches') is not False
                    target = binding.get('version_id') if resolved else None
                    reason = (None if resolved else 'git_content_mismatch' if binding.get('git_content_matches') is False
                              else binding.get('reason', 'not_captured_in_prefix'))
                    self._edge(source, target, 'reference', origin='agent_declaration',
                               reference=ref, reason=reason or ('captured_version_unavailable' if target not in self.nodes else None),
                               missing=target not in self.nodes)
            elif layer in ('model_source_occurrences', 'model_reconstructions'):
                event = self.events[event_id]
                if event['result'].get('send_state') == 'response_received' and event['result'].get('status') == 'succeeded':
                    for item in value:
                        source = item['version_id']
                        reading = dict(day=self.event_day.get(event_id), range=item['source_range'],
                                       full_source=item['full_source'])
                        self.reads[source].append(reading)
                        meta = self.nodes.get(source, {}).get('meta', {})
                        if meta.get('layer') in ('file_text', 'query_model_projection'):
                            self.reads[meta['derived_from']].append(dict(reading, representation=meta['layer']))
            elif layer == 'workspace_boundary' and version.endswith(':workspace_after'):
                for path in self.events[event_id]['result'].get('changed_paths', []):
                    item = value.get(path, {})
                    if item.get('version') in self.nodes:
                        execution_outputs[event_id].append(item['version'])
        for version, row in self.nodes.items():
            # A command's printed result depends on whatever its execution queried.
            if row['meta']['layer'] in ('tool_return', 'stdout', 'stderr'):
                execution_outputs[row['event_id']].append(version)
        for event, outputs in execution_outputs.items():
            for output in outputs:
                for source in execution_inputs[event]:
                    self._edge(output, source, 'observed_edit_input' if self.nodes[source]['meta']['layer'] == 'file_bytes' else 'same_execution')
        for edges in self.outgoing.values():
            edges.sort(key=lambda e: e['origin'] != 'agent_declaration')
        for edges in self.incoming.values():
            edges.sort(key=lambda e: e['origin'] != 'agent_declaration')
        # Reads, edits and every Bash boundary snapshot store their own version of
        # unchanged bytes; traversal treats identical bytes of one path as one node.
        self.same_bytes, self.members = {}, defaultdict(list)
        for version, row in self.nodes.items():
            if row['meta']['layer'] == 'file_bytes':
                key = (row['meta']['object_id'], row['content_hash'])
                self.same_bytes[version] = key
                self.members[key].append(version)

    def _key(self, version):
        row = self.nodes[version]
        meta, event = row['meta'], self.events[row['event_id']]
        return evidence_key(version, meta, event['query'], event['result'], event['request'].get('request'))

    def _edge(self, source, target, kind, origin='automatic_capture', **details):
        if target not in self.nodes and origin != 'agent_declaration':
            return
        edge = dict(source=source, target=target if target in self.nodes else None,
                    kind=kind, origin=origin, **details)
        self.outgoing[source].append(edge)
        if edge['target']:
            self.incoming[target].append(edge)

    def _objects(self, version):
        if version in self.records:
            return [dict(o, basis='agent_declaration') for o in self.records[version]['objects']]
        return [dict(kind=o['kind'], id=o['value'], basis=o['basis'], field=o.get('path'))
                for o in self.nodes[version]['meta'].get('objects', [])]

    def _target(self, target):
        if 'version' in target:
            handles = self.store.load_state(HANDLES) or {}
            version = handles.get(target['version'])
            if version not in self.nodes:
                raise ValueError('Unknown or unavailable version handle; use a path or SQL instead')
            return version
        matches = []
        for version, row in self.nodes.items():
            meta, event = row['meta'], self.events[row['event_id']]
            if 'path' in target and meta['layer'] == 'file_bytes' and meta['object_id'] == target['path']:
                matches.append(version)
            elif 'sql' in target and meta['layer'] == 'server_public_response' and event['query'] and event['query'][5] == target['sql']:
                matches.append(version)
            elif 'record' in target and version in self.records:
                record = self.records[version]
                if target['record'] in (record['id'], record['version']):
                    matches.append(version)
        if not matches:
            raise ValueError('No captured version matches; use another path, SQL or registered text')
        return matches[-1]

    def _describe(self, version):
        """What the agent needs about one version: handle, simulated day, what it is, problems."""
        row = self.nodes[version]
        meta, event = row['meta'], self.events[row['event_id']]
        request = event['request'].get('request') or {}
        query = event['query'] if meta['layer'] == 'server_public_response' else None
        record = self.records.get(version)
        result = dict(version=self.resolver.handle(version),
            what=pf_render.label(meta['layer'], event['request']['kind'], request, query[5] if query else None,
                                 meta.get('object_id'), event['result'].get('classification'), record),
            day=record['sim_day'] if record and record.get('sim_day') is not None else self.event_day.get(row['event_id']),
            layer=meta['layer'], operation=event['request']['kind'],
            status=event['result'].get('status', 'result_unknown'), objects=self._objects(version),
            relation_origin='agent_declaration' if record else 'automatic_capture')
        if meta['source_truncated']:
            result['truncated'] = True
        if meta['extent'] != 'full':
            result['extent'] = meta['extent']
        if event['result'].get('capture_gaps'):
            result['capture_gaps'] = event['result']['capture_gaps']
        if about := _about(meta['content_time']):
            result['about'] = about
        if self._rerun(version):
            result['what'] += ' [rerun by PF check]'
        if meta['layer'] == 'file_bytes':
            result['path'] = meta['object_id']
        if query:
            result.update(sql=query[5], bound_parameters=query[6],
                          result_coverage=event['result'].get('result_coverage', 'unknown'))
            try:
                result['rows'] = len(json.loads(self.resolver.content(version)[1])['rows'])
            except (ValueError, KeyError, TypeError):
                pass
        if 'classification' in event['result']:
            result['classification'] = event['result']['classification']
        if record:
            result.update(record=record['version'], text=record['text'], text_status=record['status'],
                          reason=record['reason'], applies_at=record['applies_at'],
                          current_revision=self.latest[self._key(version)] == version)
        if row['previous_version'] in self.nodes:
            result['previous_version'] = self.resolver.handle(row['previous_version'])
        reads = self.reads[version]
        result['model_reads'] = dict(count=len(reads), full=any(r['full_source'] for r in reads),
                                     last_day=reads[-1]['day'] if reads else None)
        return result

    def _rerun(self, version):
        """Harness reruns of cited reads: diff targets, but not the agent's own history."""
        parent = self.events.get(self.events[self.nodes[version]['event_id']]['request'].get('parent_event_id') or '')
        return bool(parent) and parent['request']['kind'].startswith('pf_')

    def _brief(self, version):
        full = self._describe(version)
        return {k: full[k] for k in ('version', 'day', 'what', 'truncated', 'extent') if k in full} | (
            {'status': full['status']} if full['status'] != 'succeeded' else {})

    def _change(self, before, after):
        """How the current version of a dependency differs from the cited one."""
        if after in self.records:
            record = self.records[after]
            return 'retired as ' + record['version'] if record['status'] == 'retired' else 'revised to ' + record['version']
        meta = self.resolver.content(before)[0]
        kind = ('query' if meta['layer'] == 'server_public_response' and self.events[meta['created_by_event']]['query']
                else 'public' if meta['layer'] == 'server_public_response'
                else 'json' if (meta.get('object_id') or '').endswith('.json') else 'text')
        return pf_render.change(kind, self.resolver.content(before)[1], self.resolver.content(after)[1])

    def _value(self, before, after, reference):
        """The cited cell(s) then and now, e.g. '>= 770: now 721 (was 812)'."""
        from .pf_stale import cell
        meta = self.resolver.content(before)[0]
        kind = 'csv' if (meta.get('object_id') or '').endswith('.csv') else 'json'
        try:
            predicate = reference.get('predicate') or {}
            if predicate.get('type') == 'compare':
                left, right = (cell(self.resolver.content(after)[1], predicate[k], kind) for k in ('left', 'right'))
                return f"{pf_render.number(left)} {predicate['op']} {pf_render.number(right)} now"
            old, new = (cell(self.resolver.content(v)[1], reference['select'], kind) for v in (before, after))
        except (ValueError, KeyError, TypeError, IndexError):
            return 'value unavailable'
        n = pf_render.number
        if predicate.get('type') == 'threshold':
            return f"{predicate['op']} {n(predicate['value'])}: now {n(new)} (was {n(old)})"
        if predicate.get('type') == 'tolerance':
            return f"within {n(predicate['amount'])} of {n(old)}: now {n(new)}"
        return f'now {n(new)} (was {n(old)})'

    def _node(self, version):
        return self.same_bytes.get(version, version)

    def _edges(self, node, reverse):
        """Edges of a version and of every captured observation of the same file bytes."""
        members = self.members.get(self.same_bytes.get(node), [node])
        edges, side = (self.incoming, 'target') if reverse else (self.outgoing, 'source')
        result = []
        for member in members:
            for edge in edges[member]:
                if edge['kind'] == 'same_content_observation':
                    continue
                result.append(edge if member == node else dict(edge, **{side: node, 'observed_as': member}))
        result.sort(key=lambda e: e['origin'] != 'agent_declaration')
        return result

    def _trace(self, root, reverse):
        follow = self.follow_execution
        queue, scheduled, rows = deque([(root, [root], False)]), {(self._node(root), False)}, []
        while queue:
            node, path, historical = queue.popleft()
            for edge in self._edges(node, reverse):
                if edge['origin'] != 'agent_declaration' and not follow:
                    continue
                target = edge['source'] if reverse else edge['target']
                current = (not reverse or not self.values['current_only'] or target not in self.records or
                           (self.latest[self._key(target)] == target and self.records[target]['status'] == 'active'))
                pure_history = historical or edge.get('reference', {}).get('purpose') == 'historical_only'
                cycle = target is not None and self._node(target) in {self._node(v) for v in path}
                depth_limit = len(path) >= self.values['depth']
                more = target is not None and any(e['origin'] == 'agent_declaration' or follow
                                                  for e in self._edges(target, reverse))
                stop = ('missing' if target is None else 'cycle' if cycle else
                        'already_expanded' if (self._node(target), pure_history) in scheduled else
                        'depth_limit' if depth_limit and more else None)
                # Older revisions can lead to active referrers; filter endpoints, not traversal.
                if current or stop == 'depth_limit':
                    rows.append(dict(edge=edge, path=path + ([target] if target else []),
                                     historical_only=pure_history, stop=stop, traversal_only=not current))
                if target and not stop and not depth_limit:
                    scheduled.add((self._node(target), pure_history))
                    queue.append((target, path + [target], pure_history))
        return rows

    def _describe_edge(self, item):
        edge = item['edge']
        result = {k: edge[k] for k in ('kind', 'origin', 'missing', 'reason') if k in edge}
        result.update(relation=RELATION.get(edge['kind'], edge['kind']), source=self._describe(edge['source']),
                      target=self._describe(edge['target']) if edge['target'] else None,
                      path=[self.resolver.handle(v) for v in item['path']], depth=_depth(item),
                      historical_only=item['historical_only'], unexpanded=item['stop'],
                      traversal_only=item['traversal_only'])
        if 'reference' in edge:
            result.update(edge['reference'])
        if 'check' in item:
            check = dict(item['check'])
            current = check['current_version']
            if edge['target'] and current:
                if 'reference' in edge and (edge['reference'].get('select') or edge['reference'].get('predicate')) \
                        and check['predicate_result'] in ('holds', 'fails'):
                    check['summary'] = self._value(edge['target'], current, edge['reference'])
                elif check['version_changed']:
                    check['summary'] = self._change(edge['target'], current)
            check['current_version'] = self.resolver.handle(current) if current else None
            check['affected_paths'] = [[self.resolver.handle(v) for v in path] for path in check['affected_paths']]
            result['check'] = check
        return result

    def _cursor(self, offset):
        value = dict(operation=self.operation, values=self.values, offset=offset, cutoff=self.cutoff)
        if self.check_state:
            value['check_state'] = self.check_state
        cursor = next((k for k, v in self.cursors.items() if v == value), None)
        if cursor is None:
            cursor = 'c' + str(len(self.cursors) + 1)
            self.cursors[cursor] = value
            self.store.save_state(self.cursor_name, self.cursors)
        return cursor

    def _page(self, rows, offset, describe, **extra):
        end = min(len(rows), offset + self.values['limit'])
        return dict(items=[describe(v) for v in rows[offset:end]],
            next_cursor=self._cursor(end) if end < len(rows) else None,
            remaining=len(rows) - end, total=len(rows), detail=self.values['detail'], **extra)

    def _read(self, target, offset):
        meta, raw = self.resolver.content(target)
        try:
            content = raw.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ValueError('This captured version is not UTF-8 text') from exc
        header = {}
        if self.values['mode'] == 'diff':
            baseline = self._target(self.values['baseline'])
            key = self._key(target)
            if key != self._key(baseline) or key[0] not in ('query', 'file_bytes', 'registered_text', 'registered_script', 'dashboard'):
                raise ValueError('Diff requires two versions of the same captured object')
            old_meta, old = self.resolver.content(baseline)
            try:
                before = old.decode('utf-8')
            except UnicodeDecodeError as exc:
                raise ValueError('Diff requires UTF-8 text') from exc
            header.update(direction='baseline_to_target', raw_equal=old == raw)
            if key[0] == 'query':
                events = [self.events[self.nodes[v]['event_id']] for v in (baseline, target)]
                comparison = [e['result'].get('comparison', {}) for e in events]
                # Rows compared after canonical sorting; null when either result lacks a comparison.
                header['rows_equal'] = None
                if all(c.get('status') == 'available' for c in comparison):
                    contents = [self.resolver.content(self.nodes[v]['event_id'] + ':comparison')[1] for v in (baseline, target)]
                    header['rows_equal'] = contents[0] == contents[1]
                    if meta['source_truncated'] or old_meta['source_truncated']:
                        header['rows_scope'] = 'returned_subset'
            header['baseline'] = self._brief(baseline)
            # splitlines with terminators preserves final-newline differences.
            lines = difflib.unified_diff(before.splitlines(keepends=True), content.splitlines(keepends=True),
                                        fromfile=header['baseline']['version'], tofile=self.resolver.handle(target))
            content = ''.join(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n' for line in lines)
        header['target'] = self._brief(target)
        end = min(len(content), offset + 30000)
        header.update(range=[offset, end], total_chars=len(content),
                      next_cursor=self._cursor(end) if end < len(content) else None)
        if end < len(content):
            header['truncated_reason'] = 'character_limit'
        prefix = encoded(header).decode() + '\n'
        from .pf_read import capture_read
        return capture_read(self, target, prefix + content[offset:end],
                            offset, end, len(content))
