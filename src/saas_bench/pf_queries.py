"""PF-only historical queries, rebuilt from immutable capture and declaration records."""
from collections import Counter, defaultdict, deque
from contextlib import closing
import difflib
import json
import re
import shlex
import time
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, ValidationError, model_validator

from .execution_capture import CapturedText, OBJECT_FIELDS, origin
from .pf_deadline import budget_scope, check
from . import evidence_handles, pf_render
from .registration_evidence import covered
from .registration_schema import Input, Text
from .sql_evidence import PUBLIC_LAYERS, encoded


class Target(Input):
    path: Text | None = None
    sql: Text | None = None
    operation_ref: Annotated[str, Field(pattern=r"^op:[0-9a-f]{32}$")] | None = None
    record: Annotated[str, Field(pattern=r'^(?:(?:ceo|growth|ops_finance):)?r[1-9][0-9]*(\.[1-9][0-9]*)?$')] | None = None
    version: Annotated[str, Field(pattern=evidence_handles.HANDLE_PATTERN)] | None = None

    @model_validator(mode='after')
    def captured_target(self):
        if sum(v is not None for v in (self.path, self.sql, self.record, self.version, self.operation_ref)) != 1:
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


class Candidates(Input):
    cursor: Annotated[str, Field(pattern=r'^c[1-9][0-9]*$')] | None = None
    role: Literal['ceo', 'growth', 'ops_finance', 'all'] | None = None
    history: bool = False
    day_from: Annotated[int, Field(ge=0)] | None = None
    day_to: Annotated[int, Field(ge=0)] | None = None
    kind: Literal['file', 'sql', 'api', 'stdout', 'stderr', 'output', 'code', 'text', 'script', 'dashboard', 'execution'] | None = None
    text: Annotated[str, Field(min_length=1, max_length=2000)] | None = None
    event: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    detail: bool = False

    @model_validator(mode='after')
    def ordered_days(self):
        if self.day_from is not None and self.day_to is not None and self.day_from > self.day_to:
            raise ValueError('day_from must not exceed day_to')
        return self


class SearchObject(Input):
    id: Text
    kind: Text | None = None


class Search(Page):
    object: SearchObject | None = None
    text: Text | None = None
    all: bool = False

    @model_validator(mode='after')
    def search_mode(self):
        if not self.cursor and (self.object is None) == (self.text is None):
            raise ValueError('Specify one object or literal text')
        if self.text is not None and self.all:
            raise ValueError('Text search already includes historical versions')
        return self


class Versions(Page):
    target: Target | None = None


class Single(Input):
    target: Target | None = None


class More(Input):
    cursor: Annotated[str, Field(pattern=r'^c[1-9][0-9]*$')]


class Trace(Page):
    target: Target | None = None
    depth: Annotated[int, Field(ge=1, le=10)] = 1
    include_execution: bool = False
    include_candidates: bool = False
    all_revisions: bool = False


class Dependents(Trace):
    current_only: bool = True


class Dependencies(Trace):
    purpose: Literal['current', 'historical_only'] = 'current'


class Read(Page):
    target: Target | None = None
    mode: Literal['history', 'content', 'diff', 'captures'] = 'content'
    baseline: Target | None = None
    full: bool = False


MODELS = dict(pf_search=Search, pf_dependencies=Dependencies, pf_dependents=Dependents, pf_read=Read,
              pf_log=Versions, pf_diff=Single, pf_blame=Single, pf_more=More, pf_candidates=Candidates)


RELATION = dict(reference='cites', same_execution='observed in same execution as', returned_range='printed from',
                public_field_range='field from', executed_by='run by', derived_from='derived from',
                observed_edit_input='edited from')


def _depth(row):
    """Edges from the queried root have depth 1; a missing target adds no node to the path."""
    return len(row['path']) - (1 if row['edge']['target'] else 0)


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


def evidence_key(version, meta, query, result, request, record=None):
    """The object whose versions are listed and diffed together."""
    record = record or {}
    if meta['layer'] in ('stdout', 'stderr', 'executed_code') and record.get('kind') == 'cli_python' \
            and (record.get('request') or {}).get('script'):
        # Every run of one script file: its outputs are versions of one object.
        return ('script_' + meta['layer'], record['request']['script'])
    if meta['layer'] in ('tool_return', 'stdout', 'stderr') and record.get('call'):
        return ('command_' + meta['layer'], encoded(record['call']).decode())
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
    return '__pycache__' in parts or parts[0] in ('sessions', '.tool-notes.jsonl')


class PFQueries:
    def __init__(self, registry, *, call_timeout=None):
        from .runtime_options import runtime_value
        self.call_timeout = runtime_value('pf_timeout', 600) if call_timeout is None else call_timeout
        if registry.mode != 'pf':
            raise ValueError('PF queries require PF mode')
        self.store, self.resolver = registry.store, registry.resolver
        self.registry = registry
        self.workspace = registry.workspace

    def decorate(self, capture, text, after):
        """The [pf: ...] line after a tool return: what it can be cited as, what it wrote,
        and notes on files whose full content it shows."""
        record = self.store.read_event(capture.event)['request']
        handles = evidence_handles.index(self.store)
        note = (record.get('request') or {}).get('note')
        noted = (['note saved' + (' (truncated to 200 characters)' if capture.facts.get('note_truncated') else '')]
                 if note else [])
        written = [after[path]['version'] for path in capture.facts.get('changed_paths', [])
                   if after and after.get(path, {}).get('version') and not noise(path)]
        wrote = ['wrote ' + ', '.join(self.resolver.handle(v) for v in written[:6]) +
                 (f' and {len(written) - 6} more' if len(written) > 6 else '')] if written else []
        if record['kind'] == 'read_file':
            for item in capture.origins:
                meta, _ = self.resolver.content(item['version_id'])
                if meta['layer'] == 'file_text':
                    return str(text) + '\n[pf: ' + self.resolver.handle(meta['derived_from']) + \
                        self._noted(meta['derived_from']) + ']'
            return text
        if record['kind'] in ('write_file', 'edit_file'):
            parts = wrote + noted
            return str(text) + '\n[pf: ' + ' | '.join(parts) + ']' if parts else text
        if record['kind'] != 'bash':
            return text
        with closing(self.store.connect()) as conn:
            observed = conn.execute('''WITH RECURSIVE children(event_id) AS (
                SELECT ? UNION SELECT r.event_id FROM requests r JOIN children c
                ON json_extract(r.request, '$.parent_event_id')=c.event_id
                WHERE json_extract(r.request, '$.kind') NOT GLOB 'pf_*')
                SELECT v.version_id,r.query_id,json_extract(r.request, '$.kind') AS kind FROM requests r LEFT JOIN versions v
                ON v.event_id=r.event_id AND json_extract(v.metadata, '$.layer')='server_public_response'
                WHERE r.event_id IN children AND json_extract(r.request, '$.kind') IN ('sql_query','public_http')
                ORDER BY r.rowid''', (capture.event,)).fetchall()
            scripts = conn.execute('''SELECT r.event_id, v.version_id FROM requests r
                LEFT JOIN versions v ON v.version_id=r.event_id || ':stdout'
                WHERE json_extract(r.request, '$.parent_event_id')=? AND json_extract(r.request, '$.kind')='cli_python'
                AND json_extract(r.request, '$.request.script') IS NOT NULL ORDER BY r.rowid''',
                (capture.event,)).fetchall()
        body = str(text)
        printed = []
        for child, version in scripts:
            if version is None:
                event = self.store.read_event(child)
                if (event['result'].get('completion') == 'unfinished'
                        and not {child + ':python_end', child + ':stdout_bytes'} & set(event['outputs'])):
                    continue
                if child + ':stdout_bytes' in event['outputs']:
                    from .execution_capture import decoded
                    raw = self.resolver.content(child + ':stdout_bytes')[1]
                    try:
                        decoded(raw)
                    except UnicodeError:
                        continue
                version = child + ':stdout'
            try:
                output = self.resolver.content(version)[1].decode('utf-8')
            except (RuntimeError, UnicodeDecodeError):
                continue
            if output:
                shown = [o for o in capture.origins if o['version_id'] == version]
                printed.append((version, covered([(0, len(output))], shown) if shown else False, output == body))
        capture.facts['pf_retrieval'] = bool(capture.facts.get('pf_calls')) and not observed and not printed
        if capture.facts['pf_retrieval']:
            return body + '\n[pf: ' + ' | '.join(wrote + noted) + ']' if wrote else text
        shown_files = self._shown_notes(after, written, body)
        if not observed and not printed and not written:
            return body + ''.join('\n' + line for line in shown_files) if shown_files else text
        again = lambda day: '' if day is None else f' (same as day {day})'
        # The first item names this return itself: a script's stdout when the return is exactly
        # that output, otherwise the command return (saved by capture.finish), so what was seen
        # is always citable. Scripts shown only in part are named as such.
        if len(printed) == 1 and printed[0][2]:
            version = printed[0][0]
            output = self.resolver.handle(version) + again(handles.repeated(version))
        else:
            name, day = handles.pending('cmd', record.get('call'), evidence_handles.body_digest(body))
            output = name + again(day)
            if printed:
                output += ', with ' + ' and '.join(('' if full else 'part of ') + self.resolver.handle(v)
                                                   for v, full, _ in printed)
        queries = [row['version_id'] for row in observed if row['query_id'] and row['version_id']]
        sql_count = sum(row['kind'] == 'sql_query' for row in observed)
        query = [self.resolver.handle(queries[0])] if len(queries) == 1 else []
        counts = []
        if len(observed) > 1:
            counts = [f'{sql_count} SQL, {len(observed) - sql_count} API']
        cursors = self.store.load_state('pf_cursors') or {}
        cursor = 'c' + str(len(cursors) + 1)
        cursors[cursor] = dict(operation='pf_read', values=dict(mode='captures',
            capture_event=capture.event, limit=20, detail=False, scope='this execution'), offset=0, cutoff=None)
        self.store.save_state('pf_cursors', cursors)
        line = '[pf: ' + ' | '.join([output] + query + counts + wrote + noted + ['captures: pf more ' + cursor]) + ']'
        return body + '\n' + line + ''.join('\n' + line for line in shown_files)

    def _noted(self, version):
        note = evidence_handles.index(self.store).note(version)
        return f' note (day {note[0] if note[0] is not None else "unknown"}): "{pf_render.short(note[1], 200)}"' if note else ''

    def _shown_notes(self, after, written, body):
        """Notes on workspace files whose whole content this output shows, e.g. after cat."""
        lines, handles = [], evidence_handles.index(self.store)
        handles.refresh()
        for path, item in (after or {}).items():
            version = item.get('version')
            group = handles.info.get(version, (None, None))[1] if version else None
            if not group or not group.get('note') or version in written or noise(path):
                continue
            text = self._text(version)
            if text and text.strip() and text.strip() in body:
                lines.append('[pf: ' + self.resolver.handle(version) + self._noted(version) + ']')
        return lines[:3]

    def execute(self, operation, args):
        """Agent-facing result: compact text, one line per item."""
        with budget_scope():
            if operation == 'pf_more':
                saved = (self.store.load_state('pf_cursors') or {}).get(More.model_validate(args).cursor)
                if not saved:
                    raise ValueError('Unknown cursor; run the pf command again')
                operation = saved['operation']
            page = self.last_answer = self.answer(operation, args)  # structured form kept for audits
            if isinstance(page, str):
                return page  # pf_read content and diff keep their header line and exact body
            if page.get('format') == 'pf.relations.v1':
                from .pf_relations import render
                return render(page)
            if operation == 'pf_candidates':
                from .pf_candidates import render
                return render(page)
            if operation == 'pf_search':
                if 'text' in self.values:
                    return self._render_text_search(page)
                if 'sections' in page:
                    return pf_render.render_search(page)
                wanted = self.values['object']
                return pf_render.render_list(page, f"pf search {wanted['id']}", wanted)
            if operation == 'pf_log':
                return pf_render.render_log(page)
            if operation == 'pf_read' and self.values.get('mode') == 'captures':
                return pf_render.render_captures(page)
            if operation == 'pf_read':
                return pf_render.render_list(page, 'Versions of ' + page['root']['what'])
            if operation == 'pf_dependents':
                return pf_render.render_dependents(page)
            return pf_render.render_dependencies(page)

    def answer(self, operation, args):
        """Structured page for one query; pf_read content and diff return their delivered text."""
        try:
            with budget_scope(timeout=self.call_timeout or float('inf')), self.resolver.cached():
                return self._answer(operation, args)
        except TimeoutError:
            self.__dict__.pop('_indexed', None)
            raise

    def _answer(self, operation, args):
        check()
        try:
            values = MODELS[operation].model_validate(args).model_dump(exclude_none=True)
        except ValidationError as exc:
            raise ValueError('; '.join('.'.join(map(str, e['loc'])) + ': ' + e['msg']
                                      for e in exc.errors(include_input=False))) from exc
        if operation == 'pf_candidates':
            if 'cursor' in values and set(args) != {'cursor'}:
                raise ValueError('Continue with cursor only, or start a new candidates query')
            from .pf_candidates import CandidateQuery
            return CandidateQuery(self.registry).answer(values)
        if operation in ('pf_dependencies', 'pf_dependents'):
            saved = (self.store.load_state('pf_cursors') or {}).get(values.get('cursor'))
            if 'cursor' not in values or saved and saved.get('relation_snapshot'):
                if 'cursor' in values and set(args) != {'cursor'}:
                    raise ValueError('Continue with cursor only, or start a new relationship query')
                from .pf_relations import RelationQuery
                return RelationQuery(self.registry).answer(operation, values)
        offset, cutoff = 0, None
        self.cursor_name = 'pf_cursors'
        self.cursors = self.store.load_state(self.cursor_name) or {}
        if 'cursor' in values:
            if set(args) != {'cursor'}:
                raise ValueError('Continue with cursor only, or omit cursor to start a new query')
            saved = self.cursors.get(values['cursor'])
            if not saved or saved['operation'] != operation:
                raise ValueError('Unknown query cursor; start a new query')
            values, offset, cutoff = saved['values'], saved['offset'], saved['cutoff']
        self._index(cutoff)
        self.operation, self.values = operation, values
        if operation == 'pf_search':
            if 'text' in values:
                return self._search_text(values['text'], offset)
            if 'object' not in values:
                raise ValueError('object is required')
            wanted = values['object']
            same = lambda o: str(o['id']) == wanted['id'] and wanted.get('kind', o['kind']) == o['kind']
            matches = [v for v in self.nodes if not self._rerun(v) and any(same(o) for o in self._objects(v))]
            if values['all'] or offset:
                return self._page(matches[::-1], offset, self._describe)
            return self._sections(matches, wanted)
        if operation == 'pf_read' and values.get('mode') == 'captures' and 'capture_event' in values:
            page = self._captures(values['capture_event'], offset)
            if cutoff is None and 'cursor' in args:
                self.cursors[args['cursor']]['cutoff'] = self.cutoff
                self.store.save_state(self.cursor_name, self.cursors)
            return page
        if 'target' not in values:
            raise ValueError('target is required')
        target = self._target(values['target'])
        if operation == 'pf_log':
            return self._log(target, offset)
        if operation == 'pf_blame':
            return self._blame(target)
        if operation == 'pf_diff':
            baseline, target = self._diff_pair(target, values['target'])
            self.operation = 'pf_read'
            self.values = values = dict(target={'version': self.resolver.handle(target)}, mode='diff',
                                        baseline={'version': self.resolver.handle(baseline)},
                                        full=False, detail=False, limit=20)
            return self._read(target, offset, baseline=baseline)
        if operation == 'pf_read':
            mode = values['mode']
            if (mode == 'diff') != ('baseline' in values):
                raise ValueError('baseline is required only for diff')
            if values['full'] and mode != 'content':
                raise ValueError('full is supported only for content reads')
            if values['detail'] and mode != 'history':
                raise ValueError('detail is supported only for history')
            if mode == 'captures':
                row = self.nodes[target]
                event = self.events[row['event_id']]
                if row['meta']['layer'] not in ('tool_return', 'stdout', 'stderr') or event['request']['kind'] not in ('bash', 'cli_python'):
                    raise ValueError('captures requires a saved command or script output')
                self.values = dict(values, capture_event=row['event_id'], scope='latest captured execution for this handle')
                return self._captures(row['event_id'], offset)
            if mode == 'history':
                key = self._key(target)
                listed = [v for v in self.nodes if self._key(v) == key and not self._rerun(v)][::-1]
                return self._page(listed, offset, self._describe, root=self._describe(listed[0] if listed else target))
            return self._read(target, offset)
        self.follow_execution = (values['include_execution'] or
                                 operation == 'pf_dependencies' and values['purpose'] == 'current')
        rows = self._trace(target, operation == 'pf_dependents')
        unexpanded = list(dict.fromkeys(self.resolver.handle(r['edge']['target']) for r in rows
                                       if r['stop'] == 'depth_limit'))
        describe = self._describe_edge
        if not values['detail']:
            if operation == 'pf_dependencies':
                rows = [r for r in rows if _depth(r) == 1]
            else:
                seen, endpoints = set(), []
                for row in rows:
                    source = row['edge']['source']
                    if source in self.records and not row['traversal_only'] and source not in seen:
                        seen.add(source)
                        endpoints.append(row)
                rows = endpoints
        return self._page(rows, offset, describe, root=self._describe(target), unexpanded=unexpanded)

    def _index(self, cutoff):
        check()
        started = time.monotonic()
        reset = not hasattr(self, '_indexed') or (cutoff is not None and cutoff < self._indexed)
        if reset:
            self.events, self.nodes, self.records = {}, {}, {}
            self.outgoing, self.incoming = defaultdict(list), defaultdict(list)
            self.latest, self.reads, self.event_day = {}, defaultdict(list), {}
            self.same_bytes, self.members = {}, defaultdict(list)
            self._files, self._outputs, self._inputs = {}, defaultdict(set), defaultdict(set)
            self._codes, self._private_pending = defaultdict(set), {}
            self._edge_keys, self._pending_events = set(), set()
            self._indexed = self._event_cutoff = 0
            self._last_day = None
        day = self._last_day
        added_events = 0
        with closing(self.store.connect()) as conn:
            conn.execute('BEGIN')
            self.cutoff = cutoff if cutoff is not None else conn.execute('SELECT coalesce(max(rowid),0) FROM versions').fetchone()[0]
            pending = list(self._pending_events)
            placeholders = ','.join('?' for _ in pending) or 'NULL'
            for row in conn.execute(f'''SELECT r.rowid AS event_row,r.event_id,r.request,s.record,q.definition FROM requests r
                    LEFT JOIN results s USING(event_id) LEFT JOIN queries q ON r.query_id=q.id
                    WHERE r.rowid>? OR r.event_id IN ({placeholders}) ORDER BY r.rowid''',
                    [self._event_cutoff, *pending]):
                check()
                self._event_cutoff = max(self._event_cutoff, row['event_row'])
                if not self.store.identity.get('role'):
                    try:
                        self.store._visible(conn, row['event_id'])
                    except KeyError:
                        continue
                request = json.loads(row['request'])
                # Host day stamps provide context; file observation dates require stamped ancestry.
                if row['event_id'] not in self.events and request['kind'] in evidence_handles.HOST_DAY_KINDS:
                    day = (request.get('request') or {}).get('day', day)
                self.event_day.setdefault(row['event_id'], day)
                self.events[row['event_id']] = dict(request=request,
                    result=json.loads(row['record']) if row['record'] else {},
                    query=json.loads(row['definition']) if row['definition'] else None)
                if row['record']:
                    self._pending_events.discard(row['event_id'])
                else:
                    self._pending_events.add(row['event_id'])
                added_events += 1
            versions = conn.execute('SELECT rowid,* FROM versions WHERE rowid>? AND rowid<=? ORDER BY rowid',
                                    (self._indexed, self.cutoff)).fetchall()
        self._last_day = day
        new_nodes = {}
        for row in versions:
            check()
            if row['event_id'] not in self.events:
                continue
            meta = json.loads(row['metadata'])
            try:
                with closing(self.store.connect()) as conn:
                    self.store.authorize_version(conn, row['event_id'], meta, public=meta['layer'] in PUBLIC_LAYERS, graph=True)
            except PermissionError:
                continue
            event = self.events[row['event_id']]
            retrieval = (meta['layer'] in ('stdout', 'stderr', 'tool_return')
                         and event['result'].get('pf_retrieval', bool(event['result'].get('pf_calls'))))
            if meta['layer'] in PUBLIC_LAYERS and not event['request']['kind'].startswith('pf_') and not retrieval:
                self.nodes[row['version_id']] = dict(row, meta=meta)
                new_nodes[row['version_id']] = self.nodes[row['version_id']]
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
                self._private_pending[row['version_id']] = (row['event_id'], meta['layer'])
        files, execution_outputs, execution_inputs = self._files, self._outputs, self._inputs
        codes, touched = self._codes, set()
        for version, row in new_nodes.items():
            if row['meta']['layer'] == 'executed_code':
                codes[row['event_id']].add(version)
        for version, row in new_nodes.items():
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
                key = (meta.get('owner_role'), meta['object_id'])
                previous = files.get(key)
                if previous and self.nodes[previous]['content_hash'] == row['content_hash']:
                    self._edge(version, previous, 'same_content_observation')
                else:
                    files[key] = version
                same = (key, row['content_hash'])
                self.same_bytes[version] = same
                self.members[same].append(version)
            if meta['layer'] in ('executed_code', 'server_public_response'):
                parent = row['event_id']
                seen = set()
                while parent in self.events and parent not in seen:
                    seen.add(parent)
                    execution_inputs[parent].add(version)
                    touched.add(parent)
                    if meta['layer'] == 'server_public_response':
                        for code in codes[parent]:
                            self._edge(version, code, 'executed_by')
                    parent = self.events[parent]['request'].get('parent_event_id')
            if meta['layer'] == 'file_bytes' and event['request']['kind'] == 'edit_file' and version.endswith('_read'):
                execution_inputs[row['event_id']].add(version)
                touched.add(row['event_id'])
        for version, (event_id, layer) in list(self._private_pending.items()):
            if not self.events[event_id]['result']:
                continue
            value = json.loads(self.resolver.content(version, graph=True)[1])
            if layer == 'agent_declaration' and value['version_id'] in self.records:
                source = value['version_id']
                refs = self.records[source]['references']
                for i, ref in enumerate(refs):
                    binding = value['references'][i] if i < len(value['references']) else {}
                    resolved = binding.get('status') == 'resolved' and binding.get('git_content_matches') is not False
                    target = binding.get('version_id') if resolved else None
                    reason = ('inaccessible' if resolved and target not in self.nodes else None if resolved else 'git_content_mismatch' if binding.get('git_content_matches') is False
                              else binding.get('reason', 'not_captured_in_prefix'))
                    self._edge(source, target, 'reference', origin='agent_declaration',
                               reference_index=i, reference=({k: v for k, v in ref.items() if k in ('purpose',)} if reason == 'inaccessible' else ref), reason=reason or ('captured_version_unavailable' if target not in self.nodes else None),
                               missing=target not in self.nodes)
            elif layer in ('model_source_occurrences', 'model_reconstructions'):
                if self.store.identity.get('role') and (self.events[event_id]['request'].get('role') != self.store.identity['role'] or self.events[event_id]['request'].get('session_id') != self.store.identity['session_id']):
                    del self._private_pending[version]
                    continue
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
                        execution_outputs[event_id].add(item['version'])
                        touched.add(event_id)
            del self._private_pending[version]
        for version, row in new_nodes.items():
            # Capture association, not a claim of numerical causality.
            if row['meta']['layer'] in ('tool_return', 'stdout', 'stderr'):
                execution_outputs[row['event_id']].add(version)
                touched.add(row['event_id'])
        for event in touched:
            for output in sorted(execution_outputs[event]):
                for source in sorted(execution_inputs[event]):
                    self._edge(output, source, 'observed_edit_input' if self.nodes[source]['meta']['layer'] == 'file_bytes' else 'same_execution')
        self._indexed = self.cutoff
        self.index_stats = dict(seconds=time.monotonic()-started, new_events=added_events,
                                new_versions=len(versions), cold=reset)

    def _key(self, version):
        row = self.nodes[version]
        meta, event = row['meta'], self.events[row['event_id']]
        key = evidence_key(version, meta, event['query'], event['result'], event['request'].get('request'), event['request'])
        return (*key, meta.get('owner_role')) if self.store.identity.get('role') else key

    def _edge(self, source, target, kind, origin='automatic_capture', **details):
        if target not in self.nodes and origin != 'agent_declaration':
            return
        edge = dict(source=source, target=target if target in self.nodes else None,
                    kind=kind, origin=origin, **details)
        identity = encoded(edge)
        if identity in self._edge_keys:
            return
        self._edge_keys.add(identity)
        self.outgoing[source].append(edge)
        if edge['target']:
            self.incoming[target].append(edge)

    def _objects(self, version):
        if version in self.records:
            return [dict(o, basis='agent_declaration') for o in self.records[version]['objects']]
        return [dict(kind=o['kind'], id=o['value'], basis=o['basis'], field=o.get('path'))
                for o in self.nodes[version]['meta'].get('objects', [])]

    def _target(self, target):
        if 'version' in target and evidence_handles.RAW_VERSION.fullmatch(target['version']):
            self.store.public_content(target['version'])
            if target['version'] not in self.nodes:
                raise ValueError('Captured version is unavailable in this cursor')
            return target['version']
        owner = self.store.identity.get('role')
        target = dict(target)
        if 'version' in target and evidence_handles.RECORD.fullmatch(target['version']):
            target = {'record': target['version']}
        for field in ('path', 'record'):
            if owner and field in target and target[field].split(':', 1)[0] in ('ceo', 'growth', 'ops_finance') and ':' in target[field]:
                owner, target[field] = target[field].split(':', 1)
        if 'path' in target and evidence_handles.VERSIONED.fullmatch(target['path']) and not any(
                row['meta'].get('object_id') == target['path'] for row in self.nodes.values()):
            target = {'version': (owner + ':' if owner else '') + target['path']}  # forecast.json@v3 written as a path
        if 'path' in target and not any(row['meta']['layer'] == 'file_bytes' and row['meta']['object_id'] == target['path']
                                        for row in self.nodes.values()):
            # An output or query named without its version, e.g. analyze.py.out or query7: the latest.
            members = [v for v in evidence_handles.index(self.store).latest((owner + ':' if owner else '') + target['path']) or [] if v in self.nodes]
            if members:
                return members[-1]
        if 'version' in target:
            members = [v for v in self.resolver.lookup(target['version']) or [] if v in self.nodes]
            if not members:
                raise ValueError('Unknown version handle; pf log <file or output> lists its versions')
            return members[-1]
        matches = []
        for version, row in self.nodes.items():
            meta, event = row['meta'], self.events[row['event_id']]
            if 'path' in target and meta['layer'] == 'file_bytes' and meta['object_id'] == target['path'] and (not owner or meta.get('owner_role') == owner):
                matches.append(version)
            elif 'sql' in target and meta['layer'] == 'server_public_response' and event['query'] and event['query'][5] == target['sql']:
                matches.append(version)
            elif 'record' in target and version in self.records:
                record = self.records[version]
                if target['record'] in (record['id'], record['version']) and (not owner or meta.get('owner_role') == owner):
                    matches.append(version)
        if not matches:
            if owner and self.store.identity.get('role'):
                from .role_policy import READABLE_ROLES
                if owner not in READABLE_ROLES[self.store.identity['role']]:
                    raise PermissionError('Evidence is inaccessible')
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
            relation_origin='agent_declaration' if record else 'automatic_capture', owner=meta.get('owner_role', 'ceo'))
        if meta['source_truncated']:
            result['truncated'] = True
        if meta['extent'] != 'full':
            result['extent'] = meta['extent']
        if event['result'].get('capture_gaps'):
            result['capture_gaps'] = event['result']['capture_gaps']
        if about := _about(meta['content_time']):
            result['about'] = about
        if (request.get('parsed') or {}).get('tool') == 'get_group_insights':
            dates = meta['content_time'].get('fields', [])
            result['snapshot_day'] = next((f['value'] for f in dates if f['field'].endswith('snapshot_day')), None)
            result['refresh_hint'] = 'research_group completes a new survey; calling get_group_insights again only retrieves it'
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
            result.update(record=result['version'], what='text ' + result['version'],
                          text=record['text'], text_status=record['status'],
                          reason=record['reason'], applies_at=record['applies_at'],
                          current_revision=self.latest[self._key(version)] == version)
        handles = evidence_handles.index(self.store)
        if previous := handles.previous(version):
            result['previous_version'] = previous
        if (day := handles.repeated(version)) is not None:
            result['unchanged_since_day'] = day
        if same := handles.same_as(version):
            result['same_content_as'] = same
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
        return {k: full[k] for k in ('version', 'day', 'what', 'owner', 'truncated', 'extent',
                                    'snapshot_day', 'refresh_hint') if k in full} | (
            {'status': full['status']} if full['status'] != 'succeeded' else {})

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
            check()
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
            result['reference_index'] = edge.get('reference_index')
        return result

    # --- pf log / diff / blame / search ---------------------------------------------

    def _groups(self, version):
        """Version groups of this version's object visible in this snapshot, oldest first."""
        handles = evidence_handles.index(self.store)
        return [g for g in handles.versions(version) if any(m in self.nodes for m in g['members'])]

    def _text(self, version):
        try:
            return self.resolver.content(version)[1].decode('utf-8')
        except UnicodeDecodeError:
            return None

    def _log(self, target, offset):
        """One line per version of the object, newest first; revisions for a registered text."""
        if target in self.records:
            key = self._key(target)
            revisions = [v for v in self.nodes if v in self.records and self._key(v) == key][::-1]
            return self._page(revisions, offset, self._describe, root=self._describe(target), kind='record')
        groups = self._groups(target)
        handles = evidence_handles.index(self.store)
        before = dict(zip(map(id, groups[1:]), groups))
        def describe(group):
            member = next(m for m in group['members'] if m in self.nodes)
            item = self._brief(member) | dict(version=handles.name(member), day=group['day'],
                                              acquisitions=len(group['members']))
            text, layer = self._text(member), self.nodes[member]['meta']['layer']
            earlier = before.get(id(group))
            if text is not None and layer == 'server_public_response':
                try:
                    item['rows'] = len(json.loads(text)['rows'])
                except (ValueError, KeyError, TypeError):
                    pass
            elif text is not None:
                previous = earlier and self._text(next(m for m in earlier['members'] if m in self.nodes))
                item['size'] = (pf_render.text_change(previous, text) if previous is not None
                                else pf_render.plural(len(text.splitlines()), 'line'))
            if note := group.get('note'):
                item['note'] = note
            if same := handles.same_as(member):
                item['same_content_as'] = same
            return item
        return self._page(groups[::-1], offset, describe, root=self._describe(target))

    def _diff_pair(self, version, requested):
        """pf diff with one argument: that version against the latest, or the last two versions."""
        if version in self.records:
            key = self._key(version)
            revisions = [v for v in self.nodes if v in self.records and self._key(v) == key]
            if '.' in requested.get('record', requested.get('version', '')):
                return version, revisions[-1]
            if len(revisions) < 2:
                raise ValueError(f"{self.records[version]['version']} has no other revision to compare")
            return revisions[-2], revisions[-1]
        groups = self._groups(version)
        newest = lambda g: [m for m in g['members'] if m in self.nodes][-1]
        if 'version' in requested or evidence_handles.VERSIONED.fullmatch(requested.get('path', '')):
            return version, newest(groups[-1])
        if len(groups) < 2:
            raise ValueError(f"{evidence_handles.index(self.store).name(version)} is the only version; nothing to compare")
        return newest(groups[-2]), newest(groups[-1])

    def _blame(self, target):
        """Each line of a text version with the version and day that introduced it."""
        if target in self.records:
            raise ValueError('pf blame works on files and outputs; pf log rN lists the revisions of a text')
        handles = evidence_handles.index(self.store)
        group = handles.group(target)
        groups = [g for g in self._groups(target) if g['k'] <= group['k']]
        origins, lines = [], []
        for g in groups:
            check()
            text = self._text(next(m for m in g['members'] if m in self.nodes))
            if text is None:
                raise ValueError('blame needs a text version')
            new = text.splitlines()
            fresh = []
            for tag, a, b, c, d in difflib.SequenceMatcher(None, lines, new, autojunk=False).get_opcodes():
                fresh += origins[a:b] if tag == 'equal' else [g] * (d - c)
            origins, lines = fresh, new
        name = handles.name(target)
        out = [f"{name}: the version and day that introduced each line"]
        previous = None
        width = max((len(f"v{g['k']} d{g['day'] if g['day'] is not None else 'unknown'}") for g in origins), default=0)
        for origin_group, line in zip(origins, lines):
            label = f"v{origin_group['k']} d{origin_group['day'] if origin_group['day'] is not None else 'unknown'}" if origin_group is not previous else ''
            out.append(f'{label:<{width}} | {line}')
            previous = origin_group
        notes = [g for g in {id(g): g for g in origins}.values() if g.get('note')]
        if notes:
            out.append('Notes: ' + ' · '.join(f"v{g['k']} (day {g['note'][0] if g['note'][0] is not None else 'unknown'}): \"{pf_render.short(g['note'][1], 120)}\""
                                              for g in sorted(notes, key=lambda g: g['k'])))
        text = '\n'.join(out)
        if len(text) > 30000:
            text = text[:30000] + f'\n[truncated; pf show {name} for the whole file]'
        return text

    def _search_text(self, term, offset):
        """Literal, case-insensitive search of public history at the cursor cutoff."""
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        matches, seen = [], set()
        handles = evidence_handles.index(self.store)
        with closing(self.store.connect()) as conn:
            for version, row in reversed(self.nodes.items()):
                check()
                if self._rerun(version):
                    continue
                key = (handles.key(version), row['content_hash'])
                if key in seen:
                    continue
                seen.add(key)
                try:
                    body = self.resolver.content(version, connection=conn)[1].decode('utf-8')
                except UnicodeDecodeError:
                    continue
                match = pattern.search(body)
                if match:
                    start, end = max(0, match.start() - 80), min(len(body), match.start() + 200)
                    matches.append(dict(source_version=version,
                        day=self.event_day[row['event_id']], snippet=body[start:end], source_range=[start, end]))
        # Bound rendered excerpts too, so a large --limit never skips hits when
        # the enclosing Bash return reaches its character limit.
        end, size = offset, 0
        for item in matches[offset:offset + self.values['limit']]:
            name = self.resolver.handle(item['source_version'])
            width = len(name) + len(item['snippet']) + 80
            if end > offset and size + width > 22000:
                break
            item['version'] = name
            size += width
            end += 1
        return dict(items=matches[offset:end], next_cursor=self._cursor(end) if end < len(matches) else None,
                    remaining=len(matches) - end, total=len(matches), detail=self.values['detail'])

    def _render_text_search(self, page):
        text = f"pf search --text {self.values['text']!r}: {page['total']} matching objects/content versions, newest first."
        sources = []
        for item in page['items']:
            text += f"\n{item['version']} · day {item['day'] if item['day'] is not None else 'unknown'} · excerpt:\n"
            sources.append(dict(version_id=item['source_version'], source_range=item['source_range'],
                                request_range=[len(text), len(text) + len(item['snippet'])], full_source=False))
            text += item['snippet']
        if page['next_cursor']:
            text += f"\n{page['remaining']} more: pf more {page['next_cursor']}."
        return CapturedText(text, sources)

    def _sections(self, matches, wanted):
        """pf search: registered texts, business writes and latest outputs about one object."""
        handles = evidence_handles.index(self.store)
        texts = [v for v in matches if v in self.records and self.latest[self._key(v)] == v
                 and self.records[v]['status'] == 'active']
        writes = [v for v in matches if self.events[self.nodes[v]['event_id']]['result'].get('classification')
                  == 'write_receipt']
        outputs, seen = [], set()
        for version in matches[::-1]:
            for edge in self.incoming.get(version, []):
                source = edge['source']
                if edge['kind'] != 'same_execution' or source not in self.nodes:
                    continue
                # What the agent saw: a command's return, or the stdout of a script it ran by name.
                event = self.events[self.nodes[source]['event_id']]
                layer = self.nodes[source]['meta']['layer']
                script = event['request']['kind'] == 'cli_python' and (event['request'].get('request') or {}).get('script')
                if not (layer == 'tool_return' or (layer == 'stdout' and script)) or self._rerun(source):
                    continue
                key = handles.key(source)
                if key not in seen:
                    seen.add(key)
                    outputs.append(source)
        total = len(matches)
        return dict(object=wanted, total=total, sections=[
            ('Active texts', [self._describe(v) for v in texts[::-1][:5]], len(texts)),
            ('Business writes', [self._describe(v) for v in writes[::-1][:5]], len(writes)),
            ('Outputs', [self._describe(v) for v in outputs[:5]], len(outputs))])

    def _captures(self, event_id, offset):
        with closing(self.store.connect()) as conn:
            self.store.readable_event(conn, event_id)
        roots = [v for v, row in self.nodes.items() if row['event_id'] == event_id
                 and row['meta']['layer'] in ('tool_return', 'stdout', 'stderr')]
        if not roots:
            raise ValueError('Captured execution is unavailable in this cursor')
        root = next((v for v in roots if self.nodes[v]['meta']['layer'] == 'tool_return'), roots[0])
        children = {event_id}
        for child, event in self.events.items():
            if event['request'].get('parent_event_id') in children and not event['request']['kind'].startswith('pf_'):
                children.add(child)
        files = {v for child in children for v in self._outputs[child]
                 if self.nodes[v]['meta']['layer'] == 'file_bytes' and not noise(self.nodes[v]['meta']['object_id'])}
        versions = [v for v, row in self.nodes.items() if v in files or (
            row['event_id'] in children and (row['meta']['layer'] == 'server_public_response' or
            (row['event_id'] != event_id and row['meta']['layer'] in ('stdout', 'stderr')
             and bool(self.resolver.content(v, public=True)[1]))))]
        def status(event):
            result = event['result']
            return 'unfinished' if result.get('completion') == 'unfinished' else result.get('status', 'result_unknown')

        calls = {child: event for child, event in self.events.items() if child in children
                 and event['request']['kind'] in ('sql_query', 'public_http')}
        responses = {self.nodes[v]['event_id'] for v in versions
                     if self.nodes[v]['meta']['layer'] == 'server_public_response'}
        missing = []
        for child, event in calls.items():
            if child not in responses:
                request = event['request'].get('request') or {}
                missing.append(dict(missing_response=True, status=status(event), day=self.event_day.get(child),
                    what=pf_render.label('server_public_response', event['request']['kind'], request,
                        event['query'][5] if event['query'] else None,
                        classification=event['result'].get('classification'))))
        gaps = sorted({gap for child in children for gap in self.events[child]['result'].get('capture_gaps', [])})
        return self._page(versions + missing, offset,
            lambda item: self._describe(item) if isinstance(item, str) else item,
            root=self._describe(root), scope=self.values['scope'], capture_gaps=gaps,
            call_statuses=dict(Counter(status(event) for event in calls.values())),
            script_statuses=dict(Counter(status(self.events[child]) for child in children
                if self.events[child]['request']['kind'] == 'cli_python')))

    def _cursor(self, offset):
        value = dict(operation=self.operation, values=self.values, offset=offset, cutoff=self.cutoff)
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

    def _read(self, target, offset, baseline=None):
        meta, raw = self.resolver.content(target, public=True)
        try:
            content = raw.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ValueError('This captured version is not UTF-8 text') from exc
        header = {}
        if self.values['mode'] == 'diff':
            baseline = baseline or self._target(self.values['baseline'])
            key = self._key(target)
            if key != self._key(baseline) or not (key[0] in ('query', 'public_read', 'file_bytes', 'registered_text', 'registered_script',
                                                              'dashboard') or key[0].startswith(('script_', 'command_'))):
                raise ValueError('Diff requires two versions of the same captured object')
            old_meta, old = self.resolver.content(baseline, public=True)
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
                    contents = [self.resolver.content(self.nodes[v]['event_id'] + ':comparison', graph=True)[1] for v in (baseline, target)]
                    header['rows_equal'] = contents[0] == contents[1]
                    if meta['source_truncated'] or old_meta['source_truncated']:
                        header['rows_scope'] = 'returned_subset'
            header['baseline'] = self._brief(baseline)
            # splitlines with terminators preserves final-newline differences.
            lines = difflib.unified_diff(before.splitlines(keepends=True), content.splitlines(keepends=True),
                                        fromfile=header['baseline']['version'], tofile=self.resolver.handle(target))
            content = ''.join(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n' for line in lines)
        header['target'] = self._brief(target)
        # Leave room for the header/cursor inside Bash's 30,000-character return.
        end = min(len(content), offset + 24000)
        header.update(range=[offset, end], total_chars=len(content),
                      next_cursor=self._cursor(end) if end < len(content) else None)
        if end < len(content):
            header['truncated_reason'] = 'character_limit'
        prefix = encoded(header).decode() + '\n'
        # The continuation command follows the body; the header keeps the delivery fields.
        more = f"\n[{len(content) - end:,} more characters: pf more {header['next_cursor']}]" if header['next_cursor'] else ''
        if self.values['mode'] == 'diff' and content and end == len(content):
            more += f"\n[Before changing this source: pf rdepend {shlex.quote(header['baseline']['version'])} --detail.]"
        from .pf_read import capture_read
        return capture_read(self, target, prefix + content[offset:end] + more,
                            offset, end, len(content))
