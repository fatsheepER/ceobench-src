"""PF-only historical queries, rebuilt from immutable capture and declaration records."""
from collections import defaultdict, deque
from contextlib import closing
import difflib
import json
import re
import time
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, ValidationError, model_validator

from .execution_capture import CapturedText, CURRENT_EVENT, OBJECT_FIELDS, origin
from . import evidence_handles, pf_render
from .registration_evidence import covered
from .registration_schema import Input, Text
from .text_registry import applies_ended
from .sql_evidence import encoded


class Target(Input):
    path: Text | None = None
    sql: Text | None = None
    record: Annotated[str, Field(pattern=r'^r[1-9][0-9]*(\.[1-9][0-9]*)?$')] | None = None
    version: Annotated[str, Field(pattern=evidence_handles.HANDLE_PATTERN)] | None = None

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


MODELS = dict(pf_search=Search, pf_dependencies=Dependencies, pf_dependents=Dependents, pf_read=Read,
              pf_log=Versions, pf_diff=Single, pf_blame=Single, pf_more=More)


RELATION = dict(reference='cites', same_execution='observed in same execution as', returned_range='printed from',
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


def evidence_key(version, meta, query, result, request, record=None):
    """The object whose versions are listed, diffed and checked together."""
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
    return '__pycache__' in parts or parts[0] == 'sessions'


class PFQueries:
    def __init__(self, registry, *, stale_checks=True, refresh=None):
        if registry.mode != 'pf':
            raise ValueError('PF queries require PF mode')
        self.store, self.resolver = registry.store, registry.resolver
        self.registry = registry
        self.workspace = registry.workspace
        self.stale_checks, self.refresh = stale_checks, refresh

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
                SELECT count(*) FROM versions v JOIN requests r USING(event_id)
                WHERE r.event_id IN children AND json_extract(v.metadata, '$.layer')='server_public_response'
                ''', (capture.event,)).fetchone()[0]
            scripts = conn.execute('''SELECT r.event_id || ':stdout' FROM requests r
                WHERE json_extract(r.request, '$.parent_event_id')=? AND json_extract(r.request, '$.kind')='cli_python'
                AND json_extract(r.request, '$.request.script') IS NOT NULL ORDER BY r.rowid''',
                (capture.event,)).fetchall()
        body = str(text)
        printed = []
        for (version,) in scripts:
            try:
                output = self.resolver.content(version)[1].decode('utf-8')
            except (RuntimeError, UnicodeDecodeError):
                continue
            if output:
                shown = [o for o in capture.origins if o['version_id'] == version]
                printed.append((version, covered([(0, len(output))], shown) if shown else False, output == body))
        # PF's own refreshes are excluded above; ordinary SQL/scripts in the same
        # shell still produce citable evidence, including failed or piped output.
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
        line = '[pf: ' + ' | '.join([output] + wrote + noted) + ']'
        return body + '\n' + line + ''.join('\n' + line for line in shown_files)

    def _noted(self, version):
        note = evidence_handles.index(self.store).note(version)
        return f' note (day {note[0]}): "{pf_render.short(note[1], 200)}"' if note else ''

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
        if operation == 'pf_more':
            saved = (self.store.load_state('pf_cursors') or {}).get(More.model_validate(args).cursor)
            if not saved:
                raise ValueError('Unknown cursor; run the pf command again')
            operation = saved['operation']
        page = self.last_answer = self.answer(operation, args)  # structured form kept for audits
        if isinstance(page, str):
            return page  # pf_read content and diff keep their header line and exact body
        if operation == 'pf_search':
            if 'text' in self.values:
                return self._render_text_search(page)
            if 'sections' in page:
                return pf_render.render_search(page)
            wanted = self.values['object']
            return pf_render.render_list(page, f"pf search {wanted['id']}", wanted)
        if operation == 'pf_log':
            return pf_render.render_log(page)
        if operation == 'pf_read':
            return pf_render.render_list(page, 'Versions of ' + page['root']['what'])
        if operation == 'pf_dependents':
            return pf_render.render_dependents(page)
        return pf_render.render_dependencies(page)

    def answer(self, operation, args):
        """Structured page for one query; pf_read content and diff return their delivered text."""
        with self.resolver.cached():
            return self._answer(operation, args)

    def _answer(self, operation, args):
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
                    self._review_after_manual(target, rows)
                    self.check_state = 'pf_check:' + event
                    self.store.save_state(self.check_state, rows)
                    if own_event:
                        self.store.complete(event)
                finally:
                    CURRENT_EVENT.reset(token)
        unexpanded = list(dict.fromkeys(self.resolver.handle(r['edge']['target']) for r in rows
                                       if r['stop'] == 'depth_limit'))
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
                          stale_check='performed' if checking else 'not_performed', unexpanded=unexpanded,
                          checked_day=self._check_day if checking else None)

    def _review_after_manual(self, target, rows):
        state = self.store.load_state('pf_review')
        if not state or target not in state['pending'] or any(r['stop'] == 'depth_limit' for r in rows):
            return  # A partial scope cannot discharge an unexamined root.
        direct = [r for r in rows if _weekly(r)]
        if not direct:
            return
        if any(r['check']['affected'] for r in direct):
            pending = state['pending'][target]
            pending['last_day'] = self._check_day
            pending['conditions'] = self._continuing_rows(rows)
            ordinary = self._ordinary_changes(rows)
            if ordinary:
                pending['reason'] = self._describe_edge(ordinary[0])
            else:
                del state['pending'][target]
        else:
            del state['pending'][target]
        self.store.save_state('pf_review', state)

    def weekly_check(self, day):
        with self.resolver.cached():
            return self._weekly_check(day)

    def _weekly_check(self, day):
        """Week-start digest of new issues, pending review and continuing conditions."""
        if not self.stale_checks:
            return None
        self._index(None)
        tracing = time.monotonic()
        self.operation = 'pf_dependencies'
        # The digest covers every reachable dependency; explicit queries remain depth-limited.
        self.values = dict(depth=len(self.nodes), purpose='current', include_execution=False, detail=False, limit=100)
        self.follow_execution = True
        state = self.store.load_state('pf_review') or {'pending': {}, 'ended': []}
        self.review_pending = state['pending']
        active = sorted((v for v, record in self.records.items()
                         if record['status'] == 'active' and self.latest[self._key(v)] == v),
                        key=lambda v: int(self.records[v]['id'][1:]), reverse=True)
        traced, ended, continuing = {}, [], set()
        eligible = set()
        for version in active:
            record = self.records[version]
            if applies_ended(record, day):
                ended.append(record['version'])
                continue
            if record.get('applies_at', {}).get('start_day', 0) > day:
                continue
            eligible.add(version)
            if version in self.review_pending:
                # Keep ordinary findings dated, but retry transient source failures
                # alongside explicit conditions and their propagation paths.
                rows = self.review_pending[version]['conditions']
                if rows:
                    traced[version] = rows
                    continuing.add(version)
                continue
            rows = self._cached_trace(version)
            if any(_weekly(r) for r in rows):
                traced[version] = rows
        self.review_pending = {v: p for v, p in self.review_pending.items() if v in eligible}
        new_ended = [v for v in ended if v not in state['ended']]
        if not traced and not new_ended and not self.review_pending:
            if state != {'pending': {}, 'ended': ended}:
                self.store.save_state('pf_review', dict(pending={}, ended=ended))
            return None
        traced_at = time.monotonic()
        skipped_pending = len(self.review_pending) - len(continuing)
        event = self.store.begin_event('pf_weekly_check', {'day': day})
        token = CURRENT_EVENT.set(event)
        try:
            self._check([row for rows in traced.values() for row in rows], day=day)
            rendering = time.monotonic()
            entries, underlying = [], []
            for version, rows in traced.items():
                tops = ([self._describe_edge(r) for r in rows
                         if (self._conditional(r) or r.get('retry_refresh')) and not r['historical_only']]
                        if version in continuing else [self._top(r, rows) for r in rows if _weekly(r)])
                # Old data changing deserves an example even behind an immutable
                # output. Only verified row additions are folded into a count.
                direct = [t for t in tops if t['check'].get('below_failed')
                          or t['check'].get('below_content_changed')
                          or t['check']['predicate_result'] in ('fails', 'cannot_check')
                          or (t['check']['version_changed'] and t['check']['predicate_result'] != 'holds'
                              and t['check'].get('change_kind') != 'append_only')]
                entries.append(dict(text=self._describe(version), total=len(tops), changed=direct))
                if not direct and any(t['check'].get('below_changed') or
                                      t['check'].get('change_kind') == 'append_only' for t in tops):
                    underlying.append(self.records[version]['version'])
                if version not in continuing:
                    ordinary = self._ordinary_changes(rows)
                    if ordinary:
                        reason = self._describe_edge(ordinary[0])
                        self.review_pending[version] = dict(first_day=day, last_day=day,
                            reason=reason, conditions=self._continuing_rows(rows))
                else:
                    # Do not advance the ordinary problem's last verification date.
                    self.review_pending[version]['conditions'] = self._continuing_rows(rows)
            text = pf_render.render_weekly(entries, day, pf=True, ended=new_ended, underlying=underlying,
                                          condition_only=len(continuing), pending=len(self.review_pending),
                                          active=len(active), eligible=len(eligible))
            if self.review_pending:
                text += (f'\nPending review: {pf_render.plural(len(self.review_pending), "active text")}; '
                         f'{pf_render.plural(skipped_pending, "earlier finding")} '
                         f'{"was" if skipped_pending == 1 else "were"} not rechecked this week. '
                         'Use text_list with review="pending" for dated findings; '
                         'pf depend rN rechecks one text. Pending does not retire a text.')
                text += pf_render.render_pending([
                    dict(record=self.records[v]['version'], **p) for v, p in self.review_pending.items()], day)
            self.store.save_state('pf_review', dict(pending=self.review_pending, ended=ended))
            version = self.store.version(event, 'weekly_check', text, layer='weekly_check')
            self.store.complete(event, index=self.index_stats, check=self.check_stats,
                                trace_seconds=traced_at-tracing, render_seconds=time.monotonic()-rendering,
                                skipped_pending_texts=skipped_pending,
                                content_reads=self.resolver.read_stats,
                                checked_texts=len(traced), pending_texts=len(self.review_pending))
        finally:
            CURRENT_EVENT.reset(token)
        return CapturedText(text, [origin(version, text)])

    def _ordinary_changes(self, rows):
        paths = {tuple(p) for r in rows if _weekly(r) for p in r['check']['affected_paths']}
        ordinary = [r for r in rows if not self._conditional(r) and not r['historical_only']
                    and (r['check']['version_changed'] or r['check']['predicate_result'] == 'cannot_check')
                    and (r['edge']['target'] is None or not self._clock(r['edge']['target']))
                    and any(tuple(p[-2:]) == (r['edge']['source'], r['edge']['target'])
                            or r['edge']['target'] is None and p[-1] == r['edge']['source'] for p in paths)]
        return sorted(ordinary, key=lambda r: (
            r['check']['predicate_result'] != 'cannot_check',
            r['check'].get('change_kind') == 'append_only'))

    @staticmethod
    def _conditional(row):
        ref = row['edge'].get('reference', {})
        return bool(ref.get('select') or ref.get('predicate'))

    def _continuing_rows(self, rows):
        failures = self.store.load_state('pf_refresh_failures') or {}
        def needs_retry(row):
            target = row['edge']['target']
            failure = failures.get(encoded(self._key(target)).decode(), {}) if target else {}
            return bool(failure) and not failure.get('permanent') and not row['historical_only']
        paths = [r['path'] for r in rows if not r['historical_only'] and (self._conditional(r) or needs_retry(r))]
        result = []
        for row in rows:
            if not any(path[:len(row['path'])] == row['path'] for path in paths):
                continue
            item = dict(row)
            item['retry_refresh'] = needs_retry(row)
            if not self._conditional(row) and not item['retry_refresh']:
                item['cached_check'] = row['check']
            else:
                item.pop('cached_check', None)
            result.append(item)
        return result

    def _cached_trace(self, root):
        cache = getattr(self, '_traces', {})
        saved = cache.get(root)
        if saved and all(self._graph_versions[n] == revision for n, revision in saved[0].items()):
            return [dict(r) for r in saved[1]]
        rows = self._trace(root, False)
        nodes = {self._node(v) for r in rows for v in r['path']}
        cache[root] = ({n: self._graph_versions[n] for n in nodes}, rows)
        self._traces = cache
        return [dict(r) for r in rows]

    def _check(self, rows, day=None):
        from .pf_stale import StaleCheck
        return StaleCheck(self).run(rows, day=day)

    def _top(self, row, rows):
        """A direct dependency, with a count of the underlying sources that changed."""
        result = self._describe_edge(row)
        check = row.get('check')
        if not check or row['historical_only']:
            return result
        below = [r for r in rows if _depth(r) > 1 and r['path'][:2] == row['path'] and r.get('check')
                 and not r['historical_only'] and (r['edge']['target'] is None or
                 (self.nodes[r['edge']['target']]['meta']['layer'] in SOURCE_LAYERS
                  and not self._clock(r['edge']['target'])))]
        # Shared ancestors may have been expanded under another path. Use propagated
        # causes, not just this branch's displayed traversal, for failed predicates.
        causes = {tuple(path[-2:]) for path in check['affected_paths'] if len(path) > 2}
        missing = {path[-1] for path in check['affected_paths'] if len(path) > 1}
        problems = {encoded(r['edge']): r for r in rows if not r['historical_only']
                    and r['check']['predicate_result'] in ('fails', 'cannot_check') and
                    ((r['edge']['source'], r['edge']['target']) in causes or
                     (r['edge']['target'] is None and r['edge']['source'] in missing))}
        failed = [r for r in problems.values() if r['check']['predicate_result'] == 'fails']
        changed = [r for r in below if r['check']['version_changed'] and r['check']['predicate_result'] in
                   ('not_declared', 'not_checked')]
        changed.sort(key=lambda r: r['check'].get('change_kind') == 'append_only')
        unknown = [r for r in problems.values() if r['check']['predicate_result'] == 'cannot_check']
        source_count = len({encoded(r['edge']) for r in below} | problems.keys())
        parts = []
        for rows_, head in ((failed, 'PREDICATE FAILS on'), (changed, 'changed:')):
            if rows_:
                example = self._describe_edge(rows_[0])
                parts.append(f"{len(rows_)} of {source_count} underlying sources {head} "
                             f"{example['target']['what']}: {example['check'].get('summary', 'changed')}")
        if unknown:
            reasons = ', '.join(sorted({str(r['check']['reason']) for r in unknown}))
            parts.append(f'{len(unknown)} could not be checked ({reasons})')
        if parts:
            result['check']['sources'] = '; '.join(parts)
        result['check'].update(below_failed=len(failed) + len(unknown), below_changed=len(changed),
                              below_content_changed=sum(r['check'].get('change_kind') != 'append_only' for r in changed))
        return result

    def _clock(self, version):
        """The simulated-day read, which changes every week by definition."""
        request = self.events[self.nodes[version]['event_id']]['request'].get('request') or {}
        return request.get('path') == '/vars'

    def _index(self, cutoff):
        started = time.monotonic()
        reset = not hasattr(self, '_indexed') or (cutoff is not None and cutoff < self._indexed)
        if reset:
            self.events, self.nodes, self.records = {}, {}, {}
            self.outgoing, self.incoming = defaultdict(list), defaultdict(list)
            self.latest, self.reads, self.event_day = {}, defaultdict(list), {}
            self._latest_agent = {}
            self.same_bytes, self.members = {}, defaultdict(list)
            self._files, self._outputs, self._inputs = {}, defaultdict(set), defaultdict(set)
            self._codes, self._private_pending = defaultdict(set), {}
            self._edge_keys, self._pending_events = set(), set()
            self._graph_versions, self._traces = defaultdict(int), {}
            self._indexed = self._event_cutoff = self._last_day = 0
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
                self._event_cutoff = max(self._event_cutoff, row['event_row'])
                try:
                    self.store._visible(conn, row['event_id'])
                except KeyError:
                    continue
                request = json.loads(row['request'])
                # The server records each weekly dashboard with its simulated day; later
                # events happen on that day. Agents reason in simulated days, not clock time.
                if row['event_id'] not in self.events and request['kind'] == 'dashboard_generation':
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
        self._check_day = max(day, self.registry.sim_day() or 0)
        new_nodes = {}
        for row in versions:
            if row['event_id'] not in self.events:
                continue
            meta = json.loads(row['metadata'])
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
                parent = self.events.get(event['request'].get('parent_event_id'), {})
                if not parent.get('request', {}).get('kind', '').startswith('pf_'):
                    self._latest_agent[self._key(row['version_id'])] = row['version_id']
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
                key = meta['object_id']
                previous = files.get(key)
                if previous and self.nodes[previous]['content_hash'] == row['content_hash']:
                    self._edge(version, previous, 'same_content_observation')
                else:
                    files[key] = version
                same = (key, row['content_hash'])
                self.same_bytes[version] = same
                self.members[same].append(version)
                self._graph_versions[same] += 1
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
        return evidence_key(version, meta, event['query'], event['result'], event['request'].get('request'),
                            event['request'])

    def _edge(self, source, target, kind, origin='automatic_capture', **details):
        if target not in self.nodes and origin != 'agent_declaration':
            return
        edge = dict(source=source, target=target if target in self.nodes else None,
                    kind=kind, origin=origin, **details)
        identity = encoded(edge)
        if identity in self._edge_keys:
            return
        self._edge_keys.add(identity)
        self._graph_versions[self._node(source)] += 1
        if target in self.nodes:
            self._graph_versions[self._node(target)] += 1
        self.outgoing[source].append(edge)
        if edge['target']:
            self.incoming[target].append(edge)

    def _objects(self, version):
        if version in self.records:
            return [dict(o, basis='agent_declaration') for o in self.records[version]['objects']]
        return [dict(kind=o['kind'], id=o['value'], basis=o['basis'], field=o.get('path'))
                for o in self.nodes[version]['meta'].get('objects', [])]

    def _target(self, target):
        if 'path' in target and evidence_handles.VERSIONED.fullmatch(target['path']) and not any(
                row['meta'].get('object_id') == target['path'] for row in self.nodes.values()):
            target = {'version': target['path']}  # forecast.json@v3 written as a path
        if 'path' in target and not any(row['meta']['layer'] == 'file_bytes' and row['meta']['object_id'] == target['path']
                                        for row in self.nodes.values()):
            # An output or query named without its version, e.g. analyze.py.out or query7: the latest.
            members = [v for v in evidence_handles.index(self.store).latest(target['path']) or [] if v in self.nodes]
            if members:
                return members[-1]
        if 'version' in target and evidence_handles.RECORD.fullmatch(target['version']):
            target = {'record': target['version']}
        if 'version' in target:
            members = [v for v in self.resolver.lookup(target['version']) or [] if v in self.nodes]
            if not members:
                raise ValueError('Unknown version handle; pf log <file or output> lists its versions')
            return members[-1]
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
        if (request.get('parsed') or {}).get('tool') == 'get_group_insights':
            dates = meta['content_time'].get('fields', [])
            result['snapshot_day'] = next((f['value'] for f in dates if f['field'].endswith('snapshot_day')), None)
            result['checked_day'] = self._check_day
            result['refresh_hint'] = 'research_group completes a new survey; calling get_group_insights again only retrieves it'
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
        return {k: full[k] for k in ('version', 'day', 'what', 'truncated', 'extent',
                                    'snapshot_day', 'checked_day', 'refresh_hint') if k in full} | (
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
            if self._rerun(member):
                item['what'] += ' [rerun by PF check]'
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
        width = max((len(f"v{g['k']} d{g['day']}") for g in origins), default=0)
        for origin_group, line in zip(origins, lines):
            label = f"v{origin_group['k']} d{origin_group['day']}" if origin_group is not previous else ''
            out.append(f'{label:<{width}} | {line}')
            previous = origin_group
        notes = [g for g in {id(g): g for g in origins}.values() if g.get('note')]
        if notes:
            out.append('Notes: ' + ' · '.join(f"v{g['k']} (day {g['note'][0]}): \"{pf_render.short(g['note'][1], 120)}\""
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
            text += f"\n{item['version']} · day {item['day']} · excerpt:\n"
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

    def _read(self, target, offset, baseline=None):
        meta, raw = self.resolver.content(target)
        try:
            content = raw.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ValueError('This captured version is not UTF-8 text') from exc
        header = {}
        if self.values['mode'] == 'diff':
            baseline = baseline or self._target(self.values['baseline'])
            key = self._key(target)
            if key != self._key(baseline) or not (key[0] in ('query', 'file_bytes', 'registered_text', 'registered_script',
                                                              'dashboard') or key[0].startswith(('script_', 'command_'))):
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
        # Leave room for the header/cursor inside Bash's 30,000-character return.
        end = min(len(content), offset + 24000)
        header.update(range=[offset, end], total_chars=len(content),
                      next_cursor=self._cursor(end) if end < len(content) else None)
        if end < len(content):
            header['truncated_reason'] = 'character_limit'
        prefix = encoded(header).decode() + '\n'
        # The continuation command follows the body; the header keeps the delivery fields.
        more = f"\n[{len(content) - end:,} more characters: pf more {header['next_cursor']}]" if header['next_cursor'] else ''
        from .pf_read import capture_read
        return capture_read(self, target, prefix + content[offset:end] + more,
                            offset, end, len(content))
