"""Current evidence comparison and dependency-edge predicates for PF queries."""
from collections import defaultdict
import csv
from decimal import Decimal, InvalidOperation
import io
import json
import operator
import os
import stat

from .execution_capture import CURRENT_EVENT
from .pf_refresh import replayable
from .registration_evidence import selected_ranges, select_row
from .sql_evidence import encoded, whole_rows


OPS = {'>': operator.gt, '>=': operator.ge, '<': operator.lt, '<=': operator.le}


def table(raw, kind):
    if kind == 'csv':
        reader = csv.DictReader(io.StringIO(raw.decode(), newline=''))
        columns = reader.fieldnames or []
        rows = list(reader)
        if not columns or any(None in row or None in row.values() for row in rows):
            raise ValueError('invalid_csv')
        data = dict(success=True, columns=columns, rows=rows)
    else:
        data = json.loads(raw)
    if len(set(data['columns'])) != len(data['columns']):
        raise ValueError('duplicate_columns')
    return data


def cell(raw, selector, kind):
    text = raw.decode()
    if 'path' in selector:
        # Reuse registration's exact JSON Pointer validation and key ambiguity checks.
        ranges = selected_ranges(text, selector, 'json')
        start, end = ranges[-1]
        value = json.loads(text[start:end])
    else:
        data = table(raw, kind)
        value = data['rows'][select_row(data['rows'], data['columns'], selector)][selector['col']]
    if value is None:
        raise ValueError('selected_value_null')
    if type(value) not in (str, bool, int, float):
        raise ValueError('selected_value_not_scalar')
    return value


def number(value, kind):
    if type(value) not in (int, float) and not (kind == 'csv' and isinstance(value, str)):
        raise ValueError('selected_value_not_numeric')
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('selected_value_not_numeric') from exc
    if not result.is_finite():
        raise ValueError('selected_value_not_finite')
    return result


def compare(resolver, before, after, reference, kind):
    """Return whole-version change and whether this dependency still holds."""
    old_meta, old = resolver.content(before)
    new_meta, new = resolver.content(after)
    events = [resolver.store.read_event(m['created_by_event']) for m in (old_meta, new_meta)]
    for meta, event in zip((old_meta, new_meta), events):
        if event['result'].get('status') != 'succeeded':
            raise ValueError('read_' + event['result'].get('status', 'failed'))
        if meta['source_truncated']:
            raise ValueError('source_truncated')
        if meta['extent'] != 'full':
            raise ValueError('incomplete_capture')
        if kind == 'query':
            comparison = event['result'].get('comparison', {})
            if comparison.get('status') != 'available':
                raise ValueError(comparison.get('reason', 'comparison_unavailable'))
    if kind in ('query', 'csv'):
        bodies = [encoded(table(raw, kind)) for raw in (old, new)]
        comparisons = [whole_rows(raw) for raw in bodies]
        for metadata, _ in comparisons:
            if metadata['status'] != 'available':
                raise ValueError(metadata['reason'])
        changed = comparisons[0][1] != comparisons[1][1]
    elif kind == 'record':
        changed = before != after
    elif kind == 'public':
        changed = encoded(json.loads(old)) != encoded(json.loads(new))
    else:
        changed = old != new
    try:
        return changed, evaluate(old, new, reference, kind, changed), None
    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
        return changed, None, str(exc)


def evaluate(old, new, reference, kind, changed):
    predicate = reference.get('predicate')
    selector = reference.get('select')
    if not predicate and not selector:
        return not changed
    if predicate and predicate['type'] == 'compare':
        if kind != 'query':
            raise ValueError('compare_requires_one_query')
        # Baseline selectors must also be valid, but the condition is evaluated now.
        for select in (predicate['left'], predicate['right']):
            number(cell(old, select, kind), kind)
        left, right = [number(cell(new, predicate[k], kind), kind) for k in ('left', 'right')]
        return OPS[predicate['op']](left, right)
    baseline, current = [cell(raw, selector, kind) for raw in (old, new)]
    if not predicate:
        if type(baseline) is not type(current):
            raise ValueError('selected_value_type_changed')
        return encoded(baseline) == encoded(current)
    baseline, current = number(baseline, kind), number(current, kind)
    if predicate['type'] == 'tolerance':
        return abs(current - baseline) <= Decimal(str(predicate['amount']))
    return OPS[predicate['op']](current, Decimal(str(predicate['value'])))


class StaleCheck:
    def __init__(self, queries):
        self.q = queries
        self.store, self.resolver = queries.store, queries.resolver

    def run(self, rows):
        current, failures, representatives = {}, {}, {}
        for row in rows:
            target = row['edge']['target']
            if target and not row['historical_only']:
                representatives.setdefault(self.q._key(target), target)
        remote = {}
        for key, version in representatives.items():
            meta = self.q.nodes[version]['meta']
            event = self.store.read_event(meta['created_by_event'])
            if meta['layer'] == 'file_bytes':
                current[key], failures[key] = self._file(meta['object_id'])
            elif meta['layer'] == 'server_public_response' and replayable(event):
                remote[version] = key
            else:
                current[key] = self.q.latest[key]
        if remote:
            try:
                if self.q.refresh is None:
                    raise ValueError('refresh_unavailable')
                refreshed = self.q.refresh(list(remote), CURRENT_EVENT.get())
                for version, key in remote.items():
                    current[key] = refreshed[version]
            except Exception:
                self.store.assert_healthy(quiescent=False)
                for key in remote.values():
                    failures[key] = 'refresh_unavailable' if self.q.refresh is None else 'refresh_failed'
        for row in rows:
            edge, target = row['edge'], row['edge']['target']
            ref = edge.get('reference', {})
            check = dict(current_version=None, version_changed=None, predicate_result='not_checked',
                         notice=None, reason=None, affected=False, affected_paths=[])
            row['check'] = check
            if row['historical_only']:
                check['reason'] = 'historical_only'
                continue
            if target is None:
                check.update(predicate_result='cannot_check', notice='cannot_check',
                             reason=edge.get('reason') or 'missing_evidence', affected=True)
                continue
            meta = self.q.nodes[target]['meta']
            event = self.q.events[self.q.nodes[target]['event_id']]
            key = self.q._key(target)
            if event['result'].get('classification') == 'write_receipt':
                check.update(current_version=target, reason='write_receipt')
                continue
            check['current_version'] = current.get(key)
            kind = ('query' if key[0] == 'query' else 'record' if target in self.q.records else
                    'public' if meta['layer'] == 'server_public_response' else
                    'csv' if meta['layer'] == 'file_bytes' and meta['object_id'].endswith('.csv') else 'other')
            try:
                if failures.get(key):
                    raise ValueError(failures[key])
                changed, holds, reason = compare(self.resolver, target, current[key], ref, kind)
                check['version_changed'] = changed
                if reason:
                    raise ValueError(reason)
                check.update(version_changed=changed, affected=not holds,
                             predicate_result=('holds' if holds else 'fails') if ref.get('predicate') or ref.get('select') else 'not_declared')
                if ref.get('predicate') and not holds:
                    check['notice'] = 'predicate_failed'
                elif changed:
                    check['notice'] = 'strong_dependency_version_changed' if edge['origin'] == 'agent_declaration' else 'version_changed'
            except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                check.update(predicate_result='cannot_check', notice='cannot_check', reason=str(exc), affected=True)
        self._propagate(rows)
        return rows

    def _file(self, name):
        event = self.store.begin_event('stale_file_read', {'path': name}, parent=CURRENT_EVENT.get())
        version = None
        try:
            path = (self.q.workspace / name).resolve()
            if not path.is_relative_to(self.q.workspace):
                raise ValueError('file_outside_workspace')
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError('not_regular_file')
                raw = stream.read()
                after = os.fstat(stream.fileno())
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('file_changed_during_read')
        except (OSError, ValueError) as exc:
            reason = 'file_missing' if isinstance(exc, FileNotFoundError) else 'file_unavailable'
            self.store.complete(event, 'failed', reason=reason)
            return None, reason
        version = self.store.version(event, 'file', raw, layer='file_bytes', object_id=name)
        self.store.complete(event)
        return version, None

    def _propagate(self, rows):
        node = self.q._node
        outgoing = defaultdict(list)
        seen = set()
        for row in rows:
            key = encoded(row['edge'])
            if not row['historical_only'] and key not in seen:
                outgoing[node(row['edge']['source'])].append(row)
                seen.add(key)

        def causes(row, path):
            target, check = row['edge']['target'], row['check']
            if check['reason'] in ('historical_only', 'write_receipt'):
                return []
            if check['predicate_result'] == 'holds':
                return []
            found = [path + ([target] if target else [])] if check['affected'] else []
            if target and node(target) not in {node(v) for v in path}:
                for child in outgoing[node(target)]:
                    found.extend(causes(child, path + [target]))
            return found

        paths = [causes(row, [row['edge']['source']]) for row in rows]
        for row, found in zip(rows, paths):
            row['check']['affected_paths'] = list(dict.fromkeys(tuple(p) for p in found))
            row['check']['affected'] = bool(found)
