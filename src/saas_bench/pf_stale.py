"""Current evidence comparison and dependency-edge predicates for PF queries."""
from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation
import io
import json
import operator
import os
import stat
import time

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


def compare(resolver, before, after, reference, kind, events=None):
    """Return whole-version change and whether this dependency still holds."""
    old_meta, old = resolver.content(before)
    new_meta, new = resolver.content(after)
    cached = events if events is not None else {}
    for meta, failure in ((old_meta, 'original_execution_failed'), (new_meta, 'refresh_failed')):
        event_id = meta['created_by_event']
        if event_id not in cached:
            cached[event_id] = resolver.store.read_event(event_id, evidence=True)
        event = cached[event_id]
        if event['result'].get('status') != 'succeeded':
            raise ValueError(failure + ":" + event['result'].get('status', 'unknown'))
        if meta['source_truncated']:
            raise ValueError('source_truncated')
        if meta['extent'] != 'full':
            raise ValueError('incomplete_capture')
        if kind == 'query':
            comparison = event['result'].get('comparison', {})
            if comparison.get('status') != 'available':
                raise ValueError(comparison.get('reason', 'comparison_unavailable'))
    change_kind = 'content_changed'
    if kind in ('query', 'csv'):
        bodies = [encoded(table(raw, kind)) for raw in (old, new)]
        comparisons = [whole_rows(raw) for raw in bodies]
        for metadata, _ in comparisons:
            if metadata['status'] != 'available':
                raise ValueError(metadata['reason'])
        changed = comparisons[0][1] != comparisons[1][1]
        tables = [json.loads(c[1]) for c in comparisons]
        if (changed and all(c[0]['scope'] == 'complete_for_query' for c in comparisons)
                and tables[0]['columns'] == tables[1]['columns']
                and not (Counter(tables[0]['rows']) - Counter(tables[1]['rows']))):
            change_kind = 'append_only'
    elif kind == 'record':
        changed = before != after
    elif kind == 'public':
        changed = encoded(json.loads(old)) != encoded(json.loads(new))
    else:
        changed = old != new
    try:
        return changed, evaluate(old, new, reference, kind, changed), None, change_kind if changed else None
    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
        return changed, None, str(exc), change_kind if changed else None


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

    def run(self, rows, day=None):
        started = time.monotonic()
        current, failures, representatives = {}, {}, {}
        retry = self.store.load_state('pf_refresh_failures') or {}
        diagnostics, compared, events = {}, {}, dict(self.q.events)
        for row in rows:
            target = row['edge']['target']
            if target and not row['historical_only'] and 'cached_check' not in row:
                representatives.setdefault(self.q._key(target), target)
        remote = {}
        for key, version in representatives.items():
            meta = self.q.nodes[version]['meta']
            event = self.store.read_event(meta['created_by_event'], evidence=True)
            if meta['layer'] == 'file_bytes':
                current[key], failures[key] = self._file(meta['object_id'], meta.get('owner_role'))
            elif meta['layer'] == 'server_public_response' and replayable(event):
                retry_key = encoded(key).decode()
                previous = retry.get(retry_key, {})
                source = self.q._latest_agent.get(key, version)
                if previous.get('source') != source:
                    previous = {}
                original = event['result']
                permanent = original.get('permanent_error', False)
                if key[0] == 'query' and original.get('http_status', 200) in (400, 403, 500):
                    # Old ledgers predate the explicit SQL error classification.
                    error = str(json.loads(self.resolver.content(version)[1]).get('error', '')).lower()
                    permanent |= original['http_status'] in (400, 403) or any(
                        hint in error for hint in ('syntax error', 'no such column', 'no such table', 'ambiguous column'))
                if day is not None and (permanent or previous.get('permanent')):
                    failures[key] = 'original_sql_error' if permanent else 'permanent_sql_error'
                elif day is not None and previous.get('retry_day', 0) > day:
                    failures[key] = f"retry_after_day_{previous['retry_day']}; last_verified_day_{previous['last_day']}"
                else:
                    remote[version] = key
                diagnostics[key] = dict(previous, source=source)
            else:
                current[key] = self.q.latest[key]
        refreshing = time.monotonic()
        if remote:
            try:
                if self.q.refresh is None:
                    raise ValueError('refresh_unavailable')
                refreshed = self.q.refresh(list(remote), CURRENT_EVENT.get())
                for version, key in remote.items():
                    current[key] = refreshed[version]
                    meta, _ = self.resolver.content(current[key])
                    receipt = self.store.read_event(meta['created_by_event'], evidence=True)
                    events[meta['created_by_event']] = receipt
                    result = receipt['result']
                    self.q._check_day = max(self.q._check_day, result.get('day') or 0)
                    mark = encoded(key).decode()
                    if result.get('status') == 'succeeded':
                        retry.pop(mark, None)
                    else:
                        previous = diagnostics[key]
                        attempts = previous.get('attempts', 0) + 1
                        checked_day = day if day is not None else self.q._check_day
                        retry[mark] = dict(source=previous['source'], attempts=attempts,
                            last_day=checked_day, retry_day=(checked_day or 0) + 7 * min(8, 2 ** min(attempts - 1, 3)),
                            permanent=result.get('permanent_error', False), result=current[key])
            except Exception:
                self.store.assert_healthy(quiescent=False)
                for key in remote.values():
                    failures[key] = 'refresh_unavailable' if self.q.refresh is None else 'refresh_failed'
        self.store.save_state('pf_refresh_failures', retry)
        comparing = time.monotonic()
        for row in rows:
            edge, target = row['edge'], row['edge']['target']
            ref = edge.get('reference', {})
            check = dict(current_version=None, version_changed=None, predicate_result='not_checked',
                         notice=None, reason=None, affected=False, affected_paths=[])
            row['check'] = check
            if 'cached_check' in row:
                row['check'] = dict(row['cached_check'])
                continue
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
                comparison_key = (target, current[key], encoded(ref), kind)
                if comparison_key not in compared:
                    try:
                        compared[comparison_key] = compare(self.resolver, target, current[key], ref, kind, events)
                    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
                        compared[comparison_key] = exc
                value = compared[comparison_key]
                if isinstance(value, Exception):
                    raise value
                changed, holds, reason, change_kind = value
                check['version_changed'] = changed
                check['change_kind'] = change_kind
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
            if (day is not None and target in getattr(self.q, 'review_pending', {})
                    and check['predicate_result'] not in ('holds', 'fails')):
                pending = self.q.review_pending[target]
                check.update(affected=True, predicate_result='cannot_check', notice='pending_review',
                             reason=f"pending_review; last_verified_day_{pending['last_day']}")
        self._propagate(rows)
        self.q.check_stats = dict(source_resolution_seconds=refreshing-started,
            refresh_seconds=comparing-refreshing, comparison_and_propagation_seconds=time.monotonic()-comparing,
            remote_sources=len(remote), shared_sources=len(representatives), unique_comparisons=len(compared))
        return rows

    def _file(self, name, owner=None):
        event = self.store.begin_event('stale_file_read', {'path': name}, parent=CURRENT_EVENT.get())
        version = None
        try:
            workspace = self.q.registry.repository(owner)
            path = workspace / name
            if path.is_symlink() or not path.resolve().is_relative_to(workspace):
                raise ValueError('file_outside_workspace')
            from .workspace_io import open_file
            with open_file(workspace, path) as stream:
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
        version = self.store.version(event, 'file', raw, layer='file_bytes', object_id=name, **({'owner_role': owner} if owner else {}))
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
