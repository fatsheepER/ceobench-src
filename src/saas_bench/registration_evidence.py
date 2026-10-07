"""Resolve declaration references against captured versions and actual model sends."""
from contextlib import closing, contextmanager
import csv
import io
import json
from pathlib import PurePosixPath
import re
import subprocess

from . import evidence_handles
from .evidence_handles import HANDLES  # noqa: F401  (re-exported for callers)


WEEK_LABEL = re.compile('week-[1-9][0-9]*')


def week_label(day):
    """Label of the weekly harness commit that closes the week containing day."""
    return f'week-{max(day or 0, 0) // 7 + 1}'


def week_commit_subject(label):
    # Must equal the runner's weekly commit message (`_commit_weeks_up_to`).
    week = int(label.split('-')[1])
    return f'Week {week} (day {week * 7}) [{label}]'


def git_reference(workspace, evidence):
    path = evidence['path']
    commit = evidence.get('commit')
    if '@' in path:
        path, suffix = path.rsplit('@', 1)
        if commit is not None and commit != suffix:
            raise ValueError('Conflicting commit references')
        commit = suffix
    p = PurePosixPath(path)
    if not path or p.is_absolute() or '..' in p.parts or p.as_posix() != path or '\x00' in path:
        raise ValueError('Evidence path must be a normalized workspace-relative file path')
    def git(*args):
        result = subprocess.run(['git', '--no-replace-objects', '-C', str(workspace), *args],
                                capture_output=True, timeout=10)
        if result.returncode:
            raise ValueError('Cannot resolve file/commit; commit the file yourself or use unknown with a reason')
        return result.stdout.decode().strip()
    if commit is None:
        full = git('rev-parse', '--verify', 'HEAD^{commit}')
    elif WEEK_LABEL.fullmatch(commit):
        subject = week_commit_subject(commit)
        matches = [line[:40] for line in git('log', '--format=%H %s').splitlines() if line[41:] == subject]
        if not matches:
            raise ValueError('That weekly commit does not exist yet; a bare path cites this week\'s file')
        full = matches[-1]
    else:
        if not re.fullmatch('[0-9a-fA-F]{1,40}', commit):
            raise ValueError('Commit must be a unique hexadecimal prefix')
        # --disambiguate requires four characters; enumerate commits for shorter inputs too.
        matches = [line.split()[0] for line in git('cat-file', '--batch-all-objects',
                   '--batch-check=%(objectname) %(objecttype)').splitlines()
                   if line.split()[1] == 'commit' and line.startswith(commit.lower())]
        if len(matches) != 1:
            raise ValueError('Commit prefix is missing or ambiguous; use a longer prefix or unknown')
        full = matches[0]
    if git('cat-file', '-t', full + ':' + path) != 'blob':
        raise ValueError('Reference must name a committed file')
    # No git show, checkout, snapshot, or commit: existence checking reads no cited contents.
    return dict(path=path, commit=commit if WEEK_LABEL.fullmatch(commit or '') else full[:7]), full


def weekly_reference(workspace, path, label):
    """Cite a working-tree file as the weekly harness commit labelled `label` will store it."""
    p = PurePosixPath(path)
    if not path or p.is_absolute() or '..' in p.parts or p.as_posix() != path or '\x00' in path:
        raise ValueError('Evidence path must be a normalized workspace-relative file path')
    target = workspace / path
    if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(workspace):
        raise ValueError('Reference must name an existing workspace file, or use unknown with a reason')
    ignored = subprocess.run(['git', '-C', str(workspace), 'check-ignore', '-q', '--', path],
                             capture_output=True, timeout=10)
    if ignored.returncode == 0:
        raise ValueError('File is ignored by Git and never committed; use unknown with a reason')
    # Bound silently to that week's closing commit; later edits in the week are included.
    return dict(path=path, commit=label)


def json_spans(text):
    """Parse JSON and locate values/keys without searching for matching value strings."""
    decoder = json.JSONDecoder()
    spans, keys = {}, {}
    def ws(i):
        while i < len(text) and text[i].isspace():
            i += 1
        return i
    def parse(i, path):
        i = ws(i)
        start = i
        if text[i] == '{':
            value, i = {}, ws(i + 1)
            while text[i] != '}':
                key_start = i
                key, end = decoder.raw_decode(text, i)
                if not isinstance(key, str) or key in value:
                    raise ValueError('Duplicate or invalid JSON key')
                i = ws(end)
                if text[i] != ':':
                    raise ValueError('Invalid JSON object')
                keys[path + (key,)] = (key_start, end)
                value[key], i = parse(i + 1, path + (key,))
                i = ws(i)
                if text[i] != ',':
                    break
                i = ws(i + 1)
            if text[i] != '}':
                raise ValueError('Invalid JSON object')
            i += 1
        elif text[i] == '[':
            value, i = [], ws(i + 1)
            while text[i] != ']':
                item, i = parse(i, path + (len(value),))
                value.append(item)
                i = ws(i)
                if text[i] != ',':
                    break
                i = ws(i + 1)
            if text[i] != ']':
                raise ValueError('Invalid JSON array')
            i += 1
        else:
            value, i = decoder.raw_decode(text, i)
        spans[path] = (start, i)
        return value, i
    try:
        value, end = parse(0, ())
        if ws(end) != len(text):
            raise ValueError('Trailing JSON data')
    except (IndexError, json.JSONDecodeError) as exc:
        raise ValueError('Evidence is not valid JSON') from exc
    return value, spans, keys


def selected_ranges(text, selector, kind):
    if selector is None:
        return [(0, len(text))]
    if kind == 'csv':
        lines = text.splitlines(keepends=True)
        reader = csv.DictReader(io.StringIO(text, newline=''))
        headers = reader.fieldnames
        if not headers or len(set(headers)) != len(headers):
            raise ValueError('CSV headers are missing or ambiguous')
        header_end = sum(map(len, lines[:reader.line_num]))
        rows, spans = [], []
        previous = reader.line_num
        for row in reader:
            rows.append(row)
            spans.append((sum(map(len, lines[:previous])), sum(map(len, lines[:reader.line_num]))))
            previous = reader.line_num
        index = select_row(rows, headers, selector)
        start, end = spans[index]
        # A record terminator is formatting, not part of the selected CSV cells.
        end = start + len(text[start:end].rstrip('\r\n'))
        return [(0, header_end), (start, end)]
    value, spans, keys = json_spans(text)
    if 'path' in selector:
        path = ()
        ranges = []
        for token in selector['path'].split('/')[1:]:
            if re.search(r'~(?![01])', token):
                raise ValueError('Invalid JSON Pointer escape')
            token = token.replace('~1', '/').replace('~0', '~')
            key = int(token) if isinstance(value, list) and token.isdecimal() else token
            path += (key,)
            try:
                value = value[key]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError('JSON selection does not exist') from exc
            if path in keys:
                ranges.append(keys[path])
        return ranges + [spans[path]]
    if not isinstance(value, dict) or not isinstance(value.get('rows'), list):
        raise ValueError('Selection requires a table result or CSV file; JSON files use path')
    rows, columns = value['rows'], value.get('columns', [])
    index = select_row(rows, columns, selector)
    fields = set(selector.get('row', {})) | {selector['col']}
    return [r for field in fields for r in (keys[('rows', index, field)], spans[('rows', index, field)])]


def select_row(rows, columns, selector):
    col = selector.get('col')
    if col not in columns:
        raise ValueError('Selected column is missing')
    if 'row' not in selector:
        if len(rows) != 1 or len(columns) != 1:
            raise ValueError('Omit row only for a 1x1 result')
        return 0
    def equal(a, b):
        return type(a) is type(b) and a == b
    matches = [i for i, row in enumerate(rows) if all(k in row and equal(row[k], v)
                for k, v in selector['row'].items())]
    if len(matches) != 1:
        raise ValueError('Row selector must match exactly one row; matched ' + ('no rows' if not matches else 'multiple rows'))
    return matches[0]


def covered(wanted, occurrences):
    ranges = sorted(item['source_range'] for item in occurrences)
    for start, end in wanted:
        for a, b in ranges:
            if a <= start:
                start = max(start, b)
        if start < end:
            return False
    return True


class EvidenceResolver:
    def __init__(self, store):
        self.store = store
        self._cache = None

    @contextmanager
    def cached(self):
        previous = self._cache
        if previous is None:
            self._cache = {}
            self._cache_bytes = 0
            self.read_stats = dict(verified_reads=0, verified_bytes=0, cache_hits=0)
        try:
            yield
        finally:
            self._cache = previous

    def content(self, version, *, connection=None):
        try:
            if self._cache is not None and version in self._cache:
                self.read_stats['cache_hits'] += 1
                return self._cache[version]
            value = self.store.get_content(version, **({'connection': connection} if connection is not None else {}))
            if self._cache is not None:
                self.read_stats['verified_reads'] += 1
                self.read_stats['verified_bytes'] += len(value[1])
            # Per-operation verified bytes; bounded memory, no cross-check integrity bypass.
            if self._cache is not None and self._cache_bytes + len(value[1]) <= 32 * 1024 * 1024:
                self._cache[version] = value
                self._cache_bytes += len(value[1])
            return value
        except (KeyError, ValueError) as exc:
            self.store.fail(exc)
            raise RuntimeError('Captured evidence is missing or corrupt; collection stopped') from exc

    def identity(self, version):
        meta, body = self.content(version)
        if meta['layer'] == 'file_text':
            raw, _ = self.content(meta['derived_from'])
            return meta['derived_from'], raw['object_id'], body.decode(), 'file'
        if meta['layer'] == 'query_model_projection':
            raw, _ = self.content(meta['derived_from'])
            return meta['derived_from'], raw['object_id'], body.decode(), 'query'
        if meta['layer'] == 'file_bytes':
            return version, meta['object_id'], body.decode('utf-8'), 'file'
        if meta['layer'] == 'registered_text':
            return version, meta['object_id'], body.decode(), 'record'
        event = self.store.read_event(meta['created_by_event'])
        if meta['layer'] == 'server_public_response' and event['query_definition']:
            return version, meta['object_id'], body.decode(), 'query'
        # Receipts and streams without a persistent object identity are immutable
        # observations. They can be cited by handle without inventing an update chain.
        return version, meta.get('object_id') or version, body.decode(), 'other'

    def handle(self, version, allocate=True):
        """Agent-facing handle, e.g. forecast.json@v3; see evidence_handles."""
        return evidence_handles.index(self.store).name(version, allocate)

    def lookup(self, handle):
        return evidence_handles.index(self.store).lookup(handle)

    @contextmanager
    def binding_scope(self, request_event=None, context_id=None):
        """Share one originating request's verified sources across all references."""
        previous = getattr(self, '_binding_context', None)
        self._binding_context = dict(request_event=request_event, context_id=context_id)
        try:
            with self.cached():
                yield
        finally:
            self._binding_context = previous

    def _request_sources(self):
        from .execution_capture import CURRENT_EVENT
        scope = self._binding_context
        if 'sources' in scope:
            return scope
        handles = evidence_handles.index(self.store)
        handles.refresh()
        if CURRENT_EVENT.get() and not scope['request_event']:
            facts = self.store.read_event(CURRENT_EVENT.get())['request']
            scope.update(request_event=facts.get('model_request_event'), context_id=facts.get('model_context_id'))
        # Offline callers and old captures lack batch origin facts. Stay in their
        # latest context; never look through earlier weeks for a better read.
        if not scope['request_event'] and not scope['context_id'] and handles.model_requests:
            scope['context_id'] = handles.model_requests[-1][1]
            for event, context in reversed(handles.model_requests):
                if context != scope['context_id']:
                    break
                result = self.store.read_event(event)['result']
                if result.get('send_state') == 'response_received' and result.get('status') == 'succeeded':
                    scope['request_event'] = event
                    break
        scope['sources'] = {}
        if scope['request_event']:
            event = self.store.read_event(scope['request_event'])
            if event['request']['kind'] != 'model_request' or (
                    scope['context_id'] != event['request']['request'].get('context_id')):
                raise ValueError('Registration origin does not match its model context')
            if (event['result'].get('send_state') != 'response_received' or
                    event['result'].get('status') != 'succeeded'):
                raise ValueError('Registration requires a successful originating model request')
            for slot in (':occurrences', ':reconstructed'):
                version = scope['request_event'] + slot
                if version not in event['outputs']:
                    continue
                for item in json.loads(self.content(version)[1]):
                    source, _, content, kind = self.identity(item['version_id'])
                    scope['sources'].setdefault(source, []).append((item, content, kind))
        return scope

    def _current_write(self, version, scope):
        handles = evidence_handles.index(self.store)
        meta, _ = self.content(version)
        kind, facts, day = handles.events[meta['created_by_event']]
        context = facts.get('model_context_id')
        outside = context != scope['context_id'] if context is not None else day != handles.day
        if outside:
            return False
        # Only successful writes count; a before-snapshot is not authored evidence.
        if meta['layer'] != 'file_bytes' or not version.endswith('_after'):
            return False
        event = self.store.read_event(meta['created_by_event'])
        return (event['result'].get('status') == 'succeeded' and
                meta.get('object_id') in event['result'].get('changed_paths',
                                                          event['result'].get('written_paths', [])))

    def resolve(self, evidence, reference, accept=None):
        if getattr(self, '_binding_context', None) is None:
            with self.binding_scope():
                return self.resolve(evidence, reference, accept)
        handles = evidence_handles.index(self.store)
        handles.refresh()
        members = None
        if 'version' in evidence:
            named = self.lookup(evidence['version'])
            if not named:
                raise ValueError('Unknown version handle; cite the handle shown in a [pf: ...] line, or a file path')
            members = set(named)
            _, object_id, _, kind = self.identity(named[-1])
            groups = handles.versions(named[-1])
        elif 'path' in evidence:
            object_id, kind = evidence['path'], 'file'
            groups = handles.groups.get(('file', object_id), [])
        elif 'record' in evidence:
            object_id, kind = 'record:' + evidence['record'].split('.')[0], 'record'
            with closing(self.store.connect()) as conn:
                candidates = [row[0] for row in conn.execute(
                    "SELECT version_id FROM versions WHERE json_extract(metadata, '$.object_id')=? ORDER BY rowid DESC",
                    (object_id,)) if row[0] in handles.info]
            groups = []
        elif 'sql' in evidence:
            object_id, kind = None, 'query'
            groups = [group for key, history in handles.groups.items()
                      if key[0] == 'query' and json.loads(key[1])[3] == evidence['sql'] for group in history]
        else:
            raise ValueError('Unsupported PF evidence reference')
        if kind != 'record':
            candidates = [v for group in reversed(groups) for v in reversed(group['members'])]
        if not candidates:
            raise ValueError('No captured evidence exists for this reference; cite a captured handle or use unknown with a reason')
        latest = candidates[0]
        if members is not None:
            candidates = [c for c in candidates if c in members]
        if accept is not None:
            candidates = [c for c in candidates if accept(c)]
            if not candidates:
                raise ValueError('The committed bytes were never captured')
        predicate = reference.get('predicate', {})
        whole = not reference.get('select') and not predicate
        if kind == 'record':
            if not whole:
                raise ValueError('Registered text supports whole-text equality only')
            wanted = evidence.get('record', '')
            for candidate in candidates:
                if '.' in wanted and json.loads(self.content(candidate)[1])['version'] != wanted:
                    continue
                return dict(version_id=candidate, latest_version_id=latest, source_truncated=False,
                            delivered_in=[], basis='registered_text')
            raise ValueError('Unknown registered text revision; use unknown with a reason')
        scope = self._request_sources()
        selectors = ([predicate['left'], predicate['right']] if predicate.get('type') == 'compare'
                     else [reference.get('select')])
        if predicate.get('type') == 'compare' and kind != 'query':
            raise ValueError('compare requires one query view')
        if kind in ('file', 'other') and not str(object_id).endswith(('.csv', '.json')) and any(selectors):
            raise ValueError('Plain text supports whole-text equality only')
        for candidate in candidates:
            matches = scope['sources'].get(candidate, [])
            written = kind == 'file' and self._current_write(candidate, scope)
            authored = written and self.authored(candidate)
            if written and (whole or authored) and (not whole or not matches):
                if whole:
                    return self._whole_binding(candidate, latest, [], 'written', fully_known=authored)
                meta, raw = self.content(candidate)
                return dict(version_id=candidate, latest_version_id=latest, source_truncated=False,
                            delivered_in=[], authored_by=meta['created_by_event'],
                            selected_ranges=selected_ranges(raw.decode(), reference.get('select'),
                                                            'csv' if str(object_id).endswith('.csv') else 'json'))
            if not matches:
                if written:
                    raise ValueError('Selected evidence was not fully delivered in this request; read the computed file first')
                continue
            for source_version in dict.fromkeys(item['version_id'] for item, _, _ in matches):
                group = [(item, content) for item, content, _ in matches if item['version_id'] == source_version]
                text = group[0][1]
                delivery = [dict(request_event=scope['request_event'], occurrence=item) for item, _ in group]
                if whole:
                    return self._whole_binding(candidate, latest, delivery, 'observed', fully_known=authored)
                ranges = [r for select in selectors for r in selected_ranges(
                    text, select, 'csv' if str(object_id).endswith('.csv') else 'json')]
                if covered(ranges, [item for item, _ in group]):
                    meta, _ = self.content(candidate)
                    return dict(version_id=candidate, latest_version_id=latest,
                                source_truncated=meta['source_truncated'], delivered_in=delivery, selected_ranges=ranges)
            raise ValueError('Selected evidence was not fully delivered in this request; use unknown with a reason')
        if whole and (members is not None or accept is not None):
            return self._whole_binding(candidates[0], latest, [],
                                       'explicit_handle' if members is not None else 'committed_bytes')
        if kind == 'query':
            raise ValueError('Evidence has not been delivered in this request: cite the command output by the '
                             'first handle in its [pf: ...] line, or read the query version first')
        raise ValueError('Evidence has not been delivered in this request or written in this context; '
                         'read it first, cite an explicit whole version, or use unknown with a reason')

    def _whole_binding(self, version, latest, delivered, basis, fully_known=False):
        from .pf_queries import PUBLIC_LAYERS
        meta, raw = self.content(version)
        if meta['layer'] not in PUBLIC_LAYERS or meta.get('pf_retrieval'):
            raise ValueError('Only public captured evidence can be cited')
        if meta['extent'] != 'full' or meta['source_truncated']:
            raise ValueError('Only part of this object was captured; cite a complete captured output or use unknown')
        ranges = [item['occurrence'] for item in delivered]
        full = covered([(0, len(raw.decode('utf-8')))], ranges)
        result = dict(version_id=version, latest_version_id=latest, source_truncated=False,
                    basis=basis, delivered_in=delivered, capture_extent=meta['extent'],
                    reading_scope='full' if full else 'authored' if fully_known else 'partial' if ranges else 'not_in_request',
                    read_ranges=[item['source_range'] for item in ranges])
        if fully_known:
            result['authored_by'] = meta['created_by_event']
        return result

    def authored(self, version):
        """Whether the model wrote or successfully edited this captured file version.

        write_file writes its content argument verbatim; a Bash command counts only when
        the complete decoded file text appears in the command the model wrote (for
        example a quoted heredoc). Successful edit_file calls own their resulting
        snapshot; their internal reads do not count as model delivery.
        """
        meta, raw = self.content(version)
        if meta['layer'] != 'file_bytes':
            return False
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            return False
        event = self.store.read_event(meta['created_by_event'])
        request = event['request']
        args = request.get('request') or {}
        if request['kind'] == 'edit_file':
            return (version.endswith('_after') and event['result'].get('status') == 'succeeded'
                    and meta['object_id'] in event['result'].get('written_paths', []))
        if request['kind'] == 'write_file':
            return args.get('content') == text
        return request['kind'] == 'bash' and bool(text) and text in (args.get('command') or '')
