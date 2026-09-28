"""Resolve declaration references against captured versions and actual model sends."""
from contextlib import closing
import csv
import io
import json
from pathlib import PurePosixPath
import re
import subprocess


# Private state name of the lineage-wide vN handle table.
HANDLES = 'registration_handles'
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

    def content(self, version):
        try:
            return self.store.get_content(version)
        except (KeyError, ValueError) as exc:
            self.store.fail(exc)
            raise RuntimeError('Captured evidence is missing or corrupt; collection stopped') from exc

    def versions(self):
        with closing(self.store.connect()) as conn:
            result = []
            for row in conn.execute('SELECT version_id,event_id,metadata FROM versions ORDER BY rowid DESC'):
                try:
                    self.store._visible(conn, row['event_id'])
                except KeyError:
                    continue
                result.append((row['version_id'], json.loads(row['metadata'])))
        return result

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

    def handle(self, version):
        # One lineage-wide table: forks copy the evidence database and keep numbering,
        # so a handle an agent saw or wrote down never changes meaning.
        name = HANDLES
        handles = self.store.load_state(name) or {}
        if version not in handles.values():
            handles['v' + str(len(handles) + 1)] = version
            self.store.save_state(name, handles)
        return next(k for k, v in handles.items() if v == version)

    def resolve(self, evidence, reference, accept=None):
        versions = self.versions()
        explicit = None
        if 'version' in evidence:
            handles = self.store.load_state(HANDLES) or {}
            explicit = handles.get(evidence['version'])
            if explicit is None:
                raise ValueError('Unknown version handle; use a path or SQL instead')
            explicit, object_id, _, kind = self.identity(explicit)
        elif 'path' in evidence:
            object_id, kind = evidence['path'], 'file'
        elif 'record' in evidence:
            object_id, kind = 'record:' + evidence['record'].split('.')[0], 'record'
        elif 'sql' in evidence:
            object_id, kind = None, 'query'
        else:
            raise ValueError('Unsupported PF evidence reference')
        candidates = []
        query_definition = (self.store.read_event(self.content(explicit)[0]['created_by_event'])['query_definition']
                            if explicit and kind == 'query' else None)
        for version, meta in versions:
            if kind != 'other' and meta['layer'] not in ('file_bytes', 'server_public_response', 'registered_text'):
                continue
            if kind == 'query' and 'sql' in evidence:
                event = self.store.read_event(meta['created_by_event'])
                if event['request'].get('candidate_sql') != evidence['sql']:
                    continue
            elif query_definition:
                definition = self.store.read_event(meta['created_by_event'])['query_definition']
                if not definition or [definition[1], *definition[3:]] != [query_definition[1], *query_definition[3:]]:
                    continue
            elif (meta.get('object_id') or version) != object_id:
                continue
            candidates.append(version)
        if not candidates:
            if kind == 'query':
                raise ValueError('No captured query has exactly this SQL text; cite the vN handle shown '
                                 'after your command instead, or use unknown with a reason')
            raise ValueError('No captured evidence exists; use unknown with a reason')
        latest = candidates[0]
        if accept is not None:
            candidates = [c for c in candidates if accept(c)]
            if not candidates:
                raise ValueError('The committed bytes were never captured')
        predicate = reference.get('predicate', {})
        if kind == 'record':
            # Registered texts cite each other directly, exactly as in the Git group
            # (design 3.2); the text is the agent's own declaration, not captured evidence.
            if reference.get('select') or predicate:
                raise ValueError('Registered text supports whole-text equality only')
            wanted_record = evidence.get('record', '')
            for candidate in candidates:
                if explicit and candidate != explicit:
                    continue
                if '.' in wanted_record and json.loads(self.content(candidate)[1])['version'] != wanted_record:
                    continue
                return dict(version_id=candidate, latest_version_id=latest, source_truncated=False,
                            delivered_in=[], basis='registered_text')
            raise ValueError('Unknown registered text revision; use unknown with a reason')
        authored = {c for c in candidates if kind == 'file' and self.authored(c)}
        for occurrence_version, meta in versions:
            if occurrence_version in authored and not (explicit and occurrence_version != explicit):
                # The model wrote these exact bytes itself, so it knows the whole file.
                if predicate.get('type') == 'compare':
                    raise ValueError('compare requires one query view')
                text = self.content(occurrence_version)[1].decode('utf-8')
                content_kind = 'csv' if str(object_id).endswith('.csv') else 'json'
                selector = reference.get('select')
                if selector and not str(object_id).endswith(('.csv', '.json')):
                    raise ValueError('Plain text supports whole-text equality only')
                return dict(version_id=occurrence_version, latest_version_id=latest, source_truncated=False,
                            delivered_in=[], authored_by=meta['created_by_event'],
                            selected_ranges=selected_ranges(text, selector, content_kind))
            if meta['layer'] != 'model_source_occurrences':
                continue
            event = self.store.read_event(meta['created_by_event'])
            if (event['result'].get('send_state') != 'response_received' or
                    event['result'].get('status') != 'succeeded'):
                continue
            occurrences = json.loads(self.content(occurrence_version)[1])
            if meta['created_by_event'] + ':reconstructed' in event['outputs']:
                occurrences += json.loads(self.content(meta['created_by_event'] + ':reconstructed')[1])
            # Newest acquired evidence in the last actual request wins, regardless of
            # message field ordering (old tool messages often recur in the same request).
            for candidate in candidates:
                if explicit and candidate != explicit:
                    continue
                matches = []
                for item in occurrences:
                    source, _, content, source_kind = self.identity(item['version_id'])
                    if source == candidate:
                        matches.append((item, content, source_kind))
                if not matches:
                    if candidate in authored:
                        break  # Repeated old messages cannot supersede a newer full write.
                    continue
                if predicate.get('type') == 'compare' and kind != 'query':
                    raise ValueError('compare requires one query view')
                selectors = ([predicate['left'], predicate['right']] if predicate.get('type') == 'compare'
                             else [reference.get('select')])
                for source_version in dict.fromkeys(item['version_id'] for item, _, _ in matches):
                    group = [(item, content) for item, content, _ in matches if item['version_id'] == source_version]
                    text = group[0][1]
                    content_kind = ('csv' if str(object_id).endswith('.csv') else 'json')
                    if kind in ('file', 'other') and not str(object_id).endswith(('.csv', '.json')) and any(selectors):
                        raise ValueError('Plain text supports whole-text equality only')
                    ranges = [r for select in selectors for r in selected_ranges(text, select, content_kind)]
                    if covered(ranges, [item for item, _ in group]):
                        candidate_meta, _ = self.content(candidate)
                        return dict(version_id=candidate, latest_version_id=latest,
                                    source_truncated=candidate_meta['source_truncated'],
                                    delivered_in=[dict(request_event=meta['created_by_event'], occurrence=item)
                                                  for item, _ in group], selected_ranges=ranges)
                if candidate in authored:
                    break  # The outer loop will use the model's complete authored bytes.
                # Never silently fall back to an older, more fully read version.
                raise ValueError('Selected evidence was not fully delivered to the model; use unknown with a reason')
        if kind == 'query':
            raise ValueError('Evidence has not been delivered to the model: you saw only what your command '
                             'printed, not this raw query result. Cite that output by its 输出 vN handle '
                             '(its queries are traced upstream), or use unknown with a reason')
        if kind == 'file':
            raise ValueError('Evidence has not been delivered to the model: you neither read nor wrote this '
                             'file version. Read it first, cite the 输出 vN handle of the command output '
                             'you saw, or use unknown with a reason')
        raise ValueError('Evidence has not been delivered to the model; use unknown with a reason')

    def authored(self, version):
        """Whether the model's own tool call contained these exact file bytes.

        write_file writes its content argument verbatim; a Bash command counts only when
        the complete decoded file text appears in the command the model wrote (for
        example a quoted heredoc). Script outputs and edits remain unassigned.
        """
        meta, raw = self.content(version)
        if meta['layer'] != 'file_bytes':
            return False
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            return False
        request = self.store.read_event(meta['created_by_event'])['request']
        args = request.get('request') or {}
        if request['kind'] == 'write_file':
            return args.get('content') == text
        return request['kind'] == 'bash' and bool(text) and text in (args.get('command') or '')
