"""PF-only historical queries, rebuilt from immutable capture and declaration records."""
from collections import defaultdict, deque
from contextlib import closing
import difflib
import json
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, ValidationError, model_validator

from .execution_capture import CapturedText, OBJECT_FIELDS
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
        'pf_search': 'Find captured evidence and registered text revisions by exact business object kind/id. Returns an index, not evidence contents. Object IDs come only from public fields or explicit declarations.',
        'pf_dependencies': 'Trace dependencies of a captured file or registered text. Explicit references first; purpose=current always also follows observed execution associations, so upstream SQL of derived files is included; for purpose=historical_only, include_execution=true adds them. With the runner stale check enabled, purpose=current refreshes all expanded SQL views and approved public reads, compares current files and evaluates reference predicates. Historical-only paths are skipped. A passing predicate stops downstream propagation; failures and unknown checks retain paths. Scripts never rerun. Results are advisory. purpose=historical_only retrieves saved history without checks.',
        'pf_dependents': 'Find references to an exact captured version, with paths and reference purposes. Defaults to current active registered texts, traversing old revisions to reach them; current_only=false also returns superseded and retired endpoints. Historical-only paths are labeled, never marked invalid. A file path or version also matches references to any captured observation of the same bytes. include_execution also returns observed execution associations.',
        'pf_read': 'Read immutable evidence (mode=content), list versions of its object (history), or compare baseline to target (diff). Content automatically uses FULL, DELTA or UNCHANGED when the current request contains a complete recoverable baseline and compact output costs fewer tokens. Set full=true with mode=content to request full text. Paths, SQL and bare rN select the latest captured version; rN.M and vN select an exact version. Diff requires the same file, query view, or registered object. SQL comparison preserves types and duplicate rows and separately reports raw order. Content/diff are paged at 30000 characters. A diff does not count as reading the target.',
    }
    return [dict(name=name, description=descriptions[name] +
                 ' Continue a page with only its cursor; pagination uses a fixed snapshot.',
                 parameters=model.model_json_schema()) for name, model in MODELS.items()]


PF_PROMPT = '''

PF tools are available: pf_search finds business objects,
pf_dependencies follows references, pf_dependents finds referrers, and pf_read
reads or compares saved versions. Index results do not deliver evidence contents;
read a version before registering a reference to it. Current-purpose pf_dependencies
automatically checks expanded dependencies when enabled by the runner. Historical-only
references and purpose=historical_only do not trigger checks. All expanded SQL views
rerun on one read-only snapshot; approved SDK reads rerun, scripts never do. A passing
predicate stops changes propagating to referrers. The checks report observations,
not whether your conclusions are correct. Other tools do not run stale checks.
Content reads may return FULL, DELTA or UNCHANGED. DELTA lists [start,end,text]
replacements using zero-based Unicode character offsets in the original baseline;
apply all replacements together. UNCHANGED reuses the named baseline exactly.
Only complete baselines and verified difference chains in this request qualify.
Use pf_read with full=true for full text. Repeating the same target immediately
after a compact read also returns full text once per target and context.
Repeated identical bash, read_file and search_files calls (same tool and arguments)
may likewise return DELTA or UNCHANGED against the previous result of that same
call in this request, named previous_same_call; the header's vN is the full result,
readable with pf_read. Repeating the call immediately after a compact result returns
full text once (for bash this reruns the command).
Execution associations are observed facts, not claims of semantic support.
Use vN handles, workspace-relative paths, exact SQL, or rN.M. A path, SQL or bare
rN in a PF query selects the latest captured version, whereas text registration
still defaults to the most recently delivered evidence. Use mode=history to find
older versions. Keep next_cursor to continue the same snapshot using only cursor.
'''


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
            matches = [v for v in self.nodes if any(o['kind'] == wanted['kind'] and
                       str(o['id']) == wanted['id'] for o in self._objects(v))]
            return self._page(matches, offset, self._describe)
        if 'target' not in values:
            raise ValueError('target is required')
        target = self._target(values['target'])
        if operation == 'pf_read':
            mode = values['mode']
            if (mode == 'diff') != ('baseline' in values):
                raise ValueError('baseline is required only for diff')
            if values['full'] and mode != 'content':
                raise ValueError('full is supported only for content reads')
            if mode == 'history':
                key = self._key(target)
                return self._page([v for v in self.nodes if self._key(v) == key], offset, self._describe)
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
                from .execution_capture import CURRENT_EVENT
                from .pf_stale import StaleCheck
                event = CURRENT_EVENT.get()
                own_event = event is None
                if own_event:
                    event = self.store.begin_event('pf_dependencies', values)
                token = CURRENT_EVENT.set(event)
                try:
                    rows = StaleCheck(self).run(rows)
                    self.check_state = 'pf_check:' + event
                    self.store.save_state(self.check_state, rows)
                    if own_event:
                        self.store.complete(event)
                finally:
                    CURRENT_EVENT.reset(token)
        return self._page(rows, offset, self._describe_edge, root=self._describe(target),
                          stale_check='performed' if checking else 'not_performed')

    def _index(self, cutoff):
        # ponytail: rebuild O(versions + edges) per query/page; persist an index if
        # measured history-query latency becomes material in longer experiments.
        self.events, self.nodes, self.records = {}, {}, {}
        self.outgoing, self.incoming = defaultdict(list), defaultdict(list)
        self.latest, self.reads = {}, defaultdict(list)
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
                self.events[row['event_id']] = dict(request=json.loads(row['request']),
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
                        reading = dict(at=event['request']['started_at'], range=item['source_range'],
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
        row = self.nodes[version]
        meta, event = row['meta'], self.events[row['event_id']]
        result = dict(version=self.resolver.handle(version), layer=meta['layer'], objects=self._objects(version),
            acquired_at=meta['acquired_at'], content_time=meta['content_time'],
            source_truncated=meta['source_truncated'], extent=meta['extent'],
            operation=event['request']['kind'], status=event['result'].get('status', 'result_unknown'),
            capture_gaps=event['result'].get('capture_gaps', []),
            relation_origin='agent_declaration' if version in self.records else 'automatic_capture')
        if meta['layer'] == 'file_bytes':
            result['path'] = meta['object_id']
        if meta['layer'] == 'server_public_response' and event['query']:
            result.update(sql=event['query'][5], bound_parameters=event['query'][6],
                          result_coverage=event['result'].get('result_coverage', 'unknown'))
        if 'classification' in event['result']:
            result['classification'] = event['result']['classification']
        request = event['request'].get('request') or {}
        if event['request']['kind'] in ('bash', 'cli_python', 'registered_script_execution'):
            result['execution'] = {k: request[k] for k in ('command', 'source', 'name') if k in request}
        if version in self.records:
            record = self.records[version]
            result.update(record=record['version'], text_status=record['status'], reason=record['reason'],
                          applies_at=record['applies_at'], current_revision=self.latest[self._key(version)] == version)
        if row['previous_version'] in self.nodes:
            result['previous_version'] = self.resolver.handle(row['previous_version'])
        reads = self.reads[version]
        result['model_reads'] = dict(count=len(reads), last=reads[-1] if reads else None)
        return result

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
        result = {k: v for k, v in edge.items() if k not in ('source', 'target', 'reference', 'observed_as')}
        result.update(source=self._describe(edge['source']),
                      target=self._describe(edge['target']) if edge['target'] else None,
                      path=[self.resolver.handle(v) for v in item['path']],
                      historical_only=item['historical_only'], unexpanded=item['stop'],
                      traversal_only=item['traversal_only'])
        if 'reference' in edge:
            result.update(edge['reference'])
        if 'check' in item:
            check = dict(item['check'])
            check['current_version'] = self.resolver.handle(check['current_version']) if check['current_version'] else None
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
        return encoded(dict(items=[describe(v) for v in rows[offset:end]],
            next_cursor=self._cursor(end) if end < len(rows) else None,
            remaining=len(rows) - end, truncated_reason='page_limit' if end < len(rows) else None,
            **extra)).decode()

    def _read(self, target, offset):
        meta, raw = self.resolver.content(target)
        try:
            content = raw.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ValueError('This captured version is not UTF-8 text') from exc
        header = dict(mode=self.values['mode'])
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
                header['row_comparison'] = dict(status='unknown', reason='comparison_unavailable')
                if all(c.get('status') == 'available' for c in comparison):
                    contents = [self.resolver.content(self.nodes[v]['event_id'] + ':comparison')[1] for v in (baseline, target)]
                    header['row_comparison'] = dict(status='compared', equal=contents[0] == contents[1],
                        scope='returned_subset' if meta['source_truncated'] or old_meta['source_truncated'] else 'complete_for_query',
                        order_compared=False)
            header.update(baseline=self._describe(baseline), target=self._describe(target))
            # splitlines with terminators preserves final-newline differences.
            lines = difflib.unified_diff(before.splitlines(keepends=True), content.splitlines(keepends=True),
                                        fromfile=header['baseline']['version'], tofile=header['target']['version'])
            content = ''.join(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n' for line in lines)
        else:
            header['target'] = self._describe(target)
        end = min(len(content), offset + 30000)
        header.update(range=[offset, end], total_chars=len(content),
                      next_cursor=self._cursor(end) if end < len(content) else None,
                      truncated_reason='character_limit' if end < len(content) else None)
        prefix = encoded(header).decode() + '\n'
        from .pf_read import capture_read
        return capture_read(self, target, prefix + content[offset:end],
                            offset, end, len(content))
