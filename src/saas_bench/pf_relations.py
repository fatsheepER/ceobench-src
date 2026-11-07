"""Read-only navigation of saved source declarations and observed capture links."""
from collections import Counter, defaultdict, deque
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath
import re
import subprocess
import uuid

from .public_errors import PublicPFError, PublicPFPermissionError
from . import evidence_handles, pf_render
from .pf_deadline import check
from .public_evidence import PublicEvidence
from .registration_evidence import git_reference, native_repository
from .registration_schema import cite_text
from .relation_registry import encode
from .sql_evidence import PUBLIC_LAYERS, now
from .workspace_io import open_file
from .workspace_layout import evidence_namespace


FORMAT = 'pf.relations.v1'
DETAIL_CHARS = 18000
TEXT_FORMAT = 'ceobench.text-register.v1'
RELATION_FORMAT = 'ceobench.relations.v1'
ROLES = ('ceo', 'growth', 'ops_finance')
EXACT_TEXT = re.compile(r'(?:(ceo|growth|ops_finance):)?(r[1-9][0-9]*\.[1-9][0-9]*)')
GIT_REF = re.compile(r'(?:(ceo|growth|ops_finance|shared):)?(.+)@([0-9a-fA-F]{40})')
GIT_INPUT = re.compile(r'(?:(ceo|growth|ops_finance|shared):)?(.+)@([0-9a-fA-F]{1,40})')
EXECUTION_KINDS = frozenset(('bash', 'cli_python', 'sql_query', 'public_http', 'read_file',
                             'write_file', 'edit_file', 'registered_script_execution', 'dashboard_generation'))


def _hash(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def _key(endpoint):
    ref = endpoint['ref']
    if 'event_ref' in ref:
        return ('event', ref['event_ref'])
    return ('operation', ref['operation_ref']) if 'operation_ref' in ref else ('cite', ref.get('cite'))


def _label(endpoint):
    if endpoint is None:
        return '(empty source set)'
    ref = endpoint['ref']
    if 'missing' in ref:
        return 'missing: ' + ref['missing']['description']
    return ref.get('cite') or ref.get('operation_ref') or ref.get('event_ref') or 'unknown'


def _path_endpoint(endpoint):
    return None if endpoint is None else dict(ref=deepcopy(endpoint['ref']),
                                             storage_role=endpoint.get('storage_role'),
                                             workspace_namespace=endpoint.get('workspace_namespace'))


class RelationQuery:
    def __init__(self, registry):
        self.registry, self.store, self.resolver = registry, registry.store, registry.resolver
        self.role = registry.role
        self.public = PublicEvidence(self.store, registry)

    def answer(self, operation, values):
        if operation not in ('pf_dependencies', 'pf_dependents'):
            raise PublicPFError('Unknown relation query operation')
        if values.get('cursor'):
            saved = (self.store.load_state('pf_cursors') or {}).get(values['cursor'])
            if not saved or saved.get('operation') != operation or not saved.get('relation_snapshot'):
                raise PublicPFError('Unknown relation cursor; start a new relation query')
            name = saved['relation_snapshot']
            snapshot = self.store.load_state(name)
            if not snapshot or snapshot.get('format') != FORMAT:
                raise PublicPFError('Relation snapshot is unavailable; start a new relation query')
            offset, detail, selection = saved['offset'], saved['detail'], saved['selection']
            self._authorize(snapshot)
        else:
            snapshot = self._snapshot(operation, values)
            name = 'pf_relations:' + uuid.uuid4().hex
            self.store.save_state(name, snapshot)
            offset, detail, selection = 0, values.get('detail', False), [0, len(snapshot['items'])]
        common = {k: deepcopy(snapshot[k]) for k in ('format', 'scope', 'cutoff', 'root', 'counts')}
        common['cutoff'] = self.public.cutoff(snapshot['cutoff'])
        common.update(total_in_scope=len(snapshot['items']), detail=detail)
        common['root'] = self._compact_endpoint(snapshot['root'])
        start, stop = selection
        if not 0 <= start <= stop <= len(snapshot['items']):
            raise PublicPFError('Invalid relation cursor selection')
        if detail:
            content = json.dumps(dict(scope=snapshot['scope'], cutoff=self.public.cutoff(snapshot['cutoff']),
                                      root=self.public.endpoint(snapshot['root']),
                                      items=[self.public.relation(item) for item in snapshot['items'][start:stop]]),
                                 ensure_ascii=True, indent=2, sort_keys=True)
            if not 0 <= offset <= len(content):
                raise PublicPFError('Invalid relation detail cursor offset')
            end = min(len(content), offset + DETAIL_CHARS)
            cursor = self._cursor(operation, name, end, True, selection) if end < len(content) else None
            return dict(common, items=[], returned=stop-start, detail_text=content[offset:end],
                        range=[offset, end], total_chars=len(content), next_cursor=cursor,
                        detail_cursor=None, has_more=cursor is not None)
        if not start <= offset <= stop:
            raise PublicPFError('Invalid relation cursor offset')
        end = min(stop, offset + snapshot['limit'])
        while end > offset + 1 and len(json.dumps([self._compact(item) for item in snapshot['items'][offset:end]], ensure_ascii=True)) > 16000:
            end -= 1
        cursor = self._cursor(operation, name, end, False, selection) if end < stop else None
        detail_cursor = self._cursor(operation, name, 0, True, [offset, end])
        items = [self._compact(item) for item in snapshot['items'][offset:end]]
        return dict(common, items=items, returned=len(items), next_cursor=cursor,
                    detail_cursor=detail_cursor, has_more=cursor is not None)

    def _cursor(self, operation, snapshot, offset, detail, selection):
        cursors = self.store.load_state('pf_cursors') or {}
        value = dict(operation=operation, relation_snapshot=snapshot, offset=offset,
                     detail=detail, selection=selection)
        for name, existing in cursors.items():
            if existing == value:
                return name
        name = 'c' + str(max((int(k[1:]) for k in cursors if re.fullmatch(r'c[1-9][0-9]*', k)), default=0) + 1)
        cursors[name] = value
        self.store.save_state('pf_cursors', cursors)
        return name

    def _roles(self):
        if not self.registry.identity:
            return [self.role]
        from .role_policy import READABLE_ROLES
        return list(READABLE_ROLES[self.role])

    def _registry(self, role, filename):
        path = self.registry.authority_path(role, filename)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return {}, None
        state = json.loads(raw)
        expected = RELATION_FORMAT if filename == 'relations.json' else TEXT_FORMAT
        if state.get('format') != expected or not isinstance(state.get('records'), dict):
            raise ValueError('Invalid private relation or text registry')
        for history in state['records'].values():
            if not isinstance(history, list) or not history:
                raise ValueError('Invalid private registry revision history')
            for record in history:
                check()
                author = record.get('author_role') if filename == 'relations.json' else record.get('author')
                if author != role:
                    raise ValueError('Private registry author differs from its workspace owner')
                if self.registry.identity:
                    identity = self.registry.identity
                    if record.get('role') != role or any(record.get(k) != getattr(identity, k) for k in ('run_id', 'world_id')):
                        raise ValueError('Private registry identity differs from its workspace owner')
                    if not all(isinstance(record.get(k), str) and record[k] for k in ('agent_id', 'session_id')):
                        raise ValueError('Private registry identity is incomplete')
        return state['records'], hashlib.sha256(raw).hexdigest()

    def _snapshot(self, operation, values):
        if not values.get('target'):
            raise PublicPFError('target is required')
        self.registries, registry_cutoff = {}, {}
        self.validations = {}
        for role in self._roles():
            for filename in ('relations.json', 'registrations.json'):
                histories, digest = self._registry(role, filename)
                self.registries[role, filename] = histories
                registry_cutoff.setdefault(role, {})[filename] = dict(sha256=digest, revisions=[
                    record.get('relation_revision', role + ':' + record.get('version', ''))
                    for history in histories.values() for record in history])
        with closing(self.store.connect()) as conn:
            conn.execute('BEGIN')
            self.conn = conn
            cutoff = {table: conn.execute('SELECT coalesce(max(rowid),0) FROM ' + table).fetchone()[0]
                      for table in ('requests', 'versions', 'results', 'deliveries', 'client_calls', 'client_interruptions')}
            cutoff.update(captured_at=now(), registries=registry_cutoff)
            self._captures(conn)
            root = self._root(values['target'])
            all_revisions = values.get('all_revisions', False) or (operation == 'pf_dependents' and not values.get('current_only', True))
            include_candidates = bool(values.get('include_candidates') or values.get('include_execution'))
            if values.get('purpose') == 'historical_only':
                include_candidates = False
            rows = self._declarations(root, operation, all_revisions)
            if include_candidates:
                rows += self._observations()
            depth = values.get('depth', 1)
            items = self._walk(rows, root, operation, depth)
            self._validate_endpoint(root)
            for item in items:
                self._validate_revision(item)
                if item.get('root_event_ref'):
                    self._validate_endpoint(self._execution_endpoint(item['root_event_ref']))
                for side in ('source', 'target'):
                    if item[side] is not None:
                        self._validate_endpoint(item[side])
            counts = Counter(item['assertion'] for item in items if item['source'] is not None)
            revisions = {item['relation_revision']: item['assertion'] for item in items if item.get('relation_revision')}
            counts = dict(endpoint_rows=sum(item['source'] is not None for item in items),
                          declaration_revisions=len(revisions), assertion_revisions=dict(Counter(revisions.values())),
                          endpoint_rows_by_assertion=dict(counts),
                          execution_observations=sum(item['assertion'] == 'execution_observation' for item in items),
                          empty_source_sets=sum(item['source'] is None for item in items))
        del self.conn
        scope = dict(operation=operation, direction='source_to_target', reader_role=self.role,
                     roles=self._roles(), depth=depth, all_revisions=all_revisions,
                     include_candidates=include_candidates, purpose=values.get('purpose', 'current'),
                     current_only=values.get('current_only', True),
                     revisions=dict(relation_batch='all_saved' if all_revisions else 'latest_active',
                         legacy_text_reference='all_saved' if all_revisions else
                         'exact_reachable_saved_revisions' if operation == 'pf_dependencies' else 'latest_active'),
                     current_means='latest saved revision; no validity judgment', time='all_saved_days')
        return dict(format=FORMAT, scope=scope, cutoff=cutoff, root=root, items=items, counts=counts,
                    validations=list(self.validations.values()), limit=min(20, values.get('limit', 20)))

    def _captures(self, conn):
        self.captures, self.bindings, self.events = {}, {}, {}
        for row in conn.execute('SELECT r.event_id,r.request,s.record,q.definition FROM requests r '
                                'LEFT JOIN results s USING(event_id) LEFT JOIN queries q ON r.query_id=q.id ORDER BY r.rowid'):
            check()
            self.events[row['event_id']] = dict(request=json.loads(row['request']),
                result=json.loads(row['record']) if row['record'] else {},
                query=json.loads(row['definition']) if row['definition'] else None)
        for row in conn.execute('SELECT rowid,* FROM versions ORDER BY rowid'):
            check()
            meta = json.loads(row['metadata'])
            if meta['layer'] not in PUBLIC_LAYERS and meta['layer'] != 'agent_declaration':
                continue
            try:
                self.store.authorize_version(conn, row['event_id'], meta,
                                             public=meta['layer'] in PUBLIC_LAYERS, graph=meta['layer'] == 'agent_declaration')
            except PermissionError:
                continue
            if meta['layer'] == 'agent_declaration':
                try:
                    _, raw = self.store.get_content(row['version_id'], connection=conn, graph=True)
                    binding = json.loads(raw)
                except (KeyError, ValueError):
                    continue
                self.bindings[binding['version_id']] = binding.get('references', [])
            else:
                event = self.events[row['event_id']]
                retrieval = event['request']['kind'].startswith('pf_') or event['result'].get('pf_retrieval')
                if retrieval and meta['layer'] in ('stdout', 'stderr', 'tool_return'):
                    continue
                self.captures[row['version_id']] = dict(row, meta=meta)
        self.text_captures = {}
        for version, row in self.captures.items():
            if row['meta']['layer'] == 'registered_text':
                try:
                    _, raw = self.store.public_content(version, connection=conn)
                    record = json.loads(raw)
                except (KeyError, ValueError):
                    continue
                role = row['meta'].get('owner_role', record.get('author', self.role))
                self.text_captures[role, record['version']] = version

    def _qualify(self, cite, role=None):
        if not self.registry.identity:
            return cite if GIT_INPUT.fullmatch(cite) else cite.removeprefix(self.role + ':')
        if cite.split(':', 1)[0] in (*ROLES, 'shared'):
            return cite
        return (role or self.role) + ':' + cite

    def _saved_endpoint(self, endpoint):
        result = deepcopy(endpoint)
        if not self.registry.identity and result['ref'].get('cite'):
            result['ref']['cite'] = self._qualify(result['ref']['cite'])
        return result

    def _root(self, target):
        if target.get('operation_ref'):
            ref = dict(operation_ref=target['operation_ref'])
            for histories in self.registries.values():
                for history in histories.values():
                    for record in history:
                        for endpoint in [record.get('target'), *(s['ref'] for s in record.get('sources', []))]:
                            if endpoint and endpoint['ref'] == ref:
                                return deepcopy(endpoint)
            return dict(kind='operation', ref=ref, source_state='unknown', readable=False,
                        producer_role=None, source_role=None, storage_role=None,
                        availability='not_saved_in_relation', reason='No saved relation snapshot for this operation')
        value = target.get('record') or target.get('version') or target.get('path')
        if value and evidence_handles.EXACT.fullmatch(value) and ':event' in value:
            event = evidence_handles.resolve_ref(self.store, 'event', value)
            self.store.readable_event(self.conn, event)
            if event not in self.events:
                raise PublicPFError('Captured execution is unavailable')
            return self._execution_endpoint(event)
        if value and not self.registry.identity:
            value = self._qualify(value)
        if value and evidence_handles.RECORD.fullmatch(value):
            owner, name = value.split(':', 1) if ':' in value else (self.role, value)
            self._authorize_role(owner)
            history = self.registries.get((owner, 'registrations.json'), {}).get(name.split('.')[0], [])
            record = next((r for r in history if r['version'] == name), None) if '.' in name else history[-1] if history else None
            if record is None:
                raise PublicPFError('Unknown registered text revision')
            return self._text_endpoint(owner, record)
        if value and (match := GIT_INPUT.fullmatch(value)):
            owner, path, commit = match.groups()
            owner = owner or getattr(self.registry, 'workspace_namespace', None) or self.role
            self._public_path(owner, path)
            workspace = self.registry.repository(owner)
            native_repository(workspace)
            if len(commit) != 40:
                _, commit = git_reference(workspace, dict(path=path, commit=commit), run=self.registry.git_run)
            cite = owner + ':' + path + '@' + commit.lower()
            for histories in self.registries.values():
                for history in histories.values():
                    for record in history:
                        for endpoint in [record.get('target'), *(s['ref'] for s in record.get('sources', []))]:
                            if endpoint and endpoint['ref'].get('cite') == cite:
                                return deepcopy(endpoint)
            raw = self._git_bytes(owner, path, commit)
            return dict(kind='file', ref=dict(cite=cite), source_state='available', readable=True,
                        producer_role=None, source_role=None if owner == 'shared' else owner,
                        storage_role=None if owner == 'shared' else owner,
                        **({'workspace_namespace': owner} if owner == 'shared' else {}),
                        sha256=hashlib.sha256(raw).hexdigest())
        if 'sql' in target:
            matches = [version for version, row in self.captures.items()
                       if row['meta']['layer'] == 'server_public_response' and self.events[row['event_id']]['query']
                       and self.events[row['event_id']]['query'][5] == target['sql']]
        elif value and evidence_handles.RAW_VERSION.fullmatch(value):
            matches = [value] if value in self.captures else []
        else:
            matches = [v for v in self.resolver.lookup(value) or [] if v in self.captures]
            if not matches:
                name = self._qualify(value) if self.registry.identity else value
                matches = [v for v in evidence_handles.index(self.store).latest(name) or [] if v in self.captures]
            if not matches:
                owner, path = (value.split(':', 1) if ':' in value and value.split(':', 1)[0] in (*ROLES, 'shared')
                               else (getattr(self.registry, 'workspace_namespace', None) or self.role, value))
                self._public_path(owner, path)
                matches = [v for v, row in self.captures.items() if row['meta']['layer'] == 'file_bytes'
                           and row['meta'].get('object_id') == path and evidence_namespace(row['meta'], self.role) == owner]
        if not matches:
            from .pf_paths import missing
            raise PublicPFError(missing(self.registry, value, prefix='No saved exact endpoint matches') +
                                '; business operations use ROLE:number')
        return self._capture_endpoint(matches[-1])

    def _text_endpoint(self, owner, record):
        return dict(kind='text', ref=dict(cite=self._qualify(record['version'], owner)), source_state='available', readable=True,
                    producer_role=owner, source_role=owner, storage_role=owner, sha256=_hash(record),
                    text=record.get('text'), record=deepcopy(record))

    def _capture_endpoint(self, version):
        row = self.captures[version]
        meta = row['meta']
        if meta['layer'] == 'registered_text':
            for (role, name), captured in self.text_captures.items():
                if captured == version:
                    history = self.registries.get((role, 'registrations.json'), {}).get(name.split('.')[0], [])
                    record = next((r for r in history if r['version'] == name), None)
                    if record:
                        return self._text_endpoint(role, record)
        role = meta.get('owner_role', self.role)
        namespace = meta.get('workspace_namespace') if meta['layer'] in ('file_bytes', 'file_text') else None
        return dict(kind='file' if meta['layer'] in ('file_bytes', 'file_text') else 'output',
                    ref=dict(cite=self._qualify(self.resolver.handle(version), namespace or role)),
                    source_state='available', readable=True,
                    producer_role=meta.get('producer_role') if namespace else role,
                    source_role=meta.get('producer_role') if namespace else role,
                    storage_role=None if namespace else role,
                    **({key: meta[key] for key in ('workspace_namespace', 'collector_role') if key in meta} if namespace else {}),
                    sha256=row['content_hash'], content_ref=version, event_ref=row['event_id'],
                    capture=dict(layer=meta['layer'], source_truncated=meta.get('truncated'),
                                 reading_scope='unknown'))

    def _declarations(self, root, operation, all_revisions):
        rows = []
        for (role, filename), histories in self.registries.items():
            for history in histories.values():
                legacy_forward = filename == 'registrations.json' and operation == 'pf_dependencies'
                records = history if all_revisions or legacy_forward else [history[-1]]
                for record in records:
                    if not all_revisions and record['status'] != 'active' and not legacy_forward:
                        continue
                    current = record is history[-1] and record['status'] == 'active'
                    if filename == 'relations.json':
                        target = self._saved_endpoint(record['target'])
                        sources = [(self._saved_endpoint(source['ref']), source.get('note'), {}) for source in record['sources']]
                        assertion, revision = record['assertion'], record['relation_revision']
                        relation, reason = record['relation'], record['reason']
                    else:
                        target = self._text_endpoint(role, record)
                        sources = [(self._legacy_source(role, record, i, ref), ref.get('note'), deepcopy(ref))
                                   for i, ref in enumerate(record.get('references', []))]
                        assertion, revision = 'author_declared', role + ':' + record['version']
                        relation, reason = role + ':' + record['id'], record.get('reason')
                    for index, (source, note, legacy) in enumerate(sources or [(None, None, {})]):
                        rows.append(dict(source=source, target=target, assertion=assertion,
                            origin='legacy_text_reference' if filename == 'registrations.json' else 'relation_batch',
                            relation=relation, relation_revision=revision, revision=record['revision'],
                            author_role=role, record_status=record['status'], current=current,
                            source_index=index if source is not None else None, reason=reason, note=note,
                            sim_day=record.get('sim_day'), purpose=legacy.get('purpose', 'current'),
                            reference=legacy or None, history=dict(role=role, path=filename,
                                id=record['id'], revision=record['revision'], sha256=_hash(record))))
        return rows

    def _legacy_source(self, role, record, index, reference):
        evidence = reference['evidence']
        if 'unknown' in evidence:
            return dict(kind='missing', ref=dict(missing=dict(description=evidence['unknown'], reason=evidence['unknown'])),
                        source_state='missing', readable=False, producer_role=None, source_role=None, storage_role=None)
        version = self.text_captures.get((role, record['version']))
        bindings = self.bindings.get(version, [])
        binding = bindings[index] if index < len(bindings) else {}
        bound = binding.get('version_id')
        if evidence.get('commit') or binding.get('git_commit'):
            owner = evidence.get('owner', role)
            original = reference.get('snapshot_source') or {}
            commit = binding.get('git_commit') or evidence.get('commit', '')
            if re.fullmatch('[0-9a-fA-F]{1,40}', commit):
                if len(commit) != 40:
                    try:
                        self._public_path(owner, evidence['path'])
                        workspace = self.registry.repository(owner)
                        native_repository(workspace)
                        _, commit = git_reference(workspace, dict(path=evidence['path'], commit=commit), run=self.registry.git_run)
                    except ValueError:
                        commit = None
                if commit:
                    return dict(kind='file', ref=dict(cite=owner + ':' + evidence['path'] + '@' + commit),
                                source_state='available', readable=True, producer_role=None if owner == 'shared' else original.get('owner'),
                                source_role=None if owner == 'shared' else original.get('owner', owner),
                                storage_role=None if owner == 'shared' else owner,
                                **({'workspace_namespace': owner} if owner == 'shared' else {}),
                                snapshot_source=deepcopy(original) or None, binding=deepcopy(binding),
                                original_reference=deepcopy(evidence))
            return dict(kind='unknown', ref=dict(cite=cite_text(evidence)), source_state='unknown', readable=False,
                        producer_role=None, source_role=None if owner == 'shared' else original.get('owner', owner),
                        storage_role=None if owner == 'shared' else owner,
                        **({'workspace_namespace': owner} if owner == 'shared' else {}),
                        availability='exact_binding_unavailable',
                        reason='No saved exact Git commit; a weekly label was not rebound', binding=deepcopy(binding))
        if bound in self.captures and binding.get('status') == 'resolved' and binding.get('git_content_matches') is not False:
            endpoint = self._capture_endpoint(bound)
            endpoint['binding'] = deepcopy(binding)
            return endpoint
        cite = cite_text(evidence)
        if cite and EXACT_TEXT.fullmatch(cite):
            owner, name = cite.split(':', 1) if ':' in cite else (role, cite)
            history = self.registries.get((owner, 'registrations.json'), {}).get(name.split('.')[0], [])
            target = next((r for r in history if r['version'] == name), None)
            if target:
                return self._text_endpoint(owner, target)
        if bound and binding.get('status') == 'resolved' and binding.get('git_content_matches') is not False:
            exact = evidence.get('version')
            return dict(kind='output', ref=dict(cite=self._qualify(exact, role) if exact else bound),
                        source_state='available', readable=False, producer_role=None,
                        source_role=None if evidence.get('owner') == 'shared' else evidence.get('owner', role),
                        storage_role=None, content_ref=bound,
                        **({'workspace_namespace': 'shared'} if evidence.get('owner') == 'shared' else {}),
                        availability='saved_content_unavailable', reason='Exact saved binding remains; captured bytes are unavailable',
                        binding=deepcopy(binding))
        return dict(kind='unknown', ref=dict(cite=cite), source_state='unknown', readable=False,
                    producer_role=None, source_role=None if evidence.get('owner') == 'shared' else evidence.get('owner', role),
                    storage_role=None, **({'workspace_namespace': 'shared'} if evidence.get('owner') == 'shared' else {}),
                    availability='exact_binding_unavailable', reason=binding.get('reason', 'No exact saved binding; reference was not rebound'),
                    binding=deepcopy(binding))

    def _execution_root(self, event):
        root, seen = event, {event}
        parent = self.events[event]['request'].get('parent_event_id')
        while parent in self.events and parent not in seen:
            check()
            seen.add(parent)
            if self.events[parent]['request']['kind'] not in EXECUTION_KINDS:
                break
            try:
                self.store.readable_event(self.conn, parent)
            except PermissionError:
                break
            root = parent
            parent = self.events[parent]['request'].get('parent_event_id')
        return root

    def _execution_endpoint(self, event):
        saved = self.events[event]
        request = saved['request']
        role = request.get('role', request.get('author', self.role))
        return dict(kind='execution', ref=dict(event_ref=event), event_ref=event,
                    source_state='available', readable=True, availability='available',
                    producer_role=role, source_role=role, storage_role=None,
                    event_kind=request['kind'], parent_event_ref=request.get('parent_event_id'),
                    request=deepcopy(request), result=deepcopy(saved['result']))

    @staticmethod
    def _member_role(version, meta):
        layer = meta['layer']
        if layer == 'file_bytes':
            if version.endswith('_after'):
                return ('observed_file' if meta.get('workspace_namespace') and not meta.get('producer_role')
                        else 'written_file')
            if version.endswith('_read'):
                return 'read_file'
            return 'observed_file'
        return dict(file_text='representation', query_model_projection='representation',
                    server_public_response='response', tool_return='tool_return',
                    stdout='stdout', stderr='stderr', executed_code='executed_code',
                    dashboard='dashboard').get(layer, layer)

    def _observations(self):
        rows = []
        for target, row in self.captures.items():
            check()
            meta = row['meta']
            event = row['event_id']
            if self.events[event]['request']['kind'] in EXECUTION_KINDS:
                root_event = self._execution_root(event)
                rows.append(dict(source=self._capture_endpoint(target), target=self._execution_endpoint(event),
                                 assertion='execution_observation', origin='automatic_capture', relation=None,
                                 relation_revision=None, kind='execution_membership', basis='execution_membership',
                                 author_role=None, current=None, purpose='current', event_ref=event,
                                 content_ref=target, member_role=self._member_role(target, meta),
                                 root_event_ref=root_event, navigation='pf candidates --history --role all --event ' + root_event))
            links = []
            if meta.get('derived_from'):
                links.append((meta['derived_from'], 'derived_from', {}))
            links += [(segment['version_id'], 'returned_range', dict(source_range=segment.get('source_range'),
                       output_range=segment.get('request_range'))) for segment in meta.get('segments', [])]
            links += [(segment['version_id'], 'public_field_range', dict(field=field['pointer'],
                       source_range=segment.get('source_range'))) for field in meta.get('public_fields', [])
                      for segment in field.get('origins', [])]
            for index, (source, kind, observation) in enumerate(links):
                if source not in self.captures:
                    continue
                rows.append(dict(source=self._capture_endpoint(source), target=self._capture_endpoint(target),
                                 assertion='execution_observation', origin='automatic_capture', relation=None,
                                 relation_revision=None, kind=kind, author_role=None, current=None,
                                 source_index=index, purpose='current', observation=observation,
                                 event_ref=event, source_event_ref=self.captures[source]['event_id'],
                                 source_content_ref=source, target_content_ref=target))
        return rows

    def _walk(self, rows, root, operation, depth):
        reverse = operation == 'pf_dependents'
        adjacency = defaultdict(list)
        for index, row in enumerate(rows):
            check()
            membership = row.get('basis') == 'execution_membership'
            endpoint = row['source'] if reverse or membership else row['target']
            if endpoint is not None and 'missing' not in endpoint['ref'] and endpoint['source_state'] != 'unknown':
                adjacency[_key(endpoint)].append((index, row))
        root_key = _key(root)
        queue = deque([(root_key, 0, [_path_endpoint(root)], [root_key], False)])
        scheduled, seen_rows, result = {root_key}, set(), []
        while queue:
            check()
            key, distance, path, path_keys, historical = queue.popleft()
            for index, row in adjacency.get(key, []):
                if index in seen_rows:
                    continue
                seen_rows.add(index)
                membership = row.get('basis') == 'execution_membership'
                endpoint = row['target'] if reverse or membership else row['source']
                next_key = _key(endpoint) if endpoint is not None and 'missing' not in endpoint['ref'] else None
                past = historical or row.get('purpose') == 'historical_only'
                cycle = next_key in path_keys if next_key else False
                already = next_key in scheduled if next_key else False
                stop = ('empty_source_set' if endpoint is None else 'missing' if endpoint.get('source_state') in ('missing', 'unknown') else
                        'execution_membership' if membership else 'cycle' if cycle else 'already_expanded' if already else
                        'depth_limit' if distance + 1 >= depth and adjacency.get(next_key) else None)
                result.append(dict(deepcopy(row), depth=distance+1, path=path + [_path_endpoint(endpoint)],
                                   historical_only=past, unexpanded=stop))
                if next_key and not stop and distance+1 < depth:
                    scheduled.add(next_key)
                    queue.append((next_key, distance+1, path + [_path_endpoint(endpoint)], path_keys + [next_key], past))
        return result

    def _validate_revision(self, item):
        history = item.get('history')
        if history:
            self.validations['revision:' + item['relation_revision']] = dict(kind='revision', **history)

    def _authorize_role(self, role):
        if self.registry.identity:
            from .role_policy import READABLE_ROLES
            if role not in READABLE_ROLES[self.role]:
                raise PublicPFPermissionError('Evidence is inaccessible')

    def _public_path(self, owner, path):
        self.registry.repository(owner)
        from .public_records import private_framework_path
        parts = PurePosixPath(path).parts
        if private_framework_path(path) or not parts or PurePosixPath(path).is_absolute() or '..' in parts or '.git' in parts or parts[0] == 'sessions' or '\x00' in path:
            raise PublicPFPermissionError('Only public business files may be queried')

    def _git_bytes(self, owner, path, commit):
        check()
        self._public_path(owner, path)
        workspace = self.registry.repository(owner)
        native_repository(workspace)
        result = self.registry.git_run(['git', '--no-replace-objects', '-C', str(workspace), 'show', commit + ':' + path],
                                       capture_output=True, timeout=10)
        if result.returncode:
            raise PublicPFError('Saved Git snapshot is unavailable')
        return result.stdout

    def _validate_endpoint(self, endpoint):
        check()
        for role in (endpoint.get('source_role'), endpoint.get('producer_role'), endpoint.get('storage_role')):
            if role:
                self._authorize_role(role)
        if endpoint['source_state'] in ('missing', 'unknown'):
            return
        ref = endpoint['ref']
        if 'event_ref' in ref:
            event = ref['event_ref']
            self.store.readable_event(self.conn, event)
            self.validations['event:' + event] = dict(kind='event', event=event,
                request_sha256=_hash(self.events[event]['request']))
            return
        if 'operation_ref' in ref:
            if endpoint.get('operation') and endpoint.get('sha256') != _hash(endpoint['operation']):
                raise ValueError('Saved operation snapshot checksum differs')
            return
        cite = ref['cite']
        if match := EXACT_TEXT.fullmatch(cite):
            owner, name = match.groups()
            owner = owner or self.role
            history = self.registries.get((owner, 'registrations.json'), {}).get(name.split('.')[0], [])
            record = next((r for r in history if r['version'] == name), None)
            if record is None or (endpoint.get('sha256') and endpoint['sha256'] != _hash(record)):
                endpoint.update(readable=False, availability='saved_content_unavailable', reason='Saved text revision is unavailable or changed')
                return
            self.validations['text:' + cite] = dict(kind='revision', role=owner, path='registrations.json',
                id=record['id'], revision=record['revision'], sha256=_hash(record))
        elif match := GIT_REF.fullmatch(cite):
            owner, path, commit = match.groups()
            owner = owner or getattr(self.registry, 'workspace_namespace', None) or self.role
            try:
                raw = self._git_bytes(owner, path, commit)
                digest = hashlib.sha256(raw).hexdigest()
                if endpoint.get('sha256') and endpoint['sha256'] != digest:
                    raise ValueError('Saved Git snapshot checksum differs')
            except (ValueError, subprocess.TimeoutExpired):
                endpoint.update(readable=False, availability='saved_content_unavailable', reason='Saved Git snapshot is unavailable or changed')
                return
            endpoint['sha256'] = digest
            self.validations['git:' + cite] = dict(kind='git', role=owner, path=path, commit=commit, sha256=digest)
        else:
            versions = [endpoint['content_ref']] if endpoint.get('content_ref') else self.resolver.lookup(cite) or []
            versions = [v for v in versions if v in self.captures]
            available = []
            for version in versions:
                row = self.captures[version]
                if endpoint.get('sha256') and row['content_hash'] != endpoint['sha256']:
                    continue
                try:
                    _, raw = self.store.public_content(version, connection=self.conn)
                except (KeyError, ValueError):
                    continue
                available.append(version)
                self.validations['capture:' + version] = dict(kind='capture', version=version, sha256=hashlib.sha256(raw).hexdigest())
            if not available:
                endpoint.update(readable=False, availability='saved_content_unavailable', reason='Saved capture bytes are unavailable or changed')
            else:
                endpoint['content_refs'] = available
                endpoint['event_refs'] = list(dict.fromkeys(self.captures[v]['event_id'] for v in available))
        endpoint.setdefault('availability', 'available')

    def _authorize(self, snapshot):
        roles = set(self._roles())
        if snapshot['scope']['reader_role'] != self.role:
            raise PublicPFPermissionError('Relation snapshot belongs to another reader')
        for endpoint in [snapshot['root'], *(endpoint for item in snapshot['items'] for endpoint in (item['source'], item['target']) if endpoint)]:
            for role in (endpoint.get('source_role'), endpoint.get('producer_role'), endpoint.get('storage_role')):
                if role:
                    self._authorize_role(role)
        loaded = {}
        with closing(self.store.connect()) as conn:
            for validation in snapshot['validations']:
                check()
                kind = validation['kind']
                if kind == 'revision':
                    role, path = validation['role'], validation['path']
                    if role not in roles:
                        raise PublicPFPermissionError('Relation author is inaccessible')
                    if (role, path) not in loaded:
                        loaded[role, path] = self._registry(role, path)[0]
                    records = loaded[role, path].get(validation['id'], [])
                    record = next((r for r in records if r['revision'] == validation['revision']), None)
                    if record is None or _hash(record) != validation['sha256']:
                        raise PublicPFError('Relation snapshot revision is unavailable or changed; start a new query')
                elif kind == 'capture':
                    try:
                        _, raw = self.store.public_content(validation['version'], connection=conn)
                    except (KeyError, ValueError) as exc:
                        raise PublicPFError('Relation snapshot content is unavailable or changed; start a new query') from exc
                    if hashlib.sha256(raw).hexdigest() != validation['sha256']:
                        raise PublicPFError('Relation snapshot content is unavailable or changed; start a new query')
                elif kind == 'event':
                    row = conn.execute('SELECT request FROM requests WHERE event_id=?', (validation['event'],)).fetchone()
                    if row is None or _hash(json.loads(row['request'])) != validation['request_sha256']:
                        raise PublicPFError('Relation snapshot event is unavailable or changed; start a new query')
                    self.store.readable_event(conn, validation['event'])
                elif kind == 'git':
                    raw = self._git_bytes(validation['role'], validation['path'], validation['commit'])
                    if hashlib.sha256(raw).hexdigest() != validation['sha256']:
                        raise PublicPFError('Relation snapshot Git content is unavailable or changed; start a new query')

    def _compact_endpoint(self, endpoint):
        if endpoint is None:
            return None
        result = self.public.endpoint(endpoint)
        for key in ('request', 'result', 'record', 'operation'):
            result.pop(key, None)
        if result['ref'].get('cite') and len(result['ref']['cite']) > 400:
            result['ref']['cite'] = pf_render.short(result['ref']['cite'], 400)
            result['ref_truncated'] = True
        if 'missing' in result['ref']:
            result['ref']['missing'] = {k: pf_render.short(str(v), 180) for k, v in result['ref']['missing'].items()}
        if endpoint.get('text'):
            result['text'] = pf_render.short(endpoint['text'], 220)
        if endpoint.get('reason'):
            result['reason'] = pf_render.short(endpoint['reason'], 180)
        if endpoint.get('operation'):
            operation = endpoint['operation']['operation']
            result['operation_summary'] = {k: operation[k] for k in ('operation_ref', 'operation_seq', 'actor', 'outcome') if k in operation}
        return result

    def _compact(self, item):
        result = self.public.relation(item)
        for key in ('reference', 'observation'):
            result.pop(key, None)
        result.update(source=self._compact_endpoint(item['source']), target=self._compact_endpoint(item['target']))
        for field in ('reason', 'note'):
            if result.get(field):
                result[field] = pf_render.short(result[field], 220)
        return result


def render(page):
    cutoff = page['cutoff']
    lines = ['PF relations: saved records only; no replay or validity judgment.',
             'Scope: ' + json.dumps(page['scope'], ensure_ascii=True, sort_keys=True),
             'Root: ' + pf_render.short(_label(page['root']), 240),
             'Snapshot cutoff: ' + json.dumps(cutoff, sort_keys=True),
             'Counts: ' + json.dumps(page['counts'], sort_keys=True),
             f"returned={page['returned']}; total_in_scope={page['total_in_scope']}; has_more={str(page['has_more']).lower()}."]
    if page.get('next_cursor'):
        lines.append('More from the same snapshot: pf more ' + page['next_cursor'])
    if page.get('detail_cursor'):
        lines.append('Complete reasons, saved operation results and missing-source fields: pf more ' + page['detail_cursor'])
    if page.get('detail'):
        lines.append(f"Detail characters {page['range'][0]}:{page['range'][1]}/{page['total_chars']}.")
        lines.append(page['detail_text'])
    for item in page['items']:
        source, target = item['source'], item['target']
        lines.append(f"{item.get('relation_revision') or item.get('kind')} | {item['assertion']} | origin={item['origin']} | "
                     f"{pf_render.short(_label(source), 240)} -> {pf_render.short(_label(target), 240)} | "
                     f"author={item.get('author_role')}; revision_status={item.get('record_status', 'observed')}; depth={item['depth']}")
        for label, endpoint in (('source', source), ('target', target)):
            if endpoint:
                lines.append(f"  {label}: state={endpoint['source_state']}; availability={endpoint.get('availability', 'declared_missing' if endpoint['source_state'] == 'missing' else 'unknown')}; "
                             f"source_role={endpoint.get('source_role')}, producer_role={endpoint.get('producer_role')}, storage_role={endpoint.get('storage_role')}")
                if endpoint.get('text'):
                    lines.append('  Text: ' + endpoint['text'])
                if endpoint.get('reason'):
                    lines.append('  ' + endpoint['reason'])
        if item.get('reason'):
            lines.append('  Reason: ' + item['reason'])
        if item.get('basis') == 'execution_membership':
            lines.append(f"  execution_membership: member={item['member_role']}; event_ref={item['event_ref']}; content_ref={item['content_ref']}. This records membership, not source adoption.")
            lines.append('  Explore this execution: ' + item['navigation'])
    if not page['items'] and not page.get('detail'):
        lines.append('No saved relationships in this scope. This does not enumerate all users of the material. --all-revisions includes superseded and retired revisions; --include-candidates adds saved execution observations.')
    return '\n'.join(lines)
