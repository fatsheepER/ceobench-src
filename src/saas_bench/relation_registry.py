"""Atomic source sets and immutable revision receipts, shared by Git and PF."""
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import os
from pathlib import PurePosixPath
import re
import shlex
import uuid

from pydantic import ValidationError

from .registration_evidence import retain_snapshot
from .registration_schema import parse_cite, cite_text
from .relation_schema import MODELS
from .run_lifecycle import RunCancelled
from .workspace_io import open_file, parent_fd


FORMAT = 'ceobench.relations.v1'


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True, allow_nan=False)


class Rejected(ValueError):
    def __init__(self, code, message, **details):
        self.code = code
        self.details = details
        super().__init__(message)


class RelationHold(RunCancelled):
    def __init__(self, receipt, detail):
        self.receipt = receipt
        super().__init__('relation_persistence_unknown', detail)


class RelationRegistry:
    def __init__(self, texts):
        self.texts = texts
        self.workspace, self.role = texts.workspace, texts.role
        self.private_root = texts.private_root
        self.path = self.private_root / 'relations.json'
        self.public_path = self.workspace / 'documents/relations.json'
        self.operation_lookup = None
        self.attempt_id = None
        self.failed = None
        self.last_result = None

    def _load(self):
        if self.path.is_symlink():
            raise ValueError('Relation storage must not be a symlink')
        if not self.path.exists():
            return dict(format=FORMAT, records={}, calls={})
        with open_file(self.private_root, self.path) as stream:
            state = json.load(stream)
        if state.get('format') != FORMAT or not isinstance(state.get('records'), dict) or not isinstance(state.get('calls'), dict):
            raise ValueError('Invalid relation storage')
        for history in state['records'].values():
            for record in history:
                if record['author_role'] != self.role:
                    raise ValueError('Relation author differs from its workspace owner')
                if self.texts.identity:
                    for key, value in self.texts.identity.fields().items():
                        if key != 'session_id' and record.get(key) != value:
                            raise ValueError('Relation identity differs from its workspace owner')
        return state

    def _publish(self, state):
        from .private_storage import publish_json
        from .public_records import public_relation_state
        from contextlib import nullcontext
        with self.texts.projection_lock or nullcontext():
            publish_json(self.private_root, self.path, state, prefix='.relation-')
            if self.texts.workspace_namespace:
                records = {}
                for registry in self.texts.peer_registries:
                    for history in registry.relations._load()['records'].values():
                        records[history[-1]['relation']] = history
                state = dict(format=FORMAT, records=records)
            publish_json(self.workspace, self.public_path, public_relation_state(state, self.texts.public_commit), prefix='.public-relation-')

    def _summary(self, items, results, status=None):
        counts = Counter(item['status'] for item in results)
        status = status or ('partial' if counts['saved'] and counts['rejected'] else
                            'saved' if counts['saved'] else 'rejected')
        return dict(protocol='r2-provenance-v1', status=status, submitted=len(items),
                    **{key: counts[key] for key in ('saved', 'rejected', 'not_attempted', 'unknown')}, items=results)

    def _hold(self, items, results, detail):
        remainder = [dict(item_id=item.get('item_id') if isinstance(item, dict) else None,
                          status='not_attempted') for item in items[len(results):]]
        self.failed = self.last_result = self._summary(items, results + remainder, 'hold')
        raise RelationHold(self.failed, detail)

    def execute(self, args, *, tool_call_id=None):
        if not self.texts.capabilities.source_declarations:
            return 'Error: Source declarations are disabled for this run'
        if self.failed:
            raise RelationHold(self.failed, 'An earlier relation save is unresolved; resume from a checkpoint')
        if not isinstance(args, dict) or set(args) != {'items'} or not isinstance(args.get('items'), list) or not args['items']:
            return encode(dict(status='rejected', error_code='invalid_request',
                               error='Provide a nonempty items list; all identity fields are host supplied',
                               submitted=0, saved=0, rejected=0, not_attempted=0, items=[]))
        items = args['items']
        session = self.texts.identity.session_id if self.texts.identity else 'local'
        key = session + ':' + (tool_call_id or uuid.uuid4().hex)
        results = []
        item_started = False
        try:
            state = self._load()
            request = encode(args)
            prior = state['calls'].get(key)
            if prior:
                if prior['request'] != request:
                    return encode(self._summary(items, [dict(item_id=i.get('item_id') if isinstance(i, dict) else None,
                        status='rejected', error_code='call_conflict', error='This logical tool call already has different arguments') for i in items]))
                if not prior['complete']:
                    self._hold(items, prior['results'], 'An earlier delivery did not finish; resume from a checkpoint')
                self.last_result = prior['receipt']
                return encode(prior['receipt'])
            state['calls'][key] = call = dict(request=request, complete=False, results=[])
            self._publish(state)
            labels = Counter(i.get('item_id') for i in items if isinstance(i, dict) and isinstance(i.get('item_id'), str))
            scope = self.texts.resolver.binding_scope() if self.texts.resolver else nullcontext()
            with scope:
                for raw in items:
                    item_started = True
                    label = raw.get('item_id') if isinstance(raw, dict) else None
                    try:
                        if isinstance(label, str) and labels[label] != 1:
                            raise Rejected('duplicate_item_id', 'item_id must be unique within the batch')
                        if not isinstance(raw, dict) or not isinstance(raw.get('op'), str) or raw['op'] not in MODELS:
                            raise Rejected('invalid_item', 'op must be create, revise or retire')
                        item = MODELS[raw['op']].model_validate(raw).model_dump(exclude_none=True)
                        record, retained = self._prepare(item, state)
                    except (Rejected, ValidationError, PermissionError, ValueError, FileNotFoundError) as exc:
                        code = (exc.code if isinstance(exc, Rejected) else 'permission_denied' if isinstance(exc, PermissionError)
                                else 'invalid_item' if isinstance(exc, ValidationError) else 'invalid_reference')
                        message = ('; '.join('.'.join(map(str, e['loc'])) + ': ' + e['msg'] for e in exc.errors(include_input=False))
                                   if isinstance(exc, ValidationError) else str(exc))
                        result = dict(item_id=label, status='rejected', error_code=code, error=message)
                        if isinstance(exc, Rejected):
                            result.update(exc.details)
                        elif isinstance(exc, ValidationError):
                            location = exc.errors(include_input=False)[0]['loc']
                            if location and location[0] == 'target':
                                result['error_location'] = 'target'
                            elif len(location) >= 3 and location[0] == 'sources' and location[2] == 'ref':
                                result['error_location'] = f'sources[{location[1]}].ref'
                            if 'error_location' in result:
                                result['correction'] = 'Use exactly one cite or operation_ref endpoint; a source may instead contain missing with description and reason.'
                        record, retained = None, []
                    else:
                        result = self._receipt(label, record)
                    for commit in retained if self.texts.capabilities.git_extra_saves else []:
                        retain_snapshot(self.workspace, commit, run=self.texts.git_run)
                    if record is not None:
                        state['records'].setdefault(record['id'], []).append(record)
                    call['results'] = results + [result]
                    self._publish(state)
                    results.append(result)
                    item_started = False
            call.update(complete=True, receipt=self._summary(items, results))
            self._publish(state)
            self.last_result = call['receipt']
            return encode(self.last_result)
        except RelationHold:
            raise
        except Exception as exc:
            if item_started and len(results) < len(items):
                raw = items[len(results)]
                results.append(dict(item_id=raw.get('item_id') if isinstance(raw, dict) else None,
                                    status='unknown', error_code='persistence_unknown',
                                    error='Publication could not be confirmed; no further items were attempted'))
            self._hold(items, results, f'{type(exc).__name__}: {exc}')

    def _prepare(self, item, state):
        previous = None
        if item['op'] == 'create':
            record_id = None
            record = dict(status='active')
        else:
            owner, record_id = (item['relation'].split(':', 1) if ':' in item['relation'] else (self.role, item['relation']))
            if owner != self.role:
                raise Rejected('permission_denied', 'Only the author may revise or retire a relation; create an inferred relation instead')
            history = state['records'].get(record_id)
            if not history:
                raise Rejected('not_found', 'Unknown relation')
            previous = history[-1]
            expected = item['expected_revision']
            if expected not in (previous['revision'], previous['relation_revision'], f"{record_id}.{previous['revision']}"):
                raise Rejected('revision_conflict', 'Expected revision differs from the latest saved revision: ' + previous['relation_revision'])
            if previous['status'] == 'retired':
                raise Rejected('retired', 'This relation is retired; create a new relation')
            record = deepcopy(previous)
        if not self.texts.capabilities.text_tools and item['op'] != 'retire':
            # Validate the whole source set before binding can save any file.
            endpoints = ([('target', item['target'])] if 'target' in item else []) + [
                (f'sources[{i}].ref', source['ref']) for i, source in enumerate(item.get('sources', []))]
            for location, endpoint in endpoints:
                try:
                    if 'cite' in endpoint and 'record' in parse_cite(endpoint['cite'], self.texts.mode == 'pf'):
                        raise Rejected('disabled_reference', 'Registered text targets and sources are disabled; use files, captured outputs or operations')
                except ValueError as exc:
                    raise self._endpoint_error(endpoint, location, exc) from exc
            inherited = []
            if previous:
                if 'target' not in item:
                    inherited.append(('target', previous['target']))
                if 'sources' not in item:
                    inherited.extend((f'sources[{i}].ref', source['ref']) for i, source in enumerate(previous['sources']))
            for location, endpoint in inherited:
                if endpoint.get('kind') == 'text':
                    raise self._endpoint_error(endpoint['ref'], location,
                        Rejected('disabled_reference', 'Replace the inherited text endpoint with a file, captured output or operation'))
        retained = []
        if 'target' in item:
            target, commits = self._bind_at(item['target'], 'target')
            retained += commits
            if previous and target['kind'] != previous['target']['kind']:
                raise self._endpoint_error(item['target'], 'target',
                    Rejected('target_kind_changed', 'Create a new relation to change target kind'))
            record['target'] = target
        if 'sources' in item:
            sources = []
            for index, source in enumerate(item['sources']):
                bound, commits = self._bind_at(source['ref'], f'sources[{index}].ref')
                retained += commits
                sources.append(dict(ref=bound, **({'note': source['note']} if 'note' in source else {})))
            record['sources'] = sources
        record.update(assertion=item.get('assertion', record.get('assertion')), reason=item['reason'])
        if record['assertion'] == 'author_declared' and record['target']['producer_role'] != self.role:
            raise self._endpoint_error(record['target']['ref'], 'target',
                Rejected('author_mismatch', 'The target producer is another role or unknown; use inferred'))
        if item['op'] == 'retire':
            record['status'] = 'retired'
        if record_id is None:
            record_id = 'l' + str(self.texts.allocate('relation'))
        record.update(id=record_id, revision=self.texts.allocate('relation_revision:' + record_id))
        record.update(author_role=self.role, sim_day=self.texts.sim_day(), attempt_id=self.attempt_id,
                      relation=self.role + ':' + record_id,
                      relation_revision=f"{self.role}:{record_id}.{record['revision']}")
        if self.texts.identity:
            record.update(self.texts.identity.fields())
        return record, list(dict.fromkeys(retained))

    def _receipt(self, label, record):
        from .public_records import public_relation
        record = public_relation(record, self.texts.public_commit)
        fields = ('relation', 'relation_revision', 'revision', 'target', 'sources', 'author_role', 'assertion', 'reason')
        return dict(item_id=label, status='saved', **{k: record[k] for k in fields}, record_status=record['status'],
                    source_state='missing' if any(s['ref']['source_state'] == 'missing' for s in record['sources']) else 'available',
                    history=dict(path='documents/relations.json', role=self.role, relation=record['relation'], revision=record['revision']))

    def _bind_at(self, endpoint, location):
        try:
            return self._bind(endpoint)
        except (Rejected, PermissionError, ValueError, FileNotFoundError) as exc:
            raise self._endpoint_error(endpoint, location, exc) from exc

    def _endpoint_error(self, endpoint, location, exc):
        code = exc.code if isinstance(exc, Rejected) else 'permission_denied' if isinstance(exc, PermissionError) else 'invalid_reference'
        details = dict(error_location=location)
        correction = 'Use an accessible public file/output reference; for a peer use ROLE:path. Copy the exact version from its receipt.'
        if code == 'permission_denied':
            correction = 'Use an authorized public reference; private files, escaping paths and symbolic links cannot be cited.'
        elif 'operation_ref' in endpoint:
            details['reference'] = dict(endpoint)
            correction = 'Copy operation_ref from a completed public business receipt; query business_operations if needed.'
        elif 'cite' in endpoint:
            # Never echo absolute host paths or storage diagnostics. Canonicalize
            # only caller-supplied, authorized public names; this is not a claim
            # that the requested version exists.
            try:
                from .pf_paths import normalize, peer_hint
                evidence = parse_cite(endpoint['cite'], self.texts.mode == 'pf')
                if 'unknown' not in evidence:
                    field = 'path' if 'path' in evidence else 'version' if 'version' in evidence else 'record'
                    evidence[field] = normalize(self.texts, evidence[field])
                    cite = cite_text(evidence)
                    details['reference'] = dict(cite=cite)
                    if field == 'record':
                        correction = 'Use an exact saved text revision ROLE:rN.K copied from its receipt or history.'
                    elif self.texts.mode == 'pf':
                        base = cite.rsplit('@v', 1)[0] if re.fullmatch(r'.+@v[1-9][0-9]*', cite) else cite
                        correction = 'Run ' + shlex.join(['pf', 'log', base]) + '; copy an available exact @vN handle for this endpoint.'
                        correction += peer_hint(cite)
                    else:
                        correction = 'Use a saved public file or an exact ROLE:path@COMMIT that exists in the named repository.'
            except (ValueError, PermissionError):
                pass
        if code == 'disabled_reference':
            correction = 'Use a file, captured output or operation_ref; text endpoints are disabled for this run.'
        elif code == 'author_mismatch':
            correction = 'Use inferred for a target produced by another role or an unknown author.'
        elif code == 'target_kind_changed':
            correction = 'Create a new relation to change the target kind.'
        details['correction'] = correction
        from .public_errors import PublicPFPermissionError
        message = ('Saved reference is unavailable' if isinstance(exc, FileNotFoundError) else
                   'Evidence is inaccessible' if isinstance(exc, PermissionError) and not isinstance(exc, PublicPFPermissionError)
                   else str(exc))
        return Rejected(code, f'{location}: {message}', **details)

    def _bind(self, endpoint):
        if 'missing' in endpoint:
            return dict(kind='missing', ref=endpoint, source_state='missing', readable=False,
                        producer_role=None, source_role=None, storage_role=None), []
        if 'operation_ref' in endpoint:
            if self.operation_lookup is None:
                raise Rejected('not_found', 'Operation references require the public business ledger')
            operation = self.operation_lookup(endpoint['operation_ref'])
            if operation is None:
                raise Rejected('not_found', 'Unknown operation_ref in this world')
            actor = operation['operation']['actor']
            from .role_policy import READABLE_ROLES
            if actor not in READABLE_ROLES[self.role]:
                raise PermissionError('Operation evidence is inaccessible')
            if operation['operation']['outcome'] == 'unfinished':
                raise Rejected('not_terminal', 'The operation has not reached a recorded outcome; query its receipt later')
            return dict(kind='operation', ref=endpoint, source_state='available', readable=True,
                        producer_role=actor, source_role=actor, storage_role=None, operation=operation,
                        sha256=hashlib.sha256(encode(operation).encode()).hexdigest()), []
        cite = endpoint['cite']
        evidence = parse_cite(cite, self.texts.mode == 'pf')
        if 'unknown' in evidence:
            raise Rejected('invalid_reference', 'Use an explicit missing source; targets cannot be missing')
        if 'record' in evidence and not re.fullmatch(r'(?:(?:ceo|growth|ops_finance):)?r[1-9][0-9]*\.[1-9][0-9]*', evidence['record']):
            raise Rejected('ambiguous_version', 'Cite an exact text revision ROLE:rN.K')
        if 'version' in evidence and not re.fullmatch(r'.+@v[1-9][0-9]*', evidence['version']):
            raise Rejected('ambiguous_version', 'Cite an exact captured version ROLE:HANDLE@vK')
        if (self.texts.mode == 'pf' and not self.texts.capabilities.text_tools
                and 'path' in evidence and 'commit' not in evidence):
            raise Rejected('ambiguous_version', 'Cite an exact captured file version ROLE:path@vN; a bare path does not identify the version you used')
        ref = dict(evidence=evidence)
        try:
            binding = self.texts._bind(ref, self.texts._load()['records'])
        except ValueError as exc:
            if str(exc).startswith('Could not save the named file snapshot'):
                raise OSError(str(exc)) from exc
            raise
        evidence = ref['evidence']
        if 'record' in evidence:
            owner, version = evidence['record'].split(':', 1) if ':' in evidence['record'] else (self.role, evidence['record'])
            path = self.texts.authority_path(owner, 'registrations.json')
            with open_file(path.parent, path) as stream:
                history = json.load(stream)['records'][version.split('.')[0]]
            record = next(r for r in history if r['version'] == version)
            producer = owner if record.get('author') == owner else None
            if self.texts.identity and (record.get('role') != owner or
                    any(record.get(k) != getattr(self.texts.identity, k) for k in ('run_id', 'world_id'))):
                producer = None
            return dict(kind='text', ref=dict(cite=owner + ':' + version), source_state='available', readable=True,
                        producer_role=producer, source_role=owner, storage_role=owner,
                        sha256=hashlib.sha256(encode(record).encode()).hexdigest()), []
        if binding.get('git_commit'):
            owner = evidence.get('owner', self.texts.workspace_namespace or self.role)
            workspace = self.texts.repository(owner)
            full, path = binding['git_commit'], evidence['path']
            parts = PurePosixPath(path).parts
            if not parts or '.git' in parts or parts[0] == 'sessions':
                raise Rejected('permission_denied', 'Only public business files may be cited')
            raw = self.texts._git_output(['git', '--no-replace-objects', '-C', str(workspace), 'show', full + ':' + path])
            source = ref.get('snapshot_source')
            producer = None if self.texts.workspace_namespace else source.get('owner', self.role) if source else self._git_producer(owner, path, full)
            original = dict(cite=cite_text(source)) if source else dict(cite=owner + ':' + path + '@' + full)
            retained_by_owner = self.texts.git_run(['git', '-C', str(workspace), 'show-ref', '--verify', '--quiet',
                'refs/ceobench/snapshots/' + full], capture_output=True, timeout=10).returncode == 0
            source_role = None if self.texts.workspace_namespace else producer if producer is not None else owner
            if not self.texts.workspace_namespace and owner != self.role and not retained_by_owner and self.texts.capabilities.git_extra_saves:
                # Preserve the requested old bytes without writing retention refs in the peer's repository.
                full = self._copy_git_bytes(path, raw, original['cite'])
                owner = self.role
            bound = dict(kind='file', ref=dict(cite=owner + ':' + path + '@' + full),
                         source_state='available', readable=True, producer_role=producer,
                         source_role=source_role, storage_role=None if self.texts.workspace_namespace else owner,
                         **(dict(workspace_namespace=owner, collector_role=self.role) if self.texts.workspace_namespace else {}),
                         snapshot_source=original,
                         sha256=hashlib.sha256(raw).hexdigest())
            return bound, [full] if owner in (self.role, self.texts.workspace_namespace) else []
        if not self.texts.resolver or 'version_id' not in binding:
            raise Rejected('not_found', 'No readable exact content version was resolved')
        meta, raw = self.texts.resolver.content(binding['version_id'])
        producer = meta.get('producer_role') if meta.get('workspace_namespace') else meta.get('owner_role')
        location = {key: meta[key] for key in ('workspace_namespace', 'collector_role', 'publisher_role', 'writer_role') if key in meta} if meta.get('workspace_namespace') else {}
        kind = 'file' if meta['layer'] in ('file_bytes', 'file_text') else 'output'
        return dict(kind=kind, ref=dict(cite=self.texts.resolver.handle(binding['version_id'])),
                    source_state='available', readable=True, producer_role=producer, source_role=producer,
                    storage_role=None if location else producer, **location, sha256=hashlib.sha256(raw).hexdigest(),
                    capture=dict(reading_scope=binding.get('reading_scope', 'unknown'),
                                 source_truncated=binding.get('source_truncated'), layer=meta['layer'])), []

    def _git_producer(self, owner, path, commit):
        if self.texts.workspace_namespace:
            return None
        workspace = self.texts.repository(owner)
        producers = set()
        registry = self.texts.authority_path(owner, 'registrations.json')
        if registry.exists():
            with open_file(registry.parent, registry) as stream:
                texts = json.load(stream)
            for history in texts['records'].values():
                for record in history:
                    for ref in record.get('references', []):
                        evidence = ref.get('evidence', {})
                        if evidence.get('path') == path and evidence.get('owner', owner) == owner and commit.startswith(evidence.get('commit', '!')):
                            if 'snapshot_source' in ref:
                                producers.add(ref['snapshot_source'].get('owner', owner))
        relation_path = self.texts.authority_path(owner, 'relations.json')
        if relation_path.exists():
            with open_file(relation_path.parent, relation_path) as stream:
                relations = json.load(stream)
            wanted = owner + ':' + path + '@' + commit
            for history in relations['records'].values():
                for record in history:
                    for ref in [record['target'], *(s['ref'] for s in record['sources'])]:
                        if ref['ref'].get('cite') == wanted:
                            producers.add(ref['producer_role'])
        return producers.pop() if len(producers) == 1 else None

    def _copy_git_bytes(self, path, raw, source):
        def git(*args, data=None):
            return self.texts._git_output(['git', '--no-replace-objects', '-c', 'core.hooksPath=/dev/null',
                '-c', 'user.name=CEO Bench', '-c', 'user.email=ceobench@localhost', '-c', 'commit.gpgsign=false',
                '-C', str(self.workspace), *args], input=data).decode().strip()
        blob = git('hash-object', '-w', '--stdin', data=raw)
        parts = PurePosixPath(path).parts
        tree = git('mktree', '-z', data=f'100644 blob {blob}\t{parts[-1]}\0'.encode())
        for part in reversed(parts[:-1]):
            tree = git('mktree', '-z', data=f'040000 tree {tree}\t{part}\0'.encode())
        return git('commit-tree', tree, data=('Registered relation snapshot of ' + source + '\n').encode())
