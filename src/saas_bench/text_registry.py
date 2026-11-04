"""Workspace declarations with private evidence bindings kept outside the workspace."""
from copy import deepcopy
from contextlib import nullcontext
import json
import re
from pathlib import Path
import subprocess
import uuid

from pydantic import ValidationError

from .execution_capture import CapturedText, CURRENT_EVENT, origin
from . import evidence_handles, pf_render
from .registration_evidence import EvidenceResolver, git_reference, native_repository, snapshot_reference, retain_snapshot
from .registration_schema import MODELS, applies_text, cite_text, internal
from .run_state import write_json
from .sql_evidence import encoded


class TextRegistry:
    def __init__(self, workspace, mode, store=None, sim_day=lambda: None, identity=None, private_root=None, capabilities=None):
        from .agent_capabilities import AgentCapabilities
        self.capabilities = AgentCapabilities.resolve(capabilities)
        if mode not in ('git', 'prefix', 'pf'):
            raise ValueError('Invalid registration mode')
        self.git_run = subprocess.run
        self.identity = identity
        self.role = identity.role if identity else "ceo"
        self.workspace = Path(workspace).resolve()
        from .private_storage import private_root as storage_root
        self.private_root = storage_root(self.workspace, identity=identity, root=private_root)
        self.path = self.private_root / 'registrations.json'
        self.public_path = self.workspace / 'documents/registrations.json'
        self.public_refs = getattr(store, 'public_refs', None)
        self.mode, self.store, self.sim_day = mode, store, sim_day
        if mode in ('prefix', 'pf') and (store is None or not store.execution_capture):
            raise ValueError('Prefix/PF registration requires execution capture')
        if store and store.path.resolve().is_relative_to(self.workspace):
            raise ValueError('PF evidence storage must be outside the agent workspace')
        self.resolver = EvidenceResolver(store) if store else None
        from .relation_registry import RelationRegistry
        self.relations = RelationRegistry(self)

    def _load(self):
        if self.path.is_symlink() or self.path.with_suffix('.json.tmp').is_symlink():
            raise ValueError('Registration storage must not be a symlink')
        if not self.path.exists():
            return dict(format='ceobench.text-register.v1', records={})
        from .workspace_io import open_file
        with open_file(self.private_root, self.path) as stream:
            value = json.load(stream)
        if value.get('format') != 'ceobench.text-register.v1' or not isinstance(value.get('records'), dict):
            raise ValueError('Invalid registration file')
        if self.identity:
            for revisions in value['records'].values():
                for record in revisions:
                    owner = {k: v for k, v in self.identity.fields().items() if k != 'session_id'}
                    if (record.get('author') != self.role or not isinstance(record.get('session_id'), str)
                            or not record['session_id'] or any(record.get(k) != v for k, v in owner.items())):
                        raise ValueError('Registration identity differs from its workspace owner')
        return value

    def _git_output(self, argv, **kwargs):
        return self.git_run(argv, capture_output=True, check=True, timeout=10, **kwargs).stdout

    def _git_reference(self, workspace, evidence):
        if self.identity:
            native_repository(workspace)
        reference, full = git_reference(workspace, evidence, run=self.git_run)
        return dict(reference, commit=full), full

    def execute(self, operation, args):
        if not self.capabilities.text_tools:
            return 'Error: Text tools are disabled for this run'
        with self.resolver.binding_scope() if self.resolver else nullcontext():
            return self._execute(operation, args)

    def _execute(self, operation, args):
        try:
            values = MODELS[self.mode][operation].model_validate(args).model_dump(exclude_none=True)
        except ValidationError as exc:
            # Do not echo the submitted value: it can contain long private identifiers.
            errors = ['.'.join(map(str, e['loc'])) + ': ' + e['msg'] for e in exc.errors(include_input=False)]
            raise ValueError('; '.join(errors)) from exc
        if operation in ('create', 'revise'):
            values = internal(values, self.mode == 'pf')
        state = self._load()
        if operation == 'list':
            self.last_result = None
            return self._list(state, **values)
        records = state['records']
        if operation == 'create':
            record_id = None
            previous = None
            record = dict(values, status='active')
        else:
            record_id = values.pop('record')
            if record_id not in records:
                raise ValueError('Unknown registered text')
            previous = records[record_id][-1]
            if previous['status'] == 'retired':
                raise ValueError('Registered text is retired; create a new text to resume it')
            record = dict(deepcopy(previous), **values)
            if operation == 'retire':
                record['status'] = 'retired'
        # Agents reason in simulated days; clock time stays in the private event record.
        record.update(sim_day=self.sim_day(), author=self.role)
        if self.identity:
            record.update(self.identity.fields())
        warnings, bindings = [], []
        if operation == 'create' or 'references' in values:
            for ref in record['references']:
                if len(ref.get('note', '')) > 200:
                    ref['note'] = ref['note'][:200]
                    warnings.append('Note truncated to 200 characters.')
                bindings.append(self._bind(ref, records))
        elif self.store:
            binding = self.store.load_state('declaration:' + previous['version'])
            bindings = binding['references'] if binding else []
        if operation == 'create' or 'references' in values:
            for ref in record['references']:
                if 'snapshot_source' in ref:
                    retain_snapshot(self.workspace, ref['evidence']['commit'], run=self.git_run)
        if record_id is None:
            record_id = 'r' + str(self.allocate('text'))
        record.update(id=record_id, revision=self.allocate('text_revision:' + record_id))
        record['version'] = f"{record_id}.{record['revision']}"
        records.setdefault(record_id, []).append(record)
        # Publish only after every reference validates and each snapshot is retained.
        from .private_storage import publish_json
        from .public_records import public_text, public_text_state
        publish_json(self.private_root, self.path, state, prefix='.registration-')
        publish_json(self.workspace, self.public_path, public_text_state(state, self.public_commit), prefix='.public-registration-')
        if self.store:
            try:
                event = CURRENT_EVENT.get()
                owned = event is None
                if owned:
                    event = self.store.begin_event('text_' + operation, values)
                version = self.store.version(event, 'registered_text', encoded(public_text(record, self.public_commit)),
                                             layer='registered_text', object_id='record:' + record_id)
                private = dict(version_id=version, references=bindings)
                self.store.version(event, 'declaration', encoded(private), layer='agent_declaration',
                                   object_id='declaration:' + record['version'])
                self.store.save_state('declaration:' + record['version'], private)
                if owned:
                    self.store.complete(event)
            except Exception as exc:
                self.store.fail(exc)
                raise RuntimeError('Text saved, but private capture failed; collection stopped') from exc
        # Structured form for audits and tests; the agent receives the text receipt.
        self.last_result = result = dict(id=record_id, version=record['version'], status=record['status'])
        if warnings:
            result['warnings'] = list(dict.fromkeys(warnings))
        retained = (operation == 'revise' and record['text'] != previous['text']
                    and 'references' not in values)
        shown = operation == 'create' or 'references' in values
        if self.mode == 'pf':
            result['evidence'] = [self._display_binding(b) for b in bindings]
            cited = [self._pf_cited(b, ref['evidence']) for b, ref in zip(bindings, record['references'])]
        else:
            # Git and prefix receipts show only what the Git group can know: the stored reference.
            cited = [self._git_cited(ref['evidence']) +
                     (' (snapshot of ' + self._git_cited(ref['snapshot_source']) + ')' if 'snapshot_source' in ref else '')
                     for ref in record['references']]
        verb = dict(create='Registered', revise='Revised', retire='Retired')[operation]
        handle = (self.role + ":" if self.identity else "") + record["version"]
        lines = [f"{verb} {handle} ({record['status']})."]
        if retained:
            lines.append('Retained references: ' + (' · '.join(cited) or 'none'))
        elif shown and cited:
            lines.append('Cited: ' + ' · '.join(cited))
        if self.mode == 'pf' and operation != 'retire' and (writes := self._week_writes(record)):
            lines.append('Business writes this week touching ' + ', '.join(writes[0]) + ': ' + ' · '.join(writes[1]))
        lines += result.get('warnings', [])
        return '\n'.join(lines)

    def _git_cited(self, evidence):
        if 'unknown' in evidence:
            return f"unknown ({evidence['unknown']})"
        if 'record' in evidence:
            return evidence['record']
        from .public_records import public_evidence_reference
        shown = public_evidence_reference(evidence, self.public_commit, self.role)
        text = cite_text(shown)
        return text + ' (original commit unavailable)' if shown.get('availability') else text

    def _pf_cited(self, binding, evidence):
        shown = self._display_binding(binding)
        if 'record' in shown:
            return shown['record']
        if shown.get('status') == 'unknown':
            return f"unknown ({shown['reason']})"
        if shown.get('status') == 'commit_only':
            seen = f"; you saw {shown['delivered']}" if shown.get('delivered') else ''
            return f"{self._git_cited(evidence)} (commit only: those bytes never reached you{seen})"
        handles = evidence_handles.index(self.store)
        group = handles.group(binding['version_id'])
        text = shown['version'] + (f" (day {group['day'] if group['day'] is not None else 'unknown'})" if group else '')
        meta, _ = self.resolver.content(binding['version_id'])
        event = self.store.read_event(meta['created_by_event'], evidence=True)
        record, result = event['request'], event['result']
        definition = event['query_definition'] if meta['layer'] == 'server_public_response' else None
        text += ' — ' + pf_render.label(meta['layer'], record['kind'], record.get('request'),
            definition[5] if definition else None, meta.get('object_id'), result.get('classification'))
        if result['status'] != 'succeeded':
            text += f" ({result['status']}" + (f"; exit code: {result['exit_code']}" if 'exit_code' in result else '') + ')'
        if note := handles.note(binding['version_id']):
            text += f' — source note: "{pf_render.short(note[1], 120)}"'
        if shown['differs']:
            text += f" (latest captured {shown['latest']}, {self._change(binding)})"
        if binding.get('reading_scope') in ('partial', 'not_in_request', 'not_read'):
            scope = 'part of body present in this request' if binding['reading_scope'] == 'partial' else 'body not present in this request'
            text += f" (whole captured object; {scope}; pf show {shown['version']} --full)"
        return text

    def _change(self, binding):
        meta, old = self.resolver.content(binding['version_id'])
        _, new = self.resolver.content(binding['latest_version_id'])
        kind = ('query' if meta['layer'] == 'server_public_response' else
                'json' if str(meta.get('object_id', '')).endswith('.json') else 'text')
        return pf_render.change(kind, old, new)

    def _week_writes(self, record):
        """This week's successful business writes whose objects share an ID with the text's objects."""
        wanted = {str(o['id']) for o in record['objects']}
        handles = evidence_handles.index(self.store)
        handles.refresh()
        touched, calls = set(), []
        for event_id, (kind, request, day) in list(handles.events.items()):
            parsed = (request.get('request') or {}).get('parsed') or {}
            if (kind != 'public_http' or day != handles.day or not parsed.get('tool') or
                    (request.get('request') or {}).get('method') == 'GET' or parsed['tool'] in evidence_handles.READ_TOOLS):
                continue
            try:
                meta = self.store.get_content(event_id + ':public_response')[0]
                status = self.store.read_event(event_id)['result'].get('status')
            except (KeyError, ValueError):
                continue
            ids = {str(o['value']) for o in meta.get('objects', [])} & wanted
            if ids and status in ('succeeded', 'partially_succeeded'):
                touched |= ids
                calls.append(f"day {day} " + pf_render.call_text(parsed))
        return (sorted(touched), calls[-6:]) if calls else None

    def public_commit(self, owner, commit):
        workspace = self.repository(owner)
        result = self.git_run(['git', '--no-replace-objects', '-C', str(workspace),
            'rev-parse', '--short=7', '--verify', commit + '^{commit}'], capture_output=True, timeout=10)
        return result.stdout.decode().strip() if result.returncode == 0 else None

    def repository(self, owner=None):
        if not self.identity:
            return self.workspace
        from .role_policy import READABLE_ROLES
        owner = owner or self.role
        if owner not in READABLE_ROLES[self.role]:
            raise PermissionError('Evidence is inaccessible')
        return self.workspace.parent / owner

    def allocate(self, namespace):
        if self.public_refs is None:
            from .public_refs import PublicRefs
            self.public_refs = PublicRefs(self.private_root / 'public-refs.sqlite')
        return self.public_refs.ordinal(namespace, self.role, uuid.uuid4().hex)

    def authority_path(self, owner, filename):
        if filename not in ('registrations.json', 'relations.json'):
            raise ValueError('Unknown private registry')
        self.repository(owner)
        return (self.private_root if owner == self.role or not self.identity else self.private_root.parent / owner) / filename

    def _reference_owner(self, evidence):
        owner = evidence.get('owner', self.role)
        if self.identity and 'path' in evidence:
            value = evidence['path']
            if value.split(':', 1)[0] in ('ceo', 'growth', 'ops_finance') and ':' in value:
                owner, evidence['path'] = value.split(':', 1)
            elif Path(value).is_absolute():
                for candidate in self.workspace.parent.iterdir():
                    if Path(value).is_relative_to(candidate):
                        owner, evidence['path'] = candidate.name, str(Path(value).relative_to(candidate))
                        break
            self.repository(owner)
            evidence['owner'] = owner
        return owner

    def _bind(self, ref, records):
        evidence = ref['evidence']
        owner = self._reference_owner(evidence)
        workspace = self.repository(owner)
        if 'unknown' in evidence:
            return dict(status='unknown', reason=evidence['unknown'])
        full = None
        source_evidence = evidence
        if self.mode == 'pf' and 'path' in evidence and 'commit' not in evidence \
                and evidence_handles.VERSIONED.fullmatch(evidence['path']):
            evidence = ref['evidence'] = {'version': evidence['path']}  # forecast.json@v3 written as a path
        if self.mode == 'pf' and evidence_handles.RECORD.fullmatch(evidence.get('version', '')):
            evidence = ref['evidence'] = {'record': evidence['version']}
        if 'record' in evidence:
            requested = evidence['record']
            record_owner = self.role
            record_store = records
            if self.identity and ':' in requested:
                record_owner, requested = requested.split(':', 1)
                from .workspace_io import open_file
                path = self.authority_path(record_owner, 'registrations.json')
                with open_file(path.parent, path) as stream:
                    record_store = json.load(stream)['records']
            history = record_store.get(requested.split('.')[0], [])
            target = (next((r for r in history if r['version'] == requested), None)
                      if '.' in requested else history[-1] if history else None)
            if target is None:
                raise ValueError('Unknown registered text revision; use unknown with a reason')
            evidence['record'] = (record_owner + ':' if self.identity else '') + target['version']
        elif self.mode in ('git', 'prefix') and 'path' in evidence and 'commit' not in evidence \
                and '@' not in evidence['path']:
            if not self.capabilities.git_extra_saves:
                raise ValueError('Immediate Git snapshots are disabled; save and commit the file, then cite path@COMMIT')
            if (ref.get('select') or ref.get('predicate')) and not evidence['path'].endswith(('.json', '.csv')):
                raise ValueError('Plain text supports whole-text equality only')
            ref['evidence'], full = snapshot_reference(workspace, evidence['path'], self.workspace,
                owner if self.identity else None, run=self.git_run)
            ref['snapshot_source'] = dict(evidence)
            if self.identity:
                ref['evidence']['owner'] = self.role
            workspace = self.workspace
        elif self.mode in ('git', 'prefix') or (
                'path' in evidence and ('commit' in evidence or '@' in evidence['path'])):
            # PF keeps the Git group's committed-file references (design 4.1).
            if 'path' not in evidence:
                raise ValueError('Evidence must be a workspace file path or a registered text rN.M; otherwise use unknown with a reason')
            ref['evidence'], full = self._git_reference(workspace, evidence)
            if self.identity:
                ref['evidence']['owner'] = owner
            if (ref.get('select') or ref.get('predicate')) and not ref['evidence']['path'].endswith(('.json', '.csv')):
                raise ValueError('Plain text supports whole-text equality only')
        elif 'path' in evidence:
            path = Path(evidence['path'])
            if path.is_absolute() or '..' in path.parts or path.as_posix() != evidence['path']:
                raise ValueError('Evidence path must be workspace-relative')
        binding = dict(status='git', git_commit=full) if full else dict(status='registered_text', **evidence)
        if self.mode == 'git':
            return binding
        accept = None
        if full:
            # Keep the Git identity distinct from the delivered PF version. Bind only a
            # delivered capture whose bytes hash to that committed blob.
            blob = self._git_output(['git', '-C', str(workspace), 'rev-parse',
                full + ':' + ref['evidence']['path']], text=True).strip()
            def accept(version):
                raw = self.store.get_content(version)[1]
                return blob == self._git_output(['git', '-C', str(workspace), 'hash-object', '--stdin'],
                                                       input=raw).decode().strip()
        lookup = source_evidence if 'snapshot_source' in ref else ref['evidence']
        try:
            try:
                binding.update(self.resolver.resolve(lookup, ref, accept=accept), status='resolved')
                if full:
                    binding['git_content_matches'] = True
            except ValueError:
                if not full:
                    raise
                # No delivered capture has the committed bytes: record the last delivered
                # version and the mismatch, which traversal reports as a missing edge.
                binding.update(self.resolver.resolve(lookup, ref), status='resolved',
                               git_content_matches=False)
        except ValueError as exc:
            if self.mode == 'pf' and not full:
                raise
            # A committed-file reference stays valid as in Git; prefix results never
            # disclose whether the private match succeeded.
            binding.update(status='unknown', reason=str(exc))
        return binding

    def _display_binding(self, binding):
        shown = dict(commit=binding['git_commit'][:7]) if binding.get('git_commit') else {}
        if binding.get('status') == 'registered_text':
            return dict(record=binding['record'])
        if binding.get('git_content_matches') is False:
            return dict(shown, status='commit_only', reason='The committed bytes were not sent to your model',
                        delivered=self.resolver.handle(binding['version_id']))
        if binding.get('status') != 'resolved':
            status = 'commit_only' if binding.get('git_commit') else 'unknown'
            return dict(shown, status=status, reason=binding.get('reason', 'not_captured_in_prefix'))
        version, latest = (self.resolver.handle(binding[k]) for k in ('version_id', 'latest_version_id'))
        return dict(shown, version=version, latest=latest, differs=version != latest)

    def _list(self, state, after, limit, raw=False):
        active = [versions[-1] for key, versions in sorted(state['records'].items(), key=lambda kv: int(kv[0][1:]))
                  if int(key[1:]) > after and versions[-1]['status'] == 'active']
        page = active[:limit]
        next_after = int(page[-1]['id'][1:]) if len(active) > limit else None
        if not raw:
            lines = [f'Saved active texts: {len(page)} of {len(active)} after r{after}.']
            for record in page:
                handle = (self.role + ':' if self.identity else '') + record['version']
                sentence = re.split(r'(?<=[.!?])\s|(?<=[。！？])|\n', record['text'], maxsplit=1)[0]
                excerpt = pf_render.short(sentence, 120)
                if sentence != record['text'] and not excerpt.endswith(('...', '…')):
                    excerpt += ' …'
                day = record['sim_day'] if record.get('sim_day') is not None else 'unknown'
                lines.append(f"{handle} · {record['status']} · saved day {day} · applies {pf_render.short(applies_text(record['applies_at']), 120)} · {excerpt}")
            if not page:
                lines.append('No active texts in this page.')
            lines.append('Full records: text_list ' + json.dumps(dict(after=after, limit=limit, raw=True), separators=(',', ':')))
            if next_after is not None:
                lines.append('Next: text_list ' + json.dumps(dict(after=next_after, limit=limit), separators=(',', ':')))
            return CapturedText('\n'.join(lines), [])
        result, origins = '{"records":[', []
        for record in page:
            if result[-1] != '[':
                result += ','
            from .public_records import public_text
            content = encoded(public_text(record, self.public_commit)).decode()
            if self.store:
                binding = self.store.load_state('declaration:' + record['version'])
                if binding and self.store.get_content(binding['version_id'])[1] == content.encode():
                    origins.append(origin(binding['version_id'], content, target=len(result)))
            result += content
        next_after = int(page[-1]['id'][1:]) if len(active) > limit else None
        result += '],"next_after":' + json.dumps(next_after) + '}'
        if self.identity:
            result = result[:-1] + ',"shareable_references":' + json.dumps({
                r['version']: self.role + ':' + r['version'] for r in page}) + '}'
        return CapturedText(result, origins)
