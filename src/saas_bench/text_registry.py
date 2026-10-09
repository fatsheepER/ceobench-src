"""Workspace declarations with private evidence bindings kept outside the workspace."""
from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess

from pydantic import ValidationError

from .execution_capture import CapturedText, CURRENT_EVENT, origin
from . import evidence_handles, pf_render
from .registration_evidence import EvidenceResolver, git_reference, week_label, weekly_reference
from .registration_schema import MODELS, cite_text, internal
from .run_state import write_json
from .sql_evidence import encoded


def applies_ended(record, day):
    """The text's stated applicability ended before this day, e.g. last week's plan for days 42-48."""
    applies = record.get('applies_at') or {}
    last = applies.get('end_day', applies.get('day'))
    return type(last) is int and last < day


class TextRegistry:
    def __init__(self, workspace, mode, store=None, sim_day=lambda: None, identity=None):
        if mode not in ('git', 'prefix', 'pf'):
            raise ValueError('Invalid registration mode')
        self.git_run = subprocess.run
        self.identity = identity
        self.role = identity.role if identity else "ceo"
        self.workspace = Path(workspace).resolve()
        self.path = self.workspace / 'registrations.json'
        self.mode, self.store, self.sim_day = mode, store, sim_day
        if mode in ('prefix', 'pf') and (store is None or not store.execution_capture):
            raise ValueError('Prefix/PF registration requires execution capture')
        if store and store.path.resolve().is_relative_to(self.workspace):
            raise ValueError('PF evidence storage must be outside the agent workspace')
        self.resolver = EvidenceResolver(store) if store else None

    def _load(self):
        if self.path.is_symlink() or self.path.with_suffix('.json.tmp').is_symlink():
            raise ValueError('Registration storage must not be a symlink')
        if not self.path.exists():
            return dict(format='ceobench.text-register.v1', records={})
        from .workspace_io import open_file
        with open_file(self.workspace, self.path) as stream:
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

    def execute(self, operation, args):
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
            record_id = 'r' + str(max((int(k[1:]) for k in records), default=0) + 1)
            previous = None
            record = dict(values, id=record_id, revision=1, status='active')
        else:
            record_id = values.pop('record')
            if record_id not in records:
                raise ValueError('Unknown registered text')
            previous = records[record_id][-1]
            if previous['status'] == 'retired':
                raise ValueError('Registered text is retired; create a new text to resume it')
            record = dict(deepcopy(previous), **values, revision=previous['revision'] + 1)
            if operation == 'retire':
                record['status'] = 'retired'
        # Agents reason in simulated days; clock time stays in the private event record.
        record.update(version=f"{record_id}.{record['revision']}", sim_day=self.sim_day(), author=self.role)
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
        records.setdefault(record_id, []).append(record)
        # Validate every reference before touching the workspace. A failed declaration
        # does not consume a revision or save partially validated references.
        if self.identity:
            import os
            import uuid
            from .workspace_io import parent_fd
            with parent_fd(self.workspace, self.path) as (directory, name):
                temp = '.registration-' + uuid.uuid4().hex
                fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
                try:
                    with os.fdopen(fd, 'w') as stream:
                        json.dump(state, stream, ensure_ascii=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temp, name, src_dir_fd=directory, dst_dir_fd=directory)
                finally:
                    try:
                        os.unlink(temp, dir_fd=directory)
                    except FileNotFoundError:
                        pass
        else:
            write_json(self.path, state)
        if self.store:
            try:
                event = CURRENT_EVENT.get()
                owned = event is None
                if owned:
                    event = self.store.begin_event('text_' + operation, values)
                version = self.store.version(event, 'registered_text', encoded(record),
                                             layer='registered_text', object_id='record:' + record_id)
                private = dict(version_id=version, references=bindings)
                if any(b.get('awaiting_week') for b in bindings):
                    pending = self.store.load_state('weekly_declarations') or []
                    self.store.save_state('weekly_declarations', pending + [record['version']])
                else:
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
        shown = operation == 'create' or 'references' in values
        if self.mode == 'pf':
            result['evidence'] = [self._display_binding(b) for b in bindings]
            cited = [self._pf_cited(b, ref['evidence']) for b, ref in zip(bindings, record['references'])]
        else:
            # Git and prefix receipts show only what the Git group can know: the stored reference.
            cited = [self._git_cited(ref['evidence']) for ref in record['references']]
        verb = dict(create='Registered', revise='Revised', retire='Retired')[operation]
        lines = [f"{verb} {record['version']} ({record['status']})."]
        if self.identity:
            lines.append(f"Shareable reference: {self.role}:{record['version']}")
        if shown and cited:
            lines.append('Cited: ' + ' · '.join(cited))
        if self.mode == 'pf' and operation != 'retire' and (writes := self._week_writes(record)):
            lines.append('Business writes this week touching ' + ', '.join(writes[0]) + ': ' + ' · '.join(writes[1]))
        lines += result.get('warnings', [])
        return '\n'.join(lines)

    def finalize_week(self, label):
        """Publish prefix declarations against the exact bytes of their closing commit.

        Provisional bindings live only in private state, never in the dependency
        graph. Historical revisions are finalized too, including inherited refs.
        """
        if self.mode != 'prefix':
            return
        pending = self.store.load_state('weekly_declarations') or []
        if not pending:
            return
        records = {r['version']: r for history in self._load()['records'].values() for r in history}
        for revision in pending[:]:
            private = self.store.load_state('declaration:' + revision)
            refs = private['references']
            if not any(b.get('awaiting_week') == label for b in refs):
                continue
            for i, (ref, binding) in enumerate(zip(records[revision]['references'], refs)):
                if binding.get('awaiting_week') == label:
                    refs[i] = self._closing_binding(ref, binding)
            if any(b.get('awaiting_week') for b in refs):
                self.store.save_state('declaration:' + revision, private)
                continue
            event = self.store.begin_event('prefix_week_binding', dict(revision=revision, week=label))
            # The declaration and its lookup state become visible together. A crash
            # cannot publish an old missing edge alongside the completed binding.
            with self.store.batch():
                self.store.version(event, 'declaration', encoded(private), layer='agent_declaration',
                                   object_id='declaration:' + revision)
                self.store.save_state('declaration:' + revision, private)
                pending.remove(revision)
                self.store.save_state('weekly_declarations', pending)
                self.store.complete(event)

    def _closing_binding(self, ref, provisional):
        binding = dict(git_week=provisional['awaiting_week'])
        try:
            evidence, full = git_reference(self.workspace, ref['evidence'], run=self.git_run)
            binding['git_commit'] = full
            raw = self._git_output(['git', '-C', str(self.workspace), 'show', full + ':' + evidence['path']])
            handles = evidence_handles.index(self.store)
            handles.refresh()
            candidates = [v for group in reversed(handles.groups.get(handles.object_key('file', evidence['path'], evidence.get('owner')), []))
                          for v in reversed(group['members'])]
            matches = [v for v in candidates if self.resolver.content(v)[1] == raw]
            if not matches:
                raise ValueError('Closing commit file bytes were not captured in the prefix')
            if provisional.get('status') == 'resolved' and provisional.get('version_id') in matches:
                # Retain actual registration-time delivery, never a later read/write.
                binding.update({k: v for k, v in provisional.items() if k != 'awaiting_week'})
                binding['latest_version_id'] = candidates[0]
            elif ref.get('select') or ref.get('predicate'):
                raise ValueError('Selected closing-commit bytes were not available at registration')
            else:
                binding.update(self.resolver._whole_binding(matches[0], candidates[0], [], 'committed_bytes'))
            binding.update(status='resolved', git_content_matches=True)
        except ValueError as exc:
            binding.update(status='unknown', reason=str(exc))
        return binding

    def assert_week_finalized(self, day):
        if self.mode == 'prefix':
            for revision in self.store.load_state('weekly_declarations') or []:
                private = self.store.load_state('declaration:' + revision)
                if any(int(b['awaiting_week'].split('-')[1]) * 7 <= day
                       for b in private['references'] if b.get('awaiting_week')):
                    raise RuntimeError('Closing-week evidence binding is incomplete: ' + revision)

    def _git_cited(self, evidence):
        if 'unknown' in evidence:
            return f"unknown ({evidence['unknown']})"
        if 'record' in evidence:
            return evidence['record']
        text = cite_text(evidence)
        return text + (" (this week's closing commit)" if evidence.get('commit') == week_label(self.sim_day()) else '')

    def _pf_cited(self, binding, evidence):
        shown = self._display_binding(binding)
        if 'record' in shown:
            return shown['record']
        if shown.get('status') == 'unknown':
            return f"unknown ({shown['reason']})"
        if shown.get('status') == 'commit_only':
            seen = f"; you saw {shown['delivered']}" if shown.get('delivered') else ''
            return f"{cite_text(evidence)} (commit only: those bytes never reached you{seen})"
        handles = evidence_handles.index(self.store)
        group = handles.group(binding['version_id'])
        text = shown['version'] + (f" (day {group['day']})" if group else '')
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

    def weekly_check(self, day):
        """Git/prefix week-start digest: cited committed files and texts compared with current ones."""
        records = self._load()['records']
        entries, ended = [], []
        for key in sorted(records, key=lambda k: int(k[1:]), reverse=True):
            record = records[key][-1]
            refs = [r for r in record['references'] if r['purpose'] == 'current' and 'unknown' not in r['evidence']]
            if record['status'] != 'active' or not refs:
                continue
            if applies_ended(record, day):
                ended.append(record['version'])
                continue
            changed = [line for line in (self._git_change(r['evidence'], records) for r in refs) if line]
            entries.append(dict(text=dict(record=record['version'], day=record.get('sim_day'), text=record['text']),
                                total=len(refs), changed=changed))
        return pf_render.render_weekly(entries, day, pf=False, ended=ended) if entries or ended else None

    def _git_change(self, evidence, records):
        if 'record' in evidence:
            requested = evidence['record']
            if self.identity and ':' in requested:
                owner, requested = requested.split(':', 1)
                peer = self.repository(owner)
                from .workspace_io import open_file
                with open_file(peer, peer / 'registrations.json') as stream:
                    records = json.load(stream)['records']
            latest = records[requested.split('.')[0]][-1]
            if latest['version'] == requested:
                return None
            return f"text {evidence['record']}: " + ('retired' if latest['status'] == 'retired' else
                                                      'revised to ' + latest['version'])
        workspace = self.repository(evidence.get('owner'))
        cited = (evidence.get('owner', '') + ':' if self.identity else '') + evidence['path'] + '@' + evidence['commit']
        try:
            _, full = git_reference(workspace, evidence, run=self.git_run)
            old = self.git_run(['git', '-C', str(workspace), 'show', f"{full}:{evidence['path']}"],
                                 capture_output=True, timeout=10, check=True).stdout
        except (ValueError, subprocess.SubprocessError) as exc:
            return f'{cited}: cannot check ({exc})'
        current = workspace / evidence['path']
        if current.is_symlink() or not current.is_file():
            return f'{cited}: the file no longer exists'
        from .workspace_io import open_file
        try:
            with open_file(workspace, current) as stream:
                new = stream.read()
        except (OSError, ValueError):
            return f'{cited}: the file no longer exists inside the workspace'
        if new == old:
            return None
        kind = 'json' if evidence['path'].endswith('.json') else 'text'
        return (f"{cited}: the current file differs ({pf_render.change(kind, old, new)}); "
                f"git diff {full[:7]} -- {evidence['path']}")

    def repository(self, owner=None):
        if not self.identity:
            return self.workspace
        from .role_policy import READABLE_ROLES
        owner = owner or self.role
        if owner not in READABLE_ROLES[self.role]:
            raise PermissionError('Evidence is inaccessible')
        return self.workspace.parent / owner

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
        full = label = None
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
                peer = self.repository(record_owner)
                with open_file(peer, peer / 'registrations.json') as stream:
                    record_store = json.load(stream)['records']
            history = record_store.get(requested.split('.')[0], [])
            target = (next((r for r in history if r['version'] == requested), None)
                      if '.' in requested else history[-1] if history else None)
            if target is None:
                raise ValueError('Unknown registered text revision; use unknown with a reason')
            evidence['record'] = (record_owner + ':' if self.identity else '') + target['version']
        elif self.mode in ('git', 'prefix') and 'path' in evidence and 'commit' not in evidence \
                and '@' not in evidence['path']:
            ref['evidence'] = weekly_reference(workspace, evidence['path'], week_label(self.sim_day()), run=self.git_run)
            label = ref['evidence']['commit']
            if self.identity:
                ref['evidence']['owner'] = owner
            if (ref.get('select') or ref.get('predicate')) and not evidence['path'].endswith(('.json', '.csv')):
                raise ValueError('Plain text supports whole-text equality only')
        elif self.mode in ('git', 'prefix') or (
                'path' in evidence and ('commit' in evidence or '@' in evidence['path'])):
            # PF keeps the Git group's committed-file references (design 4.1).
            if 'path' not in evidence:
                raise ValueError('Evidence must be a workspace file path or a registered text rN.M; otherwise use unknown with a reason')
            ref['evidence'], full = git_reference(workspace, evidence, run=self.git_run)
            if self.identity:
                ref['evidence']['owner'] = owner
            if (ref.get('select') or ref.get('predicate')) and not ref['evidence']['path'].endswith(('.json', '.csv')):
                raise ValueError('Plain text supports whole-text equality only')
        elif 'path' in evidence:
            path = Path(evidence['path'])
            if path.is_absolute() or '..' in path.parts or path.as_posix() != evidence['path']:
                raise ValueError('Evidence path must be workspace-relative')
        binding = (dict(status='git', git_commit=full) if full else
                   dict(status='git', git_week=label) if label else dict(status='registered_text', **evidence))
        if self.mode == 'git':
            return binding
        if label:
            binding['awaiting_week'] = label
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
        try:
            try:
                binding.update(self.resolver.resolve(ref['evidence'], ref, accept=accept), status='resolved')
                if full:
                    binding['git_content_matches'] = True
            except ValueError:
                if not full:
                    raise
                # No delivered capture has the committed bytes: record the last delivered
                # version and the mismatch, which traversal reports as a missing edge.
                binding.update(self.resolver.resolve(ref['evidence'], ref), status='resolved',
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

    def _list(self, state, after, limit, review=None):
        active = [versions[-1] for key, versions in sorted(state['records'].items(), key=lambda kv: int(kv[0][1:]))
                  if int(key[1:]) > after and versions[-1]['status'] == 'active']
        pending = (self.store.load_state('pf_review') or {}).get('pending', {}) if self.mode == 'pf' else {}
        checks = {}
        for record in active:
            binding = self.store.load_state('declaration:' + record['version']) if self.mode == 'pf' else None
            if binding and (check := pending.get(binding['version_id'])):
                checks[record['version']] = dict(status='pending', first_day=check['first_day'], last_day=check['last_day'],
                    reason=pf_render.dependency_line(check['reason'], relation=False))
        if review == 'pending':
            active = [record for record in active if record['version'] in checks]
        page = active[:limit]
        result, origins = '{"records":[', []
        for record in page:
            if result[-1] != '[':
                result += ','
            content = encoded(record).decode()
            if self.store:
                binding = self.store.load_state('declaration:' + record['version'])
                if binding and self.store.get_content(binding['version_id'])[1] == content.encode():
                    origins.append(origin(binding['version_id'], content, target=len(result)))
            result += content
        next_after = int(page[-1]['id'][1:]) if len(active) > limit else None
        result += '],"next_after":' + json.dumps(next_after) + '}'
        if self.mode == 'pf':
            result = result[:-1] + ',"checks":' + json.dumps({r['version']: checks.get(r['version']) for r in page}) + '}'
        if self.identity:
            result = result[:-1] + ',"shareable_references":' + json.dumps({
                r['version']: self.role + ':' + r['version'] for r in page}) + '}'
        return CapturedText(result, origins)
