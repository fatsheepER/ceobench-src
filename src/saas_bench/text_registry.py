"""Workspace declarations with private evidence bindings kept outside the workspace."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

from pydantic import ValidationError

from .execution_capture import CapturedText, CURRENT_EVENT, origin
from .registration_evidence import EvidenceResolver, git_reference
from .registration_schema import MODELS
from .run_state import write_json
from .sql_evidence import encoded, now


class TextRegistry:
    def __init__(self, workspace, mode, store=None, sim_day=lambda: None):
        if mode not in ('git', 'prefix', 'pf'):
            raise ValueError('Invalid registration mode')
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
        value = json.loads(self.path.read_text())
        if value.get('format') != 'ceobench.text-register.v1' or not isinstance(value.get('records'), dict):
            raise ValueError('Invalid registration file')
        return value

    def execute(self, operation, args):
        try:
            values = MODELS[self.mode][operation].model_validate(args).model_dump(exclude_none=True)
        except ValidationError as exc:
            # Do not echo the submitted value: it can contain long private identifiers.
            errors = ['.'.join(map(str, e['loc'])) + ': ' + e['msg'] for e in exc.errors(include_input=False)]
            raise ValueError('; '.join(errors)) from exc
        state = self._load()
        if operation == 'list':
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
        record.update(version=f"{record_id}.{record['revision']}",
                      registered_at=now(), sim_day=self.sim_day(), author='ceo')
        warnings, bindings = [], []
        if operation == 'create' or 'references' in values:
            for ref in record['references']:
                if len(ref.get('note', '')) > 200:
                    ref['note'] = ref['note'][:200]
                    warnings.append('备注已截至 200 字')
                bindings.append(self._bind(ref, records))
        elif self.store:
            binding = self.store.load_state('declaration:' + previous['version'])
            bindings = binding['references'] if binding else []
        records.setdefault(record_id, []).append(record)
        # Validate every reference before touching the workspace. A failed declaration
        # does not consume a revision or save partially validated references.
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
                self.store.version(event, 'declaration', encoded(private), layer='agent_declaration',
                                   object_id='declaration:' + record['version'])
                self.store.save_state('declaration:' + record['version'], private)
                if owned:
                    self.store.complete(event)
            except Exception as exc:
                self.store.fail(exc)
                raise RuntimeError('Text saved, but private capture failed; collection stopped') from exc
        result = dict(id=record_id, version=record['version'], status=record['status'])
        if warnings:
            result['warnings'] = list(dict.fromkeys(warnings))
        if self.mode == 'pf':
            result['evidence'] = [self._display_binding(b) for b in bindings]
        return encoded(result).decode()

    def _bind(self, ref, records):
        evidence = ref['evidence']
        if 'unknown' in evidence:
            return dict(status='unknown', reason=evidence['unknown'])
        full = None
        if 'record' in evidence:
            requested = evidence['record']
            history = records.get(requested.split('.')[0], [])
            target = (next((r for r in history if r['version'] == requested), None)
                      if '.' in requested else history[-1] if history else None)
            if target is None:
                raise ValueError('Unknown registered text revision; use unknown with a reason')
            evidence['record'] = target['version']
        elif self.mode in ('git', 'prefix') or (
                'path' in evidence and ('commit' in evidence or '@' in evidence['path'])):
            # PF keeps the Git group's committed-file references (design 4.1).
            if 'path' not in evidence:
                raise ValueError('Git references require a committed path, or unknown with a reason')
            ref['evidence'], full = git_reference(self.workspace, evidence)
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
            blob = subprocess.check_output(['git', '-C', str(self.workspace), 'rev-parse',
                full + ':' + ref['evidence']['path']], text=True).strip()
            def accept(version):
                raw = self.store.get_content(version)[1]
                return blob == subprocess.check_output(['git', '-C', str(self.workspace), 'hash-object', '--stdin'],
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
        return dict(shown, version=self.resolver.handle(binding['version_id']),
                    latest=self.resolver.handle(binding['latest_version_id']),
                    differs=binding['version_id'] != binding['latest_version_id'])

    def _list(self, state, after, limit):
        active = [versions[-1] for key, versions in sorted(state['records'].items(), key=lambda kv: int(kv[0][1:]))
                  if int(key[1:]) > after and versions[-1]['status'] == 'active']
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
        return CapturedText(result, origins)
