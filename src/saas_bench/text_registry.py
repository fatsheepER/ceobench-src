"""Workspace declarations with private evidence bindings kept outside the workspace."""
from copy import deepcopy
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
        record.update(version=f"{record_id}.{record['revision']}", sim_day=self.sim_day(), author='ceo')
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
        if shown and cited:
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
        group = evidence_handles.index(self.store).group(binding['version_id'])
        text = shown['version'] + (f" (day {group['day']})" if group else '')
        if shown['differs']:
            text += f" (as you last saw it; now {shown['latest']}, {self._change(binding)})"
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
            latest = records[evidence['record'].split('.')[0]][-1]
            if latest['version'] == evidence['record']:
                return None
            return f"text {evidence['record']}: " + ('retired' if latest['status'] == 'retired' else
                                                      'revised to ' + latest['version'])
        cited = evidence['path'] + '@' + evidence['commit']
        try:
            _, full = git_reference(self.workspace, evidence)
            old = subprocess.run(['git', '-C', str(self.workspace), 'show', f"{full}:{evidence['path']}"],
                                 capture_output=True, timeout=10, check=True).stdout
        except (ValueError, subprocess.SubprocessError) as exc:
            return f'{cited}: cannot check ({exc})'
        current = self.workspace / evidence['path']
        if current.is_symlink() or not current.is_file():
            return f'{cited}: the file no longer exists'
        new = current.read_bytes()
        if new == old:
            return None
        kind = 'json' if evidence['path'].endswith('.json') else 'text'
        return (f"{cited}: the current file differs ({pf_render.change(kind, old, new)}); "
                f"git diff {full[:7]} -- {evidence['path']}")

    def _bind(self, ref, records):
        evidence = ref['evidence']
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
            history = records.get(requested.split('.')[0], [])
            target = (next((r for r in history if r['version'] == requested), None)
                      if '.' in requested else history[-1] if history else None)
            if target is None:
                raise ValueError('Unknown registered text revision; use unknown with a reason')
            evidence['record'] = target['version']
        elif self.mode in ('git', 'prefix') and 'path' in evidence and 'commit' not in evidence \
                and '@' not in evidence['path']:
            ref['evidence'] = weekly_reference(self.workspace, evidence['path'], week_label(self.sim_day()))
            label = ref['evidence']['commit']
            if (ref.get('select') or ref.get('predicate')) and not evidence['path'].endswith(('.json', '.csv')):
                raise ValueError('Plain text supports whole-text equality only')
        elif self.mode in ('git', 'prefix') or (
                'path' in evidence and ('commit' in evidence or '@' in evidence['path'])):
            # PF keeps the Git group's committed-file references (design 4.1).
            if 'path' not in evidence:
                raise ValueError('Evidence must be a workspace file path or a registered text rN.M; otherwise use unknown with a reason')
            ref['evidence'], full = git_reference(self.workspace, evidence)
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
        version, latest = (self.resolver.handle(binding[k]) for k in ('version_id', 'latest_version_id'))
        return dict(shown, version=version, latest=latest, differs=version != latest)

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
