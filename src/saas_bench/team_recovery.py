"""Explicit, fingerprint-bound host revision and recovery approval.

The original manifest, HOLD and interrupted attempt are never rewritten.
"""
import json
import os
from pathlib import Path
import ast
import hashlib
import subprocess

from .run_state import file_hash, tree_hash

ALLOWED = {'src/saas_bench/run_lifecycle.py', 'src/saas_bench/team_batch.py',
    'src/saas_bench/team_checkpoint.py', 'src/saas_bench/team_recovery.py',
    'src/saas_bench/api_server.py', 'src/saas_bench/sql_evidence.py',
    'src/saas_bench/multi_agent.py', 'src/saas_bench/run_state.py',
    'src/saas_bench/process_boundary.py', 'src/saas_bench/execution_capture.py',
    'src/saas_bench/agents/bash_agent/tools.py', 'src/saas_bench/pf_deadline.py',
    'scripts/team_batch.py', 'scripts/team_controller.py', 'scripts/team_observer.py',
    'scripts/team_recovery.py', 'scripts/team_driver.py'}

REPAIR_BODIES = {
    'src/saas_bench/api_server.py': {'_APIHandler.handle_one_request', '_APIHandler.parse_request',
        '_APIHandler.end_headers',
        '_APIHandler._expire_request_input', '_APIHandler._start_public_request', '_APIHandler._read_request_bytes',
        '_APIHandler._capture_sql', '_APIHandler._capture_response', '_APIHandler._capture_delivery',
        '_TrackedHTTPServer.get_request', '_TrackedHTTPServer.shutdown_request', 'registered_script_classification',
        'NovaMindAPIServer.start', 'NovaMindAPIServer.stop', 'NovaMindAPIServer.public_request_state',
        'NovaMindAPIServer.checkpoint'},
    'src/saas_bench/agents/bash_agent/tools.py': {'agent_runtime', 'validate_bash_limits',
        'BashAgentToolExecutor.verify_sandbox', 'BashAgentToolExecutor._preserve_process'},
    'src/saas_bench/multi_agent.py': {'MultiAgentRuntime.checkpoint', 'MultiAgentRuntime.restore',
        'MultiAgentRuntime._begin_operation', 'MultiAgentRuntime._workspace_fingerprint',
        'MultiAgentRuntime._finish_operation', 'MultiAgentRuntime._guard_business_operations',
        'MultiAgentRuntime.request_pause', 'MultiAgentRuntime._check', 'MultiAgentRuntime._check_model_send',
        'MultiAgentRuntime.close'},
    'src/saas_bench/team_batch.py': {'close_team', 'wall_time', 'mark_interrupted', 'register_attempt',
        'prune_checkpoints', 'save_checkpoint'},
    'src/saas_bench/execution_capture.py': {'sql_client_interruption', 'capture_http', 'finish_http',
        'public_handler', 'receive_client', 'restore_sources', 'model_request', '_record_model_request'},
    'scripts/team_controller.py': {'scope_state', 'signal_scope', 'run_disposition',
        'RequestGate.stop', 'RequestGate.check_stop', 'RequestGate.__call__', 'RequestGate.response'},
    'scripts/team_observer.py': {'_capture_logs', 'poll'},
    'scripts/team_driver.py': set(),
    'scripts/team_batch.py': set(),
}


def without_import_cache(node):
    class CacheHygiene(ast.NodeTransformer):
        def visit_Assign(self, node):
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'env'
                    and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == 'dict'):
                node.value.keywords = [keyword for keyword in node.value.keywords
                    if not (keyword.arg == 'PYTHONDONTWRITEBYTECODE' and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value == '1')]
            return node
    return CacheHygiene().visit(node)


def source_files(root):
    root = Path(root)
    paths = [path for directory in ('src', 'scripts') for path in (root / directory).rglob('*')
             if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'
             and path.name != '.env' and not path.name.startswith('.env.')]
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def design_files(root):
    root = Path(root)
    paths = [path for directory in ('public/docs', 'public_sources', 'src', 'scripts')
             for path in (root / directory).rglob('*')
             if path.is_file() and path.suffix in ('.md', '.json', '.yaml', '.yml', '.toml', '.txt', '.py')
             and '__pycache__' not in path.parts and
             (directory not in ('src', 'scripts') or path.suffix != '.py')]
    if (root / 'public/novamind-client').exists():
        paths.append(root / 'public/novamind-client')
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def declared_fix(record, old, actual):
    classification = record.get('fix_classification')
    if classification not in ('harness_fix', 'design_change'):
        raise ValueError('Recovery requires explicit fix_classification')
    if classification == 'design_change':
        raise ValueError('design_change requires a new frozen batch')
    if not isinstance(record.get('reason'), str) or not record['reason'].strip():
        raise ValueError('Harness fix requires a reason')
    original = source_files(old['source_root'])
    changed = {name for name in original.keys() | actual.keys() if original.get(name) != actual.get(name)}
    if (not isinstance(record.get('affected_files'), list) or
            changed != set(record.get('changed_files', [])) or changed != set(record['affected_files'])):
        raise ValueError('Declared affected files differ from the actual revision')
    if not changed <= ALLOWED:
        raise ValueError('Recovery revision changes business code, prompts, role policy, PF help or configuration')
    new_root = Path(record['source_root'])
    if design_files(old['source_root']) != design_files(new_root):
        raise ValueError('harness_fix cannot change PF help, public client or experiment configuration')
    protected = {
        'src/saas_bench/team_batch.py': {'PARAMETERS', 'MODEL', 'ENDPOINT', 'configuration', 'prompt_templates'},
        'src/saas_bench/agents/bash_agent/tools.py': {'BASH_AGENT_TOOL_DEFS', 'NOTE_PARAMETER', 'get_bash_agent_tool_descriptions'},
        'src/saas_bench/multi_agent.py': {'__init__', '_rotate_sessions', '_observation', '_handoff',
            '_message', '_ask_batch', '_run_analyst', '_parallel', 'run'},
    }
    for name, symbols in protected.items():
        if name not in changed:
            continue
        def definitions(root):
            tree = ast.parse((Path(root) / name).read_text())
            return {symbol: ast.dump(without_import_cache(node) if name == 'src/saas_bench/multi_agent.py'
                    and symbol == '__init__' else node, include_attributes=False) for node in ast.walk(tree)
                    for symbol in ([node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else
                        [target.id for target in node.targets if isinstance(target, ast.Name)]
                        if isinstance(node, ast.Assign) else []) if symbol in symbols}
        if definitions(old['source_root']) != definitions(new_root):
            raise ValueError('harness_fix cannot change frozen settings, prompts or scheduling design')
    if 'src/saas_bench/multi_agent.py' in changed:
        class WithoutOperationJournal(ast.NodeTransformer):
            def visit_Assign(self, node):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == '_begin_operation':
                    return None
                return self.generic_visit(node)

            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == '_finish_operation':
                    return None
                return self.generic_visit(node)

            def visit_If(self, node):
                self.generic_visit(node)
                return node if node.body else None

            def visit_Try(self, node):
                had_finally = bool(node.finalbody)
                self.generic_visit(node)
                if not node.handlers and not node.orelse and had_finally and not node.finalbody:
                    return node.body
                return node

        def ceo_execution(root):
            tree = ast.parse((Path(root) / 'src/saas_bench/multi_agent.py').read_text())
            return {node.name: ast.dump(WithoutOperationJournal().visit(node), include_attributes=False)
                    for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in ('_run_ceo', '_script_output')}
        if ceo_execution(old['source_root']) != ceo_execution(new_root):
            raise ValueError('harness_fix cannot change CEO model inputs or tool execution')
        def model_inputs(root):
            tree = ast.parse((Path(root) / 'src/saas_bench/multi_agent.py').read_text())
            return [ast.dump(node, include_attributes=False) for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and
                    (isinstance(node.func, ast.Name) and node.func.id in ('BashAgent', 'system_prompt', 'get_bash_agent_tool_descriptions') or
                     isinstance(node.func, ast.Attribute) and node.func.attr in ('act', 'record_tool_result'))]
        if model_inputs(old['source_root']) != model_inputs(new_root):
            raise ValueError('harness_fix cannot change role model inputs')
    for name in changed & REPAIR_BODIES.keys():
        def design(root):
            class RepairBodies(ast.NodeTransformer):
                def __init__(self):
                    self.classes = []

                def visit_ClassDef(self, node):
                    self.classes.append(node.name)
                    self.generic_visit(node)
                    self.classes.pop()
                    return node

                def visit_FunctionDef(self, node):
                    symbol = '.'.join([*self.classes, node.name])
                    if symbol in REPAIR_BODIES[name]:
                        node.body = [ast.Pass()]
                    elif name == 'src/saas_bench/multi_agent.py' and symbol == 'MultiAgentRuntime.__init__':
                        node = without_import_cache(node)
                    elif name == 'src/saas_bench/multi_agent.py' and symbol in (
                            'MultiAgentRuntime._run_ceo', 'MultiAgentRuntime._script_output'):
                        node = WithoutOperationJournal().visit(node)
                    elif name == 'src/saas_bench/api_server.py' and symbol == '_APIHandler._send_json':
                        for handler in ast.walk(node):
                            if isinstance(handler, ast.ExceptHandler) and isinstance(handler.type, ast.Name) and handler.type.id == 'OSError':
                                handler.body = [ast.Pass()]
                    return node

            return ast.dump(RepairBodies().visit(ast.parse((Path(root) / name).read_text())), include_attributes=False)
        if design(old['source_root']) != design(new_root):
            raise ValueError('harness_fix cannot change model inputs, tool execution or frozen scheduling design')
    verify_fix_commit(new_root, record.get('fix_commit'), changed, actual)
    return changed


def verify_fix_commit(root, commit, changed, actual):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if not isinstance(commit, str) or commit != head:
        raise ValueError('Repair fix_commit must identify the execution source HEAD')
    for name in actual.keys() | changed:
        result = subprocess.run(['git', 'show', commit + ':' + name], cwd=root, capture_output=True)
        checksum = hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None
        if checksum != actual.get(name):
            raise ValueError('Repair commit does not contain the approved file: ' + name)


def create_approval(manifest_path, destination, declaration, *, source_root=None, approved=False):
    """Bind an explicit repair declaration without changing the interrupted batch."""
    from .team_batch import ROOT, attempts, read, selected
    from .run_state import write_json
    manifest_path, destination = Path(manifest_path).resolve(), Path(destination).resolve()
    old = read(manifest_path)
    root = Path(source_root or ROOT).resolve()
    actual, original = source_files(root), source_files(old['source_root'])
    changed = sorted(name for name in original.keys() | actual.keys() if original.get(name) != actual.get(name))
    evidence = {}
    for filename in declaration.get('evidence', []):
        filename = Path(filename).resolve()
        if filename.name == '.env' or filename.name.startswith('.env.'):
            raise ValueError('Credentials cannot be recovery evidence')
        evidence[str(filename)] = file_hash(filename)
    if not evidence:
        raise ValueError('Recovery approval requires evidence')
    record = dict(version=1, status='approved' if approved else 'pending',
        manifest=str(manifest_path), manifest_sha256=file_hash(manifest_path),
        source_root=str(root), source_files=actual, source_sha256=tree_hash(root / 'src'),
        runtime_source_sha256=tree_hash(root / 'src/saas_bench'),
        legacy_runtime_source_sha256=tree_hash(Path(old['source_root']) / 'src/saas_bench'),
        changed_files=changed, affected_files=declaration.get('affected_files'),
        fix_classification=declaration.get('fix_classification'), fix_commit=declaration.get('fix_commit'),
        reason=declaration.get('reason'), evidence=evidence,
        hold=str(destination / 'HOLD'), attempts={}, decisions=[])
    declared_fix(record, old, actual)
    controls = declaration.get('control_freeze_file')
    if controls:
        controls = Path(controls).resolve()
        record.update(control_freeze_file=str(controls), control_freeze_sha256=file_hash(controls))
    for run_id, overrides in declaration.get('attempts', {}).items():
        entry = selected(old, run_id)
        history = attempts(entry)
        if not history:
            raise ValueError('Approved recovery requires an existing attempt: ' + run_id)
        directory = history[-1]
        previous = read(directory / 'attempt.json')
        from .team_checkpoint import validate_snapshot
        snapshot = Path(overrides.get('checkpoint', previous['checkpoint'])).resolve()
        owner = next((attempt for attempt in history
            if snapshot.is_relative_to(attempt / 'runtime/checkpoints')), None)
        if owner is None or file_hash(owner / 'manifest.json') != file_hash(manifest_path):
            raise ValueError('Approved checkpoint belongs to another run or manifest')
        snapshot, state = validate_snapshot(snapshot)
        if (state['stage'] not in ('boundary', 'ceo') or
                any(role['identity']['run_id'] != run_id for role in state['roles'].values())):
            raise ValueError('Approved checkpoint is outside this run recoverable boundary')
        record['attempts'][run_id] = dict(attempt=previous['attempt'],
            attempt_id=previous['attempt_id'],
            attempt_sha256=file_hash(directory / 'attempt.json'), checkpoint=str(snapshot),
            checkpoint_sha256=file_hash(snapshot / 'checkpoint.json'), day=state['day'],
            roles={name: role['identity'] for name, role in state['roles'].items()},
            residual_operations=overrides.get('residual_operations', []),
            allow_unknown_business_write=overrides.get('allow_unknown_business_write', False))
        record['decisions'].append(dict(kind='recovery_checkpoint', run_id=run_id,
            seed=entry['seed'], repeat=entry['repeat'], group=entry['group'],
            checkpoint=str(snapshot), day=state['day'], stage=state['stage'],
            week_boundary=state['stage'] == state['phase'] == 'boundary' and
                state['day'] % 7 == 0 and state['week'] == {} and state['failure'] is None))
    points = list(record['decisions'])
    pairs = {(point['seed'], point['repeat']) for point in points}
    for seed, repeat in sorted(pairs):
        pair = [point for point in points if (point['seed'], point['repeat']) == (seed, repeat)]
        aligned = (len(pair) == 2 and {point['group'] for point in pair} == {'git', 'pf'} and
            len({point['day'] for point in pair}) == 1 and all(point['week_boundary'] for point in pair))
        record['decisions'].append(dict(kind='paired_recovery_alignment', seed=seed, repeat=repeat,
            alignment='same_week_boundary' if aligned else 'independent_checkpoints',
            checkpoints={point['group']: dict(checkpoint=point['checkpoint'], day=point['day'],
                stage=point['stage']) for point in pair}))
    validate_residual_declarations(record)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / 'approval.json'
    write_json(path, record)
    path.chmod(0o444)
    return dict(path=str(path), sha256=file_hash(path), status=record['status'])


def validate_residual_declarations(record):
    for run_id, saved in record['attempts'].items():
        seen = set()
        for item in saved.get('residual_operations', []):
            classification = item.get('classification')
            if classification not in ('read_only_interrupted', 'pending_read_only', 'unknown_business_write'):
                raise ValueError('Residual operation requires an explicit classification')
            if classification == 'unknown_business_write' and saved.get('allow_unknown_business_write') is not True:
                raise ValueError('Unknown business write requires explicit approval')
            if not isinstance(item.get('reason'), str) or not item['reason'].strip() or not item.get('evidence'):
                raise ValueError('Residual operation requires a reason and evidence')
            key = item.get('root'), item.get('operation')
            if key in seen:
                raise ValueError('Duplicate residual operation declaration')
            seen.add(key)
            for filename, checksum in item['evidence'].items():
                path = Path(filename)
                if not path.is_absolute() or path.name == '.env' or path.name.startswith('.env.'):
                    raise ValueError('Residual evidence must be an absolute noncredential path')
                if file_hash(path) != checksum:
                    raise ValueError('Residual recovery evidence changed')
            resource = item.get('resource_evidence')
            if resource not in item['evidence']:
                raise ValueError('Residual recovery requires bound worker resource evidence')
            resources = json.loads(Path(resource).read_text())
            unit = resources.get('units', {}).get(run_id, {})
            if resources.get('children_reaped') is not True or unit.get('scope_empty') is not True or unit.get('cleanup_error'):
                raise ValueError('Residual recovery requires proven worker scope closure')
            bound = resources.get('attempts', {}).get(run_id, {})
            if (not saved.get('attempt_id') or bound != {key: saved[key] for key in ('attempt', 'attempt_id', 'attempt_sha256')} or
                    not all(isinstance(unit.get(key), str) and unit[key] for key in ('unit', 'invocation_id', 'control_group')) or
                    item.get('worker_unit') != unit['unit'] or item.get('worker_invocation_id') != unit['invocation_id']):
                raise ValueError('Residual resource evidence must bind the original attempt and worker invocation')
            if classification in ('read_only_interrupted', 'pending_read_only'):
                from .team_checkpoint import recovery_operation_safe
                root = Path(item['root'])
                operation = root / 'private/operations' / item['operation']
                if (not root.is_absolute() or root.resolve() != root or Path(item['operation']).name != item['operation'] or
                        operation.is_symlink() or file_hash(operation) != item['sha256']):
                    raise ValueError('Approved residual operation changed')
                if not recovery_operation_safe(root, json.loads(operation.read_text())):
                    raise ValueError('Read-only residual approval requires positive operation-bound proof')


def approval(manifest_path=None):
    value = os.environ.get('CEOBENCH_RECOVERY_APPROVAL')
    if not value:
        return None
    path = Path(value).resolve()
    if file_hash(path) != os.environ.get('CEOBENCH_RECOVERY_APPROVAL_SHA256'):
        raise ValueError('Recovery approval fingerprint changed')
    record = json.loads(path.read_text())
    if record.get('version') != 1 or record.get('status') != 'approved':
        raise ValueError('Recovery approval is not approved')
    if record.get('control_freeze_file') and file_hash(record['control_freeze_file']) != record['control_freeze_sha256']:
        raise ValueError('Approved recovery controls changed')
    manifest = Path(record['manifest'])
    if manifest_path is not None and Path(manifest_path).resolve() != manifest:
        raise ValueError('Recovery approval belongs to another manifest')
    if file_hash(manifest) != record['manifest_sha256']:
        raise ValueError('Original manifest changed after approval')
    old = json.loads(manifest.read_text())
    new_root = Path(record['source_root'])
    if Path(__file__).resolve().parents[2] != new_root:
        raise ValueError('Executing source is outside approved revision')
    actual = source_files(new_root)
    if actual != record['source_files'] or tree_hash(new_root / 'src') != record['source_sha256']:
        raise ValueError('Approved execution source changed')
    declared_fix(record, old, actual)
    if tree_hash(Path(old['source_root']) / 'src') != old['source_sha256']:
        raise ValueError('Original source fingerprint changed')
    if tree_hash(new_root / 'src/saas_bench') != record['runtime_source_sha256']:
        raise ValueError('Approved runtime source changed')
    for filename, checksum in record['evidence'].items():
        if file_hash(filename) != checksum:
            raise ValueError('Recovery evidence changed: ' + filename)
    validate_residual_declarations(record)
    hold = Path(record['hold'])
    historical = manifest.parent / 'HOLD'
    if (hold == historical or not hold.is_absolute() or hold.resolve() != hold or
            not hold.is_relative_to(path.parent)):
        raise ValueError('Recovery HOLD must be independent and inside approval directory')
    if record['legacy_runtime_source_sha256'] != tree_hash(Path(old['source_root']) / 'src/saas_bench'):
        raise ValueError('Legacy runtime source changed')
    return record


def execution_root(manifest_path, manifest):
    record = approval(manifest_path)
    return Path(record['source_root'] if record else manifest['source_root'])


def hold_path(manifest_path):
    record = approval(manifest_path)
    return Path(record['hold']) if record else Path(manifest_path).parent / 'HOLD'


def recovery_source(manifest_path, entry, previous, directory):
    record = approval(manifest_path)
    if not record:
        return None
    saved = record['attempts'].get(entry['run_id'])
    if not saved or previous['attempt'] != saved['attempt']:
        return None
    if file_hash(Path(directory) / 'attempt.json') != saved['attempt_sha256']:
        raise ValueError('Original attempt changed after recovery approval')
    from .team_checkpoint import validate_snapshot, validate_recovery_source
    snapshot, state = validate_snapshot(saved['checkpoint'])
    if file_hash(snapshot / 'checkpoint.json') != saved['checkpoint_sha256']:
        raise ValueError('Approved checkpoint changed')
    if state['stage'] not in ('boundary', 'ceo') or state['day'] != saved['day']:
        raise ValueError('Approved checkpoint is outside recoverable boundary')
    if {name: role['identity'] for name, role in state['roles'].items()} != saved['roles']:
        raise ValueError('Approved role identity changed')
    if any(role['identity']['run_id'] != entry['run_id'] for role in state['roles'].values()):
        raise ValueError('Approved checkpoint belongs to another run')
    validate_recovery_source(snapshot, state, current_root=Path(directory) / 'runtime')
    return snapshot


def source_revision_allowed(snapshot, state):
    record = approval()
    if not record or state['source_sha256'] != record['legacy_runtime_source_sha256']:
        return False
    return any(str(Path(snapshot).resolve()) == saved['checkpoint'] and
        file_hash(Path(snapshot) / 'checkpoint.json') == saved['checkpoint_sha256']
        for saved in record['attempts'].values())
