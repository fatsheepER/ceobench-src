"""Explicit, fingerprint-bound host revision and recovery approval.

The original manifest, HOLD and interrupted attempt are never rewritten.
"""
import json
import os
from pathlib import Path

from .run_state import file_hash, tree_hash

ALLOWED = {'src/saas_bench/run_lifecycle.py', 'src/saas_bench/team_batch.py',
    'src/saas_bench/team_checkpoint.py', 'src/saas_bench/team_recovery.py',
    'scripts/team_controller.py', 'scripts/team_observer.py'}


def source_files(root):
    root = Path(root)
    paths = [*sorted((root / 'src').rglob('*.py')),
        *(root / 'scripts' / name for name in ('team_batch.py', 'team_controller.py', 'team_observer.py', 'round5.py'))]
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


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
    original = source_files(old['source_root'])
    changed = {name for name in original.keys() | actual.keys() if original.get(name) != actual.get(name)}
    if changed != set(record['changed_files']) or not changed <= ALLOWED:
        raise ValueError('Recovery revision changes business code or unapproved files')
    if tree_hash(Path(old['source_root']) / 'src') != old['source_sha256']:
        raise ValueError('Original source fingerprint changed')
    if tree_hash(new_root / 'src/saas_bench') != record['runtime_source_sha256']:
        raise ValueError('Approved runtime source changed')
    for filename, checksum in record['evidence'].items():
        if file_hash(filename) != checksum:
            raise ValueError('Recovery evidence changed: ' + filename)
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
