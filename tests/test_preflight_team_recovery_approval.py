import copy
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from saas_bench import team_batch as batch
from saas_bench import team_recovery as recovery
from saas_bench.run_state import file_hash, tree_hash, write_json
from test_preflight_team_batch import frozen, managed_team


def bind_approval(monkeypatch, directory, manifest_path, attempts=None):
    directory.mkdir(parents=True, exist_ok=True)
    old = batch.read(manifest_path)
    actual = recovery.source_files(batch.ROOT)
    original = recovery.source_files(old['source_root'])
    controls = directory / 'controls.json'
    evidence = directory / 'evidence.json'
    write_json(controls, dict(offline=True))
    write_json(evidence, dict(accepted=True))
    record = dict(version=1, status='approved', manifest=str(manifest_path),
        manifest_sha256=file_hash(manifest_path), source_root=str(batch.ROOT),
        source_files=actual, source_sha256=tree_hash(batch.ROOT / 'src'),
        runtime_source_sha256=tree_hash(batch.ROOT / 'src/saas_bench'),
        legacy_runtime_source_sha256=tree_hash(Path(old['source_root']) / 'src/saas_bench'),
        changed_files=sorted(name for name in original.keys() | actual.keys()
            if original.get(name) != actual.get(name)),
        evidence={str(evidence): file_hash(evidence)}, control_freeze_file=str(controls),
        control_freeze_sha256=file_hash(controls), hold=str(directory / 'HOLD'),
        attempts=attempts or {})
    path = directory / 'approval.json'
    write_json(path, record)
    monkeypatch.setenv('CEOBENCH_RECOVERY_APPROVAL', str(path))
    monkeypatch.setenv('CEOBENCH_RECOVERY_APPROVAL_SHA256', file_hash(path))
    return path, record


@pytest.fixture
def approved_revision(tmp_path, monkeypatch):
    source = tmp_path / 'legacy'
    shutil.copytree(batch.ROOT / 'src', source / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    (source / 'scripts').mkdir()
    for name in ('team_batch.py', 'team_controller.py', 'team_observer.py', 'round5.py'):
        shutil.copyfile(batch.ROOT / 'scripts' / name, source / 'scripts' / name)
    host = source / 'src/saas_bench/run_lifecycle.py'
    host.write_text(host.read_text() + '\n')
    manifest_path = tmp_path / 'formal/manifest.json'
    write_json(manifest_path, dict(source_root=str(source), source_sha256=tree_hash(source / 'src')))
    (manifest_path.parent / 'HOLD').touch()
    path, record = bind_approval(monkeypatch, tmp_path / 'recovery', manifest_path)
    return path, record, manifest_path


def test_approval_accepts_bound_host_revision_and_preserves_historical_hold(approved_revision, monkeypatch):
    _, record, manifest = approved_revision
    assert recovery.approval(manifest) == record
    assert record['changed_files'] == ['src/saas_bench/run_lifecycle.py']
    assert recovery.execution_root(manifest, batch.read(manifest)) == batch.ROOT
    assert recovery.hold_path(manifest) == Path(record['hold'])
    assert (manifest.parent / 'HOLD').exists()
    monkeypatch.delenv('CEOBENCH_RECOVERY_APPROVAL')
    assert recovery.hold_path(manifest) == manifest.parent / 'HOLD'


@pytest.mark.parametrize('damage,expected', [
    ('fingerprint', 'fingerprint'), ('source_files', 'execution source'),
    ('runtime_source_sha256', 'runtime source'), ('controls', 'controls'),
    ('evidence', 'evidence'), ('hold', 'independent'), ('manifest', 'another manifest'),
    ('original_manifest', 'Original manifest'), ('source_root', 'outside approved revision'),
])
def test_approval_refuses_drift_before_execution(approved_revision, monkeypatch, tmp_path, damage, expected):
    path, record, manifest = approved_revision
    if damage == 'fingerprint':
        monkeypatch.setenv('CEOBENCH_RECOVERY_APPROVAL_SHA256', '0' * 64)
    elif damage == 'controls':
        Path(record['control_freeze_file']).write_text('{}')
    elif damage == 'evidence':
        Path(next(iter(record['evidence']))).write_text('{}')
    elif damage == 'hold':
        record['hold'] = str(manifest.parent / 'HOLD')
    elif damage == 'manifest':
        manifest = tmp_path / 'other/manifest.json'
    elif damage == 'original_manifest':
        manifest.write_text(manifest.read_text() + '\n')
    elif damage == 'source_root':
        record['source_root'] = str(tmp_path / 'unapproved-source')
    else:
        record[damage] = {} if damage == 'source_files' else '0' * 64
    if damage not in ('fingerprint', 'manifest'):
        write_json(path, record)
        monkeypatch.setenv('CEOBENCH_RECOVERY_APPROVAL_SHA256', file_hash(path))
    with pytest.raises(ValueError, match=expected):
        recovery.approval(manifest)


def test_approval_refuses_business_changes_even_when_fingerprints_are_rebound(approved_revision, monkeypatch):
    path, record, manifest_path = approved_revision
    old = batch.read(manifest_path)
    business = Path(old['source_root']) / 'src/saas_bench/multi_agent.py'
    business.write_text(business.read_text() + '\n')
    old['source_sha256'] = tree_hash(Path(old['source_root']) / 'src')
    write_json(manifest_path, old)
    record['manifest_sha256'] = file_hash(manifest_path)
    record['legacy_runtime_source_sha256'] = tree_hash(business.parent)
    record['changed_files'].append('src/saas_bench/multi_agent.py')
    write_json(path, record)
    monkeypatch.setenv('CEOBENCH_RECOVERY_APPROVAL_SHA256', file_hash(path))
    with pytest.raises(ValueError, match='business code'):
        recovery.approval(manifest_path)


@pytest.fixture
def approved_attempt(managed_team, tmp_path, monkeypatch):
    _, directory, previous, manifest, entry = managed_team
    manifest_path = directory / 'manifest.json'
    previous.update(status='running', error='historical stop')
    write_json(directory / 'attempt.json', previous)
    snapshot = Path(previous['checkpoint'])
    state = batch.read(snapshot / 'team.json')
    saved = dict(attempt=previous['attempt'], attempt_sha256=file_hash(directory / 'attempt.json'),
        checkpoint=str(snapshot), checkpoint_sha256=file_hash(snapshot / 'checkpoint.json'),
        day=state['day'], roles={name: role['identity'] for name, role in state['roles'].items()})
    path, revision = bind_approval(monkeypatch, tmp_path / 'recovery', manifest_path, {entry['run_id']: saved})
    return SimpleNamespace(directory=directory, previous=previous, manifest=manifest, entry=entry,
        manifest_path=manifest_path, snapshot=snapshot, state=state, approval_path=path, revision=revision)


def test_audited_registration_preserves_interruption_and_frozen_attempt_limit(approved_attempt):
    directory, manifest, entry = approved_attempt.directory, approved_attempt.manifest, approved_attempt.entry
    manifest_path, snapshot = approved_attempt.manifest_path, approved_attempt.snapshot
    before = (directory / 'attempt.json').read_bytes()
    capped = copy.deepcopy(manifest)
    capped['max_attempts'] = 1
    with pytest.raises(ValueError, match='finite attempt limit'):
        batch.register_attempt(manifest_path, capped, entry, resume=True)
    assert batch.attempts(entry) == [directory]
    assert (directory / 'attempt.json').read_bytes() == before
    next_directory, later, source = batch.register_attempt(manifest_path, manifest, entry, resume=True)
    assert later['attempt'] == 2 and source == snapshot
    assert later['source_checkpoint'] == str(snapshot)
    assert file_hash(next_directory / 'manifest.json') == file_hash(manifest_path)
    assert (directory / 'attempt.json').read_bytes() == before


@pytest.mark.parametrize('damage,expected', [
    ('attempt_checksum', 'Original attempt changed'),
    ('checkpoint_checksum', 'Approved checkpoint changed'),
    ('roles', 'role identity changed'), ('day', 'recoverable boundary'),
    ('run', 'belongs to another run'),
])
def test_recovery_refuses_unbound_attempt_snapshot_and_identity(approved_attempt, monkeypatch, damage, expected):
    approved = approved_attempt
    entry = copy.deepcopy(approved.entry)
    saved = approved.revision['attempts'][entry['run_id']]
    if damage == 'attempt_checksum':
        saved['attempt_sha256'] = '0' * 64
    elif damage == 'checkpoint_checksum':
        saved['checkpoint_sha256'] = '0' * 64
    elif damage == 'roles':
        saved['roles']['ceo']['session_id'] = 'unapproved-session'
    elif damage == 'day':
        saved['day'] += 7
    elif damage == 'run':
        approved.revision['attempts'] = {'another-run': saved}
        entry['run_id'] = 'another-run'
    write_json(approved.approval_path, approved.revision)
    monkeypatch.setenv('CEOBENCH_RECOVERY_APPROVAL_SHA256', file_hash(approved.approval_path))
    before = (approved.directory / 'attempt.json').read_bytes()
    with pytest.raises(ValueError, match=expected):
        recovery.recovery_source(approved.manifest_path, entry, approved.previous, approved.directory)
    assert (approved.directory / 'attempt.json').read_bytes() == before
    assert batch.attempts(approved.entry) == [approved.directory]


@pytest.mark.parametrize('status', ['completed', 'started'])
def test_recovery_refuses_operations_after_the_approved_checkpoint(approved_attempt, status):
    approved = approved_attempt
    write_json(approved.directory / 'runtime/private/operations/post-checkpoint.json', dict(status=status))
    before = (approved.directory / 'attempt.json').read_bytes()
    with pytest.raises(RuntimeError, match='unknown|changed'):
        recovery.recovery_source(approved.manifest_path, approved.entry, approved.previous, approved.directory)
    assert (approved.directory / 'attempt.json').read_bytes() == before
    assert batch.attempts(approved.entry) == [approved.directory]


def test_source_migration_is_limited_to_exact_approved_snapshot_and_revision(managed_team, tmp_path, monkeypatch):
    _, directory, previous, _, entry = managed_team
    snapshot = Path(previous['checkpoint'])
    state = batch.read(snapshot / 'team.json')
    path, record = bind_approval(monkeypatch, tmp_path / 'recovery', directory / 'manifest.json',
        {entry['run_id']: dict(checkpoint=str(snapshot), checkpoint_sha256=file_hash(snapshot / 'checkpoint.json'))})
    assert recovery.source_revision_allowed(snapshot, state)
    assert not recovery.source_revision_allowed(tmp_path / 'unapproved-snapshot', state)
    assert not recovery.source_revision_allowed(snapshot, dict(state, source_sha256='0' * 64))
    record['attempts'][entry['run_id']]['checkpoint_sha256'] = '0' * 64
    write_json(path, record)
    monkeypatch.setenv('CEOBENCH_RECOVERY_APPROVAL_SHA256', file_hash(path))
    assert not recovery.source_revision_allowed(snapshot, state)
