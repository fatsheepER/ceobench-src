from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from saas_bench import team_batch as batch, team_recovery as recovery
from saas_bench.run_state import file_hash, tree_hash, write_json
from test_preflight_team_batch import frozen
from test_preflight_team_checkpoint import fast_simulator
from test_preflight_team_flow import raw
from test_preflight_team_readonly_recovery import managed_git
from test_preflight_team_recovery_approval import bind_approval


@pytest.mark.parametrize('managed_git', ['git', 'pf'], indirect=True)
@pytest.mark.parametrize('business', ['none', 'unknown', 'completed'])
def test_approved_recovery_rolls_back_role_tail_and_keeps_business_guard(
        managed_git, tmp_path, monkeypatch, business):
    team, directory, previous, manifest, entry = managed_git
    workspace = team.roles['ceo'].workspace
    (workspace / 'saved.txt').write_text('checkpoint content\n')
    (workspace / 'saved.txt').chmod(0o640)
    (workspace / 'saved-link').symlink_to('saved.txt')
    snapshot = batch.save_checkpoint(team, directory, previous, 'ready')
    state = batch.read(snapshot / 'team.json')
    team.roles['ceo'].executor.execute('bash', dict(command=
        'printf tail > saved.txt; chmod 600 saved.txt; rm saved-link; '
        'ln -s MEMORY.md saved-link; mkdir tail-directory; printf tail > tail.txt'))
    if business != 'none':
        team._set_phase('ceo')
        if business == 'unknown':
            set_prices = team.server.tools.set_prices
            def interrupted_write(args):
                set_prices(args)
                raise ConnectionError('Business write completed but its response was lost')
            monkeypatch.setattr(team.server.tools, 'set_prices', interrupted_write)
        status, response = raw(team, 'ceo', '/call', dict(tool='set_prices', args={'A': 37}))
        assert status == (500 if business == 'unknown' else 200), response
        assert team.server.conn.execute('SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] == 37
    team.close()
    saved_at = datetime.fromisoformat(state['saved_at']) if state.get('saved_at') else datetime.fromtimestamp(
        (snapshot / 'team.json').stat().st_mtime, timezone.utc)
    previous.update(status='paused', finished_at=(saved_at + timedelta(seconds=90)).isoformat())
    write_json(directory / 'attempt.json', previous)
    saved = dict(attempt=previous['attempt'], attempt_id=previous['attempt_id'],
        attempt_sha256=file_hash(directory / 'attempt.json'), checkpoint=str(snapshot),
        checkpoint_sha256=file_hash(snapshot / 'checkpoint.json'), day=state['day'],
        roles={name: role['identity'] for name, role in state['roles'].items()})
    path = tmp_path / 'manifest.json'
    bind_approval(monkeypatch, tmp_path / 'approval', path, {entry['run_id']: saved})
    original = {p: p.read_bytes() for p in
        (directory / 'attempt.json', snapshot / 'checkpoint.json', workspace / 'saved.txt')}
    if business != 'none':
        with pytest.raises(RuntimeError, match='Business operation'):
            recovery.recovery_source(path, entry, previous, directory)
        with pytest.raises(RuntimeError, match='Business operation'):
            batch.prepare(path, entry['run_id'], resume=True, simulator_factory=fast_simulator)
        assert batch.attempts(entry) == [directory]
    else:
        assert recovery.recovery_source(path, entry, previous, directory) == snapshot
        resumed = batch.prepare(path, entry['run_id'], resume=True, simulator_factory=fast_simulator)
        root = batch.attempts(entry)[-1] / 'runtime'
        assert resumed['status'] == 'ready' and resumed['attempt'] == 2
        assert resumed['source_checkpoint'] == str(snapshot)
        rollback = resumed['checkpoint_rollback']
        assert rollback['checkpoint'] == str(snapshot) and rollback['day'] == 0
        assert rollback['discarded_wall_seconds'] == 90
        assert rollback['checkpoint_at_estimated'] is (not bool(state.get('saved_at')))
        assert resumed['decisions'][-1]['kind'] == 'checkpoint_rollback'
        assert tree_hash(root / 'roles', exact=True) == tree_hash(snapshot / 'roles', exact=True)
        assert (root / 'roles/ceo/saved.txt').read_text() == 'checkpoint content\n'
        assert (root / 'roles/ceo/saved.txt').stat().st_mode == (snapshot / 'roles/ceo/saved.txt').stat().st_mode
        assert (root / 'roles/ceo/saved-link').readlink() == Path('saved.txt')
        assert not (root / 'roles/ceo/tail.txt').exists()
        assert not (root / 'roles/ceo/tail-directory').exists()
        assert (root / 'roles/ceo/.git').is_dir()
    assert {p: p.read_bytes() for p in original} == original


def test_approval_selects_shared_week_boundary_without_rewriting_attempts(frozen, tmp_path, monkeypatch):
    from test_preflight_team_batch import changed_manifest
    path = changed_manifest(frozen, tmp_path, lambda manifest: None)
    manifest = batch.read(path)
    entries = [row for row in manifest['runs'] if row['seed'] == manifest['seeds'][0] and row['repeat'] == 1]
    teams, snapshots, originals = [], {}, {}
    monkeypatch.setattr(recovery, 'verify_fix_commit', lambda *args: None)
    try:
        for entry in entries:
            with batch.run_lock(Path(entry['output_dir'])):
                directory, record, _ = batch.register_attempt(path, manifest, entry)
            team = batch.make_team(manifest, entry, directory)
            teams.append(team)
            team._set_phase('boundary')
            record['status'] = 'paused'
            snapshots[entry['run_id']] = batch.save_checkpoint(team, directory, record, 'week_boundary')
            team._stage = 'ceo'
            team._set_phase('ceo')
            batch.save_checkpoint(team, directory, record, 'answers_ready')
            originals[directory / 'attempt.json'] = (directory / 'attempt.json').read_bytes()
        evidence = tmp_path / 'offline-evidence.json'
        write_json(evidence, dict(offline=True))
        declaration = dict(fix_classification='harness_fix', fix_commit='offline', affected_files=[],
            reason='Select a common existing boundary', evidence=[str(evidence)],
            attempts={run_id: dict(checkpoint=str(snapshot)) for run_id, snapshot in snapshots.items()})
        receipt = recovery.create_approval(path, tmp_path / 'approval', declaration, approved=True)
        approved = batch.read(receipt['path'])
        aligned = approved['decisions'][-1]
        assert aligned['alignment'] == 'same_week_boundary'
        assert {group: point['day'] for group, point in aligned['checkpoints'].items()} == {'git': 0, 'pf': 0}
        monkeypatch.setenv('CEOBENCH_RECOVERY_APPROVAL', receipt['path'])
        monkeypatch.setenv('CEOBENCH_RECOVERY_APPROVAL_SHA256', receipt['sha256'])
        for entry in entries:
            directory = batch.attempts(entry)[-1]
            previous = batch.read(directory / 'attempt.json')
            assert recovery.recovery_source(path, entry, previous, directory) == snapshots[entry['run_id']]
        other = dict(declaration, attempts={entries[0]['run_id']:
            dict(checkpoint=str(snapshots[entries[1]['run_id']]))})
        with pytest.raises(ValueError, match='another run'):
            recovery.create_approval(path, tmp_path / 'wrong-owner', other, approved=True)
        first = batch.attempts(entries[0])[-1]
        write_json(first / 'runtime/private/operations/later-business.json',
            dict(kind='api:set_prices', status='completed', day=0))
        with pytest.raises(RuntimeError, match='Business operations changed'):
            recovery.recovery_source(path, entries[0], batch.read(first / 'attempt.json'), first)
        assert {p: p.read_bytes() for p in originals} == originals
    finally:
        monkeypatch.delenv('CEOBENCH_RECOVERY_APPROVAL', raising=False)
        monkeypatch.delenv('CEOBENCH_RECOVERY_APPROVAL_SHA256', raising=False)
        for team in teams:
            batch.close_team(team)


def test_failed_role_copy_verification_leaves_no_partial_restore(tmp_path, monkeypatch):
    from saas_bench import team_checkpoint
    from saas_bench.multi_agent import MultiAgentRuntime
    from test_preflight_team_checkpoint import make_team, dispose, clients
    team = make_team(tmp_path / 'source', 'git')
    restored = None
    try:
        snapshot = team.checkpoint()
        copy = team_checkpoint.copy_workspace
        def corrupted_copy(source, destination, **options):
            copy(source, destination, **options)
            if Path(destination).name == 'roles':
                (Path(destination) / 'ceo/MEMORY.md').write_text('corrupted copy')
        monkeypatch.setattr(team_checkpoint, 'copy_workspace', corrupted_copy)
        target = tmp_path / 'restored'
        with pytest.raises(RuntimeError, match='Restored role workspace differs'):
            MultiAgentRuntime.restore(snapshot, target, client_factory=clients(), simulator_factory=fast_simulator)
        assert not target.exists()
        monkeypatch.setattr(team_checkpoint, 'copy_workspace', copy)
        restored = MultiAgentRuntime.restore(snapshot, target, client_factory=clients(), simulator_factory=fast_simulator)
        assert (restored.roles['ceo'].workspace / 'MEMORY.md').read_bytes() == (
            snapshot / 'roles/ceo/MEMORY.md').read_bytes()
    finally:
        if restored:
            dispose(restored)
        dispose(team)


@pytest.mark.parametrize('managed_git', ['git'], indirect=True)
@pytest.mark.parametrize('legacy', [False, True])
def test_rollback_keeps_unknown_work_end_and_legacy_time_explicit(managed_git, tmp_path, legacy):
    team, directory, previous, manifest, entry = managed_git
    snapshot = batch.save_checkpoint(team, directory, previous, 'ready')
    if legacy:
        state = batch.read(snapshot / 'team.json')
        state.pop('saved_at')
        write_json(snapshot / 'team.json', state)
        receipt = batch.read(snapshot / 'checkpoint.json')
        receipt['files']['team.json'] = file_hash(snapshot / 'team.json')
        write_json(snapshot / 'checkpoint.json', receipt)
    previous['status'] = 'paused'
    write_json(directory / 'attempt.json', previous)
    with batch.run_lock(Path(entry['output_dir'])):
        _, resumed, _ = batch.register_attempt(tmp_path / 'manifest.json', manifest, entry, resume=True)
    rollback = resumed['checkpoint_rollback']
    assert rollback['discarded_until'] is None and rollback['discarded_wall_seconds'] is None
    assert rollback['checkpoint_at_estimated'] is legacy
