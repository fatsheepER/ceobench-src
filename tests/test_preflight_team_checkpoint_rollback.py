from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from saas_bench import team_batch as batch, team_recovery as recovery
from saas_bench.run_state import file_hash, tree_hash, write_json
from test_preflight_team_batch import frozen
from test_preflight_team_checkpoint import fast_simulator
from test_preflight_team_flow import raw
from test_preflight_team_readonly_recovery import managed_git


@pytest.mark.parametrize('managed_git', ['git', 'pf'], indirect=True)
@pytest.mark.parametrize('business', ['none', 'unknown', 'completed'])
def test_recovery_rolls_back_role_tail_and_all_post_checkpoint_business_writes(
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
    path = tmp_path / 'manifest.json'
    original = {p: p.read_bytes() for p in
        (directory / 'attempt.json', snapshot / 'checkpoint.json', workspace / 'saved.txt')}
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


def test_first_recovery_selects_independent_boundaries_without_approval(frozen, tmp_path, monkeypatch):
    import json
    from test_preflight_team_batch import changed_manifest
    path = changed_manifest(frozen, tmp_path, lambda manifest: None)
    manifest = batch.read(path)
    entries = [row for row in manifest['runs'] if row['seed'] == manifest['seeds'][0] and row['repeat'] == 1]
    teams, snapshots, originals = [], {}, {}
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
        monkeypatch.setenv('CEOBENCH_RECOVERY_CHECKPOINTS', json.dumps({key: str(value) for key, value in snapshots.items()}))
        for entry in entries:
            directory = batch.attempts(entry)[-1]
            write_json(directory / 'runtime/private/operations/later-business.json',
                dict(kind='api:advance', status='running', day=0))
            assert recovery.latest_checkpoint(entry) == snapshots[entry['run_id']]
        assert {p: p.read_bytes() for p in originals} == originals
    finally:
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
