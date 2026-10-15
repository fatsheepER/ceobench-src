import json
from pathlib import Path
import subprocess
import os

import pytest

from saas_bench import team_recovery as recovery
from saas_bench.run_state import file_hash, tree_hash, write_json


def snapshot(root, run_id, saved_at, *, day=21, stage='boundary'):
    root.mkdir(parents=True)
    (root / 'roles').mkdir()
    (root / 'private').mkdir()
    (root / 'world.nmdb').write_bytes(b'offline snapshot')
    write_json(root / 'team.json', dict(day=day, stage=stage, saved_at=saved_at,
        roles={name: dict(identity=dict(run_id=run_id)) for name in ('ceo', 'growth', 'ops_finance')}))
    write_json(root / 'checkpoint.json', dict(version=1, roles_hash_policy='all_files',
        files={name: file_hash(root / name) for name in ('team.json', 'world.nmdb')},
        roles_sha256=tree_hash(root / 'roles', exact=True), private_sha256=tree_hash(root / 'private')))
    return root


def test_latest_checkpoint_ignores_failed_operation_tail_and_preserves_old_files(tmp_path):
    entry = dict(run_id='git-1', output_dir=str(tmp_path / 'git'))
    first = Path(entry['output_dir']) / 'attempt-001'
    older = snapshot(first / 'runtime/checkpoints/older', entry['run_id'], '2026-10-15T01:00:00+00:00')
    newer = snapshot(first / 'runtime/checkpoints/newer', entry['run_id'], '2026-10-15T02:00:00+00:00')
    write_json(first / 'attempt.json', dict(status='running', checkpoint=str(older)))
    write_json(first / 'runtime/private/operations/tail.json', dict(kind='api:advance', status='running'))
    original = {path: path.read_bytes() for path in first.rglob('*') if path.is_file()}
    assert recovery.latest_checkpoint(entry) == newer
    assert {path: path.read_bytes() for path in original} == original


def test_explicit_checkpoint_selection_survives_repeated_recovery(tmp_path, monkeypatch):
    entry = dict(run_id='pf-1', output_dir=str(tmp_path / 'pf'))
    first = Path(entry['output_dir']) / 'attempt-001'
    boundary = snapshot(first / 'runtime/checkpoints/boundary', entry['run_id'], '2026-10-15T01:00:00+00:00')
    newest = snapshot(first / 'runtime/checkpoints/newest', entry['run_id'], '2026-10-15T02:00:00+00:00')
    write_json(first / 'attempt.json', dict(status='paused'))
    monkeypatch.setenv('CEOBENCH_RECOVERY_CHECKPOINTS', json.dumps({entry['run_id']: str(boundary)}))
    assert recovery.latest_checkpoint(entry) == boundary
    write_json(Path(entry['output_dir']) / 'attempt-002/attempt.json', dict(source_checkpoint=str(boundary)))
    assert recovery.latest_checkpoint(entry) == boundary
    resumed = snapshot(Path(entry['output_dir']) / 'attempt-002/runtime/checkpoints/resumed',
        entry['run_id'], '2026-10-15T03:00:00+00:00')
    assert recovery.latest_checkpoint(entry) == boundary
    monkeypatch.setenv('CEOBENCH_RECOVERY_CHECKPOINTS', json.dumps({entry['run_id']: str(newest)}))
    assert recovery.latest_checkpoint(entry) == newest
    monkeypatch.setenv('CEOBENCH_RECOVERY_CHECKPOINTS', json.dumps({entry['run_id']: str(boundary)}))
    assert recovery.latest_checkpoint(entry) == boundary


def test_replay_retry_stays_on_selected_past_until_new_checkpoint_is_saved(tmp_path):
    entry = dict(run_id='pf-1', output_dir=str(tmp_path / 'pf'))
    first = Path(entry['output_dir']) / 'attempt-999'
    beginning = snapshot(first / 'runtime/checkpoints/first', entry['run_id'], '2026-10-15T01:00:00+00:00', day=0)
    snapshot(first / 'runtime/checkpoints/future', entry['run_id'], '2026-10-15T02:00:00+00:00', day=70)
    latest = Path(entry['output_dir']) / 'attempt-1000'
    write_json(latest / 'attempt.json', dict(status='failed', checkpoint=str(beginning), source_checkpoint=str(beginning)))
    assert recovery.latest_checkpoint(entry) == beginning
    progressed = snapshot(latest / 'runtime/checkpoints/new', entry['run_id'], '2026-10-15T03:00:00+00:00', day=7)
    assert recovery.latest_checkpoint(entry) == progressed
    assert recovery.latest_checkpoint(entry, week=11).name == 'future'


def test_invalid_latest_checkpoint_is_reported_and_selected(tmp_path, caplog):
    entry = dict(run_id='git-1', output_dir=str(tmp_path / 'git'))
    first = Path(entry['output_dir']) / 'attempt-001'
    valid = snapshot(first / 'runtime/checkpoints/valid', entry['run_id'], '2026-10-15T01:00:00+00:00')
    broken = snapshot(first / 'runtime/checkpoints/broken', entry['run_id'], '2026-10-15T02:00:00+00:00')
    (broken / 'world.nmdb').write_bytes(b'incomplete')
    assert recovery.latest_checkpoint(entry) == broken
    assert 'checksum mismatch' in caplog.text


def test_checkpoint_discovery_survives_incomplete_attempt_metadata(tmp_path, caplog):
    entry = dict(run_id='git-1', output_dir=str(tmp_path / 'git'))
    first = Path(entry['output_dir']) / 'attempt-001'
    selected = snapshot(first / 'runtime/checkpoints/valid', entry['run_id'], '2026-10-15T01:00:00+00:00')
    (first / 'attempt.json').write_text('{"status":')
    assert recovery.latest_checkpoint(entry) == selected
    assert str(first / 'attempt.json') in caplog.text


def test_checkpoint_selection_warns_about_run_identity(tmp_path, monkeypatch, caplog):
    entry = dict(run_id='git-1', output_dir=str(tmp_path / 'git'))
    wrong = snapshot(Path(entry['output_dir']) / 'attempt-001/runtime/checkpoints/wrong', 'pf-1', '2026-10-15T01:00:00+00:00')
    monkeypatch.setenv('CEOBENCH_RECOVERY_CHECKPOINTS', json.dumps({entry['run_id']: str(wrong)}))
    assert recovery.latest_checkpoint(entry) == wrong
    assert 'run identity' in caplog.text


def test_any_saved_week_can_be_selected_after_later_attempts(tmp_path, monkeypatch):
    entry = dict(run_id='pf-1', output_dir=str(tmp_path / 'pf'))
    first = Path(entry['output_dir']) / 'attempt-001'
    beginning = snapshot(first / 'runtime/checkpoints/first', entry['run_id'], '2026-10-15T01:00:00+00:00', day=0)
    boundary = snapshot(first / 'runtime/checkpoints/week4', entry['run_id'], '2026-10-15T02:00:00+00:00')
    later = Path(entry['output_dir']) / 'attempt-002'
    write_json(later / 'attempt.json', dict(source_checkpoint=str(boundary)))
    snapshot(later / 'runtime/checkpoints/answers', entry['run_id'], '2026-10-15T03:00:00+00:00', stage='ceo')
    newest = snapshot(later / 'runtime/checkpoints/week5', entry['run_id'], '2026-10-15T04:00:00+00:00', day=28)
    monkeypatch.setenv('CEOBENCH_RECOVERY_CHECKPOINTS', 'obsolete metadata')
    original = {path: path.read_bytes() for path in Path(entry['output_dir']).rglob('*') if path.is_file()}
    assert recovery.latest_checkpoint(entry, week=1) == beginning
    assert recovery.latest_checkpoint(entry, week=4) == boundary
    assert recovery.latest_checkpoint(entry, week=5) == newest
    assert {path: path.read_bytes() for path in original} == original


def test_missing_week_warns_and_does_not_choose_a_different_saved_week(tmp_path, caplog):
    entry = dict(run_id='git-1', output_dir=str(tmp_path / 'git'))
    first = Path(entry['output_dir']) / 'attempt-001'
    snapshot(first / 'runtime/checkpoints/latest', entry['run_id'], '2026-10-15T01:00:00+00:00')
    assert recovery.latest_checkpoint(entry, week=1) is None
    assert 'No saved checkpoint for week 1' in caplog.text


@pytest.mark.parametrize('week', [None, 4])
def test_recovery_cli_reports_checkpoint_without_approval(tmp_path, week):
    from saas_bench import team_batch as batch
    entry = dict(run_id='git-1', output_dir=str(tmp_path / 'git'))
    selected = snapshot(Path(entry['output_dir']) / 'attempt-001/runtime/checkpoints/valid', entry['run_id'], '2026-10-15T01:00:00+00:00')
    manifest = tmp_path / 'manifest.json'
    write_json(manifest, dict(runs=[entry]))
    command = [os.sys.executable, str(batch.ROOT / 'scripts/team_recovery.py'),
        '--manifest', str(manifest), '--run-id', entry['run_id']]
    if week is not None:
        command.extend(['--week', str(week)])
    result = subprocess.run(command,
        env=dict(os.environ, PYTHONPATH=str(batch.ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [dict(run_id='git-1', checkpoint=str(selected),
        day=21, stage='boundary', approval_required=False)]
