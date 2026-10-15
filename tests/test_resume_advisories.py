
from saas_bench import team_batch as batch
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.run_state import write_json
from test_preflight_team_batch import changed_manifest, frozen
from test_preflight_team_checkpoint import clients, dispose, fast_simulator, make_team


def test_restore_continues_with_bad_hash_policy_workspace_and_rng(tmp_path, monkeypatch, caplog):
    team = make_team(tmp_path / 'source', 'git')
    snapshot = team.checkpoint()
    dispose(team)
    receipt = batch.read(snapshot / 'checkpoint.json')
    receipt.update(roles_hash_policy='unknown', roles_sha256='bad')
    write_json(snapshot / 'checkpoint.json', receipt)
    from saas_bench.simulation import Simulator
    monkeypatch.setattr(Simulator, 'restore_rng_states', lambda self: False)
    restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored',
        client_factory=clients(), simulator_factory=fast_simulator, force_resume=True)
    try:
        assert restored.run(stop_after_day=7).reason == 'observation_end'
        assert 'hash policy' in caplog.text and 'checksum mismatch' in caplog.text
        assert 'RNG' in caplog.text
    finally:
        dispose(restored)


def test_prepare_records_manifest_environment_and_runtime_warnings(frozen, tmp_path, monkeypatch):
    path = changed_manifest(frozen, tmp_path, lambda manifest: manifest.update(version=99))
    manifest = batch.read(path)
    entry = manifest['runs'][0]
    monkeypatch.setenv('PYTHONHASHSEED', '123')
    monkeypatch.setenv('ORACLE_MODE', '1')
    record = batch.prepare(path, entry['run_id'], force_resume=True,
        client_factory=clients(), simulator_factory=batch.simulator)
    assert record['status'] == 'ready'
    directory = batch.attempts(entry)[-1]
    saved = batch.read(directory / 'attempt.json')
    assert saved['force_resume'] is True
    messages = '\n'.join(row['message'] for row in saved['validation_warnings'])
    assert 'manifest version' in messages and 'PYTHONHASHSEED' in messages and 'oracle' in messages
    assert (directory / 'warnings.jsonl').read_text()


def test_bad_time_metadata_is_unknown_and_does_not_abort(tmp_path, caplog):
    directory = tmp_path / 'attempt-001'
    checkpoint = directory / 'runtime/checkpoints/old'
    write_json(checkpoint / 'team.json', dict(source_root=str(directory / 'runtime'), saved_at='bad-date'))
    write_json(directory / 'attempt.json', dict(status='paused', checkpoint=str(checkpoint),
        execution_started_at='also-bad', wall_seconds=3))
    result = batch.wall_time([directory])
    assert result['wall_seconds'] is None and result['time_metadata_errors']
    assert 'time' in caplog.text.lower()


def test_resume_keeps_missing_receipt_and_bad_prior_time_advisory(frozen, tmp_path):
    path = changed_manifest(frozen, tmp_path, lambda manifest: None)
    entry = batch.read(path)['runs'][0]
    first = batch.prepare(path, entry['run_id'], simulator_factory=batch.simulator)
    from pathlib import Path
    snapshot = Path(first['checkpoint'])
    (snapshot / 'checkpoint.json').rename(snapshot / 'original-checkpoint.json')
    first.update(started_at='invalid-time', wall_seconds=3, status='failed')
    first.pop('finished_at', None)
    write_json(batch.attempts(entry)[0] / 'attempt.json', first)
    resumed = batch.prepare(path, entry['run_id'], resume=True, force_resume=True,
        simulator_factory=batch.simulator)
    assert resumed['status'] == 'ready' and resumed['attempt'] == 2
    assert resumed['source_checkpoint'] == str(snapshot)
    messages = '\n'.join(row['message'] for row in resumed['validation_warnings'])
    assert 'checkpoint.json' in messages and 'isoformat' in messages


def test_public_archive_contains_advisory_dependency():
    import subprocess
    import sys
    archive = batch.ROOT / 'public/novamind-operation'
    result = subprocess.run([sys.executable, '-I', '-c',
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from saas_bench.run_warnings import advisory; "
        "assert advisory.__module__ == 'saas_bench.run_warnings'", str(archive)],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
