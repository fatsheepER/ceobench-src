from pathlib import Path

import pytest

from saas_bench import team_batch as batch
from saas_bench.run_state import file_hash, write_json
from test_preflight_team_batch import changed_manifest, frozen, managed_team


@pytest.mark.parametrize('legacy_cap', [None, 3])
def test_manual_resumes_have_no_cumulative_limit(frozen, tmp_path, legacy_cap):
    def policy(manifest):
        if legacy_cap is not None:
            manifest.pop('attempt_policy', None)
            manifest['max_attempts'] = legacy_cap
    path = changed_manifest(frozen, tmp_path, policy)
    manifest = batch.validate_manifest(path)
    entry = manifest['runs'][0]
    previous = None
    for number in range(1, 7):
        record = batch.prepare(path, entry['run_id'], resume=number > 1)
        directory = batch.attempts(entry)[-1]
        assert record['attempt'] == number and record['status'] == 'ready'
        assert record['source_checkpoint'] == previous
        assert file_hash(directory / 'manifest.json') == file_hash(path)
        assert batch.read(Path(record['checkpoint']) / 'team.json')['stage'] == 'boundary'
        previous = record['checkpoint']
        record['status'] = 'paused'
        write_json(directory / 'attempt.json', record)
    assert len(batch.attempts(entry)) == 6


def test_resume_ignores_historical_hold_operation_tail_and_active_lock(managed_team, caplog):
    _, directory, record, manifest, entry = managed_team
    path = directory / 'manifest.json'
    record['status'] = 'paused'
    write_json(directory / 'attempt.json', record)
    for _ in range(3):
        directory, record, _ = batch.register_attempt(path, manifest, entry, resume=True)
        record['status'] = 'paused'
        write_json(directory / 'attempt.json', record)
    assert record['attempt'] == 4
    before = (directory / 'attempt.json').read_bytes()
    hold = path.parent / 'HOLD'
    hold.touch()
    created, resumed, source = batch.register_attempt(path, manifest, entry, resume=True)
    assert source == Path(record['checkpoint']) and created.name == 'attempt-005'
    assert hold.exists()
    with batch.run_lock(Path(entry['output_dir'])):
        pass
    assert 'already active' in caplog.text
    operations = directory / 'runtime/private/operations'
    operations.mkdir(parents=True)
    write_json(operations / 'unsnapshotted.json', dict(status='completed'))
    created, resumed, source = batch.register_attempt(path, manifest, entry, resume=True)
    assert source == Path(record['checkpoint']) and created.name == 'attempt-006'
    assert (directory / 'attempt.json').read_bytes() == before


def test_attempt_numbering_remains_monotonic_beyond_three_digits(tmp_path):
    path = tmp_path / 'manifest.json'
    write_json(path, {})
    for number in (999, 1000):
        directory = tmp_path / f'attempt-{number:03}'
        write_json(directory / 'manifest.json', {})
        write_json(directory / 'attempt.json', dict(status='paused', execution_started=False,
            attempt=number, manifest_sha256=file_hash(path), checkpoint=None))
    entry = dict(output_dir=str(tmp_path), run_id='offline', category='engineering')
    assert [p.name for p in batch.attempts(entry)] == ['attempt-999', 'attempt-1000']
    directory, record, _ = batch.register_attempt(path, {}, entry, resume=True)
    assert record['attempt'] == 1001 and directory.name == 'attempt-1001'


@pytest.mark.parametrize('policy', [
    dict(attempt_policy='automatic_retry'), dict(max_attempts=0), dict(max_attempts=True),
])
def test_historical_attempt_policy_does_not_block_recovery(frozen, tmp_path, policy):
    path = changed_manifest(frozen, tmp_path, lambda manifest: manifest.update(policy))
    assert batch.validate_manifest(path)['attempt_policy'] == policy.get('attempt_policy', 'automatic_checkpoint_recovery')


def test_concurrent_manual_registrations_preserve_both_attempts(frozen, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from test_preflight_team_batch import changed_manifest
    path = changed_manifest(frozen, tmp_path, lambda manifest: None)
    manifest = batch.read(path)
    entry = manifest['runs'][0]
    Path(entry['output_dir']).mkdir(parents=True)
    barrier = threading.Barrier(2)
    create = batch.tempfile.mkdtemp
    def together(*args, **kwargs):
        barrier.wait(timeout=5)
        return create(*args, **kwargs)
    monkeypatch.setattr(batch.tempfile, 'mkdtemp', together)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: batch.register_attempt(path, manifest, entry), range(2)))
    assert {row[1]['attempt'] for row in results} == {1, 2}
    assert len(batch.attempts(entry)) == 2
    assert len({batch.read(row[0] / 'attempt.json')['attempt_id'] for row in results}) == 2
