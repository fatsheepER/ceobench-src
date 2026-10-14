from datetime import datetime, timedelta, timezone
from pathlib import Path

from saas_bench import team_batch as batch
from saas_bench.run_state import write_json
from saas_bench.team_results import summarize_pairs
from test_preflight_team_usage import PRICES, body, log, receipts
from test_preflight_team_batch import frozen, managed_team


def test_retained_calls_and_discarded_calls_are_separate_by_role(tmp_path):
    entry = dict(output_dir=str(tmp_path), category='formal')
    first, second = tmp_path / 'attempt-001', tmp_path / 'attempt-002'
    for directory in (first, second):
        (directory / 'runtime/private/ceo').mkdir(parents=True)
        (directory / 'runtime/private/growth').mkdir()
    kept = receipts('kept', 'ceo', body(10, 2), cost=.1)
    abandoned = receipts('discarded', 'growth', body(30, 4), cost=.2)
    resumed = receipts('resumed', 'ceo', body(20, 3), cost=.1)
    abandoned[-1]['timestamp'] = '2026-10-07T00:00:04+00:00'
    log(first / 'runtime/private/ceo/model-usage.jsonl', kept)
    log(first / 'runtime/private/growth/model-usage.jsonl', abandoned)
    log(second / 'runtime/private/ceo/model-usage.jsonl', kept + resumed)
    valid, waste = batch.run_usage(entry, PRICES)
    assert valid['team']['complete']['total_tokens'] == 35
    assert waste['team']['complete']['total_tokens'] == 34
    assert waste['roles']['growth']['complete']['total_tokens'] == 34
    assert waste['roles']['growth']['wall_seconds'] == 4
    assert {row['call_id'] for row in waste['records']} == {'discarded'}


def test_repeated_rollback_excludes_abandoned_attempts_and_downtime(tmp_path):
    stamp = datetime(2026, 10, 15, tzinfo=timezone.utc)
    history = [tmp_path / f'attempt-{number:03}' for number in (1, 2, 3)]
    source = history[0] / 'runtime/checkpoints/first'
    write_json(source / 'team.json', dict(source_root=str(history[0] / 'runtime'),
        saved_at=(stamp + timedelta(seconds=80)).isoformat()))
    for directory, seconds, start in zip(history, (100, 50, 30), (0, 1000, 2000)):
        write_json(directory / 'attempt.json', dict(wall_seconds=seconds,
            execution_started_at=(stamp + timedelta(seconds=start)).isoformat(),
            source_checkpoint=str(source) if directory != history[0] else None))
    result = batch.wall_time(history)
    assert result['wall_seconds'] == 110
    assert result['waste_wall_seconds'] == 70
    assert result['wall_seconds'] + result['waste_wall_seconds'] == 180


def test_pair_comparison_uses_valid_time_and_exposes_waste_separately():
    rows = [dict(seed=42, repeat=1, group=group, run_id=group, status='running',
                 wall_seconds=duration, waste_wall_seconds=waste)
            for group, duration, waste in [('git', 10, 90), ('pf', 20, 1)]]
    pair = summarize_pairs(rows)['pairs'][0]
    assert pair['comparison']['wall_seconds']['pf_over_git'] == 2
    assert pair['waste']['git']['wall_seconds'] == 90


def test_preparation_failure_uses_checkpoint_logs_and_final_failure_tail_is_waste(tmp_path):
    entry = dict(output_dir=str(tmp_path), category='formal', run_id='git')
    first, second = tmp_path / 'attempt-001', tmp_path / 'attempt-002'
    private = first / 'runtime/private/ceo'
    private.mkdir(parents=True)
    kept = receipts('kept', 'ceo', body(10, 2), cost=.1)
    log(private / 'model-usage.jsonl', kept + receipts('discarded', 'ceo', body(30, 4), cost=.2))
    checkpoint = first / 'runtime/checkpoints/complete'
    (checkpoint / 'private/ceo').mkdir(parents=True)
    (checkpoint / 'roles').mkdir()
    log(checkpoint / 'private/ceo/model-usage.jsonl', kept)
    (checkpoint / 'world.nmdb').write_bytes(b'complete world')
    write_json(checkpoint / 'team.json', dict(day=21, stage='boundary',
        saved_at='2026-10-15T01:00:00+00:00', source_root=str(first / 'runtime'),
        roles={name: dict(identity=dict(run_id='git')) for name in ('ceo', 'growth', 'ops_finance')}))
    from saas_bench.run_state import file_hash, tree_hash
    write_json(checkpoint / 'checkpoint.json', dict(version=1,
        files={name: file_hash(checkpoint / name) for name in ('team.json', 'world.nmdb')},
        roles_sha256=tree_hash(checkpoint / 'roles', exact=True),
        private_sha256=tree_hash(checkpoint / 'private')))
    write_json(first / 'attempt.json', dict(status='failed', checkpoint=str(checkpoint)))
    write_json(second / 'attempt.json', dict(status='failed', attempt=2, source_checkpoint=str(checkpoint)))
    valid, waste = batch.run_usage(entry, PRICES)
    assert valid['team']['complete']['total_tokens'] == 12
    assert waste['team']['complete']['total_tokens'] == 34
    from unittest.mock import patch
    manifest = dict(category='formal', runs=[dict(entry, seed=42, repeat=1, group='git')],
                    d0={}, pricing=PRICES)
    with patch.object(batch, 'validate_manifest', return_value=manifest):
        report = batch.summarize(tmp_path / 'manifest.json')
    assert report['resources']['team']['complete']['total_tokens'] == 12
    assert report['runs'][0]['waste']['team']['complete']['total_tokens'] == 34


def test_unreadable_historical_time_metadata_remains_unknown(tmp_path):
    first, second = tmp_path / 'attempt-001', tmp_path / 'attempt-002'
    first.mkdir()
    (first / 'attempt.json').write_text('{unfinished')
    write_json(second / 'attempt.json', dict(wall_seconds=3))
    report = batch.wall_time([first, second])
    assert report['wall_seconds'] is None
    assert report['wall_seconds_known'] == 3
    assert report['time_metadata_errors'][0]['attempt'] == 'attempt-001'


def test_failed_model_request_does_not_publish_a_new_checkpoint(managed_team, monkeypatch):
    from saas_bench.multi_agent import RunOutcome
    from saas_bench.run_lifecycle import RunCancelled
    team, directory, record, manifest, entry = managed_team
    original = record['checkpoint']
    before = set((team.root / 'checkpoints').iterdir())
    def fail_model(**kwargs):
        usage = team.roles['ceo'].usage
        monkeypatch.setattr(usage.lifecycle, 'sleep', lambda seconds: None)
        try:
            usage.call('chat', dict(model=team.roles['ceo'].agent.model),
                       lambda: (_ for _ in ()).throw(ValueError('offline failed request')), day=0)
        except RunCancelled:
            pass
        team.failure = 'model_request_attempt_limit'
        return RunOutcome(day=0, reason='failed')
    monkeypatch.setattr(team, 'run', fail_model)
    result = batch.execute(team, manifest, entry, directory, record, directory / 'STOP')
    assert result['checkpoint'] == original
    assert set((team.root / 'checkpoints').iterdir()) == before
    assert result['result']['usage']['team']['billed_units'] == 0
    assert result['result']['waste']['roles']['ceo']['billed_units'] == 1
