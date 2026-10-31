import shutil

import pytest

from saas_bench import team_batch as batch
from saas_bench.run_state import write_json
from test_preflight_team_usage import PRICES, body, log, receipts


def usage_fixture(tmp_path, fault, *, resumed=False):
    entry = dict(output_dir=str(tmp_path), category='formal', run_id='git',
                 seed=42, repeat=1, group='git')
    first, second = [tmp_path / f'attempt-{number:03}' for number in (1, 2)]
    checkpoint = first / 'runtime/checkpoints/selected'
    ceo_kept = receipts('ceo-kept', 'ceo', body(10, 2), cost=.1)
    ceo_tail = receipts('ceo-tail', 'ceo', body(30, 4), cost=.2)
    growth_kept = receipts('growth-kept', 'growth', body(20, 3), cost=.1)
    growth_tail = receipts('growth-tail', 'growth', body(40, 5), cost=.2)
    for role, kept, tail in [('ceo', ceo_kept, ceo_tail),
                             ('growth', growth_kept, growth_tail)]:
        current = first / f'runtime/private/{role}/model-usage.jsonl'
        saved = checkpoint / f'private/{role}/model-usage.jsonl'
        current.parent.mkdir(parents=True)
        saved.parent.mkdir(parents=True)
        log(current, kept + tail)
        log(saved, kept)
    write_json(checkpoint / 'team.json', dict(source_root=str(first / 'runtime'),
        saved_at='2026-10-07T00:00:01+00:00',
        roles={role: dict(usage=dict(calls=1)) for role in ('ceo', 'growth')}))
    write_json(checkpoint / 'checkpoint.json', dict(log_cutoffs={
        str(path.relative_to(checkpoint)): path.stat().st_size
        for path in checkpoint.glob('private/*/model-usage.jsonl')}))
    saved = checkpoint / 'private/ceo/model-usage.jsonl'
    if fault == 'missing':
        saved.unlink()
    elif fault == 'truncated':
        saved.write_bytes(b'{"event":"request","call_id":"ceo-kept')
    else:
        saved.write_bytes(b'{malformed}\n')
    write_json(first / 'attempt.json', dict(status='paused', checkpoint=str(checkpoint)))
    if resumed:
        for role, rows in [('ceo', receipts('ceo-new', 'ceo', body(50, 6), cost=.3)),
                           ('growth', growth_kept)]:
            path = second / f'runtime/private/{role}/model-usage.jsonl'
            path.parent.mkdir(parents=True)
            log(path, rows)
        write_json(second / 'attempt.json', dict(status='finished', execution_started=True,
            source_checkpoint=str(checkpoint)))
    else:
        write_json(second / 'attempt.json', dict(status='paused', execution_started=True,
            source_checkpoint=str(checkpoint)))
    return entry, checkpoint


@pytest.mark.parametrize('fault', ['missing', 'truncated', 'malformed'])
def test_selected_usage_gap_is_unknown_not_invented_waste(tmp_path, monkeypatch, fault):
    entry, checkpoint = usage_fixture(tmp_path, fault)
    manifest = dict(category='formal', runs=[entry], d0={}, pricing=PRICES)
    monkeypatch.setattr(batch, 'validate_manifest', lambda path: manifest)
    report = batch.summarize(tmp_path / 'manifest.json')
    row, = report['runs']
    valid, waste = row['usage'], row['waste']
    assert valid['team']['known']['total_tokens'] == 23
    assert valid['team']['complete']['total_tokens'] is None
    assert valid['team']['wall_seconds'] is None
    assert valid['missing_receipts']
    assert 'ceo-kept' not in {record['call_id'] for record in waste['records']}
    assert waste['roles']['growth']['known']['total_tokens'] == 45
    assert waste['team']['complete']['total_tokens'] is None
    assert valid['unclassified']['status'] == 'retained_membership_unknown'
    assert valid['unclassified']['team']['known']['total_tokens'] == 46
    assert {record['call_id'] for record in valid['unclassified']['records']} == {'ceo-kept', 'ceo-tail'}
    assert report['resources']['team']['known']['total_tokens'] == 23
    assert report['resources']['team']['complete']['total_tokens'] is None
    assert report['pairs']['pairs'][0]['comparison']['total_tokens']['pf_over_git'] is None
    assert not list(tmp_path.rglob('HOLD'))


def test_new_usage_after_restore_does_not_hide_missing_inherited_log(tmp_path, monkeypatch):
    entry, checkpoint = usage_fixture(tmp_path, 'missing', resumed=True)
    manifest = dict(category='formal', runs=[entry], d0={}, pricing=PRICES)
    monkeypatch.setattr(batch, 'validate_manifest', lambda path: manifest)
    report = batch.summarize(tmp_path / 'manifest.json')
    row, = report['runs']
    assert row['usage']['team']['known']['total_tokens'] == 79
    assert row['usage']['team']['complete']['total_tokens'] is None
    assert {record['call_id'] for record in row['usage']['records']} == {'ceo-new', 'growth-kept'}
    assert 'ceo-kept' not in {record['call_id'] for record in row['waste']['records']}
    assert report['resources']['team']['complete']['total_tokens'] is None
    assert not list(tmp_path.rglob('HOLD'))


def test_selected_data_failure_before_execution_preserves_previous_pending(tmp_path):
    entry = dict(output_dir=str(tmp_path), category='formal', run_id='git')
    first, second = [tmp_path / f'attempt-{number:03}' for number in (1, 2)]
    path = first / 'runtime/private/ceo/model-usage.jsonl'
    path.parent.mkdir(parents=True)
    log(path, receipts('pending', 'ceo', body(30, 4), cost=.2))
    write_json(first / 'attempt.json', dict(status='paused', wall_seconds=10))
    write_json(second / 'attempt.json', dict(status='paused',
        reason='checkpoint_data_missing', execution_started=False,
        source_checkpoint=str(first / 'runtime/checkpoints/selected')))
    valid, waste = batch.run_usage(entry, PRICES)
    assert {record['call_id'] for record in valid['pending']['records']} == {'pending'}
    assert not waste['records']
    timing = batch.wall_time([first, second])
    assert timing['pending_wall_seconds'] == 10
    assert timing['waste_wall_seconds'] == 0


@pytest.mark.parametrize('fault', ['missing', 'truncated', 'malformed'])
def test_known_checkpoint_role_gap_preserves_other_role_totals(tmp_path, fault):
    entry, _ = usage_fixture(tmp_path, fault)
    valid, waste = batch.run_usage(entry, PRICES)

    assert valid['team']['complete']['total_tokens'] is None
    assert valid['roles']['ceo']['complete']['total_tokens'] is None
    assert valid['roles']['growth']['complete']['total_tokens'] == 23
    assert valid['roles']['growth']['wall_seconds'] == 0
    assert waste['roles']['growth']['complete']['total_tokens'] == 45
    assert waste['roles']['growth']['wall_seconds'] == 0
    assert not list(tmp_path.rglob('HOLD'))


def external_descendant(tmp_path):
    source = tmp_path / 'source'
    _, first_checkpoint = usage_fixture(source, 'missing', resumed=True)
    second = source / 'attempt-002'
    selected = second / 'runtime/checkpoints/descendant'
    shutil.copytree(second / 'runtime/private', selected / 'private')
    write_json(selected / 'team.json', dict(
        source_root=str(second / 'runtime'),
        saved_at='2026-10-07T00:00:02+00:00',
        roles=dict(ceo=dict(usage=dict(calls=2)), growth=dict(usage=dict(calls=1)))))
    write_json(selected / 'checkpoint.json', dict(log_cutoffs={
        str(path.relative_to(selected)): path.stat().st_size
        for path in selected.glob('private/*/model-usage.jsonl')}))
    write_json(second / 'attempt.json', dict(status='finished', execution_started=True,
        source_checkpoint=str(first_checkpoint), checkpoint=str(selected)))

    destination = tmp_path / 'destination'
    current = destination / 'attempt-001'
    shutil.copytree(selected / 'private', current / 'runtime/private')
    write_json(current / 'attempt.json', dict(status='finished', execution_started=True,
        source_checkpoint=str(selected)))
    entry = dict(output_dir=str(destination), category='formal', run_id='git',
                 seed=42, repeat=1, group='git')
    return entry, second


def test_external_checkpoint_keeps_missing_ancestor_usage_unknown(tmp_path):
    entry, _ = external_descendant(tmp_path)
    valid, waste = batch.run_usage(entry, PRICES)

    assert valid['team']['known']['total_tokens'] == 79
    assert valid['team']['complete']['total_tokens'] is None
    assert valid['roles']['ceo']['complete']['total_tokens'] is None
    assert valid['roles']['growth']['complete']['total_tokens'] == 23
    assert valid['unclassified']['status'] == 'retained_membership_unknown'
    assert {row['call_id'] for row in valid['unclassified']['records']} == {
        'ceo-kept', 'ceo-tail'}
    assert not waste['records']
    assert not list(tmp_path.rglob('HOLD'))


def test_unreadable_external_ancestry_cannot_claim_complete_usage(tmp_path):
    entry, second = external_descendant(tmp_path)
    (second / 'attempt.json').unlink()
    valid, _ = batch.run_usage(entry, PRICES)

    assert valid['team']['known']['total_tokens'] == 79
    assert valid['team']['complete']['total_tokens'] is None
    assert valid['team']['wall_seconds'] is None
    assert valid['missing_receipts']
    assert not list(tmp_path.rglob('HOLD'))


@pytest.mark.parametrize('status', ['paused', 'finished'])
def test_current_usage_gap_does_not_poison_independent_historical_waste(tmp_path, status):
    entry = dict(output_dir=str(tmp_path), category='formal', run_id='git')
    first, second = [tmp_path / f'attempt-{number:03}' for number in (1, 2)]
    old_path = first / 'runtime/private/ceo/model-usage.jsonl'
    current_path = second / 'runtime/private/ceo/model-usage.jsonl'
    old_path.parent.mkdir(parents=True)
    current_path.parent.mkdir(parents=True)

    old = receipts('old-waste', 'ceo', body(10, 2), cost=.1)
    old[-1]['timestamp'] = '2026-10-07T00:00:04+00:00'
    log(old_path, old)
    current = receipts('current', 'ceo', body(30, 4), cost=.2)
    current[0]['timestamp'] = '2026-10-07T00:01:00+00:00'
    current[-1]['timestamp'] = '2026-10-07T00:01:03+00:00'
    log(current_path, current)
    write_json(first / 'attempt.json', dict(status='finished', execution_started=True))
    write_json(second / 'attempt.json', dict(status=status, execution_started=True,
        replay_week=1, source_checkpoint=None, checkpoint=None))

    _, before = batch.run_usage(entry, PRICES)
    assert before['roles']['ceo']['complete']['total_tokens'] == 12
    assert before['roles']['ceo']['wall_seconds'] == 4
    with current_path.open('ab') as stream:
        stream.write(b'{"event":"request","call_id":"unfinished')

    retained, waste = batch.run_usage(entry, PRICES)
    current_usage = retained['pending'] if status == 'paused' else retained
    assert current_usage['roles']['ceo']['known']['total_tokens'] == 34
    assert current_usage['roles']['ceo']['complete']['total_tokens'] is None
    assert current_usage['roles']['ceo']['wall_seconds'] is None
    assert current_usage['roles']['ceo']['wall_seconds_known'] == 3
    assert any(gap['reason'] == 'unterminated_log_tail'
               for gap in current_usage['missing_receipts'])

    assert {row['call_id'] for row in waste['records']} == {'old-waste'}
    assert waste['roles']['ceo']['known']['total_tokens'] == 12
    assert waste['roles']['ceo']['complete']['total_tokens'] == 12
    assert waste['roles']['ceo']['wall_seconds'] == 4
    assert waste['roles']['ceo']['wall_seconds_missing_calls'] == []
    assert waste['team']['complete']['total_tokens'] == 12
    assert waste['missing_receipts'] == []
    if status == 'finished':
        from saas_bench.team_usage import aggregate
        assert {key: value for key, value in retained['pending'].items()
                if key != 'status'} == aggregate([], PRICES)
    assert not list(tmp_path.rglob('HOLD'))


def test_finished_simulator_tail_has_no_pending_or_waste_partition(tmp_path):
    from saas_bench.team_usage import aggregate

    entry = dict(output_dir=str(tmp_path), category='formal', run_id='git')
    attempt = tmp_path / 'attempt-001'
    ceo_path = attempt / 'runtime/private/ceo/model-usage.jsonl'
    ceo_path.parent.mkdir(parents=True)
    log(ceo_path, receipts('retained-ceo', 'ceo', body(10, 2), cost=.1))
    simulator_path = ceo_path.parent.parent / 'simulator-model-usage.jsonl'
    log(simulator_path, receipts('retained-simulator', 'simulator', body(20, 4), cost=.2))
    with simulator_path.open('ab') as stream:
        stream.write(b'{"event":"request","call_id":"unfinished-simulator')
    write_json(attempt / 'attempt.json', dict(status='finished', execution_started=True,
        source_checkpoint=None, checkpoint=None))

    retained, waste = batch.run_usage(entry, PRICES)
    assert retained['team']['complete']['total_tokens'] == 12
    assert retained['resources']['simulator']['known']['total_tokens'] == 24
    assert retained['resources']['simulator']['complete']['total_tokens'] is None
    assert any(gap['category'] == 'simulator' for gap in retained['missing_receipts'])
    empty = aggregate([], PRICES)
    assert {key: value for key, value in retained['pending'].items()
            if key != 'status'} == empty
    assert waste == empty
    assert not list(tmp_path.rglob('HOLD'))
