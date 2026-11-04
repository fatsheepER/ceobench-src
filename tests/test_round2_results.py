import pytest

from saas_bench.team_results import summarize_pairs


@pytest.mark.parametrize('end', [28, 56, 70, 112])
def test_complete_pair_uses_assigned_observation_end(end):
    runs = [dict(run_id=f'{group}-one', seed=101, repeat=1, group=group,
                 status='reached_d112' if end == 112 else 'reached_stop_day',
                 day=end, stop_after_day=end) for group in ('git', 'pf')]
    result = summarize_pairs(runs)
    assert result.get('complete_observation_pairs') == 1
    assert result['pairs'][0]['observation_end'] == end
    assert result['pairs'][0]['complete_observation'] is True


@pytest.mark.parametrize('fault', ['different_end', 'early_day', 'late_day', 'capture_fault'])
def test_observation_pair_does_not_hide_incomplete_or_mismatched_runs(fault):
    runs = [dict(run_id=f'{group}-one', seed=101, repeat=1, group=group,
                 status='reached_stop_day', day=56, stop_after_day=56)
            for group in ('git', 'pf')]
    if fault == 'different_end':
        runs[1]['stop_after_day'] = 28
    elif fault == 'early_day':
        runs[1]['day'] = 49
    elif fault == 'late_day':
        runs[1]['day'] = 63
    else:
        runs[1]['evidence_fault'] = True
    result = summarize_pairs(runs)
    assert result.get('complete_observation_pairs') == 0
    assert result['pairs'][0]['complete_observation'] is False
