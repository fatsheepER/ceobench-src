from saas_bench.team_results import summarize_pairs
from saas_bench.team_usage import aggregate
from test_preflight_team_usage import PRICES, body, log, receipts


def test_pair_table_keeps_roles_waste_known_subtotals_and_missing(tmp_path):
    rows = []
    for group in ('git', 'pf'):
        kept = receipts(group + '-good', 'ceo', body(10, 2))
        missing = receipts(group + '-missing', 'ceo', body(20, 3))
        missing[-1]['usage']['input_tokens'] = None
        missing[-1]['response']['usage'].pop('prompt_tokens')
        usage = aggregate([log(tmp_path / (group + '.jsonl'), kept + missing)], PRICES)
        waste = aggregate([log(tmp_path / (group + '-waste.jsonl'), receipts(group + '-discarded', 'growth', body(100, 5)))], PRICES)
        rows.append(dict(run_id=group, group=group, seed=1048, repeat=1, status='reached_stop_day',
            day=56, stop_after_day=56, usage=usage, waste=waste, wall_seconds=7, waste_wall_seconds=9))
    result = summarize_pairs(rows)
    table = result['usage_table']
    selected = lambda group, scope, role: next(r for r in table if (r['group'], r['scope'], r['role']) == (group, scope, role))
    ceo = selected('pf', 'retained', 'ceo')
    assert ceo['input_tokens'] is None and ceo['known']['input_tokens'] == 10
    assert ceo['missing']['input_tokens'] == 1
    assert ceo['output_tokens'] == 5 and ceo['cached_tokens'] == 6 and ceo['reasoning_tokens'] == 2
    assert ceo['raw_billed_cost_usd'] is None
    assert selected('pf', 'waste', 'growth')['total_tokens'] == 105
    assert selected('pf', 'retained', 'team')['group_wall_seconds'] == 7
    assert selected('pf', 'waste', 'team')['group_wall_seconds'] == 9
    assert selected('pf', 'retained', 'ops_finance')['input_tokens'] is None
    assert result['pairs'][0]['comparison']['input_tokens']['pf_over_git'] is None
    assert result['pairs'][0]['comparison']['output_tokens']['pf_minus_git'] == 0


def test_unstarted_runs_have_null_usage_and_time_in_table():
    result = summarize_pairs([dict(run_id='unstarted', group='git', seed=1049, repeat=1, status='not_started')])
    assert len(result['usage_table']) == 20
    assert all(r['input_tokens'] is None and r['model_wall_seconds'] is None and r['group_wall_seconds'] is None
               for r in result['usage_table'])


def test_unclassified_inherited_usage_remains_separate(tmp_path):
    unknown = aggregate([log(tmp_path / 'unknown.jsonl', receipts('unknown', 'growth', body(30, 4)))], PRICES)
    unknown['status'] = 'retained_membership_unknown'
    run = dict(run_id='pf', group='pf', seed=1048, repeat=1, status='paused', usage=dict(unclassified=unknown))
    table = summarize_pairs([run])['usage_table']
    row = next(r for r in table if r['scope'] == 'unclassified' and r['role'] == 'growth')
    assert row['input_tokens'] == 30 and row['output_tokens'] == 4
    assert row['classification_status'] == 'retained_membership_unknown'
    assert all(r['input_tokens'] is None for r in table if r['scope'] in ('retained', 'waste', 'pending'))
