"""Offline team outcomes and paired comparisons with explicit missing values."""

from .database import get_cash, get_mrr
from .scoring import cash_at_advance, score_predictions


PRIMARY_METRICS = ('input_tokens', 'output_tokens', 'total_tokens', 'uniform_cost_usd')


def summarize_run(conn, *, run_id, group, seed, repeat, day, reason, usage,
                  attempt=None, stop_after_day=112, wall_seconds=None):
    if group not in ('git', 'pf'):
        raise ValueError('Unknown experiment group')
    cash = get_cash(conn)
    if reason in ('failed', 'error', 'timeout', 'world_operation_failed'):
        status, scoring_outcome = 'technical_failure', 'failed'
    elif reason in ('paused', 'hold', 'cancelled'):
        status, scoring_outcome = 'paused', 'running'
    elif reason in ('bankrupt', 'natural_end') and cash < 0:
        status, scoring_outcome = 'natural_bankruptcy', 'bankrupt'
    elif day >= stop_after_day and reason in ('observation_end', 'natural_end', 'completed'):
        status, scoring_outcome = 'reached_d112' if stop_after_day == 112 else 'reached_stop_day', 'completed'
    else:
        raise ValueError('Run reason and reached day do not describe a terminal or paused outcome')
    predictions = score_predictions(conn, day, outcome=scoring_outcome)
    unexpired = [row for row in predictions['rows'] if row['target_day'] > day]
    return dict(run_id=run_id, group=group, seed=seed, repeat=repeat, attempt=attempt,
        status=status, day=day, stop_reason=reason, stop_after_day=stop_after_day,
        cash_at_advance=cash_at_advance(conn, day), mrr=get_mrr(conn), wall_seconds=wall_seconds,
        usage=usage, predictions=predictions, unexpired_predictions=unexpired)


def summarize_pairs(runs):
    """Keep every assigned run and compare only complete primary metrics."""
    runs = list(runs)
    indexed = {}
    for run in runs:
        key = (run['seed'], run['repeat'])
        pair = indexed.setdefault(key, {})
        if run['group'] not in ('git', 'pf') or run['group'] in pair:
            raise ValueError('A pair requires exactly one assigned run per group')
        pair[run['group']] = run
    pairs = []
    for (seed, repeat), pair in sorted(indexed.items()):
        git, pf = pair.get('git'), pair.get('pf')
        comparison = {}
        for metric in PRIMARY_METRICS:
            values = {group: (run.get('usage') or {}).get('team', {}).get('complete', {}).get(metric)
                for group, run in pair.items()}
            left, right = values.get('git'), values.get('pf')
            missing = [group for group, value in (('git', left), ('pf', right)) if value is None]
            comparison[metric] = dict(pf_minus_git=right - left if not missing else None,
                pf_over_git=right / left if not missing and left != 0 else None,
                missing_groups=missing, ratio_reason='missing_metric' if missing else
                    ('zero_git_denominator' if left == 0 else None))
        complete = bool(git and pf and git['status'] == pf['status'] == 'reached_d112')
        bankruptcy = bool(git and pf and 'natural_bankruptcy' in (git['status'], pf['status']))
        common_day = min(git['day'], pf['day']) // 7 * 7 if bankruptcy and all(
            run.get('day') is not None for run in (git, pf)) else None
        pairs.append(dict(seed=seed, repeat=repeat, run_ids={g: r['run_id'] for g, r in pair.items()},
            statuses={g: r['status'] for g, r in pair.items()}, complete_d112=complete,
            common_completed_week_end=common_day,
            comparison=comparison, missing_groups=[g for g in ('git', 'pf') if g not in pair]))
    return dict(runs=runs, pairs=pairs, allocated_runs=len(runs), complete_d112_pairs=sum(p['complete_d112'] for p in pairs),
        status_counts={status: sum(r['status'] == status for r in runs) for status in sorted({r['status'] for r in runs})},
        by_seed={str(seed): [p for p in pairs if p['seed'] == seed] for seed in sorted({key[0] for key in indexed})})
