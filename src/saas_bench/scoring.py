"""Offline cash forecast scoring; never calls a model."""

import math

from .database import get_predictions


# Ledger categories written only by the agent's own purchase tools (tools.py market and
# group research, start_research_project). The agent acts on a week-start day only after
# that day's advance, so on the target day these entries follow the dashboard it saw.
WEEK_START_PURCHASES = ('market_research', 'group_research', 'research_project')


def cash_at_advance(conn, day):
    """Cash when the advance reaching `day` finished, before that day's new-week purchases."""
    marks = ','.join('?' * len(WEEK_START_PURCHASES))
    return conn.execute(f'SELECT COALESCE(SUM(amount), 0) FROM ledger WHERE day < ? '
                        f'OR (day = ? AND category NOT IN ({marks}))',
                        (day, day, *WEEK_START_PURCHASES)).fetchone()[0]


def interval_score(lower, upper, actual):
    if not all(math.isfinite(v) for v in (lower, upper, actual)) or lower > upper:
        raise ValueError('Finite ordered interval and actual cash are required')
    return upper - lower + 40 * max(lower - actual, actual - upper, 0)


def score_predictions(conn, current_day, *, outcome='running', planned_end_day=None,
                      submit_start=None, submit_end=None):
    rows = []
    actuals = {}
    for prediction in get_predictions(conn):
        row = dict(prediction)
        submitted = row['submit_day']
        if submit_start is not None and submitted < submit_start:
            continue
        if submit_end is not None and submitted > submit_end:
            continue
        target = submitted + row['horizon_days']
        row.update(target_day=target, status='pending', actual_cash=None, interval_score=None)
        if planned_end_day is not None and target > planned_end_day:
            row['status'] = 'excluded_after_planned_end'
        elif target > current_day:
            if outcome == 'bankrupt':
                row['status'] = 'unmatured_bankruptcy'
            elif outcome in ('failed', 'error', 'timeout'):
                row['status'] = 'unmatured_failure'
        else:
            if target not in actuals:
                actuals[target] = (cash_at_advance(conn, target), conn.execute(
                    'SELECT COALESCE(SUM(amount), 0) FROM ledger WHERE day <= ?', (target,)).fetchone()[0])
            actual, through_day = actuals[target]
            point, lower, upper = (row[k] for k in ('predicted_value', 'predicted_lower', 'predicted_upper'))
            # Scored against the dashboard cash at the target advance; the old end-of-day
            # ledger total (including that day's purchases) is kept for comparison.
            row.update(actual_cash=actual, ledger_cash_through_target_day=through_day)
            if any(v is None or not math.isfinite(v) for v in (point, lower, upper)) or not lower <= point <= upper:
                row['status'] = 'invalid_prediction'
            else:
                row.update(status='scored', interval_score=interval_score(lower, upper, actual),
                           absolute_error=abs(point - actual), covered=lower <= actual <= upper,
                           interval_width=upper - lower,
                           absolute_percentage_error=abs(point - actual) / abs(actual) if actual else None)
        rows.append(row)
    main = [row['interval_score'] for row in rows if row['horizon_days'] == 28 and row['status'] == 'scored']
    counts = {status: sum(row['status'] == status for row in rows) for status in sorted({row['status'] for row in rows})}
    return {'current_day': current_day, 'outcome': outcome, 'rows': rows, 'status_counts': counts,
            'four_week_count': len(main), 'four_week_mean_interval_score': sum(main) / len(main) if main else None}
