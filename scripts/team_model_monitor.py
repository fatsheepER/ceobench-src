from datetime import datetime
from pathlib import Path

import round5
from saas_bench.model_usage import add_billing, billing_assessment, billing_totals


def record_billing(row, state, group, path):
    if row['event'] == 'request':
        state.setdefault('billing_requests', {})[row['call_id']] = row.get('timestamp')
    if row['event'] != 'response':
        return None
    receipts = state.setdefault('billing_groups', {}).setdefault(group, {}).setdefault('receipts', {})
    if row['call_id'] in receipts:
        return None
    timestamp = row.get('request_timestamp') or state.get('billing_requests', {}).get(row['call_id'])
    cost = billing_assessment(row.get('usage') or {}, row.get('api'), row.get('pricing'),
        at=datetime.fromisoformat(timestamp) if timestamp else None, recorded_cost=row.get('cost_usd'))
    fields = cost['missing_fields'] if round5.model_issue(row) is None else []
    receipts[row['call_id']] = dict(timestamp=row.get('timestamp') or timestamp or '',
        call_id=row['call_id'], role=row.get('role'), path=str(path), cost=cost, missing_fields=fields)
    return group, row['call_id']


def billing_warnings(state, changed):
    """Replay response chronology across role logs and copied attempt prefixes."""
    changed = {tuple(key) for key in changed}
    warnings = []
    for group, summary in state.get('billing_groups', {}).items():
        calls = missing = streak = 0
        costs = billing_totals()
        for receipt in sorted(summary['receipts'].values(), key=lambda row: (row['timestamp'], row['call_id'])):
            calls += 1
            missing += bool(receipt['missing_fields'])
            streak = streak + 1 if receipt['missing_fields'] else 0
            add_billing(costs, receipt['cost'])
            triggers = []
            if missing * 20 > calls:
                triggers.append('missing_over_five_percent')
            if streak >= 20:
                triggers.append('consecutive_missing_20')
            if receipt['missing_fields'] and triggers and (group, receipt['call_id']) in changed:
                warnings.append(dict(receipt, event='billing_fields_missing', group=group,
                    calls=calls, missing_calls=missing, consecutive_missing=streak, triggers=triggers))
        summary.update(calls=calls, missing_calls=missing, consecutive_missing=streak, costs=costs)
    return warnings


def finish_billing(state, complete):
    state['billing_coverage'] = 'complete' if complete else 'rebuilding'
    notices = billing_warnings(state, state.get('billing_pending', []) if complete else [])
    if complete:
        state['billing_pending'] = []
        state.pop('billing_rebuild', None)
    return notices


def poll(output, state):
    output = Path(output)
    offsets = state.setdefault('offsets', {})
    seen = {tuple(key) for key in state.get('seen_receipts', [])}
    if 'billing_groups' not in state and seen:
        offsets.update(dict.fromkeys(offsets, 0))
        state['billing_rebuild'] = True
    state.setdefault('billing_groups', {})
    errors = []
    changed = state.setdefault('billing_pending', [])
    complete = True
    for path in sorted((output / 'engineering/runs').glob('*/attempt-*/runtime/private/**/*model-usage.jsonl')):
        rows, offset = round5.read_increment(path, offsets.get(str(path), 0))
        for row in rows:
            event = row['event']
            key = (event, row.get('attempt_id') if event.startswith('http_') else row.get('call_id'))
            if not key[1]:
                raise ValueError('Immutable model receipt ID is required')
            issue = round5.model_issue(row)
            if key in seen:
                if state.get('billing_rebuild'):
                    group = path.relative_to(output / 'engineering/runs').parts[0]
                    record_billing(row, state, group, path)
                continue
            seen.add(key)
            group = path.relative_to(output / 'engineering/runs').parts[0]
            if recorded := record_billing(row, state, group, path):
                changed.append(recorded)
            if issue or event in ('http_error', 'capture_failed', 'model_call_error'):
                errors.append(dict(path=str(path), event=event, issue=issue,
                                   call_id=row.get('call_id'), attempt_id=row.get('attempt_id')))
        offsets[str(path)] = offset
        complete &= offset == path.stat().st_size
    state['seen_receipts'] = sorted(seen)
    return errors + finish_billing(state, complete)
