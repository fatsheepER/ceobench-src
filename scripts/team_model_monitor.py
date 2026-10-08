from pathlib import Path

import round5


def poll(output, state):
    output = Path(output)
    offsets = state.setdefault('offsets', {})
    seen = {tuple(key) for key in state.get('seen_receipts', [])}
    errors = []
    for path in sorted((output / 'engineering/runs').glob('*/attempt-*/runtime/private/**/*model-usage.jsonl')):
        rows, offset = round5.read_increment(path, offsets.get(str(path), 0))
        for row in rows:
            event = row['event']
            key = (event, row.get('attempt_id') if event.startswith('http_') else row.get('call_id'))
            if not key[1]:
                raise ValueError('Immutable model receipt ID is required')
            issue = round5.model_issue(row)
            missing_cost = event == 'response' and not row.get('error') and row.get('cost_usd') is None
            if issue in ('model_route_changed', 'simulator_thinking_changed', 'served_model_changed') or missing_cost:
                (output / 'engineering/HOLD').touch()
            if key in seen:
                continue
            seen.add(key)
            if issue or event in ('http_error', 'capture_failed', 'model_call_error'):
                errors.append(dict(path=str(path), event=event, issue=issue,
                                   call_id=row.get('call_id'), attempt_id=row.get('attempt_id')))
            if missing_cost:
                errors.append(dict(path=str(path), event='missing_uniform_cost', call_id=row.get('call_id')))
        offsets[str(path)] = offset
    state['seen_receipts'] = sorted(seen)
    return errors
