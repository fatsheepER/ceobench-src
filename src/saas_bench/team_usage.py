"""Offline accounting of immutable team calls and physical HTTP attempts."""

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .model_usage import FIELDS, cost_usd, load_pricing, usage_values
from .role_policy import ROLES


CATEGORIES = ('formal', 'simulator', 'shared_d0', 'engineering', 'abandoned')
METRICS = (*FIELDS, 'total_tokens', 'uniform_cost_usd', 'raw_billed_cost_usd', 'recorded_cost_usd')


def _number(value, *, tokens=False):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Receipt values must be finite nonnegative numbers')
    if tokens and int(value) != value:
        raise ValueError('Token counts must be integers')
    return value


def _body(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else {}
    except ValueError:
        result = {}
        for line in value.splitlines():
            if not line.startswith('data:'):
                continue
            try:
                frame = json.loads(line[5:].strip())
            except ValueError:
                continue
            if not isinstance(frame, dict):
                continue
            frame = frame.get('message') or frame.get('response') or frame
            if not isinstance(frame, dict):
                continue
            if frame.get('model'):
                result['model'] = frame['model']
            if isinstance(frame.get('usage'), dict):
                result.setdefault('usage', {}).update(frame['usage'])
        return result


def _raw_cost(row, body):
    usage = body.get('usage') or {}
    for source, keys in ((row, ('raw_cost_usd', 'billed_cost_usd')),
                         (body, ('cost_usd', 'cost')), (usage, ('cost_usd', 'cost'))):
        for key in keys:
            if source.get(key) is not None:
                return _number(source[key])
    return None


def _summary(records):
    known = {field: None for field in METRICS}
    missing = dict.fromkeys(METRICS, 0)
    for record in records:
        for field in METRICS:
            value = record['metrics'][field]
            if value is None:
                missing[field] += 1
            else:
                known[field] = (known[field] or 0) + value
    return dict(billed_units=len(records), known=known, missing=missing,
                complete={field: known[field] if records and not missing[field] else None for field in METRICS},
                failed_http_attempts=sum(record['failed_http_attempt'] for record in records),
                unreturned_requests=len({r['call_id'] for r in records if r['unreturned']}),
                no_response_requests=len({r['call_id'] for r in records if r['no_response']}))


def _attempt_order(attempt):
    request = attempt.get('http_request', {})
    timestamp = request.get('timestamp') or attempt.get('http_response', {}).get('timestamp')
    at = datetime.fromisoformat(timestamp) if timestamp else datetime.min.replace(tzinfo=timezone.utc)
    if at.tzinfo is None:
        raise ValueError('Receipt timestamp requires a timezone')
    retry = request.get('sdk_retry')
    return at, int(retry) if retry is not None else -1


def aggregate(paths, prices, *, start_day=0, end_day=None, category='formal'):
    """Read logs without changing them; categories map to paths when a resource ledger is supplied.

    An HTTP attempt is the billing unit. Its final logical receipt supplies parsed
    fields, never a second bill. Frozen prices use the served model and request time.
    """
    if isinstance(prices, (str, Path)):
        prices = load_pricing(prices)
    if not isinstance(prices, dict) or not all(prices.get(key) for key in ('source', 'basis', 'rates')):
        raise ValueError('Frozen pricing requires source, basis and rates')
    if category not in CATEGORIES:
        raise ValueError('Unknown primary resource category: ' + category)
    groups = paths if isinstance(paths, Mapping) else {category: paths}
    calls, attempts, seen, duplicates = {}, {}, {}, 0
    for resource, files in groups.items():
        if resource not in CATEGORIES:
            raise ValueError('Unknown resource category: ' + resource)
        files = [files] if isinstance(files, (str, Path)) else files
        for path in files:
            with Path(path).open() as stream:
                for line_number, line in enumerate(stream, 1):
                    row = json.loads(line)
                    event = row.get('event')
                    if event not in ('request', 'response', 'http_request', 'http_response', 'http_error'):
                        raise ValueError('Unknown model receipt event: ' + str(event))
                    call_id, attempt_id = row.get('call_id'), row.get('attempt_id')
                    if not call_id or (event.startswith('http_') and not attempt_id):
                        raise ValueError('Immutable call and HTTP attempt IDs are required')
                    assigned = 'simulator' if resource in ('formal', 'engineering') and row.get('role') == 'simulator' else resource
                    key = (event, attempt_id if event.startswith('http_') else call_id)
                    value = (assigned, row)
                    if key in seen:
                        if seen[key] != value:
                            raise ValueError('Conflicting immutable model receipt: ' + str(key))
                        duplicates += 1
                        continue
                    seen[key] = value
                    call = calls.setdefault(call_id, dict(attempts=[], category=assigned, role=row.get('role')))
                    if call['category'] != assigned or call['role'] != row.get('role'):
                        raise ValueError('Call identity or category changed: ' + call_id)
                    row = dict(row, receipt_path=str(Path(path).resolve()), receipt_line=line_number)
                    if event.startswith('http_'):
                        attempt = attempts.setdefault(attempt_id, dict(call_id=call_id))
                        if attempt['call_id'] != call_id:
                            raise ValueError('HTTP attempt changed logical call: ' + attempt_id)
                        if attempt_id not in call['attempts']:
                            call['attempts'].append(attempt_id)
                        attempt[event] = row
                    else:
                        call[event] = row
    records, gaps = [], []
    for call_id, call in calls.items():
        request, response = call.get('request', {}), call.get('response')
        day = request.get('day')
        if day is None:
            gaps.append(dict(call_id=call_id, reason='missing_request_day'))
        elif day < start_day or (end_day is not None and day >= end_day):
            continue
        role = call['role']
        if call['category'] == 'formal' and role not in ROLES:
            raise ValueError('Formal receipt requires a team role: ' + str(role))
        ordered = sorted(call['attempts'], key=lambda key: (_attempt_order(attempts[key]), key))
        final_attempt = ordered[-1] if ordered else None
        if response and response.get('attempt_id'):
            final_attempt = response['attempt_id']
            if final_attempt not in ordered:
                raise ValueError('Logical response references an absent HTTP attempt')
        for attempt_id in ordered or [None]:
            attempt = attempts.get(attempt_id, {})
            http = attempt.get('http_response', {})
            http_request = attempt.get('http_request', {})
            logical = response if attempt_id == final_attempt else None
            body = _body(http.get('body')) if attempt_id else _body((logical or {}).get('response'))
            api = request.get('api') or (logical or {}).get('api')
            if api not in ('chat', 'responses', 'messages'):
                gaps.append(dict(call_id=call_id, attempt_id=attempt_id, reason='missing_api'))
                usage = dict.fromkeys(FIELDS)
            else:
                usage = usage_values(body, api)
            if logical:
                for field, value in (logical.get('usage') or {}).items():
                    if field in FIELDS and value is not None:
                        usage[field] = value
            usage = {field: _number(usage.get(field), tokens=True) for field in FIELDS}
            model = ((logical or {}).get('response') or {}).get('model') or body.get('model')
            timestamp = http_request.get('timestamp') or request.get('timestamp')
            rates = prices['rates'].get(model)
            timed = rates and any(key in rates for key in ('valid_from', 'valid_until', 'peak_schedule'))
            at = datetime.fromisoformat(timestamp) if timestamp else datetime.min.replace(tzinfo=timezone.utc)
            if timestamp and at.tzinfo is None:
                raise ValueError('Receipt timestamp requires a timezone')
            fee = cost_usd(usage, api, rates, at=at) if api and not (timed and not timestamp) else None
            failed = bool(attempt_id and (http.get('error') or http.get('status', 0) >= 400 or attempt.get('http_error')))
            if attempt_id and not http_request:
                gaps.append(dict(call_id=call_id, attempt_id=attempt_id, reason='missing_http_request'))
            if attempt_id and not http and not attempt.get('http_error'):
                gaps.append(dict(call_id=call_id, attempt_id=attempt_id, reason='unreturned_http_request'))
            metrics = dict(usage, total_tokens=(usage['input_tokens'] + usage['output_tokens']
                if usage['input_tokens'] is not None and usage['output_tokens'] is not None else None),
                uniform_cost_usd=fee, raw_billed_cost_usd=_raw_cost(http or logical or {}, body),
                recorded_cost_usd=_number((logical or {}).get('cost_usd')))
            records.append(dict(call_id=call_id, attempt_id=attempt_id, role=role,
                category=call['category'], day=day, model=model, api=api, metrics=metrics,
                failed_http_attempt=failed, unreturned=response is None,
                no_response=response is None or response.get('response') is None,
                request_receipt=dict(path=request.get('receipt_path'), line=request.get('receipt_line')),
                response_receipt=dict(path=(http or logical or {}).get('receipt_path'),
                    line=(http or logical or {}).get('receipt_line'))))
    team_records = [r for r in records if r['category'] == category and r['role'] in ROLES]
    return dict(pricing=prices, primary_category=category, team=_summary(team_records),
        roles={role: _summary([r for r in team_records if r['role'] == role]) for role in ROLES},
        resources={resource: _summary([r for r in records if r['category'] == resource]) for resource in CATEGORIES},
        all_resources=_summary(records), calls=len({r['call_id'] for r in records}),
        attempts=sum(r['attempt_id'] is not None for r in records), duplicates=duplicates,
        missing_receipts=gaps, failed_attempt_usage=_summary([r for r in records if r['failed_http_attempt']]),
        records=records)
