"""Private SDK call and HTTP attempt receipts; unknown usage stays unknown."""

from contextvars import ContextVar
from datetime import datetime, timezone
import base64
import functools
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

import httpx

from .run_lifecycle import RunCancelled, WorkerLifecycle


_CALL = ContextVar('model_call', default=None)
_DEADLINE = ContextVar('model_request_deadline', default=None)
_ATTEMPT = ContextVar('model_request_attempt', default=None)
MODEL_REQUEST_ATTEMPTS = 5
MODEL_RETRY_DELAYS = (10, 20, 40, 80)
MODEL_ATTEMPT_SECONDS = 600
MODEL_REQUEST_SECONDS = MODEL_REQUEST_ATTEMPTS * MODEL_ATTEMPT_SECONDS + sum(MODEL_RETRY_DELAYS)
FIELDS = ('input_tokens', 'output_tokens', 'cached_tokens', 'cache_creation_tokens', 'reasoning_tokens')
_SECRETS = {'authorization', 'proxy-authorization', 'api_key', 'api-key', 'x-api-key',
            'aws_access_key', 'aws_secret_key', 'aws_session_token'}


def check_model_call():
    if current := _CALL.get():
        current[0]._check_deadline()


def plain(value):
    if hasattr(value, 'model_dump'):
        value = value.model_dump(mode='json', exclude_unset=True)
    if isinstance(value, dict):
        return {k: '[REDACTED]' if k.lower() in _SECRETS else plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def response_error(response, api=None):
    """Provider errors can arrive with HTTP 200 and permissive SDK decoding."""
    if not isinstance(response, dict):
        return 'Model response must be an object' if api else None
    if response.get('error'):
        return 'Provider error: ' + json.dumps(response['error'], ensure_ascii=False)
    if api:
        field = {'chat': 'choices', 'responses': 'output', 'messages': 'content'}[api]
        items = response.get(field)
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            return f'Model response {field} must be a list of objects'
        if api == 'chat' and (not items or not isinstance(items[0].get('message'), dict)):
            return 'Model response choices must contain an assistant message'
    return None


def usage_values(response, api):
    raw = plain(response)
    raw = raw if isinstance(raw, dict) else {}
    usage = raw.get('usage') or {}
    chat = api == 'chat'
    anthropic = api == 'messages'
    input_tokens = usage.get('prompt_tokens' if chat else 'input_tokens')
    output_tokens = usage.get('completion_tokens' if chat else 'output_tokens')
    read = (usage.get('cache_read_input_tokens') if anthropic else
            (usage.get('prompt_tokens_details' if chat else 'input_tokens_details') or {}).get('cached_tokens'))
    if chat and read is None:
        read = usage.get('prompt_cache_hit_tokens')
    write = usage.get('cache_creation_input_tokens')
    if write is None and not anthropic:
        details = usage.get('prompt_tokens_details' if chat else 'input_tokens_details') or {}
        write = details.get('cache_creation_input_tokens', details.get('cache_write_tokens'))
    # Anthropic reports uncached input separately; the common input count includes caches.
    if anthropic:
        input_tokens = sum((input_tokens, read, write)) if all(v is not None for v in (input_tokens, read, write)) else None
    reasoning = (usage.get('completion_tokens_details' if chat else 'output_tokens_details') or {}).get('reasoning_tokens')
    return dict(zip(FIELDS, (input_tokens, output_tokens, read, write, reasoning)))


def usage_delta(previous, current):
    """Known subtotals and missing counts for calls made since a summary snapshot."""
    calls = current['calls'] - previous['calls']
    missing = {field: current['missing'][field] - previous['missing'][field] for field in FIELDS}
    known = {field: None if calls and missing[field] == calls else
             (current['known'][field] or 0) - (previous['known'][field] or 0) for field in FIELDS}
    return dict(calls=calls, known=known, missing=missing)


def summarize_usage_log(path, start_day=0, end_day=None):
    """Sum every response by its request's day in [start_day, end_day).

    Regenerations and failed calls count even when they produce no tool action.
    Missing usage remains separate from the known subtotals.
    """
    summary = dict(calls=0, errors=0, known=dict.fromkeys(FIELDS), missing=dict.fromkeys(FIELDS, 0),
                   known_cost_usd=None, missing_cost=0, unreturned_requests=0, costs=billing_totals())
    pending = {}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            call_id = row.get('call_id')
            if row['event'] == 'request':
                if row['day'] >= start_day and (end_day is None or row['day'] < end_day):
                    pending[call_id] = row.get('timestamp')
            elif row['event'] == 'response' and call_id in pending:
                timestamp = pending.pop(call_id)
                summary['calls'] += 1
                summary['errors'] += row.get('error') is not None
                usage = row.get('usage') or {}
                for field in FIELDS:
                    value = usage.get(field)
                    if value is None:
                        summary['missing'][field] += 1
                    else:
                        summary['known'][field] = (summary['known'][field] or 0) + value
                cost = row.get('cost_usd')
                if cost is None:
                    summary['missing_cost'] += 1
                else:
                    summary['known_cost_usd'] = (summary['known_cost_usd'] or 0) + cost
                add_billing(summary['costs'], billing_assessment(usage, row.get('api'), row.get('pricing'),
                    at=datetime.fromisoformat(timestamp) if timestamp else None, recorded_cost=cost))
    summary['unreturned_requests'] = len(pending)
    return summary


def load_pricing(path):
    """Load a sourced price table; the manifest stores its contents, not its path."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError('Pricing must be a JSON object')
    if not all(isinstance(data.get(k), str) and data[k].strip() for k in ('source', 'basis')):
        raise ValueError('Pricing requires source and basis')
    if not isinstance(data.get('rates'), dict) or not data['rates']:
        raise ValueError('Pricing requires model rates')
    for model, rates in data['rates'].items():
        if not model or not isinstance(rates, dict) or not rates:
            raise ValueError('Invalid model pricing')
        for key, value in rates.items():
            if key == 'peak_schedule':
                if value != 'weekday_01_04_06_10_utc':
                    raise ValueError('Unsupported peak pricing schedule')
            elif key in ('valid_from', 'valid_until'):
                if datetime.fromisoformat(value).tzinfo is None:
                    raise ValueError('Price validity requires a timezone')
            elif key not in ('input', 'output', 'cache_read', 'cache_write', 'peak_multiplier') or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Price must be a finite nonnegative USD/1k token rate')
        if 'valid_from' in rates and 'valid_until' in rates and datetime.fromisoformat(rates['valid_from']) >= datetime.fromisoformat(rates['valid_until']):
            raise ValueError('Invalid price validity interval')
    return data


def cost_usd(usage, api, rates, at=None):
    """Rates are explicit dollars per 1k tokens for this exact served model."""
    if not rates or any(usage[k] is None for k in ('input_tokens', 'output_tokens', 'cached_tokens')):
        return None
    at = at or datetime.now(timezone.utc)
    if ('valid_from' in rates and at < datetime.fromisoformat(rates['valid_from'])) or ('valid_until' in rates and at >= datetime.fromisoformat(rates['valid_until'])):
        return None
    write = usage['cache_creation_tokens'] if api == 'messages' else 0
    if write is None:
        return None
    counts = {'input': usage['input_tokens'] - usage['cached_tokens'] - write,
              'output': usage['output_tokens'], 'cache_read': usage['cached_tokens'], 'cache_write': write}
    if any(n < 0 or (n and k not in rates) for k, n in counts.items()):
        return None
    multiplier = 1
    if rates.get('peak_schedule') == 'weekday_01_04_06_10_utc':
        utc = at.astimezone(timezone.utc)
        if utc.weekday() < 5 and (1 <= utc.hour < 4 or 6 <= utc.hour < 10):
            multiplier = rates.get('peak_multiplier', 1)
    return multiplier * sum(n * rates.get(k, 0) / 1000 for k, n in counts.items())


def billing_assessment(usage, api, rates, *, at=None, recorded_cost=None):
    """Frozen uniform fees; raw billing fields remain unchanged."""
    usage = {field: usage.get(field) for field in FIELDS}
    complete = all(usage[field] is not None for field in ('input_tokens', 'output_tokens'))
    missing = ([field for field in ('cached_tokens', 'cache_creation_tokens') if usage[field] is None] +
               (['cost_usd'] if recorded_cost is None else [])) if complete else []
    result = dict(kind='unknown', usd=None, lower_usd=None, upper_usd=None, missing_fields=missing)
    timed = rates and any(key in rates for key in ('valid_from', 'valid_until', 'peak_schedule'))
    if not complete or not rates or api not in ('chat', 'responses', 'messages') or timed and at is None:
        return result
    at = at or datetime.min.replace(tzinfo=timezone.utc)
    if at.tzinfo is None:
        raise ValueError('Receipt timestamp requires a timezone')
    uncertain_read = usage['cached_tokens'] is None
    uncertain_write = api == 'messages' and usage['cache_creation_tokens'] is None
    if not uncertain_read and not uncertain_write:
        fee = cost_usd(usage, api, rates, at=at)
        if fee is not None:
            result.update(kind='exact' if recorded_cost is not None else 'recomputed', usd=fee)
        return result
    read = usage['cached_tokens']
    write = usage['cache_creation_tokens'] if api == 'messages' else 0
    if uncertain_read and uncertain_write:
        endpoints = [(0, 0), (usage['input_tokens'], 0), (0, usage['input_tokens'])]
    elif uncertain_read:
        endpoints = [(0, write), (usage['input_tokens'] - write, write)]
    else:
        endpoints = [(read, 0), (read, usage['input_tokens'] - read)]
    amounts = [cost_usd(dict(usage, cached_tokens=cached, cache_creation_tokens=created), api, rates, at=at)
               for cached, created in endpoints]
    if all(amount is not None for amount in amounts):
        result.update(kind='interval', lower_usd=min(amounts), upper_usd=max(amounts))
    return result


def billing_totals():
    return dict(exact=dict(calls=0, subtotal_usd=None), recomputed=dict(calls=0, subtotal_usd=None),
                interval=dict(calls=0, lower_usd=None, upper_usd=None), unknown_calls=0,
                missing_field_calls=0, unclassified_prior_calls=0,
                missing_fields=dict.fromkeys(('cached_tokens', 'cache_creation_tokens', 'cost_usd'), 0))


def add_billing(totals, cost):
    totals['missing_field_calls'] += bool(cost['missing_fields'])
    for field in cost['missing_fields']:
        totals['missing_fields'][field] += 1
    if cost['kind'] == 'unknown':
        totals['unknown_calls'] += 1
        return
    category = totals[cost['kind']]
    category['calls'] += 1
    if cost['kind'] == 'interval':
        for field in ('lower_usd', 'upper_usd'):
            category[field] = (category[field] or 0) + cost[field]
    else:
        category['subtotal_usd'] = (category['subtotal_usd'] or 0) + cost['usd']


class _Stream(httpx.SyncByteStream):
    def __init__(self, source, finish, require_done=False, check=lambda: None):
        self.source, self.finish = source, finish
        self.check = check
        self.chunks = []
        self.finished = False
        self.require_done = require_done

    def _finish(self, error=None):
        if not self.finished:
            self.finished = True
            self.finish(b''.join(self.chunks), error)

    def __iter__(self):
        try:
            for chunk in self.source:
                self.chunks.append(chunk)
                self.check()
                yield chunk
        except BaseException as exc:
            self._finish(type(exc).__name__)
            raise
        else:
            if self.require_done and b'data: [DONE]' not in b''.join(self.chunks).splitlines():
                self._finish('stream_closed_before_completion')
                raise httpx.ReadError('Chat completion stream ended before [DONE]')
            self._finish()

    def close(self):
        try:
            self.source.close()
        finally:
            lines = b''.join(self.chunks).rstrip().splitlines()
            completed = bool(lines and lines[-1] == b'data: [DONE]')
            self._finish(None if completed else 'stream_closed_before_completion')


class _ModelAttempt(WorkerLifecycle):
    def __init__(self, lifecycle, deadline):
        super().__init__()
        self.lifecycle, self.deadline = lifecycle, deadline

    def check(self, *, retry=False):
        self.lifecycle.check(retry=retry)
        if self.reason or time.monotonic() >= self.deadline:
            self.cancel('model_attempt_timeout')
            raise TimeoutError('Model request attempt exceeded its time budget')


class ModelUsage:
    def __init__(self, path, role, pricing=None, evidence_store=None, token_counter=None, identity=None, expected_model=None):
        self.identity = identity
        self.expected_model = expected_model
        self.message_ids = []
        self.world_state = lambda: None
        self.request_check = lambda: None
        self.call_world_state = None
        self.evidence_store = evidence_store
        self.token_counter = token_counter
        self.source_records = []
        self.context_id = uuid.uuid4().hex if evidence_store or identity else None
        self.last_request_event = None
        self.path = Path(path) if path else None
        self.role = role
        self.pricing = pricing or {}
        self.lock = threading.RLock()
        self.call_lock = threading.RLock()
        self.lifecycle = WorkerLifecycle()
        self._clients = []
        self.summary = {'calls': 0, 'errors': 0, 'known': {k: None for k in FIELDS},
                        'missing': {k: 0 for k in FIELDS}, 'known_cost_usd': None, 'missing_cost': 0,
                        'costs': billing_totals(),
                        'http_attempts': 0, 'failed_http_attempts': 0, 'failed_attempts_without_usage': 0}

    def write(self, event, **values):
        if self.path:
            entry = dict(event=event, role=self.role, timestamp=datetime.now(timezone.utc).isoformat(), **values)
            if self.identity:
                entry.update(self.identity.fields(), context_id=self.context_id, message_ids=list(self.message_ids),
                             world_state_id=self.call_world_state)
            with self.lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open('a') as stream:
                    stream.write(json.dumps(plain(entry), ensure_ascii=False, allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())

    def attach(self, client):
        """Keep SDK retry policy and its configured transport, including test transports."""
        http_client = getattr(client, '_client', None)
        if not isinstance(http_client, httpx.Client):
            return client
        if client not in self._clients:
            self._clients.append(client)
        if getattr(http_client, '_ceobench_recorded', False):
            return client
        send = http_client.send

        @functools.wraps(send)
        def recorded_send(request, *args, **kwargs):
            recorder, call_id = _CALL.get() or (self, None)
            recorder._check_deadline()
            recorder.request_check()
            if deadline := _DEADLINE.get():
                if attempt := _ATTEMPT.get():
                    deadline = min(deadline, attempt.deadline)
                remaining = max(.001, deadline - time.monotonic())
                request.extensions['timeout'] = {phase: min(value, remaining) if value is not None else remaining
                    for phase, value in request.extensions.get('timeout', {}).items()}
            attempt_id = uuid.uuid4().hex
            failure_recorded = False
            with recorder.lock:
                recorder.summary['http_attempts'] += 1
            raw_body = request.read()
            capture_event = None
            capture_done = False
            if recorder.evidence_store:
                from .execution_capture import model_request
                from .run_lifecycle import RunCancelled
                from .sql_evidence import EvidencePending
                # Required evidence is durable before sending a model request.
                try:
                    capture_event = model_request(recorder.evidence_store, raw_body, recorder.source_records,
                                                  call_id, attempt_id, recorder.context_id, recorder.message_ids)
                except EvidencePending as exc:
                    raise RunCancelled('sql_evidence_pending') from exc
                except Exception as exc:
                    recorder.evidence_store.fail(exc)
                    raise
            body = raw_body.decode('utf-8')
            try:
                body = json.loads(body)
            except ValueError:
                pass
            recorder.write('http_request', call_id=call_id, attempt_id=attempt_id,
                       method=request.method, endpoint=str(request.url.copy_with(query=None, username='', password='')),
                       client_headers={k: request.headers[k] for k in ('user-agent', 'x-opencode-session') if k in request.headers},
                       sdk_retry=request.headers.get('x-stainless-retry-count'), body=body)
            try:
                # Read through our stream wrapper, so interrupted SSE bodies are retained too.
                streaming = kwargs.get('stream', False)
                response = send(request, *args, **dict(kwargs, stream=True))
                def finish(content, error):
                    nonlocal failure_recorded, capture_done
                    try:
                        reported = json.loads(content)
                    except (ValueError, UnicodeError):
                        reported = None
                    error = error or response_error(reported)
                    if capture_event and not capture_done:
                        capture_done = True
                        try:
                            recorder.evidence_store.complete(capture_event, 'failed' if error or response.status_code >= 400 else 'succeeded',
                                send_state='response_received', http_status=response.status_code, response_error=error)
                            if not error and response.status_code < 400:
                                recorder.last_request_event = capture_event
                        except Exception as exc:
                            recorder.evidence_store.fail(exc)
                    try:
                        body = content.decode('utf-8')
                    except UnicodeDecodeError:
                        body = {'base64': base64.b64encode(content).decode('ascii')}
                    if response.status_code >= 400 or error:
                        with recorder.lock:
                            recorder.summary['failed_http_attempts'] += 1
                            recorder.summary['failed_attempts_without_usage'] += not bool(
                                reported.get('usage') if isinstance(reported, dict) else None)
                        failure_recorded = True
                    recorder.write('http_response', call_id=call_id, attempt_id=attempt_id,
                               status=response.status_code, request_id=response.headers.get('request-id') or response.headers.get('x-request-id'),
                               body=body, error=error)
                if response.is_stream_consumed:
                    finish(response.content, None)
                else:
                    require_done = (streaming and request.url.path.endswith('/chat/completions')
                                    and response.headers.get('content-type', '').startswith('text/event-stream'))
                    response.stream = _Stream(response.stream, finish, require_done=require_done,
                                              check=recorder._check_deadline)
                    if not streaming:
                        response.read()
                return response
            except BaseException as exc:
                if capture_event and not capture_done:
                    try:
                        recorder.evidence_store.complete(capture_event, 'failed', send_state='unknown', error=type(exc).__name__)
                    except Exception as capture_error:
                        recorder.evidence_store.fail(capture_error)
                if not failure_recorded:
                    with recorder.lock:
                        recorder.summary['failed_http_attempts'] += 1
                        recorder.summary['failed_attempts_without_usage'] += 1
                recorder.write('http_error', call_id=call_id, attempt_id=attempt_id, error=type(exc).__name__)
                raise

        http_client.send = recorded_send
        http_client._ceobench_recorded = True
        return client

    def call(self, api, request, invoke, **context):
        deadline = time.monotonic() + MODEL_REQUEST_SECONDS
        token = _DEADLINE.set(deadline)
        acquired = False
        policies = []
        try:
            while not acquired:
                self.lifecycle.check()
                if time.monotonic() >= deadline:
                    raise TimeoutError('Model call lock exceeded the request time budget')
                acquired = self.call_lock.acquire(timeout=max(.001, min(.1, deadline - time.monotonic())))
            self._check_deadline()
            for client in self._clients:
                policies.append((client, client.max_retries))
                client.max_retries = 0
                self.lifecycle.watch_client(client)
            with self.lifecycle.model_call():
                return self._call(api, request, invoke, **context)
        finally:
            for client, retries in policies:
                client.max_retries = retries
            if acquired:
                self.call_lock.release()
            _DEADLINE.reset(token)

    def _check_deadline(self):
        self.lifecycle.check()
        if attempt := _ATTEMPT.get():
            attempt.check()

    def _invoke_attempt(self, invoke):
        remaining = max(.001, min(MODEL_ATTEMPT_SECONDS, _DEADLINE.get() - time.monotonic()))
        attempt = _ModelAttempt(self.lifecycle, time.monotonic() + remaining)
        token = _ATTEMPT.set(attempt)
        timer = threading.Timer(remaining, attempt.cancel, args=('model_attempt_timeout',))
        timer.daemon = True
        hooks = []
        try:
            for client in self._clients:
                http = client._client
                previous = getattr(http, '_ceobench_lifecycle', None)
                attempt.watch_client(client)
                hooks.append((http, http.event_hooks['request'][-1], previous))
            timer.start()
            with attempt.model_call():
                return invoke()
        finally:
            timer.cancel()
            if timer.ident is not None:
                timer.join(timeout=1)
            for http, hook, previous in hooks:
                http.event_hooks['request'].remove(hook)
                http._ceobench_lifecycle = previous
            _ATTEMPT.reset(token)

    def _call(self, api, request, invoke, **context):
        self.request_check()
        self.last_request_event = None
        self.call_world_state = self.world_state()
        replacements = []
        if self.evidence_store:
            from .execution_capture import text_sources
            from .pf_read import prepare_request
            try:
                replacements = prepare_request(self.evidence_store, request, self.context_id, self.token_counter)
            except Exception as exc:
                self.evidence_store.fail(exc)
                raise
            self.source_records = text_sources(request)
        call_id = uuid.uuid4().hex
        started = datetime.now(timezone.utc)
        token = _CALL.set((self, call_id))
        response, error, model_attempts, last_model_error = None, None, 0, None
        self.write('request', call_id=call_id, api=api, request=request, **context)
        try:
            for model_attempts in range(1, MODEL_REQUEST_ATTEMPTS + 1):
                self._check_deadline()
                response = None
                try:
                    response = self._invoke_attempt(invoke)
                    self._check_deadline()
                    raw = plain(response)
                    if message := response_error(raw, api):
                        raise ValueError(message)
                    if self.expected_model is not None and raw.get('model') != self.expected_model:
                        raise ValueError('Returned model differs from frozen configuration')
                    return response
                except Exception as exc:
                    last_model_error = type(exc).__name__
                    self.lifecycle.check()
                    if model_attempts == MODEL_REQUEST_ATTEMPTS:
                        raise RunCancelled('model_request_attempt_limit') from exc
                    self.lifecycle.sleep(MODEL_RETRY_DELAYS[model_attempts - 1])
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            try:
                raw = plain(response)
                usage = usage_values(raw, api)
                served_model = raw.get('model') if isinstance(raw, dict) else None
                cost = cost_usd(usage, api, self.pricing.get(served_model), at=started)
                billing = billing_assessment(usage, api, self.pricing.get(served_model),
                                             at=started, recorded_cost=cost)
                with self.lock:
                    summary = self.summary
                    totals = summary.setdefault('costs', dict(billing_totals(),
                                                              unclassified_prior_calls=summary['calls']))
                    summary['calls'] += 1
                    summary['errors'] += error is not None
                    for field, value in usage.items():
                        if value is None:
                            summary['missing'][field] += 1
                        else:
                            summary['known'][field] = (summary['known'][field] or 0) + value
                    if cost is None:
                        summary['missing_cost'] += 1
                    else:
                        summary['known_cost_usd'] = (summary['known_cost_usd'] or 0) + cost
                    add_billing(totals, billing)
                    self.write('response', call_id=call_id, api=api, response=raw, usage=usage,
                               error=error, cost_usd=cost, pricing=self.pricing.get(served_model),
                               cost=billing, request_timestamp=started.isoformat(),
                               model_attempts=model_attempts, last_model_error=last_model_error)
            finally:
                _CALL.reset(token)
                if replacements:
                    from .pf_read import restore_request
                    restore_request(request, replacements)
