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
import uuid

import httpx


_CALL = ContextVar('model_call', default=None)
FIELDS = ('input_tokens', 'output_tokens', 'cached_tokens', 'cache_creation_tokens', 'reasoning_tokens')
_SECRETS = {'authorization', 'proxy-authorization', 'api_key', 'api-key', 'x-api-key',
            'aws_access_key', 'aws_secret_key', 'aws_session_token'}


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
            if key in ('valid_from', 'valid_until'):
                if datetime.fromisoformat(value).tzinfo is None:
                    raise ValueError('Price validity requires a timezone')
            elif key not in ('input', 'output', 'cache_read', 'cache_write') or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
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
    return sum(n * rates.get(k, 0) / 1000 for k, n in counts.items())


class _Stream(httpx.SyncByteStream):
    def __init__(self, source, finish):
        self.source, self.finish = source, finish
        self.chunks = []
        self.finished = False

    def _finish(self, error=None):
        if not self.finished:
            self.finished = True
            self.finish(b''.join(self.chunks), error)

    def __iter__(self):
        try:
            for chunk in self.source:
                self.chunks.append(chunk)
                yield chunk
        except BaseException as exc:
            self._finish(type(exc).__name__)
            raise
        else:
            self._finish()

    def close(self):
        try:
            self.source.close()
        finally:
            self._finish('stream_closed_before_completion')


class ModelUsage:
    def __init__(self, path, role, pricing=None, evidence_store=None, token_counter=None):
        self.evidence_store = evidence_store
        self.token_counter = token_counter
        self.source_records = []
        self.context_id = uuid.uuid4().hex if evidence_store else None
        self.path = Path(path) if path else None
        self.role = role
        self.pricing = pricing or {}
        self.lock = threading.RLock()
        self.summary = {'calls': 0, 'errors': 0, 'known': {k: None for k in FIELDS},
                        'missing': {k: 0 for k in FIELDS}, 'known_cost_usd': None, 'missing_cost': 0,
                        'http_attempts': 0, 'failed_http_attempts': 0, 'failed_attempts_without_usage': 0}

    def write(self, event, **values):
        if self.path:
            entry = dict(event=event, role=self.role, timestamp=datetime.now(timezone.utc).isoformat(), **values)
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
        if getattr(http_client, '_ceobench_recorded', False):
            return client
        send = http_client.send

        @functools.wraps(send)
        def recorded_send(request, *args, **kwargs):
            recorder, call_id = _CALL.get() or (self, None)
            attempt_id = uuid.uuid4().hex
            failure_recorded = False
            with recorder.lock:
                recorder.summary['http_attempts'] += 1
            raw_body = request.read()
            capture_event = None
            capture_done = False
            if recorder.evidence_store:
                from .execution_capture import model_request
                # Required evidence is durable before sending a model request.
                try:
                    capture_event = model_request(recorder.evidence_store, raw_body, recorder.source_records,
                                                  call_id, attempt_id, recorder.context_id)
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
                    response.stream = _Stream(response.stream, finish)
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
        response, error = None, None
        self.write('request', call_id=call_id, api=api, request=request, **context)
        try:
            response = invoke()
            if message := response_error(plain(response), api):
                raise ValueError(message)
            return response
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            try:
                raw = plain(response)
                usage = usage_values(raw, api)
                served_model = raw.get('model') if isinstance(raw, dict) else None
                cost = cost_usd(usage, api, self.pricing.get(served_model), at=started)
                with self.lock:
                    summary = self.summary
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
                    self.write('response', call_id=call_id, api=api, response=raw, usage=usage,
                               error=error, cost_usd=cost, pricing=self.pricing.get(served_model))
            finally:
                _CALL.reset(token)
                if replacements:
                    from .pf_read import restore_request
                    restore_request(request, replacements)
