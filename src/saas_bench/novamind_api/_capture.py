"""Append-only observations for standard clients; no evidence read capability."""
import builtins
from contextvars import ContextVar
from functools import wraps
import io
import http.client
import socket
import json
import os
import sys
import urllib.error
import urllib.request
import uuid

_CALLS = ContextVar('novamind_observations', default=None)
_CONTEXT = 'NOVAMIND_CAPTURE_CONTEXT'


class _UnixConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        if self.timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
            self.sock.settimeout(self.timeout)
        self.sock.connect(os.environ['NOVAMIND_API_SOCKET'])


class _UnixHandler(urllib.request.HTTPHandler):
    def http_open(self, request):
        return self.do_open(_UnixConnection, request)


def _transport(request, *args, **kwargs):
    if os.environ.get('NOVAMIND_API_SOCKET'):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}), _UnixHandler()).open(request, *args, **kwargs)
    return urllib.request.urlopen(request, *args, **kwargs)


def submit(body):
    port = os.environ.get('NOVAMIND_API_PORT') or ('1' if os.environ.get('NOVAMIND_API_SOCKET') else None)
    if not port:
        return False
    request = urllib.request.Request(f'http://127.0.0.1:{int(port)}/_capture',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    try:
        with _transport(request, timeout=5) as response:
            return json.load(response).get('accepted') is True
    except Exception:
        # Server-side expected receipt / unfinished execution prevents a healthy checkpoint.
        return False


def observed(fn):
    @wraps(fn)
    def call(*args, **kwargs):
        context = os.environ.get(_CONTEXT)
        if not context:
            return fn(*args, **kwargs)
        state = _CALLS.get()
        outer = state is None
        state = state if state is not None else {'calls': [], 'projection': []}
        start = len(state['calls'])
        token = _CALLS.set(state) if outer else None
        result, error = None, None
        try:
            result = fn(*args, **kwargs)
            return result
        except BaseException as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
            raise
        finally:
            for item in state['calls'][start:]:
                record = item['record']
                record.setdefault('transformations', []).append(dict(operation=fn.__qualname__, returned=result, error=error))
                if not fn.__module__.endswith(('_public_cli', 'novamind_cli')) or fn.__name__ == '_api_call':
                    record.update(returned=result, error=error)
            if outer:
                _CALLS.reset(token)
                for item in state['calls']:
                    record = dict(item['record'], projection=state['projection'])
                    submit(dict(context=context, call=item['call'], record=record))
    return call


def urlopen(request, *args, **kwargs):
    state = _CALLS.get()
    context = os.environ.get(_CONTEXT)
    if state is None or not context or request.full_url.rsplit('/', 1)[-1] in ('health', 'game-status', 'checkpoint'):
        return _transport(request, *args, **kwargs)
    call = uuid.uuid4().hex
    request.add_header('X-Capture-Context', context)
    request.add_header('X-Capture-Call', call)
    record = {'state': 'not_read'}
    state['calls'].append(dict(call=call, record=record))
    def wrap(response):
        read = response.read
        def observed_read(*a, **k):
            try:
                data = read(*a, **k)
            except BaseException as exc:
                partial = getattr(exc, 'partial', b'')
                record.update(state='partial', body=partial.hex())
                raise
            record.update(state='received', body=record.get('body', '') + data.hex())
            try:
                record['parsed'] = json.loads(bytes.fromhex(record['body']))
            except (ValueError, UnicodeError):
                pass
            return data
        response.read = observed_read
        return response
    try:
        return wrap(_transport(request, *args, **kwargs))
    except urllib.error.HTTPError as exc:
        wrap(exc)
        raise
    except BaseException:
        record['state'] = 'transport_error'
        raise


def print(*args, **kwargs):
    state = _CALLS.get()
    target = kwargs.get('file') or sys.stdout
    if state is not None and target in (sys.stdout, sys.stderr):
        text = io.StringIO()
        builtins.print(*args, **dict(kwargs, file=text))
        try:
            sink = os.fstat(target.fileno())
            sink = [sink.st_dev, sink.st_ino]
        except (OSError, AttributeError, io.UnsupportedOperation):
            sink = None
        state['projection'].append({'stream': 'stderr' if target is sys.stderr else 'stdout',
                                    'text': text.getvalue(), 'sink': sink})
    return builtins.print(*args, **kwargs)


def script_start(code, source, env):
    parent = os.environ.get(_CONTEXT)
    if not parent:
        return None
    token = uuid.uuid4().hex
    if submit(dict(context=parent, call=token, record=dict(kind='python_start', code=code, source=source))):
        env[_CONTEXT] = token
        return parent, token
    return None


def script_end(identity, stdout, stderr, exit_code, error=None):
    if identity:
        parent, token = identity
        submit(dict(context=parent, call=token, record=dict(kind='python_end', stdout=stdout.hex(),
               stderr=stderr.hex(), exit_code=exit_code, error=error)))


def decode_output(raw):
    with io.TextIOWrapper(io.BytesIO(raw)) as stream:
        return stream.read()
