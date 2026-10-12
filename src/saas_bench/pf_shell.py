"""A per-Bash PF service. The sandbox receives only a client and this socket."""
import json
import os
import re
from pathlib import Path, PurePosixPath
import socket
import socketserver
import tempfile
import threading
import time

from . import pf_cli
from .evidence_handles import RAW_VERSION
from .execution_capture import CapturedText, CURRENT_EVENT, ExecutionCapture
from .pf_deadline import Budget, budget_scope


CLIENT = '''import json, os, socket, sys
channel = socket.socket(socket.AF_UNIX)
address = os.environ['PF_SOCKET']
parent_fd = None
if len(os.fsencode(address)) >= 108:
    parent, name = os.path.split(address)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    address = '/proc/self/fd/%d/%s' % (parent_fd, name)
try:
    channel.connect(address)
finally:
    if parent_fd is not None: os.close(parent_fd)
sink = os.fstat(1)
channel.sendall((json.dumps(dict(argv=sys.argv[1:], cwd=os.getcwd(),
                               stdout_sink=[sink.st_dev, sink.st_ino])) + '\\n').encode())
reply = json.load(channel.makefile('rb'))
channel.close()
try:
    sys.stdout.write(reply['stdout'])
    sys.stdout.flush()
except BrokenPipeError:
    os._exit(141)
sys.stderr.write(reply['stderr'])
sys.exit(reply['code'])
'''


class ShellService:
    def __init__(self, executor):
        self.executor = executor
        self.capture = executor.capture
        self.calls, self.outputs = [], []
        self.cancelled = threading.Event()
        self.deadline = time.monotonic() + executor.bash_timeout
        self.check_cancelled = None
        self.connection = None
        self.directory = tempfile.TemporaryDirectory(prefix='ceobench-pf-', dir=getattr(executor, 'temp_root', None))
        self.root = Path(self.directory.name)
        self.guest = '/opt/pf' if executor._bwrap() else str(self.root)
        client = self.root / 'pf'
        try:
            client.write_text('#!' + executor.python + '\n' + CLIENT)
            client.chmod(0o555)
        except BaseException:
            self.directory.cleanup()
            raise
        service = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                service.connection = self.connection
                try:
                    budget = Budget(service.deadline, cancelled=service.cancelled,
                                    check=service.check_cancelled)
                    with budget_scope(budget=budget) as budget:
                        data = bytearray()
                        while not data.endswith(b'\n') and len(data) <= 65536:
                            self.connection.settimeout(min(30, budget.remaining()))
                            chunk = self.connection.recv(4096)
                            if not chunk:
                                break
                            data.extend(chunk)
                        if len(data) > 65536:
                            raise ValueError('PF request too large')
                        response = service.call(json.loads(bytes(data)), budget)
                except Exception as exc:
                    response = dict(stdout='', stderr='pf: ' + str(exc) + '\n',
                                    code=124 if isinstance(exc, TimeoutError) else 2)
                try:
                    self.wfile.write(json.dumps(response).encode())
                except OSError:
                    pass  # The child result remains captured even if its pipe consumer exits.
                finally:
                    service.connection = None

        class Server(socketserver.UnixStreamServer):
            def server_bind(server):
                from .runtime_temp import bind_unix
                bind_unix(server.socket, server.server_address)
        try:
            self.server = Server(str(self.root / 'socket'), Handler)
        except BaseException:
            self.directory.cleanup()
            raise
        self.server.timeout = .02
        self.thread = threading.Thread(target=self._serve, daemon=True)
        try:
            self.thread.start()
        except BaseException:
            self.server.server_close()
            self.directory.cleanup()
            raise

    def _serve(self):
        while not self.cancelled.is_set():
            self.server.handle_request()

    def set_deadline(self, deadline, check=None):
        self.deadline, self.check_cancelled = deadline, check

    def call(self, request, budget=None):
        argv, cwd = request.get('argv'), request.get('cwd')
        if not isinstance(argv, list) or any(not isinstance(v, str) for v in argv):
            raise ValueError('PF argv must be strings')
        if not isinstance(cwd, str):
            raise ValueError('PF working directory is required')
        root = PurePosixPath(self.executor.guest_root)
        directory = PurePosixPath(os.path.normpath(cwd))
        relative = PurePosixPath(os.path.relpath(directory, root))
        operation, args, error = 'pf_usage', {}, None
        try:
            operation, args = pf_cli.parse_argv(argv)
            if not directory.is_relative_to(root) and operation != 'pf_help':
                raise ValueError('PF working directory must be inside the workspace')
            for key in ('target', 'baseline'):
                target = args.get(key, {})
                field = next((f for f in ('path', 'version') if f in target), None)
                if field and relative.parts and not RAW_VERSION.fullmatch(target[field]) and target[field].split(':', 1)[0] not in ('ceo', 'growth', 'ops_finance'):
                    value = target[field]
                    # Numbered outputs/texts are global; file and script names follow cwd.
                    if not re.fullmatch(r'(?:r\d+(?:\.\d+)?|(?:cmd|query|read|receipt)\d+(?:@v\d+)?)', value):
                        target[field] = os.path.normpath(str(relative / value))
        except ValueError as exc:
            error = str(exc)
        capture = ExecutionCapture(self.executor.evidence_store)
        parent_args = self.capture.store.read_event(self.capture.event)['request']['request']
        audit = dict(operation=operation, arguments=args, event_id=None, outcome='running',
                     argv=argv, cwd=str(relative), exit_code=None, note=parent_args.get('note'))
        self.calls.append(audit)
        capture.begin(operation, args, parent=self.capture.event, argv=argv, cwd=str(relative),
                      command=parent_args.get('command'), note=parent_args.get('note'))
        audit['event_id'] = capture.event
        token = CURRENT_EVENT.set(capture.event)
        code, outcome = 0, 'succeeded'
        try:
            if error:
                result, code, outcome = error, 2, 'usage_error'
            elif operation == 'pf_help':
                result = pf_cli.HELP
            else:
                try:
                    result = self.executor.pf_queries.execute(operation, args)
                    if budget:
                        budget.check()
                except TimeoutError as exc:
                    result, code, outcome = 'Error: ' + str(exc), 124, 'timed_out'
                except (ValueError, KeyError, PermissionError) as exc:
                    result, code, outcome = 'Error: ' + str(exc), 1, 'execution_error'
                except Exception as exc:
                    result, code, outcome = 'Error: ' + str(exc), 1, 'execution_error'
            capture.origins.extend(getattr(result, 'origins', []))
            result = capture.finish(result, 'timed_out' if outcome == 'timed_out' else
                                    'succeeded' if code == 0 else 'failed',
                                    exit_code=code, stdout_sink=request.get('stdout_sink'))
        finally:
            CURRENT_EVENT.reset(token)
        audit.update(outcome=outcome, exit_code=code)
        self.outputs.append((result, request.get('stdout_sink'), code))
        return dict(stdout=str(result) if code == 0 else '', stderr=str(result) + '\n' if code else '', code=code)

    def project(self, text):
        from .execution_capture import slice_origins
        origins, offset = [], 0
        partial = False
        for result, sink, code in self.outputs:
            start = str(text).find(str(result), offset)
            if code == 0 and sink == self.capture.facts.get('stdout_sink') and start >= 0:
                origins.extend(slice_origins(result.origins, 0, len(result), start))
            else:
                self.capture.facts.setdefault('capture_gaps', []).append('pf_pipeline_source_ranges_unknown')
            if code == 0 and start >= 0:
                offset = start + len(result)
            if code == 0 and result.pf_read and (start < 0 or
                    len(self.outputs) == 1 and str(text) != result):
                partial = True
        if partial:
            self.capture.facts['pf_delivery'] = dict(complete=False, source_range=None,
                view='shell_source_ranges_unknown',
                original_reads=[result.pf_read['id'] for result, _, code in self.outputs
                                if code == 0 and result.pf_read])
            notice = ('[PF final delivery: partial or transformed shell view; source ranges unknown. '
                      'Original PF headers describe responses generated before shell processing.]\n')
            origins = slice_origins(origins, 0, len(text), len(notice))
            self.capture.origins = slice_origins(self.capture.origins, 0, len(text), len(notice))
            text = notice + text
        read = None
        if len(self.outputs) == 1 and self.outputs[0][2] == 0:
            result = self.outputs[0][0]
            if result.pf_read:
                read = result.pf_read
                if str(text) != result:
                    meta, _ = self.capture.store.get_content(read['id'])
                    version = self.capture.store.version(self.capture.event, 'pf_shell_view', str(text),
                        layer='pf_read_full', target=meta['target'], key=meta['key'],
                        mode=meta['mode'], read_range=None, complete=False, force_full=True,
                        view='shell_source_ranges_unknown', original_read=result.pf_read['id'])
                    read = {'id': version}
        return CapturedText(text, origins, read)

    def close(self):
        self.cancelled.set()
        if connection := self.connection:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.thread.join(timeout=.5)
        closed = not self.thread.is_alive()
        self.capture.facts['pf_calls'] = self.calls
        self.capture.facts['pf_cleanup'] = dict(closed=closed,
            unfinished_event_ids=[call['event_id'] for call in self.calls
                                  if call['outcome'] == 'running' and call['event_id']])
        if not closed:
            from .agents.bash_agent.tools import ProcessBoundaryError
            raise ProcessBoundaryError('PF host request remains active after cancellation; execution stopped')
        self.server.server_close()
        self.directory.cleanup()
