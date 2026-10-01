"""A per-Bash PF service. The sandbox receives only a client and this socket."""
import json
import os
import re
from pathlib import Path, PurePosixPath
import socketserver
import tempfile
import threading

from . import pf_cli
from .execution_capture import CapturedText, CURRENT_EVENT, ExecutionCapture


CLIENT = '''import json, os, socket, sys
channel = socket.socket(socket.AF_UNIX)
channel.connect(os.environ['PF_SOCKET'])
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
        self.directory = tempfile.TemporaryDirectory(prefix='ceobench-pf-')
        self.root = Path(self.directory.name)
        self.guest = '/opt/pf' if executor._bwrap() else str(self.root)
        client = self.root / 'pf'
        client.write_text('#!' + executor.python + '\n' + CLIENT)
        client.chmod(0o555)
        service = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.connection.settimeout(30)
                try:
                    data = self.rfile.readline(65537)
                    if len(data) > 65536:
                        raise ValueError('PF request too large')
                    response = service.call(json.loads(data))
                except Exception as exc:
                    response = dict(stdout='', stderr='pf: ' + str(exc) + '\n', code=2)
                try:
                    self.wfile.write(json.dumps(response).encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass  # The child result remains captured even if its pipe consumer exits.

        self.server = socketserver.UnixStreamServer(str(self.root / 'socket'), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()

    def call(self, request):
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
                if field and relative.parts:
                    value = target[field]
                    # Numbered outputs/texts are global; file and script names follow cwd.
                    if not re.fullmatch(r'(?:r\d+(?:\.\d+)?|(?:cmd|query|read|receipt)\d+(?:@v\d+)?)', value):
                        target[field] = os.path.normpath(str(relative / value))
        except ValueError as exc:
            error = str(exc)
        capture = ExecutionCapture(self.executor.evidence_store)
        parent_args = self.capture.store.read_event(self.capture.event)['request']['request']
        capture.begin(operation, args, parent=self.capture.event, argv=argv, cwd=str(relative),
                      command=parent_args.get('command'), note=parent_args.get('note'))
        token = CURRENT_EVENT.set(capture.event)
        code, outcome = 0, 'succeeded'
        try:
            if error:
                result, code, outcome = error, 2, 'usage_error'
            elif operation == 'pf_help':
                result = pf_cli.USAGE
            else:
                try:
                    result = self.executor.pf_queries.execute(operation, args)
                except (ValueError, KeyError) as exc:
                    result, code, outcome = 'Error: ' + str(exc), 1, 'execution_error'
            capture.origins.extend(getattr(result, 'origins', []))
            result = capture.finish(result, 'succeeded' if code == 0 else 'failed',
                                    exit_code=code, stdout_sink=request.get('stdout_sink'))
        finally:
            CURRENT_EVENT.reset(token)
        audit = dict(operation=operation, arguments=args, event_id=capture.event, outcome=outcome,
                     argv=argv, cwd=str(relative), exit_code=code, note=parent_args.get('note'))
        self.calls.append(audit)
        self.outputs.append((result, request.get('stdout_sink'), code))
        return dict(stdout=str(result) if code == 0 else '', stderr=str(result) + '\n' if code else '', code=code)

    def project(self, text):
        """Only direct stdout proves ranges. Pipelines retain their literal Bash return."""
        from .execution_capture import slice_origins
        origins, offset = [], 0
        for result, sink, code in self.outputs:
            start = str(text).find(str(result), offset)
            if code == 0 and sink == self.capture.facts.get('stdout_sink') and start >= 0:
                origins.extend(slice_origins(result.origins, 0, len(result), start))
                offset = start + len(result)
            else:
                self.capture.facts.setdefault('capture_gaps', []).append('pf_pipeline_source_ranges_unknown')
        read = None
        if len(self.outputs) == 1 and self.outputs[0][2] == 0:
            result = self.outputs[0][0]
            if result.pf_read:
                read = result.pf_read
                if str(text) != result:
                    meta, _ = self.capture.store.get_content(read['id'])
                    version = self.capture.store.version(self.capture.event, 'pf_shell_view', str(text),
                        layer='pf_read_full', target=meta['target'], key=meta['key'],
                        mode=meta['mode'], read_range=[0, 0], complete=False, force_full=True,
                        view='shell_source_ranges_unknown')
                    read = {'id': version}
        return CapturedText(text, origins, read)

    def close(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.directory.cleanup()
