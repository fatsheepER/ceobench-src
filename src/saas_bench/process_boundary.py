"""A per-command supervisor; unfinished descendants retain their sandbox."""
import json
import shlex
import socket
import subprocess
import sys
import time
import uuid

# This code runs inside the same sandbox as Bash, with no engine imports.
SUPERVISOR = r'''
import ctypes, json, os, signal, socket, subprocess, sys
port, key, command = int(sys.argv[1]), sys.argv[2], sys.argv[3]
channel = socket.create_connection(('127.0.0.1', port))
channel.sendall((key + '\n').encode())
linux = sys.platform == 'linux'
if linux:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), 'Cannot supervise descendants')
proc = subprocess.Popen(['bash', '-c', command])
code = proc.wait()
unknown = None
children = []
def report(phase):
    channel.sendall((json.dumps(dict(exit_code=code, children=children, unknown=unknown,
        phase=phase, coverage='subreaper_wait' if linux else 'process_group')) + '\n').encode())
try:
    if linux:
        # Adopt and reap orphaned descendants, including detached process groups.
        # The host drains both output pipes and enforces the original deadline.
        while True:
            try:
                info = os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            except ChildProcessError:
                break
            if info is None:
                children = ['running']
                report('draining')
                os.waitpid(-1, 0)
                children = []
                continue
            os.waitpid(info.si_pid, 0)
    else:
        import time
        while True:
            probe = subprocess.Popen(['ps', '-axo', 'pid=,ppid=,pgid=,stat='], stdout=subprocess.PIPE, text=True)
            rows = probe.communicate()[0]
            children = [int(row.split()[0]) for row in rows.splitlines()
                        if len(row.split()) == 4 and int(row.split()[2]) == os.getpgrp()
                        and int(row.split()[0]) not in (os.getpid(), probe.pid) and not row.split()[3].startswith('Z')]
            if not children:
                break
            report('draining')
            time.sleep(.01)
except Exception as exc:
    unknown = type(exc).__name__ + ': ' + str(exc)
report('unknown' if unknown else 'closed')
channel.close()
if children or unknown:
    while True:
        signal.pause()
if code < 0:
    os.kill(os.getpid(), -code)
sys.exit(code)
'''


class BoundaryOpen(RuntimeError):
    def __init__(self, process, record, stdout, stderr):
        super().__init__('Bash descendants did not finish before the command deadline; branch paused'
                         if record.get('timed_out') else 'Cannot verify Bash process completion; branch paused')
        self.process, self.record = process, record
        self.stdout, self.stderr = stdout, stderr


class Boundary:
    def __init__(self, command, python=sys.executable):
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen(1)
        self.key = uuid.uuid4().hex
        self.command = shlex.join([python, '-c', SUPERVISOR,
                                  str(self.listener.getsockname()[1]), self.key, command])
        self.channel = None
        self.record = None

    def close(self):
        if self.channel:
            self.channel.close()
        self.listener.close()

    def communicate(self, process, timeout):
        try:
            return self._communicate(process, timeout)
        except subprocess.TimeoutExpired:
            raise
        except BoundaryOpen:
            raise
        except Exception as exc:
            raise BoundaryOpen(process, dict(exit_code=process.poll(), children=[], unknown=str(exc)), b'', b'') from exc

    def _communicate(self, process, timeout):
        deadline = time.monotonic() + timeout
        self.listener.settimeout(min(0.05, max(0.001, timeout)))
        while self.channel is None:
            try:
                self.channel, _ = self.listener.accept()
            except TimeoutError:
                if process.poll() is not None:
                    self.record = dict(exit_code=process.returncode, children=[], unknown=None,
                                       coverage='supervisor_start_failed')
                    return process.communicate()
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(process.args, timeout)
        self.channel.setblocking(False)
        data = b''
        stdout, stderr = b'', b''
        authenticated = False
        def receive(chunk):
            nonlocal data, authenticated
            data += chunk
            while b'\n' in data:
                line, data = data.split(b'\n', 1)
                if not authenticated:
                    if line.decode() != self.key:
                        raise RuntimeError('Invalid process boundary channel')
                    authenticated = True
                else:
                    self.record = json.loads(line)
                    if self.record['unknown']:
                        raise BoundaryOpen(process, self.record, stdout, stderr)
        while True:
            try:
                chunk = self.channel.recv(65536)
                receive(chunk)
            except BlockingIOError:
                chunk = None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if self.record and self.record['children']:
                    raise BoundaryOpen(process, dict(self.record, timed_out=True), stdout, stderr)
                raise subprocess.TimeoutExpired(process.args, timeout, stdout, stderr)
            try:
                stdout, stderr = process.communicate(timeout=min(0.05, remaining))
                if self.record is None or self.record.get('phase') != 'closed':
                    # The supervisor sends the boundary before exiting. Drain its socket.
                    self.channel.settimeout(max(.001, deadline - time.monotonic()))
                    while self.record is None or self.record.get('phase') != 'closed':
                        chunk = self.channel.recv(65536)
                        if not chunk:
                            raise RuntimeError('Supervisor exited without a process boundary')
                        receive(chunk)
                return stdout, stderr
            except subprocess.TimeoutExpired as exc:
                stdout, stderr = exc.output or b'', exc.stderr or b''
