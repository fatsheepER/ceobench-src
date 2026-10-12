"""A per-command supervisor; unfinished descendants retain their sandbox."""
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import tempfile
import uuid

DEFAULT_MEMORY_LIMIT_BYTES = 1024**3


def command_memory_limit(value=None):
    if value is None:
        value = os.environ.get('CEOBENCH_BASH_MEMORY_MAX_BYTES', str(DEFAULT_MEMORY_LIMIT_BYTES))
        try:
            value = int(value)
        except ValueError:
            raise ValueError('Bash memory limit must be a positive integer in bytes') from None
    if type(value) is not int or value < 32 * 1024**2:
        raise ValueError('Bash memory limit must be an integer of at least 32 MiB')
    return value

# This code runs inside the same sandbox as Bash, with no engine imports.
SUPERVISOR = r'''
import ctypes, json, os, resource, signal, socket, subprocess, sys, time
address, key, command = sys.argv[1], sys.argv[2], sys.argv[3]
verified = sys.argv[4] == 'verified'
memory_limit = int(sys.argv[5])
if address.startswith('/'):
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    channel.connect(address)
else:
    channel = socket.create_connection(('127.0.0.1', int(address)))
channel.sendall((key + '\n').encode())
linux = sys.platform == 'linux'
proc = None
spawn_errno = None
code = None
unknown = None
children = []
stopping = False
reaped = []
signaled = []
def report(phase):
    channel.sendall((json.dumps(dict(exit_code=code, children=children, unknown=unknown,
        phase=phase, coverage='no_target_start' if proc is None else 'subreaper_wait' if linux else 'process_group',
        stop_requested=stopping, signaled=signaled, reaped=reaped,
        memory_limit_bytes=memory_limit, memory_limit_kind='per_process_address_space',
        target_started=proc is not None, spawn_errno=spawn_errno)) + '\n').encode())
try:
    if linux:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0):
            raise OSError(ctypes.get_errno(), 'Cannot supervise descendants')
    # Set limits here, never in preexec_fn in the threaded host worker.
    hard_limit = resource.getrlimit(resource.RLIMIT_AS)[1]
    if hard_limit != resource.RLIM_INFINITY:
        memory_limit = min(memory_limit, hard_limit)
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    proc = subprocess.Popen(['bash', '-c', command])
    if verified:
        if not linux:
            raise RuntimeError('Verified descendant cleanup requires Linux')
        channel.setblocking(False)
        data = b''
        reported = False
        while True:
            try:
                chunk = channel.recv(4096)
                if not chunk:
                    raise RuntimeError('Host disconnected before boundary closed')
                data += chunk
                while b'\n' in data:
                    line, data = data.split(b'\n', 1)
                    if line != b'stop':
                        raise RuntimeError('Invalid boundary control request')
                    stopping = True
            except BlockingIOError:
                pass
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    children = []
                    break
                if not pid:
                    children = ['running']
                    break
                reaped.append(pid)
                if pid == proc.pid:
                    code = proc.returncode = os.waitstatus_to_exitcode(status)
            if not children:
                break
            if stopping:
                # Killing a parent adopts its detached descendants on the next pass.
                with open('/proc/self/task/%d/children' % os.getpid()) as stream:
                    children = [int(pid) for pid in stream.read().split()]
                for pid in children:
                    try:
                        os.kill(pid, signal.SIGKILL)
                        if pid not in signaled:
                            signaled.append(pid)
                    except ProcessLookupError:
                        pass
            if code is not None and not reported:
                report('draining')
                reported = True
            time.sleep(.01)
    else:
        code = proc.wait()
    if linux and not verified:
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
    elif not linux and not verified:
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
    if proc is None:
        spawn_errno = getattr(exc, 'errno', None)
        code = 125
        sys.stderr.write(type(exc).__name__ + ': ' + str(exc) + '\n')
    else:
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
    def __init__(self, command, python=sys.executable, *, isolated=False, verified=False, memory_limit_bytes=None, temp_root=None):
        memory_limit_bytes = command_memory_limit(memory_limit_bytes)
        self.socket_dir = None
        self.listener = None
        self.channel = None
        try:
            self.socket_dir = tempfile.TemporaryDirectory(prefix='ceobench-boundary-', dir=temp_root) if isolated else None
            self.listener = socket.socket(socket.AF_UNIX if isolated else socket.AF_INET)
            if isolated:
                from .runtime_temp import bind_unix
                bind_unix(self.listener, self.socket_dir.name + '/socket')
            else:
                self.listener.bind(('127.0.0.1', 0))
            self.listener.listen(1)
        except BaseException:
            self.close()
            raise
        self.key = uuid.uuid4().hex
        self.command = shlex.join([python, '-c', SUPERVISOR,
                                  '/run/novamind-boundary/socket' if isolated else str(self.listener.getsockname()[1]), self.key, command,
                                  'verified' if verified else 'legacy', str(memory_limit_bytes)])
        self.verified = verified
        self.channel = None
        self.record = None

    def close(self):
        if self.channel:
            self.channel.close()
        if self.listener:
            self.listener.close()
        if self.socket_dir:
            self.socket_dir.cleanup()

    def communicate(self, process, timeout, *, started_at=None, check=None):
        try:
            if self.verified:
                return self._communicate_verified(process, timeout, started_at, check)
            return self._communicate(process, timeout, started_at)
        except subprocess.TimeoutExpired:
            raise
        except BoundaryOpen:
            raise
        except Exception as exc:
            raise BoundaryOpen(process, dict(exit_code=process.poll(), children=[], unknown=str(exc)), b'', b'') from exc

    def _communicate_verified(self, process, timeout, started_at, check=None):
        started_at = time.monotonic() if started_at is None else started_at
        deadline = started_at + timeout
        cleanup_started = None
        stdout, stderr, data = b'', b'', b''
        authenticated = False
        stop_sent = False
        cancellation = None
        self.listener.settimeout(.01)
        try:
            while True:
                now = time.monotonic()
                if check and cancellation is None:
                    from .run_lifecycle import RunCancelled
                    try:
                        check()
                    except RunCancelled as exc:
                        cancellation = str(exc)
                        cleanup_started = now
                if cleanup_started is None and now >= deadline:
                    cleanup_started = now
                if cleanup_started is not None and now >= cleanup_started + 5:
                    raise RuntimeError('Could not verify cleanup within 5 seconds')
                if self.channel is None:
                    try:
                        self.channel, _ = self.listener.accept()
                        self.channel.setblocking(False)
                    except TimeoutError:
                        if process.poll() is not None:
                            stdout, stderr = process.communicate(timeout=5)
                            self.record = dict(phase='closed', coverage='preauthentication_no_target',
                                target_started=False, exit_code=process.returncode, children=[], unknown=None,
                                direct_child_reaped=True, streams_collected=True, descendants_closed=None,
                                boundary_closed=True, elapsed_seconds=time.monotonic() - started_at,
                                timed_out=False, cancelled=cancellation, cleanup_seconds=0, timeout_trigger_seconds=None)
                            return stdout, stderr
                if self.channel is not None:
                    try:
                        chunk = self.channel.recv(65536)
                        data += chunk
                        while b'\n' in data:
                            line, data = data.split(b'\n', 1)
                            if not authenticated:
                                if line.decode() != self.key:
                                    raise RuntimeError('Invalid process boundary channel')
                                authenticated = True
                            else:
                                self.record = json.loads(line)
                                if self.record.get('unknown'):
                                    raise RuntimeError(self.record['unknown'])
                        if not chunk and (not self.record or self.record.get('phase') != 'closed'):
                            raise RuntimeError('Supervisor exited without a closed boundary')
                    except BlockingIOError:
                        pass
                    if cleanup_started is not None and authenticated and not stop_sent:
                        if not self.record or self.record.get('phase') != 'closed':
                            self.channel.sendall(b'stop\n')
                        stop_sent = True
                remaining = (cleanup_started + 5 if cleanup_started is not None else deadline) - time.monotonic()
                try:
                    stdout, stderr = process.communicate(timeout=max(.001, min(.01, remaining)))
                except subprocess.TimeoutExpired as exc:
                    stdout, stderr = exc.output or b'', exc.stderr or b''
                    continue
                if self.record and self.record.get('phase') == 'closed':
                    no_target = (self.record.get('coverage') == 'no_target_start' and
                                 self.record.get('target_started') is False)
                    if self.record.get('children') or self.record.get('coverage') != 'subreaper_wait' and not no_target:
                        raise RuntimeError('Supervisor did not prove descendant closure')
                    finished = time.monotonic()
                    if cleanup_started is None and finished >= deadline:
                        cleanup_started = finished
                    self.record.update(boundary_closed=True, descendants_closed=None if no_target else True,
                        direct_child_reaped=True, streams_collected=True,
                        elapsed_seconds=(cleanup_started or finished) - started_at,
                        timed_out=cleanup_started is not None and cancellation is None,
                        cancelled=cancellation,
                        timeout_trigger_seconds=cleanup_started - started_at if cleanup_started is not None and cancellation is None else None,
                        cleanup_seconds=finished - cleanup_started if cleanup_started is not None else 0)
                    return stdout, stderr
        except Exception as exc:
            if isinstance(exc, subprocess.TimeoutExpired):
                stdout, stderr = exc.output or stdout, exc.stderr or stderr
            finished = time.monotonic()
            self.record = dict(self.record or {}, exit_code=process.poll(), unknown=str(exc),
                boundary_closed=False, timed_out=cleanup_started is not None and cancellation is None,
                cancelled=cancellation,
                elapsed_seconds=(cleanup_started or finished) - started_at,
                timeout_trigger_seconds=cleanup_started - started_at if cleanup_started is not None and cancellation is None else None,
                cleanup_seconds=finished - cleanup_started if cleanup_started is not None else 0)
            raise BoundaryOpen(process, self.record, stdout, stderr) from exc

    def _communicate(self, process, timeout, started_at=None):
        deadline = (time.monotonic() if started_at is None else started_at) + timeout
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
