from contextlib import contextmanager
import os
from pathlib import Path
import select
import signal
import socket
import threading
import time
import weakref


class RunCancelled(BaseException):
    pass


class WorkerLifecycle:
    def __init__(self, hold=None, parent_pid=None):
        self.hold = Path(hold) if hold else None
        self.parent_pid = parent_pid
        self.reason = None
        self._model = threading.local()
        self._wake = threading.Event()
        self._thread = None
        self._depth = 0
        self._sockets = weakref.WeakSet()
        self._socket_lock = threading.RLock()

    def __enter__(self):
        if self._depth:
            self._depth += 1
            return self
        self._depth = 1
        self._old_handler = signal.signal(signal.SIGUSR1, self._signal) if threading.current_thread() is threading.main_thread() else None
        if self.parent_pid is not None:
            try:
                self._parent = os.pidfd_open(self.parent_pid)
                if os.getppid() != self.parent_pid:
                    self.reason = 'supervisor_exited'
                self._pipe = os.pipe()
                self._thread = threading.Thread(target=self._watch, daemon=True)
                self._thread.start()
            except BaseException:
                if self._old_handler is not None:
                    signal.signal(signal.SIGUSR1, self._old_handler)
                if hasattr(self, '_parent'):
                    os.close(self._parent)
                raise
        ready = os.environ.get('CEOBENCH_LIFECYCLE_READY')
        self._ready = Path(ready) if ready else None
        if ready:
            from .run_state import write_json
            write_json(Path(ready), dict(pid=os.getpid()))
        return self

    def _watch(self):
        ready, _, _ = select.select([self._parent, self._pipe[0]], [], [])
        if self._parent in ready:
            self.reason = 'supervisor_exited'
            self._wake.set()
            os.kill(os.getpid(), signal.SIGUSR1)

    def _signal(self, signum, frame):
        self.cancel('supervisor_cancelled')
        if getattr(self._model, 'active', False):
            raise RunCancelled(self.reason)

    def cancel(self, reason='requested_pause'):
        self.reason = self.reason or reason
        self._wake.set()
        with self._socket_lock:
            for channel in list(self._sockets):
                try:
                    channel.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def watch_client(self, client):
        http_client = getattr(client, '_client', None)
        if http_client is None or not hasattr(http_client, 'event_hooks'):
            return
        if getattr(http_client, '_ceobench_lifecycle', None) is self:
            return
        http_client._ceobench_lifecycle = self

        def request_hook(request):
            trace = request.extensions.get('trace')
            channel = tls_socket = None
            handshakes = getattr(self._model, 'handshakes', None)
            if handshakes is None:
                handshakes = self._model.handshakes = set()

            def traced(event, info):
                nonlocal channel, tls_socket
                if event in ('connection.start_tls.complete', 'connection.start_tls.failed',
                             'proxy.start_tls.complete', 'proxy.start_tls.failed'):
                    with self._socket_lock:
                        if tls_socket is not None:
                            tls_socket.close()
                            handshakes.discard(tls_socket)
                            self._sockets.discard(tls_socket)
                            tls_socket = None
                if trace:
                    trace(event, info)
                if event in ('connection.connect_tcp.complete', 'connection.start_tls.complete',
                             'proxy.start_tls.complete'):
                    channel = info['return_value'].get_extra_info('socket')
                    if channel is not None:
                        with self._socket_lock:
                            self._sockets.add(channel)
                            if self.reason:
                                try:
                                    channel.shutdown(socket.SHUT_RDWR)
                                except OSError:
                                    pass
                elif event in ('connection.start_tls.started', 'proxy.start_tls.started') and channel is not None:
                    with self._socket_lock:
                        # TLS takes the TCP fd before its blocking handshake completes.
                        tls_socket = socket.fromfd(channel.fileno(), channel.family, channel.type, channel.proto)
                        handshakes.add(tls_socket)
                        self._sockets.add(tls_socket)
                self.check()

            request.extensions['trace'] = traced
            self.check()

        http_client.event_hooks['request'].append(request_hook)

    def check(self, *, retry=False):
        if self.parent_pid is not None and hasattr(self, '_parent'):
            if select.select([self._parent], [], [], 0)[0]:
                self.reason = 'supervisor_exited'
        if retry and self.hold and self.hold.name == 'STOP' and self.hold.exists():
            self.cancel('requested_pause')
        if self.reason:
            raise RunCancelled(self.reason)

    @contextmanager
    def model_call(self):
        try:
            self._model.active = True
            self.check()
            yield
        finally:
            self._model.active = False
            with self._socket_lock:
                for channel in getattr(self._model, 'handshakes', ()):
                    channel.close()
                    self._sockets.discard(channel)
                self._model.handshakes = set()
            self.check()

    def sleep(self, seconds):
        deadline = time.monotonic() + seconds
        while True:
            self.check(retry=True)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._wake.wait(min(remaining, .25))

    def __exit__(self, *exc):
        self._depth -= 1
        if self._depth:
            return
        self._model.active = False
        if self._thread:
            os.write(self._pipe[1], b'x')
            self._thread.join()
            for fd in (*self._pipe, self._parent):
                os.close(fd)
            del self._parent
        if self._old_handler is not None:
            # A supervisor may already have queued a signal after reading ready.
            handler = signal.SIG_IGN if self._ready and self._old_handler == signal.SIG_DFL else self._old_handler
            signal.signal(signal.SIGUSR1, handler)
        if self._ready:
            self._ready.unlink(missing_ok=True)
