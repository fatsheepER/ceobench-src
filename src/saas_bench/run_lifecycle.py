from contextlib import contextmanager
import os
from pathlib import Path
import select
import signal
import threading
import time


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

    def __enter__(self):
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
        self.reason = self.reason or 'supervisor_cancelled'
        self._wake.set()
        if getattr(self._model, 'active', False):
            raise RunCancelled(self.reason)

    def cancel(self, reason='requested_pause'):
        self.reason = self.reason or reason
        self._wake.set()

    def check(self, *, retry=False):
        if self.parent_pid is not None and hasattr(self, '_parent'):
            if select.select([self._parent], [], [], 0)[0]:
                self.reason = 'supervisor_exited'
        if retry and self.hold and self.hold.exists():
            self.reason = self.reason or 'hold_during_retry'
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

    def sleep(self, seconds):
        deadline = time.monotonic() + seconds
        while True:
            self.check(retry=True)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._wake.wait(min(remaining, .25))

    def __exit__(self, *exc):
        self._model.active = False
        if self._thread:
            os.write(self._pipe[1], b'x')
            self._thread.join()
            for fd in (*self._pipe, self._parent):
                os.close(fd)
            del self._parent
        if self._old_handler is not None:
            signal.signal(signal.SIGUSR1, self._old_handler)
