"""One host PF budget shared by tracing, refresh and transport."""
from contextlib import contextmanager
from contextvars import ContextVar
import threading
import time

from .run_lifecycle import RunCancelled


CURRENT_BUDGET = ContextVar('pf_budget', default=None)
CALL_TIMEOUT_SECONDS = 120


class Budget:
    def __init__(self, deadline, *, cancelled=None, check=None):
        self.deadline = deadline
        self.cancelled = cancelled if cancelled is not None else threading.Event()
        self.external_check = check

    def check(self):
        if self.external_check:
            try:
                self.external_check()
            except RunCancelled as exc:
                self.cancelled.set()
                raise TimeoutError('PF call cancelled') from exc
        if self.cancelled.is_set():
            raise TimeoutError('PF call cancelled')
        if time.monotonic() >= self.deadline:
            raise TimeoutError('PF call time limit exceeded')

    def remaining(self):
        self.check()
        return max(0, self.deadline - time.monotonic())

    def expired(self):
        try:
            self.check()
        except Exception:
            return True
        return False


def current_budget():
    return CURRENT_BUDGET.get()


def check():
    if budget := current_budget():
        budget.check()


@contextmanager
def budget_scope(*, timeout=CALL_TIMEOUT_SECONDS, budget=None):
    parent = budget or current_budget()
    deadline = time.monotonic() + timeout
    budget = Budget(min(deadline, parent.deadline) if parent else deadline,
                    cancelled=parent.cancelled if parent else None,
                    check=parent.external_check if parent else None)
    token = CURRENT_BUDGET.set(budget)
    try:
        budget.check()
        yield budget
        budget.check()
    finally:
        CURRENT_BUDGET.reset(token)
