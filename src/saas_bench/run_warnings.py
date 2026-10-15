from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

from .run_state import file_hash, write_json


@contextmanager
def advisory():
    try:
        yield
    except Exception as exc:
        logging.warning('%s: %s; continuing as requested', type(exc).__name__, exc)


def optional_hash(path):
    with advisory():
        return file_hash(path)
    return None


class AttemptWarnings(logging.Handler):
    def __init__(self, initial=()):
        super().__init__(logging.WARNING)
        self.rows = list(initial)
        self.record = self.directory = None

    def __enter__(self):
        if not logging.getLogger().handlers:
            logging.basicConfig()
        logging.getLogger().addHandler(self)
        return self

    def bind(self, record, directory):
        self.record, self.directory = record, Path(directory)
        self.persist()

    def emit(self, event):
        self.rows.append(dict(time=datetime.fromtimestamp(event.created, timezone.utc).isoformat(),
                              message=event.getMessage()))
        self.persist()

    def persist(self):
        if self.record is None:
            return
        self.record['validation_warnings'] = list(self.rows)
        try:
            write_json(self.directory / 'attempt.json', self.record)
            (self.directory / 'warnings.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in self.rows))
        except Exception:
            self.handleError(logging.LogRecord(__name__, logging.WARNING, __file__, 0,
                'Could not persist attempt warnings', (), None))

    def __exit__(self, *exc):
        logging.getLogger().removeHandler(self)
