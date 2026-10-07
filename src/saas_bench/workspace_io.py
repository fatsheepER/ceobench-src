"""Open resolved workspace files without following replacement directory symlinks."""
from contextlib import contextmanager
import os
from pathlib import Path
import stat


@contextmanager
def parent_fd(root, path, *, create=False):
    parts = Path(path).relative_to(root).parts
    if not parts or any(p in ('.', '..') for p in parts):
        raise ValueError('Expected a file inside workspace')
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            if create:
                try:
                    os.mkdir(part, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd, parts[-1]
    finally:
        os.close(fd)


@contextmanager
def open_file(root, path, mode='rb'):
    writing = mode == 'wb'
    with parent_fd(root, path, create=writing) as (directory, name):
        flags = os.O_WRONLY | os.O_CREAT if writing else os.O_RDONLY
        fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=directory)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError('Expected a regular workspace file')
            if writing:
                os.ftruncate(fd, 0)
            with os.fdopen(fd, mode) as stream:
                fd = None
                yield stream
        finally:
            if fd is not None:
                os.close(fd)
