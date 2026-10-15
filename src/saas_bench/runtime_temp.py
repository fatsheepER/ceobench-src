"""Owned temporary storage and Unix sockets under long run directories."""
from contextlib import contextmanager
import os
import logging
from pathlib import Path
import tempfile


@contextmanager
def unix_address(path):
    path = os.fspath(path)
    if len(os.fsencode(path)) < 108:
        yield path
        return
    parent, name = os.path.split(path)
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield f'/proc/self/fd/{descriptor}/{name}'
    finally:
        os.close(descriptor)


def bind_unix(channel, path):
    with unix_address(path) as address:
        channel.bind(address)


def connect_unix(channel, path):
    with unix_address(path) as address:
        channel.connect(address)


def check_tmpdir(path, *, min_bytes=64 * 1024**2, min_inodes=4096):
    path = Path(path).resolve(strict=True)
    status = os.statvfs(path)
    available_bytes = status.f_bavail * status.f_frsize
    available_inodes = status.f_favail if status.f_files else None
    if available_bytes < min_bytes or available_inodes is not None and available_inodes < min_inodes:
        logging.warning('Insufficient temporary storage at %s: %s bytes, %s inodes', path, available_bytes, available_inodes)
    # Btrfs does not report an inode limit. Prove the directory accepts a file.
    with tempfile.TemporaryFile(dir=path) as probe:
        probe.write(b'tmpdir-precheck')
        probe.flush()
    return dict(path=str(path), available_bytes=available_bytes, available_inodes=available_inodes)


def prepare_tmpdir(path):
    path = Path(path)
    if path.is_symlink():
        logging.warning('Temporary directory is a symlink: %s', path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    check_tmpdir(path)
    return path.resolve()
