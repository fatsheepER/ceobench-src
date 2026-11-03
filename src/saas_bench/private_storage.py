"""Private authority paths and atomic, disposable workspace projections."""
import json
import os
from pathlib import Path
import uuid

from .workspace_io import parent_fd


def private_root(workspace, *, identity=None, root=None, readable=()):
    workspace = Path(workspace).resolve()
    target = Path(root).resolve() if root is not None else (
        workspace.parent / '.ceobench-private' / workspace.name)
    mounts = [workspace, *(Path(path).resolve() for path in readable)]
    mounts += [Path(path) for path in ('/usr', '/bin', '/lib', '/lib64', '/etc', '/sbin')]
    if any(target.is_relative_to(mount) for mount in mounts):
        raise ValueError('Private storage must be outside all readable mounts')
    target.mkdir(parents=True, mode=0o700, exist_ok=True)
    return target


def publish_json(root, path, value, *, prefix):
    with parent_fd(root, path, create=True) as (directory, name):
        temporary = prefix + uuid.uuid4().hex
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        try:
            with os.fdopen(descriptor, 'w') as stream:
                json.dump(value, stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
            os.fsync(directory)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
