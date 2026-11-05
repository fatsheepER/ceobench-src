"""Resolve public PF path aliases before applying version and access checks."""
import errno
from pathlib import PurePosixPath
import shlex

from . import evidence_handles
from .public_errors import PublicPFError, PublicPFPermissionError
from .public_records import private_framework_path
from .role_policy import ROLES


def normalize(registry, value):
    """Return ROLE:relative-name (or the standalone name), retaining any @vN.

    This maps names only. It never chooses a version from read timestamps, reads
    file bytes, or requires an old captured file to still exist in the workspace.
    """
    if '\x00' in value:
        raise PublicPFError('Reference must not contain a NUL character')
    if evidence_handles.RAW_VERSION.fullmatch(value) or evidence_handles.EXACT.fullmatch(value):
        return value  # Opaque capture/event identity; the store rechecks visibility.
    owner, path = registry.role, value
    qualified = ':' in value and value.split(':', 1)[0] in ROLES
    if qualified:
        owner, path = value.split(':', 1)
    suffix = ''
    if match := evidence_handles.VERSIONED.fullmatch(path):
        path, suffix = match.group(1), '@v' + match.group(2)
    workspace = registry.workspace
    if not qualified and PurePosixPath(path).is_absolute():
        roots = {role: workspace.parent / role for role in ROLES} if registry.identity else {owner: workspace}
        found = next(((role, str(PurePosixPath(path).relative_to(root))) for role, root in roots.items()
                      if PurePosixPath(path).is_relative_to(root)), None)
        if found is None:
            raise PublicPFPermissionError('Path is outside the authorized role workspaces; use ROLE:path')
        owner, path = found
    elif not qualified and path.startswith('../') and registry.identity:
        parts = path.split('/')
        if len(parts) < 3 or parts[1] not in ROLES:
            raise PublicPFPermissionError('Path escapes the role workspace; use ROLE:path')
        owner, path = parts[1], '/'.join(parts[2:])
    try:
        root = registry.repository(owner)
    except PermissionError as exc:
        raise PublicPFPermissionError('Evidence is inaccessible') from exc
    if not registry.identity and qualified and owner != registry.role:
        raise PublicPFPermissionError('This run has no such role workspace')
    if PurePosixPath(path).is_absolute() or root.resolve() != root:
        raise PublicPFPermissionError('Path is outside the authorized role workspace')
    parts = []
    for part in path.split('/'):
        if part in ('', '.'):
            continue
        if part == '..':
            if not parts:
                raise PublicPFPermissionError('Path escapes the role workspace; use ROLE:path')
            parts.pop()
            continue
        parts.append(part)
        relative = '/'.join(parts)
        if '.git' in parts or parts[0] == 'sessions' or private_framework_path(relative):
            raise PublicPFPermissionError('Only public workspace files may be referenced')
        # An exact @vN names stored bytes, not the current filesystem object.
        # Do not traverse even an ancestor link when resolving saved history.
        # The resolver still requires a real capture and rechecks its visibility.
        if suffix:
            continue
        try:
            is_link = (root / relative).is_symlink()
        except OSError as exc:
            if exc.errno != errno.ENAMETOOLONG:
                raise
            raise PublicPFError('Reference path component is too long; use a captured handle or a shorter public path') from exc
        if is_link:
            raise PublicPFPermissionError('Symbolic links are not public PF file references; use the original ROLE:path')
    if not parts:
        raise PublicPFError('Reference must name a file or captured output inside a role workspace')
    return (owner + ':' if registry.identity else '') + '/'.join(parts) + suffix


def missing(registry, value=None, *, prefix='No captured version matches'):
    message = prefix + '; use a captured file/output handle or public SQL'
    if registry.capabilities.text_tools:
        message += ', or a registered text revision'
    if value:
        message += peer_hint(value) or '. For a peer file use ROLE:path (for example growth:reports/forecast.txt)'
    return message


def peer_hint(value):
    path = value.split(':', 1)[1] if ':' in value and value.split(':', 1)[0] in ROLES else value
    if match := evidence_handles.VERSIONED.fullmatch(path):
        path = match.group(1)
    first, separator, rest = path.partition('/')
    if separator and first in ROLES:
        return '. This is a local directory path; for the peer use ' + shlex.join(['pf', 'log', first + ':' + rest])
    return ''
