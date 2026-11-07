"""Explicit business-file layout; actor authority remains role-bound."""
from dataclasses import dataclass
from pathlib import PurePosixPath

LEGACY = 'role-v1'
SHARED = 'shared-v1'
MANAGED = ('docs', 'novamind-operation', 'documents')


@dataclass(frozen=True)
class FileKey:
    namespace: str
    relative_path: str

    def __post_init__(self):
        path = PurePosixPath(self.relative_path)
        if not self.namespace or path.is_absolute() or '..' in path.parts or str(path) != self.relative_path:
            raise ValueError('Expected a normalized workspace file identity')


def evidence_namespace(metadata, fallback=None):
    return (metadata.get('workspace_namespace') if metadata.get('layer') in ('file_bytes', 'file_text') else None) or metadata.get('owner_role', fallback)


def workspace_names(layout):
    from .role_policy import ROLES
    if layout == SHARED:
        return ('shared',)
    if layout == LEGACY:
        return ROLES
    raise ValueError('Unknown workspace layout: ' + str(layout))


def managed_path(relative):
    parts = PurePosixPath(relative).parts
    return bool(parts) and parts[0] in MANAGED
