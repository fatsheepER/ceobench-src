"""Best-effort private file observations, independent of business success."""
import hashlib
import json
import logging
import os
from pathlib import Path
import stat
import threading
import uuid

from .workspace_io import open_file
from .workspace_layout import managed_path


def stamp(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class WorkspaceObserver:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.blobs = self.root / 'blobs'
        self.blobs.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.active = {}

    def _scan(self, workspace):
        from .execution_capture import excluded
        entries, stamps, gaps = {}, {}, []
        def skip(relative):
            return excluded(relative) or managed_path(relative) or any(part.startswith('.registered-') for part in Path(relative).parts)
        def error(exc):
            gaps.append(dict(path=str(exc.filename), reason=type(exc).__name__))
        root_stamp = stamp(workspace.lstat())
        for directory, dirs, files in os.walk(workspace, followlinks=False, onerror=error):
            parent = Path(directory)
            dirs[:] = [name for name in dirs if not skip((parent / name).relative_to(workspace).as_posix())]
            for name in sorted(dirs + files):
                path = parent / name
                relative = path.relative_to(workspace).as_posix()
                if skip(relative):
                    continue
                try:
                    before = path.lstat()
                    entry = dict(mode=before.st_mode, size=before.st_size)
                    if stat.S_ISLNK(before.st_mode):
                        entry.update(type='symlink', target=os.readlink(path))
                    elif stat.S_ISREG(before.st_mode):
                        with open_file(workspace, path) as stream:
                            if stamp(os.fstat(stream.fileno())) != stamp(before):
                                raise RuntimeError('changed_before_read')
                            raw = stream.read()
                            if stamp(os.fstat(stream.fileno())) != stamp(before):
                                raise RuntimeError('changed_during_read')
                        digest = hashlib.sha256(raw).hexdigest()
                        target = self.blobs / digest
                        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                            temporary = self.blobs / ('.blob-' + uuid.uuid4().hex)
                            try:
                                with temporary.open('xb') as stream:
                                    stream.write(raw)
                                os.replace(temporary, target)
                            finally:
                                temporary.unlink(missing_ok=True)
                        entry.update(type='file', sha256=digest)
                    else:
                        entry['type'] = 'directory' if stat.S_ISDIR(before.st_mode) else 'special'
                    if stamp(path.lstat()) != stamp(before):
                        raise RuntimeError('changed_after_read')
                    entries[relative], stamps[relative] = entry, stamp(before)
                except (OSError, ValueError, RuntimeError) as exc:
                    gaps.append(dict(path=relative, reason=str(exc)))
        for relative, signature in stamps.items():
            try:
                stable = stamp((workspace / relative).lstat()) == signature
            except OSError:
                stable = False
            if not stable:
                entries.pop(relative, None)
                gaps.append(dict(path=relative, reason='changed_during_tree_scan'))
        if stamp(workspace.lstat()) != root_stamp:
            gaps.append(dict(path='.', reason='directory_changed_during_tree_scan'))
        return dict(entries=entries, gaps=gaps, coverage='file_acquisitions_not_atomic_tree')

    def begin(self, executor, tool, args, request, context, call):
        observation = dict(id=uuid.uuid4().hex, **executor.identity.fields(), tool=tool, arguments=args,
                           model_request_event=request, model_context_id=context, tool_call_id=call,
                           concurrent_executions=[], gaps=[])
        with self.lock:
            for other in self.active.values():
                other['concurrent_executions'].append(observation['id'])
                observation['concurrent_executions'].append(other['id'])
            self.active[observation['id']] = observation
        if not executor.evidence_store:
            try:
                observation['before'] = self._scan(executor.workspace_path)
            except Exception as exc:
                observation['gaps'].append(dict(phase='before', reason=str(exc)))
        executor.last_capture_event = None
        return observation

    def finish(self, executor, observation):
        if observation is None:
            return
        try:
            if executor.evidence_store:
                from contextlib import closing
                event = executor.last_capture_event
                observation['capture_event'] = event
                with closing(executor.evidence_store.connect()) as conn:
                    for row in conn.execute("SELECT v.version_id,v.metadata,b.content FROM versions v JOIN blobs b ON b.content_hash=v.content_hash WHERE v.event_id=? AND json_extract(v.metadata,'$.layer')='workspace_boundary'", (event,)):
                        phase = row['version_id'].rsplit('workspace_', 1)[-1]
                        if phase:
                            observation[phase] = dict(reused_capture=True, manifest=json.loads(row['content']))
                if not all(phase in observation for phase in ('before', 'after')):
                    observation['gaps'].append(dict(reason='capture_boundary_not_available'))
            elif getattr(executor, '_bash_facts', {}).get('boundary_closed', True):
                observation['after'] = self._scan(executor.workspace_path)
            else:
                observation['gaps'].append(dict(phase='after', reason='execution_boundary_open'))
        except Exception as exc:
            observation['gaps'].append(dict(reason=str(exc)))
        finally:
            with self.lock:
                self.active.pop(observation['id'], None)
            observation.update(status=getattr(executor, 'last_status', None),
                               attribution='observed_changes_not_exclusive_authorship',
                               coverage='tool_boundaries_only_internal_writes_unobserved')
            try:
                (self.root / (observation['id'] + '.json')).write_text(json.dumps(observation, ensure_ascii=False))
            except Exception:
                logging.exception('Private workspace observation could not be saved')
