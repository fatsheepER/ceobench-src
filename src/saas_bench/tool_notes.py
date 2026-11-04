"""Public Git artifacts for one noted call and its exact file snapshots."""
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import uuid

from .registration_evidence import snapshot_reference, retain_snapshot
from .workspace_io import open_file, parent_fd


def record_git_note(executor, note, output):
    if not executor.capabilities.tool_notes or not executor.capabilities.git_extra_saves:
        raise ValueError('Git note snapshots are disabled for this run')
    registry = executor.text_registry
    number = executor.public_refs.ordinal('noted_call', executor.role, uuid.uuid4().hex)
    relative = Path('tool-outputs') / f'call{number}.txt'
    day = registry.sim_day()
    lines = [f'Call: {executor.role}:call{number}', f'Tool: {note["tool"]}',
             f'Role: {executor.role}', f'Day: {day if day is not None else "unknown"}',
             f'Status: {note["status"]}', f'Note: {note["note"]}', 'Changed files:']
    snapshots, stamps = [], {}
    message = '\n'.join(lines[:-1]) + '\n'
    for path in note['files']:
        if path.startswith('tool-outputs/'):
            continue
        source = executor.workspace_path / path
        if not source.exists():
            lines.append(f'- {executor.role}:{path} (deleted; no resulting version)')
            continue
        try:
            evidence, commit = snapshot_reference(executor.workspace_path, path, executor.workspace_path,
                executor.role, run=executor.run_private)
            argv = ['git', '--no-replace-objects', '-c', 'core.hooksPath=/dev/null',
                    '-c', 'user.name=CEO Bench', '-c', 'user.email=ceobench@localhost',
                    '-c', 'commit.gpgsign=false', '-C', str(executor.workspace_path)]
            tree = executor.run_private(argv + ['rev-parse', commit + '^{tree}'],
                capture_output=True, check=True, timeout=10).stdout.decode().strip()
            commit = executor.run_private(argv + ['commit-tree', tree], input=message.encode(),
                capture_output=True, check=True, timeout=10).stdout.decode().strip()
            evidence['commit'] = commit
            retain_snapshot(executor.workspace_path, commit, run=executor.run_private)
            short = registry.public_commit(executor.role, commit)
            if short is None:
                raise ValueError('Retained snapshot is unavailable')
        except (ValueError, OSError, subprocess.SubprocessError):
            logging.exception('Could not snapshot a noted file')
            lines.append(f'- {executor.role}:{path} (snapshot unavailable; inspect current file)')
        else:
            snapshots.append(evidence)
            from .execution_capture import decoded
            with open_file(executor.workspace_path, source) as stream:
                raw = stream.read()
                state = os.fstat(stream.fileno())
            try:
                text_hash = hashlib.sha256(decoded(raw).encode()).hexdigest()
            except UnicodeError:
                pass
            else:
                stamps[path] = dict(size=state.st_size, mtime_ns=state.st_mtime_ns, text_sha256=text_hash)
            lines.append(f'- {executor.role}:{path}@{short}')
    lines.extend(['', 'Output:', output])
    data = '\n'.join(lines).encode()
    with parent_fd(executor.workspace_path, executor.workspace_path / relative, create=True) as (directory, name):
        descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(directory)
    try:
        evidence, commit = snapshot_reference(executor.workspace_path, relative.as_posix(), executor.workspace_path,
            executor.role, run=executor.run_private)
        retain_snapshot(executor.workspace_path, commit, run=executor.run_private)
        short = registry.public_commit(executor.role, commit)
        if short is None:
            raise ValueError('Retained call artifact snapshot is unavailable')
    except (ValueError, OSError, subprocess.SubprocessError):
        logging.exception('Could not snapshot a noted call artifact')
        evidence = dict(path=relative.as_posix(), snapshot_status='unavailable')
        reference = relative.as_posix() + ' (snapshot unavailable; current artifact only)'
    else:
        reference = relative.as_posix() + '@' + short
    note.update(output_artifact=evidence, file_snapshots=snapshots, file_stamps=stamps, sim_day=day)
    return f'[Call output and note: {executor.role}:{reference}]'


def current_file_note(executor, path, text):
    if not executor.capabilities.tool_notes or not executor.capabilities.git_extra_saves:
        return ''
    registry = executor.text_registry
    if registry is None or registry.mode not in ('git', 'prefix'):
        return ''
    owner, name = executor._file_owner(path), executor._file_name(path)
    registry.repository(owner)
    private = executor.private_root if owner == executor.role else executor.private_root.parent / owner
    log = private / 'tool-notes.jsonl'
    if not log.exists():
        return ''
    state = path.stat()
    stamp = dict(size=state.st_size, mtime_ns=state.st_mtime_ns,
                 text_sha256=hashlib.sha256(text.encode()).hexdigest())
    with open_file(private, log) as stream:
        records = stream.read().splitlines()
    for raw in reversed(records):
        record = json.loads(raw)
        if name not in record.get('files', []):
            continue
        if record.get('file_stamps', {}).get(name) != stamp:
            return ''
        day = record.get('sim_day')
        return (f'[File note: role={owner}; day={day if day is not None else "unknown"}; '
                f'status={record["status"]}; {record["note"]}]')
    return ''
