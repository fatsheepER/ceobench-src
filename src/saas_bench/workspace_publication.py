"""Immutable PF execution trees and the single authority for publication outcomes."""
from contextlib import contextmanager, closing
from dataclasses import dataclass, field
import ctypes
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import subprocess
import tempfile
import time
import uuid

from .workspace_layout import FileKey, managed_path
from .public_errors import PublicPFError

class PublicationHold(RuntimeError):
    pass


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class Execution:
    id: str
    actor: str
    root: Path
    public: str
    before: str
    bases: dict
    drafts: tuple
    private_inputs: bool = False
    output_before: dict = field(default_factory=dict)
    automatic: bool = False
    git_before: str | None = None


class WorkspacePublication:
    def __init__(self, public, private, namespace='shared', *, timeout=600, lifecycle=None):
        self.public = Path(public)
        self.private = Path(private)
        self.namespace, self.timeout, self.lifecycle = namespace, timeout, lifecycle
        self.private.mkdir(parents=True, exist_ok=True)
        for name in ('blobs', 'manifests', 'candidates', 'executions', 'projections', 'git-repositories'):
            (self.private / name).mkdir(exist_ok=True)
        self.database = self.private / 'authority.sqlite'
        with closing(self.connect()) as conn, conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS head (id INTEGER PRIMARY KEY CHECK(id=1), manifest TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS authors (actor TEXT PRIMARY KEY, bases BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS executions (id TEXT PRIMARY KEY, record BLOB NOT NULL, candidate TEXT, outcome BLOB);
                CREATE TABLE IF NOT EXISTS drafts (id TEXT PRIMARY KEY, actor TEXT NOT NULL, candidate TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS finish_confirmations (actor TEXT, day INTEGER, signature TEXT, confirmed INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(actor,day,signature));
                CREATE TABLE IF NOT EXISTS draft_reminders (actor TEXT, day INTEGER, PRIMARY KEY(actor,day));
                CREATE TABLE IF NOT EXISTS private_git (actor TEXT PRIMARY KEY, snapshot TEXT NOT NULL, private INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS discards (id TEXT PRIMARY KEY, actor TEXT, candidate TEXT);
                CREATE TABLE IF NOT EXISTS resolutions (id TEXT PRIMARY KEY, record BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS evidence (id TEXT PRIMARY KEY, event TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS projected (id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS initial (manifest TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS host_outputs (actor TEXT, path TEXT, entry BLOB, private INTEGER, PRIMARY KEY(actor,path));
            ''')
        with self.lock():
            if self.head() is None:
                initial = self.scan(self.public)
                manifest = self.manifest(initial)
                with closing(self.connect()) as conn, conn:
                    conn.execute('INSERT INTO head VALUES (1,?)', (manifest,))
                    conn.execute('INSERT INTO initial VALUES (?)', (manifest,))

    def connect(self):
        conn = sqlite3.connect(self.database, timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA synchronous=FULL')
        return conn

    @contextmanager
    def lock(self):
        with (self.private / 'publish.lock').open('a+b') as stream:
            deadline = time.monotonic() + self.timeout
            while True:
                if self.lifecycle:
                    self.lifecycle.check()
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise PublicationHold('Publication lock deadline exceeded')
                    time.sleep(.01)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def stage(self, name, execution=None):
        """Persistence fault boundary used by real-process recovery tests."""

    def put(self, collection, data):
        key = checksum(data)
        target = self.private / collection / key
        if target.exists():
            if target.read_bytes() != data:
                raise PublicationHold('Immutable workspace object is corrupt')
            return key
        fd, name = tempfile.mkstemp(dir=target.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, target)
            sync_directory(target.parent)
        finally:
            Path(name).unlink(missing_ok=True)
        return key

    def get(self, collection, key):
        data = (self.private / collection / key).read_bytes()
        if checksum(data) != key:
            raise PublicationHold('Immutable workspace object is corrupt')
        return data

    def manifest(self, tree):
        return self.put('manifests', encoded(tree))

    def tree(self, manifest):
        return json.loads(self.get('manifests', manifest))

    def head(self):
        with closing(self.connect()) as conn:
            row = conn.execute('SELECT manifest FROM head WHERE id=1').fetchone()
            return row[0] if row else None

    def excluded(self, path):
        from .execution_capture import excluded
        return path.split('/')[0] == 'tool-outputs' or excluded(path) or managed_path(path) or any(p.startswith('.registered-') for p in Path(path).parts)

    def scan(self, root):
        from .workspace_io import open_file
        tree = {}
        def incomplete(error):
            raise PublicationHold('Workspace scan is incomplete: ' + str(error)) from error
        for directory, dirs, files in os.walk(root, followlinks=False, onerror=incomplete):
            dirs[:] = sorted(n for n in dirs if not self.excluded((Path(directory) / n).relative_to(root).as_posix()))
            for name in sorted(dirs + files):
                path = Path(directory) / name
                relative = path.relative_to(root).as_posix()
                if self.excluded(relative):
                    continue
                FileKey(self.namespace, relative)
                info = path.lstat()
                entry = dict(mode=stat.S_IMODE(info.st_mode))
                if stat.S_ISREG(info.st_mode):
                    with open_file(root, path) as stream:
                        data = stream.read()
                    entry.update(kind='file', blob=self.put('blobs', data))
                elif stat.S_ISLNK(info.st_mode):
                    entry.update(kind='symlink', target=os.readlink(path))
                elif stat.S_ISDIR(info.st_mode):
                    entry.update(kind='directory')
                else:
                    raise PublicationHold('Unsupported workspace file type: ' + relative)
                tree[relative] = entry
        return tree

    def materialize(self, tree, root):
        root.mkdir(parents=True)
        for relative, entry in sorted(tree.items(), key=lambda item: (len(Path(item[0]).parts), item[0])):
            FileKey(self.namespace, relative)
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if entry['kind'] == 'directory':
                path.mkdir(exist_ok=True)
            elif entry['kind'] == 'symlink':
                path.symlink_to(entry['target'])
            else:
                path.write_bytes(self.get('blobs', entry['blob']))
        for relative, entry in sorted(tree.items(), reverse=True):
            if entry['kind'] != 'symlink':
                (root / relative).chmod(entry['mode'])
        return root

    def copy_managed(self, root, *, include_git=True):
        from .workspace_layout import MANAGED
        for name in (*MANAGED, *(['.git'] if include_git else [])):
            source = self.public / name
            if source.is_dir():
                shutil.copytree(source, root / name, symlinks=True)
            elif source.exists():
                shutil.copy2(source, root / name)

    def begin(self, actor, *, execution_id=None, store=None, automatic=False, inspect_git=False):
        identity = execution_id or uuid.uuid4().hex
        with self.lock(), closing(self.connect()) as conn, conn:
            if store is not None:
                self.project_pending_evidence(store)
            if conn.execute('SELECT 1 FROM executions WHERE id=?', (identity,)).fetchone():
                raise PublicationHold('Execution already exists; recover its outcome without rerunning tools')
            for row in conn.execute('SELECT record FROM executions WHERE outcome IS NULL'):
                record = json.loads(row[0])
                if not record.get('owner') or record['owner'] != self.process_owner(record.get('pid')):
                    raise PublicationHold('Unfinished prior process execution; restore a selected checkpoint')
            public = conn.execute('SELECT manifest FROM head WHERE id=1').fetchone()[0]
            tree = self.tree(public)
            author = conn.execute('SELECT bases FROM authors WHERE actor=?', (actor,)).fetchone()
            bases = json.loads(author[0]) if author and not automatic else dict(tree)
            if author is None:
                conn.execute('INSERT INTO authors VALUES (?,?)', (actor, encoded(bases)))
            drafts = []
            for row in ([] if automatic else conn.execute('SELECT * FROM drafts WHERE actor=? ORDER BY rowid', (actor,))):
                candidate = json.loads(self.get('candidates', row['candidate']))
                drafts.append(dict(id=row['id'], candidate=row['candidate']))
                original = self.tree(candidate['after'])
                for path, entry in sorted(candidate['files'].items(), key=lambda item: (len(Path(item[0]).parts), item[0])):
                    if entry is None or entry['kind'] != 'directory':
                        for child in list(tree):
                            if child.startswith(path + '/'):
                                tree.pop(child)
                    if entry is not None:
                        for parent in Path(path).parents:
                            name = parent.as_posix()
                            if name != '.' and tree.get(name, {}).get('kind') != 'directory':
                                tree[name] = original[name]
                    if entry is None:
                        tree.pop(path, None)
                    else:
                        tree[path] = entry
            before = self.manifest(tree)
            outputs = [] if automatic else list(conn.execute('SELECT path,entry,private FROM host_outputs WHERE actor=?', (actor,)))
            output_before = {row['path']: json.loads(row['entry']) for row in outputs}
            private_git = conn.execute('SELECT private FROM private_git WHERE actor=?', (actor,)).fetchone()
            private_inputs = bool(drafts) or any(row['private'] for row in outputs) or bool(inspect_git and not automatic and private_git and private_git[0])
            record = dict(id=identity, actor=actor, public=public, before=before, bases=bases, drafts=drafts,
                          private_inputs=private_inputs, output_before=output_before, automatic=automatic,
                          pid=os.getpid(), owner=self.process_owner(os.getpid()))
            conn.execute('INSERT INTO executions(id,record) VALUES (?,?)', (identity, encoded(record)))
        root = self.materialize(tree, self.private / 'executions' / identity)
        self.copy_managed(root, include_git=False)
        self.prepare_git(root, actor, automatic=automatic)
        self.materialize(output_before, root / 'tool-outputs')
        self.stage('execution_started', identity)
        return Execution(identity, actor, root, public, before, bases, tuple(drafts), private_inputs, output_before, automatic, self.git_fingerprint(root))

    def prepare_git(self, root, actor, *, automatic=False):
        from .registration_evidence import native_repository
        with self.lock(), closing(self.connect()) as conn:
            if not (self.public / '.git').exists():
                return
            native_repository(self.public)
            row = None if automatic else conn.execute('SELECT snapshot FROM private_git WHERE actor=?', (actor,)).fetchone()
            source = self.private / 'git-repositories' / row[0] if row else self.public
            native_repository(source)
            self.safe_git(source)
            shutil.copytree(source / '.git', root / '.git', symlinks=True)
            for item in (self.public / '.git/objects').rglob('*'):
                target = root / '.git/objects' / item.relative_to(self.public / '.git/objects')
                if item.is_dir():
                    target.mkdir(exist_ok=True)
                elif not target.exists():
                    shutil.copy2(item, target, follow_symlinks=False)
                elif target.read_bytes() != item.read_bytes():
                    raise PublicationHold('Public Git object differs from the private copy')
            result = subprocess.run(['git', '--no-replace-objects', '-C', str(self.public), 'rev-parse', 'HEAD'],
                                    capture_output=True, check=True, timeout=10)
            reference = root / '.git/refs/pf-public/head'
            reference.parent.mkdir(parents=True, exist_ok=True)
            reference.write_bytes(result.stdout)
            env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
            env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_ATTR_NOSYSTEM='1',
                       GIT_INDEX_FILE=str(root / '.git/index'), GIT_OBJECT_DIRECTORY=str(root / '.git/objects'))
            refreshed = subprocess.run(['git', '--no-replace-objects', '--git-dir=' + str(self.public / '.git'),
                '--work-tree=' + str(root), '-c', 'core.fsmonitor=false', '-c', 'core.hooksPath=/dev/null',
                'update-index', '--no-split-index', '--refresh'], env=env, capture_output=True, timeout=10)
            if refreshed.returncode not in (0, 1):
                raise PublicationHold('Private Git index refresh failed: ' + refreshed.stderr.decode(errors='replace'))

    @staticmethod
    def git_fingerprint(root):
        git = root / '.git'
        if not git.exists() and not git.is_symlink():
            return None
        WorkspacePublication.safe_git(root)
        return checksum(encoded({p.relative_to(git).as_posix(): checksum(p.read_bytes())
            for p in git.rglob('*') if p.is_file()}))

    @staticmethod
    def safe_git(root):
        git = root / '.git'
        if git.is_symlink() or not git.is_dir() or any(
                path.is_symlink() or not (path.is_file() or path.is_dir()) for path in git.rglob('*')):
            raise PublicationHold('Private Git metadata contains unsupported links or special files')

    def save_git(self, execution):
        if execution.automatic or not (execution.root / '.git').exists() or self.git_fingerprint(execution.root) == execution.git_before:
            return
        from .registration_evidence import native_repository
        native_repository(execution.root)
        self.safe_git(execution.root)
        snapshot = uuid.uuid4().hex
        target = self.private / 'git-repositories' / snapshot
        target.mkdir()
        shutil.copytree(execution.root / '.git', target / '.git', symlinks=True)
        for directory, _, files in os.walk(target, topdown=False):
            for name in files:
                with (Path(directory) / name).open('rb') as stream:
                    os.fsync(stream.fileno())
            sync_directory(directory)
        sync_directory(target.parent)
        with self.lock(), closing(self.connect()) as conn, conn:
            prior = conn.execute('SELECT private FROM private_git WHERE actor=?', (execution.actor,)).fetchone()
            private = bool(prior and prior[0]) or self.git_fingerprint(execution.root) != execution.git_before
            conn.execute('INSERT OR REPLACE INTO private_git VALUES (?,?,?)', (execution.actor, snapshot, private))

    def outcome(self, identity):
        with closing(self.connect()) as conn:
            row = conn.execute('SELECT outcome FROM executions WHERE id=?', (identity,)).fetchone()
            if row is None or row[0] is None:
                raise PublicationHold('Publication outcome unknown; do not rerun the tool')
            return json.loads(row[0])

    def _merge_entry(self, base, current, candidate):
        if current == base or current == candidate:
            return candidate, False
        if candidate == base:
            return current, False
        if not all(entry and entry['kind'] == 'file' for entry in (base, current, candidate)):
            return None, True
        if current['mode'] != base['mode'] and candidate['mode'] != base['mode'] and current['mode'] != candidate['mode']:
            return None, True
        data = [self.get('blobs', entry['blob']) for entry in (base, current, candidate)]
        if any(b'\0' in value for value in data):
            return None, True
        try:
            for value in data:
                value.decode('utf-8')
        except UnicodeError:
            return None, True
        with tempfile.TemporaryDirectory(dir=self.private) as temporary:
            paths = [Path(temporary) / str(i) for i in range(3)]
            for path, value in zip(paths, data):
                path.write_bytes(value)
            result = subprocess.run(['git', 'merge-file', '-p', str(paths[2]), str(paths[0]), str(paths[1])],
                                    capture_output=True, timeout=min(30, self.timeout))
        if 1 <= result.returncode <= 127:
            return None, True
        if result.returncode:
            raise PublicationHold('Text merge failed: ' + result.stderr.decode(errors='replace'))
        return dict(kind='file', mode=current['mode'] if candidate['mode'] == base['mode'] else candidate['mode'],
                    blob=self.put('blobs', result.stdout)), False

    def _candidate(self, execution, after, status, closure):
        before = self.tree(execution.before)
        changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
        files = {path: after.get(path) for path in changed}
        bases = {path: execution.bases.get(path) for path in changed}
        inherited = {}
        for draft in execution.drafts:
            old = json.loads(self.get('candidates', draft['candidate']))
            if set(changed) & old['files'].keys():
                inherited[draft['id']] = draft['candidate']
                for path in old['files']:
                    files[path] = after.get(path)
                    bases[path] = old['bases'][path]
        outputs = self.scan(execution.root / 'tool-outputs')
        outputs = {path: entry for path, entry in outputs.items() if execution.output_before.get(path) != entry}
        return dict(execution=execution.id, actor=execution.actor, public=execution.public,
                    before=execution.before, after=self.manifest(after), files=files, bases=bases,
                    changed=changed, inherited=inherited, status=status, closure=closure,
                    host_outputs=outputs, private_inputs=execution.private_inputs, automatic=execution.automatic)

    def finish(self, execution, *, status='succeeded', closure, store=None, capture_event=None, private_output=False):
        with closing(self.connect()) as conn:
            row = conn.execute('SELECT candidate,outcome FROM executions WHERE id=?', (execution.id,)).fetchone()
        if row is None:
            raise PublicationHold('Unknown workspace execution')
        if not closure.get('boundary_closed') or status == 'result_unknown':
            raise PublicationHold('Workspace execution boundary is not closed')
        if row['candidate']:
            candidate_id = row['candidate']
            candidate = json.loads(self.get('candidates', candidate_id))
            if candidate['status'] != status or candidate['closure'] != closure:
                raise PublicationHold('Conflicting execution finalization')
        else:
            git_changed = self.git_fingerprint(execution.root) != execution.git_before
            self.save_git(execution)
            private_output |= git_changed
            after = self.scan(execution.root)
            self.stage('blobs_durable', execution.id)
            candidate = self._candidate(execution, after, status, closure)
            candidate['capture_event'] = capture_event
            candidate['private_inputs'] |= private_output
            candidate_id = self.put('candidates', encoded(candidate))
            self.stage('candidate_durable', execution.id)
            with closing(self.connect()) as conn, conn:
                conn.execute('UPDATE executions SET candidate=? WHERE id=? AND candidate IS NULL', (candidate_id, execution.id))
            self.stage('intent_durable', execution.id)
        if row['outcome']:
            self.recover_projections(store)
            return json.loads(row['outcome'])
        try:
            return self._publish(execution, candidate_id, candidate, store=store)
        except Exception as exc:
            try:
                result = self.outcome(execution.id)
            except PublicationHold:
                raise exc
            self.recover_projections(store)
            return result

    def _publish(self, execution, candidate_id, candidate, against=None, store=None):
        deadline = time.monotonic() + self.timeout
        while True:
            if time.monotonic() >= deadline:
                raise PublicationHold('Publication deadline exceeded')
            head = self.head()
            current, before = self.tree(head), self.tree(candidate['public'])
            files = candidate['files']
            advanced = [path for path in files if current.get(path) != candidate['bases'].get(path)]
            stale = [path for path in files if candidate['bases'].get(path) != before.get(path)]
            conflicts, merged = [], dict(current)
            if not files:
                state = 'unchanged'
            elif candidate['status'] != 'succeeded':
                state = 'retained_partial'
            elif against is not None and head != against:
                state = 'needs_confirmation'
            elif against is None and (stale or candidate['inherited']):
                state = 'needs_confirmation'
            else:
                for path, entry in files.items():
                    value, conflict = (entry, False) if against is not None else self._merge_entry(before.get(path), current.get(path), entry)
                    if conflict:
                        conflicts.append(path)
                    elif value is None:
                        merged.pop(path, None)
                    else:
                        merged[path] = value
                for path in merged:
                    for parent in Path(path).parents:
                        if str(parent) != '.' and merged.get(parent.as_posix(), {}).get('kind') != 'directory':
                            conflicts.append(path)
                state = 'conflict' if conflicts else 'merged' if any(current.get(p) != before.get(p) for p in files) else 'published'
            published = state in ('published', 'merged')
            target = self.manifest(merged) if published else head
            self.stage('manifest_durable', execution.id)
            projection = None
            if published:
                projection = self.materialize(merged, self.private / 'projections' / uuid.uuid4().hex)
                self.copy_managed(projection)
            result = dict(execution=execution.id, candidate=candidate_id, status=state, previous_head=head, head=target,
                          changed=candidate['changed'], paths=sorted(files), advanced=advanced,
                          needs_confirmation=stale, conflicts=sorted(set(conflicts)), author_after=candidate['after'],
                          actor=execution.actor, capture_event=candidate.get('capture_event'),
                          outputs_public=not candidate.get('private_inputs') and candidate['status'] == 'succeeded' and state in ('published', 'merged', 'unchanged'))
            with self.lock(), closing(self.connect()) as conn, conn:
                if store is not None:
                    self.project_pending_evidence(store)
                conn.execute('BEGIN IMMEDIATE')
                row = conn.execute('SELECT outcome FROM executions WHERE id=?', (execution.id,)).fetchone()
                if row['outcome']:
                    return json.loads(row['outcome'])
                if conn.execute('SELECT manifest FROM head WHERE id=1').fetchone()[0] != head:
                    if projection:
                        shutil.rmtree(projection)
                    continue
                revisions_valid = all(tuple(conn.execute('SELECT candidate FROM drafts WHERE id=? AND actor=?', (identity, execution.actor)).fetchone() or ()) == (revision,)
                                      for identity, revision in candidate['inherited'].items())
                if not revisions_valid:
                    published = False
                    result.update(status='not_published', head=head, reason='candidate_revision_changed', outputs_public=False)
                    if projection:
                        shutil.rmtree(projection)
                        projection = None
                if published:
                    conn.execute('UPDATE head SET manifest=? WHERE id=1', (target,))
                    bases = json.loads(conn.execute('SELECT bases FROM authors WHERE actor=?', (execution.actor,)).fetchone()[0])
                    if not candidate.get('automatic'):
                        bases.update(files)
                    conn.execute('UPDATE authors SET bases=? WHERE actor=?', (encoded(bases), execution.actor))
                if revisions_valid:
                    for identity in candidate['inherited']:
                        conn.execute('DELETE FROM drafts WHERE id=? AND actor=?', (identity, execution.actor))
                if files and not published and (revisions_valid or not candidate.get('resolution')):
                    conn.execute('INSERT INTO drafts VALUES (?,?,?)', (execution.id, execution.actor, candidate_id))
                for path, entry in candidate.get('host_outputs', {}).items():
                    conn.execute('INSERT OR REPLACE INTO host_outputs VALUES (?,?,?,?)',
                        (execution.actor, path, encoded(entry), not result['outputs_public']))
                result['sequence'] = conn.execute("SELECT COALESCE(MAX(json_extract(outcome,'$.sequence')),0)+1 FROM executions").fetchone()[0]
                conn.execute('UPDATE executions SET outcome=? WHERE id=?', (encoded(result), execution.id))
                self.stage('before_commit', execution.id)
                conn.commit()
                self.stage('committed', execution.id)
                if projection:
                    self._exchange_projection(projection)
                    self.stage('projection_swapped', execution.id)
                if store is not None:
                    self.project_evidence(store, result)
                    self.stage('evidence_durable', execution.id)
            self.stage('receipt', execution.id)
            return result

    def _exchange_projection(self, projection):
        if (projection / '.git').exists():
            shutil.rmtree(projection / '.git')
        if (self.public / '.git').exists():
            shutil.copytree(self.public / '.git', projection / '.git', symlinks=True)
        libc = ctypes.CDLL(None, use_errno=True)
        rename = libc.renameat2
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(projection), -100, os.fsencode(self.public), 2):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        sync_directory(self.public.parent)
        sync_directory(projection.parent)

    def recover_projection(self):
        self.recover_projections()

    @staticmethod
    def process_owner(pid):
        if not isinstance(pid, int):
            return None
        try:
            fields = Path('/proc/' + str(pid) + '/stat').read_text().rsplit(')', 1)[1].split()
            return str(pid) + ':' + fields[19]
        except (FileNotFoundError, ProcessLookupError):
            return None

    def recover_projections(self, store=None):
        while True:
            head = self.head()
            projection = self.materialize(self.tree(head), self.private / 'projections' / uuid.uuid4().hex)
            self.copy_managed(projection)
            with self.lock():
                if self.head() != head:
                    shutil.rmtree(projection)
                    continue
                self._exchange_projection(projection)
                if store is not None:
                    self.project_pending_evidence(store)
                return

    def project_pending_evidence(self, store):
        with closing(self.connect()) as conn:
            rows = conn.execute("SELECT e.outcome FROM executions e LEFT JOIN projected p ON p.id=e.id WHERE e.outcome IS NOT NULL AND p.id IS NULL ORDER BY json_extract(e.outcome,'$.sequence')").fetchall()
        for row in rows:
            self.project_evidence(store, json.loads(row[0]))

    def project_evidence(self, store, result):
        if result['status'] in ('published', 'merged'):
            self.publish_evidence(store, result['execution'], result['head'], result['paths'], result['actor'])
        if result.get('capture_event'):
            store.workspace_output_visibility(result['capture_event'], result['outputs_public'])
        with closing(self.connect()) as conn, conn:
            conn.execute('INSERT OR IGNORE INTO projected VALUES (?)', (result['execution'],))

    def resolve(self, actor, identity, *, against, revision, store=None):
        with closing(self.connect()) as conn:
            row = conn.execute('SELECT candidate FROM drafts WHERE id=? AND actor=?', (identity, actor)).fetchone()
            if row is None or row[0] != revision:
                raise PublicPFError('Candidate revision does not match the current draft')
        original = json.loads(self.get('candidates', revision))
        new_id = uuid.uuid4().hex
        candidate = dict(original, execution=new_id, inherited={identity: revision}, status='succeeded', capture_event=None, resolution=True)
        candidate_id = self.put('candidates', encoded(candidate))
        record = dict(id=new_id, actor=actor, against=against, revision=revision, draft=identity,
                      pid=os.getpid(), owner=self.process_owner(os.getpid()))
        with closing(self.connect()) as conn, conn:
            conn.execute('INSERT INTO executions(id,record,candidate) VALUES (?,?,?)', (new_id, encoded(record), candidate_id))
            conn.execute('INSERT INTO resolutions VALUES (?,?)', (new_id, encoded(record)))
        execution = Execution(new_id, actor, self.private / 'executions' / original['execution'],
                              original['public'], original['before'], original['bases'], ())
        try:
            return self._publish(execution, candidate_id, candidate, against=against, store=store)
        except Exception as exc:
            try:
                result = self.outcome(new_id)
            except PublicationHold:
                raise PublicationHold('Resolution outcome is unknown; manual recovery required') from exc
            self.recover_projections(store)
            return result

    def seed_evidence(self, store):
        with self.lock():
            with closing(self.connect()) as conn:
                head = conn.execute('SELECT manifest FROM initial').fetchone()[0]
            self.publish_evidence(store, 'initial', head, self.tree(head), None)
            self.project_pending_evidence(store)

    def publish_evidence(self, store, identity, head, files, actor):
        with closing(self.connect()) as conn, conn:
            row = conn.execute('SELECT event FROM evidence WHERE id=?', (identity,)).fetchone()
            event = row[0] if row else store.begin_event('workspace_publication', dict(execution=identity, head=head))
            if row is None:
                conn.execute('INSERT INTO evidence VALUES (?,?)', (identity, event))
        from copy import copy
        with closing(store.connect()) as conn:
            request = json.loads(conn.execute('SELECT request FROM requests WHERE event_id=?', (event,)).fetchone()[0])
        writer = copy(store)
        writer.identity = dict(store.identity, branch_id=event.split('/')[1], role=request.get('role', store.identity.get('role')))
        tree = self.tree(head)
        with writer.batch():
            for index, path in enumerate(sorted(files)):
                entry = tree.get(path)
                entry = entry or {'kind': 'deleted'}
                slot = 'published_' + str(index)
                try:
                    writer.get_content(event + ':' + slot)
                except KeyError:
                    writer.version(event, slot, self.get('blobs', entry['blob']) if entry['kind'] == 'file' else b'', layer='file_bytes',
                        object_id=path, workspace_namespace=self.namespace, workspace_publication=True,
                        publication_head=head, producer_role=actor, workspace_entry=entry,
                        extent='full' if entry['kind'] == 'file' else 'metadata_only')
            if writer.read_event(event)['result']['status'] == 'result_unknown':
                writer.complete(event)

    def draft_rows(self, actor):
        with closing(self.connect()) as conn:
            return [dict(row) for row in conn.execute('SELECT * FROM drafts WHERE actor=? ORDER BY id', (actor,))]

    @staticmethod
    def draft_ref(row):
        return row['id'] + '@' + row['candidate']

    def draft(self, actor, reference):
        rows = self.draft_rows(actor)
        row = next((row for row in rows if self.draft_ref(row) == reference), None)
        if row is None:
            raise PublicPFError('Draft revision is unavailable. Run pf workspace status.')
        return row, json.loads(self.get('candidates', row['candidate']))

    def differences(self, candidate, head):
        current = self.tree(head)
        text = []
        for path, entry in sorted(candidate['files'].items()):
            base = candidate['bases'].get(path)
            for label, value in (('shared', current.get(path)), ('draft', entry)):
                if base == value:
                    continue
                def lines(item):
                    if item is None:
                        return ['[missing]\n']
                    if item['kind'] != 'file':
                        return [json.dumps(item, sort_keys=True) + '\n']
                    data = self.get('blobs', item['blob'])
                    try:
                        if b'\0' in data:
                            raise UnicodeError()
                        return data.decode('utf-8').splitlines(keepends=True)
                    except UnicodeError:
                        return ['[binary blob=' + item['blob'] + ']\n']
                old_ref = base.get('blob', checksum(encoded(base))) if base else 'missing'
                new_ref = value.get('blob', checksum(encoded(value))) if value else 'missing'
                if base and value and base['kind'] == value['kind'] == 'file' and base['mode'] != value['mode']:
                    text.append(f'--- {path} base@{old_ref}\n+++ {path} {label}@{new_ref}\n'
                                f"old mode {base['mode']:04o}\nnew mode {value['mode']:04o}\n")
                diff = difflib.unified_diff(lines(base), lines(value),
                    fromfile=path + ' base@' + old_ref, tofile=path + ' ' + label + '@' + new_ref)
                text.extend(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n' for line in diff)
        return ''.join(text)

    def receipt(self, result, *, full=False):
        if result['status'] == 'unchanged':
            return ''
        candidate = json.loads(self.get('candidates', result['candidate']))
        if result.get('reason') == 'candidate_revision_changed' and candidate.get('resolution'):
            return '[PF workspace not_published] Selected draft revision is no longer current. Existing drafts were preserved.\n' + \
                'published_head=' + result['head'] + '\nRun pf workspace status to select a current draft.'
        ref = result['execution'] + '@' + result['candidate']
        header = '[PF workspace ' + result['status'] + '] ' + ', '.join(result['paths'])
        header += '\ndraft=' + ref + '; published_head=' + result['head']
        header += '; compared_head=' + result['previous_head']
        if result['status'] == 'needs_confirmation':
            header += '\nVERSION CHANGED: confirmation required. Draft saved; not published.'
        elif result['status'] in ('conflict', 'retained_partial', 'not_published'):
            header += '\nDraft saved; not published.'
        if result['advanced']:
            header += '\nShared version advanced; result=' + result['status'] + '.'
        if result['status'] in ('published', 'merged'):
            return header
        diff = self.differences(candidate, result['previous_head'])
        if not full and len(diff) > 6000:
            diff = diff[:6000] + '\n[diff truncated]\n'
        header += '\n' + diff
        return header + '\nFull diff: pf workspace resolve ' + ref + ' --full\n' + \
            'Choose this public version: pf workspace resolve ' + ref + ' --against ' + result['previous_head'] + \
            '\nOr: pf workspace discard ' + ref + '\nSuccessful business API calls remain successful. Do not repeat them to fix files.'

    def status(self, actor):
        rows = self.draft_rows(actor)
        with closing(self.connect()) as conn:
            author = conn.execute('SELECT bases FROM authors WHERE actor=?', (actor,)).fetchone()
        bases = json.loads(author[0]) if author else self.tree(self.head())
        lines = ['published_head=' + self.head() + '; drafts=' + str(len(rows)),
                 'comparison_base=' + self.manifest(bases)]
        for row in rows:
            candidate = json.loads(self.get('candidates', row['candidate']))
            lines.append('draft=' + self.draft_ref(row) + '; paths=' + ', '.join(sorted(candidate['files'])))
        return '\n'.join(lines)

    def workspace_command(self, actor, args, *, day, store=None):
        operation = args['operation']
        if operation == 'status':
            if args.get('confirm_finish'):
                return self.confirm_finish(actor, day, args['confirm_finish'])
            return self.status(actor)
        row, candidate = self.draft(actor, args['candidate'])
        if operation == 'discard':
            with self.lock(), closing(self.connect()) as conn, conn:
                changed = conn.execute('DELETE FROM drafts WHERE id=? AND actor=? AND candidate=?',
                    (row['id'], actor, row['candidate'])).rowcount
                if not changed:
                    raise PublicPFError('Draft revision changed. Run pf workspace status.')
                conn.execute('INSERT INTO discards VALUES (?,?,?)', (row['id'], actor, row['candidate']))
            return 'discarded draft=' + args['candidate'] + '; audit retained; files not published.'
        if not args.get('against'):
            with closing(self.connect()) as conn:
                result = json.loads(conn.execute('SELECT outcome FROM executions WHERE id=?', (row['id'],)).fetchone()[0])
            result['previous_head'] = result['head'] = self.head()
            return self.receipt(result, full=args.get('full', False))
        against = args['against']
        if len(against) != 64 or any(c not in '0123456789abcdef' for c in against):
            raise PublicPFError('Use the exact published_head from status or resolve.')
        result = self.resolve(actor, row['id'], against=against, revision=row['candidate'], store=store)
        return self.receipt(result, full=args.get('full', False))

    def draft_signature(self, actor, day):
        return checksum(encoded(dict(actor=actor, day=day, drafts=self.draft_rows(actor))))

    def week_reminder(self, actor, day):
        with self.lock(), closing(self.connect()) as conn, conn:
            if not self.draft_rows(actor) or conn.execute('SELECT 1 FROM draft_reminders WHERE actor=? AND day=?', (actor, day)).fetchone():
                return ''
            conn.execute('INSERT INTO draft_reminders VALUES (?,?)', (actor, day))
            return 'Your unpublished drafts remain private.\n' + self.status(actor) + '\nUse pf workspace resolve or discard.'

    def check_finish(self, actor, day):
        with self.lock(), closing(self.connect()) as conn, conn:
            if not self.draft_rows(actor):
                return ''
            signature = self.draft_signature(actor, day)
            row = conn.execute('SELECT confirmed FROM finish_confirmations WHERE actor=? AND day=? AND signature=?',
                (actor, day, signature)).fetchone()
            if row and row[0]:
                return ''
            conn.execute('INSERT OR IGNORE INTO finish_confirmations VALUES (?,?,?,0)', (actor, day, signature))
            return 'Weekly finish not completed: your drafts are unpublished.\n' + self.status(actor) + \
                '\nResolve or discard them, or explicitly keep this set with: pf workspace status --confirm-finish ' + signature + \
                '\nThen send your final weekly message or request next-week separately. No business operation was repeated.'

    def confirm_finish(self, actor, day, signature):
        with self.lock(), closing(self.connect()) as conn, conn:
            if signature != self.draft_signature(actor, day):
                raise PublicPFError('Draft set changed; attempt weekly finish again.')
            changed = conn.execute('UPDATE finish_confirmations SET confirmed=1 WHERE actor=? AND day=? AND signature=?',
                (actor, day, signature)).rowcount
            if not changed:
                raise PublicPFError('No finish reminder matches this role, week and draft set.')
            return 'finish_confirmed; drafts remain private. Send your final weekly message or request next-week separately.'
