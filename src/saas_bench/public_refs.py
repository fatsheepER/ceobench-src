"""Private, durable public-reference allocation outside rollback snapshots."""
from contextlib import contextmanager
from pathlib import Path
import re
import sqlite3


class PublicReferenceUnavailable(RuntimeError):
    pass


class PublicRefs:
    def __init__(self, path, *, snapshot=None, require_existing=False):
        self.path = Path(path).resolve()
        self.unavailable = None
        if require_existing and not self.path.is_file():
            self.unavailable = 'Public reference allocator is missing; new references require the original allocation history'
            self._read_path = Path(snapshot).resolve() if snapshot else self.path
            return
        self._read_path = self.path
        self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        with self._connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS references_by_key (
                kind TEXT NOT NULL, role TEXT NOT NULL, private_key TEXT NOT NULL,
                number INTEGER NOT NULL CHECK(number>0), public_ref TEXT NOT NULL,
                PRIMARY KEY(kind,role,private_key), UNIQUE(kind,public_ref), UNIQUE(kind,role,number))''')
            conn.execute('''CREATE TABLE IF NOT EXISTS counters (
                kind TEXT NOT NULL, role TEXT NOT NULL, last_number INTEGER NOT NULL,
                PRIMARY KEY(kind,role))''')
            conn.commit()
        self.path.chmod(0o600)
        if snapshot and Path(snapshot).is_file():
            self.reserve(snapshot)

    def require_available(self):
        if not self.path.is_file() and self.unavailable is None:
            self.unavailable = 'Public reference allocator is missing; new references require the original allocation history'
        if self.unavailable:
            raise PublicReferenceUnavailable(self.unavailable)

    def reserve(self, snapshot):
        if self.unavailable:
            return
        with self._connect() as conn:
            conn.execute('ATTACH DATABASE ? AS frozen', (str(Path(snapshot).resolve()),))
            conn.execute('BEGIN IMMEDIATE')
            conflict = conn.execute('''SELECT 1 FROM frozen.references_by_key f
                JOIN main.references_by_key l ON l.kind=f.kind AND
                    ((l.role=f.role AND l.private_key=f.private_key) OR l.public_ref=f.public_ref OR
                     (l.role=f.role AND l.number=f.number))
                WHERE l.private_key<>f.private_key OR l.public_ref<>f.public_ref OR l.role<>f.role OR l.number<>f.number
                LIMIT 1''').fetchone()
            if conflict:
                self.unavailable = 'Public reference allocator conflicts with the selected checkpoint; new references require the original allocation history'
                self._read_path = Path(snapshot).resolve()
                return
            conn.execute('INSERT OR IGNORE INTO main.references_by_key SELECT * FROM frozen.references_by_key')
            conn.execute('''INSERT INTO main.counters SELECT * FROM frozen.counters WHERE 1
                ON CONFLICT(kind,role) DO UPDATE SET last_number=MAX(counters.last_number,excluded.last_number)''')
            conn.commit()

    @contextmanager
    def _connect(self, *, readonly=False):
        location = self._read_path.as_uri() + '?mode=ro' if readonly else str(self.path)
        conn = sqlite3.connect(location, uri=readonly, timeout=30)
        try:
            conn.execute('PRAGMA synchronous=FULL')
            yield conn
        finally:
            conn.close()

    def assign(self, kind, role, private_key):
        if kind not in ('operation', 'event', 'capture'):
            raise ValueError('Unsupported public reference kind')
        return self._allocate(kind, role, private_key, kind)[0]

    def ordinal(self, namespace, role, private_key):
        return self._allocate('ordinal:' + namespace, role, private_key, None)[1]

    def _allocate(self, namespace, role, private_key, kind):
        if not isinstance(namespace, str) or not namespace or not re.fullmatch(r'[a-z][a-z0-9_]*', role):
            raise ValueError('Public references require a namespace and role')
        if not isinstance(private_key, str) or not private_key:
            raise ValueError('Public references require a private identity')
        if self.unavailable:
            if not self._read_path.is_file():
                raise PublicReferenceUnavailable(self.unavailable)
            with self._connect(readonly=True) as conn:
                row = conn.execute('SELECT role,public_ref,number FROM references_by_key WHERE kind=? AND role=? AND private_key=?',
                                   (namespace, role, private_key)).fetchone()
            if row and row[0] == role:
                return row[1], row[2]
            raise PublicReferenceUnavailable(self.unavailable)
        with self._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            row = conn.execute('SELECT role,public_ref,number FROM references_by_key WHERE kind=? AND role=? AND private_key=?',
                               (namespace, role, private_key)).fetchone()
            if row:
                return row[1], row[2]
            previous = conn.execute('SELECT last_number FROM counters WHERE kind=? AND role=?', (namespace, role)).fetchone()
            number = previous[0] + 1 if previous else 1
            public_ref = f'{role}:{"" if kind in (None, "operation") else kind}{number}'
            conn.execute('INSERT INTO references_by_key VALUES(?,?,?,?,?)', (namespace, role, private_key, number, public_ref))
            conn.execute('''INSERT INTO counters VALUES(?,?,?) ON CONFLICT(kind,role)
                DO UPDATE SET last_number=excluded.last_number''', (namespace, role, number))
            conn.commit()
            return public_ref, number

    def resolve(self, kind, public_ref):
        if self.unavailable and not self._read_path.is_file():
            return None
        with self._connect(readonly=True) as conn:
            row = conn.execute('SELECT private_key FROM references_by_key WHERE kind=? AND public_ref=?',
                               (kind, public_ref)).fetchone()
            return row[0] if row else None

    def snapshot(self, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._connect(readonly=True) as conn:
            target = sqlite3.connect(destination)
            try:
                conn.backup(target)
            finally:
                target.close()
        destination.chmod(0o600)
        return str(destination)
