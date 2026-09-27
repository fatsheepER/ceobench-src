"""Private SQL evidence, never mounted into the agent workspace.

Blob/version storage adapted from provenancefs/store.py at
924e7f7c3c094f8db18dfb17e2fe64c77ec1b843. Unlike record_write there,
every execution has its own version, including identical responses.
"""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid

FORMAT = 'ceobench.evidence-records.v1'


def encoded(value):
    return json.dumps(value, ensure_ascii=True, separators=(',', ':'), sort_keys=True).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def whole_rows(body, losses=()):
    """Compare public rows only, with explicit types and duplicate multiplicity."""
    data = json.loads(body)
    if not data.get('success'):
        return {'status': 'unsupported', 'reason': 'query_failed'}, None
    columns = data['columns']
    reason = next(iter(losses), None)
    if len(set(columns)) != len(columns):
        reason = 'duplicate_columns'
    if reason:
        return {'status': 'unsupported', 'reason': reason}, None

    def typed(value):
        if value is None:
            return ['null']
        if isinstance(value, bool):
            return ['bool', value]
        if isinstance(value, int):
            return ['int', str(value)]
        if isinstance(value, float) and math.isfinite(value):
            return ['float64', value.hex()]
        if isinstance(value, str):
            return ['text', value]
        raise ValueError('unsupported_value')

    try:
        rows = [encoded([[col, typed(row[col])] for col in columns]).decode()
                for row in data['rows']]
    except (KeyError, ValueError) as exc:
        return {'status': 'unsupported', 'reason': str(exc)}, None
    content = encoded({'format': 'whole-row-v1', 'columns': columns, 'rows': sorted(rows)})
    return {'status': 'available', 'scope': 'returned_subset' if data.get('truncated') else 'complete_for_query'}, content


class SQLEvidenceStore:
    def __init__(self, path, identity):
        self.path = Path(path).resolve()
        self.identity = identity
        for key in ('run_id', 'branch_id', 'data_source_id'):
            if not re.fullmatch(r'[A-Za-z0-9_-]+', identity.get(key, '')):
                raise ValueError('Invalid evidence identity: ' + key)
        if identity.get('format') != FORMAT:
            raise ValueError('Unsupported evidence format')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fault_path = self.path.with_suffix('.fault.json')
        self.fault = None
        self._local = threading.local()
        with closing(self.connect()) as conn, conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS identity (value BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS branches (
                    id TEXT PRIMARY KEY, parent TEXT, fork_seq INTEGER);
                CREATE TABLE IF NOT EXISTS blobs (
                    content_hash TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL, content BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS queries (id TEXT PRIMARY KEY, definition BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS requests (
                    event_id TEXT PRIMARY KEY, branch TEXT NOT NULL, seq INTEGER NOT NULL,
                    query_id TEXT, request BLOB NOT NULL, UNIQUE(branch, seq));
                CREATE TABLE IF NOT EXISTS results (
                    event_id TEXT PRIMARY KEY REFERENCES requests(event_id), record BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS versions (
                    version_id TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES requests(event_id),
                    previous_version TEXT, content_hash TEXT NOT NULL REFERENCES blobs(content_hash),
                    metadata BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS deliveries (
                    event_id TEXT PRIMARY KEY REFERENCES requests(event_id), record BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS capture_contexts (
                    token TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES requests(event_id));
                CREATE TABLE IF NOT EXISTS client_calls (
                    token TEXT PRIMARY KEY, event_id TEXT NOT NULL REFERENCES requests(event_id),
                    received BLOB);
                CREATE TABLE IF NOT EXISTS private_state (name TEXT PRIMARY KEY, value BLOB NOT NULL);
                CREATE INDEX IF NOT EXISTS versions_object_id ON versions(json_extract(metadata, '$.object_id'));
            ''')
            source = encoded({k: identity[k] for k in ('run_id', 'data_source_id', 'format')})
            existing = conn.execute('SELECT value FROM identity').fetchone()
            if existing and existing[0] != source:
                raise ValueError('Evidence database identity mismatch')
            if not existing:
                conn.execute('INSERT INTO identity VALUES (?)', (source,))
            branch = identity['branch_id']
            parent, cutoff = identity.get('parent_branch'), identity.get('fork_seq')
            existing = conn.execute('SELECT parent, fork_seq FROM branches WHERE id=?', (branch,)).fetchone()
            if existing and tuple(existing) != (parent, cutoff):
                raise ValueError('Evidence branch ancestry mismatch')
            if not existing:
                if parent and (not conn.execute('SELECT 1 FROM branches WHERE id=?', (parent,)).fetchone()
                               or cutoff != self.sequence(conn, parent)):
                    raise ValueError('Fork must use the saved parent cutoff')
                conn.execute('INSERT INTO branches VALUES (?,?,?)', (branch, parent, cutoff))

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=1)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=FULL')
        return conn

    def sequence(self, conn, branch=None):
        return conn.execute('SELECT coalesce(max(seq),0) FROM requests WHERE branch=?',
                            (branch or self.identity['branch_id'],)).fetchone()[0]

    def fail(self, exc):
        from .run_state import write_json
        previous = self.fault or {}
        if self.fault_path.exists():
            try:
                previous = json.loads(self.fault_path.read_bytes())
            except (OSError, ValueError):
                pass
        self.fault = dict(previous, reason=str(exc), time=now(), capture_status='missing')
        try:
            write_json(self.fault_path, self.fault)
        except OSError:
            pass  # The server's in-memory fault also blocks checkpoint publication.

    @property
    def execution_capture(self):
        return self.identity.get('capture_scope') == 'execution'

    def assert_healthy(self, quiescent=True, settle=0):
        """With settle, wait that long for server threads to record sends already observed by the client."""
        deadline = time.monotonic() + settle
        while True:
            if self.fault or self.fault_path.exists():
                raise RuntimeError('SQL evidence capture failed; collection stopped')
            if not quiescent:
                return
            with closing(self.connect()) as conn:
                pending = conn.execute('''SELECT event_id, event_id IN (SELECT event_id FROM results) FROM requests
                    WHERE event_id NOT IN (SELECT event_id FROM results)
                       OR (coalesce(json_extract(request, '$.requires_delivery'), 1) = 1
                           AND event_id NOT IN (SELECT event_id FROM deliveries))
                    ORDER BY 2 LIMIT 1''').fetchone()
                missing = conn.execute('SELECT token FROM client_calls WHERE received IS NULL LIMIT 1').fetchone()
                unknown = conn.execute("SELECT event_id FROM results WHERE json_extract(record, '$.status') = 'result_unknown' LIMIT 1").fetchone()
            # The response bytes reach the client before its handler records the send.
            if pending and pending[1] and time.monotonic() < deadline:
                time.sleep(.01)
                continue
            break
        if pending:
            raise RuntimeError('Unconfirmed SQL evidence outcome: ' + pending[0])
        if missing:
            raise RuntimeError('Unconfirmed client receive outcome: ' + missing[0])
        if unknown:
            raise RuntimeError('Unconfirmed execution outcome: ' + unknown[0])

    def begin_event(self, kind, request=None, parent=None, requires_delivery=False, admitted=False, **facts):
        deadline = time.monotonic() + 180
        while True:
            self.assert_healthy(quiescent=False)
            with closing(self.connect()) as conn, conn:
                conn.execute('BEGIN IMMEDIATE')
                paused = conn.execute("SELECT value FROM private_state WHERE name='admission_paused'").fetchone()
                if parent or admitted or not paused or not json.loads(paused[0]):
                    if parent:
                        self._visible(conn, parent)
                    seq = self.sequence(conn) + 1
                    event = f"{self.identity['run_id']}/{self.identity['branch_id']}/{seq}"
                    record = dict(event_id=event, seq=seq, author='ceo', kind=kind, request=request,
                                  started_at=now(), parent_event_id=parent, requires_delivery=requires_delivery,
                                  query_id=None, **facts)
                    conn.execute('INSERT INTO requests VALUES (?,?,?,?,?)',
                                 (event, self.identity['branch_id'], seq, None, encoded(record)))
                    return event
            if time.monotonic() >= deadline:
                raise TimeoutError('Checkpoint admission did not reopen')
            time.sleep(.01)

    def version(self, event, slot, payload, *, layer, object_id=None, **metadata):
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
        if not isinstance(payload, bytes):
            raise TypeError('Evidence content must be bytes or text')
        version = event + ':' + slot
        sha = digest(payload)
        with self._writer() as conn:
            self._visible(conn, event)
            previous = self._latest(conn, object_id)[0] if object_id else None
            record = dict(layer=layer, object_id=object_id, created_by_event=event, extent='full',
                          source_truncated=False, acquired_at=now(),
                          content_time={'status': 'unknown', 'reason': 'not_declared'})
            record.update(metadata)
            conn.execute('INSERT OR IGNORE INTO blobs VALUES (?,?,?)', (sha, len(payload), payload))
            conn.execute('INSERT INTO versions VALUES (?,?,?,?,?)',
                         (version, event, previous, sha, encoded(record)))
        return version

    def _latest(self, conn, object_id, layer=None):
        # Indexed by versions_object_id; visibility follows the branch ancestry.
        for row in conn.execute("SELECT version_id,event_id,content_hash FROM versions "
                                "WHERE json_extract(metadata, '$.object_id')=? "
                                "AND (? IS NULL OR json_extract(metadata, '$.layer')=?) ORDER BY rowid DESC",
                                (object_id, layer, layer)):
            try:
                self._visible(conn, row['event_id'])
                return row['version_id'], row['content_hash']
            except KeyError:
                continue
        return None, None

    def latest_version(self, object_id, layer=None):
        """The newest visible version of an object and its content hash."""
        conn = getattr(self._local, 'batch', None)
        if conn is not None:
            return self._latest(conn, object_id, layer)
        with closing(self.connect()) as conn:
            return self._latest(conn, object_id, layer)

    @contextmanager
    def _writer(self):
        conn = getattr(self._local, 'batch', None)
        if conn is not None:
            yield conn
            return
        with closing(self.connect()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            yield conn

    @contextmanager
    def batch(self):
        """Group many version writes into one durable transaction (one fsync)."""
        if getattr(self._local, 'batch', None) is not None:
            yield
            return
        conn = self.connect()
        try:
            conn.execute('BEGIN IMMEDIATE')
            self._local.batch = conn
            yield
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            self._local.batch = None
            conn.close()

    def complete(self, event, status='succeeded', **facts):
        record = dict(status=status, completed_at=None if status == 'result_unknown' else now(),
                      capture_status='missing' if self.fault or self.fault_path.exists() else 'complete', capture_gaps=[])
        record.update(facts)
        with closing(self.connect()) as conn, conn:
            conn.execute('INSERT INTO results VALUES (?,?)', (event, encoded(record)))

    def context(self, event, token=None):
        token = token or uuid.uuid4().hex
        if not re.fullmatch('[a-f0-9]{32}', token):
            raise ValueError('Invalid capture context')
        with closing(self.connect()) as conn, conn:
            self._visible(conn, event)
            conn.execute('INSERT INTO capture_contexts VALUES (?,?)', (token, event))
        return token

    def parent(self, token, active=True):
        if not token:
            return None
        with closing(self.connect()) as conn:
            row = conn.execute('SELECT event_id FROM capture_contexts WHERE token=?', (token,)).fetchone()
            if row is None:
                raise ValueError('Unknown capture context')
            self._visible(conn, row[0])
            if active and conn.execute('SELECT 1 FROM results WHERE event_id=?', (row[0],)).fetchone():
                raise ValueError('Capture context has ended')
            return row[0]

    def bind_client(self, event, token):
        if not re.fullmatch('[a-f0-9]{32}', token or ''):
            raise ValueError('Invalid client call key')
        with closing(self.connect()) as conn, conn:
            conn.execute('INSERT INTO client_calls VALUES (?,?,NULL)', (token, event))

    def received(self, token, payload):
        with closing(self.connect()) as conn, conn:
            row = conn.execute('SELECT event_id,received FROM client_calls WHERE token=?', (token,)).fetchone()
            if row is None or row['received'] is not None:
                raise ValueError('Unknown or already recorded client call')
            event = row['event_id']
            self._visible(conn, event)
        # All fields are client observations, never server-authored facts.
        for slot, value in payload.items():
            self.version(event, 'client_' + slot, bytes.fromhex(value) if slot == 'body' else encoded(value),
                         layer='program_' + slot, origin='standard_client_observation')
        with closing(self.connect()) as conn, conn:
            conn.execute('UPDATE client_calls SET received=? WHERE token=? AND received IS NULL',
                         (encoded(dict(receive_state=payload.get('state', 'received'), time=now())), token))

    def save_state(self, name, value):
        with closing(self.connect()) as conn, conn:
            conn.execute('INSERT OR REPLACE INTO private_state VALUES (?,?)', (name, encoded(value)))

    def load_state(self, name):
        with closing(self.connect()) as conn:
            row = conn.execute('SELECT value FROM private_state WHERE name=?', (name,)).fetchone()
        return json.loads(row[0]) if row else None

    def begin(self, raw_body, policy, parent=None):
        if self.fault or self.fault_path.exists():
            raise RuntimeError('SQL evidence capture failed; collection stopped')
        try:
            body = json.loads(raw_body) if raw_body else {}
        except (ValueError, UnicodeDecodeError):
            body = None
        sql = body.get('sql') if isinstance(body, dict) else None
        definition = ["query-v1", self.identity['run_id'], self.identity['branch_id'],
                      self.identity['data_source_id'], policy, sql, None]
        query = digest(encoded(definition)) if isinstance(sql, str) and sql.strip() else None
        with closing(self.connect()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            seq = self.sequence(conn) + 1
            event = f"{self.identity['run_id']}/{self.identity['branch_id']}/{seq}"
            if query:
                conn.execute('INSERT OR IGNORE INTO queries VALUES (?,?)', (query, encoded(definition)))
            request = dict(event_id=event, seq=seq, author='ceo', kind='sql_query',
                           started_at=now(), candidate_sql=sql, bound_parameters=None,
                           raw_request_hex=raw_body.hex(), parent_event_id=parent,
                           parent_reason=None if parent else 'unpropagated_context', query_id=query)
            conn.execute('INSERT INTO requests VALUES (?,?,?,?,?)',
                         (event, self.identity['branch_id'], seq, query, encoded(request)))
        return event

    def finish(self, event, status, headers, body, execution):
        started = time.monotonic()
        comparison, content = whole_rows(body, execution.get('losses', ()))
        result = json.loads(body)
        state = ('succeeded' if status == 200 else 'timed_out' if status == 504 else
                 'rejected' if status in (400, 403) else 'failed')
        record = dict(execution, status=state, http_status=status, headers=headers,
                      completed_at=now(), capture_status='complete', capture_gaps=[],
                      result_coverage=('truncated' if result.get('truncated') else
                                       'complete_for_query') if status == 200 else 'unknown',
                      comparison=comparison, receive_state='unknown')
        with closing(self.connect()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            request = conn.execute('SELECT query_id FROM requests WHERE event_id=?', (event,)).fetchone()
            previous = conn.execute('''SELECT v.version_id FROM versions v JOIN requests r USING(event_id)
                WHERE r.query_id=? AND v.version_id LIKE '%:public_response'
                ORDER BY v.rowid DESC LIMIT 1''', (request[0],)).fetchone()
            for slot, payload in [('public_response', body), ('comparison', content)]:
                if payload is None:
                    continue
                sha = digest(payload)
                conn.execute('INSERT OR IGNORE INTO blobs VALUES (?,?,?)', (sha, len(payload), payload))
                metadata = dict(layer='server_public_response' if slot == 'public_response' else 'comparison',
                                object_id=request[0], created_by_event=event,
                                extent='full', source_truncated=bool(result.get('truncated')),
                                acquired_at=record['completed_at'],
                                content_time={'status': 'unknown', 'reason': 'not_declared'})
                if slot == 'comparison':
                    metadata['derived_from'] = event + ':public_response'
                conn.execute('INSERT INTO versions VALUES (?,?,?,?,?)',
                             (event + ':' + slot, event,
                              previous[0] if previous and slot == 'public_response' else None,
                              sha, encoded(metadata)))
            record['capture_seconds'] = time.monotonic() - started
            conn.execute('INSERT INTO results VALUES (?,?)', (event, encoded(record)))

    def delivered(self, event, state):
        with closing(self.connect()) as conn, conn:
            conn.execute('INSERT INTO deliveries VALUES (?,?)',
                         (event, encoded(dict(send_state=state, receive_state='unknown', time=now()))))

    def _visible(self, conn, event):
        row = conn.execute('SELECT branch,seq FROM requests WHERE event_id=?', (event,)).fetchone()
        if row:
            branch, limit = self.identity['branch_id'], None
            while branch:
                if row['branch'] == branch and (limit is None or row['seq'] <= limit):
                    return
                ancestry = conn.execute('SELECT parent,fork_seq FROM branches WHERE id=?', (branch,)).fetchone()
                branch, limit = ancestry
        raise KeyError('Evidence is outside this branch: ' + event)

    def read_event(self, event):
        with closing(self.connect()) as conn:
            self._visible(conn, event)
            request = json.loads(conn.execute('SELECT request FROM requests WHERE event_id=?', (event,)).fetchone()[0])
            result = conn.execute('SELECT record FROM results WHERE event_id=?', (event,)).fetchone()
            delivery = conn.execute('SELECT record FROM deliveries WHERE event_id=?', (event,)).fetchone()
            definition = conn.execute('SELECT definition FROM queries WHERE id=?', (request['query_id'],)).fetchone()
            client = conn.execute('SELECT received FROM client_calls WHERE event_id=?', (event,)).fetchone()
            versions = conn.execute('SELECT version_id,metadata FROM versions WHERE event_id=? ORDER BY rowid', (event,)).fetchall()
            inputs = sorted({meta['derived_from'] for row in versions if (meta := json.loads(row['metadata'])).get('derived_from')})
            return dict(request=request, inputs=inputs, outputs=[row['version_id'] for row in versions],
                        client_receipt=json.loads(client[0]) if client and client[0] else {'receive_state': 'unknown'}, result=json.loads(result[0]) if result else
                        dict(status='result_unknown', completed_at=None, capture_status='missing'),
                        query_definition=json.loads(definition[0]) if definition else None,
                        delivery=json.loads(delivery[0]) if delivery else
                        dict(send_state='unknown', receive_state='unknown'))

    def get_content(self, version):
        with closing(self.connect()) as conn:
            row = conn.execute('SELECT * FROM versions WHERE version_id=?', (version,)).fetchone()
            if row is None:
                raise KeyError(version)
            self._visible(conn, row['event_id'])
            blob = conn.execute('SELECT content,size_bytes FROM blobs WHERE content_hash=?', (row['content_hash'],)).fetchone()
            if blob is None or len(blob[0]) != blob[1] or digest(blob[0]) != row['content_hash']:
                raise ValueError('Evidence blob checksum mismatch')
            return dict(version_id=version, previous_version=row['previous_version'],
                        blob_sha256=row['content_hash'], **json.loads(row['metadata'])), bytes(blob[0])

    def snapshot(self, target):
        self.assert_healthy()
        with closing(self.connect()) as conn, closing(sqlite3.connect(target)) as backup:
            conn.backup(backup)
            # Admission is live coordination, never restored as a permanent pause.
            backup.execute("DELETE FROM private_state WHERE name='admission_paused'")
            backup.commit()
            cutoff = self.sequence(conn)
        return dict(identity=self.identity, cutoff=cutoff, sha256=digest(Path(target).read_bytes()))
