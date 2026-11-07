"""Agent-facing evidence handles: an object name and that object's version number.

A handle names what a version belongs to and which version of it this is, e.g.
forecast.json@v3, query7@v2 (with its SQL shown alongside), analyze.py.out@v4 (the
stdout of that script) or cmd5@v1 (a command's return). The version number rises only
when the object's content differs from its previous version; every execution is still
recorded, and repeated acquisitions of the same content share one handle.

Object and version ordinals are allocated durably outside rollback state. Exact event
and capture references preserve individual executions even when content is identical.
"""
from collections import defaultdict
from contextlib import closing, contextmanager
import json
import re
import threading

from .public_errors import PublicPFError
from .sql_evidence import digest, encoded
from .role_policy import READABLE_ROLES
from .pf_deadline import check
from .workspace_layout import evidence_namespace


# Private state name of the per-kind object number table, e.g. {"query": {key: 7}}.
HANDLES = 'registration_handles'
# Harness-generated reads; mirrors execution_capture.READ_TOOLS (kept here to avoid a cycle).
READ_TOOLS = frozenset({'get_social_posts', 'get_cost_info', 'list_research_projects',
                        'get_market_overview', 'get_group_insights'})
STREAMS = {'stdout': 'out', 'stderr': 'err', 'executed_code': 'code'}
# Calls that may carry an agent note (design 3.2); mirrors bash_agent.tools.NOTE_TOOLS.
NOTE_KINDS = frozenset({'bash', 'write_file', 'edit_file'})
HOST_DAY_KINDS = frozenset({'dashboard_generation', 'registered_script_execution'})
VERSIONED = re.compile(r'(.+)@v([1-9][0-9]*)')
RAW_VERSION = re.compile(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[1-9][0-9]*:.+')
SINGLE = re.compile(r'([a-z_]+?)([1-9][0-9]*)')
# Accepted wherever the agent may write a handle; registered texts keep rN / rN.M.
HANDLE_PATTERN = r'^((?:(?:ceo|growth|ops_finance):)?(?:.+@v[1-9][0-9]*|[a-z_]+[1-9][0-9]*|r[1-9][0-9]*(\.[1-9][0-9]*)?)|[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[1-9][0-9]*:.+)$'
EXACT = re.compile(r'(ceo|growth|ops_finance):(event|capture)([1-9][0-9]*)')
RECORD = re.compile(r'(?:(?:ceo|growth|ops_finance):)?r[1-9][0-9]*(\.[1-9][0-9]*)?')


def body_digest(text):
    """Signature of a tool return before PF appended its handles."""
    return digest(str(text).encode('utf-8', 'surrogatepass'))


def one_line(sql):
    return ' '.join(str(sql).split())


def index(store):
    """The handle index shared by every reader of this store instance."""
    value = getattr(store, '_handle_index', None)
    epoch = store.workspace_visibility_epoch()
    if value is None or getattr(value, 'visibility_epoch', None) != epoch:
        value = store._handle_index = HandleIndex(store)
        value.visibility_epoch = epoch
    return value


def authority(store):
    from .public_refs import PublicRefs
    if not hasattr(store, 'public_refs'):
        store.public_refs = PublicRefs(store.path.parent / 'public-refs.sqlite')
    return store.public_refs


@contextmanager
def allocation(store):
    from .public_refs import PublicReferenceUnavailable
    try:
        yield authority(store)
    except PublicReferenceUnavailable as exc:
        store.fail(exc, failure_stage='public_reference_allocation')
        raise


def public_ref(store, kind, private_key):
    if private_key is None:
        return None
    event = private_key.split(':', 1)[0] if kind == 'capture' else private_key
    with closing(store.connect()) as conn:
        row = conn.execute('SELECT request FROM requests WHERE event_id=?', (event,)).fetchone()
    if row is None:
        raise PublicPFError('Captured reference is unavailable')
    request = json.loads(row[0])
    role = request.get('role', request.get('author', store.identity.get('role', 'ceo')))
    with allocation(store) as refs:
        return refs.assign(kind, role, private_key)


def resolve_ref(store, kind, ref):
    if not EXACT.fullmatch(ref):
        return ref
    value = authority(store).resolve(kind, ref)
    if value is None:
        raise PublicPFError('Captured reference is unavailable')
    return value


class HandleIndex:
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self.version_rowid = self.request_rowid = 0
        self.day = None
        self.events = {}                    # visible event -> (kind, request record, day)
        self.model_requests = []            # visible model sends, in acquisition order
        self.notes = {}                     # visible event -> (day, note) the agent attached to that call
        self.note_sources = {}
        self.info = {}                      # version -> (object key, group)
        self.groups = defaultdict(list)     # object key -> [group], oldest first
        self.code = {}                      # cli_python event -> executed code hash
        self.chain = None
        self.ordinals = {}
        self.bases = {}

    # --- incremental build ----------------------------------------------------

    def _ancestry(self, conn):
        chain, branch, limit = {}, self.store.identity['branch_id'], None
        while branch and branch not in chain:
            check()
            chain[branch] = limit
            row = conn.execute('SELECT parent,fork_seq FROM branches WHERE id=?', (branch,)).fetchone()
            branch, limit = (row[0], row[1]) if row else (None, None)
        return chain

    def refresh(self):
        check()
        with self.lock, closing(self.store.connect()) as conn:
            conn.execute('BEGIN')
            if self.chain is None:
                self.chain = self._ancestry(conn)
            for row in conn.execute('SELECT rowid,event_id,branch,seq,request FROM requests WHERE rowid>? '
                                    'ORDER BY rowid', (self.request_rowid,)):
                check()
                self.request_rowid = row['rowid']
                if not self.store.identity.get('role') and (row['branch'] not in self.chain or
                        (self.chain[row['branch']] is not None and row['seq'] > self.chain[row['branch']])):
                    continue
                record = json.loads(row['request'])
                args = record.get('request') if isinstance(record.get('request'), dict) else {}
                if record['kind'] in HOST_DAY_KINDS:
                    self.day = args.get('day', self.day)
                if record['kind'] == 'model_request' and record.get('role') == self.store.identity.get('role') and record.get('session_id') == self.store.identity.get('session_id'):
                    self.model_requests.append((row['event_id'], args.get('context_id')))
                # Keep only what names an object; commands and wire bodies stay in the store.
                self.events[row['event_id']] = (record['kind'], dict(
                    model_context_id=record.get('model_context_id'),
                    call=record.get('call'), request={k: args[k] for k in ('method', 'path', 'parsed', 'script', 'name')
                                                      if k in args}), self.day)
                # A note on a call covers what it ran, e.g. the scripts of a bash command.
                allowed_note = not self.store.identity.get('role') or record.get('role', record.get('author')) in READABLE_ROLES[self.store.identity['role']]
                if allowed_note and record['kind'] in NOTE_KINDS and isinstance(args.get('note'), str) and args['note'].strip():
                    self.notes[row['event_id']] = (self.day, args['note'].strip())
                elif allowed_note and record.get('parent_event_id') in self.notes and record['kind'] == 'cli_python':
                    self.notes[row['event_id']] = self.notes[record['parent_event_id']]
            rows = conn.execute('''SELECT v.rowid,v.version_id,v.event_id,v.content_hash,v.metadata,q.definition
                FROM versions v JOIN requests r USING(event_id) LEFT JOIN queries q ON r.query_id=q.id
                WHERE v.rowid>? ORDER BY v.rowid''', (self.version_rowid,)).fetchall()
        # A query's row comparison is written in the same transaction as its public response.
        comparisons = {row['event_id']: row['content_hash'] for row in rows
                       if row['version_id'].endswith(':comparison')}
        with self.lock:
            for row in rows:
                check()
                self.version_rowid = max(self.version_rowid, row['rowid'])
                if row['event_id'] in self.events:
                    meta = json.loads(row['metadata'])
                    if self.store.identity.get('role'):
                        try:
                            with closing(self.store.connect()) as conn:
                                self.store.authorize_version(conn, row['event_id'], meta, public=True)
                        except PermissionError:
                            continue
                    self._add(row, meta, comparisons)

    def _add(self, row, meta, comparisons):
        version, event_id = row['version_id'], row['event_id']
        kind, record, day = self.events[event_id]
        layer = meta['layer']
        if layer == 'executed_code' and kind == 'cli_python':
            self.code[event_id] = row['content_hash']
        if layer in ('file_text', 'query_model_projection'):
            # Projections are named by the version they project.
            if meta.get('derived_from') in self.info:
                self.info[version] = self.info[meta['derived_from']]
                self.note_sources[version] = self.note_sources.get(meta['derived_from'])
            return
        key = self._object(version, event_id, kind, record, layer, meta, row['definition'])
        if self.store.identity.get('role'):
            key = (*key, evidence_namespace(meta, event_id.split('/')[1]))
        signature = (comparisons.get(event_id) or row['content_hash'] if key[0] == 'query' else
                     meta.get('body_sha256') or row['content_hash'])
        if meta.get('workspace_publication'):
            signature = digest(encoded(meta['workspace_entry']))
        groups = self.groups[key]
        previous = groups[-1]['members'][-1] if groups and groups[-1]['signature'] == signature else None
        if groups and groups[-1]['signature'] == signature:
            group = groups[-1]
            group['members'].append(version)
        else:
            group = dict(key=key, k=self._version_number(key, version), signature=signature, members=[version], day=day)
            groups.append(group)
        self.info[version] = (key, group)
        self.bases[self.base(key)] = key
        # The note of the call that produced this content: its outputs and the files it wrote,
        # not versions it merely observed (a read, or the before-snapshot of a later command).
        produced = (layer != 'file_bytes' or
                    (meta.get('producer_role') is not None
                     if meta.get('workspace_namespace') else version.endswith('_after')))
        self.note_sources[version] = version if produced else self.note_sources.get(previous)
        if produced and event_id in self.notes:
            group.setdefault('note', self.notes[event_id])

    def _object(self, version, event_id, kind, record, layer, meta, definition):
        args = record.get('request') or {}
        if layer == 'file_bytes':
            return ('file', meta['object_id'])
        if layer == 'server_public_response':
            if definition:
                query = json.loads(definition)
                return ('query', encoded([query[1], *query[3:]]).decode())
            parsed = args.get('parsed') if isinstance(args.get('parsed'), dict) else {}
            if args.get('method') == 'GET' or parsed.get('tool') in READ_TOOLS:
                return ('read', encoded([args.get('method'), args.get('path'), args.get('parsed')]).decode())
            return ('receipt', version)
        if layer == 'dashboard':
            return ('dashboard',)
        if layer == 'registered_text':
            return ('record', version)
        if layer == 'registered_script':
            return ('weekly', meta['object_id'].split(':', 1)[-1])
        if layer in STREAMS:
            stream = STREAMS[layer]
            if kind == 'cli_python':
                if args.get('script'):
                    return ('script', args['script'], stream)
                code = self.code.get(event_id)
                if code:
                    return ('inline', code, stream)
            if kind == 'registered_script_execution' and layer == 'executed_code' and args.get('name'):
                return ('weekly_code', args['name'])
            if record.get('call') and stream != 'code':
                return ('cmd_stream', encoded(record['call']).decode(), stream)
        if layer == 'tool_return' and record.get('call'):
            return ('cmd' if kind == 'bash' else 'call', encoded(record['call']).decode())
        return ('node', layer, version)

    # --- names ----------------------------------------------------------------

    def _number(self, kind, key, allocate, owner=None):
        identity = (kind, owner or self.store.identity.get('role', 'ceo'), key)
        if identity not in self.ordinals:
            with allocation(self.store) as refs:
                self.ordinals[identity] = refs.ordinal('object:' + kind, identity[1], key)
        return self.ordinals[identity]

    def _version_number(self, key, version):
        owner = key[-1] if self.store.identity.get('role') else 'ceo'
        with allocation(self.store) as refs:
            return refs.ordinal('version:' + encoded(key).decode(), owner, version)

    def object_key(self, kind, name, owner=None):
        key = (kind, name)
        scope = owner or (self.store.identity.get('workspace_namespace') if kind == 'file' else None)
        return (*key, scope or self.store.identity['role']) if self.store.identity.get('role') else key

    def base(self, key, allocate=True):
        if self.store.identity.get('role'):
            return key[-1] + ':' + self._base(key[:-1], allocate, key[-1])
        return self._base(key, allocate)

    def _base(self, key, allocate=True, owner=None):
        """The object part of a handle, e.g. forecast.json, query7 or analyze.py.out."""
        kind = key[0]
        if kind == 'file':
            return key[1]
        if kind == 'dashboard':
            return 'dashboard'
        if kind == 'weekly':
            return 'weekly:' + key[1]
        if kind == 'weekly_code':
            return 'weekly:' + key[1] + '.code'
        if kind == 'script':
            return key[1] + '.' + key[2]
        if kind == 'inline':
            return f"inline{self._number('inline', key[1], allocate, owner)}.{key[2]}"
        if kind == 'cmd_stream':
            return f"cmd{self._number('cmd', key[1], allocate, owner)}.{key[2]}"
        if kind == 'node':
            return f"{key[1]}{self._number('node', key[2], allocate, owner)}"
        return f"{kind}{self._number(kind, key[1], allocate, owner)}"

    def name(self, version, allocate=True):
        with self.lock:
            if version not in self.info:
                self.refresh()
            if version not in self.info:
                raise KeyError(version)
            key, group = self.info[version]
            if key[0] == 'record':
                name = json.loads(self.store.get_content(version)[1])['version']
                return key[-1] + ':' + name if self.store.identity.get('role') else name
            if key[0] in ('receipt', 'node'):
                return self.base(key, allocate)  # A single immutable observation.
            return f"{self.base(key, allocate)}@v{group['k']}"

    def pending(self, kind, call, signature, version, allocate=True):
        """Handle the next return of this call will receive before it is stored, and the day
        its identical content was first returned (None if the content is new)."""
        with self.lock:
            self.refresh()
            key = self.object_key(kind, encoded(call).decode())
            groups = self.groups.get(key, [])
            same = bool(groups) and groups[-1]['signature'] == signature
            k = groups[-1]['k'] if same else self._version_number(key, version)
            return f'{self.base(key, allocate)}@v{k}', groups[-1]['day'] if same else None

    # --- facts used in agent-facing text --------------------------------------

    def group(self, version):
        with self.lock:
            if version not in self.info:
                self.refresh()
            return self.info.get(version, (None, None))[1]

    def key(self, version):
        with self.lock:
            if version not in self.info:
                self.refresh()
            return self.info.get(version, (None, None))[0]

    def repeated(self, version):
        """Day this version's content was first acquired, if this is a later acquisition."""
        group = self.group(version)
        if not group or group['members'][0] == version or group['key'][0] in ('record', 'receipt', 'node'):
            return None
        return group['day']

    def same_as(self, version):
        """An earlier version of the same object with identical content, e.g. after A→B→A."""
        group = self.group(version)
        if not group or group['key'][0] in ('record', 'receipt', 'node'):
            return None
        earlier = next((g for g in self.groups[group['key']][:self.groups[group['key']].index(group)]
                        if g['signature'] == group['signature']), None)
        return self.name(earlier['members'][0]) if earlier else None

    def note(self, version):
        self.refresh()
        source = self.note_sources.get(version)
        return self.notes.get(source.split(':', 1)[0]) if source else None

    def note_context(self, version):
        if not self.note(version):
            return None
        event = self.store.read_event(self.note_sources[version].split(':', 1)[0], evidence=True)
        request = event['request']
        return dict(role=request.get('role', request.get('author', 'ceo')),
                    day=self.note(version)[0], status=event['result'].get('status', 'result_unknown'))

    def latest(self, base):
        """Versions (oldest first) of an object's newest version, named without @vK, or None."""
        with self.lock:
            self.refresh()
            key = self._key_for(base)
            return list(self.groups[key][-1]['members']) if key else None

    def versions(self, version):
        """Every version group of this version's object, oldest first."""
        group = self.group(version)
        return list(self.groups[group['key']]) if group else []

    def previous(self, version):
        group = self.group(version)
        if not group or self.groups[group['key']].index(group) == 0 or group['key'][0] in ('record', 'receipt', 'node'):
            return None
        return self.name(self.groups[group['key']][self.groups[group['key']].index(group) - 1]['members'][-1])

    # --- resolution -----------------------------------------------------------

    def _key_for(self, base):
        role = self.store.identity.get('role')
        namespace = self.store.identity.get('workspace_namespace')
        if role and base.split(':', 1)[0] not in (*READABLE_ROLES, 'shared'):
            shared = self.bases.get(namespace + ':' + base) if namespace else None
            if shared is not None and shared[0] == 'file':
                return shared
            base = role + ':' + base
        return self.bases.get(base)

    def lookup(self, handle):
        with self.lock:
            self.refresh()
            if match := EXACT.fullmatch(handle):
                kind = match.group(2)
                raw = resolve_ref(self.store, kind, handle)
                with closing(self.store.connect()) as conn:
                    if kind == 'event':
                        self.store.readable_event(conn, raw)
                        rows = conn.execute('SELECT version_id,metadata FROM versions WHERE event_id=? ORDER BY rowid', (raw,))
                    else:
                        rows = conn.execute('SELECT version_id,metadata FROM versions WHERE version_id=?', (raw,))
                    found = []
                    for row in rows:
                        event = row['version_id'].split(':', 1)[0]
                        try:
                            self.store.authorize_version(conn, event, json.loads(row['metadata']), public=True)
                        except PermissionError:
                            if kind == 'capture':
                                raise
                            continue
                        if row['version_id'] in self.info:
                            found.append(row['version_id'])
                    return found or None
            if m := VERSIONED.fullmatch(handle):
                key = self._key_for(m.group(1))
                group = next((g for g in self.groups.get(key, []) if g['k'] == int(m.group(2))), None)
                return list(group['members']) if group else None
            return next(([version] for version in self.info
                         if self.info[version][0][0] in ('receipt', 'node') and self.name(version) == handle), None)
