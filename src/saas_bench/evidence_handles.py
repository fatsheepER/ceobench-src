"""Agent-facing evidence handles: an object name and that object's version number.

A handle names what a version belongs to and which version of it this is, e.g.
forecast.json@v3, query7@v2 (with its SQL shown alongside), analyze.py.out@v4 (the
stdout of that script) or cmd5@v1 (a command's return). The version number rises only
when the object's content differs from its previous version; every execution is still
recorded, and repeated acquisitions of the same content share one handle.

Version numbers follow the chronological order of the versions visible in this branch,
so they are deterministic and never change once assigned. Objects without a natural
name (queries, reads, commands, receipts) are numbered in the order first shown; that
table lives in private state and is copied with the evidence database at a fork.
"""
from collections import defaultdict
from contextlib import closing
import json
import re
import threading

from .sql_evidence import digest, encoded


# Private state name of the per-kind object number table, e.g. {"query": {key: 7}}.
HANDLES = 'registration_handles'
# Harness-generated reads; mirrors execution_capture.READ_TOOLS (kept here to avoid a cycle).
READ_TOOLS = frozenset({'get_social_posts', 'get_cost_info', 'list_research_projects',
                        'get_market_overview', 'get_group_insights'})
STREAMS = {'stdout': 'out', 'stderr': 'err', 'executed_code': 'code'}
# Calls that may carry an agent note (design 3.2); mirrors bash_agent.tools.NOTE_TOOLS.
NOTE_KINDS = frozenset({'bash', 'write_file', 'edit_file'})
VERSIONED = re.compile(r'(.+)@v([1-9][0-9]*)')
SINGLE = re.compile(r'([a-z_]+?)([1-9][0-9]*)')
# Accepted wherever the agent may write a handle; registered texts keep rN / rN.M.
HANDLE_PATTERN = r'^(.+@v[1-9][0-9]*|[a-z_]+[1-9][0-9]*|r[1-9][0-9]*(\.[1-9][0-9]*)?)$'
RECORD = re.compile(r'r[1-9][0-9]*(\.[1-9][0-9]*)?')


def body_digest(text):
    """Signature of a tool return before PF appended its handles."""
    return digest(str(text).encode('utf-8', 'surrogatepass'))


def one_line(sql):
    return ' '.join(str(sql).split())


def index(store):
    """The handle index shared by every reader of this store instance."""
    value = getattr(store, '_handle_index', None)
    if value is None:
        value = store._handle_index = HandleIndex(store)
    return value


class HandleIndex:
    def __init__(self, store):
        self.store = store
        self.lock = threading.RLock()
        self.version_rowid = self.request_rowid = 0
        self.day = 0
        self.events = {}                    # visible event -> (kind, request record, day)
        self.model_requests = []            # visible model sends, in acquisition order
        self.notes = {}                     # visible event -> (day, note) the agent attached to that call
        self.info = {}                      # version -> (object key, group)
        self.groups = defaultdict(list)     # object key -> [group], oldest first
        self.code = {}                      # cli_python event -> executed code hash
        self.chain = None

    # --- incremental build ----------------------------------------------------

    def _ancestry(self, conn):
        chain, branch, limit = {}, self.store.identity['branch_id'], None
        while branch and branch not in chain:
            chain[branch] = limit
            row = conn.execute('SELECT parent,fork_seq FROM branches WHERE id=?', (branch,)).fetchone()
            branch, limit = (row[0], row[1]) if row else (None, None)
        return chain

    def refresh(self):
        with self.lock, closing(self.store.connect()) as conn:
            if self.chain is None:
                self.chain = self._ancestry(conn)
            for row in conn.execute('SELECT rowid,event_id,branch,seq,request FROM requests WHERE rowid>? '
                                    'ORDER BY rowid', (self.request_rowid,)):
                self.request_rowid = row['rowid']
                if row['branch'] not in self.chain or (self.chain[row['branch']] is not None
                                                       and row['seq'] > self.chain[row['branch']]):
                    continue
                record = json.loads(row['request'])
                if record['kind'] == 'dashboard_generation':
                    # Later events happen on the simulated day of the latest weekly dashboard.
                    self.day = (record.get('request') or {}).get('day', self.day)
                args = record.get('request') if isinstance(record.get('request'), dict) else {}
                if record['kind'] == 'model_request':
                    self.model_requests.append((row['event_id'], args.get('context_id')))
                # Keep only what names an object; commands and wire bodies stay in the store.
                self.events[row['event_id']] = (record['kind'], dict(
                    model_context_id=record.get('model_context_id'),
                    call=record.get('call'), request={k: args[k] for k in ('method', 'path', 'parsed', 'script', 'name')
                                                      if k in args}), self.day)
                # A note on a call covers what it ran, e.g. the scripts of a bash command.
                if record['kind'] in NOTE_KINDS and isinstance(args.get('note'), str) and args['note'].strip():
                    self.notes[row['event_id']] = (self.day, args['note'].strip())
                elif record.get('parent_event_id') in self.notes and record['kind'] == 'cli_python':
                    self.notes[row['event_id']] = self.notes[record['parent_event_id']]
            rows = conn.execute('''SELECT v.rowid,v.version_id,v.event_id,v.content_hash,v.metadata,q.definition
                FROM versions v JOIN requests r USING(event_id) LEFT JOIN queries q ON r.query_id=q.id
                WHERE v.rowid>? ORDER BY v.rowid''', (self.version_rowid,)).fetchall()
        # A query's row comparison is written in the same transaction as its public response.
        comparisons = {row['event_id']: row['content_hash'] for row in rows
                       if row['version_id'].endswith(':comparison')}
        with self.lock:
            for row in rows:
                self.version_rowid = max(self.version_rowid, row['rowid'])
                if row['event_id'] in self.events:
                    self._add(row, json.loads(row['metadata']), comparisons)

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
            return
        key = self._object(version, event_id, kind, record, layer, meta, row['definition'])
        signature = (comparisons.get(event_id) or row['content_hash'] if key[0] == 'query' else
                     meta.get('body_sha256') or row['content_hash'])
        groups = self.groups[key]
        if groups and groups[-1]['signature'] == signature:
            group = groups[-1]
            group['members'].append(version)
        else:
            group = dict(key=key, k=len(groups) + 1, signature=signature, members=[version], day=day)
            groups.append(group)
        self.info[version] = (key, group)
        # The note of the call that produced this content: its outputs and the files it wrote,
        # not versions it merely observed (a read, or the before-snapshot of a later command).
        produced = layer != 'file_bytes' or version.endswith('_after')
        if produced and event_id in self.notes:
            group['note'] = self.notes[event_id]

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

    def _numbers(self):
        return self.store.load_state(HANDLES) or {}

    def _number(self, kind, key, allocate):
        table = self._numbers()
        numbers = table.setdefault(kind, {})
        if key not in numbers:
            if not allocate:
                return len(numbers) + 1
            numbers[key] = len(numbers) + 1
            self.store.save_state(HANDLES, table)
        return numbers[key]

    def base(self, key, allocate=True):
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
            return f"inline{self._number('inline', key[1], allocate)}.{key[2]}"
        if kind == 'cmd_stream':
            return f"cmd{self._number('cmd', key[1], allocate)}.{key[2]}"
        if kind == 'node':
            return f"{key[1]}{self._number('node', key[2], allocate)}"
        return f"{kind}{self._number(kind, key[1], allocate)}"

    def name(self, version, allocate=True):
        with self.lock:
            if version not in self.info:
                self.refresh()
            if version not in self.info:
                raise KeyError(version)
            key, group = self.info[version]
            if key[0] == 'record':
                return json.loads(self.store.get_content(version)[1])['version']
            if key[0] in ('receipt', 'node'):
                return self.base(key, allocate)  # A single immutable observation.
            return f"{self.base(key, allocate)}@v{group['k']}"

    def pending(self, kind, call, signature, allocate=True):
        """Handle the next return of this call will receive before it is stored, and the day
        its identical content was first returned (None if the content is new)."""
        with self.lock:
            self.refresh()
            key = (kind, encoded(call).decode())
            groups = self.groups.get(key, [])
            same = bool(groups) and groups[-1]['signature'] == signature
            k = groups[-1]['k'] if same else len(groups) + 1
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
        earlier = next((g for g in self.groups[group['key']][:group['k'] - 1]
                        if g['signature'] == group['signature']), None)
        return self.name(earlier['members'][0]) if earlier else None

    def note(self, version):
        """(day, text) of the latest note on a call that produced this version's content."""
        group = self.group(version)
        return group.get('note') if group else None

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
        if not group or group['k'] == 1 or group['key'][0] in ('record', 'receipt', 'node'):
            return None
        return self.name(self.groups[group['key']][group['k'] - 2]['members'][-1])

    # --- resolution -----------------------------------------------------------

    def _key_for(self, base):
        numbers = self._numbers()
        def numbered(kind, n, *rest):
            key = next((k for k, v in numbers.get(kind, {}).items() if v == int(n)), None)
            return (kind if not rest else rest[0], key, *rest[1:]) if key is not None else None
        candidates = []
        if m := re.fullmatch(r'(query|read|cmd|call)([1-9][0-9]*)', base):
            candidates.append(numbered(m.group(1), m.group(2)))
        if m := re.fullmatch(r'cmd([1-9][0-9]*)\.(out|err)', base):
            candidates.append(numbered('cmd', m.group(1), 'cmd_stream', m.group(2)))
        if m := re.fullmatch(r'inline([1-9][0-9]*)\.(out|err|code)', base):
            candidates.append(numbered('inline', m.group(1), 'inline', m.group(2)))
        if base == 'dashboard':
            candidates.append(('dashboard',))
        if base.startswith('weekly:'):
            name = base[len('weekly:'):]
            if name.endswith('.code'):
                candidates.append(('weekly_code', name[:-len('.code')]))
            candidates.append(('weekly', name))
        if m := re.fullmatch(r'(.+)\.(out|err|code)', base):
            candidates.append(('script', m.group(1), m.group(2)))
        candidates.append(('file', base))
        return next((k for k in candidates if k and k in self.groups), None)

    def lookup(self, handle):
        """Versions (oldest first) that a handle denotes, or None if it denotes nothing."""
        with self.lock:
            self.refresh()
            if m := VERSIONED.fullmatch(handle):
                key = self._key_for(m.group(1))
                groups = self.groups.get(key, []) if key else []
                k = int(m.group(2))
                return list(groups[k - 1]['members']) if k <= len(groups) else None
            if m := SINGLE.fullmatch(handle):
                numbers = self._numbers()
                for kind in ('receipt', 'node'):
                    version = next((k for k, v in numbers.get(kind, {}).items() if v == int(m.group(2))), None)
                    if version in self.info and self.name(version, allocate=False) == handle:
                        return [version]
            return None
