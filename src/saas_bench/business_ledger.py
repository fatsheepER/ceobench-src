from contextlib import contextmanager
from copy import deepcopy
import json
import re
import uuid


OPERATION_COLUMNS = {
    'operation_ref': 'TEXT PRIMARY KEY. Stable role:number reference, e.g. growth:1. Numbers are never reused after rollback.',
    'operation_seq': 'INTEGER. Operation admission order on the retained route.',
    'actor': 'TEXT. Role bound by the host listener.',
    'initiator': 'TEXT. model, registered_script, or host; bound by the host.',
    'tool': 'TEXT. Authorized business tool or next_week.',
    'day': 'INTEGER. Simulation day at admission.',
    'parameters_json': 'TEXT. Complete public request parameters as JSON.',
    'outcome': 'TEXT. unfinished, succeeded, rejected, partial, or unknown. Unfinished does not imply zero effects or a live worker.',
    'result_json': 'TEXT. Complete tool result including per-item failures, or canonical next_week terminal outcome; NULL until confirmed.',
}
EFFECT_COLUMNS = {
    'effect_ref': 'TEXT PRIMARY KEY. Stable operation reference plus local effect index.',
    'operation_ref': 'TEXT. Operation that produced this effect.',
    'effect_seq': 'INTEGER. Actual committed transaction segment order on the retained route.',
    'item_index': 'INTEGER. Zero-based effect order inside this segment.',
    'day': 'INTEGER. Simulation day at the commit.',
    'kind': 'TEXT. Public row insert/update/delete, runtime_config, or advance_progress.',
    'target_json': 'TEXT. Public target identity as JSON.',
    'before_json': 'TEXT. Public values before this effect, or JSON null.',
    'after_json': 'TEXT. Public values after this effect, or JSON null.',
    'applied_parameters_json': 'TEXT. Normalized applied public values as JSON.',
    'result_json': 'TEXT. Effect-specific public result as JSON.',
}
TABLE_DOCS = {
    'business_operations': {'description': 'Authorized business calls shared by Git and PF. Query deliberately; default receipts remain concise. See docs/business-ledger.md.', 'columns': OPERATION_COLUMNS},
    'business_effects': {'description': 'Actual committed effects. Effects in one transaction share effect_seq. Advance effects are commit progress summaries, not simulator row audits.', 'columns': EFFECT_COLUMNS},
    'business_ledger_metadata': {'description': 'Ledger inception on this world. Older history is not reconstructed.', 'columns': {'singleton': 'INTEGER PRIMARY KEY. Always 1.', 'inception_day': 'INTEGER. Day the ledger schema was first installed.'}},
}

TOOL_TABLES = {
    **{name: ('config_history',) for name in ('set_prices', 'set_model_tiers', 'set_daily_spend', 'set_capacity_tier', 'set_usage_quotas')},
    'send_enterprise_deal': ('enterprise_turns', 'subscriptions'),
    'reject_enterprise_deal': ('enterprise_turns', 'subscriptions'),
    'post_social_media': ('agent_social_media_posts',),
    'research_market': ('ledger', 'group_info_levels', 'segment_discovery', 'notifications'),
    'research_group': ('ledger', 'pending_group_research'),
    'start_research_project': ('ledger', 'research_projects'),
}
QUEUE_COLUMNS = ('id', 'group_id', 'from_level', 'to_level', 'cost', 'started_day', 'expected_completion_day', 'status')


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def ensure_schema(conn, day):
    if conn.in_transaction:
        raise RuntimeError('Business ledger initialization requires a clean world connection')
    conn.execute('''CREATE TABLE IF NOT EXISTS business_operations (
        operation_ref TEXT PRIMARY KEY, operation_seq INTEGER UNIQUE NOT NULL,
        actor TEXT NOT NULL, initiator TEXT NOT NULL, tool TEXT NOT NULL, day INTEGER NOT NULL,
        parameters_json TEXT NOT NULL, outcome TEXT NOT NULL, result_json TEXT)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS business_effects (
        effect_ref TEXT PRIMARY KEY, operation_ref TEXT NOT NULL REFERENCES business_operations(operation_ref),
        effect_seq INTEGER NOT NULL, item_index INTEGER NOT NULL, day INTEGER NOT NULL,
        kind TEXT NOT NULL, target_json TEXT NOT NULL, before_json TEXT NOT NULL,
        after_json TEXT NOT NULL, applied_parameters_json TEXT NOT NULL, result_json TEXT NOT NULL,
        UNIQUE(effect_seq,item_index))''')
    conn.execute('CREATE INDEX IF NOT EXISTS business_effect_operation ON business_effects(operation_ref,effect_seq,item_index)')
    conn.execute('CREATE TABLE IF NOT EXISTS business_ledger_metadata(singleton INTEGER PRIMARY KEY CHECK(singleton=1), inception_day INTEGER NOT NULL)')
    conn.execute('INSERT OR IGNORE INTO business_ledger_metadata VALUES(1,?)', (day,))
    conn.commit()


def migrate_operation_refs(conn, public_refs):
    legacy = conn.execute("SELECT operation_ref,actor FROM business_operations ORDER BY operation_seq").fetchall()
    replacements = [(old, public_refs.assign('operation', actor, old)) for old, actor in legacy
                    if re.fullmatch(r'op:[0-9a-f]{32}', old)]
    if not replacements:
        return
    conn.execute('PRAGMA defer_foreign_keys=ON')
    try:
        for old, new in replacements:
            conn.execute('UPDATE business_operations SET operation_ref=? WHERE operation_ref=?', (new, old))
            rows = conn.execute('SELECT effect_ref FROM business_effects WHERE operation_ref=?', (old,)).fetchall()
            for (effect,) in rows:
                conn.execute('UPDATE business_effects SET operation_ref=?,effect_ref=? WHERE effect_ref=?',
                             (new, new + effect[len(old):], effect))
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def read_operation(conn, operation_ref):
    """Read only the public ledger projection; callers own the world lock."""
    cursor = conn.execute('SELECT ' + ','.join(OPERATION_COLUMNS) +
        ' FROM business_operations WHERE operation_ref=?', (operation_ref,))
    row = cursor.fetchone()
    if row is None:
        return None
    operation = dict(zip(OPERATION_COLUMNS, row))
    cursor = conn.execute('SELECT ' + ','.join(EFFECT_COLUMNS) +
        ' FROM business_effects WHERE operation_ref=? ORDER BY effect_seq,item_index', (operation_ref,))
    return dict(operation=operation, effects=[dict(zip(EFFECT_COLUMNS, row)) for row in cursor])


class BusinessOperation:
    def __init__(self, conn, *, actor, initiator, tool, day, parameters, public_refs):
        if conn.in_transaction:
            raise RuntimeError('Business operation requires a clean world connection')
        if initiator not in ('host', 'model', 'registered_script'):
            raise ValueError('Invalid host business initiator')
        self.conn, self.day, self.tool = conn, day, tool
        self.internal_ref = 'op:' + uuid.uuid4().hex
        self.ref = public_refs.assign('operation', actor, self.internal_ref)
        self.parameters = deepcopy(parameters)
        self.next_effect = 0
        self.commit_failed = False
        self.unknown = False
        conn.execute('''INSERT INTO business_operations
            VALUES(?,(SELECT COALESCE(MAX(operation_seq),0)+1 FROM business_operations),?,?,?,?,?,'unfinished',NULL)''',
            (self.ref, actor, initiator, tool, day, encode(parameters)))
        conn.commit()
        sequence = conn.execute('SELECT operation_seq FROM business_operations WHERE operation_ref=?',
                                (self.ref,)).fetchone()[0]
        self._receipt = dict(operation_ref=self.ref, operation_seq=sequence,
                             outcome='unfinished', effect_seqs=None)

    def receipt(self):
        # A timed-out caller must not reacquire the world lock held by late work.
        return deepcopy(self._receipt)

    @contextmanager
    def bind(self, tools, simulator=None, *, tables=None, advance=False, extra_owners=()):
        adapter = CommitConnection(self, simulator=simulator, advance=advance)
        owners = [owner for owner in (tools, simulator, *extra_owners) if owner is not None and hasattr(owner, "conn")]
        previous = [(owner, owner.conn) for owner in owners]
        try:
            adapter.install(TOOL_TABLES.get(self.tool, ()) if tables is None else tables)
            for owner in owners:
                owner.conn = adapter
            yield adapter
        finally:
            for owner, conn in previous:
                owner.conn = conn
            adapter.uninstall()

    def finish(self, result=None, *, unknown=False):
        if unknown:
            self.unknown = True
        elif self.unknown or self.commit_failed:
            raise RuntimeError('Business outcome is unknown; manual checkpoint resume required')
        if self.conn.in_transaction:
            if unknown:
                return
            raise RuntimeError('Business operation returned with an uncommitted transaction')
        effect_seqs = [row[0] for row in self.conn.execute(
            'SELECT DISTINCT effect_seq FROM business_effects WHERE operation_ref=? ORDER BY effect_seq', (self.ref,))]
        if unknown:
            outcome, public = 'unknown', {'success': False, 'error': 'Business operation outcome is unknown; manual checkpoint resume required'}
        else:
            public = (dict(success=result.success, message=result.message, data=result.data,
                           error=None if result.success else result.message)
                      if hasattr(result, 'success') else result)
            success = bool(public.get('success'))
            items = (public.get('data') or {}).get('results', []) if isinstance(public.get('data'), dict) else []
            partial = any(isinstance(item, dict) and (item.get('success') is False or bool(item.get('error'))) for item in items)
            outcome = 'partial' if effect_seqs and (not success or partial) else 'succeeded' if success else 'rejected'
        serialized = encode(public)
        previous = tuple(self.conn.execute(
            'SELECT outcome,result_json FROM business_operations WHERE operation_ref=?', (self.ref,)).fetchone())
        if previous[0] != 'unfinished':
            if previous[0] != outcome or json.loads(previous[1]) != json.loads(serialized):
                raise RuntimeError('Conflicting business operation finalization')
            return
        self.conn.execute('UPDATE business_operations SET outcome=?,result_json=? WHERE operation_ref=?',
                          (outcome, serialized, self.ref))
        self.conn.commit()
        self._receipt = dict(self._receipt, outcome=outcome, effect_seqs=effect_seqs)


class CommitConnection:
    def __init__(self, operation, *, simulator=None, advance=False):
        self.operation, self.raw = operation, operation.conn
        self.simulator, self.advance = simulator, advance
        self.triggers = []
        self.start_changes = self.raw.total_changes
        self.previous_progress = self.progress() if advance else None

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def progress(self):
        reached = getattr(self.simulator, 'current_day', self.operation.day)
        exists = self.raw.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='_rng_states'").fetchone()
        row = self.raw.execute("SELECT state_json FROM _rng_states WHERE name='all'").fetchone() if exists else None
        completed = json.loads(row[0]).get('_sim_state', {}).get('current_day') if row else None
        return dict(reached_day=reached, last_completed_day=completed)

    def install(self, tables):
        if self.raw.in_transaction:
            raise RuntimeError('Business capture requires a clean world connection')
        from .database import TABLE_DOCS as public_docs
        self.raw.execute('''CREATE TEMP TABLE IF NOT EXISTS _business_pending (
            position INTEGER PRIMARY KEY, kind TEXT, target_json TEXT, before_json TEXT,
            after_json TEXT, applied_parameters_json TEXT, result_json TEXT)''')
        self.raw.execute('DELETE FROM _business_pending')
        self.raw.commit()
        for table in tables:
            columns = list(QUEUE_COLUMNS if table == 'pending_group_research' else public_docs[table]['columns'])
            info = self.raw.execute(f'PRAGMA main.table_info("{table}")').fetchall()
            keys = [row[1] for row in sorted(info, key=lambda row: row[5]) if row[5]]
            if not keys or not set(columns).issubset({row[1] for row in info}):
                raise RuntimeError('Business projection schema mismatch')
            def obj(prefix, names):
                return 'json_object(' + ','.join(f"'{col}',{prefix}.\"{col}\"" for col in names) + ')'
            for action in ('INSERT', 'UPDATE', 'DELETE'):
                old = obj('OLD', columns) if action != 'INSERT' else "'null'"
                new = obj('NEW', columns) if action != 'DELETE' else "'null'"
                timing = 'AFTER'
                if table == 'config_history' and action == 'INSERT':
                    # REPLACE deletes the old row before AFTER INSERT can observe it.
                    timing = 'BEFORE'
                    old = f"COALESCE((SELECT {obj('prior', columns)} FROM main.\"{table}\" prior ORDER BY day DESC LIMIT 1),'null')"
                target = f"json_object('table','{table}','key',json({obj('OLD' if action == 'DELETE' else 'NEW', keys)}))"
                name = f'_business_{table}_{action.lower()}'
                event = 'UPDATE OF ' + ','.join(f'"{col}"' for col in columns) if action == 'UPDATE' else action
                self.raw.execute(f'''CREATE TEMP TRIGGER "{name}" {timing} {event} ON main."{table}"
                    BEGIN INSERT INTO _business_pending(kind,target_json,before_json,after_json,applied_parameters_json,result_json)
                    VALUES('{action.lower()}',{target},{old},{new},{new},'{{}}'); END''')
                self.triggers.append(name)
        self.start_changes = self.raw.total_changes

    def uninstall(self):
        # DDL leaves a pending business transaction intact on failure.
        for name in self.triggers:
            self.raw.execute(f'DROP TRIGGER IF EXISTS temp."{name}"')

    def effect(self, kind, target, before, after, applied=None, result=None):
        self.raw.execute('''INSERT INTO _business_pending(kind,target_json,before_json,after_json,applied_parameters_json,result_json)
            VALUES(?,?,?,?,?,?)''', (kind, encode(target), encode(before), encode(after),
                                    encode(after if applied is None else applied), encode(result or {})))

    def commit(self):
        try:
            return self._commit()
        except BaseException:
            self.operation.commit_failed = True
            raise

    def _commit(self):
        if not self.raw.in_transaction:
            return self.raw.commit()
        progress = self.progress() if self.advance else None
        if self.advance and self.raw.total_changes > self.start_changes:
            self.effect('advance_progress', {'scope': 'world_transaction'}, self.previous_progress,
                        progress, {}, {'scope': 'commit_progress_not_row_audit'})
        rows = self.raw.execute('SELECT kind,target_json,before_json,after_json,applied_parameters_json,result_json FROM _business_pending ORDER BY position').fetchall()
        segment = self.raw.execute('SELECT COALESCE(MAX(effect_seq),0)+1 FROM business_effects').fetchone()[0]
        day = getattr(self.simulator, 'current_day', self.operation.day) if self.advance else self.operation.day
        for index, row in enumerate(rows):
            self.raw.execute('INSERT INTO business_effects VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                (f'{self.operation.ref}/effect:{self.operation.next_effect + index}', self.operation.ref,
                 segment, index, day, *tuple(row)))
        self.raw.execute('DELETE FROM _business_pending')
        self.raw.commit()
        self.operation.next_effect += len(rows)
        self.start_changes = self.raw.total_changes
        self.previous_progress = progress

    def rollback(self):
        self.raw.rollback()
        self.start_changes = self.raw.total_changes

    def execute(self, sql, parameters=()):
        token = re.sub(r'/\*.*?\*/|--[^\n]*', '', sql, flags=re.S).strip().split(None, 1)[0].upper()
        if token in ('COMMIT', 'END'):
            self.commit()
            return self.raw.cursor()
        if token in ('RELEASE', 'SAVEPOINT'):
            raise RuntimeError('Business transactions require explicit commit; savepoints are not supported here')
        return self.raw.execute(sql, parameters)

    def cursor(self, *args, **kwargs):
        return CommitCursor(self, self.raw.cursor(*args, **kwargs))

    def executescript(self, script):
        raise RuntimeError('Business transactions cannot use implicit executescript commits')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        return False


class CommitCursor:
    def __init__(self, connection, cursor):
        self.connection, self.raw = connection, cursor

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def __iter__(self):
        return iter(self.raw)

    def execute(self, sql, parameters=()):
        self.raw = self.connection.execute(sql, parameters)
        return self

    def executescript(self, script):
        return self.connection.executescript(script)


def publish_runtime_config(tools, tool_name, setting_type, settings, replacements):
    from .database import record_config_override
    conn = tools.conn
    if conn.in_transaction:
        raise RuntimeError('Runtime configuration publication requires a clean transaction')
    prepared = deepcopy(replacements)
    before = {name: deepcopy(getattr(tools.config, name)) for name in prepared}
    try:
        record_config_override(conn, tools.current_day, tool_name, setting_type, settings)
        if isinstance(conn, CommitConnection):
            conn.effect('runtime_config', {'setting_type': setting_type}, before, prepared, settings)
        for name, value in prepared.items():
            setattr(tools.config, name, value)
        conn.commit()
    except BaseException:
        # SQLite retaining the transaction proves no successful commit occurred.
        # An exception after COMMIT cannot safely be called a no-change failure.
        if conn.in_transaction:
            for name, value in before.items():
                setattr(tools.config, name, value)
            conn.rollback()
        raise
