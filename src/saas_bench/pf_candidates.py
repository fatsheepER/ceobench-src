"""Frozen, event-preserving discovery of public captures, without building a graph."""
from collections import Counter, defaultdict
from contextlib import closing
import json
import shlex
import uuid

from .public_errors import PublicPFError, PublicPFPermissionError
from . import evidence_handles, pf_render, sql_access
from .execution_capture import CURRENT_EVENT
from .pf_deadline import check
from .sql_evidence import now
from .public_evidence import PublicEvidence


LAYERS = dict(file_bytes='file', file_text='file', server_public_response='api', stdout='stdout', stderr='stderr',
              tool_return='output', executed_code='code', registered_text='text',
              registered_script='script', dashboard='dashboard')
EVENT_KINDS = frozenset(('bash', 'cli_python', 'sql_query', 'public_http', 'read_file', 'write_file',
                        'edit_file', 'registered_script_execution', 'dashboard_generation'))
FORMAT = 'pf.candidates.v1'
DETAIL_CHARS = 18000


class CandidateQuery:
    def __init__(self, registry):
        self.registry, self.store, self.resolver = registry, registry.store, registry.resolver
        self.role = self.store.identity.get('role', 'ceo')
        self.public = PublicEvidence(self.store)
        self.cursors = self.store.load_state('pf_cursors') or {}

    def answer(self, values):
        if 'cursor' in values:
            saved = self.cursors.get(values['cursor'])
            if not saved or saved.get('operation') != 'pf_candidates':
                raise PublicPFError('Unknown candidate cursor; start a new pf candidates query')
            name = saved.get('candidate_snapshot')
            snapshot = self.store.load_state(name) if name else None
            if not snapshot or snapshot.get('format') != FORMAT:
                raise PublicPFError('Candidate snapshot is unavailable; start a new pf candidates query')
            offset, detail, selection = saved['offset'], saved['detail'], saved['selection']
        else:
            snapshot = self._snapshot(values)
            name = 'pf_candidates:' + uuid.uuid4().hex
            self.store.save_state(name, snapshot)
            offset, detail, selection = 0, values['detail'], [0, len(snapshot['items'])]
        self._authorize(snapshot)
        common = dict(scope=self.public.scope(snapshot['scope']), cutoff=self.public.cutoff(snapshot['cutoff']), total_in_scope=len(snapshot['items']),
                      detail=detail, model_delivery_scope=self.public.delivery_scope(snapshot['model_delivery_scope']))
        for field in ('view', 'sql_access_coverage'):
            if field in snapshot and snapshot[field] is not None:
                common[field] = snapshot[field]
        start, stop = selection
        if not 0 <= start <= stop <= len(snapshot['items']):
            raise PublicPFError('Invalid candidate cursor selection')
        if detail:
            selected = snapshot['items'][start:stop] if snapshot.get('view') == 'sql_accesses' else [
                dict(item, **snapshot['events'][item['event_ref']]) for item in snapshot['items'][start:stop]]
            selected = [self.public.directory(item) if snapshot.get('view') == 'sql_accesses' else self.public.candidate(item) for item in selected]
            content = json.dumps(selected, ensure_ascii=True, indent=2, sort_keys=True)
            end = min(len(content), offset + DETAIL_CHARS)
            if not 0 <= offset <= len(content):
                raise PublicPFError('Invalid candidate detail cursor offset')
            cursor = self._cursor(name, end, True, selection) if end < len(content) else None
            return dict(common, items=[], returned=stop - start, detail_text=content[offset:end],
                        range=[offset, end], total_chars=len(content), next_cursor=cursor, has_more=cursor is not None)
        if not start <= offset <= stop:
            raise PublicPFError('Invalid candidate cursor offset')
        end = min(stop, offset + 20)
        cursor = self._cursor(name, end, False, selection) if end < stop else None
        detail_cursor = self._cursor(name, 0, True, [offset, end]) if end > offset else None
        items = [self.public.directory(item) if snapshot.get('view') == 'sql_accesses' else self.public.candidate(item) for item in snapshot['items'][offset:end]]
        return dict(common, items=items, returned=len(items), next_cursor=cursor, has_more=cursor is not None,
                    detail_cursor=detail_cursor)

    def _cursor(self, snapshot, offset, detail, selection):
        value = dict(operation='pf_candidates', candidate_snapshot=snapshot, offset=offset,
                     detail=detail, selection=selection)
        for name, existing in self.cursors.items():
            if existing == value:
                return name
        name = 'c' + str(max([int(k[1:]) for k in self.cursors] or [0]) + 1)
        self.cursors[name] = value
        self.store.save_state('pf_cursors', self.cursors)
        return name

    def _event_allowed(self, conn, event):
        self.store.authorize_event_metadata(conn, event['event_id'])
        if not self.store.identity.get('role'):
            self.store._visible(conn, event['event_id'])
        else:
            from .role_policy import READABLE_ROLES
            if event.get('role', event.get('author')) not in READABLE_ROLES[self.role]:
                raise PublicPFPermissionError('Candidate event is inaccessible')

    def _authorize(self, snapshot):
        with closing(self.store.connect()) as conn:
            items = snapshot.get('sql_authorization_items', []) + (
                [] if snapshot.get('view') == 'sql_accesses' else snapshot['items'])
            for item in items:
                check()
                sql_access.authorize_projection(item.get('sql_access'))
                request = conn.execute('SELECT request FROM requests WHERE event_id=?', (item['event_ref'],)).fetchone()
                if request is None:
                    raise PublicPFError('Candidate snapshot event is unavailable')
                self._event_allowed(conn, json.loads(request[0]))
                if item['content_ref']:
                    row = conn.execute('SELECT event_id,content_hash,metadata FROM versions WHERE version_id=?',
                                       (item['content_ref'],)).fetchone()
                    if row is None or row['content_hash'] != item['sha256']:
                        raise PublicPFError('Candidate snapshot content is unavailable or changed')
                    self.store.authorize_version(conn, row['event_id'], json.loads(row['metadata']), public=True)
                else:
                    row = conn.execute('SELECT request FROM requests WHERE event_id=?', (item['event_ref'],)).fetchone()
                    if row is None:
                        raise PublicPFError('Candidate snapshot event is unavailable')
                    self._event_allowed(conn, json.loads(row[0]))

    def _snapshot(self, values):
        values = dict(values)
        if values.get('event'):
            values['event'] = evidence_handles.resolve_ref(self.store, 'event', values['event'])
        today = self.registry.sim_day()
        role = values.get('role', self.role)
        explicit_days = 'day_from' in values or 'day_to' in values
        if not values['history'] and not explicit_days and type(today) is not int:
            raise PublicPFError('Current simulated day is unknown; specify --history or a day range')
        start = values.get('day_from') if explicit_days or values['history'] else today // 7 * 7
        end = values.get('day_to') if explicit_days or values['history'] else start + 6
        scope = dict(role=role, role_basis='execution_initiator', day_from=start, day_to=end,
                     kind=values.get('kind') or ('sql' if values.get('table') or values.get('sql_accesses') else None),
                     table=values.get('table'), column=values.get('column'), text=values.get('text'),
                     event=values.get('event'), time='explicit_days' if explicit_days else
                     'all_history' if values['history'] else 'current_simulated_week')
        with closing(self.store.connect()) as conn:
            conn.execute('BEGIN')
            cutoff = {table: conn.execute('SELECT coalesce(max(rowid),0) FROM ' + table).fetchone()[0]
                      for table in ('requests', 'versions', 'results', 'deliveries', 'client_calls', 'client_interruptions')}
            cutoff['captured_at'] = now()
            events, days, last_days = {}, {}, {}
            for row in conn.execute('''SELECT r.rowid,r.event_id,r.request,s.record,q.definition,d.record AS delivery FROM requests r
                    LEFT JOIN results s USING(event_id) LEFT JOIN queries q ON r.query_id=q.id
                    LEFT JOIN deliveries d USING(event_id) ORDER BY r.rowid'''):
                check()
                request = json.loads(row['request'])
                result = json.loads(row['record']) if row['record'] else {}
                actor = request.get('role', request.get('author', 'ceo'))
                parent = request.get('parent_event_id')
                day = result.get('day')
                if request['kind'] in evidence_handles.HOST_DAY_KINDS:
                    day = (request.get('request') or {}).get('day', day)
                    if type(day) is int:
                        last_days[actor] = day
                if type(day) is not int:
                    day = days.get(parent, last_days.get(actor))
                days[row['event_id']] = day
                events[row['event_id']] = dict(request=request, result=result, row=row['rowid'], day=day,
                                              query=json.loads(row['definition']) if row['definition'] else None,
                                              transport=dict(server_delivery=json.loads(row['delivery']) if row['delivery'] else
                                                             {'send_state': 'unknown'}, client_receipts=[]), stream_gaps=[])
            for row in conn.execute('''SELECT c.event_id,c.received,i.record FROM client_calls c
                    LEFT JOIN client_interruptions i USING(token) ORDER BY c.rowid'''):
                events[row['event_id']]['transport']['client_receipts'].append(dict(
                    receipt=json.loads(row['received']) if row['received'] else {'receive_state': 'unknown'},
                    interruption=json.loads(row['record']) if row['record'] else None))
            excluded, event = set(), CURRENT_EVENT.get()
            while event in events and event not in excluded:
                excluded.add(event)
                event = events[event]['request'].get('parent_event_id')
            wanted = values.get('event')
            descendants = set()
            if wanted:
                if wanted not in events:
                    raise PublicPFError('Unknown candidate event')
                if events[wanted]['request']['kind'] not in EVENT_KINDS and not any(
                        json.loads(row[0])['layer'] in LAYERS for row in conn.execute(
                        'SELECT metadata FROM versions WHERE event_id=?', (wanted,))):
                    raise PublicPFPermissionError('Candidate event is inaccessible')
                try:
                    self.store.readable_event(conn, wanted)
                except PermissionError:
                    if events[wanted]['request']['kind'] not in EVENT_KINDS:
                        raise
                    self._event_allowed(conn, events[wanted]['request'])
                descendants.add(wanted)
                for event, record in events.items():
                    if record['request'].get('parent_event_id') in descendants:
                        descendants.add(event)
            deliveries, delivery_scope = self._deliveries(conn, events)
            versions = conn.execute('SELECT rowid,* FROM versions ORDER BY rowid').fetchall()
            by_version = {row['version_id']: row for row in versions}
            layers = defaultdict(set)
            for row in versions:
                layers[row['event_id']].add(json.loads(row['metadata'])['layer'])
            for event_id, available in layers.items():
                events[event_id]['stream_gaps'] = [stream + '_bytes_without_public_text' for stream in ('stdout', 'stderr')
                    if stream + '_bytes' in available and stream not in available and events[event_id]['request']['kind'] == 'cli_python']
            items, seen = [], set()
            for row in versions:
                check()
                event = events[row['event_id']]
                request, result = event['request'], event['result']
                meta = json.loads(row['metadata'])
                layer = meta['layer']
                if layer not in LAYERS or request['kind'].startswith('pf_') or row['event_id'] in excluded:
                    continue
                if result.get('pf_retrieval', bool(result.get('pf_calls'))) and layer in ('tool_return', 'stdout', 'stderr'):
                    continue
                source = by_version.get(meta.get('derived_from')) if layer == 'file_text' else None
                if layer == 'file_text':
                    if source is None:
                        raise PublicPFError('Captured file read source is unavailable')
                    if source['event_id'] == row['event_id']:
                        continue
                    meta['object_id'] = json.loads(source['metadata'])['object_id']
                if layer in ('file_bytes', 'file_text') and (meta.get('object_id', '').startswith('sessions/') or
                        '__pycache__' in meta.get('object_id', '').split('/') or meta.get('object_id') == '.tool-notes.jsonl'):
                    continue
                try:
                    self._event_allowed(conn, request)
                    self.store.authorize_version(conn, row['event_id'], meta, public=True)
                except (KeyError, PermissionError):
                    continue
                seen.add(row['event_id'])
                item = self._item(event, meta, row, deliveries, events, source=source)
                if self._matches(item, scope, descendants):
                    if scope['text'] and not self._text_match(item, scope['text'], conn):
                        continue
                    items.append((event['row'], row['rowid'], item))
            for event_id, event in events.items():
                request = event['request']
                if event_id in seen or event_id in excluded or request['kind'] not in EVENT_KINDS:
                    continue
                if event['result'].get('pf_retrieval', bool(event['result'].get('pf_calls'))):
                    continue
                try:
                    self._event_allowed(conn, request)
                except (KeyError, PermissionError):
                    continue
                item = self._item(event, {}, None, deliveries, events)
                if self._matches(item, scope, descendants) and (not scope['text'] or self._text_match(item, scope['text'], conn)):
                    items.append((event['row'], 0, item))
        items.sort(key=lambda value: (value[0], value[1]), reverse=True)
        candidates = [item for _, _, item in items]
        sql_items = {item['event_ref']: item for item in candidates if item['kind'] == 'sql'}
        access_counts = Counter(item['sql_access']['status'] for item in sql_items.values())
        coverage = dict(query_events=len(sql_items), unknown_events=access_counts['unknown'],
                        not_started_events=access_counts['not_started'], recorded_events=access_counts['recorded'],
                        recorded_empty_events=sum(item['sql_access']['status'] == 'recorded' and
                            not item['sql_access']['public_reads'] for item in sql_items.values()),
                        basis='role/day/kind/event/text scope before table/column matching; no match does not prove no access')
        authorization = [{key: item[key] for key in ('event_ref', 'content_ref', 'sha256', 'sql_access')}
                         for item in sql_items.values()]
        candidates = [item for item in candidates if not scope['table'] or
                      item['kind'] == 'sql' and sql_access.matches(item['sql_access'], scope['table'], scope['column'])]
        for item in candidates:
            if item['kind'] == 'sql':
                item['access_match_bases'] = sorted({row['basis'] for row in item['sql_access']['public_reads']
                    if (scope['table'] is None or row['table'] == scope['table']) and
                       (scope['column'] is None or row['column'] == scope['column'])})
        if values.get('sql_accesses'):
            candidates = self._directory(candidates, scope)
            details = {}
        else:
            details = {item['event_ref']: dict(request=item.pop('request'), receipt=item.pop('receipt')) for item in candidates}
        return dict(format=FORMAT, scope=scope, cutoff=cutoff, items=candidates, events=details,
                    model_delivery_scope=delivery_scope, view='sql_accesses' if values.get('sql_accesses') else 'candidates',
                    sql_access_coverage=coverage if sql_items or scope['kind'] == 'sql' else None,
                    sql_authorization_items=authorization)

    @staticmethod
    def _directory(candidates, scope):
        pairs = defaultdict(dict)
        for item in candidates:
            check()
            for read in item['sql_access']['public_reads']:
                if (scope['table'] is None or read['table'] == scope['table']) and (
                        scope['column'] is None or read['column'] == scope['column']):
                    key = read['table'], read['column']
                    event = pairs[key].setdefault(item['event_ref'], dict(item=item, bases=set()))
                    event['bases'].add(read['basis'])
        rows = []
        for (table, column), events in sorted(pairs.items(), key=lambda pair: (pair[0][0], pair[0][1] or '')):
            check()
            command = ['pf', 'candidates', '--role', scope['role'], '--kind', 'sql', '--table', table]
            if column is not None:
                command += ['--column', column]
            if scope['day_from'] is None and scope['day_to'] is None:
                command += ['--history']
            for key in ('day_from', 'day_to', 'text', 'event'):
                if scope[key] is not None:
                    command += ['--' + key.replace('_', '-'), str(scope[key])]
            items = [entry['item'] for entry in events.values()]
            rows.append(dict(table=table, column=column, query_events=len(events),
                outcomes=dict(Counter(item['status'] for item in items)),
                truncated_events=sum(item['receipt'].get('result_coverage') == 'truncated' for item in items),
                observation_bases=dict(Counter(basis for entry in events.values() for basis in entry['bases'])),
                navigation=shlex.join(command),
                navigation_scope='table_and_column' if column is not None else 'all_columns_for_table',
                executions=[dict(event_ref=item['event_ref'], content_ref=item['content_ref'], version=item['version'],
                                 status=item['status'], result_coverage=item['receipt'].get('result_coverage', 'unknown'),
                                 details=shlex.join(['pf', 'candidates', '--history', '--role', 'all', '--event', item['event_ref'], '--detail']),
                                 show=shlex.join(['pf', 'show', item['content_ref'], '--full']) if item['content_ref'] else None)
                            for item in items]))
        return rows

    @staticmethod
    def _matches(item, scope, descendants):
        return ((scope['role'] == 'all' or scope['role'] == item['initiator_role']) and
                (scope['day_from'] is None or item['day'] is not None and item['day'] >= scope['day_from']) and
                (scope['day_to'] is None or item['day'] is not None and item['day'] <= scope['day_to']) and
                (scope['kind'] is None or item['kind'] == scope['kind']) and
                (scope['event'] is None or item['event_ref'] in descendants))

    def _text_match(self, item, text, conn):
        if item.get('note') and text.casefold() in item['note'][1].casefold():
            return True
        if text.casefold() in item['name'].casefold() or (item['kind'] != 'file' and
                text.casefold() in json.dumps(item['request'], ensure_ascii=False).casefold()):
            return True
        if item['content_ref']:
            _, raw = self.store.get_content(item['content_ref'], connection=conn, public=True)
            try:
                return text.casefold() in raw.decode('utf-8').casefold()
            except UnicodeDecodeError:
                pass
        return False

    def _item(self, event, meta, row, deliveries, events, source=None):
        request, result = event['request'], event['result']
        event_id, layer = request['event_id'], meta.get('layer')
        actor = request.get('role', request.get('author', 'ceo'))
        owner = meta.get('owner_role', actor)
        namespace = meta.get('workspace_namespace') if layer in ('file_bytes', 'file_text') else None
        kind = 'sql' if request['kind'] == 'sql_query' and layer in (None, 'server_public_response') else LAYERS.get(layer, 'execution')
        if layer is None and request['kind'] == 'public_http':
            kind = 'api'
        version = row['version_id'] if row else None
        producer = actor
        if layer == 'file_bytes' and not (version.endswith('_after') and owner == actor):
            producer = None
        if layer == 'file_text':
            source_request = events[source['event_id']]['request']
            source_actor = source_request.get('role', source_request.get('author', 'ceo'))
            producer = source_actor if source['version_id'].endswith('_after') and owner == source_actor else None
        if namespace:
            producer = meta.get('producer_role')
            owner = producer
        gaps = set(result.get('capture_gaps', [])) | set(event['stream_gaps'])
        ancestor, visited = request.get('parent_event_id'), set()
        while ancestor in events and ancestor not in visited:
            visited.add(ancestor)
            parent = events[ancestor]
            gaps.update(parent['result'].get('capture_gaps', []))
            ancestor = parent['request'].get('parent_event_id')
        name = event['query'][5] if kind == 'sql' and event['query'] else (meta.get('object_id') or
            pf_render.label(layer or 'server_public_response', request['kind'], request.get('request') or {},
                            classification=result.get('classification')))
        detail_request = dict(kind=request['kind'], arguments=request.get('request'), query=event['query'],
                              parent_event_ref=request.get('parent_event_id'), started_at=request.get('started_at'))
        receipt = {k: result[k] for k in ('status', 'completion', 'completed_at', 'exit_code', 'classification',
                   'capture_status', 'capture_gaps', 'result_coverage', 'error', 'error_kind', 'http_status',
                   'changed_paths', 'written_paths', 'deleted_paths', 'boundary_closed', 'snapshot_status',
                   'failure_stage', 'error_type') if k in result}
        status = 'unfinished' if result.get('completion') == 'unfinished' else result.get('status', 'result_unknown')
        receipt.update(status=status, transport=event['transport'])
        item = dict(event_ref=event_id, content_ref=version, version=self.resolver.handle(version) if version else None,
                    kind=kind, name=name, day=event['day'], status=status, initiator_role=actor,
                    initiator='registered_script' if request['kind'] == 'registered_script_execution' or any(
                        events[parent]['request']['kind'] == 'registered_script_execution' for parent in visited) else
                        'model' if request.get('model_request_event') or any(
                        events[parent]['request'].get('model_request_event') for parent in visited) else 'unknown',
                    producer_role=producer, source_role=owner, storage_role=actor if version and not meta.get('workspace_namespace') else None,
                    **({key: meta[key] for key in ('workspace_namespace', 'collector_role') if key in meta} if namespace else {}),
                    sha256=row['content_hash'] if row else None,
                    capture=dict(state='available' if version else 'missing', layer=layer,
                                 extent=meta.get('extent', 'unknown'), source_truncated=meta.get('source_truncated'),
                                 derived_from=meta.get('derived_from'), transformation=meta.get('transformation'),
                                 gaps=sorted(gaps)),
                    model_delivery=deliveries.get(version, dict(historical=dict(status='unknown', occurrences=[]),
                                                               current=dict(status='unknown', occurrences=[]))),
                    association='execution_observation', request=detail_request, receipt=receipt)
        if version and (note := evidence_handles.index(self.store).note(version)):
            item['note'] = note
        if kind == 'sql':
            item['sql_access'] = sql_access.projection(result.get('sql_access'))
            item['access_match_bases'] = sorted({read['basis'] for read in item['sql_access']['public_reads']})
        return item

    def _deliveries(self, conn, events):
        eligible = [(event_id, event) for event_id, event in events.items()
                    if event['request']['kind'] == 'model_request' and
                    event['request'].get('role', event['request'].get('author', 'ceo')) == self.role and
                    event['result'].get('status') == 'succeeded' and event['result'].get('send_state') == 'response_received']
        current, active, current_basis = None, CURRENT_EVENT.get(), 'unknown'
        while active in events:
            request = events[active]['request']
            if request.get('model_request_event'):
                current = request['model_request_event']
                current_basis = 'originating_request'
                break
            active = request.get('parent_event_id')
        if current is None:
            current = next((event_id for event_id, event in reversed(eligible)
                            if event['request'].get('session_id') == self.store.identity.get('session_id')), None)
        if current and current_basis == 'unknown':
            current_basis = 'latest_successful_request_in_session'
        found = defaultdict(lambda: dict(historical=dict(status='unknown', occurrences=[]),
                                        current=dict(status='unknown', occurrences=[])))
        for event_id, event in eligible:
            check()
            for suffix in (':occurrences', ':reconstructed'):
                row = conn.execute('SELECT metadata FROM versions WHERE version_id=?', (event_id + suffix,)).fetchone()
                if row is None:
                    continue
                try:
                    _, raw = self.store.get_content(event_id + suffix, connection=conn, graph=True)
                except (KeyError, PermissionError):
                    continue
                for occurrence in json.loads(raw):
                    check()
                    version = occurrence['version_id']
                    row = conn.execute('SELECT metadata FROM versions WHERE version_id=?', (version,)).fetchone()
                    if row is None:
                        continue
                    meta = json.loads(row[0])
                    representation = meta['layer'] if meta['layer'] in ('file_text', 'query_model_projection') else None
                    sources = (version, meta['derived_from']) if representation else (version,)
                    item = dict(request_event=event_id, day=event['day'], source_range=occurrence['source_range'],
                                full_source=occurrence['full_source'], representation=representation)
                    for source in sources:
                        for group in ('historical', 'current') if event_id == current else ('historical',):
                            record = found[source][group]
                            if item not in record['occurrences']:
                                record['occurrences'].append(item)
                            if record['status'] != 'full':
                                record['status'] = 'full' if item['full_source'] and not representation else 'partial_or_representation'
        return dict(found), dict(reader_role=self.role, current_request=current, current_request_basis=current_basis,
                                 historical='successful requests of this reader role; private peer delivery is not queried',
                                 current='originating model request before this query; summary is not source delivery')


def render(page):
    scope = page['scope']
    lines = ['PF candidates: execution observations, not proof of use.',
             'Scope: ' + json.dumps(scope, ensure_ascii=True, sort_keys=True),
             'Snapshot cutoff: ' + json.dumps(page['cutoff'], sort_keys=True),
             f"returned={page['returned']}; total_in_scope={page['total_in_scope']}; has_more={str(page['has_more']).lower()}.",
             'Model delivery: this reader only; historical/current request before query; unknown is not unread.']
    if page.get('next_cursor'):
        lines.append('More from the same snapshot: pf more ' + page['next_cursor'])
    if page.get('detail_cursor'):
        lines.append('Exact events, capture gaps and ranges for this page: pf more ' + page['detail_cursor'])
    directory = page.get('view') == 'sql_accesses'
    if directory:
        lines[0] = 'PF SQL access directory: authorization observations, not proof of use.'
        lines.append('Counts are distinct query events per pair; observation basis counts overlap. Column-free reads name only a table.')
        lines.append('Navigation starts a fresh query in the same scope and may include later captures; these frozen counts do not change.')
    if 'sql_access_coverage' in page:
        lines.append('SQL access coverage: ' + json.dumps(page['sql_access_coverage'], sort_keys=True))
        lines.append(sql_access.SEMANTICS)
    if page['detail']:
        lines.append(f"Detail characters {page['range'][0]}:{page['range'][1]}/{page['total_chars']}; requests and receipt metadata, not source bodies.")
        lines.append(page['detail_text'])
    for item in page['items']:
        if directory:
            lines.append(f"{item['table']}.{item['column'] or '<column-free>'} | query_events={item['query_events']} | "
                         f"outcomes={json.dumps(item['outcomes'], sort_keys=True)} | truncated={item['truncated_events']} | "
                         f"bases={json.dumps(item['observation_bases'], sort_keys=True)}\n  {item['navigation']}")
            if item['navigation_scope'] == 'all_columns_for_table':
                lines.append('  Table-wide navigation includes other column observations; exact events are in details.')
            continue
        capture, delivery = item['capture'], item['model_delivery']
        extent = 'missing' if capture['state'] == 'missing' else 'truncated' if capture['source_truncated'] else capture['extent']
        lines.append(f"{pf_render.short(item['version'], 220) if item['version'] else 'No content handle'} | event={item['event_ref']} | day={item['day']} | {item['kind']} "
                     f"{pf_render.short(item['name'], 100)} | {item['status']} | capture={extent} | "
                     f"delivery history={delivery['historical']['status']}, current={delivery['current']['status']} | "
                     f"source={item['source_role']}, producer={item['producer_role']}, storage={item['storage_role']}")
        if 'sql_access' in item:
            lines.append('  authorization=' + item['sql_access']['status'] + '; match_bases=' + ','.join(item['access_match_bases']))
    if not page['items'] and not page['detail']:
        lines.append('No candidates in this scope. Use --history or --role all explicitly to broaden it.')
    lines.append('Read content: pf show HANDLE --full. Event details: pf candidates --history --event EVENT --detail.')
    return '\n'.join(lines)
