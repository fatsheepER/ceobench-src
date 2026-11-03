"""Finite public metadata projections; business arguments and bodies remain opaque."""
from copy import deepcopy
import shlex

from . import evidence_handles
from .public_errors import public_diagnostics


def fields(value, names):
    return {key: deepcopy(value[key]) for key in names if key in value}


class PublicEvidence:
    def __init__(self, store, registry=None):
        self.store = store
        self.registry = registry

    def ref(self, kind, value):
        return evidence_handles.public_ref(self.store, kind, value) if value else None

    def cutoff(self, value):
        result = fields(value, ('requests', 'versions', 'results', 'deliveries', 'client_calls',
                                'client_interruptions', 'captured_at'))
        if 'registries' in value:
            result['registries'] = {role: {name: {'revision_count': len(record['revisions'])}
                for name, record in files.items()} for role, files in value['registries'].items()}
        return result

    def scope(self, value):
        result = deepcopy(value)
        if result.get('event'):
            result['event'] = self.ref('event', result['event'])
        return result

    def arguments(self, value):
        if not isinstance(value, dict):
            return value
        return fields(value, ('command', 'path', 'content', 'old_text', 'new_text', 'note', 'method',
                              'parsed', 'sql', 'params', 'parameters', 'script', 'source', 'code', 'name',
                              'args', 'day', 'offset', 'limit', 'start_line', 'end_line'))

    def request(self, value):
        result = fields(value, ('kind', 'started_at'))
        result['arguments'] = self.arguments(value.get('arguments', value.get('request')))
        query = value.get('query')
        if query:
            result['query'] = dict(sql=deepcopy(query[5]), bound_parameters=deepcopy(query[6]))
        parent = value.get('parent_event_ref', value.get('parent_event_id'))
        if parent:
            result['parent_event_ref'] = self.ref('event', parent)
        return result

    @staticmethod
    def transport(value):
        allowed = ('send_state', 'receive_state', 'status', 'completed_at', 'received_at',
                   'http_status', 'interrupted_at')
        result = {}
        if 'server_delivery' in value:
            result['server_delivery'] = fields(value['server_delivery'], allowed) | public_diagnostics(value['server_delivery'])
        if 'client_receipts' in value:
            result['client_receipts'] = [dict(receipt=fields(row['receipt'], allowed) | public_diagnostics(row['receipt']),
                interruption=(fields(row['interruption'], allowed) | public_diagnostics(row['interruption'])) if row.get('interruption') else None)
                for row in value['client_receipts']]
        return result

    def receipt(self, value):
        result = fields(value, ('status', 'completion', 'completed_at', 'exit_code', 'classification',
            'capture_status', 'result_coverage', 'http_status',
            'changed_paths', 'written_paths', 'deleted_paths', 'boundary_closed', 'snapshot_status',
            )) | public_diagnostics(value)
        if 'transport' in value:
            result['transport'] = self.transport(value['transport'])
        return result

    def delivery(self, value):
        return {name: dict(status=group['status'], occurrences=[
            fields(item, ('day', 'source_range', 'full_source', 'representation')) |
            {'request_event': self.ref('event', item['request_event'])} for item in group['occurrences']])
            for name, group in value.items() if name in ('historical', 'current')}

    def delivery_scope(self, value):
        result = fields(value, ('reader_role', 'current_request_basis', 'historical', 'current'))
        result['current_request'] = self.ref('event', value.get('current_request'))
        return result

    def capture(self, value):
        result = fields(value, ('state', 'layer', 'extent', 'source_truncated', 'transformation',
                                'reading_scope')) | public_diagnostics(value)
        if value.get('derived_from'):
            result['derived_from'] = self.ref('capture', value['derived_from'])
        return result

    def candidate(self, item):
        result = fields(item, ('version', 'kind', 'name', 'day', 'status', 'initiator_role', 'initiator',
                              'producer_role', 'source_role', 'storage_role', 'association', 'sql_access',
                              'access_match_bases', 'note'))
        result.update(event_ref=self.ref('event', item['event_ref']),
                      content_ref=self.ref('capture', item['content_ref']),
                      capture=self.capture(item['capture']), model_delivery=self.delivery(item['model_delivery']))
        if 'request' in item:
            result['request'] = self.request(item['request'])
        if 'receipt' in item:
            result['receipt'] = self.receipt(item['receipt'])
        return result

    def directory(self, item):
        result = fields(item, ('table', 'column', 'query_events', 'outcomes', 'truncated_events',
                              'observation_bases', 'navigation_scope'))
        result['navigation'] = item['navigation']
        result['executions'] = []
        for entry in item['executions']:
            event, capture = self.ref('event', entry['event_ref']), self.ref('capture', entry['content_ref'])
            result['executions'].append(fields(entry, ('version', 'status', 'result_coverage')) |
                dict(event_ref=event, content_ref=capture,
                     details=shlex.join(['pf', 'candidates', '--history', '--role', 'all', '--event', event, '--detail']),
                     show=shlex.join(['pf', 'show', capture, '--full']) if capture else None))
        words = shlex.split(result['navigation'])
        if '--event' in words:
            offset = words.index('--event') + 1
            words[offset] = self.ref('event', words[offset])
            result['navigation'] = shlex.join(words)
        return result

    def endpoint(self, endpoint):
        if endpoint is None:
            return None
        from .public_records import public_relation_endpoint, public_text
        result = public_relation_endpoint(endpoint, self.registry.public_commit)
        result.update(fields(endpoint, ('availability', 'reason', 'text', 'event_kind')))
        if endpoint.get('record'):
            result['record'] = public_text(endpoint['record'], self.registry.public_commit)
        for key, kind in (('content_ref', 'capture'), ('event_ref', 'event'), ('parent_event_ref', 'event')):
            if endpoint.get(key):
                result[key] = self.ref(kind, endpoint[key])
        if endpoint['ref'].get('event_ref'):
            result['ref'] = {'event_ref': self.ref('event', endpoint['ref']['event_ref'])}
        if 'capture' in endpoint:
            result['capture'] = self.capture(endpoint['capture'])
        if 'request' in endpoint:
            result['request'] = self.request(endpoint['request'])
        if 'result' in endpoint:
            result['result'] = self.receipt(endpoint['result'])
        return result

    def relation(self, item):
        result = fields(item, ('assertion', 'origin', 'relation', 'relation_revision', 'revision',
            'author_role', 'record_status', 'current', 'source_index', 'reason', 'note', 'sim_day',
            'purpose', 'kind', 'basis', 'member_role', 'depth', 'historical_only', 'unexpanded'))
        result.update(source=self.endpoint(item['source']), target=self.endpoint(item['target']))
        if 'path' in item:
            result['path'] = []
            for endpoint in item['path']:
                if endpoint is None:
                    label = '(empty source set)'
                else:
                    ref = self.endpoint(endpoint)['ref']
                    label = ('missing: ' + ref['missing']['description'] if 'missing' in ref else
                             ref.get('cite') or ref.get('operation_ref') or ref.get('event_ref') or 'unknown')
                result['path'].append(label)
        if item.get('reference'):
            from .public_records import public_reference
            result['reference'] = public_reference(item['reference'], self.registry.public_commit, item.get('author_role'))
        if item.get('observation'):
            result['observation'] = fields(item['observation'], ('source_range', 'output_range', 'field'))
        for key, kind in (('event_ref', 'event'), ('root_event_ref', 'event'), ('source_event_ref', 'event'),
                          ('content_ref', 'capture'), ('source_content_ref', 'capture'), ('target_content_ref', 'capture')):
            if item.get(key):
                result[key] = self.ref(kind, item[key])
        if item.get('root_event_ref'):
            result['navigation'] = 'pf candidates --history --role all --event ' + result['root_event_ref']
        return result
