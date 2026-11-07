"""Business projections of private text and relation authority records."""
from copy import deepcopy
from pathlib import PurePosixPath
import re


TEXT_FIELDS = ('id', 'version', 'revision', 'status', 'text', 'objects', 'applies_at', 'reason', 'sim_day', 'author')
RELATION_FIELDS = ('id', 'revision', 'status', 'relation', 'relation_revision', 'author_role', 'sim_day', 'assertion', 'reason')
ENDPOINT_FIELDS = ('kind', 'source_state', 'readable', 'producer_role', 'source_role', 'storage_role',
                   'workspace_namespace', 'collector_role')


def fields(record, names):
    return {name: deepcopy(record[name]) for name in names if name in record}


def private_framework_path(path):
    parts = PurePosixPath(str(path)).parts
    if not parts:
        return False
    return parts[0] in ('registrations.json', 'relations.json', '.tool-notes.jsonl', '.daily_jobs.json') or parts[0].startswith(('.registration-', '.relation-'))


def public_evidence_reference(evidence, commit_ref, owner=None):
    result = fields(evidence, ('path', 'commit', 'owner', 'record', 'unknown', 'version'))
    commit = result.get('commit')
    if isinstance(commit, str) and re.fullmatch('[0-9a-fA-F]{40}', commit):
        short = commit_ref(result.get('owner', owner), commit)
        if short is None:
            result.pop('commit')
            result['availability'] = 'original_commit_unavailable'
        else:
            result['commit'] = short
    return result


def public_reference(reference, commit_ref, owner=None):
    result = fields(reference, ('purpose', 'note', 'select', 'predicate'))
    for name in ('evidence', 'snapshot_source'):
        if reference.get(name):
            result[name] = public_evidence_reference(reference[name], commit_ref, owner)
    return result


def public_text(record, commit_ref):
    result = fields(record, TEXT_FIELDS)
    result['references'] = [public_reference(reference, commit_ref, record.get('author'))
                            for reference in record.get('references', [])]
    return result


def public_text_state(state, commit_ref):
    return dict(format=state['format'], records={key: [public_text(record, commit_ref) for record in history]
        for key, history in state['records'].items()})


def _reference(reference, commit_ref, owner=None):
    result = fields(reference, ('cite', 'operation_ref'))
    if 'missing' in reference:
        result['missing'] = fields(reference['missing'], ('description', 'reason', 'last_known_ref'))
    if match := re.fullmatch(r'(?:(ceo|growth|ops_finance|shared):)?(.+)@([0-9a-fA-F]{40})', result.get('cite', '')):
        role, path, commit = match.groups()
        short = commit_ref(role or owner, commit)
        if short is None:
            result = dict(missing=dict(description='Original committed source is unavailable',
                reason='The source repository no longer contains the saved commit.',
                last_known_ref=(role + ':' if role else '') + path))
        else:
            result['cite'] = (role + ':' if role else '') + path + '@' + short
    return result


def public_relation_endpoint(endpoint, commit_ref):
    result = fields(endpoint, ENDPOINT_FIELDS)
    result['ref'] = _reference(endpoint.get('ref', {}), commit_ref, endpoint.get('workspace_namespace') or endpoint.get('storage_role'))
    if endpoint.get('snapshot_source'):
        source = endpoint['snapshot_source']
        result['snapshot_source'] = (public_evidence_reference(source, commit_ref, endpoint.get('source_role'))
            if 'path' in source else _reference(source, commit_ref, endpoint.get('source_role')))
    if 'missing' in result['ref']:
        result.update(source_state='missing', readable=False)
    if 'capture' in endpoint:
        result['capture'] = fields(endpoint['capture'], ('reading_scope', 'source_truncated', 'layer'))
    if 'operation' in endpoint:
        from .business_ledger import OPERATION_COLUMNS, EFFECT_COLUMNS
        operation = endpoint['operation']
        result['operation'] = dict(operation=fields(operation['operation'], OPERATION_COLUMNS),
                                   effects=[fields(item, EFFECT_COLUMNS) for item in operation.get('effects', [])])
    return result


def public_relation(record, commit_ref):
    result = fields(record, RELATION_FIELDS)
    result['target'] = public_relation_endpoint(record['target'], commit_ref)
    result['sources'] = [dict(ref=public_relation_endpoint(source['ref'], commit_ref), **fields(source, ('note',)))
                         for source in record['sources']]
    return result


def public_relation_state(state, commit_ref):
    return dict(format=state['format'], records={key: [public_relation(record, commit_ref) for record in history]
        for key, history in state['records'].items()})
