"""PF read delivery at the final request boundary, with replay-based accounting."""
from collections import Counter
from contextlib import closing
import difflib
import hashlib
from itertools import accumulate
import json
import shlex

from .execution_capture import CapturedText, CURRENT_EVENT, at_pointer, origin, slice_origins, text_sources
from . import evidence_handles
from .sql_evidence import encoded


def capture_read(query, target, text, start, end, total):
    store = query.store
    event = CURRENT_EVENT.get()
    own_event = event is None
    if own_event:
        event = store.begin_event('pf_read', query.values)
    mode = query.values['mode']
    header, body = str(text).split('\n', 1)
    header = dict(json.loads(header), delivery='FULL' if mode == 'content' else 'DIFF')
    prefix = encoded(header).decode() + '\n'
    sources = [origin(target, query.resolver.content(target)[1].decode(), start, end, len(prefix))] if mode == 'content' else []
    meta = query.nodes[target]['meta']
    read = store.version(event, 'read_full', prefix + body, layer='pf_read_full',
        target=target, key=list(query._key(target)), mode=mode, read_range=[start, end],
        complete=mode == 'content' and start == 0 and end == total and
                 not meta['source_truncated'] and meta['extent'] == 'full',
        force_full=query.values['full'])
    if own_event:
        store.complete(event)
    return CapturedText(prefix + body, sources, {'id': read})


def capture_view(store, text, view):
    """Save exactly the piped response and only the source ranges it still contains."""
    from .pf_cli import apply_view
    shown = apply_view(text, view)
    if shown == text:
        return text
    start = 0 if view[0] == 'head' else len(text) - len(shown)
    end = start + len(shown)
    sources = slice_origins(text.origins, start, end)
    meta, _ = store.get_content(text.pf_read['id'])
    a, b = meta['read_range']
    prefix = str(text).index('\n') + 1
    clip = lambda pos: a + max(0, min(b - a, pos - prefix))
    read = store.version(meta['created_by_event'], 'read_view', shown, layer='pf_read_full',
        target=meta['target'], key=meta['key'], mode=meta['mode'],
        read_range=[clip(start), clip(end)], complete=False, force_full=True, view=list(view))
    return CapturedText(shown, sources, {'id': read})


def make_delta(before, after):
    old, new = before.splitlines(keepends=True), after.splitlines(keepends=True)
    offsets = [0, *accumulate(map(len, old))]
    # ponytail: line replacements; use a character diff only if single-line
    # evidence produces enough full fallbacks to justify its additional cost.
    return [[offsets[a], offsets[b], ''.join(new[c:d])]
            for tag, a, b, c, d in difflib.SequenceMatcher(None, old, new).get_opcodes()
            if tag != 'equal']


def apply_delta(before, edits):
    parts, end = [], 0
    for start, stop, replacement in edits:
        if (type(start) is not int or type(stop) is not int or not isinstance(replacement, str)
                or not end <= start <= stop <= len(before)):
            raise ValueError('Invalid delta replacement range')
        parts.extend((before[end:start], replacement))
        end = stop
    return ''.join([*parts, before[end:]])


def _compact(full, mode, handle, edits):
    header = json.loads(full.split('\n', 1)[0])
    header.update(delivery=mode, base=handle)
    header['full_command'] = 'pf show ' + shlex.quote(header['target']['version']) + ' --full'
    if mode == 'DELTA':
        header['patch_format'] = 'unicode-replacements-v1'
    return encoded(header).decode() + '\n' + (encoded(edits).decode() if mode == 'DELTA' else '')


def _compact_tool(mode, target_handle, edits, base):
    """Compact form of a repeated tool read; the base is its previous result in this request."""
    header = dict(delivery=mode, base=base, target=target_handle)
    header['full_command'] = 'pf show ' + shlex.quote(target_handle) + ' --full'
    if mode == 'DELTA':
        header['patch_format'] = 'unicode-replacements-v1'
    return encoded(header).decode() + '\n' + (encoded(edits).decode() if mode == 'DELTA' else '')


def _payload(meta, full, mode, delivery):
    if meta.get('read_kind') == 'tool_call':
        return _compact_tool(mode, delivery['target_handle'], delivery['edits'], delivery['base_handle'])
    return _compact(full, mode, delivery['base_handle'], delivery['edits'])


_COUNTS = {}


def _count(counter, text):
    if counter is None:
        return None
    # Every tool return in the context is counted per request; cache successful counts.
    key = (id(counter), json.dumps(counter.metadata, sort_keys=True, default=str),
           hashlib.sha256(text.encode('utf-8', 'surrogatepass')).hexdigest())
    if key in _COUNTS:
        return _COUNTS[key]
    try:
        value = counter.count(text)
    except Exception:
        return None
    if type(value) is not int or value < 0:
        return None
    _COUNTS[key] = value
    return value


def _literal_bases(store, source, available):
    from .pf_queries import PUBLIC_LAYERS, evidence_key
    for item in source['origins']:
        if not item['full_source']:
            continue
        version = item['version_id']
        meta, raw = store.get_content(version)
        if meta['layer'] not in PUBLIC_LAYERS or meta['source_truncated'] or meta['extent'] != 'full':
            continue
        text = raw.decode('utf-8')
        a, b = item['request_range']
        if item['source_range'] != [0, len(text)] or source['text'][a:b] != text:
            continue
        if meta['layer'] == 'file_text':
            version = meta['derived_from']
            meta, original = store.get_content(version)
            if original != raw:
                continue  # Universal-newline projections cannot recover original bytes.
        event = store.read_event(meta['created_by_event'])
        key = evidence_key(version, meta, event['query_definition'], event['result'], event['request'].get('request'),
                           event['request'])
        available[version] = dict(text=text, key=list(key),
            chain=[dict(version=version, pointer=source['pointer'], range=[a, b])])


def _recent(store, context_id):
    with closing(store.connect()) as conn:
        rows = conn.execute('''SELECT r.event_id,r.seq FROM requests r JOIN results s USING(event_id)
            WHERE r.branch=? AND json_extract(r.request,'$.kind')='model_request'
              AND json_extract(r.request,'$.request.context_id')=?
              AND json_extract(s.record,'$.status')='succeeded'
              AND json_extract(s.record,'$.send_state')='response_received'
            ORDER BY r.seq DESC LIMIT 2''', (store.identity['branch_id'], context_id)).fetchall()
    if not rows:
        return [], 0, 0
    try:
        reads = json.loads(store.get_content(rows[0]['event_id'] + ':pf_reads')[1])
    except KeyError:
        reads = []
    return reads, rows[0]['seq'], rows[1]['seq'] if len(rows) > 1 else 0


def _choose(store, read_id, meta, full, available, counter, context, recent):
    full_tokens = _count(counter, full)
    choice = dict(mode='FULL', reason='no_complete_base', payload=full, base=None, edits=[],
                  candidate_tokens=dict(FULL=full_tokens, DELTA=None, UNCHANGED=None),
                  tokenizer=counter.metadata if counter else None, recovery_of=None)
    if meta['mode'] == 'diff':
        return dict(choice, mode='DIFF', reason='active_diff')
    previous, last_seq, prior_seq = recent
    spent_key = 'pf_read_recovered:' + store.identity['branch_id'] + ':' + context
    spent = store.load_state(spent_key) or []
    tool = meta.get('read_kind') == 'tool_call'
    # A tool call is identified by tool and arguments; a PF read by target and range.
    mark = json.dumps(meta['key']) if tool else meta['target']
    target_sha = store.get_content(meta['target'])[0]['blob_sha256'] if tool else None
    for item in reversed(previous):
        # A repeated tool call only recovers the compact result if it returns the same text;
        # a rerun whose output changed is an ordinary new read.
        same = (item.get('key') == meta['key'] and store.get_content(item['target'])[0]['blob_sha256'] == target_sha
                if tool else item['target'] == meta['target'] and item['range'] == meta['read_range'])
        if item['mode'] not in ('DELTA', 'UNCHANGED') or not same or item['read_id'] == read_id:
            continue
        old_meta, _ = store.get_content(item['read_id'])
        old = store.read_event(old_meta['created_by_event'])['request']
        current = store.read_event(meta['created_by_event'])['request']
        adjacent = old['event_id'].split('/')[1] == store.identity['branch_id']
        adjacent = adjacent and prior_seq < old['seq'] < last_seq < current['seq']
        # Only a compact read delivered in the immediately preceding request is recovered;
        # an explicit full read elsewhere is an ordinary FULL read without pairing.
        if adjacent and (meta['force_full'] or mark not in spent):
            if item['read_id'] not in spent:
                store.save_state(spent_key, [*spent, item['read_id'], mark])
                return dict(choice, reason='requested_full' if meta['force_full'] else 'adjacent_recovery',
                            recovery_of=item['read_id'])
    if meta['force_full']:
        return dict(choice, reason='requested_full')
    if not meta['complete']:
        return dict(choice, reason='partial_or_truncated')
    if full_tokens is None:
        return dict(choice, reason='tokenizer_unavailable' if counter is None else 'token_count_failed')
    candidates = [(v, data) for v, data in available.items() if data['key'] == meta['key']]
    if not candidates:
        return choice
    base, data = candidates[-1]
    target = store.get_content(meta['target'])[1].decode('utf-8')
    try:
        edits = make_delta(data['text'], target)
        valid = apply_delta(data['text'], edits) == target
    except (ValueError, TypeError):
        valid = False
    if not valid:
        return dict(choice, reason='delta_verification_failed')
    # Tool-call reads name their own target (for pf_read full text); PF reads name the base.
    # Allocate names after selecting a compact candidate, then check its final cost.
    handles = evidence_handles.index(store)
    handle = handles.name(base, allocate=False)
    mode = 'UNCHANGED' if target == data['text'] else 'DELTA'
    labels = dict(base_handle=handle)
    if tool:
        labels['target_handle'] = handles.name(meta['target'], allocate=False)
    payload = _payload(meta, full, mode, dict(labels, edits=edits))
    tokens = _count(counter, payload)
    choice['candidate_tokens'][mode] = tokens
    if tokens is None:
        return dict(choice, reason='token_count_failed')
    if tokens >= full_tokens:
        return dict(choice, reason='compact_not_smaller')
    if handles.name(base) != handle:
        raise RuntimeError('Read baseline handle changed')
    if tool:
        # Allocating an unnamed base can advance this other command's object number.
        labels['target_handle'] = handles.name(meta['target'])
        payload = _payload(meta, full, mode, dict(labels, edits=edits))
        tokens = _count(counter, payload)
        choice['candidate_tokens'][mode] = tokens
        if tokens is None:
            return dict(choice, reason='token_count_failed')
        if tokens >= full_tokens:
            return dict(choice, reason='compact_not_smaller')
    return dict(choice, mode=mode, reason='unchanged' if mode == 'UNCHANGED' else 'delta_smaller',
                payload=payload, base=base, edits=edits, **labels)


def _replace(request, pointer, value):
    parent, key = pointer.rsplit('/', 1)
    target = at_pointer(request, parent)
    key = int(key) if isinstance(target, list) else key.replace('~1', '/').replace('~0', '~')
    target[key] = value


def prepare_request(store, request, context_id, counter):
    """Render a temporary request. The conversation keeps its exact full fallback."""
    sources = text_sources(request)
    if not any(s.get('pf_read') for s in sources):
        return []
    available, replacements = {}, []
    recent = _recent(store, context_id)
    for source in sources:
        if not source.get('pf_read'):
            _literal_bases(store, source, available)
            continue
        read_id = source['pf_read']['id']
        meta, raw = store.get_content(read_id)
        full = raw.decode('utf-8')
        if source['text'] != full:
            raise ValueError('PF read differs from its saved full response')
        key = 'pf_read_choice:' + store.identity['branch_id'] + ':' + read_id
        choice = store.load_state(key)
        if choice is None:
            choice = _choose(store, read_id, meta, full, available, counter, context_id, recent)
            store.save_state(key, choice)
        delivery = dict(choice, materialization=False)
        target = store.get_content(meta['target'])[1].decode('utf-8')
        chain = []
        if choice['mode'] in ('DELTA', 'UNCHANGED'):
            base = available.get(choice['base'])
            try:
                valid = base is not None and apply_delta(base['text'], choice['edits']) == target
            except (ValueError, TypeError):
                valid = False
            counted = (counter is not None and counter.metadata == choice['tokenizer']
                       and _count(counter, choice['payload']) is not None and _count(counter, full) is not None)
            if not valid or not counted:
                reason = ('base_missing' if base is None else 'delta_verification_failed') if not valid else 'tokenizer_unavailable'
                delivery.update(mode='FULL', reason=reason,
                                payload=full, materialization=True)
            else:
                chain = base['chain']
        actual = delivery.pop('payload')
        tool = meta.get('read_kind') == 'tool_call'
        delivery.update(read_id=read_id, target=meta['target'], range=meta['read_range'],
                        full_version=read_id, complete=meta['complete'], context_id=context_id,
                        actual_tokens=_count(counter, actual), full_tokens=_count(counter, full),
                        tokenizer=counter.metadata if counter else None, chain=chain,
                        key=meta['key'], read_kind=meta.get('read_kind', 'pf_read'))
        pf = dict(id=read_id, delivery=delivery)
        if tool:
            # Keep the tool return's own evidence origins (files, SQL, streams): they are
            # delivered literally when FULL and by exact reconstruction otherwise.
            origins = list(source['origins']) if delivery['mode'] == 'FULL' else []
            pf['source_origins'] = source['origins']
            if delivery['mode'] == 'FULL':
                _literal_bases(store, source, available)
        elif meta.get('view'):
            origins = list(source['origins'])
        else:
            prefix = actual.index('\n') + 1
            origins = ([origin(meta['target'], target, *meta['read_range'], prefix)]
                       if delivery['mode'] == 'FULL' else [])
        value = CapturedText(actual, origins, pf)
        replacements.append((source['pointer'], at_pointer(request, source['pointer']), value))
        if meta['complete']:
            available[meta['target']] = dict(text=target, key=meta['key'], chain=[
                *chain, dict(version=meta['target'], pointer=source['pointer'], range=[0, len(actual)])])
    for pointer, _, value in replacements:
        _replace(request, pointer, value)
    return [(pointer, original) for pointer, original, _ in replacements]


def restore_request(request, replacements):
    for pointer, value in replacements:
        _replace(request, pointer, value)


def _message_id(body, pointer, event):
    parts = pointer.split('/')[1:]
    value = body
    for key in parts:
        if isinstance(value, dict):
            for field in ('tool_call_id', 'tool_use_id', 'call_id', 'id'):
                if isinstance(value.get(field), str):
                    return value[field]
        value = value[int(key)] if isinstance(value, list) else value[key]
    return event + ':' + pointer


def record_request(store, event, body, sources, context):
    """Verify compact reconstruction against the serialized wire, never a cache flag."""
    available, reads, reconstructed = {}, [], []
    for source in sources:
        if not source.get('pf_read'):
            _literal_bases(store, source, available)
            continue
        read_id = source['pf_read']['id']
        meta, full = store.get_content(read_id)
        full = full.decode('utf-8')
        actual = at_pointer(body, source['pointer'])
        tool = meta.get('read_kind') == 'tool_call'
        delivery = source['pf_read'].get('delivery') or dict(
            read_id=read_id, target=meta['target'], range=meta['read_range'], full_version=read_id,
            mode='FULL' if meta['mode'] == 'content' else 'DIFF', reason='not_prepared',
            actual_tokens=None, full_tokens=None, tokenizer=None, recovery_of=None, materialization=False,
            key=meta['key'], read_kind=meta.get('read_kind', 'pf_read'))
        target = store.get_content(meta['target'])[1].decode('utf-8')
        chain = []
        if delivery['mode'] in ('DELTA', 'UNCHANGED'):
            base = available.get(delivery['base'])
            if (not meta['complete'] or base is None or
                    apply_delta(base['text'], delivery['edits']) != target or
                    actual != _payload(meta, full, delivery['mode'], delivery)):
                raise ValueError('Compact PF read cannot be reconstructed from the final request')
            chain = base['chain']
            common = dict(request_range=[0, len(actual)], reconstructible=True,
                representation=delivery['mode'], reconstructed_from=chain, read_id=read_id,
                reader=store.identity.get('role', 'ceo'), context_id=context, json_pointer=source['pointer'],
                send_state_event_id=event)
            if tool:
                reconstructed.extend(dict(common, version_id=item['version_id'], source_range=item['source_range'],
                                          full_source=item['full_source'])
                                     for item in source['pf_read'].get('source_origins', []))
            else:
                reconstructed.append(dict(common, version_id=meta['target'], source_range=[0, len(target)],
                                          full_source=False))
        elif actual != full:
            raise ValueError('PF full response differs from saved evidence')
        elif tool:
            _literal_bases(store, source, available)
        read = dict(delivery, context_id=context, json_pointer=source['pointer'],
                    message_id=_message_id(body, source['pointer'], event), chain=chain)
        read['actual_version'] = store.version(event, 'pf_payload_' + str(len(reads)), actual, layer='pf_read_payload')
        reads.append(read)
        if meta['complete']:
            available[meta['target']] = dict(text=target, key=meta['key'], chain=[
                *chain, dict(version=meta['target'], pointer=source['pointer'], range=[0, len(actual)])])
    store.version(event, 'pf_reads', encoded(reads), layer='pf_read_ledger')
    if reconstructed:
        store.version(event, 'reconstructed', encoded(reconstructed), layer='model_reconstructions')


def accounting(store):
    """Rebuild the ledger from actual successful HTTP requests; retain exclusions."""
    reads, excluded = [], Counter()
    with closing(store.connect()) as conn:
        events = conn.execute('''SELECT r.event_id,s.record FROM requests r LEFT JOIN results s USING(event_id)
            WHERE json_extract(r.request,'$.kind')='model_request' ORDER BY r.rowid''').fetchall()
    for row in events:
        try:
            event = store.read_event(row['event_id'])
            ledger = json.loads(store.get_content(row['event_id'] + ':pf_reads')[1])
        except KeyError:
            continue
        if event['result'].get('status') != 'succeeded' or event['result'].get('send_state') != 'response_received':
            excluded[event['result']['status']] += len(ledger)
            continue
        reads.extend(dict(item, request_event=row['event_id']) for item in ledger)
    paired = {item['recovery_of'] for item in reads if item.get('recovery_of')}
    recovery_reads = {item['read_id'] for item in reads if item.get('recovery_of')}
    gross, conservative, replay, materialized, diff_tokens, missing, diff_missing = 0, 0, 0, 0, 0, 0, 0
    for item in reads:
        actual, full = item['actual_tokens'], item['full_tokens']
        if item['mode'] == 'DIFF':
            diff_tokens += actual or 0
            diff_missing += actual is None
            continue
        if actual is None or full is None:
            missing += 1
            continue
        gross += full - actual
        if item['read_id'] in recovery_reads:
            # The recovered full text never occurs in the full-read counterfactual, so
            # every occurrence of it is extra input on top of the zeroed compact read.
            replay += actual
        elif item['read_id'] not in paired:
            conservative += full - actual
        if item['materialization']:
            # Already sent as full text: zero saving against the counterfactual, no deduction.
            materialized += actual
    first = {item['read_id']: item for item in reversed(reads)}
    return dict(reads=len(first), occurrences=len(reads), excluded_occurrences=dict(excluded),
        read_kinds=dict(Counter(v.get('read_kind', 'pf_read') for v in first.values())),
        read_modes=dict(Counter(v['mode'] for v in first.values())),
        occurrence_modes=dict(Counter(v['mode'] for v in reads)),
        reasons=dict(Counter(v['reason'] for v in reads)), recovery_pairs=len(paired),
        missing_token_counts=missing, known_gross_saved_tokens=gross,
        known_conservative_saved_tokens=conservative, known_recovery_replay_tokens=replay,
        materialized_full_tokens=materialized,
        net_saved_tokens=None if missing else conservative - replay,
        active_diff_tokens=None if diff_missing else diff_tokens, active_diff_missing_counts=diff_missing)


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from .sql_evidence import SQLEvidenceStore
    parser = argparse.ArgumentParser(description='Rebuild PF read costs from a saved run.')
    parser.add_argument('run', type=Path)
    run = parser.parse_args().run
    manifest = json.loads((run / 'manifest.json').read_bytes())
    print(json.dumps(accounting(SQLEvidenceStore(run / 'sql-evidence.sqlite', manifest['sql_evidence'])), indent=2))
