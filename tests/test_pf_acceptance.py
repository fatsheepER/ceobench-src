"""Stage 4 acceptance: one constructed history, real SDK wires, independent costs.

Set CEOBENCH_STAGE4_ARTIFACTS to export evidence with the pinned official tokenizer.
Ordinary offline regression uses explicitly synthetic byte costs.
"""
from collections import Counter as Counts
from contextlib import closing
import json
import os
from pathlib import Path

import httpx
from openai import OpenAI

from saas_bench.execution_capture import at_pointer, restore_sources, text_sources
from saas_bench.model_usage import ModelUsage
from saas_bench.payload_tokens import load_counter
from saas_bench.pf_read import accounting
from saas_bench.pf_refresh import refresh
from saas_bench.public_sql import execute_query
from saas_bench.sql_evidence import digest
from test_pf_read import Counter, sample
from test_pf_stale import record_sql
from test_preflight_usage import reply
from test_public_sql import server
from test_text_registry import captured, declaration, workspace


def test_constructed_trajectory_queries_and_request_costs(workspace, tmp_path, server):
    destination = os.environ.get('CEOBENCH_STAGE4_ARTIFACTS')
    counter = load_counter('deepseek', 'deepseek-flash') if destination else Counter()
    assert counter is not None
    store, registry, executor = captured(workspace, tmp_path)
    server.sql_evidence = store
    executor.pf_queries.refresh = lambda versions, parent: refresh(server, versions, parent)
    wires, requests, actions = [], [], []
    request = dict(model='deepseek-flash', messages=[])

    def handle(wire):
        wires.append(json.loads(wire.content))
        # No fabricated provider usage: this is an offline mechanism check.
        return httpx.Response(200, json=reply('chat', usage=False))

    def tool(name, **args):
        output = executor.execute(name, args)
        assert not output.startswith('Error:'), output
        call_id = f'tool-{len(actions)}'
        request['messages'].extend([
            dict(role='assistant', content='', tool_calls=[dict(id=call_id, type='function',
                 function=dict(name=name, arguments=json.dumps(args, ensure_ascii=False)))]),
            dict(role='tool', tool_call_id=call_id, content=output),
        ])
        actions.append(dict(id=call_id, tool=name, args=args, output=str(output)))
        return output

    def query(name, **args):
        if not name.startswith('pf_'):
            output = tool(name, **args)
            return json.loads(output) if name == 'text_list' else executor.text_registry.last_result
        if name in ('pf_dependencies', 'pf_dependents') and 'cursor' not in args:
            args['detail'] = True
        tool(name, **args)  # the agent-facing text goes into the request
        return executor.pf_queries.last_answer

    def read(**args):
        return tool('pf_read', **args)

    def captured_file(content):
        tool('write_file', path='facts.txt', content=content)
        return read(target={'path': 'facts.txt'})

    def refresh_count():
        with closing(store.connect()) as conn:
            return conn.execute("SELECT count(*) FROM results WHERE json_extract(record,'$.refresh_of') IS NOT NULL").fetchone()[0]

    with OpenAI(api_key='offline', base_url='http://127.0.0.1:1/v1', max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        usage = ModelUsage(tmp_path / 'requests.jsonl', 'agent', evidence_store=store, token_counter=counter)
        usage.context_id = 'week-1'
        usage.attach(client)

        def send(label):
            original = json.dumps(request, ensure_ascii=False)
            usage.call('chat', request, lambda: client.chat.completions.create(**request))
            assert json.dumps(request, ensure_ascii=False) == original
            with closing(store.connect()) as conn:
                event = conn.execute("SELECT event_id FROM requests WHERE json_extract(request,'$.kind')='model_request' ORDER BY seq DESC LIMIT 1").fetchone()[0]
            wire = json.loads(store.get_content(event + ':wire')[1])
            assert wire == wires[-1]
            ledger = json.loads(store.get_content(event + ':pf_reads')[1])
            available = {}
            for item in ledger:
                actual = at_pointer(wire, item['json_pointer'])
                assert actual == store.get_content(item['actual_version'])[1].decode()
                full_meta, full = store.get_content(item['full_version'])
                full = full.decode()
                header, payload = actual.split('\n', 1)
                header = json.loads(header)
                assert item['context_id'] == usage.context_id
                assert item['message_id'] == at_pointer(wire, item['json_pointer'].rsplit('/', 1)[0])['tool_call_id']
                assert item['range'] == full_meta['read_range'] == header['range']
                assert item['actual_tokens'] == counter.count(actual)
                assert item['full_tokens'] == counter.count(full)
                assert item['tokenizer'] == counter.metadata
                if item['mode'] == 'DIFF':
                    continue
                if item['mode'] in ('DELTA', 'UNCHANGED'):
                    assert item['actual_tokens'] < item['full_tokens']
                    restored = available[header['base']]
                    # Independent patch oracle: edit backwards, not via apply_delta.
                    if item['mode'] == 'DELTA':
                        for start, end, replacement in reversed(json.loads(payload)):
                            restored = restored[:start] + replacement + restored[end:]
                    assert item['chain']
                else:
                    restored = payload
                target = store.get_content(item['target'])[1].decode()
                assert restored == target == full.split('\n', 1)[1]
                available[header['target']['version']] = restored
            requests.append(dict(label=label, event=event, context=usage.context_id, reads=ledger))
            return ledger

        # Read V before promoting it to A's dependency, then read A before F cites it.
        sql = "SELECT category,NULLIF(amount,-1) AS amount,40 AS floor FROM ledger"
        version = record_sql(store, sql, execute_query(server, sql))
        original_sql = store.get_content(version)[1]
        original = sample()
        full = captured_file(original)
        sql_read = read(target={'sql': sql})
        assert [r['mode'] for r in send('first full reads')] == ['FULL', 'FULL']
        selector = dict(row={'category': 'operations'}, col='amount')
        reference = dict(evidence={'version': json.loads(sql_read.split('\n')[0])['target']['version']},
                         purpose='current', select=selector,
                         predicate=dict(type='threshold', op='>=', value=40), note='Keep at least forty')
        query('text_create', **declaration(references=[reference], text='A: sufficient amount'))
        read(target={'record': 'r1'})
        send('deliver assumption A')
        query('text_create', **declaration({'record': 'r1.1'}, text='F: forecast based on A'))
        compare_ref = dict(evidence=reference['evidence'], purpose='current', predicate=dict(
            type='compare', left=selector, op='>', right=dict(row={'category': 'operations'}, col='floor')))
        query('text_create', **declaration(references=[compare_ref], text='C: amount exceeds floor'))
        read(target={'record': 'r3'})
        send('deliver two-cell assumption C')
        query('text_create', **declaration({'record': 'r3.1'}, text='G: forecast based on C'))
        query('text_create', **declaration(references=[dict(evidence={'record': 'r1.1'}, purpose='historical_only')]))
        query('text_create', **declaration())
        read(target={'path': 'facts.txt'})
        assert send('unchanged file')[-1]['mode'] == 'UNCHANGED'

        def check_forecasts(result, reason=None):
            for record in ('r2', 'r4'):
                before = server.conn.serialize()
                declarations = registry.path.read_bytes()
                count = refresh_count()
                rows = query('pf_dependencies', target={'record': record}, purpose='current')['items']
                assert refresh_count() == count + 1
                assert server.conn.serialize() == before
                assert registry.path.read_bytes() == declarations
                upstream, leaf = rows
                assert leaf['check']['predicate_result'] == result
                assert upstream['check']['version_changed'] is False
                assert upstream['check']['affected'] is (result != 'holds')
                assert upstream['check']['affected_paths'] == ([] if result == 'holds' else [leaf['path']])
                assert leaf['check']['notice'] == {
                    'holds': 'strong_dependency_version_changed', 'fails': 'predicate_failed',
                    'cannot_check': 'cannot_check'}[result]
                if reason:
                    assert reason in leaf['check']['reason']

        server.conn.execute('UPDATE ledger SET amount=43')
        server.conn.commit()
        check_forecasts('holds')
        captured_file(original.replace('记录 012:', '修订 012:'))
        assert send('predicate holds; first delta')[-1]['mode'] == 'DELTA'
        server.conn.execute('UPDATE ledger SET amount=39')
        server.conn.commit()
        check_forecasts('fails')
        final = original.replace('记录 012:', '修订 012:').replace('记录 041:', '修订 041:').removesuffix('\r\n')
        delta2 = captured_file(final)
        assert send('predicate fails; second delta')[-1]['mode'] == 'DELTA'
        # Only a full read in the request right after the compact read is a recovery pair.
        recovery = read(target={'path': 'facts.txt'}, full=True)
        row = send('adjacent explicit full recovery')[-1]
        assert row['mode'] == 'FULL' and row['recovery_of'] == delta2.pf_read['id']

        # The same F -> A -> V paths must retain each concrete unknown reason.
        for label, amounts, unavailable, reason in [
            ('NULL', [-1], False, 'selected_value_null'),
            ('missing row', [], False, 'no rows'),
            ('ambiguous rows', [43, 43], False, 'multiple rows'),
            ('wrong type', ['unknown'], False, 'not_numeric'),
            ('truncated', [43] * 5001, False, 'source_truncated'),
            ('refresh unavailable', [43], True, 'refresh_failed:failed'),
        ]:
            server.conn.execute('DELETE FROM ledger')
            server.conn.executemany("INSERT INTO ledger(day,category,amount) VALUES(0,'operations',?)", [(v,) for v in amounts])
            server.conn.commit()
            server._operation_failed = unavailable
            check_forecasts('cannot_check', reason)
            send(label)
        server._operation_failed = False

        # A saved page remains stable after the world changes, and history never refreshes.
        page = query('pf_dependencies', target={'record': 'r2'}, purpose='current', limit=1)
        count = refresh_count()
        server.conn.execute('UPDATE ledger SET amount=42')
        server.conn.commit()
        tail = query('pf_dependencies', cursor=page['next_cursor'])
        assert tail['items'][0]['check']['predicate_result'] == 'holds'
        assert query('pf_dependencies', cursor=page['next_cursor']) == tail
        assert query('pf_dependencies', target={'record': 'r5'})['items'][1]['check']['reason'] == 'historical_only'
        missing = query('pf_dependencies', target={'record': 'r6'})['items'][0]
        assert missing['target'] is None and missing['unexpanded'] == 'missing'
        assert missing['check']['reason'] == 'No saved evidence'
        query('text_revise', record='r1', text='A revised', reason='Constructed correction')
        history = query('pf_read', target={'record': 'r1'}, mode='history')
        assert [r['record'] for r in history['items']] == ['r1.2', 'r1.1']
        reverse = query('pf_dependents', target={'version': reference['evidence']['version']})
        assert {'r2.1', 'r5.1'} <= {r['source']['record'] for r in reverse['items']}
        assert next(r for r in reverse['items'] if r['source']['record'] == 'r5.1')['historical_only']
        page = query('pf_search', object={'kind': 'plan', 'id': 'B'}, limit=1, all=True)
        records = [page['items'][0]['record']]
        while page['next_cursor']:
            page = query('pf_search', cursor=page['next_cursor'])
            records.extend(r['record'] for r in page['items'])
        assert set(records) == {f'r{i}.1' for i in range(1, 7)} | {'r1.2'}
        query('text_list')
        query('pf_dependencies', target={'record': 'r2'}, purpose='historical_only')
        assert refresh_count() == count
        # Deliberately malformed capture graph, not a claimed real execution history.
        event = store.begin_event('constructed_cycle')
        a = store.version(event, 'a', 'A', layer='stdout', derived_from=event + ':b')
        store.version(event, 'b', 'B', layer='stdout', derived_from=a)
        store.complete(event)
        cycle = query('pf_dependencies', target={'version': registry.resolver.handle(a)}, include_execution=True, depth=10)
        assert len(cycle['items']) == 2 and cycle['items'][-1]['unexpanded'] == 'cycle'
        send('revision, pagination, history, missing reference and cycle')

        diff = read(target={'path': 'facts.txt'}, baseline={'version': json.loads(full.split('\n')[0])['target']['version']}, mode='diff')
        assert send('active diff counted separately')[-1]['mode'] == 'DIFF'
        read(target={'path': 'facts.txt'}, full=True)
        row = send('non-adjacent explicit full')[-1]
        assert row['mode'] == 'FULL' and row['reason'] == 'requested_full' and row['recovery_of'] is None

        # Restore private source identities from a serialized context, then prune its base.
        sources = text_sources(request)
        request = json.loads(json.dumps(request))
        restore_sources(request, sources)
        send('restored same-week context')
        assert wires[-1] == wires[-2]
        kept = {full.pf_read['id'], delta2.pf_read['id']}
        pairs = [request['messages'][i:i + 2] for i in range(0, len(request['messages']), 2)]
        request['messages'] = [m for pair in pairs if getattr(pair[1]['content'], 'pf_read', None)
                               and pair[1]['content'].pf_read['id'] in kept for m in pair]
        rows = send('pruned intermediate baseline')
        assert [r['mode'] for r in rows] == ['FULL', 'FULL'] and rows[-1]['materialization']
        request['messages'] = []
        usage.context_id = 'week-2'
        read(target={'path': 'facts.txt'})
        row = send('new week starts without a baseline')[0]
        assert row['mode'] == 'FULL' and row['reason'] == 'no_complete_base'

    # Recount wire payloads independently; the adjacent recovery pair above determines
    # adjustments, not accounting() flags. Materialized replays already cost full text.
    paired = {delta2.pf_read['id'], recovery.pf_read['id']}
    table = []
    for req, wire in zip(requests, wires, strict=True):
        actual = full_cost = conservative = diff_cost = extra = materialized = 0
        for item in req['reads']:
            cost = counter.count(at_pointer(wire, item['json_pointer']))
            baseline = counter.count(store.get_content(item['full_version'])[1].decode())
            if item['read_id'] == diff.pf_read['id']:
                diff_cost += cost
                continue
            actual += cost
            full_cost += baseline
            if item['read_id'] not in paired:
                conservative += baseline - cost
            if item['read_id'] == recovery.pf_read['id']:
                extra += cost
            if req['label'] == 'pruned intermediate baseline' and item['read_id'] == delta2.pf_read['id']:
                materialized += cost
        table.append(dict(request=req['label'], modes=[r['mode'] for r in req['reads']],
            actual_tokens=actual, full_tokens=full_cost, conservative_savings=conservative,
            recovery_replay_tokens=extra, materialized_full_tokens=materialized,
            net_saved_tokens=conservative - extra, active_diff_tokens=diff_cost))
    summary = accounting(store)
    assert summary['known_gross_saved_tokens'] == sum(r['full_tokens'] - r['actual_tokens'] for r in table)
    assert summary['known_conservative_saved_tokens'] == sum(r['conservative_savings'] for r in table)
    assert summary['known_recovery_replay_tokens'] == sum(r['recovery_replay_tokens'] for r in table) > 0
    assert summary['materialized_full_tokens'] == sum(r['materialized_full_tokens'] for r in table) > 0
    assert summary['net_saved_tokens'] == sum(r['net_saved_tokens'] for r in table)
    assert summary['active_diff_tokens'] == sum(r['active_diff_tokens'] for r in table) > 0
    assert summary['recovery_pairs'] == 1 and summary['missing_token_counts'] == 0
    assert summary['occurrences'] == sum(len(r['reads']) for r in requests) > summary['reads']
    assert summary['occurrence_modes'] == dict(Counts(r['mode'] for req in requests for r in req['reads']))
    assert usage.summary['calls'] == len(wires) == usage.summary['missing_cost']
    assert usage.summary['known']['input_tokens'] is None
    assert store.get_content(version)[1] == original_sql
    store.assert_healthy()
    with closing(store.connect()) as conn:
        assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
        for row in conn.execute('SELECT content_hash,content,size_bytes FROM blobs'):
            assert digest(row['content']) == row['content_hash'] and len(row['content']) == row['size_bytes']
    if destination:
        folder = Path(destination)
        folder.mkdir(parents=True, exist_ok=True)
        report = dict(kind='constructed_offline', tokenizer=counter.metadata, provider_model_calls=0,
                      requests=requests, manual_table=table, summary=summary, mock_usage=usage.summary)
        (folder / 'acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        (folder / 'actions.json').write_text(json.dumps(actions, ensure_ascii=False, indent=2))
        (folder / 'requests.jsonl').write_bytes((tmp_path / 'requests.jsonl').read_bytes())
        (folder / 'manifest.json').write_text(json.dumps(dict(sql_evidence=store.identity)))
        store.snapshot(folder / 'sql-evidence.sqlite')
