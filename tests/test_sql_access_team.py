from collections import Counter
import json
import re
import shlex

from saas_bench.multi_agent import MultiAgentRuntime
from test_pf_candidates import day, query
from test_preflight_team_flow import ADVANCE, calls, final, make_team
from test_receipt_outputs import artifact


def test_sql_access_model_discovers_reads_and_declares_exact_query():
    turns = Counter()
    saved = {}

    def respond(team, role, body):
        if role == 'ceo':
            return calls(('bash', {'command': ADVANCE}))
        if role == 'ops_finance':
            return final('No additional action.')
        turns[role] += 1
        results = [m['content'] for m in body['messages'] if m['role'] == 'tool']
        if turns[role] == 1:
            assert '--sql-accesses' in body['messages'][0]['content']
            return calls(('bash', {'command': 'pf candidates --history --sql-accesses --table ledger --column note'}))
        if turns[role] == 2:
            assert 'authorization observations' in results[-1]
            assert 'MODEL_SQL_ACCESS_PAYLOAD' not in results[-1]
            command = next(line.strip() for line in results[-1].splitlines() if line.strip().startswith('pf candidates '))
            return calls(('bash', {'command': command}))
        if turns[role] == 3:
            assert 'MODEL_SQL_ACCESS_PAYLOAD' not in results[-1]
            cursor = re.search(r'Exact events[^\n]+: pf more (c[0-9]+)', results[-1])[1]
            return calls(('bash', {'command': 'pf more ' + cursor}))
        if turns[role] == 4:
            fragment = results[-1].split('not source bodies.\n', 1)[1]
            items, _ = json.JSONDecoder().raw_decode(fragment)
            item, = items
            saved.update(item)
            assert item['model_delivery']['current']['status'] == 'unknown'
            assert 'MODEL_SQL_ACCESS_PAYLOAD' not in results[-1]
            return calls(('bash', {'command': 'pf show ' + shlex.quote(item['content_ref']) + ' --full'}))
        if turns[role] == 5:
            assert 'MODEL_SQL_ACCESS_PAYLOAD' in results[-1]
            return calls(('write_file', {'path': 'access-note.txt', 'content': 'Retain the observed ledger note.'}))
        if turns[role] == 6:
            return calls(('escalate', {'items': [dict(item_id='ledger-source', op='create',
                target={'cite': 'growth:access-note.txt@v1'}, sources=[dict(ref={'cite': saved['version']})],
                assertion='author_declared', reason='Keep the exact query I read.')] }))
        if turns[role] == 7:
            assert json.loads(results[-1])['saved'] == 1
            return calls(('bash', {'command': 'pf candidates --history --table ledger --column note'}))
        assert saved['version'] in results[-1] and 'delivery history=full' in results[-1]
        saved['history_tool_text'] = results[-1]
        return final('The saved note cites the query that I read.')

    with make_team('pf', respond, 'sql-access-model') as team:
        team.server.conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES(0,'operations',-12.34,'MODEL_SQL_ACCESS_PAYLOAD')")
        team.server.conn.commit()
        growth = team.roles['growth']
        day(growth.store, 0)
        output = growth.executor.execute('bash', {'command': './novamind-operation query "SELECT amount,note FROM ledger WHERE amount=-12.34"'})
        assert 'MODEL_SQL_ACCESS_PAYLOAD' in output
        peer = team.roles['ops_finance']
        _, public = query(peer.executor, 'pf candidates --history --role growth --table ledger --column note')
        assert public['items'][0]['model_delivery']['historical']['status'] == 'unknown'
        assert 'model_request_wire' not in json.dumps(public)
        assert team.run(stop_after_day=7).reason == 'observation_end', team.failure
        assert turns['growth'] == 8
        wire = [row['body'] for row in team.test_wire if row['role'] == 'growth']
        usage = [json.loads(line) for line in growth.usage.path.read_text().splitlines()]
        assert [row['body'] for row in usage if row['event'] == 'http_request'] == wire
        assert public['items'][0]['content_ref'] == saved['content_ref']
        artifact('sql-access-model.json', dict(wire=wire, usage=usage, candidate=saved,
            history=saved['history_tool_text'], peer=public, relations=growth.registry.relations._load()))


def test_sql_access_checkpoint_restores_directory_and_discards_future_queries(tmp_path):
    from test_preflight_team_checkpoint import clients, dispose, fast_simulator, make_team as checkpoint_team
    original = checkpoint_team(tmp_path / 'original', 'pf')
    restored = None
    try:
        growth = original.roles['growth']
        growth.executor.execute('bash', {'command': './novamind-operation query "SELECT * FROM customers,subscriptions,research_projects LIMIT 0"'})
        _, directory = query(growth.executor, 'pf candidates --history --sql-accesses')
        assert directory['has_more']
        checkpoint = original.checkpoint()
        growth.executor.execute('bash', {'command': './novamind-operation query "SELECT amount FROM ledger"'})
        _, later = query(growth.executor, 'pf candidates --history --table ledger --column amount')
        original_cursors = growth.store.load_state('pf_cursors')
        restored = MultiAgentRuntime.restore(checkpoint, tmp_path / 'restored', client_factory=clients(), simulator_factory=fast_simulator)
        resumed = restored.roles['growth']
        _, frozen = query(resumed.executor, 'pf more ' + directory['next_cursor'])
        assert frozen['cutoff'] == directory['cutoff']
        assert not any(row['table'] == 'ledger' for row in frozen['items'])
        assert '[exit code: 1]' in resumed.executor.execute('bash', {'command': 'pf more ' + later['detail_cursor']})
        _, after = query(resumed.executor, 'pf candidates --history --table ledger --column amount')
        assert after['total_in_scope'] == 0
        assert restored.run(stop_after_day=7).reason == 'observation_end', restored.failure
        assert growth.store.load_state('pf_cursors') == original_cursors
        artifact('sql-access-checkpoint.json', dict(selected=str(checkpoint), directory=directory, discarded=later, restored=frozen))
    finally:
        if restored:
            dispose(restored)
        dispose(original)
