"""Event-based accounting does not depend on the text shown to the agent."""
import json
import sqlite3

from scripts.analyze_pf_run import pf_calls, refresh_events, response_calls


def test_refresh_receipts_multiple_calls_and_legacy_logs():
    with sqlite3.connect(':memory:') as conn:
        conn.executescript('CREATE TABLE requests(event_id TEXT, request TEXT);'
                           'CREATE TABLE results(event_id TEXT, record TEXT);')
        conn.execute('INSERT INTO requests VALUES (?,?)', ('parent', json.dumps(
            dict(kind='pf_weekly_check', request=dict(day=490)))))
        for i, (status, code, attempted) in enumerate([
                ('succeeded', 200, True), ('failed', 500, True), ('failed', 504, True),
                ('failed', 503, False)]):
            conn.execute('INSERT INTO requests VALUES (?,?)', (str(i), json.dumps(
                dict(kind='sql_query', parent_event_id='parent'))))
            conn.execute('INSERT INTO results VALUES (?,?)', (str(i), json.dumps(
                dict(status=status, http_status=code, execution=dict(attempted=attempted)))))
        assert [r['outcome'] for r in refresh_events(conn)] == ['succeeded', 'error', 'timeout', 'not_attempted']
    for response in ({'choices': [{'message': {'tool_calls': [1, 2]}}]},
                     {'output': [{'type': 'function_call'}, {'type': 'function_call'}]},
                     {'content': [{'type': 'tool_use'}, {'type': 'tool_use'}]}):
        assert len(response_calls(response)) == 2
    audit = dict(operation='pf_log', arguments={}, event_id=None, outcome='succeeded')
    tool = dict(tool='bash', arguments=dict(command='pf log a'), result='')
    assert len(pf_calls(dict(tool, pf_call=audit), set(), {})) == 1
    assert len(pf_calls(dict(tool, pf_calls=[audit, audit]), set(), {})) == 2
