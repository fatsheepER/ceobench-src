"""Event-based accounting does not depend on the text shown to the agent."""
import json
import sqlite3

from scripts.analyze_pf_run import pf_calls, refresh_events, response_calls, tool_batch_accounting


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


def test_json_rejection_excludes_the_whole_batch_without_hiding_missing_results():
    def response(second, calls):
        return dict(timestamp=f'2026-10-02T00:00:0{second}Z', day=28, turn=second,
                    raw_response={'choices': [{'message': {'tool_calls': calls}}]})

    rejected = [dict(id='bad-write', function=dict(name='write_file',
                  arguments='{"content": "analysis", "<parameter: "/workspace/notes/week4_analysis.md"}')),
                dict(id='unaccepted-bash', function=dict(name='bash', arguments='{"command":"echo snapshot"}'))]
    accepted = [dict(id='write', function=dict(name='write_file', arguments='{"content":"analysis"}')),
                dict(id='bash', function=dict(name='bash', arguments='{"command":"echo snapshot"}'))]
    raw = [response(1, rejected), response(2, accepted)]
    timing = [dict(timestamp='2026-10-02T00:00:03Z', day=28, event='model_tool_batch', calls=2)]
    tools = [dict(day=28, tool='write_file', call_id='write'), dict(day=28, tool='bash', call_id='bash')]
    audit = tool_batch_accounting(raw, tools, timing)
    assert audit['status'] == 'passed'
    assert audit['counts'] == dict(declared=4, accepted=2, rejected=2, completed=2, cancelled=0, unaccounted=0)
    assert audit['rejected_batches'][0]['ids'] == ['bad-write', 'unaccepted-bash']
    assert tool_batch_accounting(raw, tools[:1], timing)['missing_ids'] == {'bash': 1}
    for extra in (tools[0], dict(day=28, tool='bash', call_id='unaccepted-bash')):
        assert tool_batch_accounting(raw, tools + [extra], timing)['status'] == 'failed'
    assert tool_batch_accounting(raw, tools, [])['status'] == 'failed'
    assert tool_batch_accounting(raw, tools, timing + timing)['status'] == 'failed'


def test_cross_week_cancellation_is_a_result_of_an_accepted_call():
    raw = [dict(timestamp='2026-10-02T00:00:01Z', day=28, raw_response={'output': [
        dict(type='function_call', id='provider-id', call_id='advance', name='bash', arguments='{}'),
        dict(type='function_call', call_id='later', name='bash', arguments='{}')]})]
    timing = [dict(timestamp='2026-10-02T00:00:02Z', day=28, event='model_tool_batch', calls=2)]
    tools = [dict(day=28, tool='bash', call_id='advance'),
             dict(day=35, tool='bash', call_id='later', outcome='cancelled')]
    audit = tool_batch_accounting(raw, tools, timing)
    assert audit['status'] == 'passed'
    assert audit['counts'] == dict(declared=2, accepted=2, rejected=0, completed=1, cancelled=1, unaccounted=0)
    assert audit['weeks'][28]['cancelled'] == 1
    assert tool_batch_accounting(raw, tools[:1], timing)['missing_ids'] == {'later': 1}


def test_anthropic_acceptance_and_invalid_acceptance_receipts():
    raw = [dict(timestamp='2026-10-02T00:00:01Z', day=28, raw_response={'content': [
        dict(type='tool_use', id='call', name='bash', input=dict(command='echo ok'))]})]
    timing = [dict(timestamp='2026-10-02T00:00:02Z', day=28, event='model_tool_batch', calls=1)]
    tools = [dict(day=28, tool='bash', call_id='call')]
    assert tool_batch_accounting(raw, tools, timing)['status'] == 'passed'
    for event in (dict(timing[0], calls=2), dict(timing[0], day=35),
                  dict(timing[0], timestamp='2026-10-02T00:00:00Z')):
        assert tool_batch_accounting(raw, tools, [event])['status'] == 'failed'
