import json
import shlex
import shutil
from pathlib import Path

import pytest

from saas_bench.pf_queries import PFQueries
from saas_bench.text_registry import TextRegistry
from test_execution_capture import captured as http_captured, records
from test_public_sql import server
from test_receipt_outputs import artifact, model_receives
from test_sql_evidence import settled
from test_text_registry import captured, workspace


def day(store, value):
    event = store.begin_event('dashboard_generation', {'day': value})
    store.version(event, 'dashboard', 'Day ' + str(value), layer='dashboard')
    store.complete(event)


def query(executor, command='pf candidates'):
    output = executor.execute('bash', {'command': command})
    assert '[exit code:' not in output and not output.startswith('Error:'), output
    page = executor.pf_queries.last_answer
    artifact('candidates.json', page)
    return output, page


def drain(executor, page):
    items = list(page['items'])
    while page['next_cursor']:
        _, page = query(executor, 'pf more ' + page['next_cursor'])
        items += page['items']
    return items


def test_cross_call_candidates_keep_q0_queries_files_and_repeat_events(http_captured):
    api, store, executor = http_captured
    today = [0]
    registry = TextRegistry(executor.workspace_path, 'pf', store, sim_day=lambda: today[0])
    executor.text_registry, executor.pf_queries = registry, PFQueries(registry)
    shutil.copy(Path(__file__).parents[1] / 'public/novamind-operation', executor.workspace_path)
    day(store, 0)
    q0 = executor.execute('bash', {'command': './novamind-operation query "SELECT 0 AS early_q0"'})
    assert 'early_q0' in q0
    today[0] = 7
    api.tools.set_current_day(7)
    day(store, 7)
    sql = './novamind-operation query "SELECT 1 AS q1"; ./novamind-operation query "SELECT 2 AS q2"'
    for _ in range(2):
        assert 'q1' in executor.execute('bash', {'command': sql})
    code = "from pathlib import Path; [Path(f'F{i}.txt').write_text(f'projection {i}') for i in range(24)]"
    executor.execute('bash', {'command': shlex.quote(executor.python) + ' -c ' + shlex.quote(code)})
    output, page = query(executor)
    assert page['returned'] == 20 and page['has_more']
    assert page['scope']['day_from'] == 7 and page['scope']['day_to'] == 13
    assert page['scope']['role'] == 'ceo' and 'cutoff' in output.lower()
    assert 'not proof of use' in output
    cursor = page['next_cursor']
    first_page = page
    executor.execute('write_file', {'path': 'later.txt', 'content': 'after cutoff'})
    all_items = drain(executor, first_page)
    assert len(all_items) == first_page['total_in_scope']
    assert {r['name'] for r in all_items if r['kind'] == 'file' and r['name'].startswith('F')} == {f'F{i}.txt' for i in range(24)}
    assert not any(r['name'] == 'later.txt' for r in all_items)
    queries = [r for r in all_items if r['kind'] == 'sql']
    assert len(queries) == 4 and len({r['event_ref'] for r in queries}) == 4
    assert len({r['version'] for r in queries}) == 2
    assert len({r['content_ref'] for r in queries}) == 4
    _, second = query(executor, 'pf more ' + cursor)
    assert second['cutoff'] == first_page['cutoff']
    _, older = query(executor, 'pf candidates --history --kind sql --text early_q0')
    assert older['total_in_scope'] == 1 and older['items'][0]['day'] == 0
    before = len([r for r in records(store) if r['request']['kind'] == 'sql_query'])
    item = older['items'][0]
    full = executor.execute('bash', {'command': 'pf show ' + shlex.quote(item['content_ref']) + ' --full'})
    assert 'early_q0' in full
    details, detail = query(executor, 'pf candidates --history --event ' + shlex.quote(item['event_ref']) + ' --detail')
    assert 'SELECT 0 AS early_q0' in details and 'request' in details and 'receipt' in details
    assert len([r for r in records(store) if r['request']['kind'] == 'sql_query']) == before
    model_receives(store, executor.workspace_path, output)
    settled(api)
    store.assert_healthy()


def test_candidates_cursor_freezes_late_result_and_rejects_invalid_snapshot(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    day(store, 7)
    pending = store.begin_event('public_http', {'method': 'GET', 'path': '/vars', 'parsed': {}})
    output, page = query(executor, 'pf candidates --event ' + pending)
    assert page['items'][0]['capture']['state'] == 'missing'
    assert page['items'][0]['status'] == 'result_unknown'
    cursor = page['detail_cursor']
    store.version(pending, 'response', b'{"success":true}', layer='server_public_response')
    store.complete(pending, 'succeeded')
    detail_text, detail_page = query(executor, 'pf more ' + cursor)
    assert 'result_unknown' in detail_text and detail_page['cutoff'] == page['cutoff']
    _, fresh = query(executor, 'pf candidates --event ' + pending)
    assert fresh['items'][0]['status'] == 'succeeded'
    cursors = store.load_state('pf_cursors')
    snapshot = cursors[cursor]['candidate_snapshot']
    store.save_state(snapshot, None)
    failed = executor.execute('bash', {'command': 'pf more ' + cursor})
    assert 'snapshot' in failed.lower() and '[exit code: 1]' in failed
    assert 'succeeded' in query(executor, 'pf candidates --event ' + pending)[1]['items'][0]['status']
    store.assert_healthy()


@pytest.mark.parametrize('args', ['--role stranger', '--day-from -1', '--day-from 14 --day-to 7', '--kind secret', '--event absent', 'extra'])
def test_candidates_bad_filters_are_explicit_errors(workspace, tmp_path, args):
    _, _, executor = captured(workspace, tmp_path)
    output = executor.execute('bash', {'command': 'pf candidates ' + args})
    assert '[exit code:' in output and ('Error:' in output or 'pf candidates:' in output), output


def test_candidates_capture_truncation_binary_streams_and_partial_failure(http_captured, monkeypatch):
    api, store, executor = http_captured
    registry = TextRegistry(executor.workspace_path, 'pf', store, sim_day=lambda: 0)
    executor.text_registry, executor.pf_queries = registry, PFQueries(registry)
    shutil.copy(Path(__file__).parents[1] / 'public/novamind-operation', executor.workspace_path)
    day(store, 0)
    monkeypatch.setitem(api.runtime_environment, 'CEOBENCH_SQL_ROW_LIMIT', '1')
    result = executor.execute('bash', {'command': './novamind-operation query "SELECT 1 AS n UNION ALL SELECT 2"; ./novamind-operation query "SELECT no_column FROM ledger"; printf saved > partial.txt; exit 1'})
    assert '[exit code: 1]' in result and 'captures: pf more' in result
    _, page = query(executor, 'pf candidates --kind sql')
    assert {item['status'] for item in page['items']} == {'succeeded', 'failed'}
    assert any(item['capture']['source_truncated'] for item in page['items'])
    assert all('unobserved_internal_file_reads' in item['capture']['gaps'] for item in page['items'])
    _, files = query(executor, 'pf candidates --kind file --text partial.txt')
    assert len(files['items']) == 1 and files['items'][0]['status'] == 'failed'
    assert 'saved' in executor.execute('bash', {'command': 'pf show ' + files['items'][0]['version'] + ' --full'})
    executor.execute('write_file', {'path': 'binary.py', 'content': "import sys;sys.stdout.buffer.write(b'\\xff');sys.stderr.write('known stderr')"})
    executor.execute('bash', {'command': './novamind-operation python binary.py'})
    _, streams = query(executor, 'pf candidates --kind code --text binary.py')
    assert streams['items'] and any('stdout_bytes_without_public_text' in item['capture']['gaps'] for item in streams['items'])
    settled(api)
    store.assert_healthy()


def test_candidates_do_not_accept_private_events_even_without_team_identity(workspace, tmp_path):
    store, _, executor = captured(workspace, tmp_path)
    event = store.begin_event('model_request', {'context_id': 'private'})
    store.version(event, 'wire', 'PRIVATE_MODEL_CONTENT', layer='model_request_wire')
    store.complete(event)
    result = executor.execute('bash', {'command': 'pf candidates --history --event ' + event})
    assert '[exit code: 1]' in result and 'inaccessible' in result and 'PRIVATE_MODEL_CONTENT' not in result
    store.assert_healthy()


def test_candidate_deadline_closes_public_client_and_next_call_continues(workspace, tmp_path, monkeypatch):
    import time
    from saas_bench.pf_candidates import CandidateQuery
    from saas_bench.pf_deadline import check, current_budget
    store, _, executor = captured(workspace, tmp_path)
    day(store, 7)
    executor.execute('write_file', {'path': 'source.txt', 'content': 'saved'})
    original = CandidateQuery._text_match
    def slow(self, item, text, conn):
        assert current_budget() is not None
        while True:
            check()
            time.sleep(.005)
    monkeypatch.setattr(CandidateQuery, '_text_match', slow)
    executor.pf_queries.call_timeout = .1
    failed = executor.execute('bash', {'command': 'pf candidates --text saved'})
    assert 'time limit exceeded' in failed
    assert executor._pf_service is None
    monkeypatch.setattr(CandidateQuery, '_text_match', original)
    executor.pf_queries.call_timeout = 600
    _, page = query(executor, 'pf candidates --kind file --text source.txt')
    assert page['items'][0]['name'] == 'source.txt'
    assert executor.execute('bash', {'command': 'printf next-call'}) == 'next-call'
    store.assert_healthy()
