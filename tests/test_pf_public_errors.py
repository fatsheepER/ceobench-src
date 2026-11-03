"""Unexpected PF failures retain private diagnostics outside model-visible replies."""
from contextlib import closing
import json

import pytest

from test_pf_candidates import query
from test_receipt_outputs import model_receives
from test_sql_evidence import event_ids
from test_text_registry import captured, workspace


@pytest.mark.parametrize('error_type', [RuntimeError, ValueError, KeyError, PermissionError, TimeoutError])
def test_pf_shell_exception_keeps_private_diagnostic_and_next_call(workspace, tmp_path, monkeypatch, error_type):
    store, _, executor = captured(workspace, tmp_path)
    secret = '/private/attempt-secret/evidence.sqlite session-private-' + 'e' * 64
    original = executor.pf_queries.execute

    def fail(operation, arguments):
        raise error_type(secret)

    monkeypatch.setattr(executor.pf_queries, 'execute', fail)
    output = executor.execute('bash', {'command': 'pf log file.txt'})
    assert '[exit code:' in output and secret not in output
    child = next(store.read_event(event) for event in event_ids(store)
                 if store.read_event(event)['request']['kind'] == 'pf_log')
    assert secret in child['result']['private_error']['message']
    assert secret not in model_receives(store, workspace, output)
    monkeypatch.setattr(executor.pf_queries, 'execute', original)
    assert 'after-failure' in executor.execute('bash', {'command': 'printf after-failure'})
    store.assert_healthy()


def test_pf_contract_error_remains_actionable(workspace, tmp_path):
    store, _, executor = captured(workspace, tmp_path)
    output = executor.execute('bash', {'command': 'pf more c999'})
    assert 'Unknown cursor; run the pf command again' in output
    assert 'Unknown cursor' in model_receives(store, workspace, output)
    store.assert_healthy()


def test_pf_candidate_diagnostics_project_before_details_and_model(workspace, tmp_path):
    store, _, executor = captured(workspace, tmp_path)
    executor.execute('write_file', {'path': 'business.txt', 'content': '{"sha256":"business-digest","session_id":"business-session"}'})
    event = next(event for event in event_ids(store) if store.read_event(event)['request']['kind'] == 'write_file')
    secret = '/private/attempt-secret/session-raw/' + 'f' * 64
    with closing(store.connect()) as conn, conn:
        result = json.loads(conn.execute('SELECT record FROM results WHERE event_id=?', (event,)).fetchone()[0])
        result.update(error=secret, error_kind='service', capture_gaps=['unobserved_internal_file_reads', secret])
        conn.execute('UPDATE results SET record=? WHERE event_id=?', (json.dumps(result), event))
        conn.commit()
    output, page = query(executor, 'pf candidates --history --kind file --text business.txt --detail')
    assert secret not in model_receives(store, workspace, output)
    chunks = [page['detail_text']]
    while page['next_cursor']:
        output, page = query(executor, 'pf more ' + page['next_cursor'])
        chunks.append(page['detail_text'])
        assert secret not in model_receives(store, workspace, output)
    detail = ''.join(chunks)
    assert secret not in detail
    assert 'unobserved_internal_file_reads' in detail and 'unspecified_capture_gap' in detail
    assert 'details_retained_privately' in detail and 'service' in detail
    assert secret in store.read_event(event)['result']['error']
    body = executor.execute('bash', {'command': 'pf show business.txt --full'})
    assert 'business-digest' in body and 'business-session' in body
    assert secret not in model_receives(store, workspace, body)
    store.assert_healthy()
