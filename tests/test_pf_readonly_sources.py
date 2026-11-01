"""Source navigation reads captured records without evaluating the current world."""
from contextlib import closing
import json
import re
import urllib.error

import pytest

from saas_bench.pf_queries import PFQueries
from pf_fixtures import chain
from test_pf_queries import relation_details
from test_preflight_integration import offline_runner, packed_public
from test_public_sql import server
from test_text_registry import workspace, call, declaration, send


def source_events(store):
    with closing(store.connect()) as conn:
        return conn.execute("""SELECT event_id FROM requests WHERE query_id IS NOT NULL
            OR json_extract(request, '$.kind')='public_http' ORDER BY rowid""").fetchall()


def test_cli_source_navigation_preserves_capture_without_replay(workspace, tmp_path, server):
    store, registry, executor = chain(workspace, tmp_path, server,
        select={'col': 'amount'}, predicate={'type': 'threshold', 'op': '>=', 'value': 40})
    original = source_events(store)
    binding = store.load_state('declaration:r1.1')['references'][0]['version_id']
    handle = registry.resolver.handle(binding)
    raw = registry.resolver.content(binding)[1]
    server.conn.execute('UPDATE ledger SET amount=1')
    server.conn.commit()
    for command in ('pf depend r2 --detail', 'pf depend r2 --detail --history',
                    f'pf rdepend {handle} --detail', f'pf log {handle}', f'pf show {handle} --full'):
        output = executor.execute('bash', {'command': command})
        assert not output.startswith('Error:'), output
        assert source_events(store) == original, command
        assert 'PREDICATE FAILS' not in output and 'predicate holds' not in output
        assert 'Checked day' not in output and 'Pending review' not in output
    assert registry.resolver.content(binding)[1] == raw
    page = relation_details(executor, 'pf depend r2 --depth 2 --detail')
    assert any(row['source'] and row['source']['ref'].get('cite') == handle for row in page['items'])
    assert all('check' not in row for row in page['items'])
    assert not store.load_state('pf_review')
    send(store, executor.execute('text_list', {'raw': True}))
    revised = call(registry, 'revise', record='r2', reason='Reviewed saved source', text='Revised claim')
    assert revised['version'] == 'r2.2'
    history = executor.execute('bash', {'command': 'pf show r2.1 --full'})
    assert 'Keep the plan' in history


def test_legacy_findings_and_cursor_checks_are_not_delivered(workspace, tmp_path, server):
    store, registry, executor = chain(workspace, tmp_path, server)
    with closing(store.connect()) as conn:
        cutoff = conn.execute('SELECT coalesce(max(rowid), 0) FROM versions').fetchone()[0]
    cursor = 'c999'
    cursors = store.load_state('pf_cursors') or {}
    cursors[cursor] = dict(operation='pf_dependencies', values=dict(
        target={'record': 'r2'}, detail=True, limit=1, depth=2,
        include_execution=False, purpose='historical_only'), offset=1, cutoff=cutoff,
        check_state='pf_check:legacy')
    store.save_state('pf_cursors', cursors)
    version = store.load_state('declaration:r1.1')['version_id']
    store.save_state('pf_review', {'pending': {version: dict(first_day=7, last_day=7,
        reason={'target': None, 'reason': 'OLD_FINDING', 'check': None})}, 'ended': []})
    store.save_state('pf_check:legacy', [{'stale': 'OLD_FINDING'}])
    executor.pf_queries = PFQueries(registry)
    listed = json.loads(executor.execute('text_list', {'raw': True}))
    assert len(listed['records']) == 2
    assert 'checks' not in listed and 'OLD_FINDING' not in json.dumps(listed)
    continued = executor.execute('bash', {'command': f'pf more {cursor}'})
    assert not continued.startswith('Error:') and '[exit code:' not in continued, continued
    assert 'OLD_FINDING' not in continued
    assert executor.pf_queries.last_answer['items']
    assert 'check' not in executor.pf_queries.last_answer['items'][0]
    assert executor.pf_queries.last_answer['items'][0]['target']['version'] == registry.resolver.handle(
        store.load_state('declaration:r1.1')['references'][0]['version_id'])


def test_packed_public_reads_remain_explicit_and_refresh_endpoint_is_gone(offline_runner):
    runner = offline_runner(text_registration='pf')
    command = ('./novamind-operation python-c "import novamind_api as nm; '
               "print(nm.vars.current_day); print(nm.query('SELECT COUNT(*) AS n FROM ledger'))\"")
    output = runner._execute_tool('bash', {'command': command})
    assert '[exit code:' not in output, output
    handle = re.search(r'\[pf: (cmd\d+@v\d+)', output)[1]
    store = runner.evidence_store
    original = source_events(store)
    assert len(original) >= 2
    for cmd in (f'pf depend {handle} --detail', f'pf log {handle}', f'pf show {handle} --full'):
        result = runner._execute_tool('bash', {'command': cmd})
        assert '[exit code:' not in result, result
        assert source_events(store) == original
    with pytest.raises(urllib.error.HTTPError) as exc:
        runner._http_post('/pf-refresh', {'versions': [], 'parent': 'old-check'})
    assert exc.value.code == 404
    before_rerun = source_events(store)
    result = runner._execute_tool('bash', {'command': command})
    assert '[exit code:' not in result, result
    assert len(source_events(store)) == len(before_rerun) + 2
