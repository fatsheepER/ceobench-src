"""Public PF metadata preserves exact acquisitions without exposing archive identity."""
import json
import re
import shlex
import shutil
from pathlib import Path

from saas_bench.pf_queries import PFQueries
from saas_bench.text_registry import TextRegistry
from test_execution_capture import captured as http_captured
from test_pf_candidates import day, query
from test_public_sql import server
from test_receipt_outputs import model_receives
from test_text_registry import captured, workspace


def details(executor, command):
    _, page = query(executor, command)
    chunks = [page['detail_text']]
    while page['next_cursor']:
        _, page = query(executor, 'pf more ' + page['next_cursor'])
        chunks.append(page['detail_text'])
    return json.loads(''.join(chunks))


def test_public_candidate_aliases_keep_acquisitions_and_business_payload(http_captured):
    api, store, executor = http_captured
    registry = TextRegistry(executor.workspace_path, 'pf', store, sim_day=lambda: 7)
    executor.text_registry, executor.pf_queries = registry, PFQueries(registry)
    shutil.copy(Path(__file__).parents[1] / 'public/novamind-operation', executor.workspace_path)
    day(store, 7)
    command = './novamind-operation query ' + shlex.quote("SELECT 'business-session' AS session_id, '" + 'a' * 64 + "' AS sha256")
    for _ in range(2):
        executor.execute('bash', {'command': command})
    output, page = query(executor, 'pf candidates --history --kind sql')
    assert len(page['items']) == 2
    first, second = page['items']
    assert re.fullmatch(r'ceo:event[1-9][0-9]*', first['event_ref'])
    assert re.fullmatch(r'ceo:capture[1-9][0-9]*', first['content_ref'])
    assert first['event_ref'] != second['event_ref'] and first['content_ref'] != second['content_ref']
    assert first['version'] == second['version']
    assert 'sha256' not in first and 'sha256' not in first['capture']
    body = executor.execute('bash', {'command': 'pf show ' + first['content_ref'] + ' --full'})
    assert 'business-session' in body and 'a' * 64 in body
    exact = details(executor, 'pf candidates --history --event ' + first['event_ref'] + ' --detail')
    assert {item['event_ref'] for item in exact} == {first['event_ref']}
    assert exact[0]['request']['query']['sql'].startswith('SELECT ')
    assert 'run_id' not in json.dumps(exact) and 'data_source_id' not in json.dumps(exact)
    model_receives(store, executor.workspace_path, output)
    store.assert_healthy()


def test_note_search_points_to_produced_capture_without_body_range(workspace, tmp_path):
    store, _, executor = captured(workspace, tmp_path)
    executor.execute('bash', {'command': 'printf stable', 'note': 'first acquisition purpose'})
    executor.execute('bash', {'command': 'printf stable', 'note': 'second acquisition purpose'})
    output = executor.execute('bash', {'command': "pf search --text 'first acquisition purpose'"})
    page = executor.pf_queries.last_answer
    assert page['total'] == 1, output
    item = page['items'][0]
    assert item['match_basis'] == 'note' and item['note'][1] == 'first acquisition purpose'
    assert 'source_range' not in item
    assert re.fullmatch(r'ceo:capture[1-9][0-9]*', item['content_ref'])
    body = executor.execute('bash', {'command': 'pf show ' + item['content_ref'] + ' --full'})
    assert 'stable' in body and 'second acquisition purpose' not in body
    assert '.tool-notes.jsonl' not in output
    model_receives(store, executor.workspace_path, output)


def test_search_paging_navigation_precedes_results(workspace, tmp_path):
    _, _, executor = captured(workspace, tmp_path)
    for i in range(22):
        executor.execute('write_file', {'path': f'match{i}.txt', 'content': 'needle'})
    output = executor.execute('bash', {'command': 'pf search --text needle'})
    assert output.index('pf more ') < output.index('excerpt:'), output


def test_reference_ordinals_survive_rollback_without_rebinding(workspace, tmp_path):
    from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
    from saas_bench.sql_evidence import SQLEvidenceStore
    from test_sql_evidence import identity
    store, _, executor = captured(workspace, tmp_path)
    day(store, 7)
    executor.execute('write_file', {'path': 'stable.txt', 'content': 'kept'})
    _, first = query(executor, 'pf candidates --kind file --text stable.txt')
    kept = first['items'][0]
    checkpoint = tmp_path / 'checkpoint.sqlite'
    receipt = store.snapshot(checkpoint)
    executor.execute('write_file', {'path': 'stable.txt', 'content': 'discarded'})
    _, later = query(executor, 'pf candidates --kind file --text stable.txt')
    discarded = later['items'][0]
    (workspace / 'stable.txt').write_text('kept')
    restored = SQLEvidenceStore(checkpoint, identity('resumed', parent_branch='prefix', fork_seq=receipt['cutoff'], capture_scope='execution'))
    if hasattr(store, 'public_refs'):
        restored.public_refs = store.public_refs
    registry = TextRegistry(workspace, 'pf', restored, sim_day=lambda: 7)
    resumed = BashAgentToolExecutor(workspace, evidence_store=restored, text_registry=registry)
    assert 'kept' in resumed.execute('bash', {'command': 'pf show ' + kept['content_ref'] + ' --full'})
    resumed.execute('write_file', {'path': 'stable.txt', 'content': 'new future'})
    _, current = query(resumed, 'pf candidates --kind file --text stable.txt')
    replacement = current['items'][0]
    for field in ('version', 'content_ref', 'event_ref'):
        assert replacement[field] not in (kept[field], discarded[field])
    for handle in (discarded['version'], discarded['content_ref']):
        failed = resumed.execute('bash', {'command': 'pf show ' + handle + ' --full'})
        assert '[exit code:' in failed and 'new future' not in failed
    failed = resumed.execute('bash', {'command': 'pf candidates --history --event ' + discarded['event_ref']})
    assert '[exit code:' in failed and 'new future' not in failed
    assert 'new future' in resumed.execute('bash', {'command': 'pf show ' + replacement['content_ref'] + ' --full'})


def test_exact_aliases_work_for_peers_and_frozen_pages_recheck_permission(monkeypatch):
    from test_preflight_team_flow import final, make_team
    from test_preflight_team_evidence import write
    with make_team('pf', lambda *args: final('done'), 'public-reference-peers') as team:
        growth, ceo = team.roles['growth'], team.roles['ceo']
        day(growth.store, 0)
        write(team, 'growth', 'peer exact bytes', 'source.txt')
        _, page = query(growth.executor, 'pf candidates --history --kind file --text source.txt')
        item = page['items'][0]
        assert 'peer exact bytes' in ceo.executor.execute('bash', {'command': 'pf show ' + item['content_ref'] + ' --full'})
        _, peer = query(ceo.executor, 'pf candidates --history --role all --event ' + item['event_ref'])
        assert any(row['content_ref'] == item['content_ref'] for row in peer['items'])
        monkeypatch.setattr('saas_bench.role_policy.READABLE_ROLES', dict(ceo=('ceo',), growth=('growth',), ops_finance=('ops_finance',)))
        for command in ('pf show ' + item['content_ref'] + ' --full', 'pf more ' + peer['detail_cursor']):
            failed = ceo.executor.execute('bash', {'command': command})
            assert '[exit code:' in failed and 'peer exact bytes' not in failed
