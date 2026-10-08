"""Compact PF returns and the week-start check of registered texts."""
import json
import re

import pytest

from saas_bench.pf_refresh import refresh
from saas_bench.public_sql import execute_query
from saas_bench.text_registry import TextRegistry
from test_text_registry import workspace, captured, call, declaration, git, send
from test_public_sql import server
from test_pf_stale import chain, cite, record_sql
from test_preflight_integration import offline_runner, packed_public
from test_stage5_prep import fake_weeks

CLOCK = re.compile(r'\d{4}-\d\d-\d\dT\d\d:\d\d')


def test_long_chain_reports_failures_and_explicit_check_coverage(workspace, tmp_path, server):
    store, registry, executor = chain(workspace, tmp_path, server, select={'col': 'amount'},
        predicate={'type': 'threshold', 'op': '>=', 'value': 40})
    for i in range(3, 7):
        call(registry, 'create', **declaration({'record': f'r{i-1}'}))
    server.conn.execute('UPDATE ledger SET amount=39')
    server.conn.commit()
    weekly = executor.weekly_check(14)
    assert 'Checked this week: 6 texts; 6 with changed evidence:' in weekly, weekly
    assert 'r6.1 (day 7)' in weekly and 'PREDICATE FAILS' in weekly
    for suffix in ('', ' --detail'):
        limited = executor.execute('bash', {'command': 'pf depend r6' + suffix})
        assert 'depth limit' in limited and 'r4.1' in limited, limited
        assert 'pf depend r4.1' in limited
    historical = call(registry, 'create', **declaration(references=[dict(
        evidence={'record': 'r6'}, purpose='historical_only')]))
    assert historical['version'] not in executor.weekly_check(14)
    server.conn.execute('UPDATE ledger SET amount=45')
    server.conn.commit()
    assert 'Checked this week: 6 texts; cited files and texts are unchanged and no predicate fails.' in executor.weekly_check(14)
    server.conn.execute('DELETE FROM ledger')
    server.conn.commit()
    unknown = executor.weekly_check(14)
    assert 'Checked this week: 6 texts; 6 with changed evidence:' in unknown and 'could not be checked' in unknown
    # Old revisions still support current descendants even after their own text is retired.
    for i in range(1, 6):
        call(registry, 'retire', record=f'r{i}', reason='Kept only as an earlier dependency')
    assert 'Checked this week: 1 text; 1 with changed evidence:' in executor.weekly_check(14)


def test_shared_dependency_failure_is_reported_on_each_declared_path(workspace, tmp_path, server):
    store, registry, executor = chain(workspace, tmp_path, server, select={'col': 'amount'},
        predicate={'type': 'threshold', 'op': '>=', 'value': 40})
    call(registry, 'create', **declaration(references=[
        dict(cite='r2'), dict(cite='r1')]))
    server.conn.execute('UPDATE ledger SET amount=39')
    server.conn.commit()
    result = executor.pf_queries.execute('pf_dependencies', dict(target={'record': 'r3'}, depth=3))
    assert result.count('PREDICATE FAILS') == 2, result


def test_compact_lines_carry_day_what_and_value_without_clock_time(workspace, tmp_path, server):
    store, registry, executor = chain(workspace, tmp_path, server, select={'col': 'amount'},
        predicate={'type': 'threshold', 'op': '>=', 'value': 40}, note='Keep at least forty')
    server.conn.execute('UPDATE ledger SET amount=39')
    server.conn.commit()
    compact = executor.execute('pf_dependencies', {'target': {'record': 'r2'}})
    lines = compact.splitlines()
    assert lines[0].startswith('r2.1 · day 7 · text r2.1 (active): "Keep the plan"')
    # One line per declared reference; the failed threshold shows the value then and now.
    assert len([l for l in lines if re.match(r'\d+\. ', l)]) == 1
    assert 'cites' in lines[2] and 'text r1.1' in lines[2]
    assert '1 of 1 underlying sources PREDICATE FAILS on SQL: SELECT amount FROM ledger: >= 40: now 39 (was 42)' in lines[2]
    detail = executor.execute('pf_dependencies', {'target': {'record': 'r2'}, 'detail': True})
    assert re.search(r'PREDICATE FAILS \(now query\d+@v\d+\): >= 40: now 39', detail) and 'note: Keep at least forty' in detail
    assert 'SQL (1 rows): SELECT amount FROM ledger' in detail
    for text in (compact, detail, executor.execute('pf_search', {'object': {'kind': 'plan', 'id': 'B'}, 'detail': True}),
                 executor.execute('pf_read', {'target': {'record': 'r1'}, 'mode': 'history', 'detail': True}),
                 executor.execute('pf_read', {'target': {'record': 'r1'}}), executor.execute('text_list', {}),
                 registry.path.read_text()):
        assert not CLOCK.search(text), text
    header = json.loads(executor.execute('pf_read', {'target': {'record': 'r1'}}).split('\n', 1)[0])
    assert set(header) == {'delivery', 'next_cursor', 'range', 'target', 'total_chars'}
    assert set(header['target']) == {'version', 'day', 'what', 'owner'}
    assert header['target']['owner'] == 'ceo'


def test_search_and_history_are_newest_first_with_one_optional_detail_line(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    for i in range(3):
        call(registry, 'create', **declaration(text=f'plan {i}'))
    lines = executor.execute('pf_search', {'object': {'kind': 'plan', 'id': 'B'}, 'limit': 2, 'all': True}).splitlines()
    assert lines[0] == 'pf search B: 3 saved, newest first; showing 1–2.'
    assert '"plan 2"' in lines[1] and '"plan 1"' in lines[2] and '1 more older: pf more c1.' in lines[3]
    detail = executor.execute('pf_search', {'object': {'kind': 'plan', 'id': 'B'}, 'detail': True, 'all': True}).splitlines()
    assert len(detail) == 1 + 3 * 2 + 1 and detail[2].startswith('    B declared')
    assert executor.execute('pf_read', {'target': {'record': 'r1'}, 'detail': True}).startswith('Error:')


def test_query_changes_name_the_rows_that_differ(workspace, tmp_path, server):
    store, registry, executor = captured(workspace, tmp_path)
    server.sql_evidence = store
    executor.pf_queries.refresh = lambda versions, parent: refresh(server, versions, parent)
    sql = 'SELECT category, SUM(amount) AS total FROM ledger GROUP BY category'
    cite(store, registry, executor, record_sql(store, sql, execute_query(server, sql)))
    server.conn.execute("UPDATE ledger SET amount=amount+5")
    server.conn.commit()
    text = executor.execute('pf_dependencies', {'target': {'record': 'r1'}})
    assert re.search(r'cites (query\d+)@v1 · .* changed \(now \1@v2\): 1 of 1 rows differ; category="operations": total \d+→\d+', text), text
    # The diff hint names the two versions just compared.
    assert re.search(r'See what changed: pf diff (query\d+)@v1 \1@v2\.', text), text
    # The rerun is a diff target, not part of the agent's own history listing.
    history = executor.execute('pf_read', {'target': {'sql': sql}, 'mode': 'history'})
    assert history.startswith('Versions of SQL: ') and ': 1 saved' in history and 'rerun' not in history


def test_weekly_check_reruns_citations_and_lists_only_changed_texts(workspace, tmp_path, server):
    store, registry, executor = chain(workspace, tmp_path, server, select={'col': 'amount'},
        predicate={'type': 'threshold', 'op': '>=', 'value': 40})
    call(registry, 'create', **declaration(text='Unrelated note'))  # unknown evidence only: not checked
    server.conn.execute('UPDATE ledger SET amount=45')
    server.conn.commit()
    before = server.conn.serialize()
    quiet = executor.weekly_check(14)
    assert server.conn.serialize() == before
    assert quiet == ('=== Check of your registered texts (day 14) ===\n'
                     'Registered texts: 3 active; 3 within their applies window.\n'
                     'Checked this week: 2 texts; cited files and texts are unchanged and no predicate fails.')
    server.conn.execute('UPDATE ledger SET amount=39')
    server.conn.commit()
    loud = executor.weekly_check(21)
    assert 'Checked this week: 2 texts; 2 with changed evidence:' in loud and 'r1.1 (day 7): "Keep the plan"' in loud
    assert re.search(r'PREDICATE FAILS \(now query\d+@v\d+\): >= 40: now 39 \(was 42\)', loud), loud
    # r2 cites r1, whose predicate failed: the failure reaches it through the declared chain.
    assert 'r2.1 (day 7)' in loud and loud.endswith('Details: pf depend r2.')
    assert not CLOCK.search(loud)
    # The digest is saved with its own origin so request source mappings stay complete.
    assert loud.origins and store.get_content(loud.origins[0]['version_id'])[1].decode() == str(loud)
    # With checks disabled (the ablation) there is no week-start push.
    executor.pf_queries.stale_checks = False
    assert executor.weekly_check(21) is None


@pytest.mark.parametrize('mode', ['git', 'prefix'])
def test_git_weekly_check_compares_cited_commits_with_current_files(workspace, tmp_path, mode):
    store, registry, _ = captured(workspace, tmp_path, mode) if mode == 'prefix' else (
        None, TextRegistry(workspace, 'git', sim_day=lambda: 7), None)
    call(registry, 'create', **declaration({'path': 'evidence.json'}))
    call(registry, 'create', **declaration({'record': 'r1'}, text='Depends on r1'))
    git(workspace, 'add', '.')
    git(workspace, 'commit', '-qm', 'Week 2 (day 14) [week-2]')
    first = registry.weekly_check(14)
    assert first == '=== Check of your registered texts (day 14) ===\nChecked this week: 2 texts; cited files and texts are unchanged.'
    (workspace / 'evidence.json').write_text('{"n":9}')
    call(registry, 'revise', record='r1', reason='Updated')
    second = registry.weekly_check(21)
    week2 = git(workspace, 'rev-parse', 'HEAD')[:7]
    assert (f'evidence.json@week-2: the current file differs (1 values differ; n 7→9); '
            f'git diff {week2} -- evidence.json') in second
    assert 'text r1.1: revised to r1.2' in second
    for word in ('PF', 'pf_', 'vN', 'compare'):
        assert word not in second, word
    if mode == 'prefix':  # Prefix output is the Git group's, word for word.
        baseline = TextRegistry(workspace, 'git', sim_day=lambda: 7)
        assert baseline.weekly_check(21) == second


@pytest.mark.parametrize('mode', ['off', 'prefix', 'pf'])
def test_runner_puts_the_weekly_check_after_the_new_week_dashboard(offline_runner, monkeypatch, mode):
    runner = offline_runner(text_registration=mode, stop_after_day=7)
    if mode != 'off':
        runner.agent.current_day = 0
        runner._execute_tool('text_create', declaration(applies_at={'start_day': 0}))
        runner._execute_tool('text_create', declaration({'record': 'r1'}, text='Depends on r1', applies_at={'start_day': 0}))
        runner._execute_tool('text_revise', dict(record='r1', reason='Corrected'))
        if mode == 'pf':  # A cited query output is rerun through the host's refresh endpoint.
            output = runner._execute_tool('bash', {'command': './novamind-operation query "SELECT COUNT(*) AS n FROM ledger"'})
            send(runner.evidence_store, output)
            handle = re.search(r'\[pf: (cmd\d+@v\d+)', output).group(1)
            runner._execute_tool('text_create', declaration({'version': handle}, text='Ledger size', applies_at={'start_day': 0}))
        runner.agent.current_day = -1
    requests = fake_weeks(runner, monkeypatch)
    assert runner.run(verbose=False)['outcome'] == 'stopped'
    first = requests[0]['messages'][1]['content']
    assert first.startswith('=== Week 0 Dashboard')
    if mode == 'off':
        assert 'Weekly check' not in first
        return
    head, check = first.split('\n\n=== Check of your registered texts (day 0) ===\n')
    assert 'r2.1 (day 0): "Depends on r1"' in check
    assert 'revised to r1.2' in check
    if mode == 'pf':  # r3's cited query output is unchanged, so it is not listed.
        assert 'r3.1' not in check and 'refresh' not in check, check
