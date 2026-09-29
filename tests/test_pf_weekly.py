"""Compact PF returns, the week-start check of registered texts, and the threshold tip."""
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
    assert set(header['target']) == {'version', 'day', 'what'}


def test_search_and_history_are_newest_first_with_one_optional_detail_line(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    for i in range(3):
        call(registry, 'create', **declaration(text=f'plan {i}'))
    lines = executor.execute('pf_search', {'object': {'kind': 'plan', 'id': 'B'}, 'limit': 2}).splitlines()
    assert lines[0] == 'pf_search plan=B: 3 saved, newest first; showing 1–2.'
    assert '"plan 2"' in lines[1] and '"plan 1"' in lines[2] and '1 more older: {"cursor": "c1"}' in lines[3]
    detail = executor.execute('pf_search', {'object': {'kind': 'plan', 'id': 'B'}, 'detail': True}).splitlines()
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
    assert re.search(r'"baseline": \{"version": "(query\d+)@v1"\}, "target": \{"version": "\1@v2"\}', text), text
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
    assert quiet.startswith('=== Weekly check of your registered texts (day 14) ===')
    assert 'None of the cited evidence changed.' in quiet and 'Unchanged: r2.1, r1.1' in quiet
    server.conn.execute('UPDATE ledger SET amount=39')
    server.conn.commit()
    loud = executor.weekly_check(21)
    assert 'r1.1 (day 7): "Keep the plan" — 1 of 1 cited sources flagged' in loud
    assert re.search(r'PREDICATE FAILS \(now query\d+@v\d+\): >= 40: now 39 \(was 42\)', loud), loud
    assert 'r2.1 (day 7)' in loud and 'Details: pf_dependencies {"target": {"record": "rN"}}.' in loud
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
    assert 'None of the cited evidence changed.' in first and 'Unchanged: r2.1, r1.1' in first
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


def test_threshold_tip_is_pf_only_optional_and_once_a_week(workspace, tmp_path):
    week = [7]
    store, registry, executor = captured(workspace, tmp_path)
    registry.sim_day = lambda: week[0]
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    text = 'Keep B at $99 while S2 new B subscriptions stay >= 770; raise 15 -> 18 later'
    first = call(registry, 'create', **declaration({'path': 'evidence.json'}, text=text))
    assert '(">= 770")' in first['tip'] and 'Optional' in first['tip']
    assert 'tip' not in call(registry, 'create', **declaration({'path': 'evidence.json'}, text=text))
    week[0] = 14
    ref = dict(evidence={'path': 'evidence.json'}, purpose='current', select={'path': '/n'},
               predicate={'type': 'threshold', 'op': '>=', 'value': 5})
    assert 'tip' not in call(registry, 'create', **declaration(references=[ref], text=text))
    assert 'tip' not in call(registry, 'create', **declaration(text='Plain plan without a number'))
    assert 'tip' in call(registry, 'create', **declaration({'path': 'evidence.json'}, text=text))
    git_registry = TextRegistry(workspace, 'git', sim_day=lambda: 7)
    git_registry.path = tmp_path / 'git-registrations.json'
    assert 'tip' not in call(git_registry, 'create', **declaration({'path': 'evidence.json'}, text=text))


@pytest.mark.parametrize('mode', ['off', 'prefix', 'pf'])
def test_runner_puts_the_weekly_check_after_the_new_week_dashboard(offline_runner, monkeypatch, mode):
    runner = offline_runner(text_registration=mode, stop_after_day=7)
    if mode != 'off':
        runner.agent.current_day = 0
        runner._execute_tool('text_create', declaration())
        runner._execute_tool('text_create', declaration({'record': 'r1'}, text='Depends on r1'))
        runner._execute_tool('text_revise', dict(record='r1', reason='Corrected'))
        if mode == 'pf':  # A cited query output is rerun through the host's refresh endpoint.
            output = runner._execute_tool('bash', {'command': './novamind-operation query "SELECT COUNT(*) AS n FROM ledger"'})
            send(runner.evidence_store, output)
            handle = re.search(r'\[输出: (cmd\d+@v\d+)', output).group(1)
            runner._execute_tool('text_create', declaration({'version': handle}, text='Ledger size'))
        runner.agent.current_day = -1
    requests = fake_weeks(runner, monkeypatch)
    assert runner.run(verbose=False)['outcome'] == 'stopped'
    first = requests[0]['messages'][1]['content']
    assert first.startswith('=== Week 0 Dashboard')
    if mode == 'off':
        assert 'Weekly check' not in first
        return
    head, check = first.split('\n\n=== Weekly check of your registered texts (day 0) ===\n')
    assert 'r2.1 (day 0): "Depends on r1" — 1 of 1 cited sources flagged' in check
    assert 'revised to r1.2' in check
    if mode == 'pf':
        assert 'Unchanged: r3.1' in check and 'refresh' not in check, check
