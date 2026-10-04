"""Prioritize observed changes without inferring that a business claim is false."""
import json

import pytest

from saas_bench.pf_stale import compare
from test_text_registry import workspace, captured, call, declaration
from test_pf_stale import record_sql, cite


def table(rows, columns=('n',), **extra):
    return dict(success=True, columns=list(columns), rows=rows, row_count=len(rows), **extra)


@pytest.mark.parametrize('old,new,kind', [
    (table([{'n': 1}]), table([{'n': 2}, {'n': 1}]), 'append_only'),
    (table([{'n': 1}, {'n': 1}]), table([{'n': 1}, {'n': 2}]), 'content_changed'),
    (table([{'n': 1}]), table([{'n': 1.0}, {'n': 2}]), 'content_changed'),
    (table([{'n': 1}]), table([]), 'content_changed'),
    (table([{'n': 1}]), table([{'n': 1, 'm': 2}], ('n', 'm')), 'content_changed'),
])
def test_row_additions_require_unchanged_typed_rows_and_duplicate_counts(workspace, tmp_path, old, new, kind):
    store, registry, _ = captured(workspace, tmp_path)
    a = record_sql(store, 'SELECT n FROM sample', old)
    b = record_sql(store, 'SELECT n FROM sample', new)
    changed, holds, reason, change_kind = compare(registry.resolver, a, b, {}, 'query')
    assert changed and not holds and reason is None and change_kind == kind


def test_truncated_query_cannot_be_classified_as_only_additions(workspace, tmp_path):
    store, registry, _ = captured(workspace, tmp_path)
    a = record_sql(store, 'SELECT n FROM sample', table([{'n': 1}], truncated=True))
    b = record_sql(store, 'SELECT n FROM sample', table([{'n': 1}, {'n': 2}], truncated=True))
    with pytest.raises(ValueError, match='source_truncated'):
        compare(registry.resolver, a, b, {}, 'query')


def output_with_sources(store, registry, executor, values):
    event = store.begin_event('bash', {'command': 'python report.py'})
    for sql, body in values.items():
        record_sql(store, sql, body, parent=event)
    version = store.version(event, 'output', 'Stored report', layer='tool_return')
    store.complete(event)
    cite(store, registry, executor, version)


def test_mixed_upstream_changes_choose_old_data_change_and_keep_dated_reminder(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    data = {'SELECT n FROM daily': table([{'n': 1}]), 'SELECT n FROM estimate': table([{'n': 9}])}
    output_with_sources(store, registry, executor, data)
    data['SELECT n FROM daily'] = table([{'n': 1}, {'n': 2}])
    data['SELECT n FROM estimate'] = table([{'n': 16}])
    calls = []
    def refresh(versions, parent):
        calls.append(versions)
        result = {}
        for v in versions:
            sql = store.read_event(store.get_content(v)[0]['created_by_event'])['query_definition'][5]
            result[v] = record_sql(store, sql, data[sql], parent=parent)
        return result
    query = executor.pf_queries
    query.refresh = refresh
    first = query.weekly_check(7)
    assert 'r1.1' in first and '9→16' in first
    pending = next(iter(store.load_state('pf_review')['pending'].values()))
    assert pending['reason']['target']['sql'] == 'SELECT n FROM estimate'
    assert pending['reason']['check']['change_kind'] == 'content_changed'
    second = query.weekly_check(14)
    assert len(calls) == 1
    assert 'r1.1 (last verified day 7)' in second and '9→16' in second
    listed = json.loads(executor.execute('text_list', {'review': 'pending'}))
    assert listed['checks']['r1.1']['last_day'] == 7
    # Manual checking updates the saved reason as well as its date.
    data['SELECT n FROM estimate'] = table([{'n': 20}])
    query.registry.sim_day = lambda: 14
    executor.execute('pf_dependencies', {'target': {'record': 'r1'}, 'detail': True})
    updated = next(iter(store.load_state('pf_review')['pending'].values()))
    assert updated['last_day'] == 14 and '9→20' in updated['reason']['check']['summary']


def test_only_additions_stay_pending_without_content_reminders(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    sql = 'SELECT n FROM daily'
    output_with_sources(store, registry, executor, {sql: table([{'n': 1}])})
    query = executor.pf_queries
    query.refresh = lambda versions, parent: {v: record_sql(store, sql, table([{'n': 1}, {'n': 2}]), parent=parent) for v in versions}
    first = query.weekly_check(7)
    assert 'Only added rows' in first and 'Pending review: 1 text' in first
    query.refresh = lambda *_: pytest.fail('Pending source was rerun')
    second = query.weekly_check(14)
    assert '1 text with only added rows' in second
    assert 'last verified day' not in second
    assert next(iter(store.load_state('pf_review')['pending'].values()))['last_day'] == 7


def test_pending_content_examples_have_a_fixed_bound(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    for n in range(5):
        executor.execute('write_file', {'path': f'f{n}.txt', 'content': 'old'})
        call(registry, 'create', **declaration({'path': f'f{n}.txt'}))
        (workspace / f'f{n}.txt').write_text('new')
    executor.pf_queries.weekly_check(7)
    second = executor.pf_queries.weekly_check(14)
    assert second.count('last verified day 7') == 3
    assert '2 more texts have earlier content-change findings' in second
