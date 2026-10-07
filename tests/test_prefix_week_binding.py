"""Closing commits, private declarations and their fork/checkpoint boundary."""
from contextlib import closing

import pytest

from saas_bench.agents.bash_agent.run_test import BashAgentRunner
from saas_bench.pf_queries import PFQueries
from saas_bench.text_registry import TextRegistry
from test_text_registry import workspace, captured, declaration, send, git, call
from test_preflight_integration import offline_runner, packed_public, advance


def close_week(workspace, registry):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.agent_workspace = workspace
    runner.tool_executor = type('Executor', (), {'text_registry': registry})()
    runner._commit_weeks_up_to(14)


@pytest.mark.parametrize('early_read', [False, True])
def test_closing_commit_binds_later_write_and_inherited_revisions(workspace, tmp_path, early_read):
    store, registry, executor = captured(workspace, tmp_path, 'prefix')
    if early_read:
        send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration({'path': 'evidence.json'}))
    call(registry, 'revise', record='r1', text='Refined plan', reason='wording')
    public = registry.path.read_bytes()
    executor.execute('write_file', {'path': 'evidence.json', 'content': '{"n":99}'})
    with pytest.raises(RuntimeError, match='incomplete'):
        registry.assert_week_finalized(14)
    # A mid-week restore retains deferred declarations.
    registry = TextRegistry(workspace, 'prefix', store, sim_day=lambda: 14)
    close_week(workspace, registry)
    assert registry.path.read_bytes() == public
    assert not store.load_state('weekly_declarations')
    for revision in ('r1.1', 'r1.2'):
        b = store.load_state('declaration:' + revision)['references'][0]
        assert b['status'] == 'resolved' and b['git_content_matches']
        assert store.get_content(b['version_id'])[1] == git(workspace, 'show', 'HEAD:evidence.json').encode()
        assert b['basis'] == 'committed_bytes' and b['reading_scope'] == 'not_in_request'
        assert not b['delivered_in'] and 'authored_by' not in b
    with closing(store.connect()) as conn:
        before = store.sequence(conn)
    close_week(workspace, registry)
    with closing(store.connect()) as conn:
        assert store.sequence(conn) == before
    q = PFQueries(TextRegistry(workspace, 'pf', store))
    q._index(None)
    for revision in ('r1.1', 'r1.2'):
        source = store.load_state('declaration:' + revision)['version_id']
        refs = [e for e in q.outgoing[source] if e['origin'] == 'agent_declaration']
        assert len(refs) == 1 and not refs[0]['missing']


def test_captured_but_unread_document_can_match_committed_bytes(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path, 'prefix')
    send(store, executor.execute('bash', {'command': 'cat evidence.json'}))
    call(registry, 'create', **declaration({'path': 'evidence.json'}))
    assert store.load_state('declaration:r1.1')['references'][0]['status'] == 'unknown'
    close_week(workspace, registry)
    b = store.load_state('declaration:r1.1')['references'][0]
    assert b['status'] == 'resolved' and b['basis'] == 'committed_bytes'
    assert b['reading_scope'] == 'not_in_request'


def test_failed_finalization_publishes_no_partial_binding(workspace, tmp_path, monkeypatch):
    store, registry, executor = captured(workspace, tmp_path, 'prefix')
    call(registry, 'create', **declaration({'path': 'evidence.json'}))
    executor.execute('write_file', {'path': 'evidence.json', 'content': '{"n":99}'})
    before = store.load_state('declaration:r1.1')
    complete = store.complete
    def fail_binding(event, **facts):
        if store.read_event(event)['request']['kind'] == 'prefix_week_binding':
            raise OSError('injected binding failure')
        return complete(event, **facts)
    monkeypatch.setattr(store, 'complete', fail_binding)
    with pytest.raises(OSError, match='injected binding failure'):
        close_week(workspace, registry)
    assert store.latest_version('declaration:r1.1')[0] is None
    assert store.load_state('declaration:r1.1') == before
    assert store.load_state('weekly_declarations') == ['r1.1']
    with pytest.raises(RuntimeError, match='incomplete'):
        registry.assert_week_finalized(14)
    with pytest.raises(RuntimeError, match='Unconfirmed'):
        store.assert_healthy()


@pytest.mark.parametrize('selected', [False, True])
def test_closing_commit_keeps_actual_registration_read(workspace, tmp_path, selected):
    store, registry, executor = captured(workspace, tmp_path, 'prefix')
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    ref = dict(evidence={'path': 'evidence.json'}, purpose='current')
    if selected:
        ref['select'] = {'path': '/n'}
    call(registry, 'create', **declaration(references=[ref]))
    old = store.load_state('declaration:r1.1')['references'][0]
    close_week(workspace, registry)
    b = store.load_state('declaration:r1.1')['references'][0]
    assert b['status'] == 'resolved' and b['delivered_in'] == old['delivered_in']


@pytest.mark.parametrize('problem', ['uncaptured', 'deleted', 'selected_changed'])
def test_closing_commit_records_specific_gaps(workspace, tmp_path, problem):
    store, registry, executor = captured(workspace, tmp_path, 'prefix')
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    ref = dict(evidence={'path': 'evidence.json'}, purpose='current')
    if problem == 'selected_changed':
        ref['select'] = {'path': '/n'}
    call(registry, 'create', **declaration(references=[ref]))
    if problem == 'deleted':
        (workspace / 'evidence.json').unlink()
    elif problem == 'uncaptured':
        (workspace / 'evidence.json').write_text('{"n":99}')
    else:
        executor.execute('write_file', {'path': 'evidence.json', 'content': '{"n":99}'})
    close_week(workspace, registry)
    b = store.load_state('declaration:r1.1')['references'][0]
    assert b['status'] == 'unknown' and b['reason']
    assert 'this request' not in b['reason']
    assert not store.load_state('weekly_declarations')


def test_finalized_prefix_forks_share_public_bytes_and_exact_evidence(offline_runner, tmp_path):
    from saas_bench.run_state import clone_sql_run
    runner = offline_runner(text_registration='prefix')
    runner.agent.current_day = 0
    runner._execute_tool('write_file', {'path': 'facts.json', 'content': '{"n":1}'})
    runner._execute_tool('text_create', declaration({'path': 'facts.json'}))
    runner._execute_tool('write_file', {'path': 'facts.json', 'content': '{"n":2}'})
    assert advance(runner)['success']
    runner._commit_weeks_up_to(7)
    runner._save_checkpoint(7)
    public = runner.tool_executor.text_registry.path.read_bytes()
    for mode in ('git', 'pf'):
        child = offline_runner(clone_sql_run(runner.workspace_dir, tmp_path / mode, mode, text_registration=mode))
        assert child.tool_executor.text_registry.path.read_bytes() == public
        if mode == 'pf':
            b = child.evidence_store.load_state('declaration:r1.1')['references'][0]
            assert child.evidence_store.get_content(b['version_id'])[1] == b'{"n":2}'
            assert 'unavailable' not in child.tool_executor.weekly_check(7)
        else:
            assert child.evidence_store is None
