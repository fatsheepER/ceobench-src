"""Prefix registration matches the saved snapshot without future-week rebinding."""
from contextlib import closing

import pytest

from saas_bench.agents.bash_agent.run_test import BashAgentRunner
from saas_bench.pf_queries import PFQueries
from saas_bench.text_registry import TextRegistry
from test_text_registry import workspace, captured, declaration, send, git, call
from test_preflight_integration import offline_runner, packed_public, advance
from test_pf_queries import relation_details


def close_week(workspace, registry):
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.agent_workspace = workspace
    runner.tool_executor = type('Executor', (), {'text_registry': registry})()
    runner._commit_weeks_up_to(14)


@pytest.mark.parametrize('early_read', [False, True])
def test_snapshot_binding_and_inherited_revisions_do_not_change_at_week_close(workspace, tmp_path, early_read):
    store, registry, executor = captured(workspace, tmp_path, 'prefix')
    if early_read:
        send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration({'path': 'evidence.json'}))
    call(registry, 'revise', record='r1', text='Refined plan', reason='wording')
    public = registry.path.read_bytes()
    before = {revision: store.load_state('declaration:' + revision) for revision in ('r1.1', 'r1.2')}
    for value in before.values():
        b = value['references'][0]
        assert b['status'] == ('resolved' if early_read else 'unknown')
        assert git(workspace, 'show', b['git_commit'] + ':evidence.json') == '{"n":7}'
    executor.execute('write_file', {'path': 'evidence.json', 'content': '{"n":99}'})
    registry = TextRegistry(workspace, 'prefix', store, sim_day=lambda: 14)
    close_week(workspace, registry)
    assert registry.path.read_bytes() == public
    assert not store.load_state('weekly_declarations')
    for revision, value in before.items():
        assert store.load_state('declaration:' + revision) == value
    with closing(store.connect()) as conn:
        sequence = store.sequence(conn)
    close_week(workspace, registry)
    with closing(store.connect()) as conn:
        assert store.sequence(conn) == sequence
    q = PFQueries(TextRegistry(workspace, 'pf', store))
    q._index(None)
    for revision, value in before.items():
        refs = [e for e in q.outgoing[value['version_id']] if e['origin'] == 'agent_declaration']
        assert len(refs) == 1 and refs[0]['missing'] is not early_read


def test_later_capture_cannot_retroactively_claim_registration_time_delivery(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path, 'prefix')
    call(registry, 'create', **declaration({'path': 'evidence.json'}))
    before = store.load_state('declaration:r1.1')
    assert before['references'][0]['status'] == 'unknown'
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    close_week(workspace, registry)
    assert store.load_state('declaration:r1.1') == before


@pytest.mark.parametrize('selected', [False, True])
def test_snapshot_keeps_actual_registration_read_after_overwrite(workspace, tmp_path, selected):
    store, registry, executor = captured(workspace, tmp_path, 'prefix')
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    ref = dict(evidence={'path': 'evidence.json'}, purpose='current')
    if selected:
        ref['select'] = {'path': '/n'}
    call(registry, 'create', **declaration(references=[ref]))
    old = store.load_state('declaration:r1.1')['references'][0]
    executor.execute('write_file', {'path': 'evidence.json', 'content': '{"n":99}'})
    close_week(workspace, registry)
    b = store.load_state('declaration:r1.1')['references'][0]
    assert b == old and b['status'] == 'resolved' and b['git_content_matches']
    assert store.get_content(b['version_id'])[1] == b'{"n":7}'


@pytest.mark.parametrize('problem', ['uncaptured', 'deleted', 'selected_changed'])
def test_post_registration_changes_do_not_rebind_sources(workspace, tmp_path, problem):
    store, registry, executor = captured(workspace, tmp_path, 'prefix')
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    ref = dict(evidence={'path': 'evidence.json'}, purpose='current')
    if problem == 'selected_changed':
        ref['select'] = {'path': '/n'}
    call(registry, 'create', **declaration(references=[ref]))
    before = store.load_state('declaration:r1.1')
    if problem == 'deleted':
        (workspace / 'evidence.json').unlink()
    elif problem == 'uncaptured':
        (workspace / 'evidence.json').write_text('{"n":99}')
    else:
        executor.execute('write_file', {'path': 'evidence.json', 'content': '{"n":99}'})
    close_week(workspace, registry)
    assert store.load_state('declaration:r1.1') == before
    assert not store.load_state('weekly_declarations')


def test_prefix_forks_share_registration_time_public_bytes_and_exact_evidence(offline_runner, tmp_path):
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
            assert child.evidence_store.get_content(b['version_id'])[1] == b'{"n":1}'
            graph = relation_details(child.tool_executor, 'pf depend r1 --detail')
            source = graph['items'][0]['source']
            assert graph['items'][0]['target']['ref']['cite'] == 'r1.1'
            assert 'binding' not in source and 'sha256' not in source
            owner_path, commit = source['ref']['cite'].rsplit('@', 1)
            assert owner_path == 'ceo:facts.json'
            assert commit == git(child.agent_workspace, 'rev-parse', '--short=7', b['git_commit'])
            assert git(child.agent_workspace, 'rev-parse', commit) == b['git_commit']
            assert child._execute_tool('bash', {'command': 'git show ' + commit + ':facts.json'}) == '{"n":1}'
            assert child.evidence_store.load_state('declaration:r1.1')['references'][0] == b
            assert child.tool_executor.text_registry.path.read_bytes() == public
        else:
            assert child.evidence_store is None
