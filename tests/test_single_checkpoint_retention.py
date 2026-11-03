from saas_bench.agents.bash_agent.run_test import BashAgentRunner
from saas_bench.run_state import file_hash, tree_hash
from saas_bench.legacy_privacy import PUBLIC_HISTORY_FORMAT
from saas_bench.public_refs import PublicRefs


def test_single_agent_restore_preserves_snapshots_and_previous_workspaces(tmp_path):
    root = tmp_path / 'new-attempt'
    snapshot = root / 'checkpoints' / ('a' * 32)
    saved_workspace = snapshot / 'agent_workspace'
    saved_workspace.mkdir(parents=True)
    (saved_workspace / 'MEMORY.md').write_text('saved memory')
    for name, contents in {'world.nmdb': 'saved world', 'session.json': '{}',
                           'server_state.json': '{}', 'manifest.json': '{}'}.items():
        (snapshot / name).write_text(contents)
    older = root / 'checkpoints' / ('b' * 32)
    older.mkdir()
    (older / 'checkpoint.json').write_text('older immutable generation')
    refs = PublicRefs(tmp_path / 'public-refs.sqlite')
    reference = refs.assign('operation', 'ceo', 'retained-operation')
    refs.snapshot(snapshot / 'public-refs.sqlite')
    (snapshot / 'private-agent').mkdir()
    checkpoint = dict(version=2, snapshot_id='a' * 32, session_id='session',
        public_history_format=PUBLIC_HISTORY_FORMAT,
        private_agent_sha256=tree_hash(snapshot / 'private-agent'),
        public_refs=dict(path=str(refs.path), snapshot='public-refs.sqlite'),
        files={name: file_hash(snapshot / name) for name in
               ('world.nmdb', 'session.json', 'server_state.json', 'manifest.json', 'public-refs.sqlite')},
        request_logs={}, workspace_sha256=tree_hash(saved_workspace))
    before = tree_hash(root / 'checkpoints', exact=True)
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.workspace_dir = root
    runner.agent_workspace = root / 'agent_workspace'
    runner.logs_dir = root / 'logs'
    runner.sql_evidence_config = None
    runner.agent_workspace.mkdir()
    (runner.agent_workspace / 'MEMORY.md').write_text('discarded memory')
    runner._restore_checkpoint_files(checkpoint)
    assert (runner.agent_workspace / 'MEMORY.md').read_text() == 'saved memory'
    previous = list(root.glob('previous-*'))
    assert len(previous) == 1
    assert (previous[0] / 'MEMORY.md').read_text() == 'discarded memory'
    runner._restore_checkpoint_files(checkpoint)
    assert len(list(root.glob('previous-*'))) == 2
    assert tree_hash(root / 'checkpoints', exact=True) == before
    assert runner._restored_public_refs.resolve('operation', reference) == 'retained-operation'
