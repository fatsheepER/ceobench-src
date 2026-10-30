import json
from pathlib import Path
import pytest
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.team_checkpoint import validate_snapshot
from test_preflight_team_checkpoint import make_team, dispose, clients, fast_simulator
from test_preflight_team_flow import raw


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_empty_ceo_script_pass_clears_results_and_preserves_recovery(tmp_path, mode):
    source = make_team(tmp_path / 'source', mode)
    restored = None
    try:
        status, response = raw(source, 'ceo', '/daily-scripts', {'name': 'same.py'}, 'DELETE')
        assert status == 200 and response['success']
        snapshot = source.checkpoint()
        operations = source.root / 'private/operations'
        before = {path.name: path.read_bytes() for path in operations.glob('*.json')}
        source.server.role_script_results['ceo'] = [{'status': 'failed'}]
        assert source._script_output('ceo') == ''
        assert source.server.role_script_results['ceo'] == []
        assert {path.name: path.read_bytes() for path in operations.glob('*.json')} == before
        restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored',
            client_factory=clients(), simulator_factory=fast_simulator)
        assert restored.server.tools.current_day == source.server.tools.current_day == 0
        assert restored.server.get_daily_scripts('ceo') == {}
        for runtime in (source, restored):
            assert runtime.run(stop_after_day=7).reason == 'observation_end', runtime.failure
        assert restored.server.tools.current_day == source.server.tools.current_day == 7
        sql = 'SELECT day,category,amount,note FROM ledger ORDER BY rowid'
        assert [tuple(row) for row in restored.server.conn.execute(sql)] == [
            tuple(row) for row in source.server.conn.execute(sql)]
    finally:
        if restored:
            dispose(restored)
        dispose(source)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_nonempty_ceo_script_pass_is_discarded_by_old_checkpoint_restore(tmp_path, mode):
    source = make_team(tmp_path / 'source', mode)
    try:
        snapshot = source.checkpoint()
        assert 'same.py' in source._script_output('ceo')
        operations = [json.loads(path.read_text()) for path in
            (source.root / 'private/operations').glob('*.json')]
        wrappers = [op for op in operations if op['kind'] == 'ceo_scripts']
        assert [{key: op[key] for key in ('kind', 'day', 'status')} for op in wrappers] == [
            dict(kind='ceo_scripts', day=0, status='completed')]
        assert wrappers[0]['wrapper_version'] == 1
        assert wrappers[0]['recovery']['classification'] == 'unproven'
        restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored',
            client_factory=clients(), simulator_factory=fast_simulator)
        try:
            assert not (restored.roles['ceo'].workspace / 'script-days.txt').exists()
        finally:
            dispose(restored)
    finally:
        dispose(source)


@pytest.mark.parametrize("mode", ["git", "pf"])
def test_completed_post_checkpoint_write_is_rolled_back(tmp_path, mode):
    source = make_team(tmp_path / "source", mode)
    restored = None
    try:
        snapshot = source.checkpoint()
        status, response = raw(source, "ceo", "/call", dict(tool="set_prices", args={"A":29}))
        assert status == 200 and response["success"]
        operations = [json.loads(p.read_text()) for p in (source.root/"private/operations").glob("*.json")]
        assert operations and all(op["status"] == "completed" for op in operations)
        validate_snapshot(snapshot)
        restored = MultiAgentRuntime.restore(snapshot, tmp_path/"restored", client_factory=clients(), simulator_factory=fast_simulator)
        assert restored.server.conn.execute("SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1").fetchone()[0] != 29
        dispose(restored)
        restored = None
        fresh = source.checkpoint()
        restored = MultiAgentRuntime.restore(fresh, tmp_path/"fresh", client_factory=clients(), simulator_factory=fast_simulator)
        assert restored.server.tools.current_day == source.server.tools.current_day
        assert restored.server.conn.execute("SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1").fetchone()[0] == 29
    finally:
        if restored: dispose(restored)
        dispose(source)


def test_real_restore_rolls_back_portable_link_targets(tmp_path):
    source = make_team(tmp_path / 'source', 'git')
    restored = None
    try:
        workspace = source.roles['ceo'].workspace
        for name in ('before.txt', 'after.txt'):
            (workspace / name).write_text('Same file contents.\n')
        link = workspace / 'portable-link'
        link.symlink_to('before.txt')
        snapshot = source.checkpoint()
        restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored',
            client_factory=clients(), simulator_factory=fast_simulator)
        assert (restored.roles['ceo'].workspace / 'portable-link').readlink() == Path('before.txt')
        dispose(restored)
        restored = None
        link.unlink()
        link.symlink_to('after.txt')
        restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'rolled-back',
            client_factory=clients(), simulator_factory=fast_simulator)
        assert (restored.roles['ceo'].workspace / 'portable-link').readlink() == Path('before.txt')
        assert link.readlink() == Path('after.txt')
    finally:
        if restored:
            dispose(restored)
        dispose(source)


from test_preflight_team_batch import frozen, changed_manifest


@pytest.mark.parametrize('tail_owner', ['snapshot_source', 'latest_attempt'])
def test_batch_registers_new_attempt_and_discards_post_checkpoint_operations(frozen, tmp_path, tail_owner):
    from saas_bench import team_batch as batch
    from saas_bench.run_state import write_json
    path = changed_manifest(frozen, tmp_path, lambda manifest: None)
    manifest = batch.validate_manifest(path)
    entry = manifest['runs'][0]
    with batch.run_lock(Path(entry['output_dir'])):
        first, record, _ = batch.register_attempt(path, manifest, entry)
        team = batch.make_team(manifest, entry, first)
        try:
            snapshot = team.checkpoint()
        finally:
            batch.close_team(team)
        record.update(status='paused', checkpoint=str(snapshot))
        write_json(first/'attempt.json', record)
        target = first
        if tail_owner == 'latest_attempt':
            target, later, _ = batch.register_attempt(path, manifest, entry, resume=True)
            later.update(status='paused', checkpoint=str(snapshot))
            write_json(target/'attempt.json', later)
        write_json(target/'runtime/private/operations/tail.json',
            dict(kind='api:research_market', day=7, status='completed'))
        before = batch.attempts(entry)
        original = {directory / 'attempt.json': (directory / 'attempt.json').read_bytes() for directory in before}
        created, resumed, source = batch.register_attempt(path, manifest, entry, resume=True)
        assert source == snapshot and resumed['source_checkpoint'] == str(snapshot)
        assert created.name == f'attempt-{len(before)+1:03}'
        assert {path: path.read_bytes() for path in original} == original
