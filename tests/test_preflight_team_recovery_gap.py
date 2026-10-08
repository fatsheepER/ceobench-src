import json
from pathlib import Path
import pytest
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.team_checkpoint import validate_snapshot, validate_recovery_source
from test_preflight_team_checkpoint import make_team, dispose, clients, fast_simulator
from test_preflight_team_flow import raw

@pytest.mark.parametrize("mode", ["git", "pf"])
def test_completed_post_checkpoint_write_cannot_be_replayed(tmp_path, mode):
    source = make_team(tmp_path / "source", mode)
    restored = None
    try:
        snapshot = source.checkpoint()
        status, response = raw(source, "ceo", "/call", dict(tool="set_prices", args={"A":29}))
        assert status == 200 and response["success"]
        operations = [json.loads(p.read_text()) for p in (source.root/"private/operations").glob("*.json")]
        assert operations and all(op["status"] == "completed" for op in operations)
        validate_snapshot(snapshot)
        with pytest.raises(RuntimeError, match="after checkpoint"):
            restored = MultiAgentRuntime.restore(snapshot, tmp_path/"restored", client_factory=clients(), simulator_factory=fast_simulator)
        assert not (tmp_path/"restored").exists()
        fresh = source.checkpoint()
        _, state = validate_snapshot(fresh)
        validate_recovery_source(fresh, state)
        restored = MultiAgentRuntime.restore(fresh, tmp_path/"fresh", client_factory=clients(), simulator_factory=fast_simulator)
        assert restored.server.tools.current_day == source.server.tools.current_day
        assert restored.server.conn.execute("SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1").fetchone()[0] == 29
    finally:
        if restored: dispose(restored)
        dispose(source)


@pytest.mark.parametrize("change", ["added_completed", "changed_completed", "missing", "latest_attempt_tail"])
def test_receipt_coverage_includes_completed_and_latest_attempt_operations(tmp_path, change):
    from saas_bench.run_state import write_json
    snapshot = tmp_path/"checkpoint"
    source = tmp_path/"source"
    latest = tmp_path/"latest"
    receipt = dict(kind="api:research_market", day=7, status="completed")
    for root in (snapshot, source, latest):
        write_json(root/"private/operations/covered.json", receipt)
    state = {"source_root":str(source)}
    validate_recovery_source(snapshot, state, current_root=latest)
    if change == "added_completed":
        write_json(source/"private/operations/new.json", receipt)
    elif change == "changed_completed":
        write_json(source/"private/operations/covered.json", dict(receipt, day=14))
    elif change == "missing":
        (source/"private/operations/covered.json").unlink()
    else:
        write_json(latest/"private/operations/new.json", receipt)
    with pytest.raises(RuntimeError, match="after checkpoint"):
        validate_recovery_source(snapshot, state, current_root=latest)


def test_unavailable_source_refuses_automatic_recovery(tmp_path):
    state = {"source_root":str(tmp_path/"missing-source")}
    with pytest.raises(RuntimeError, match="unavailable"):
        validate_recovery_source(tmp_path/"checkpoint", state)


from test_preflight_team_batch import frozen, changed_manifest


@pytest.mark.parametrize('tail_owner', ['snapshot_source', 'latest_attempt'])
def test_batch_refuses_uncovered_operations_before_new_attempt_registration(frozen, tmp_path, tail_owner):
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
        with pytest.raises(RuntimeError, match='after checkpoint'):
            batch.register_attempt(path, manifest, entry, resume=True)
        assert batch.attempts(entry) == before
        assert not (Path(entry['output_dir'])/f'attempt-{len(before)+1:03}').exists()
