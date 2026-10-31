from pathlib import Path

import pytest

from saas_bench import team_batch as batch
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.run_state import tree_hash, write_json
from test_preflight_team_checkpoint import clients, dispose, fast_simulator, make_team
from test_preflight_team_controller import controller
from test_preflight_team_observer import observer
from test_preflight_team_batch import changed_manifest, frozen


def test_finished_controller_honors_explicit_checkpoint_without_week(observer, controller, monkeypatch):
    import json
    from test_preflight_group_join import group_batch
    launches, children = group_batch(observer, controller, monkeypatch, paired=True)
    entry = observer.manifest['runs'][0]
    previous = Path(entry['output_dir']) / 'attempt-001/attempt.json'
    write_json(previous, dict(status='finished', attempt=1, day=28))
    original = previous.read_bytes()
    selected = previous.parent / 'runtime/checkpoints/operator-selected'
    monkeypatch.setenv('CEOBENCH_RECOVERY_CHECKPOINTS', json.dumps({entry['run_id']: str(selected)}))
    controller.module.supervise(controller.policy, resume=True)
    assert launches == ['s101-git', 's101-pf']
    assert all(child.poll() is not None for _, child in children)
    assert previous.read_bytes() == original


def test_explicit_d0_replay_excludes_all_previous_execution_time(tmp_path):
    first, second = [tmp_path / f'attempt-{number:03}' for number in (1, 2)]
    write_json(first / 'attempt.json', dict(status='finished', wall_seconds=100))
    write_json(second / 'attempt.json', dict(status='finished', wall_seconds=30,
        source_checkpoint=None, replay_week=1))
    result = batch.wall_time([first, second])
    assert result['wall_seconds'] == 30
    assert result['waste_wall_seconds'] == 100
    assert result['pending_wall_seconds'] == 0


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_missing_selected_conversation_is_not_replaced_with_fresh_context(tmp_path, mode):
    team = make_team(tmp_path / 'source', mode)
    try:
        snapshot = team.checkpoint()
    finally:
        dispose(team)
    missing = snapshot / 'private/growth/conversation.json'
    missing.unlink()
    original = tree_hash(snapshot, exact=True)
    restored = None
    try:
        with pytest.raises((ValueError, RuntimeError, FileNotFoundError), match='growth.*conversation|conversation.*growth'):
            restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored',
                client_factory=clients(), simulator_factory=fast_simulator)
    finally:
        if restored is not None:
            dispose(restored)
    assert tree_hash(snapshot, exact=True) == original


@pytest.mark.parametrize('publication', ['snapshot', 'pointer'])
def test_required_checkpoint_failure_pauses_before_the_next_model_request(frozen, tmp_path, monkeypatch, publication):
    from saas_bench import team_checkpoint
    path = changed_manifest(frozen, tmp_path, lambda manifest: [row.update(stop_after_day=7) for row in manifest['runs']])
    entry = next(row for row in batch.read(path)['runs'] if row['group'] == 'git')
    original_save = batch.save_checkpoint
    injected, previous = [], []

    def save(team, directory, record, boundary):
        if boundary != 'answers_ready' or injected:
            return original_save(team, directory, record, boundary)
        previous.append(record['checkpoint'])
        with monkeypatch.context() as fault:
            if publication == 'snapshot':
                def freeze(*args):
                    injected.append('world snapshot write failed')
                    raise OSError(injected[-1])
                fault.setattr(team_checkpoint, '_freeze', freeze)
            else:
                write = team_checkpoint.write_json
                def fail_pointer(target, value):
                    if Path(target) == team.root / 'checkpoint.json':
                        injected.append('checkpoint pointer write failed')
                        raise OSError(injected[-1])
                    return write(target, value)
                fault.setattr(team_checkpoint, 'write_json', fail_pointer)
            return original_save(team, directory, record, boundary)

    monkeypatch.setattr(batch, 'save_checkpoint', save)
    def simulator(conn, config, rng):
        sim = fast_simulator(conn, config, rng)
        sim.shock_manager = batch.simulator(conn, config, rng).shock_manager
        return sim
    result = batch.prepare(path, entry['run_id'], live=True,
        client_factory=clients(), simulator_factory=simulator)
    assert len(injected) == 1
    assert result['status'] == 'paused'
    assert result['hold'] and result['day'] == 0
    assert result['checkpoint'] == previous[0]
    attempt, = batch.attempts(entry)
    assert (attempt / 'HOLD').exists()
    assert injected[0] in result['hold_detail']


def test_truncated_usage_keeps_logical_duration_unknown(tmp_path):
    from saas_bench.team_usage import aggregate
    from test_preflight_team_usage import PRICES, body, log, receipts
    path = tmp_path / 'model-usage.jsonl'
    rows = receipts('known', 'ceo', body(10, 2), cost=.1)
    rows[-1]['timestamp'] = '2026-10-07T00:00:04+00:00'
    log(path, rows)
    with path.open('a') as output:
        output.write('{"event":"request","call_id":"unfinished')
    result = aggregate([path], PRICES)
    assert result['team']['wall_seconds'] is None
    assert result['roles']['ceo']['wall_seconds'] is None
    assert result['team']['wall_seconds_known'] == 4
    assert result['team']['wall_seconds_missing_calls']


@pytest.mark.parametrize('missing', ['role_workspace', 'role_week', 'conversation_field'])
def test_selected_checkpoint_requires_actual_recovery_state(tmp_path, missing):
    import shutil
    team = make_team(tmp_path / 'source', 'git')
    try:
        snapshot = team.checkpoint()
    finally:
        dispose(team)
    if missing == 'role_workspace':
        shutil.rmtree(snapshot / 'roles/growth')
        detail = 'roles/growth'
    elif missing == 'role_week':
        state = batch.read(snapshot / 'team.json')
        state.update(stage='ceo', phase='ceo')
        write_json(snapshot / 'team.json', state)
        detail = 'role_weeks'
    else:
        path = snapshot / 'private/growth/conversation.json'
        conversation = batch.read(path)
        conversation.pop('current_day')
        write_json(path, conversation)
        detail = 'current_day'
    original = tree_hash(snapshot, exact=True)
    restored = None
    try:
        with pytest.raises(ValueError, match=detail):
            restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored',
                client_factory=clients(), simulator_factory=fast_simulator)
    finally:
        if restored is not None:
            dispose(restored)
    assert tree_hash(snapshot, exact=True) == original


def test_pf_conversation_hash_drift_preserves_usable_saved_context(tmp_path, caplog):
    import json
    team = make_team(tmp_path / 'source', 'pf')
    try:
        team.roles['growth'].agent.add_input('SAVED-CONTEXT-MARKER', day=0)
        snapshot = team.checkpoint()
    finally:
        dispose(team)
    path = snapshot / 'private/growth/conversation.json'
    payload = batch.read(path)
    path.write_text(json.dumps(payload, indent=4))
    original = tree_hash(snapshot, exact=True)
    restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored',
        client_factory=clients(), simulator_factory=fast_simulator)
    try:
        assert any(message.content == 'SAVED-CONTEXT-MARKER'
                   for message in restored.roles['growth'].agent.conversation)
        assert 'checksum differs from private source state' in caplog.text
    finally:
        dispose(restored)
    assert tree_hash(snapshot, exact=True) == original


@pytest.mark.parametrize('selection', ['{unfinished', '[]', '{"git":null}', '{"git":""}'])
def test_invalid_explicit_selection_cannot_choose_another_checkpoint(tmp_path, monkeypatch, selection):
    from saas_bench.team_recovery import latest_checkpoint
    monkeypatch.setenv('CEOBENCH_RECOVERY_CHECKPOINTS', selection)
    entry = dict(output_dir=str(tmp_path), run_id='git')
    with pytest.raises(ValueError, match='CEOBENCH_RECOVERY_CHECKPOINTS|Requested checkpoint'):
        latest_checkpoint(entry)
    assert latest_checkpoint(entry, week=1) is None


@pytest.mark.parametrize('group', ['git', 'pf'])
def test_missing_actual_selected_state_holds_new_attempt_and_preserves_source(frozen, tmp_path, monkeypatch, group):
    import json
    path = changed_manifest(frozen, tmp_path, lambda manifest: None)
    entry = next(row for row in batch.read(path)['runs'] if row['group'] == group)
    first = batch.prepare(path, entry['run_id'], simulator_factory=batch.simulator)
    checkpoint = Path(first['checkpoint'])
    (checkpoint / 'private/growth/conversation.json').unlink()
    source_attempt, = batch.attempts(entry)
    original = tree_hash(source_attempt, exact=True)
    monkeypatch.setenv('CEOBENCH_RECOVERY_CHECKPOINTS', json.dumps({entry['run_id']: str(checkpoint)}))
    resumed = batch.prepare(path, entry['run_id'], resume=True, simulator_factory=batch.simulator)
    assert resumed['status'] == 'paused' and resumed['hold']
    assert resumed['reason'] == 'checkpoint_data_missing' and resumed['execution_started'] is False
    assert 'private/growth/conversation.json' in resumed['hold_detail']
    assert resumed['source_checkpoint'] == str(checkpoint)
    assert resumed['attempt'] == 2
    assert (batch.attempts(entry)[-1] / 'HOLD').exists()
    assert tree_hash(source_attempt, exact=True) == original


@pytest.mark.parametrize('missing', ['conversation', 'ceo_started', 'weekly_request_ids', 'observation'])
def test_initialized_role_requires_complete_saved_context(tmp_path, missing):
    team = make_team(tmp_path / 'source', 'git')
    try:
        team._rotate_sessions()
        team._role_input('ceo', 'SAVED-CEO-INPUT', day=0)
        team._stage = 'ceo'
        team._set_phase('ceo')
        team._week = dict(observation='SAVED-CEO-INPUT', ceo_started=True, directions='', weekly_request_ids=[])
        snapshot = team.checkpoint()
    finally:
        dispose(team)
    path = snapshot / ('private/ceo/conversation.json' if missing == 'conversation' else 'team.json')
    payload = batch.read(path)
    if missing == 'conversation':
        payload['conversation'] = []
    else:
        payload['week'].pop(missing)
    write_json(path, payload)
    restored = None
    try:
        with pytest.raises(ValueError, match=missing):
            restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored',
                client_factory=clients(), simulator_factory=fast_simulator)
    finally:
        if restored is not None:
            dispose(restored)
