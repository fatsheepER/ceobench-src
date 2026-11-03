"""Missing reference authority pauses restored workers before any model request."""
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

from saas_bench import team_batch as batch
from saas_bench.run_state import tree_hash, write_json
from test_checkpoint_driver import driver
from test_preflight_team_batch import frozen
from test_preflight_integration import offline_runner, packed_public
from test_text_registry import declaration, receipt


def break_authority(checkpoint, fault, snapshot):
    refs = checkpoint['public_refs']
    if fault == 'locator':
        checkpoint.pop('public_refs')
    elif fault == 'authority':
        path = Path(refs['path'])
        path.rename(path.with_suffix('.preserved'))
    else:
        with sqlite3.connect(Path(snapshot) / refs['snapshot']) as frozen:
            row = frozen.execute('SELECT kind,role,private_key FROM references_by_key LIMIT 1').fetchone()
        assert row
        with sqlite3.connect(refs['path']) as conn:
            conn.execute('UPDATE references_by_key SET private_key=? WHERE kind=? AND role=? AND private_key=?',
                         ('conflicting-private-identity', *row))


@pytest.mark.parametrize('group', ['git', 'pf'])
@pytest.mark.parametrize('fault', ['authority', 'locator', 'conflict'])
def test_driver_restores_selected_world_then_holds_without_model_requests(driver, group, fault):
    policy, entry = driver.policy('reference-history', group)
    _, original, first = driver.run(policy, entry)
    assert original['status'] == 'finished'
    selected = driver.root / 'selected-copy'
    shutil.copytree(first['checkpoints']['answers_ready'], selected)
    saved = batch.read(selected / 'team.json')
    break_authority(saved, fault, selected)
    write_json(selected / 'team.json', saved)
    preserved = tree_hash(selected, exact=True)
    attempt, record, proof = driver.run(policy, entry, checkpoint=selected, resume=True, expected=3)
    assert record['status'] == 'paused' and record['hold'] is True
    assert record['runtime_failure'] == 'public_reference_history_unavailable'
    assert record['source_checkpoint'] == str(selected)
    assert record['day'] == saved['day']
    assert proof['wire'] == []
    assert proof['ready']['world_equal'] and proof['ready']['messages_equal']
    for role in ('ceo', 'growth', 'ops_finance'):
        def requests(path):
            return [row for line in path.read_text().splitlines()
                    if (row := json.loads(line))['event'] == 'http_request']
        assert requests(attempt / 'runtime/private' / role / 'model-usage.jsonl') == requests(
            selected / 'private' / role / 'model-usage.jsonl')
    assert batch.read(attempt / 'HOLD')['manual_resume_required'] is True
    assert batch.read(attempt / 'HOLD')['reason'] == record['runtime_failure']
    assert Path(record['checkpoint']).is_dir()
    assert tree_hash(selected, exact=True) == preserved


@pytest.mark.parametrize('fault', ['authority', 'locator', 'conflict'])
def test_standalone_wrapper_holds_selected_checkpoint_before_model_send(offline_runner, monkeypatch, fault):
    source = offline_runner(text_registration='git')
    source.agent.current_day = 0
    assert receipt(source._execute_tool('text_create', declaration(text='retained')))['version'] == 'r1.1'
    source._save_checkpoint(0)
    source._stop_server()
    checkpoint = json.loads((source.workspace_dir / 'checkpoint.json').read_text())
    break_authority(checkpoint, fault, source.workspace_dir / 'checkpoints' / checkpoint['snapshot_id'])
    write_json(source.workspace_dir / 'checkpoint.json', checkpoint)
    original = tree_hash(source.workspace_dir, exact=True)
    restored = offline_runner(source.workspace_dir)
    assert restored.public_refs.unavailable
    monkeypatch.setattr(restored, 'setup', lambda: None)
    sent = []
    def forbidden(*args, **kwargs):
        sent.append(True)
        raise AssertionError('Reference HOLD must happen before a model call')
    monkeypatch.setattr(restored.client.chat.completions, 'create', forbidden)
    result = restored.run(verbose=False)
    assert sent == []
    assert result['outcome'] == 'paused' and result['hold'] is True
    assert result['reason'] == 'public_reference_history_unavailable'
    assert result['snapshot_id'] == checkpoint['snapshot_id'] and result['days_run'] == checkpoint['day']
    usage = restored.logs_dir / 'agent_requests.jsonl'
    assert not usage.exists() or not any(json.loads(line)['event'] == 'http_request' for line in usage.read_text().splitlines())
    hold = json.loads((restored.workspace_dir / 'HOLD').read_text())
    assert hold['manual_resume_required'] and hold['reason'] == result['reason']
    assert json.loads((restored.workspace_dir / 'checkpoint.json').read_text()) == checkpoint
    assert tree_hash(source.workspace_dir, exact=True) == original
