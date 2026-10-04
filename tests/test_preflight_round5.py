import importlib.util
import json
from pathlib import Path

import pytest

from saas_bench.run_state import checkpoint_directory, tree_hash
from test_preflight_integration import advance, offline_runner, packed_public


spec = importlib.util.spec_from_file_location('round5', Path(__file__).parents[1] / 'scripts/round5.py')
round5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(round5)


def test_go_override_replaces_old_official_simulator(monkeypatch):
    monkeypatch.setenv('CEOBENCH_SIMULATOR_LLM_PROVIDER', 'deepseek')
    monkeypatch.setenv('CEOBENCH_SIMULATOR_LLM_MODEL', 'deepseek-flash')
    monkeypatch.setenv('BOSSBENCH_LLM_REPLAY_DB', 'old-test')
    round5.environment()
    import os
    assert os.environ['CEOBENCH_SIMULATOR_LLM_PROVIDER'] == 'opencode'
    assert os.environ['CEOBENCH_SIMULATOR_LLM_MODEL'] == round5.MODEL
    assert 'BOSSBENCH_LLM_REPLAY_DB' not in os.environ


def test_monitor_detects_both_roles_and_usage_failures():
    for role in ('agent', 'simulator'):
        assert round5.model_issue(dict(role=role, event='http_response', status=200, error='Provider error')) == 'http_or_provider_error'
        assert round5.model_issue(dict(role=role, event='http_response', status=429)) == 'http_or_provider_error'
        assert round5.model_issue(dict(role=role, event='http_error')) == 'transport_error'
        assert round5.model_issue(dict(role=role, event='response', error='ValueError')) == 'model_call_error'
        assert round5.model_issue(dict(role=role, event='response', usage={'input_tokens': None})) == 'missing_usage'
        response = dict(role=role, event='response', response={'model': round5.MODEL},
                        usage=dict(input_tokens=10, output_tokens=2, cached_tokens=0), cost_usd=.01)
        assert round5.model_issue(response) is None
        response['cost_usd'] = None
        assert round5.model_issue(response) == 'missing_cost'
    assert round5.model_issue(dict(event='tool_result', result='Traceback (most recent call last): ordinary agent script')) is None
    request = dict(role='simulator', event='http_request', endpoint=round5.ENDPOINT + 'chat/completions',
                   body=dict(model=round5.MODEL, thinking={'type': 'disabled'}, reasoning_effort='none'))
    assert round5.model_issue(request) is None
    request['body']['reasoning_effort'] = 'high'
    assert round5.model_issue(request) == 'simulator_thinking_changed'
    request['endpoint'] = 'https://api.deepseek.com/chat/completions'
    assert round5.model_issue(request) == 'model_route_changed'


def test_incremental_receipts_do_not_lose_partial_lines(tmp_path):
    path = tmp_path / 'requests.jsonl'
    path.write_bytes(b'{"event":"request"}\n{"event":')
    rows, cursor = round5.read_increment(path, 0)
    assert rows == [{'event': 'request'}]
    with path.open('ab') as stream:
        stream.write(b'"response"}\n')
    rows, next_cursor = round5.read_increment(path, cursor)
    assert rows == [{'event': 'response'}]
    assert next_cursor == path.stat().st_size
    assert round5.read_increment(path, next_cursor) == ([], next_cursor)


def test_sealed_generation_stays_fixed_and_forks_restore_independently(offline_runner, tmp_path):
    source = offline_runner(execution_capture=True, text_registration='prefix')
    source._execute_tool('write_file', {'path': 'MEMORY.md', 'content': 'Offline prefix memory, fixed at D28'})
    for day in (7, 14, 21, 28):
        assert advance(source)['success']
        source._commit_weeks_up_to(day)
    source._save_checkpoint(28)
    cp = source._load_checkpoint()
    snapshot = checkpoint_directory(source.workspace_dir, cp)
    original = tree_hash(snapshot)
    sealed = tmp_path / 'A28'
    receipt = round5.seal_state(source.workspace_dir, sealed, cp['snapshot_id'])
    assert receipt['day'] == 28 and receipt['capture_cutoff'] == cp['sql_evidence']['cutoff']
    assert sealed.stat().st_mode & 0o222 == 0
    assert advance(source)['success']
    source._save_checkpoint(35)
    assert json.loads((sealed / 'checkpoint.json').read_text())['day'] == 28
    copies = {}
    for group in ('git', 'pf'):
        path = round5.fork_state(sealed, tmp_path / group, group, 'round5-offline-' + group)
        assert round5.validate_start(group, 42, 35, path) == 28
        fork = offline_runner(path)
        copies[group] = fork
        assert fork._get_game_status()['day'] == 28
        assert (fork.agent_workspace / 'MEMORY.md').read_bytes() == (snapshot / 'agent_workspace/MEMORY.md').read_bytes()
        assert fork.agent_workspace.stat().st_mode & 0o200
    assert not list(copies['git'].workspace_dir.rglob('sql-evidence*'))
    assert copies['pf'].evidence_store
    right = (copies['pf'].agent_workspace / 'MEMORY.md').read_bytes()
    copies['git']._execute_tool('write_file', {'path': 'MEMORY.md', 'content': 'Isolated offline mutation'})
    assert (copies['pf'].agent_workspace / 'MEMORY.md').read_bytes() == right
    assert tree_hash(snapshot) == original
    assert tree_hash(checkpoint_directory(sealed, cp)) == original
    assert round5.validate_start('prefix', 42, 84, sealed) == 28
    with pytest.raises(ValueError, match='seed mismatch'):
        round5.validate_start('prefix', 43, 84, sealed)
    with pytest.raises(ValueError, match='complete week boundary'):
        round5.validate_start('git', 42, 112, copies['git'].workspace_dir)
    with pytest.raises(ValueError, match='starts fresh'):
        round5.validate_start('prefix', 43, 28, sealed)
    git = copies['git']
    for day in (35, 42):
        assert advance(git)['success']
        git._commit_weeks_up_to(day)
    git._save_checkpoint(42)
    output = tmp_path / 'continuation'
    output.mkdir()
    previous = round5.stage_pointer(output, 'git', 42, 112)
    previous.write_text(json.dumps(dict(path=str(git.workspace_dir.resolve()), status='stopped',
        reached_stop=False, result={'days_run': 42})))
    (git.workspace_dir / 'pause-receipt.json').write_text(json.dumps({'day': 42}))
    assert round5.validate_start('git', 42, 112, git.workspace_dir, 2, output) == 42
    assert round5.stage_pointer(output, 'git', 42, 112, 2).name == 'git-A-to112-attempt2.json'
    with pytest.raises(ValueError, match='complete week boundary'):
        round5.validate_start('git', 42, 112, git.workspace_dir)
    previous.write_text(json.dumps(dict(path=str(git.workspace_dir.resolve()), status='running',
        reached_stop=False, result={'days_run': 42})))
    with pytest.raises(ValueError, match='complete week boundary'):
        round5.validate_start('git', 42, 112, git.workspace_dir, 2, output)


def test_receipt_audit_preserves_unknown_attempts_and_rejects_pending_or_changed_routes(tmp_path):
    path = tmp_path / 'agent_requests.jsonl'
    rows = [
        dict(event='request', call_id='call', day=42),
        dict(event='http_request', call_id='call', attempt_id='failed',
             endpoint=round5.ENDPOINT+'chat/completions',
             body=dict(model=round5.MODEL, reasoning_effort='high', thinking={'type':'enabled'})),
        dict(event='http_error', call_id='call', attempt_id='failed', error='ConnectTimeout'),
        dict(event='http_request', call_id='call', attempt_id='success',
             endpoint=round5.ENDPOINT+'chat/completions',
             body=dict(model=round5.MODEL, reasoning_effort='high', thinking={'type':'enabled'})),
        dict(event='http_response', call_id='call', attempt_id='success', status=200),
        dict(event='response', call_id='call', usage=dict(input_tokens=10, output_tokens=2,cached_tokens=0),
             cost_usd=.001, response={'model':round5.MODEL}),
    ]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    audit = round5.audit_model_receipts(path, 'agent')
    assert audit['failed_http_attempts'] == 1
    assert audit['gaps'] == [dict(role='agent', event='http_error',call_id='call',attempt_id='failed',issue='transport_error')]
    assert audit['usage']['known_cost_usd'] == .001
    assert audit['usage']['missing_cost'] == 0
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows[:-1]))
    with pytest.raises(ValueError,match='Unreturned'):
        round5.audit_model_receipts(path, 'agent')
    rows[1]['endpoint']='https://api.deepseek.com/chat/completions'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='model_route_changed'):
        round5.audit_model_receipts(path, 'agent')
