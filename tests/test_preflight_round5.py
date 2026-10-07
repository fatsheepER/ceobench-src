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


@pytest.mark.parametrize('seed', [42, 43])
def test_sealed_generation_stays_fixed_and_forks_restore_independently(offline_runner, tmp_path, seed):
    source = offline_runner(execution_capture=True, text_registration='prefix', seed=seed)
    source._execute_tool('write_file', {'path': 'MEMORY.md', 'content': 'Offline prefix memory, fixed at D28'})
    for day in (7, 14, 21, 28):
        assert advance(source)['success']
        source._commit_weeks_up_to(day)
    source._save_checkpoint(28)
    cp = source._load_checkpoint()
    snapshot = checkpoint_directory(source.workspace_dir, cp)
    original = tree_hash(snapshot)
    sealed = tmp_path / (round5.SEEDS[seed] + '28')
    receipt = round5.seal_state(source.workspace_dir, sealed, cp['snapshot_id'])
    assert receipt['day'] == 28 and receipt['capture_cutoff'] == cp['sql_evidence']['cutoff']
    assert sealed.stat().st_mode & 0o222 == 0
    assert advance(source)['success']
    source._save_checkpoint(35)
    assert json.loads((sealed / 'checkpoint.json').read_text())['day'] == 28
    copies = {}
    for group in ('git', 'pf'):
        path = round5.fork_state(sealed, tmp_path / group, group, 'round5-offline-' + group)
        assert round5.validate_start(group, seed, 35, path) == 28
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
    assert round5.validate_start('prefix', seed, 84, sealed) == 28
    with pytest.raises(ValueError, match='seed mismatch'):
        round5.validate_start('prefix', 44, 84, sealed)
    with pytest.raises(ValueError, match='complete week boundary'):
        round5.validate_start('git', seed, 112, copies['git'].workspace_dir)
    with pytest.raises(ValueError, match='starts fresh'):
        round5.validate_start('prefix', 43, 28, sealed)
    git = copies['git']
    for day in (35, 42):
        assert advance(git)['success']
        git._commit_weeks_up_to(day)
    git._save_checkpoint(42)
    output = tmp_path / 'continuation'
    output.mkdir()
    previous = round5.stage_pointer(output, 'git', seed, 112)
    previous.write_text(json.dumps(dict(path=str(git.workspace_dir.resolve()), status='stopped',
        reached_stop=False, result={'days_run': 42})))
    (git.workspace_dir / 'pause-receipt.json').write_text(json.dumps({'day': 42}))
    assert round5.validate_start('git', seed, 112, git.workspace_dir, 2, output) == 42
    assert round5.stage_pointer(output, 'git', seed, 112, 2).name == f'git-{round5.SEEDS[seed]}-to112-attempt2.json'
    with pytest.raises(ValueError, match='complete week boundary'):
        round5.validate_start('git', seed, 112, git.workspace_dir)
    previous.write_text(json.dumps(dict(path=str(git.workspace_dir.resolve()), status='running',
        reached_stop=False, result={'days_run': 42})))
    with pytest.raises(ValueError, match='complete week boundary'):
        round5.validate_start('git', seed, 112, git.workspace_dir, 2, output)
    for boundary in ('new_week', 'same_week'):
        git.agent.current_day = 42 if boundary == 'same_week' else 35
        git._save_checkpoint(42)
        checkpoint = git._load_checkpoint()
        assert checkpoint['context_boundary'] == boundary
        receipt = dict(status='paused', day=42, snapshot_id=checkpoint['snapshot_id'],
                       context_boundary=boundary)
        previous.write_text(json.dumps(dict(path=str(git.workspace_dir.resolve()), status='paused',
            reached_stop=False, result=dict(days_run=42, snapshot_id=checkpoint['snapshot_id']))))
        pause = git.workspace_dir / 'pause-receipt.json'
        pause.write_text(json.dumps(receipt))
        assert round5.validate_start('git', seed, 112, git.workspace_dir, 2, output) == 42
        for field, wrong in [('day', 35), ('snapshot_id', 'wrong'), ('context_boundary', 'wrong')]:
            pause.write_text(json.dumps(dict(receipt, **{field: wrong})))
            with pytest.raises(ValueError, match='complete week boundary'):
                round5.validate_start('git', seed, 112, git.workspace_dir, 2, output)


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


@pytest.mark.parametrize('service_failure', [False, True])
def test_monitor_observes_real_sql_failures_without_holding_on_bad_queries(tmp_path, monkeypatch, capfd,
                                                                         service_failure):
    import sqlite3
    from types import SimpleNamespace
    from saas_bench.api_server import NovaMindAPIServer
    from saas_bench.database import init_database
    from test_public_sql import request
    run, output = tmp_path / 'run', tmp_path / 'monitor'
    (run / 'logs').mkdir(parents=True)
    output.mkdir()
    args = SimpleNamespace(output_dir=output, mode='git', seed=42, stop_after_day=35,
                           attempt=1, continue_from=None)
    pointer = round5.stage_pointer(output, 'git', 42, 35)
    pointer.write_text(json.dumps(dict(path=str(run), run_id='test', status='completed')))
    conn = init_database(':memory:')
    server = NovaMindAPIServer(SimpleNamespace(workspace_path=tmp_path / 'workspace', current_day=0), conn=conn)
    server.start()
    connect = sqlite3.connect
    class Reader(sqlite3.Connection):
        def execute(self, sql, *args):
            if sql == 'SELECT 1 AS private_body':
                error = sqlite3.OperationalError('disk I/O error')
                error.sqlite_errorcode, error.sqlite_errorname = sqlite3.SQLITE_IOERR_WRITE, 'SQLITE_IOERR_WRITE'
                raise error
            return super().execute(sql, *args)
    def failed_reader(*args, **kwargs):
        return connect(*args, **kwargs, factory=Reader)
    try:
        if service_failure:
            monkeypatch.setattr(sqlite3, 'connect', failed_reader)
        status, _body = request(server, {'sql': 'SELECT 1 AS private_body' if service_failure else 'SELECT FROM private_body'})
        assert status == 500
    finally:
        monkeypatch.setattr(sqlite3, 'connect', connect)
        server.stop()
        conn.close()
    noise = 'unrelated stderr\n' * (1024**2 // 17 + 1) if service_failure else ''
    (run / 'logs' / 'api_server_stderr.log').write_text(noise + capfd.readouterr().err)
    monkeypatch.setattr(round5.subprocess, 'Popen', lambda *args, **kwargs: SimpleNamespace(pid=123, poll=lambda: 0))
    monkeypatch.setattr(round5, 'quota_health', lambda: {})
    round5.monitor(args)
    events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    sql_events = [row for row in events if row['event'] == 'sql_attention']
    assert bool(sql_events) is service_failure
    assert (output / 'hold.json').exists() is service_failure
    if service_failure:
        alert, = sql_events[0]['sql_issues']
        assert alert['sqlite_errorcode'] == sqlite3.SQLITE_IOERR_WRITE
        assert alert['sqlite_errorname'] == 'SQLITE_IOERR_WRITE'
        assert 'private_body' not in json.dumps(events)
    health = json.loads((output / 'health.json').read_text())
    assert health['sql_issues'] == (sql_events[0]['sql_issues'] if service_failure else [])


def test_sql_log_cursor_preserves_partial_lines_and_resets_for_new_runs(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    for run in (first, second):
        (run / 'logs').mkdir(parents=True)
    path = first / 'logs' / 'api_server_stderr.log'
    row = dict(error_kind='service', failure_stage='setup', sqlite_errorcode=778,
               sqlite_errorname='SQLITE_IOERR_WRITE', executed_sql='PRIVATE QUERY')
    line = b'[public_sql] ' + json.dumps(row).encode() + b'\n'
    path.write_bytes(b'unstructured stderr\n' + line[:30])
    issues, cursor = round5.sql_failures(first, {}, drain=True)
    assert issues == []
    with path.open('ab') as stream:
        stream.write(line[30:])
    issues, cursor = round5.sql_failures(first, cursor)
    assert len(issues) == 1 and 'PRIVATE' not in json.dumps(issues)
    assert round5.sql_failures(first, cursor)[0] == []
    (second / 'logs' / 'api_server_stderr.log').write_bytes(line)
    issues, cursor = round5.sql_failures(second, cursor)
    assert len(issues) == 1
    (second / 'logs' / 'api_server_stderr.log').write_bytes(b'old\n')
    assert round5.sql_failures(second, cursor)[0] == []
