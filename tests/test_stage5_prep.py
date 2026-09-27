"""Pre-stage-5 repairs: capture cost, Git forks, harness stop, group texts and tool-call reads."""
from contextlib import closing
import json
import re

import httpx
import pytest

from saas_bench.execution_capture import ExecutionCapture
from saas_bench.pf_queries import PFQueries
from saas_bench.pf_read import accounting, apply_delta
from saas_bench.registration_schema import MODELS, registration_prompt, tool_definitions
from test_text_registry import workspace, captured, call, declaration, git, send
from test_preflight_integration import offline_runner, packed_public, advance
from test_pf_read import Counter, deliver


def file_versions(store):
    with closing(store.connect()) as conn:
        return conn.execute("SELECT count(*) FROM versions WHERE json_extract(metadata,'$.layer')='file_bytes'").fetchone()[0]


def test_boundary_snapshots_reuse_unchanged_versions_in_one_transaction(workspace, tmp_path):
    store, _, executor = captured(workspace, tmp_path, 'prefix')
    executor.execute('bash', {'command': 'printf A > a.txt'})
    baseline = file_versions(store)
    executor.execute('bash', {'command': 'true'})
    assert file_versions(store) == baseline, 'unchanged bytes must not get new versions'
    executor.execute('bash', {'command': 'printf B > a.txt'})
    assert file_versions(store) == baseline + 1
    executor.execute('bash', {'command': 'printf A > a.txt'})
    assert file_versions(store) == baseline + 2, 'a revert differs from the newest version'
    latest, sha = store.latest_version('a.txt', 'file_bytes')
    meta, raw = store.get_content(latest)
    assert raw == b'A' and meta['previous_version'] and store.get_content(meta['previous_version'])[1] == b'B'
    # Manifests still name one version per path, and it holds exactly the observed bytes.
    capture = ExecutionCapture(store)
    capture.begin('probe', {})
    connects, original = [], store.connect
    def counted():
        connects.append(1)
        return original()
    store.connect = counted
    try:
        items = capture.snapshot(workspace, 'before')
    finally:
        store.connect = original
    assert len(connects) == 1, 'one snapshot is one transaction'
    for path, item in items.items():
        if item.get('type') == 'file':
            assert store.get_content(item['version'])[0]['blob_sha256'] == item['sha256'], path
    boundary = store.get_content(capture.event + ':workspace_before')[0]
    assert boundary['reused_file_versions'] == sum(1 for i in items.values() if i.get('type') == 'file')
    assert not store.fault


def test_group_forks_require_a_completed_week(offline_runner, tmp_path):
    from saas_bench.run_state import clone_sql_run
    prefix = offline_runner(text_registration='prefix')
    prefix.agent.current_day = 0
    prefix._save_checkpoint(0)
    with pytest.raises(ValueError, match='completed week'):
        clone_sql_run(prefix.workspace_dir, tmp_path / 'pf', 'pf', text_registration='pf')


def test_stop_day_is_harness_only_and_validated(offline_runner):
    for day in (10, 42, 0):
        with pytest.raises(ValueError, match='whole week'):
            offline_runner(stop_after_day=day)


def fake_weeks(runner, monkeypatch, suffix=''):
    from openai import OpenAI
    from test_preflight_usage import reply
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        body = reply('chat')
        command = "./novamind-operation next-week 'fixed offline action'" + ' 100000 -100000 1000000' * 4 + suffix
        body['choices'][0]['message']['tool_calls'] = [dict(id=f'week-{len(requests)}', type='function',
            function=dict(name='bash', arguments=json.dumps({'command': command})))]
        return httpx.Response(200, json=body)
    runner.client.close()
    runner.client = OpenAI(api_key='offline-only', base_url='https://api.deepseek.com', max_retries=0,
                           http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    runner.agent.client = runner.agent.usage_recorder.attach(runner.client)
    monkeypatch.setattr(runner, 'setup', lambda: None)
    return requests


@pytest.mark.parametrize('mode', ['off', 'prefix', 'pf'])
def test_trimmed_week_output_still_checkpoints_and_stops(offline_runner, monkeypatch, mode):
    runner = offline_runner(text_registration=mode, stop_after_day=7)
    requests = fake_weeks(runner, monkeypatch, suffix=' | tail -1')
    act = runner.agent.act
    def before_stop(observation, reward, done, info):
        assert info['day'] < 7, 'Agent called again after the requested stop day'
        return act(observation, reward, done, info)
    monkeypatch.setattr(runner.agent, 'act', before_stop)
    result = runner.run(verbose=False)
    assert result['outcome'] == 'stopped' and result['days_run'] == 7
    assert len(requests) == 1
    assert runner._load_checkpoint()['day'] == 7
    assert runner._load_checkpoint()['context_boundary'] == 'new_week'


def test_prefix_stops_at_fork_day_then_git_and_pf_branches_continue(offline_runner, monkeypatch, tmp_path):
    from saas_bench.run_state import clone_sql_run
    prefix = offline_runner(text_registration='prefix', stop_after_day=14)
    requests = fake_weeks(prefix, monkeypatch)
    result = prefix.run(verbose=False)
    assert result['outcome'] == 'stopped' and result['days_run'] == 14
    assert len(requests) == 2
    # The Agent only ever sees the configured horizon, never the harness stop.
    system = requests[0]['messages'][0]['content']
    assert '42 simulated days (6 weeks' in system and '14' not in re.findall(r'\d+ simulated days', system)[0]
    assert prefix._load_checkpoint()['context_boundary'] == 'new_week'
    manifest = json.loads((prefix.workspace_dir / 'manifest.json').read_text())
    assert 'stop_after_day' not in json.dumps(manifest)
    for mode in ('git', 'pf'):
        branch_dir = clone_sql_run(prefix.workspace_dir, tmp_path / mode, mode, text_registration=mode)
        branch = offline_runner(branch_dir, stop_after_day=28)
        more = fake_weeks(branch, monkeypatch)
        outcome = branch.run(verbose=False)
        assert outcome['outcome'] == 'stopped' and outcome['days_run'] == 28 and len(more) == 2
        assert branch._load_checkpoint()['day'] == 28
        assert (branch.evidence_store is None) == (mode == 'git')
        assert not list(branch_dir.rglob('sql-evidence*')) if mode == 'git' else (branch_dir / 'sql-evidence.sqlite').exists()


def test_git_and_prefix_see_no_pf_text_and_pf_sees_its_own_rules():
    git_text = registration_prompt(pf=False) + json.dumps(tool_definitions(pf=False))
    pf_text = registration_prompt(pf=True) + json.dumps(tool_definitions(pf=True))
    for word in ('PF', 'pf_', 'vN', 'handle', '"sql"', 'compare', 'delivered', 'sent to'):
        assert word not in git_text, word
    for word in ('"sql"', '"version"', 'compare', 'pf_dependencies', 'path@commit', 'last sent to'):
        assert word in pf_text, word
    assert 'Predicates are only stored' in git_text and 'Predicates are only stored' not in pf_text
    assert MODELS['prefix'] is MODELS['git']
    with pytest.raises(Exception):
        MODELS['git']['create'].model_validate(declaration({'sql': 'SELECT 1'}))
    MODELS['pf']['create'].model_validate(declaration({'sql': 'SELECT 1'}))


def test_pf_accepts_committed_file_references(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path, 'pf')
    head = git(workspace, 'rev-parse', 'HEAD')
    # Not yet sent to the model: still valid as in Git, reported as commit only.
    result = call(registry, 'create', **declaration({'path': 'evidence.json@' + head[:5]}))
    assert result['evidence'][0]['commit'] == head[:7] and result['evidence'][0]['status'] == 'commit_only'
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    (workspace / 'evidence.json').write_text('{"n":8}')
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    # The committed bytes ({"n":7}) were delivered earlier; bind that capture, not the newest read.
    result = call(registry, 'create', **declaration({'path': 'evidence.json', 'commit': head[:7]}))
    shown = result['evidence'][0]
    assert shown['commit'] == head[:7] and shown['version'] != shown['latest'] and shown['differs']
    binding = store.load_state('declaration:' + result['version'])['references'][0]
    assert binding['git_content_matches'] and store.get_content(binding['version_id'])[1] == b'{"n":7}'
    # A bare path keeps the PF rule: the version last sent to the model.
    bare = store.load_state('declaration:' + call(registry, 'create', **declaration({'path': 'evidence.json'}))['version'])
    assert store.get_content(bare['references'][0]['version_id'])[1] == b'{"n":8}'
    with pytest.raises(ValueError, match='Commit prefix'):
        call(registry, 'create', **declaration({'path': 'evidence.json@' + 'f' * 12}))


def sample(tag=''):
    return ''.join(f'row {i:03d}{tag if i == 40 else ""}: a useful observation about this business\n' for i in range(100))


def pf_executor(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path, 'pf')
    executor.pf_queries = PFQueries(registry)
    return store, registry, executor


def test_repeated_tool_calls_are_delivered_compactly_and_reconstruct(workspace, tmp_path):
    store, registry, executor = pf_executor(workspace, tmp_path)
    (workspace / 'data.txt').write_text(sample())
    first = executor.execute('bash', {'command': 'cat data.txt'})
    assert first.pf_read, 'eligible PF tool returns become reads'
    second = executor.execute('bash', {'command': 'cat data.txt'})
    _, ledger = deliver(store, [first, second])
    assert [r['mode'] for r in ledger] == ['FULL', 'UNCHANGED']
    assert ledger[1]['read_kind'] == 'tool_call' and ledger[1]['base_handle'] == 'previous_same_call'
    (workspace / 'data.txt').write_text(sample(' revised'))
    third = executor.execute('bash', {'command': 'cat data.txt'})
    other = executor.execute('bash', {'command': 'cat data.txt | head -n 100'})
    event, ledger = deliver(store, [first, second, third, other])
    assert ledger[2]['mode'] == 'DELTA' and ledger[3]['mode'] == 'FULL'
    assert apply_delta(str(second), ledger[2]['edits']) == str(third)
    wire = json.loads(store.get_content(event + ':wire')[1])
    header = json.loads(wire['messages'][2]['content'].split('\n', 1)[0])
    assert header['base'] == 'previous_same_call' and re.fullmatch(r'v\d+', header['target'])
    # The named full result is readable with pf_read.
    full = executor.execute('pf_read', {'target': {'version': header['target']}})
    assert full.split('\n', 1)[1] == str(third)
    summary = accounting(store)
    assert summary['read_kinds']['tool_call'] >= 4 and summary['net_saved_tokens'] > 0


def test_tool_call_recovery_truncation_and_file_delivery(workspace, tmp_path):
    store, registry, executor = pf_executor(workspace, tmp_path)
    (workspace / 'facts.txt').write_text(sample())
    a = executor.execute('read_file', {'path': 'facts.txt'})
    b = executor.execute('read_file', {'path': 'facts.txt'})
    _, ledger = deliver(store, [a, b])
    assert ledger[1]['mode'] == 'UNCHANGED'
    # A compact file read still counts as delivering that file version (reconstruction).
    result = call(registry, 'create', **declaration({'path': 'facts.txt'}))
    assert result['evidence'][0]['version'].startswith('v')
    # Repeating the call right after a compact result returns full text once.
    c = executor.execute('read_file', {'path': 'facts.txt'})
    _, ledger = deliver(store, [a, b, c])
    assert ledger[2]['mode'] == 'FULL' and ledger[2]['reason'] == 'adjacent_recovery'
    assert ledger[2]['recovery_of'] == b.pf_read['id']
    # Harness-truncated output is never compacted.
    (workspace / 'big.txt').write_text('x' * 40000)
    big1 = executor.execute('bash', {'command': 'cat big.txt'})
    big2 = executor.execute('bash', {'command': 'cat big.txt'})
    _, ledger = deliver(store, [big1, big2])
    assert ledger[1]['mode'] == 'FULL' and ledger[1]['reason'] == 'partial_or_truncated'
    # Git and prefix groups never produce compact tool reads.
    store2, _, prefix = captured(workspace, tmp_path / 'prefix', 'prefix')
    assert prefix.execute('read_file', {'path': 'facts.txt'}).pf_read is None


def test_same_content_object_reads_compare_with_the_previous_output(offline_runner, workspace, tmp_path):
    # A leading cd into the workspace itself does not make a different read.
    store, registry, executor = pf_executor(workspace, tmp_path)
    (workspace / 'data.txt').write_text(sample())
    plain = executor.execute('bash', {'command': 'cat data.txt'})
    moved = executor.execute('bash', {'command': f'cd {executor.guest_root} && cat data.txt'})
    _, ledger = deliver(store, [plain, moved])
    assert [r['mode'] for r in ledger] == ['FULL', 'UNCHANGED']
    # Rerunning the same script is compared with its last output, whatever surrounds it.
    runner = offline_runner(text_registration='pf')
    runner.agent.current_day = 0
    body = ''.join(f"print('row {i:03d}: a stable report line about this business')\n" for i in range(60))
    first = runner._execute_tool('bash', {'command': "cat > report.py <<'EOF'\n" + body + "EOF\n./novamind-operation python report.py"})
    second = runner._execute_tool('bash', {'command': 'echo rerun; ./novamind-operation python ./report.py'})
    event, ledger = deliver(runner.evidence_store, [first, second])
    assert ledger[1]['mode'] in ('DELTA', 'UNCHANGED') and ledger[1]['base_handle'] == 'previous_output_of:report.py'
    if ledger[1]['mode'] == 'DELTA':
        assert apply_delta(str(first), ledger[1]['edits']) == str(second)
    wire = json.loads(runner.evidence_store.get_content(event + ':wire')[1])
    assert json.loads(wire['messages'][1]['content'].split('\n', 1)[0])['base'] == 'previous_output_of:report.py'
    # Inline Python and ordinary commands keep the same-call identity.
    other = runner._execute_tool('bash', {'command': './novamind-operation python -c "print(1)"'})
    _, ledger = deliver(runner.evidence_store, [first, other])
    assert ledger[1]['mode'] == 'FULL'
