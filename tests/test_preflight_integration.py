import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from saas_bench.agents.bash_agent.run_test import BashAgentRunner
from saas_bench.db_protection import load_session_db
from saas_bench.run_state import checkpoint_directory, tree_hash

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope='module')
def packed_public(tmp_path_factory):
    provided = os.environ.get('CEOBENCH_TEST_PUBLIC')
    if provided:
        return Path(provided)
    spec = importlib.util.spec_from_file_location('build_public', ROOT / 'scripts/build_public.py')
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    build.PUBLIC_DIR = tmp_path_factory.mktemp('packed-public')
    build.build()
    return build.PUBLIC_DIR


@pytest.fixture
def offline_runner(tmp_path, monkeypatch, packed_public):
    monkeypatch.setenv('NOVAMIND_PUBLIC_DIR', str(packed_public))
    monkeypatch.setenv('PYTHONHASHSEED', '0')
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'offline-only')
    monkeypatch.setenv('CEOBENCH_SIMULATOR_LLM_PROVIDER', 'deepseek')
    monkeypatch.setenv('CEOBENCH_SIMULATOR_LLM_MODEL', 'test-model')
    monkeypatch.delenv('CEOBENCH_DASHBOARD_URL', raising=False)
    monkeypatch.delenv('BOSSBENCH_LLM_REPLAY_DB', raising=False)
    monkeypatch.delenv('ORACLE_MODE', raising=False)
    monkeypatch.setattr('saas_bench.agents.bash_agent.run_test.load_env_file', lambda _: {})
    from saas_bench.config import BenchmarkConfig
    def config(**kwargs):
        return BenchmarkConfig(**dict(kwargs, drift_grace_period_days=0,
                                      competitor_event_late_cutoff_days=0, competitor_event_mean_interval=3))
    monkeypatch.setattr('saas_bench.agents.bash_agent.run_test.BenchmarkConfig', config)
    popen = subprocess.Popen
    def launch(args, *other, **kwargs):
        if len(args) > 1 and str(args[1]).endswith('novamind-operation') and kwargs.get('env', {}).get('NOVAMIND_SERVER_MODE') == '1':
            args = [args[0], str(ROOT / 'tests/preflight_server.py'), *args[1:]]
        return popen(args, *other, **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', launch)
    runners = []
    def create(restore=None, **options):
        runner = BashAgentRunner(model='test-model', provider='deepseek', api_key='offline-only',
                                total_days=42, workspace_base=tmp_path, continue_from=restore,
                                run_kind=os.environ.get('CEOBENCH_TEST_KIND', 'engineering'), **options)
        runners.append(runner)
        runner.setup()
        if restore:
            runner._restore_from_checkpoint(runner._load_checkpoint())
        return runner
    yield create
    for runner in runners:
        runner._stop_server()
        runner.client.close()


def advance(runner):
    return runner._http_post('/next-week', {'rationale': 'fixed offline action',
        'predictions': {h: {'point': 100000, 'lower': -100000, 'upper': 1000000}
                        for h in ('cash_1wk', 'cash_4wk', 'cash_12wk', 'cash_26wk')}}, timeout=120)


def test_agent_zipapp_excludes_engine_and_survives_restore(offline_runner, packed_public):
    import shlex
    import zipfile
    from saas_bench.run_state import file_hash

    runner = offline_runner()
    expected = ['__main__.py', '_client_capture.pyc', '_public_cli.pyc']
    # Inspect bytes from the actual sandbox with sitecustomize disabled. Import
    # restrictions cannot protect engine code or decryption keys in a zipapp.
    probe = "import json, zipfile; print(json.dumps(sorted(zipfile.ZipFile('novamind-operation').namelist())))"
    command = 'python -S -c ' + shlex.quote(probe)
    assert json.loads(runner._execute_tool('bash', {'command': command})) == expected
    client_hash = file_hash(packed_public / 'novamind-client')
    assert file_hash(runner.agent_workspace / 'novamind-operation') == client_hash
    with zipfile.ZipFile(packed_public / 'novamind-operation') as archive:
        assert 'saas_bench/server_entry.pyc' in archive.namelist()
    denied = runner._execute_tool('bash', {'command': 'NOVAMIND_SERVER_MODE=1 ./novamind-operation --help'})
    assert 'Simulation server is managed by the host Runner.' in denied
    assert '[exit code: 1]' in denied
    query = runner._execute_tool('bash', {'command': './novamind-operation query "SELECT COUNT(*) AS n FROM ledger"'})
    assert json.loads(query)['rows'][0]['n'] >= 0
    assert advance(runner)['success']
    runner._commit_weeks_up_to(7)
    runner._save_checkpoint(7)
    runner._stop_server()
    restored = offline_runner(runner.workspace_dir)
    assert file_hash(restored.agent_workspace / 'novamind-operation') == client_hash
    assert json.loads(restored._execute_tool('bash', {'command': command})) == expected
    assert advance(restored)['success']


def test_agent_cannot_reinitialize_simulator(offline_runner):
    runner = offline_runner()
    funding = './novamind-operation query "SELECT day, amount FROM ledger WHERE category=\'initial_funding\'"'
    before = json.loads(runner._execute_tool('bash', {'command': funding}))['rows']
    probe = ("import os, urllib.error, urllib.request\n"
             "url = 'http://127.0.0.1:' + os.environ['NOVAMIND_API_PORT'] + '/reinitialize'\n"
             "try: urllib.request.urlopen(urllib.request.Request(url, data=b'{}', method='POST'))\n"
             "except urllib.error.HTTPError as error: print(error.code)\n")
    runner._execute_tool('write_file', {'path': 'reinit_probe.py', 'content': probe})
    assert runner._execute_tool('bash', {'command': 'python -S reinit_probe.py'}).strip() == '404'
    # A rejected control request must not leave writes for the next week to commit.
    assert json.loads(runner._execute_tool('bash', {'command': funding}))['rows'] == before
    assert advance(runner)['success']
    assert json.loads(runner._execute_tool('bash', {'command': funding}))['rows'] == before


def business_state(runner):
    directory = checkpoint_directory(runner.workspace_dir, runner._load_checkpoint())
    conn = load_session_db(directory / 'world.nmdb')
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
              if not row[0].startswith('sqlite_')]
    result = {}
    for table in tables:
        columns = [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")') if row[1] != 'submitted_at']
        select = ','.join('"' + c + '"' for c in columns)
        result[table] = sorted([list(row) for row in conn.execute(f'SELECT {select} FROM "{table}"')], key=str)
    conn.close()
    return result


def clone(source, destination):
    checkpoint = source._load_checkpoint()
    directory = checkpoint_directory(source.workspace_dir, checkpoint)
    destination.mkdir()
    for name in ('manifest.json', 'checkpoint.json', 'config.json'):
        shutil.copy2(source.workspace_dir / name, destination / name)
    target = destination / 'checkpoints' / checkpoint['snapshot_id']
    target.parent.mkdir()
    shutil.copytree(directory, target)
    return destination


@pytest.mark.parametrize('split_day', [21, 35])
def test_packed_continuous_restore_and_independent_clones(offline_runner, tmp_path, split_day):
    first = offline_runner()
    workspace = first.agent_workspace
    (workspace / 'MEMORY.md').write_text('retain the weekly metric')
    metric = workspace / 'metric.py'
    metric.write_text("import novamind_api as nm\nprint('version A', nm.query('SELECT COUNT(*) AS n FROM ledger'))")
    assert first._http_post('/daily-scripts', {'name': 'metric', 'content': metric.read_text()})['success']
    metric.write_text("print('version B')")
    for _ in range(split_day // 7):
        receipt = advance(first)
        assert receipt['success'], receipt
        assert 'version A' in receipt['dashboard'] and '[stderr]' not in receipt['dashboard']
    assert first._get_game_status()['day'] == split_day
    first.agent._refresh_context(f'day {split_day}', split_day)
    first.agent.current_day = split_day
    first.agent.record_tool_result('completed metric registration')
    first._git_commit_workspace('offline checkpoint')
    first._save_checkpoint(split_day)
    source_snapshot = checkpoint_directory(first.workspace_dir, first._load_checkpoint())
    assert (source_snapshot / 'world.nmdb').is_file()
    assert not (source_snapshot / 'agent_workspace' / 'sessions' / first._session_id / 'world.nmdb').exists()
    assert (first.agent_workspace / 'sessions' / first._session_id / 'world.nmdb').is_file()
    checkpoint_hash = tree_hash(source_snapshot)
    left = offline_runner(clone(first, tmp_path / 'left'))
    right = offline_runner(clone(first, tmp_path / 'right'))
    assert len({r._server_proc.pid for r in (first, left, right)}) == 3
    assert len({r._server_port for r in (first, left, right)}) == 3
    for restored in (left, right):
        assert restored.agent._last_observation == 'completed metric registration'
        assert restored.agent._observation_recorded
        assert (restored.agent_workspace / 'MEMORY.md').read_text() == 'retain the weekly metric'
        assert restored._git('rev-parse', 'HEAD').stdout == first._git('rev-parse', 'HEAD').stdout
        assert restored._http_get('/daily-scripts') == first._http_get('/daily-scripts')
        assert (restored.logs_dir / 'simulator_requests.jsonl').read_bytes() == (first.logs_dir / 'simulator_requests.jsonl').read_bytes()
    for _ in range((42 - split_day) // 7):
        expected = advance(first)
        for restored in (left, right):
            assert advance(restored) == expected
    for runner in (first, left, right):
        runner._save_checkpoint(42)
    expected = business_state(first)
    for restored in (left, right):
        assert business_state(restored) == expected
        summary = json.loads((restored.workspace_dir / 'usage_summary.json').read_text())
        assert summary['simulator'] == json.loads((first.workspace_dir / 'usage_summary.json').read_text())['simulator']
    assert expected['api_costs'] and expected['competitor_events']
    assert expected['_hidden_leads_per_1k_snapshot']
    scoring = subprocess.run([sys.executable, str(ROOT / 'scripts/score_predictions.py'), str(first.workspace_dir)],
                             capture_output=True, text=True, check=True)
    assert json.loads(scoring.stdout)['four_week_count'] == 3
    before = business_state(right)
    (left.agent_workspace / 'MEMORY.md').write_text('left only')
    assert left._http_post('/daily-scripts', {'name': 'metric', 'content': "print('version B')"})['success']
    receipt = advance(left)
    assert 'version B' in receipt['dashboard'] and 'version A' not in receipt['dashboard']
    left._save_checkpoint(49)
    assert business_state(right) == before
    assert 'version A' in json.dumps(before['_registered_scripts'])
    assert (right.agent_workspace / 'MEMORY.md').read_text() == 'retain the weekly metric'
    assert checkpoint_hash == tree_hash(source_snapshot)


def test_failed_publication_and_unknown_operation_keep_previous_snapshot(offline_runner, monkeypatch):
    runner = offline_runner()
    runner._save_checkpoint(0)
    pointer = (runner.workspace_dir / 'checkpoint.json').read_bytes()
    from saas_bench import run_state
    replace = run_state.os.replace
    def fail_publish(source, destination):
        if Path(destination) == runner.workspace_dir / 'checkpoint.json':
            raise OSError('injected pointer publication failure')
        return replace(source, destination)
    monkeypatch.setattr(run_state.os, 'replace', fail_publish)
    with pytest.raises(OSError, match='publication'):
        runner._save_checkpoint(0)
    assert (runner.workspace_dir / 'checkpoint.json').read_bytes() == pointer
    assert checkpoint_directory(runner.workspace_dir, runner._load_checkpoint()).is_dir()
    runner._begin_operation('tool', 0)
    with pytest.raises(ValueError, match='unknown'):
        runner._load_checkpoint()
    # Analysis can still inspect the last complete generation directly.
    assert checkpoint_directory(runner.workspace_dir, json.loads(pointer)).is_dir()


def test_checkpoint_http_does_not_accept_paths_and_predictions_reject_nonfinite(offline_runner, tmp_path):
    import urllib.error
    runner = offline_runner()
    with pytest.raises(urllib.error.HTTPError) as error:
        runner._http_post('/checkpoint', {'expected_day': 0, 'path': str(tmp_path / 'injected')})
    assert error.value.code == 400 and not (tmp_path / 'injected').exists()
    for bad in (float('nan'), float('inf'), None):
        with pytest.raises(urllib.error.HTTPError) as error:
            runner._http_post('/next-week', {'rationale': 'test', 'predictions': {
                h: {'point': bad, 'lower': 0, 'upper': 100} for h in ('cash_1wk', 'cash_4wk', 'cash_12wk', 'cash_26wk')}})
        assert error.value.code == 400
    assert runner._get_game_status()['day'] == 0


@pytest.mark.parametrize('execution', [False, True])
def test_full_harness_uses_packed_cli_and_fake_agent_requests(offline_runner, monkeypatch, execution):
    import httpx
    from openai import OpenAI
    from test_preflight_usage import reply
    runner = offline_runner(execution_capture=execution)
    rejected = runner._execute_tool('bash', {'command':
        "./novamind-operation next-week 'invalid forecast'" + ' 20.7 22.1 19.3' * 4})
    assert '[exit code: 1]' in rejected and 'Error:' in rejected
    assert 'Submitted cash forecasts' not in rejected
    requests = []
    amounts = ' 20.7 19.3 22.1 20.7e6 19300000 22100000 -49.15 -50 -48 0.001 -0.002 0.004'
    forecasts = [
        '+7 days: point=USD 20.7, lower=USD 19.3, upper=USD 22.1',
        '+28 days: point=USD 20700000.0, lower=USD 19300000.0, upper=USD 22100000.0',
        '+84 days: point=USD -49.15, lower=USD -50.0, upper=USD -48.0',
        '+182 days: point=USD 0.001, lower=USD -0.002, upper=USD 0.004',
    ]
    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) > 6:
            raise KeyboardInterrupt('Harness failed to advance the week')
        observation = '\n'.join(m['content'] for m in requests[-1]['messages'] if m['role'] == 'user')
        assert observation.count('Forecast units:') == 1
        assert 'All 12 cash forecast values for next-week are USD.' in observation
        assert 'For 20.7 million USD, submit 20700000 or 20.7e6.' in observation
        if len(requests) > 1:
            assert f'Submitted cash forecasts at D{(len(requests) - 2) * 7} (USD; horizons from submission):' in observation
            assert all(forecast in observation for forecast in forecasts)
        body = reply('chat')
        command = "./novamind-operation next-week 'fixed offline action'" + amounts
        body['choices'][0]['message']['tool_calls'] = [dict(id=f'week-{len(requests)}', type='function',
            function=dict(name='bash', arguments=json.dumps({'command': command})))]
        return httpx.Response(200, json=body)
    runner.client.close()
    runner.client = OpenAI(api_key='offline-only', base_url='https://api.deepseek.com', max_retries=0,
                           http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    runner.agent.client = runner.agent.usage_recorder.attach(runner.client)
    monkeypatch.setattr(runner, 'setup', lambda: None)
    result = runner.run(verbose=False)
    assert result['days_run'] == 42 and result['outcome'] == 'completed'
    assert runner._server_proc is None
    assert len(requests) == 6
    assert all(not any(m['role'] == 'tool' for m in request['messages']) for request in requests)
    checkpoint = runner._load_checkpoint()
    assert checkpoint['total_turns'] == 6
    tool_results = [json.loads(line) for line in (runner.logs_dir / f'tool_results_{runner.run_id}.jsonl').read_text().splitlines()]
    submissions = [row['result'] for row in tool_results if row['tool'] == 'bash']
    assert len(submissions) == 6
    assert all(all(forecast in receipt for forecast in forecasts) for receipt in submissions)
    conn = load_session_db(checkpoint_directory(runner.workspace_dir, checkpoint) / 'world.nmdb')
    try:
        stored = [tuple(row) for row in conn.execute('''SELECT submit_day,horizon_days,
            predicted_value,predicted_lower,predicted_upper FROM predictions ORDER BY submit_day,horizon_days''')]
    finally:
        conn.close()
    expected = [(7, 20.7, 19.3, 22.1), (28, 20700000.0, 19300000.0, 22100000.0),
                (84, -49.15, -50.0, -48.0), (182, 0.001, -0.002, 0.004)]
    assert stored == [(day, *forecast) for day in range(0, 42, 7) for forecast in expected]
    summary = json.loads((runner.workspace_dir / 'usage_summary.json').read_text())
    assert summary['agent']['calls'] == 6 and summary['agent']['known']['input_tokens'] == 60
    assert summary['simulator']['calls'] > 0


@pytest.mark.parametrize('first_response', ['no_tool', 'invalid_json', 'http_error', 'missing_usage'])
def test_harness_counts_every_response_when_act_regenerates(offline_runner, monkeypatch, first_response):
    import httpx
    from openai import OpenAI
    from test_preflight_usage import reply
    runner = offline_runner(stop_after_day=14)
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        assert len(requests) <= 4
        body = reply('chat')
        if len(requests) % 2:
            if first_response == 'http_error':
                return httpx.Response(503, json={'error': {'message': 'offline retry'}})
            body['choices'][0]['finish_reason'] = 'length'
            body['usage']['completion_tokens'] = 16384
            if first_response == 'invalid_json':
                body['choices'][0]['message']['tool_calls'] = [dict(id=f'bad-{len(requests)}', type='function',
                    function=dict(name='bash', arguments='{invalid'))]
            elif first_response == 'missing_usage':
                body.pop('usage')
        else:
            body['usage']['prompt_tokens'] = 20
            command = "./novamind-operation next-week 'fixed offline action'" + ' 100000 -100000 1000000' * 4
            body['choices'][0]['message']['tool_calls'] = [dict(id=f'week-{len(requests)}', type='function',
                function=dict(name='bash', arguments=json.dumps({'command': command})))]
        return httpx.Response(200, json=body)
    runner.client.close()
    runner.client = OpenAI(api_key='offline-only', base_url='https://api.deepseek.com', max_retries=0,
                           http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    runner.agent.client = runner.agent.usage_recorder.attach(runner.client)
    monkeypatch.setattr(runner, 'setup', lambda: None)
    monkeypatch.setattr('time.sleep', lambda _: None)
    result = runner.run(verbose=False)
    assert result['days_run'] == 14 and result['outcome'] == 'stopped'
    assert len(requests) == 4
    timing = [json.loads(line) for line in runner.timing_log_file.read_text().splitlines()]
    acts = [row for row in timing if row['event'] == 'llm_call']
    days = [row for row in timing if row['event'] == 'day_summary']
    assert len(acts) == len(days) == 2
    first_known = first_response in ('no_tool', 'invalid_json')
    expected = dict(input_tokens=30 if first_known else 20, output_tokens=16386 if first_known else 2,
                    cached_tokens=6 if first_known else 3, reasoning_tokens=2 if first_known else 1)
    for field, value in expected.items():
        assert [row[field] for row in acts] == [value, value]
        assert [row['day_' + field] for row in days] == [value, value]
    assert [row['model_calls'] for row in acts] == [2, 2]
    assert [row['day_model_calls'] for row in days] == [2, 2]
    assert [row['usage_missing']['input_tokens'] for row in acts] == [int(not first_known)] * 2
    assert [row['day_usage_missing']['input_tokens'] for row in days] == [int(not first_known)] * 2
    assert all(row['cache_creation_tokens'] is None and row['usage_missing']['cache_creation_tokens'] == 2
               for row in acts)
    summary = json.loads((runner.workspace_dir / 'usage_summary.json').read_text())['agent']
    assert summary['calls'] == 4 and summary['known']['input_tokens'] == 2 * expected['input_tokens']
    for field in ('input_tokens', 'output_tokens', 'cached_tokens', 'reasoning_tokens'):
        assert sum(row[field] for row in acts) == sum(row['day_' + field] for row in days) == summary['known'][field]


def test_harness_timeout_stops_branch_without_publishing_unknown_state(offline_runner, monkeypatch):
    from saas_bench.environment import Action
    runner = offline_runner()
    runner._save_checkpoint(0)
    previous = (runner.workspace_dir / 'checkpoint.json').read_bytes()
    monkeypatch.setattr(runner, 'setup', lambda: None)
    runner.tool_executor.bash_timeout = 0.2
    monkeypatch.setattr(runner.agent, 'act', lambda *args: Action(tool='bash',
        arguments={'command': 'python -c "import time; time.sleep(30)"'}))
    with pytest.raises(RuntimeError, match='unknown'):
        runner.run(verbose=False)
    assert runner._server_proc is None
    assert (runner.workspace_dir / 'checkpoint.json').read_bytes() == previous
    assert 'unknown' in json.loads((runner.workspace_dir / 'branch_stop.json').read_text())['reason']
    with pytest.raises(ValueError, match='unknown'):
        runner._load_checkpoint()


def test_packed_public_sql_policy(offline_runner):
    """Exercise the rebuilt engine through HTTP, shipped SDK, CLI and weekly scripts."""
    runner = offline_runner()
    workspace = runner.agent_workspace
    positive = runner._http_post('/query', {'sql': 'SELECT * FROM subscriptions LIMIT 1'})
    assert positive['success'] and 'first_billing_done' not in positive['columns']
    import urllib.error
    attacks = [
        'WITH c AS (SELECT 1) UPDATE ledger SET amount=99',
        'SELECT * FROM group_insight_snapshots',
        'SELECT actual_completion_day AS done_on FROM research_projects',
    ]
    env = dict(os.environ, NOVAMIND_API_PORT=str(runner._server_port), PYTHONPATH=str(workspace / 'docs'))
    for sql in ['SELECT count(*) AS n FROM ledger', *attacks]:
        good = sql not in attacks
        try:
            response = runner._http_post('/query', {'sql': sql})
            assert good and response['success']
        except urllib.error.HTTPError:
            assert not good
        cli = subprocess.run([sys.executable, str(workspace / 'novamind-operation'),
                              'query', sql],
                             cwd=workspace, env=env, text=True, capture_output=True, timeout=10)
        assert (cli.returncode == 0) == good, cli.stdout + cli.stderr
        sdk = subprocess.run([sys.executable, '-c', 'import novamind_api as nm; print(nm.query(' + repr(sql) + '))'],
                             cwd=workspace, env=env, text=True, capture_output=True, timeout=10)
        assert (sdk.returncode == 0) == good, sdk.stdout + sdk.stderr
    assert runner._http_post('/daily-scripts', {'name': 'sql-policy', 'content':
        "import novamind_api as nm\nprint(nm.query('SELECT count(*) AS n FROM ledger'))"})['success']
    result = advance(runner)
    assert result['success']
    assert 'row_count' in result['dashboard']


def test_packed_sql_evidence_restore_and_fork(offline_runner, tmp_path):
    from contextlib import closing
    from saas_bench.sql_evidence import SQLEvidenceStore
    from saas_bench.run_state import clone_sql_run
    runner = offline_runner(sql_capture=True)
    workspace = runner.agent_workspace
    sql = 'SELECT count(*) AS n FROM ledger'
    original = runner._http_post('/query', {'sql': sql})
    env = dict(os.environ, NOVAMIND_API_PORT=str(runner._server_port), PYTHONPATH=str(workspace / 'docs'))
    for command in ([sys.executable, str(workspace / 'novamind-operation'), 'query', sql],
                    [sys.executable, '-c', 'import novamind_api as nm; print(nm.query(' + repr(sql) + '))']):
        assert subprocess.run(command, cwd=workspace, env=env, capture_output=True, timeout=10).returncode == 0
    runner._http_post('/daily-scripts', {'name': 'capture', 'content':
        "import novamind_api as nm\nprint(nm.query('SELECT count(*) AS n FROM ledger'))"})
    assert advance(runner)['success']
    runner._save_checkpoint(7)
    checkpoint = runner._load_checkpoint()
    assert checkpoint['sql_evidence']['cutoff'] == 4
    store = SQLEvidenceStore(runner.workspace_dir / 'sql-evidence.sqlite', runner.sql_evidence_config)
    version = runner.run_id + '/prefix/1:public_response'
    assert json.loads(store.get_content(version)[1]) == original
    directory = checkpoint_directory(runner.workspace_dir, checkpoint)
    assert not list((directory / 'agent_workspace').rglob('sql-evidence*'))
    runner._stop_server()
    restored = offline_runner(runner.workspace_dir)
    assert not restored.execution_capture and restored.evidence_store is None
    with pytest.raises(ValueError, match='configuration mismatch'):
        offline_runner(runner.workspace_dir, execution_capture=True)
    restored._http_post('/query', {'sql': sql})
    restored._save_checkpoint(7)
    assert restored._load_checkpoint()['sql_evidence']['cutoff'] == 5
    left = offline_runner(clone_sql_run(restored.workspace_dir, tmp_path / 'capture-left', 'left'))
    right = offline_runner(clone_sql_run(restored.workspace_dir, tmp_path / 'capture-right', 'right'))
    for child in (left, right):
        child._http_post('/query', {'sql': 'SELECT 42 AS x'})
        child._save_checkpoint(7)
        child_store = SQLEvidenceStore(child.workspace_dir / 'sql-evidence.sqlite', child.sql_evidence_config)
        assert child_store.get_content(version)[1] == store.get_content(version)[1]
        branch = child.sql_evidence_config['branch_id']
        assert child_store.get_content(child.run_id + '/' + branch + '/1:public_response')[1]
        other = 'right' if branch == 'left' else 'left'
        with pytest.raises(KeyError):
            child_store.get_content(child.run_id + '/' + other + '/1:public_response')
    left._stop_server()
    again = offline_runner(left.workspace_dir)
    again._http_post('/query', {'sql': 'SELECT 43 AS x'})
    again._save_checkpoint(7)
    assert again._load_checkpoint()['sql_evidence']['cutoff'] == 2
    with closing(store.connect()) as conn:
        assert store.sequence(conn) == 5


def test_packed_execution_capture_scripts_cache_restore_and_fork(offline_runner, tmp_path):
    import shlex
    from contextlib import closing
    from saas_bench.run_state import clone_sql_run
    from test_sql_evidence import event_ids
    runner = offline_runner(execution_capture=True)
    store = runner.evidence_store
    code = "import novamind_api as nm; print(nm.query('SELECT 17 AS n')); print('中' * 600)"
    result = runner.tool_executor.execute('bash', {'command': './novamind-operation python-c ' + shlex.quote(code)})
    assert '17' in result and '中' * 600 in result, result
    events = [store.read_event(e) for e in event_ids(store)]
    python_event = next(r['request']['event_id'] for r in events if r['request']['kind'] == 'cli_python')
    assert store.get_content(python_event + ':code')[1].decode() == code
    query = next(r for r in events if r['request']['kind'] == 'sql_query')
    assert query['request']['parent_event_id'] == python_event
    assert query['client_receipt']['receive_state'] == 'received'
    script = runner.agent_workspace / 'registered.py'
    script.write_text("print('version A' + '中' * 600)")
    runner._http_post('/daily-scripts', {'name': 'metric', 'content': script.read_text()})
    registered = store.load_state('script_versions')['metric']
    script.write_text("print('version B')")
    assert 'version A' in advance(runner)['dashboard']
    dashboard = runner._get_dashboard()
    assert 'version B' not in dashboard and '中' * 600 not in dashboard
    generated = store.load_state('dashboard')['version']
    assert runner._get_dashboard() == dashboard
    assert store.load_state('dashboard')['version'] == generated
    meta, body = store.get_content(generated)
    assert any(o['source_range'][1] - o['source_range'][0] == 500 for o in meta['segments'])
    assert b'version A' in store.get_content(registered)[1]
    assert any(r['request']['kind'] == 'registered_script_execution' for r in
               (store.read_event(e) for e in event_ids(store)))
    runner._save_checkpoint(7)
    snapshot = checkpoint_directory(runner.workspace_dir, runner._load_checkpoint())
    assert not list((snapshot / 'agent_workspace').rglob('*evidence.sqlite'))
    runner._stop_server()
    restored = offline_runner(runner.workspace_dir)
    assert restored._get_dashboard() == dashboard
    assert restored.evidence_store.load_state('dashboard')['version'] == generated
    restored._save_checkpoint(7)
    left = offline_runner(clone_sql_run(restored.workspace_dir, tmp_path / 'execution-left', 'left'))
    right = offline_runner(clone_sql_run(restored.workspace_dir, tmp_path / 'execution-right', 'right'))
    for child in (left, right):
        assert child.evidence_store.get_content(registered)[1] == store.get_content(registered)[1]
        text = child.tool_executor.execute('write_file', {'path': 'branch.txt', 'content': child.sql_evidence_config['branch_id']})
        version = text.origins[0]['version_id']
        other = right if child is left else left
        with pytest.raises(KeyError):
            other.evidence_store.get_content(version)
        child._save_checkpoint(7)
        child.evidence_store.assert_healthy()


def test_six_week_execution_capture_matches_uncaptured_run(offline_runner, tmp_path):
    """Pair the public and tool bytes across two restore boundaries."""
    import urllib.request
    from saas_bench.run_state import clone_sql_run
    from test_sql_evidence import event_ids

    runners = [offline_runner(), offline_runner(execution_capture=True)]
    traces = [[], []]
    script_a = "print('registered version A')"
    script_b = "print('registered version B')"
    for week in range(6):
        for index, runner in enumerate(runners):
            execute = runner.tool_executor.execute
            if week == 0:
                traces[index].append(execute('write_file', {'path': 'comparison.txt', 'content': 'week 0'}))
                traces[index].append(execute('read_file', {'path': 'comparison.txt'}))
                traces[index].append(execute('search_files', {'pattern': 'week', 'path': 'comparison.txt'}))
                traces[index].append(execute('glob_files', {'pattern': 'comparison.txt'}))
                traces[index].append(runner._http_post('/daily-scripts', {'name': 'metric', 'content': script_a}))
            else:
                traces[index].append(execute('edit_file', {'path': 'comparison.txt',
                    'old_string': f'week {week - 1}', 'new_string': f'week {week}'}))
            if week == 3:
                traces[index].append(runner._http_post('/daily-scripts', {'name': 'metric', 'content': script_b}))
            if week == 5:
                request = urllib.request.Request(runner._server_url('/daily-scripts'),
                    data=b'{"name":"metric"}', headers={'Content-Type': 'application/json'}, method='DELETE')
                with urllib.request.urlopen(request, timeout=10) as response:
                    traces[index].append(json.loads(response.read()))
            traces[index].append(runner._http_post('/query', {'sql': 'SELECT 7 AS n'}))
            traces[index].append(runner._http_post('/query', {'sql': 'SELECT 7 AS n WHERE 0'}))
            traces[index].append(execute('bash', {'command': './novamind-operation query "SELECT 7 AS n"'}))
            traces[index].append(execute('bash', {'command':
                "python -c \"import sys; print('stdout'); print('stderr', file=sys.stderr); sys.exit(3)\""}))
            traces[index].append(advance(runner))
            traces[index].append(runner._get_dashboard())
            traces[index].append(runner._get_dashboard())
        assert traces[0] == traces[1]
        assert runners[0]._get_game_status() == runners[1]._get_game_status()
        if week in (2, 4):
            day = (week + 1) * 7
            for index, runner in enumerate(runners):
                if day == 21:
                    runner.agent._refresh_context(f'day {day}', day)
                    runner.agent.current_day = day
                runner._save_checkpoint(day)
                runner._stop_server()
                runners[index] = offline_runner(runner.workspace_dir)
            assert runners[0]._load_checkpoint()['context_boundary'] == runners[1]._load_checkpoint()['context_boundary']
            assert runners[0]._load_checkpoint()['context_boundary'] == ('same_week' if day == 21 else 'new_week')
            assert runners[0]._get_dashboard() == runners[1]._get_dashboard()
    for runner in runners:
        runner._save_checkpoint(42)
    assert business_state(runners[0]) == business_state(runners[1])
    assert runners[0]._http_get('/daily-scripts') == runners[1]._http_get('/daily-scripts')
    assert runners[0].agent_workspace.joinpath('comparison.txt').read_bytes() == runners[1].agent_workspace.joinpath('comparison.txt').read_bytes()
    assert runners[0].agent_workspace.joinpath('comparison.txt').read_bytes() == b'week 5'
    assert (checkpoint_directory(runners[0].workspace_dir, runners[0]._load_checkpoint()) /
        'server_state.json').read_bytes() == (checkpoint_directory(runners[1].workspace_dir,
        runners[1]._load_checkpoint()) / 'server_state.json').read_bytes()
    store = runners[1].evidence_store
    store.assert_healthy()
    kinds = {store.read_event(event)['request']['kind'] for event in event_ids(store)}
    assert {'bash', 'sql_query', 'registered_script_execution', 'public_http'} <= kinds
    destination = os.environ.get('CEOBENCH_STAGE2_ARTIFACTS')
    if destination:
        output = Path(destination)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'six-week-pair.json').write_text(json.dumps({
            'evidence_kind': 'actually_executed_local_offline', 'weeks': 6,
            'split_days': [21, 35], 'compared_actions': len(traces[0]),
            'exact_tool_and_public_values': True, 'business_state_equal': True,
            'event_kinds': sorted(kinds), 'captured_run_id': runners[1].run_id,
        }, indent=2))
        (output / 'comparison-values.json').write_text(json.dumps([
            {'index': number, 'uncaptured': left, 'captured': right, 'equal': left == right}
            for number, (left, right) in enumerate(zip(*traces))
        ], indent=2, ensure_ascii=False))
        metadata = []
        for runner in runners:
            checkpoint = runner._load_checkpoint()
            conn = load_session_db(checkpoint_directory(runner.workspace_dir, checkpoint) / 'world.nmdb')
            try:
                submitted = [row[0] for row in conn.execute('SELECT submitted_at FROM predictions ORDER BY rowid')]
            finally:
                conn.close()
            metadata.append({'run_id': runner.run_id, 'snapshot_id': checkpoint['snapshot_id'],
                             'prediction_submitted_at': submitted})
        (output / 'comparison-metadata.json').write_text(json.dumps(metadata, indent=2))
        store.snapshot(output / 'six-week-evidence.sqlite')


def test_stage2_workspace_overhead(offline_runner, monkeypatch):
    """Opt-in native 1x/10x workload with paired, alternating measurements."""
    import hashlib
    import httpx
    import platform
    import shlex
    import statistics
    import time
    from collections import defaultdict
    from openai import OpenAI
    from saas_bench import execution_capture
    from saas_bench.model_usage import ModelUsage
    from test_preflight_usage import reply
    from test_sql_evidence import event_ids

    destination = os.environ.get('CEOBENCH_STAGE2_ARTIFACTS')
    if not destination:
        pytest.skip('Set CEOBENCH_STAGE2_ARTIFACTS for the stage 2 native workload')
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    report = {'evidence_kind': 'actually_executed_local_synthetic_workload',
              'external_provider_calls': 0, 'platform': platform.platform(),
              'python': sys.version, 'historical_basis': {'files': 124, 'bytes': 2480041},
              'scales': {}}
    code = '''import json, time
import novamind_api as n
from novamind_api import _capture
times = []
submit = _capture.submit
def measured(record):
    start = time.perf_counter()
    try:
        return submit(record)
    finally:
        times.append(time.perf_counter() - start)
_capture.submit = measured
print(n.query('SELECT 7 AS n'))
open('stage2-callback.json', 'w').write(json.dumps(times))'''
    command = './novamind-operation python-c ' + shlex.quote(code)
    for scale in (1, 10):
        runners = [offline_runner(), offline_runner(execution_capture=True)]
        count, byte_count = 124 * scale, 2480041 * scale
        for runner in runners:
            for number in range(count):
                size = byte_count // count + (byte_count % count if number == count - 1 else 0)
                block = hashlib.sha256(f'stage2-{number}'.encode()).digest()
                (runner.agent_workspace / f'stage2-load-{number:04d}.bin').write_bytes(
                    (block * ((size + len(block) - 1) // len(block)))[:size])
        store = runners[1].evidence_store
        phase = defaultdict(float)
        def timed(obj, name, key):
            original = getattr(obj, name)
            def call(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    phase[key] += time.perf_counter() - start
            monkeypatch.setattr(obj, name, call)
            return original
        original_digest = timed(execution_capture, 'digest', 'hash')
        original_version = timed(store, 'version', 'storage')
        original_mapping = timed(execution_capture, 'model_request', 'request_mapping')
        usage = [ModelUsage(None, 'agent', evidence_store=runner.evidence_store) for runner in runners]
        clients = [usage[i].attach(OpenAI(api_key='offline-only', max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=reply('chat')))))) for i in range(2)]
        samples = {False: [], True: []}
        try:
            def measure(index, warmup=False):
                runner = runners[index]
                before = dict(phase)
                size_before = sum(p.stat().st_size for p in
                    (store.path, Path(str(store.path) + '-wal')) if p.exists())
                known = set(event_ids(store)) if index else set()
                start = time.perf_counter()
                result = runner.tool_executor.execute('bash', {'command': command})
                request = dict(model='test-model', messages=[{'role': 'tool',
                    'tool_call_id': 'stage2-fixture', 'content': result}])
                usage[index].call('chat', request,
                    lambda: clients[index].chat.completions.create(**request))
                total = time.perf_counter() - start
                assert 'row_count' in result and 'timed out' not in result.lower()
                callback = sum(json.loads((runner.agent_workspace / 'stage2-callback.json').read_text()))
                row = {'total_seconds': total, 'callback_roundtrip_seconds': callback}
                if index:
                    events = set(event_ids(store)) - known
                    bash = next(store.read_event(event) for event in events
                                if store.read_event(event)['request']['kind'] == 'bash')
                    facts = bash['result']
                    assert facts['status'] == 'succeeded'
                    row.update(before_scan_seconds=facts['before_scan_seconds'],
                               after_scan_seconds=facts['after_scan_seconds'])
                    row.update({key + '_seconds': phase[key] - before.get(key, 0)
                        for key in ('hash', 'storage', 'request_mapping')})
                    row['storage_growth_bytes'] = sum(p.stat().st_size for p in
                        (store.path, Path(str(store.path) + '-wal')) if p.exists()) - size_before
                    if warmup:
                        items = json.loads(store.get_content(
                            bash['request']['event_id'] + ':workspace_before')[1])
                        report['scales'][str(scale)] = {'synthetic_files': count,
                            'synthetic_bytes': byte_count,
                            'scanned_files': sum(x['type'] == 'file' for x in items.values()),
                            'scanned_bytes': sum(x['size'] for x in items.values() if x['type'] == 'file')}
                if not warmup:
                    samples[bool(index)].append(row)
            measure(0, warmup=True)
            measure(1, warmup=True)
            for repeat in range(5):
                for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
                    measure(index)
            for index in (0, 1):
                values = samples[bool(index)]
                report['scales'][str(scale)]['capture_on' if index else 'capture_off'] = {
                    'raw': values,
                    'median': {key: statistics.median(v[key] for v in values) for key in values[0]},
                    'maximum': {key: max(v[key] for v in values) for key in values[0]}}
            store.assert_healthy()
        finally:
            for client in clients:
                client.close()
            monkeypatch.setattr(execution_capture, 'digest', original_digest)
            monkeypatch.setattr(execution_capture, 'model_request', original_mapping)
            monkeypatch.setattr(store, 'version', original_version)
    (output / 'overhead.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))


@pytest.mark.parametrize('action,body', [
    ('price', {'tool': 'set_prices', 'args': {'A': 31}}),
    ('research', {'tool': 'start_research_project', 'args': {'tier': 1}}),
    ('week', {'rationale': 'fixed offline action', 'predictions': {
        h: {'point': 100000, 'lower': -100000, 'upper': 1000000}
        for h in ('cash_1wk', 'cash_4wk', 'cash_12wk', 'cash_26wk')}}),
])
def test_crashed_outer_action_is_never_replayed(offline_runner, monkeypatch, tmp_path, action, body):
    """Kill the client after a real world commit, before the public response is sent."""
    from test_sql_evidence import event_ids
    import time
    endpoint = '/next-week' if action == 'week' else '/call'
    marker, release = tmp_path / 'response-pending', tmp_path / 'release-response'
    monkeypatch.setenv('CEOBENCH_TEST_PAUSE_RESPONSE_PATH', endpoint)
    monkeypatch.setenv('CEOBENCH_TEST_RESPONSE_MARKER', str(marker))
    monkeypatch.setenv('CEOBENCH_TEST_RESPONSE_RELEASE', str(release))
    runner = offline_runner(execution_capture=True)
    runner._save_checkpoint(0)
    pointer = (runner.workspace_dir / 'checkpoint.json').read_bytes()
    runner._begin_operation(action, 0)
    code = '''import json, os, sys, urllib.request
request = urllib.request.Request(sys.argv[1], data=sys.argv[2].encode(),
    headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=120) as response:
    assert json.loads(response.read())['success']
os._exit(77)
'''
    child = subprocess.Popen([sys.executable, '-c', code, runner._server_url(endpoint), json.dumps(body)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 30
        while not marker.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(.01)
        assert marker.exists(), child.communicate(timeout=2)
        if action == 'price':
            state = runner._http_post('/query', {'sql': 'SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1'})
            assert state['rows'][0]['price_A'] == 31
        elif action == 'research':
            state = runner._http_post('/query', {'sql': 'SELECT project_id FROM research_projects WHERE tier=1'})
            assert len(state['rows']) == 1
        else:
            assert runner._get_game_status()['day'] == 7
        child.kill()
        child.communicate(timeout=5)
        assert child.returncode < 0
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=5)
        release.write_text('continue')
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        relevant = [runner.evidence_store.read_event(event) for event in event_ids(runner.evidence_store)
                    if runner.evidence_store.read_event(event)['request']['kind'] == 'public_http'
                    and runner.evidence_store.read_event(event)['request']['request']['path'] == endpoint]
        if relevant and relevant[0]['result']['status'] == 'succeeded':
            break
        time.sleep(.01)
    relevant = [runner.evidence_store.read_event(event) for event in event_ids(runner.evidence_store)
                if runner.evidence_store.read_event(event)['request']['kind'] == 'public_http'
                and runner.evidence_store.read_event(event)['request']['request']['path'] == endpoint]
    assert len(relevant) == 1
    assert relevant[0]['result']['status'] == 'succeeded'
    assert relevant[0]['delivery']['receive_state'] == 'unknown'
    assert (runner.workspace_dir / 'checkpoint.json').read_bytes() == pointer
    with pytest.raises(ValueError, match='outcome unknown'):
        runner._load_checkpoint()
    with pytest.raises(ValueError, match='outcome unknown'):
        offline_runner(runner.workspace_dir)
    destination = os.environ.get('CEOBENCH_STAGE2_ARTIFACTS')
    if destination:
        output = Path(destination)
        output.mkdir(parents=True, exist_ok=True)
        (output / f'crash-{action}.json').write_text(json.dumps({
            'evidence_kind': 'actually_executed_local_process_exit',
            'action': action, 'fault_boundary': 'world_mutated_before_public_response',
            'child_exit_code': child.returncode, 'public_attempts': len(relevant),
            'public_result_status': relevant[0]['result']['status'],
            'receive_state': relevant[0]['delivery']['receive_state'],
            'checkpoint_unchanged': True, 'restore_refused': True}, indent=2))


def test_git_workspace_maintenance_is_synchronous(offline_runner):
    runner = offline_runner(execution_capture=True)
    for option in ('gc.autoDetach', 'maintenance.autoDetach'):
        assert runner._git('config', '--get', option).stdout.strip() == 'false'
        runner._git('config', option, 'true', check=True)
    runner._git_init_workspace()  # Existing repositories need the same boundary rule.
    for option in ('gc.autoDetach', 'maintenance.autoDetach'):
        assert runner._git('config', '--get', option).stdout.strip() == 'false'
    for i in range(3):
        result = runner.tool_executor.execute('bash', {
            'command': f'git commit --allow-empty -q -m "foreground maintenance {i}" && echo committed'})
        assert 'committed' in result
    assert getattr(runner.tool_executor, 'preserved_process', None) is None
    runner.evidence_store.assert_healthy()


def test_harness_preserves_unfinished_descendants_and_server(offline_runner, monkeypatch):
    import signal
    from saas_bench.environment import Action
    runner = offline_runner(execution_capture=True)
    runner.tool_executor.bash_timeout = .3
    runner._save_checkpoint(0)
    pointer = (runner.workspace_dir / 'checkpoint.json').read_bytes()
    monkeypatch.setattr(runner, 'setup', lambda: None)
    monkeypatch.setattr(runner.agent, 'act', lambda *args: Action(tool='bash',
        arguments={'command': 'sleep 60 >/dev/null 2>&1 & echo parent-returned'}))
    try:
        with pytest.raises(RuntimeError, match='descendants did not finish before the command deadline'):
            runner.run(verbose=False)
        assert runner._server_proc.poll() is None
        assert runner.tool_executor.preserved_process.poll() is None
        assert (runner.workspace_dir / 'checkpoint.json').read_bytes() == pointer
        assert json.loads((runner.workspace_dir / 'sql-evidence.fault.json').read_text())['preserve_scene']
        with pytest.raises(ValueError, match='unknown'):
            runner._load_checkpoint()
    finally:
        proc = runner.tool_executor.preserved_process
        if proc:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate(timeout=5)
