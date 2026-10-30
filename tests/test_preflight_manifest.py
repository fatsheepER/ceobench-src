import json
import platform

import pytest

from saas_bench.run_state import artifact_hashes, verify_build, write_json, runtime_versions, tree_hash, file_hash
from test_preflight_integration import offline_runner, packed_public


@pytest.mark.parametrize('changed', ['docs/novamind_api/client.py', 'novamind-client'])
def test_artifact_drift_is_recorded_and_manifest_writes_are_atomic(tmp_path, changed, caplog):
    (tmp_path / 'docs' / 'novamind_api').mkdir(parents=True)
    (tmp_path / 'novamind-operation').write_text('bundle')
    (tmp_path / 'novamind-client').write_text('client')
    sdk = tmp_path / 'docs' / 'novamind_api' / 'client.py'
    sdk.write_text('sdk')
    original = dict(python=platform.python_version(), artifacts=artifact_hashes(tmp_path), runtime=runtime_versions())
    write_json(tmp_path / 'build.json', original)
    assert verify_build(tmp_path) == original
    with pytest.raises(ValueError):
        write_json(tmp_path / 'build.json', {'bad': float('nan')})
    assert json.loads((tmp_path / 'build.json').read_text()) == original
    (tmp_path / changed).write_text('changed')
    assert verify_build(tmp_path) == original
    assert 'artifacts' in caplog.text


def test_build_records_python_sdk_and_source_drift(tmp_path, caplog):
    public = tmp_path / 'public'
    (public / 'docs' / 'novamind_api').mkdir(parents=True)
    (public / 'novamind-operation').write_text('bundle')
    (public / 'novamind-client').write_text('client')
    (tmp_path / 'src').mkdir()
    (tmp_path / 'scripts').mkdir()
    source = tmp_path / 'src' / 'engine.py'
    source.write_text('original')
    builder = tmp_path / 'scripts' / 'build_public.py'
    builder.write_text('builder')
    baseline = dict(python=platform.python_version(), artifacts=artifact_hashes(public), runtime=runtime_versions(),
                    source_sha256=tree_hash(tmp_path / 'src'), builder_sha256=file_hash(builder))
    write_json(public / 'build.json', baseline)
    assert verify_build(public, tmp_path) == baseline
    write_json(public / 'build.json', dict(baseline, python='0.0.0'))
    assert verify_build(public, tmp_path)['python'] == '0.0.0'
    assert 'Python' in caplog.text
    write_json(public / 'build.json', dict(baseline, runtime={}))
    assert verify_build(public, tmp_path)['runtime'] == {}
    assert 'SDK' in caplog.text
    write_json(public / 'build.json', baseline)
    source.write_text('changed')
    assert verify_build(public, tmp_path) == baseline
    assert 'source' in caplog.text


def test_simulator_inherits_saved_configuration_and_honors_explicit_environment(tmp_path, monkeypatch):
    from dataclasses import asdict
    from saas_bench.config import BenchmarkConfig
    from saas_bench.server_entry import _session_config
    config = BenchmarkConfig()
    config.social_post_llm_provider = config.enterprise_llm_provider = 'deepseek'
    config.social_post_llm_model = config.enterprise_llm_model = 'test-model'
    path = tmp_path / 'manifest.json'
    write_json(path, {'benchmark_config': asdict(config)})
    monkeypatch.setenv('CEOBENCH_RUN_MANIFEST', str(path))
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-only')
    monkeypatch.delenv('CEOBENCH_SIMULATOR_LLM_PROVIDER', raising=False)
    monkeypatch.delenv('CEOBENCH_SIMULATOR_LLM_MODEL', raising=False)
    assert json.loads(json.dumps(asdict(_session_config(42, 7, 100)))) == json.loads(json.dumps(asdict(config)))
    monkeypatch.setenv('CEOBENCH_SIMULATOR_LLM_MODEL', 'different-model')
    restored = _session_config(42, 7, 100)
    assert restored.social_post_llm_model == restored.enterprise_llm_model == 'different-model'


def test_single_agent_resume_applies_explicit_configuration_to_requests_and_simulator(
        offline_runner, tmp_path, monkeypatch):
    import httpx
    from saas_bench.agents.bash_agent.run_test import BashAgentRunner
    from saas_bench.run_state import checkpoint_directory
    source = offline_runner()
    source._save_checkpoint(0)
    original = (source.workspace_dir / 'manifest.json').read_bytes()
    monkeypatch.setenv('CEOBENCH_SIMULATOR_LLM_MODEL', 'changed-simulator')
    resumed = BashAgentRunner(continue_from=source.workspace_dir, workspace_base=tmp_path,
        api_key='offline', model='changed-agent', provider='openai', base_url='http://127.0.0.1:9/v1',
        seed=12345, scenario='unrecognized-scenario', total_days=56, initial_cash=250000., reasoning_effort='low')
    try:
        resumed.setup()
        resumed._restore_from_checkpoint(resumed._load_checkpoint())
        resumed._save_checkpoint(0)
        snapshot = checkpoint_directory(resumed.workspace_dir, resumed._load_checkpoint())
        session = json.loads((snapshot / 'session.json').read_text())
        assert session['seed'] == 12345 and session['total_days'] == 56
        assert session['initial_cash'] == 250000. and session['scenario'] == 'unrecognized-scenario'
        assert session['benchmark_config']['total_days'] == 56
        assert session['simulator_llm']['social_post_llm_model'] == 'changed-simulator'
        sent = []
        def respond(_, request):
            sent.append(json.loads(request.read()))
            return httpx.Response(200, json=dict(id='changed', object='response', status='completed',
                model='changed-agent', output=[dict(type='function_call', id='fc1', call_id='call1',
                    name='bash', arguments=json.dumps(dict(command='echo changed')))],
                usage=dict(input_tokens=20, output_tokens=10, input_tokens_details=dict(cached_tokens=0))))
        monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', respond)
        resumed.agent.act('Current day is zero.', 0, False, info={'day': 0})
        assert sent[-1]['model'] == 'changed-agent' and sent[-1]['reasoning']['effort'] == 'low'
        assert '56' in sent[-1]['instructions']
        assert str(resumed.client.base_url) == 'http://127.0.0.1:9/v1/'
        assert resumed.provider == 'openai'
        assert (resumed.seed, resumed.scenario, resumed.total_days, resumed.initial_cash) == (
            12345, 'unrecognized-scenario', 56, 250000.)
        manifest = json.loads((resumed.workspace_dir / 'manifest.json').read_text())
        overrides = {row['field']: row for row in manifest['configuration_overrides']}
        assert overrides['model']['old'] == 'test-model' and overrides['model']['new'] == 'changed-agent'
        assert overrides['social_post_llm_model']['new'] == 'changed-simulator'
        assert overrides['social_post_llm_model']['effective_at'] == 'simulator_start_in_new_recovery_attempt'
        assert (source.workspace_dir / 'manifest.json').read_bytes() == original
    finally:
        resumed._stop_server()
        resumed.client.close()


@pytest.mark.parametrize('source_capture', ['off', 'sql', 'execution'])
def test_single_agent_resume_can_change_capture_without_claiming_historical_coverage(
        offline_runner, tmp_path, source_capture):
    from saas_bench.agents.bash_agent.run_test import BashAgentRunner
    enable = source_capture != 'execution'
    source = offline_runner(sql_capture=source_capture != 'off', execution_capture=source_capture == 'execution',
        text_registration='prefix' if source_capture == 'execution' else 'off')
    source.agent.current_day = 0
    source._save_checkpoint(0)
    resumed = BashAgentRunner(continue_from=source.workspace_dir, workspace_base=tmp_path, api_key='offline',
        execution_capture=enable, sql_capture=enable, text_registration='prefix' if enable else 'off', pf_stale_checks=False)
    try:
        resumed.setup()
        resumed._restore_from_checkpoint(resumed._load_checkpoint())
        assert resumed.execution_capture is enable
        assert (resumed.evidence_store is not None) is enable
        manifest = json.loads((resumed.workspace_dir / 'manifest.json').read_text())
        assert bool(manifest.get('sql_evidence')) is enable
        if enable:
            assert manifest['capture_history']['source_checkpoint_coverage'] == (
                'sql_only' if source_capture == 'sql' else 'unavailable')
            assert resumed.agent.current_day == -1 and resumed.agent.conversation == []
    finally:
        resumed._stop_server()
        resumed.client.close()


def test_recovery_weekly_usage_does_not_recount_old_physical_retries(tmp_path):
    from copy import deepcopy
    from types import SimpleNamespace
    from saas_bench.agents.bash_agent.run_test import BashAgentRunner
    from saas_bench.model_usage import ModelUsage
    summary = ModelUsage(tmp_path / 'requests.jsonl', 'agent', {}).summary
    summary['retried_http_attempts'] = 2
    runner = object.__new__(BashAgentRunner)
    runner.workspace_dir = tmp_path
    runner.agent = SimpleNamespace(usage_recorder=SimpleNamespace(summary=deepcopy(summary)), total_turns=0)
    runner.recovery_source = dict(usage=dict(agent=deepcopy(summary), simulator=deepcopy(summary)), attempt_id='new')
    runner._http_post = lambda *_: dict(day=7, usage=deepcopy(summary))
    runner._weekly_record(dict(day=7))
    current = json.loads((tmp_path / 'usage_summary.json').read_text())['attempt_usage']['agent']
    assert current['calls'] == 0 and current['errors'] == 0 and current['missing_cost'] == 0
    assert set(current['missing'].values()) == {0}


def test_opencode_routes_both_roles_without_deepseek_credentials(tmp_path, monkeypatch):
    from saas_bench.agents.bash_agent.run_test import BashAgentRunner
    from saas_bench.config import BenchmarkConfig
    from saas_bench.server_entry import _apply_simulator_llm_config, _create_simulator_openai_client
    monkeypatch.setattr('saas_bench.agents.bash_agent.run_test.load_env_file', lambda _: {})
    monkeypatch.setenv('OPENCODE_API_KEY', 'go-only-test-key')
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    for suffix in ('PROVIDER', 'MODEL'):
        monkeypatch.delenv('CEOBENCH_SIMULATOR_LLM_' + suffix, raising=False)
    runner = BashAgentRunner(provider='opencode', model='deepseek-v4.1-flash', workspace_base=tmp_path)
    env = runner._server_environment()
    assert env['CEOBENCH_SIMULATOR_LLM_PROVIDER'] == 'opencode'
    assert env['CEOBENCH_SIMULATOR_LLM_MODEL'] == 'deepseek-v4.1-flash'
    for name in ('CEOBENCH_SIMULATOR_LLM_PROVIDER', 'CEOBENCH_SIMULATOR_LLM_MODEL', 'CEOBENCH_MODEL_SESSION'):
        monkeypatch.setenv(name, env[name])
    config = BenchmarkConfig()
    _apply_simulator_llm_config(config)
    client = _create_simulator_openai_client(config)
    assert str(client.base_url) == str(runner.client.base_url) == 'https://opencode.ai/zen/go/v1/'
    assert client.api_key == 'go-only-test-key'
    assert client.default_headers['User-Agent'] == runner.client.default_headers['User-Agent'] == 'CEO-Bench/1.0'
    assert client.default_headers['x-opencode-session'] == runner.client.default_headers['x-opencode-session'] + ':simulator'
    assert runner._server_environment()['CEOBENCH_MODEL_SESSION'] == env['CEOBENCH_MODEL_SESSION']
    config.enterprise_llm_provider = 'deepseek'
    with pytest.raises(ValueError, match='one endpoint'):
        _create_simulator_openai_client(config)
    runner.client.close()
    client.close()


def test_checkpoint_pricing_survives_removed_or_changed_source_file(offline_runner, tmp_path):
    path = tmp_path / 'price.json'
    data = dict(source='https://opencode.ai/docs/go/', basis='USD/1k quota', rates={'test-model': {'input': .001}})
    path.write_text(json.dumps(data))
    first = offline_runner(pricing_file=path)
    first._save_checkpoint(0)
    original = (first.workspace_dir / 'manifest.json').read_bytes()
    path.unlink()
    restored = offline_runner(first.workspace_dir)
    assert restored._pricing == data['rates']
    data['rates']['test-model']['input'] = 9
    path.write_text(json.dumps(data))
    changed = offline_runner(first.workspace_dir, pricing_file=path)
    assert changed._pricing == {'test-model': {'input': 9}}
    assert changed.workspace_dir != first.workspace_dir
    assert (first.workspace_dir / 'manifest.json').read_bytes() == original
