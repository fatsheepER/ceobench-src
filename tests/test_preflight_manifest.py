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


def test_simulator_restores_checkpoint_configuration_after_environment_drift(tmp_path, monkeypatch):
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
    assert restored.social_post_llm_model == restored.enterprise_llm_model == 'test-model'


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
    assert changed._pricing == {'test-model': {'input': .001}}
    assert changed.workspace_dir != first.workspace_dir
    assert (first.workspace_dir / 'manifest.json').read_bytes() == original
