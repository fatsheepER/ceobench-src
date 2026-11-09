"""Provider changes must not reuse another account's same-named model price."""
import json

import pytest

from saas_bench.agents.bash_agent.run_test import BashAgentRunner
from saas_bench.run_state import checkpoint_directory, tree_hash
from test_chat_provider_switching import endpoint, isolated_environment


@pytest.mark.parametrize('selection', ['implicit', 'explicit', 'same_provider'])
def test_public_checkpoint_same_alias_pricing_is_provider_scoped(tmp_path, endpoint, selection):
    url, wire = endpoint
    alias = 'shared-provider-model-alias'
    old = dict(source='offline original provider rates', basis='USD per 1000 tokens', rates={alias:
        dict(input=.001, output=.002, cache_read=.0001, cache_write=.001)})
    new = dict(source='offline explicitly selected provider rates', basis='USD per 1000 tokens', rates={alias:
        dict(input=.003, output=.004, cache_read=.0002, cache_write=.002)})
    old_path, new_path = tmp_path / 'old-pricing.json', tmp_path / 'new-pricing.json'
    old_path.write_text(json.dumps(old))
    new_path.write_text(json.dumps(new))
    source = BashAgentRunner(provider='opendesign', model=alias, base_url=url + 'source/',
        pricing_file=old_path, workspace_base=tmp_path / 'source')
    resumed = None
    try:
        source.setup()
        assert source.agent.act('priced original dashboard', 0, False, {'day': 0}).tool == 'read_file'
        source.agent.record_tool_result('completed original fixture read')
        source._save_checkpoint(0)
        source._stop_server()
        original_manifest = (source.workspace_dir / 'manifest.json').read_bytes()
        original_log = source.agent.usage_recorder.path.read_bytes()
        historical = [row for row in map(json.loads, original_log.splitlines()) if row['event'] == 'response']
        assert len(historical) == 1 and historical[0]['cost_usd'] > 0
        original_cost = source.agent.usage_recorder.summary['known_cost_usd']
        snapshot = checkpoint_directory(source.workspace_dir, source._load_checkpoint())
        original_snapshot = tree_hash(snapshot, exact=True)
        provider = 'opendesign' if selection == 'same_provider' else 'deepseek'
        resumed = BashAgentRunner(continue_from=source.workspace_dir, provider=provider, model=alias,
            base_url=url + 'resumed/', pricing_file=new_path if selection == 'explicit' else None,
            workspace_base=tmp_path / 'resumed')
        resumed.setup()
        resumed._restore_from_checkpoint(resumed._load_checkpoint())
        manifest = json.loads((resumed.workspace_dir / 'manifest.json').read_text())
        expected = new['rates'] if selection == 'explicit' else old['rates'] if selection == 'same_provider' else {}
        assert resumed._pricing == manifest['benchmark_config']['model_pricing'] == expected
        assert resumed.agent.usage_recorder.summary['known_cost_usd'] == original_cost
        assert resumed.agent.act('continued fixture read', 0, False, {'day': 0}).tool == 'read_file'
        assert wire[-1]['path'] == '/resumed/chat/completions' and wire[-1]['body']['model'] == alias
        entries = [json.loads(line) for line in resumed.agent.usage_recorder.path.read_text().splitlines()]
        responses = [row for row in entries if row['event'] == 'response']
        assert responses[:-1] == historical
        current = responses[-1]
        if selection == 'implicit':
            assert current['cost_usd'] is None and current['pricing'] is None
            assert resumed.agent.usage_recorder.summary['known_cost_usd'] == original_cost
            assert resumed.agent.usage_recorder.summary['missing_cost'] == 1
            assert any(row['field'] == 'pricing' and row['old'] == old
                and row['new']['rates'] == {} for row in resumed.configuration_overrides)
        else:
            assert current['pricing'] == expected[alias] and current['cost_usd'] > 0
            assert resumed.agent.usage_recorder.summary['known_cost_usd'] > original_cost
        assert (source.workspace_dir / 'manifest.json').read_bytes() == original_manifest
        assert source.agent.usage_recorder.path.read_bytes() == original_log
        assert tree_hash(snapshot, exact=True) == original_snapshot
    finally:
        for runner in (source, resumed):
            if runner:
                runner._stop_server()
                runner.client.close()
