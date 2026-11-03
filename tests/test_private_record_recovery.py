"""Standalone recovery keeps reference allocation outside selected snapshots."""
import json

import pytest

from saas_bench.run_state import checkpoint_directory, tree_hash
from test_preflight_integration import offline_runner, packed_public
from test_text_registry import declaration, receipt


def register(runner, text):
    return receipt(runner._execute_tool('text_create', declaration(text=text)))


def test_repeated_real_standalone_recovery_does_not_rebind_discarded_refs(offline_runner):
    original = offline_runner(text_registration='git')
    original.agent.current_day = 0
    assert register(original, 'retained text')['version'] == 'r1.1'
    original._save_checkpoint(0)
    selected = checkpoint_directory(original.workspace_dir, original._load_checkpoint())
    frozen = tree_hash(selected, exact=True)
    assert register(original, 'discarded original text')['version'] == 'r2.1'
    original._stop_server()
    first = offline_runner(original.workspace_dir)
    assert register(first, 'first recovery text')['version'] == 'r3.1'
    first._stop_server()
    second = offline_runner(original.workspace_dir)
    assert register(second, 'second recovery text')['version'] == 'r4.1'
    assert second.tool_executor.public_refs.path == original.tool_executor.public_refs.path
    assert not original.tool_executor.public_refs.path.is_relative_to(original.workspace_dir)
    assert tree_hash(selected, exact=True) == frozen
    state = second.tool_executor.text_registry._load()['records']
    assert set(state) == {'r1', 'r4'}
    result = second._execute_tool('text_create', declaration({'record': 'r2.1'}))
    assert result.startswith('Error:') and 'Unknown registered text revision' in result


@pytest.mark.parametrize('missing', ['authority', 'locator'])
def test_current_standalone_checkpoint_cannot_restart_numbering(offline_runner, missing):
    original = offline_runner(text_registration='git')
    original.agent.current_day = 0
    register(original, 'retained text')
    original._save_checkpoint(0)
    original._stop_server()
    checkpoint = json.loads((original.workspace_dir / 'checkpoint.json').read_text())
    if missing == 'authority':
        authority = original.tool_executor.public_refs.path
        authority.rename(authority.with_suffix('.unavailable'))
    else:
        checkpoint.pop('public_refs', None)
        (original.workspace_dir / 'checkpoint.json').write_text(json.dumps(checkpoint))
    recovered = offline_runner(original.workspace_dir)
    assert recovered.tool_executor.public_refs.unavailable
    result = recovered._execute_tool('text_create', declaration(text='must not reuse numbers'))
    assert result.startswith('Error:') and 'allocation history' in result
    assert set(recovered.tool_executor.text_registry._load()['records']) == {'r1'}


def test_standalone_operation_allocator_is_shared_with_server_across_recovery(offline_runner):
    def operate(runner, spend):
        response = runner._http_post('/call', {'tool': 'set_targeted_ad_spend', 'args': {'targeted_spend': {'social_media': {'S1': spend}}}})
        assert response['success'], response
        return response['operation']['operation_ref']
    original = offline_runner(text_registration='git')
    original.agent.current_day = 0
    assert operate(original, 3) == 'ceo:1'
    original._save_checkpoint(0)
    assert operate(original, 4) == 'ceo:2'
    original._stop_server()
    first = offline_runner(original.workspace_dir)
    assert operate(first, 5) == 'ceo:3'
    first._stop_server()
    second = offline_runner(original.workspace_dir)
    assert operate(second, 6) == 'ceo:4'
