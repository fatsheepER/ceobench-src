"""Public operation identities remain short and survive selected-point rollback."""
import json
import re
import shlex

import pytest

from saas_bench.multi_agent import MultiAgentRuntime
from test_business_ledger_public import effects, query, records
from test_preflight_team_checkpoint import clients, dispose, fast_simulator, make_team as checkpoint_team
from test_preflight_team_flow import ADVANCE, calls, final, make_team, raw


def operate(team, tier):
    team._set_phase('analysis')
    status, response = raw(team, 'growth', '/call', {'tool': 'set_capacity_tier', 'args': {'tier': tier}})
    assert status == 200 and response['success'], response
    return response['operation']['operation_ref']


def declare_operation(team, ref):
    result = team.roles['growth'].executor.execute('escalate', {'items': [{
        'item_id': 'operation-source', 'op': 'create', 'target': {'operation_ref': ref},
        'sources': [{'ref': {'missing': {'description': 'Initial observation', 'reason': 'No saved source'}}}],
        'assertion': 'author_declared', 'reason': 'Record the exact business decision.'}]})
    assert json.loads(result)['saved'] == 1, result
    return result


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_short_operation_refs_reach_sdk_cli_sql_and_actual_model_input(mode):
    count = 0
    program = "import json; from novamind_api import infrastructure; print(json.dumps(infrastructure.set_capacity_tier(3)))"

    def respond(team, role, body):
        nonlocal count
        if role == 'ceo':
            return calls(('bash', {'command': ADVANCE}))
        if role == 'growth':
            count += 1
            if count == 1:
                return calls(('bash', {'command': './novamind-operation python-c ' + shlex.quote(program)}))
        return final('Done.')

    with make_team(mode, respond, 'short-operations') as team:
        assert team.run(stop_after_day=7).reason == 'observation_end', team.failure
        assert [(r['actor'], r['operation_ref']) for r in records(team)] == [
            ('growth', 'growth:1'), ('ceo', 'ceo:1')]
        usage = [json.loads(line) for line in team.roles['growth'].usage.path.read_text().splitlines()]
        sent = json.dumps([row['body'] for row in usage if row['event'] == 'http_request'])
        assert 'growth:1' in sent and not re.search(r'op:[0-9a-f]{32}', sent)
        for role in ('ceo', 'growth', 'ops_finance'):
            nested = query(team, "SELECT json_object('operation',o.operation_ref,'effect',e.effect_ref) AS nested "
                "FROM business_operations o JOIN business_effects e USING(operation_ref) "
                "WHERE o.operation_ref='growth:1'", role)
            assert nested and all(json.loads(row['nested'])['operation'] == 'growth:1' for row in nested)
            assert all(json.loads(row['nested'])['effect'].startswith('growth:1/effect:') for row in nested)
        assert operate(team, 4) == 'growth:2'
        declare_operation(team, 'growth:1')
        if mode == 'pf':
            output = team.roles['ops_finance'].executor.execute('bash', {'command': 'pf depend growth:1 --detail'})
            assert 'Initial observation' in output and 'growth:1' in output, output


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_repeated_old_checkpoint_restore_never_rebinds_discarded_operation_number(tmp_path, mode):
    original = checkpoint_team(tmp_path / 'original', mode)
    restored = []
    try:
        assert operate(original, 3) == 'growth:1'
        declare_operation(original, 'growth:1')
        original._set_phase('boundary')
        checkpoint = original.checkpoint()
        assert operate(original, 4) == 'growth:2'
        original_rows = records(original)
        for expected in (3, 4):
            team = MultiAgentRuntime.restore(checkpoint, tmp_path / f'restored-{expected}',
                client_factory=clients(), simulator_factory=fast_simulator)
            restored.append(team)
            assert [row['operation_ref'] for row in records(team)] == ['growth:1']
            assert operate(team, expected + 2) == f'growth:{expected}'
            assert query(team, "SELECT operation_ref FROM business_operations WHERE operation_ref='growth:2'") == []
            assert all(row['operation_ref'] != 'growth:2' for row in effects(team))
            if mode == 'pf':
                team._set_phase('ceo')
                output = team.roles['ceo'].executor.execute('bash', {'command': 'pf depend growth:1 --detail'})
                assert 'Initial observation' in output and 'growth:1' in output, output
        assert records(original) == original_rows
    finally:
        for team in restored:
            dispose(team)
        dispose(original)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_missing_live_allocator_restores_reads_but_holds_before_new_business_write(tmp_path, mode):
    original = checkpoint_team(tmp_path / 'original', mode)
    restored = None
    try:
        assert operate(original, 3) == 'growth:1'
        original._set_phase('boundary')
        checkpoint = original.checkpoint()
        assert operate(original, 4) == 'growth:2'
        allocator = original.public_refs.path
        allocator.rename(allocator.with_suffix('.preserved'))
        restored = MultiAgentRuntime.restore(checkpoint, tmp_path / 'restored',
            client_factory=clients(), simulator_factory=fast_simulator)
        assert [row['operation_ref'] for row in records(restored)] == ['growth:1']
        restored._set_phase('analysis')
        before = restored.server.conn.serialize()
        status, response = raw(restored, 'growth', '/call', {'tool': 'set_capacity_tier', 'args': {'tier': 5}})
        assert status == 500, response
        assert response['error'] == 'operation_reference_unavailable'
        assert response['manual_resume_required'] is True
        assert 'allocation history' in json.dumps(response)
        assert restored.server._operation_failed
        assert restored.server.conn.serialize() == before
        assert not allocator.exists()
    finally:
        if restored:
            dispose(restored)
        dispose(original)


def test_allocator_serializes_parallel_roles_and_reopens_durable_reservations(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from saas_bench.public_refs import PublicRefs
    refs = PublicRefs(tmp_path / 'private' / 'refs.sqlite')
    with ThreadPoolExecutor(max_workers=6) as pool:
        allocated = list(pool.map(lambda i: refs.assign('operation', 'growth', f'private:{i}'), range(30)))
    assert sorted(int(ref.split(':')[1]) for ref in allocated) == list(range(1, 31))
    second = PublicRefs(refs.path)
    assert [second.assign('operation', 'growth', f'private:{i}') for i in range(30)] == allocated
    assert second.assign('operation', 'ceo', 'private:ceo') == 'ceo:1'
    assert second.assign('operation', 'growth', 'private:next') == 'growth:31'
    assert second.resolve('operation', allocated[0]) == 'private:0'


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_legacy_checkpoint_ledger_gets_short_refs_without_changing_original(tmp_path, mode):
    original = checkpoint_team(tmp_path / 'original', mode)
    restored = None
    legacy = 'op:' + 'a' * 32
    try:
        assert operate(original, 3) == 'growth:1'
        conn = original.server.conn
        conn.execute('PRAGMA defer_foreign_keys=ON')
        conn.execute('UPDATE business_operations SET operation_ref=?', (legacy,))
        conn.execute("UPDATE business_effects SET operation_ref=?,effect_ref=? || substr(effect_ref,9)", (legacy, legacy))
        conn.commit()
        original._set_phase('boundary')
        checkpoint = original.checkpoint()
        metadata = checkpoint / 'team.json'
        state = json.loads(metadata.read_text())
        del state['public_refs']
        state.pop('public_history_format', None)
        metadata.write_text(json.dumps(state))
        original_state = conn.serialize()
        restored = MultiAgentRuntime.restore(checkpoint, tmp_path / 'restored',
            client_factory=clients(), simulator_factory=fast_simulator)
        rows = records(restored)
        assert len(rows) == 1 and rows[0]['operation_ref'] == 'growth:2'
        assert rows[0]['actor'] == 'growth' and rows[0]['tool'] == 'set_capacity_tier'
        assert json.loads(rows[0]['parameters_json']) == {'tier': 3}
        changed = effects(restored)
        assert changed and all(row['operation_ref'] == 'growth:2' for row in changed)
        assert all(row['effect_ref'].startswith('growth:2/effect:') for row in changed)
        assert legacy not in json.dumps(rows + changed)
        assert conn.serialize() == original_state
    finally:
        if restored:
            dispose(restored)
        dispose(original)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_failed_ledger_admission_reserves_number_before_business_work(tmp_path, mode):
    original = checkpoint_team(tmp_path / 'original', mode)
    restored = None
    try:
        checkpoint = original.checkpoint()
        original._set_phase('analysis')
        conn = original.server.conn
        before = conn.execute('SELECT capacity_tier FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0]
        conn.execute("CREATE TEMP TRIGGER reject_operation BEFORE INSERT ON business_operations BEGIN SELECT RAISE(ABORT,'operation admission fault'); END")
        status, response = raw(original, 'growth', '/call', {'tool': 'set_capacity_tier', 'args': {'tier': 3}})
        assert status == 500, response
        assert original.server._operation_failed
        assert conn.execute('SELECT capacity_tier FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] == before
        assert conn.execute('SELECT COUNT(*) FROM business_operations').fetchone()[0] == 0
        assert original.public_refs.resolve('operation', 'growth:1').startswith('op:')
        restored = MultiAgentRuntime.restore(checkpoint, tmp_path / 'restored',
            client_factory=clients(), simulator_factory=fast_simulator)
        assert operate(restored, 4) == 'growth:2'
        assert [row['operation_ref'] for row in records(restored)] == ['growth:2']
    finally:
        if restored:
            dispose(restored)
        dispose(original)
