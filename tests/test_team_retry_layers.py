"""The sealed Git/PF clones delegate every physical retry to ModelUsage."""

import json

import httpx

from saas_bench import team_batch as batch
from saas_bench.role_policy import ROLES
from test_preflight_team_batch import frozen
from test_preflight_team_flow import calls, completion


def test_same_seed_clones_retry_each_role_in_one_layer(frozen, tmp_path, monkeypatch):
    _, manifest, _, _ = frozen
    seed = manifest['runs'][0]['seed']
    entries = [next(entry for entry in manifest['runs']
                    if entry['seed'] == seed and entry['group'] == group)
               for group in ('git', 'pf')]
    assert entries[0]['d0'] == entries[1]['d0']
    received = {}

    def factory(group):
        def client_for_role(role):
            wire = received[group, role] = []

            def respond(request):
                assert client.max_retries == 0
                wire.append(request)
                assert len(wire) <= 5
                if len(wire) < 5:
                    return httpx.Response(503, json={'error': {'message': 'temporary provider overload'}})
                body = json.loads(request.content)
                return completion(calls(('read_file', {'path': 'MEMORY.md'})), body.get('stream', False))

            client = batch.checked_client(role, f'retry-layers-{group}',
                                          transport=httpx.MockTransport(respond))
            return client

        return client_for_role

    teams = []
    try:
        for entry in entries:
            team = batch.make_team(manifest, entry, tmp_path / entry['group'],
                                   client_factory=factory(entry['group']))
            teams.append(team)
            world_before = team.server.conn.serialize()
            for role in ROLES:
                runtime = team.roles[role]
                waits = []
                runtime.usage.lifecycle = runtime.agent.lifecycle
                monkeypatch.setattr(runtime.usage.lifecycle, 'sleep', waits.append)
                assert runtime.agent.client.max_retries == 0

                action = runtime.agent.act('Inspect the existing workspace.', 0, False, {'day': 0})

                assert action.tool == 'read_file' and action.arguments == {'path': 'MEMORY.md'}
                assert runtime.agent.client.max_retries == 0
                wire = received[entry['group'], role]
                assert len(wire) == 5
                assert [request.headers['x-stainless-retry-count'] for request in wire] == ['0'] * 5
                assert all(request.content == wire[0].content for request in wire)
                assert waits == [10, 20, 40, 80]

                rows = [json.loads(line) for line in runtime.usage.path.read_text().splitlines()]
                logical_requests = [row for row in rows if row['event'] == 'request']
                logical_responses = [row for row in rows if row['event'] == 'response']
                attempts = [row for row in rows if row['event'] == 'http_request']
                responses = [row for row in rows if row['event'] == 'http_response']
                assert len(logical_requests) == len(logical_responses) == 1
                assert logical_requests[0]['outer_attempt'] == 1
                assert logical_responses[0]['model_attempts'] == 5
                assert logical_responses[0]['error'] is None
                assert len({row['call_id'] for row in rows}) == 1
                assert len(attempts) == len({row['attempt_id'] for row in attempts}) == 5
                assert {row['attempt_id'] for row in attempts} == {row['attempt_id'] for row in responses}
                assert [row['status'] for row in responses] == [503] * 4 + [200]
                assert runtime.usage.summary['calls'] == 1
                assert runtime.usage.summary['http_attempts'] == 5
                assert runtime.usage.summary['failed_http_attempts'] == 4
            assert team.server.tools.current_day == 0
            assert team.server.conn.serialize() == world_before
        assert set(received) == {(group, role) for group in ('git', 'pf') for role in ROLES}
    finally:
        for team in teams:
            batch.close_team(team)
