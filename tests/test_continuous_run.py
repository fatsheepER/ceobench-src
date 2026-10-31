from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading

import httpx
from openai import OpenAI
import pytest

from saas_bench.model_usage import ModelUsage
from saas_bench.role_policy import ROLES
from test_preflight_team_checkpoint import fast_simulator
from test_preflight_team_flow import ADVANCE, calls, completion, final, make_team
from test_preflight_usage import reply
from test_team_script_state import register_cli, save_evidence


@pytest.mark.parametrize('role', [*ROLES, 'simulator'])
def test_default_network_retry_continues_past_five_and_missing_usage(tmp_path, monkeypatch, role):
    monkeypatch.delenv('CEOBENCH_MODEL_REQUEST_ATTEMPTS', raising=False)
    requests, waits = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            failed = len(requests) <= 6
            response = {'error': {'message': 'temporary outage'}} if failed else reply('chat')
            response.pop('usage', None)
            body = json.dumps(response).encode()
            self.send_response(503 if failed else 200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    serving = threading.Thread(target=server.serve_forever)
    recorder = ModelUsage(tmp_path / 'usage.jsonl', role)
    monkeypatch.setattr(recorder.lifecycle, 'sleep', waits.append)
    client = recorder.attach(OpenAI(api_key='offline', max_retries=2,
        base_url=f'http://127.0.0.1:{server.server_port}/v1',
        http_client=httpx.Client(timeout=3, trust_env=False)))
    serving.start()
    try:
        request = dict(model='test-model', messages=[dict(role='user', content='unchanged request')])
        for _ in range(2):
            response = recorder.call('chat', request, lambda: client.chat.completions.create(**request))
            assert response.choices[0].message.content == 'hello'
        assert len(requests) == 8 and all(value == requests[0] for value in requests)
        assert waits == [10, 20, 40, 80, 80, 80]
        rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
        logical = [row for row in rows if row['event'] == 'response']
        assert [row['model_attempts'] for row in logical] == [7, 1]
        assert all(row['error'] is None and row['cost_usd'] is None for row in logical)
        assert all(all(value is None for value in row['usage'].values()) for row in logical)
        assert all(value is None for value in recorder.summary['known'].values())
        assert recorder.summary['missing_cost'] == 2
        assert client.max_retries == 2 and recorder.lifecycle.reason is None
        if target := os.environ.get('CEOBENCH_R227_SAMPLES'):
            path = Path(target)
            path.mkdir(parents=True, exist_ok=True)
            (path / ('retry-' + role + '.json')).write_text(json.dumps(dict(requests=requests, usage=rows), indent=2))
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        serving.join(3)
    assert not serving.is_alive()


@pytest.mark.parametrize('fault', ['choice_index', 'tool_index'])
@pytest.mark.parametrize('complete', [False, True])
def test_invalid_stream_indexes_correct_only_after_completion(tmp_path, monkeypatch, fault, complete):
    from saas_bench.agents.bash_agent.agent import BashAgent
    from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
    requests, waits = [], []

    def respond(request):
        requests.append(json.loads(request.content))
        assert len(requests) <= 2
        if len(requests) == 2:
            return completion(calls(('read_file', dict(path='valid.txt'))), stream=True)
        base = dict(id='bad-index', object='chat.completion.chunk', created=1, model='test-model')
        delta = dict(role='assistant', content='partial') if fault == 'choice_index' else dict(
            role='assistant', tool_calls=[dict(index=1, id='invalid', type='function',
                function=dict(name='read_file', arguments='{"path":"invalid.txt"}'))])
        frames = [dict(base, choices=[dict(index=1 if fault == 'choice_index' else 0,
                                         delta=delta, finish_reason='stop')]),
                  dict(base, choices=[], usage=reply('chat')['usage'])]
        body = ''.join('data: ' + json.dumps(frame) + '\n\n' for frame in frames)
        if complete:
            body += 'data: [DONE]\n\n'
        return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                              stream=httpx.ByteStream(body.encode()))

    client = OpenAI(api_key='offline', base_url='https://opencode.ai/zen/go/v1', max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    usage = ModelUsage(tmp_path / 'usage.jsonl', 'ceo')
    monkeypatch.setattr(usage.lifecycle, 'sleep', waits.append)
    agent = BashAgent(get_bash_agent_tool_descriptions(), client, model='test-model',
                      workspace_path=tmp_path, usage_recorder=usage)
    try:
        action = agent.act('Read the file', 0, False, {'day': 0})
        assert action.tool == 'read_file' and action.arguments == {'path': 'valid.txt'}
        assert len(requests) == 2 and waits == ([] if complete else [10])
        if complete:
            assert 'previous response was invalid' in requests[1]['messages'][-1]['content']
        else:
            assert requests[0] == requests[1]
        rows = [json.loads(line) for line in usage.path.read_text().splitlines()]
        logical = [row for row in rows if row['event'] == 'response']
        assert len(logical) == (2 if complete else 1)
        if complete:
            assert logical[0]['error'] == 'InvalidModelResponse'
            assert logical[0]['usage']['input_tokens'] == 10
    finally:
        client.close()


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('role', ROLES)
@pytest.mark.parametrize('failure', ['timeout', 'killed'])
def test_closed_registered_failure_returns_to_owner_and_preserves_partner(mode, role, failure):
    received = []

    def respond(team, sender, body):
        if sender == role:
            text = json.dumps(body['messages'])
            assert 'PARTIAL-' + role in text
            assert 'FOLLOWING-SCRIPT' in text
            assert 'timed out' in text if failure == 'timeout' else 'exit code: 137' in text
            received.append(text)
        return calls(('bash', dict(command=ADVANCE))) if sender == 'ceo' else final(sender + ' report')

    with make_team(mode, respond, 'closed-script-failure') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        code = ("import time, os, signal\nimport novamind_api as n\n"
                "with open('executions.txt', 'a') as out: out.write('once\\n')\n"
                "n.infrastructure.set_capacity_tier(2)\n" + f"print('PARTIAL-{role}', flush=True)\n")
        code += "time.sleep(60)\n" if failure == 'timeout' else "os.kill(os.getpid(), signal.SIGKILL)\n"
        register_cli(team, role, 'first.py', code)
        register_cli(team, role, 'second.py', "print('FOLLOWING-SCRIPT')\n")
        team.server.role_executors[role].bash_timeout = .5
        outcome = team.run(stop_after_day=7)
        save_evidence(team, 'closed-' + role + '-' + failure, outcome=vars(outcome))
        assert (outcome.day, outcome.reason) == (7, 'observation_end'), team.failure
        assert received and team.failure is None
        records = team.server.role_script_results[role]
        assert len(records) == 2 and records[0]['classification'] != 'succeeded'
        assert all(record['facts']['boundary_closed'] for record in records)
        assert (Path(records[0]['evidence']) / 'stdout.bin').read_bytes() == f'PARTIAL-{role}\n'.encode()
        assert (team.roles[role].workspace / 'executions.txt').read_text() == 'once\n'
        assert all(runtime.week.weekly_status == 'completed' for runtime in team.roles.values())
        assert not any(executor.preserved_process for executor in team.server.role_executors.values())


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('stdout', ['', 'REAL-PARTIAL'])
def test_public_python_timeout_delivers_actual_partial_output_and_next_call(mode, stdout, monkeypatch):
    seen = []

    def respond(team, role, body):
        if role == 'growth':
            if not seen:
                seen.append('started')
                return calls(('bash', dict(command='./novamind-operation python timeout.py')))
            result = body['messages'][-1]
            assert result['role'] == 'tool'
            rendered = result['content']
            assert 'unfinished' in rendered.lower() and 'timed out' in rendered.lower()
            assert rendered.startswith(stdout + '\n' if stdout else '[stderr]\nREAL-STDERR\n')
            seen.append('received')
        return calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final(role + ' report')

    with make_team(mode, respond, 'public-partial') as team:
        executor = team.roles['growth'].executor
        executor.execute('write_file', dict(path='timeout.py', content=(
            "import sys, time\n" + (f"print('{stdout}', flush=True)\n" if stdout else '') +
            "print('REAL-STDERR', file=sys.stderr, flush=True)\ntime.sleep(60)\n")))
        monkeypatch.setitem(executor.extra_env, 'CEOBENCH_PUBLIC_PYTHON_TIMEOUT_SECONDS', '.2')
        outcome = team.run(stop_after_day=7)
        save_evidence(team, 'public-partial-' + str(bool(stdout)), outcome=vars(outcome))
        assert outcome.reason == 'observation_end', team.failure
        assert seen == ['started', 'received']
        if mode == 'pf':
            store = team.roles['growth'].store
            with store.connect() as connection:
                ids = [row[0] for row in connection.execute(
                    'SELECT event_id FROM requests WHERE branch=?', (store.identity['branch_id'],))]
            event, = [store.read_event(event) for event in ids
                      if store.read_event(event)['request']['kind'] == 'cli_python']
            assert store.get_content(event['request']['event_id'] + ':stdout_bytes')[1] == (
                (stdout + '\n').encode() if stdout else b'')
            store.assert_healthy()


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('timeout', [False, True])
def test_registered_undecodable_output_reaches_owner_with_raw_bytes(mode, timeout):
    received = []

    def respond(team, role, body):
        if role == 'growth':
            content = body['messages'][-1]['content']
            assert r'\xff' in content and 'escaped raw bytes' in content
            assert 'FOLLOWING-SCRIPT' in content
            if timeout:
                assert 'timed out' in content and 'partial' in content
            received.append(content)
        return calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final(role + ' report')

    with make_team(mode, respond, 'registered-binary') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        register_cli(team, 'growth', 'binary.py', "import os, time\nos.write(1, b'\\xff')\n" +
                     ('time.sleep(60)\n' if timeout else ''))
        register_cli(team, 'growth', 'second.py', "print('FOLLOWING-SCRIPT')\n")
        team.server.role_executors['growth'].bash_timeout = .5
        outcome = team.run(stop_after_day=7)
        save_evidence(team, 'registered-binary-' + str(timeout), outcome=vars(outcome))
        assert (outcome.day, outcome.reason) == (7, 'observation_end'), team.failure
        assert len(received) == 1
        record = team.server.role_script_results['growth'][0]
        assert record['classification'] == 'script_error'
        assert (Path(record['evidence']) / 'stdout.bin').read_bytes() == b'\xff'
        assert (Path(record['evidence']) / 'stderr.bin').read_bytes() == b''
        if mode == 'pf':
            team.roles['growth'].store.assert_healthy()


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_rolled_back_prediction_failure_has_terminal_receipt_and_agent_retries(mode):
    from test_advance_terminal import operations
    turns = []

    def respond(team, role, body):
        if role == 'ceo':
            if turns:
                result = body['messages'][-1]
                assert result['role'] == 'tool' and 'prediction_save_failed' in result['content']
                advances = [(path, value) for path, value in operations(team).items() if value['kind'] == 'api:advance']
                assert len(advances) == 1
                path, receipt = advances[0]
                assert receipt['status'] == 'failed'
                assert receipt['advance']['progression'] == 'not_advanced'
                assert team.server.tools.current_day == 0
                assert not team.server.conn.in_transaction and not team.server._operation_failed
                assert team.server.conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0] == 0
                frozen = path.read_bytes()
                team._finish_operation(path, advance_outcome=receipt['advance'])
                assert path.read_bytes() == frozen
                with pytest.raises(RuntimeError, match='Conflicting operation finalization'):
                    team._finish_operation(path, advance_outcome=dict(receipt['advance'], status='completed'))
                team.server.conn.execute('DROP TRIGGER deny_prediction_effect')
            turns.append('advance')
            assert len(turns) <= 2
            return calls(('bash', dict(command=ADVANCE)))
        return final(role + ' report')

    with make_team(mode, respond, 'prediction-rejection') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        team.server.conn.execute("CREATE TEMP TRIGGER deny_prediction_effect BEFORE INSERT ON business_effects "
            "WHEN json_extract(NEW.target_json,'$.table')='predictions' "
            "BEGIN SELECT RAISE(ABORT,'prediction rejected'); END")
        before = team.checkpoint()
        outcome = team.run(stop_after_day=7)
        save_evidence(team, 'prediction-rejection', outcome=vars(outcome),
                      operations={str(path): value for path, value in operations(team).items()},
                      prior_checkpoint=str(before))
        assert (outcome.day, outcome.reason) == (7, 'observation_end'), team.failure
        assert len(turns) == 2 and team.failure is None
        assert all(value['status'] != 'running' for value in operations(team).values())


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('fault', ['after_commit', 'rollback', 'receipt'])
def test_prediction_uncertainty_or_required_receipt_failure_holds(mode, fault, monkeypatch):
    from saas_bench.business_ledger import BusinessOperation, CommitConnection
    from saas_bench.run_lifecycle import RunCancelled
    from test_advance_terminal import operations
    turns = []

    def respond(team, role, body):
        if role == 'ceo':
            turns.append(body)
            assert len(turns) == 1
            return calls(('bash', dict(command=ADVANCE)))
        return final(role + ' report')

    with make_team(mode, respond, 'unsafe-prediction-' + fault) as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        checkpoint = team.checkpoint()
        if fault == 'after_commit':
            original = CommitConnection._commit

            def fail_commit(adapter):
                original(adapter)
                if adapter.operation.tool == 'next_week':
                    raise OSError('commit completed but acknowledgement lost')

            monkeypatch.setattr(CommitConnection, '_commit', fail_commit)
        else:
            team.server.conn.execute("CREATE TEMP TRIGGER deny_prediction_effect BEFORE INSERT ON business_effects "
                "WHEN json_extract(NEW.target_json,'$.table')='predictions' "
                "BEGIN SELECT RAISE(ABORT,'prediction rejected'); END")
            if fault == 'rollback':
                def fail_rollback(adapter):
                    raise OSError('rollback did not complete')
                monkeypatch.setattr(CommitConnection, 'rollback', fail_rollback)
            else:
                original = BusinessOperation.finish

                def fail_receipt(operation, result=None, **kwargs):
                    if result and result.get('error') == 'prediction_save_failed':
                        raise OSError('required rejection receipt cannot persist')
                    return original(operation, result, **kwargs)

                monkeypatch.setattr(BusinessOperation, 'finish', fail_receipt)
        outcome = team.run(stop_after_day=7)
        save_evidence(team, 'unsafe-prediction-' + fault, outcome=vars(outcome),
            operations={str(path): value for path, value in operations(team).items()},
            prior_checkpoint=str(checkpoint))
        assert outcome.reason == 'paused' and outcome.day == 0
        assert len(turns) == 1 and team.server._operation_failed
        with pytest.raises(RunCancelled):
            team._check_model_send()
        with pytest.raises((RuntimeError, RunCancelled)):
            team.checkpoint()
        assert checkpoint.exists()
        assert all(not runtime.executor.preserved_process for runtime in team.roles.values())


@pytest.mark.parametrize('method', ['get_action', 'continue_with_tool_results', 'get_action_with_history'])
def test_legacy_agent_missing_usage_is_null_and_next_request_continues(method):
    from saas_bench.config import BenchmarkConfig
    from saas_bench.database import init_database
    from saas_bench.llm import AgentLLM, CostTracker
    requests = []

    def respond(request):
        requests.append(request)
        response = reply('responses')
        response.pop('usage')
        return httpx.Response(200, json=response)

    client = OpenAI(api_key='offline', max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    connection = init_database(':memory:')
    config = BenchmarkConfig(agent_llm_model='test-model')
    agent = AgentLLM(client, connection, config, CostTracker(connection, config))
    try:
        for _ in range(2):
            args = ([], [], [], 0) if method == 'continue_with_tool_results' else ({'day': 0, 'cash': 1000}, [], 0)
            response, cost, *_ = getattr(agent, method)(*args)
            assert response.output_text == 'hello' and cost is None
        assert len(requests) == 2
        rows = connection.execute('SELECT input_tokens, output_tokens, cost_usd FROM api_costs').fetchall()
        assert len(rows) == 2 and all(tuple(row) == (None, None, None) for row in rows)
    finally:
        client.close()
        connection.close()
