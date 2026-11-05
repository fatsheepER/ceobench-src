from collections import Counter
from contextlib import closing
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import threading

import pytest

from saas_bench import team_batch as batch
from saas_bench.db_protection import load_session_db
from saas_bench.role_policy import ROLES
from saas_bench.run_state import file_hash, tree_hash, write_json
from test_preflight_team_batch import changed_manifest, frozen
from test_preflight_team_checkpoint import fast_simulator
from test_preflight_team_flow import ADVANCE, calls, completion, final
from test_team_script_state import register_cli
from test_team_stop_residuals import controller_module


def _rows(conn):
    return {name: sorted((tuple(row) for row in conn.execute('SELECT * FROM "' + name + '"')), key=repr)
            for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_stat%'")}


def _jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _worker(spec_path, argv):
    from saas_bench.agents.bash_agent import run_test

    spec = batch.read(spec_path)
    proof = dict(wire=[], checkpoints={}, errors=[], closed=None)
    clients, holder = [], {}
    lock = threading.Lock()
    connect = socket.socket.connect

    def local_only(channel, address):
        if isinstance(address, tuple) and address[0] not in ('127.0.0.1', '::1', 'localhost'):
            raise AssertionError('Checkpoint driver regression prohibits external connections')
        return connect(channel, address)

    def no_credentials(*args, **kwargs):
        raise AssertionError('Checkpoint driver regression cannot load credentials')

    socket.socket.connect = local_only
    run_test.load_env_file = no_credentials
    os.environ['OPENCODE_API_KEY'] = 'OFFLINE-ONLY'
    os.environ['OPENAI_API_KEY'] = 'OFFLINE-ONLY'
    checked_client, prepare = batch.checked_client, batch.prepare
    save, close = batch.save_checkpoint, batch.close_team

    def client(*args, **kwargs):
        if kwargs.get('model_config'):
            kwargs['model_config'] = dict(kwargs['model_config'], base_url=batch.ENDPOINT)
        result = checked_client(*args, **kwargs)
        clients.append(result)
        return result

    def local_world(conn, config, rng):
        simulator = fast_simulator(conn, config, rng)
        simulator.shock_manager = batch.simulator(conn, config, rng).shock_manager
        if spec.get('bankrupt_day') is not None:
            step = simulator.step_week
            def settle():
                result = step()
                if result.day == spec['bankrupt_day']:
                    conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES(?,?,-2000000,?)",
                                 (result.day, 'operations', 'offline natural termination'))
                    conn.commit()
                return result
            simulator.step_week = settle
        return simulator

    def prepare_with_local_world(*args, **kwargs):
        return prepare(*args, **kwargs, simulator_factory=local_world)

    def checkpoint(team, attempt, record, boundary):
        holder['team'] = team
        if boundary == 'ready':
            source = record.get('source_checkpoint')
            ready = dict(source=source, day=team.server.tools.current_day, stage=team._stage,
                configuration=team.configuration, generation_policy={role: dict(
                    memory_max_chars=runtime.agent.memory_max_chars,
                    max_output_tokens=runtime.agent.max_output_tokens,
                    invalid_generation_attempts=runtime.agent.invalid_generation_attempts)
                    for role, runtime in team.roles.items()})
            if source:
                snapshot = Path(source)
                saved = batch.read(snapshot / 'team.json')
                with closing(load_session_db(snapshot / 'world.nmdb')) as world:
                    ready['world_equal'] = _rows(team.server.conn) == _rows(world)
                ready['roles_equal'] = tree_hash(team.root / 'roles', exact=True) == tree_hash(snapshot / 'roles', exact=True)
                ready['weeks_equal'] = {
                    role: runtime.week.snapshot() if runtime.week else None
                    for role, runtime in team.roles.items()} == saved['role_weeks']
                ready['messages_equal'] = [asdict(message) for message in team.messages] == saved['messages']
                ready['rng_equal'] = team.server.tools.rng.bit_generator.state == saved['tools_rng']
                ready['log_prefixes_equal'] = all(
                    (team.root / name).read_bytes()[:cutoff] == (snapshot / name).read_bytes()[:cutoff]
                    for name, cutoff in batch.read(snapshot / 'checkpoint.json')['log_cutoffs'].items())
            else:
                for role in ROLES:
                    register_cli(team, role, 'once.py',
                        "import novamind_api as n\n"
                        "with open('script-count.txt', 'a') as out: out.write('once\\n')\n"
                        "print(n.infrastructure.get_cost_info())\n"
                        + f"print('SAVED-SCRIPT-{role}')\n")
            proof['ready'] = ready
        snapshot = save(team, attempt, record, boundary)
        if snapshot:
            proof['checkpoints'].setdefault(boundary, str(snapshot))
        return snapshot

    def close_team(team, record=None):
        sockets = [*team.server.role_sockets.values(), *team.server.script_role_sockets.values()]
        try:
            proof['finished'] = dict(day=team.server.tools.current_day,
                messages=[asdict(message) for message in team.messages],
                weeks={role: runtime.week.snapshot() if runtime.week else None
                       for role, runtime in team.roles.items()},
                script_counts={role: (runtime.workspace / 'script-count.txt').read_text()
                               if (runtime.workspace / 'script-count.txt').exists() else ''
                               for role, runtime in team.roles.items()},
                script_receipts={role: [dict(name=row['name'], classification=row['classification'],
                    boundary_closed=row['facts']['boundary_closed']) for row in team.server.role_script_results[role]]
                    for role in ROLES},
                operations=[dict(row) for row in team.server.conn.execute(
                    'SELECT actor,initiator,tool,outcome,operation_ref FROM business_operations ORDER BY operation_seq')],
                usage_matches_wire={role: [row['body'] for row in _jsonl(runtime.usage.path)
                    if row['event'] == 'http_request'][-len(sent):] == sent if sent else True
                    for role, runtime in team.roles.items()
                    for sent in [[row['body'] for row in proof['wire'] if row['role'] == role]]},
                logical_calls={role: [row['call_id'] for row in _jsonl(runtime.usage.path)
                    if row['event'] == 'response'] for role, runtime in team.roles.items()})
        finally:
            close(team, record)
            proof['closed'] = dict(
                live_role_threads=[thread.name for thread in threading.enumerate()
                                   if thread.name.startswith('team-role')],
                sockets_remaining=[str(path) for path in sockets if Path(path).exists()],
                public_requests=team.server.public_request_state())

    def response(team, role, body):
        if role == 'ceo':
            if team.phase == 'ceo_opening':
                if not any(message.get('name') == 'ask_analyst' for message in body['messages']):
                    return calls(('ask_analyst', dict(role='growth', message='OPENING-CHECK')))
                return final('SAVED-OPENING-DIRECTIONS')
            return calls(('bash', dict(command=ADVANCE)))
        if team._workers[role].question:
            if not any('MODEL-WRITE-DONE' in str(message.get('content', ''))
                       for message in body['messages'] if message['role'] == 'tool'):
                code = "import novamind_api as n; n.marketing.set_daily_spend(operations=123); n.infrastructure.set_capacity_tier(2); print('MODEL-WRITE-DONE')"
                return calls(('bash', dict(command='./novamind-operation python-c ' + shlex.quote(code))))
            return final('SAVED-OPENING-ANSWER')
        return final('SAVED-WEEKLY-REPORT-' + role)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                role = (self.headers.get('x-ceobench-session') or self.headers['x-opencode-session']).rsplit(':', 1)[-1]
                team = holder['team']
                with lock:
                    proof['wire'].append(dict(role=role, day=team.server.tools.current_day,
                                              phase=team.phase, body=body, endpoint=self.path))
                    if len(proof['wire']) > spec.get('max_requests', 16):
                        raise AssertionError('Unexpected repeated model turns in one-week fixture')
                if spec.get('stop_on_day') == team.server.tools.current_day:
                    (Path(argv[argv.index('--policy') + 1]).parent / 'STOP').touch()
                if spec.get('reject_run') == argv[argv.index('--worker') + 1]:
                    payload = json.dumps({'error': {'message': 'offline permanent rejection', 'code': spec.get('reject_code')}}).encode()
                    content_type, status = 'application/json', spec.get('reject_status', 400)
                else:
                    if 'input' in body:
                        messages = []
                        for item in body['input']:
                            if item.get('type') == 'function_call':
                                messages.append(dict(role='assistant', name=item['name'], content=item['arguments']))
                            elif item.get('type') == 'function_call_output':
                                messages.append(dict(role='tool', content=item['output']))
                            elif 'role' in item:
                                messages.append(item)
                        chat = response(team, role, {'messages': messages})
                        assistant = chat
                        output = [dict(id='rs1', type='reasoning', summary=[], encrypted_content='offline-reasoning')]
                        for call in assistant.get('tool_calls') or []:
                            output.append(dict(id='fc-' + call['id'], type='function_call', call_id=call['id'],
                                name=call['function']['name'], arguments=call['function']['arguments'], status='completed'))
                        if assistant.get('content'):
                            output.append(dict(id='msg1', type='message', role='assistant', status='completed',
                                content=[dict(type='output_text', text=assistant['content'], annotations=[])]))
                        result = dict(id='resp1', object='response', created_at=1, status='completed', model=body['model'],
                            output=output, usage=dict(input_tokens=10, output_tokens=2, total_tokens=12,
                                input_tokens_details={'cached_tokens': 3}, output_tokens_details={'reasoning_tokens': 1}))
                        payload, content_type = json.dumps(result).encode(), 'application/json'
                    else:
                        reply = completion(response(team, role, body), body.get('stream', False))
                        payload, content_type = reply.content, reply.headers['content-type']
                    status = 200
            except BaseException as exc:
                proof['errors'].append(type(exc).__name__ + ': ' + str(exc))
                payload, content_type, status = b'{"error":{"message":"offline fixture rejected request"}}', 'application/json', 400
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    serving = threading.Thread(target=lambda: server.serve_forever(poll_interval=.05), name='checkpoint-local-model')
    batch.ENDPOINT = f'http://127.0.0.1:{server.server_port}/opencode.ai/v1'
    batch.checked_client, batch.prepare = client, prepare_with_local_world
    batch.save_checkpoint, batch.close_team = checkpoint, close_team
    serving.start()
    try:
        return controller_module().main(argv)
    finally:
        server.shutdown()
        server.server_close()
        serving.join(3)
        proof['clients_closed'] = [client._client.is_closed for client in clients]
        proof['model_server_closed'] = not serving.is_alive() and server.socket.fileno() == -1
        write_json(spec['proof'], proof)


class Driver:
    def __init__(self, frozen, root, monkeypatch, *, worker_options=None, budget_seconds=240):
        self.frozen, self.root, self.monkeypatch = frozen, root, monkeypatch
        self.worker_options = worker_options or {}
        self.budget_seconds = budget_seconds
        self.children, self.units, self.samples = [], [], []
        self.finished, self.expired = threading.Event(), threading.Event()
        self.policies = []
        scripts = batch.ROOT / 'scripts'
        monkeypatch.syspath_prepend(str(scripts))
        module_spec = importlib.util.spec_from_file_location('checkpoint_driver_entry', scripts / 'team_driver.py')
        self.driver = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(self.driver)
        controller = self.driver.controller
        popen, poll = subprocess.Popen, controller.poll
        monkeypatch.setenv('OPENCODE_API_KEY', 'OFFLINE-ONLY')
        monkeypatch.setenv('PYTHON_DOTENV_DISABLED', '1')
        monkeypatch.delenv('CEOBENCH_RECOVERY_CHECKPOINTS', raising=False)
        monkeypatch.delenv('BOSSBENCH_LLM_REPLAY_DB', raising=False)
        monkeypatch.delenv('ORACLE_MODE', raising=False)

        def launch(command, **kwargs):
            if '--worker' not in command:
                return popen(command, **kwargs)
            self.units.append(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--unit=')))
            index = command.index(sys.executable)
            spec = root / f'worker-{len(self.children) + 1}.json'
            write_json(spec, dict(self.worker_options, proof=str(spec.with_suffix('.proof.json'))))
            child = popen([*command[:index], sys.executable, str(Path(__file__).resolve()),
                           str(spec), *command[index + 2:]], **kwargs)
            self.children.append((child, spec))
            return child

        monkeypatch.setattr(controller.subprocess, 'Popen', launch)
        monkeypatch.setattr(controller, 'poll', lambda *args: poll(*args, quota_fn=lambda: {
            window: dict(status='ok', percent=1) for window in ('rolling', 'weekly', 'monthly')}))
        self.watchdog = threading.Thread(target=self._timeout, name='checkpoint-driver-budget')
        self.watchdog.start()

    def _timeout(self):
        if not self.finished.wait(self.budget_seconds):
            self.expired.set()
            for policy in self.policies:
                (policy.parent / 'STOP').touch()
            if not self.finished.wait(8):
                self._kill()

    def _kill(self):
        for unit in self.units:
            subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=all', '--signal=SIGKILL', unit],
                           capture_output=True, timeout=5)
        for child, _ in self.children:
            if child.poll() is None:
                child.kill()

    def policy(self, name, group):
        def select(manifest):
            entry = next(row for row in manifest['runs'] if row['group'] == group)
            entry['stop_after_day'] = 7
            manifest['runs'] = [entry]

        manifest_path = changed_manifest(self.frozen, self.root / name, select)
        manifest = batch.read(manifest_path)
        entry, = manifest['runs']
        directory = self.root / name / 'controller'
        policy = directory / 'policy.json'
        identity = {key: manifest[key] for key in ('batch_id', 'source_root', 'public_dir', 'source_commit')}
        identity.update(manifest=str(manifest_path), manifest_sha256=file_hash(manifest_path),
                        host_logs=str(directory / 'host-logs'),
                        runs=[{key: entry[key] for key in ('run_id', 'output_dir')}])
        write_json(policy, dict(identity=identity))
        self.policies.append(policy)
        return policy, entry

    def run(self, policy, entry, *, checkpoint=None, week=None, resume=False, expected=0):
        before = {path: tree_hash(path, exact=True) for path in batch.attempts(entry)}
        start = len(self.children)
        self.monkeypatch.setenv('CEOBENCH_RECOVERY_CHECKPOINTS', json.dumps(
            {entry['run_id']: str(checkpoint)} if checkpoint is not None else {}))
        args = ['--policy', str(policy), '--live', '--run-id', entry['run_id']]
        if resume:
            args.append('--resume')
        if week is not None:
            args.extend(['--week', str(week)])
        code = self.driver.main(args)
        controller = batch.read(policy.parent / 'controller-exit.json')
        proofs = [batch.read(spec.with_suffix('.proof.json')) for _, spec in self.children[start:]
                  if spec.with_suffix('.proof.json').exists()]
        sample = dict(selection=str(checkpoint) if checkpoint is not None else None, week=week,
                      resume=resume, code=code, controller=controller, workers=proofs)
        self.samples.append(sample)
        assert not self.expired.is_set(), f'Real driver regression exceeded its {self.budget_seconds}-second test budget'
        assert controller['children_reaped'] and controller['memory_sampler']['closed']
        assert all(unit['scope_empty'] for unit in controller['units'].values())
        assert all(child.poll() is not None for child, _ in self.children[start:])
        assert {path: tree_hash(path, exact=True) for path in before} == before, 'Recovery changed an old attempt tree'
        assert len(self.children) == start + 1, 'An explicit driver invocation must launch exactly one worker'
        assert len(batch.attempts(entry)) == len(before) + 1, 'Each selection must allocate exactly one new attempt'
        assert len(proofs) == 1, 'Worker must publish resource-closure evidence'
        proof, = proofs
        assert proof['errors'] == []
        assert proof['model_server_closed'] and all(proof['clients_closed'])
        if proof['closed'] is not None:
            assert proof['closed']['live_role_threads'] == proof['closed']['sockets_remaining'] == []
            assert proof['closed']['public_requests']['active'] == []
            assert proof['closed']['public_requests']['queued'] is False
        assert code == expected, sample
        assert batch.read(policy.parent / 'driver-exit.json')['returncode'] == expected
        attempt = batch.attempts(entry)[-1]
        return attempt, batch.read(attempt / 'attempt.json'), proof

    def close(self):
        self.finished.set()
        self.watchdog.join(3)
        for unit in self.units:
            subprocess.run(['systemctl', '--user', 'stop', unit], capture_output=True, timeout=5)
        for child, _ in self.children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=3)
        if destination := os.environ.get('CEOBENCH_R228_SAMPLES'):
            samples = []
            for sample in self.samples:
                workers = []
                for proof in sample['workers']:
                    workers.append({key: value for key, value in proof.items() if key != 'wire'} | dict(
                        requests=[dict(role=row['role'], day=row['day'], phase=row['phase'],
                            message_roles=[message['role'] for message in row['body']['messages']],
                            script_marker_count=json.dumps(row['body']['messages']).count('SAVED-SCRIPT-' + row['role']),
                            opening_answer_count=json.dumps(row['body']['messages']).count('SAVED-OPENING-ANSWER'))
                            for row in proof['wire']]))
                samples.append(dict(sample, workers=workers))
            write_json(Path(destination) / (self.root.name + '.json'), dict(invocations=samples,
                       children_reaped=all(child.returncode is not None for child, _ in self.children),
                       watchdog_closed=not self.watchdog.is_alive(), budget_expired=self.expired.is_set()))
        assert not self.watchdog.is_alive()


@pytest.fixture
def driver(frozen, tmp_path, monkeypatch):
    harness = Driver(frozen, tmp_path, monkeypatch)
    try:
        yield harness
    finally:
        harness.close()


def _finished(record, proof, *, source=None):
    assert record['status'] == 'finished' and record['day'] == 7
    assert record.get('source_checkpoint') == (str(source) if source is not None else None)
    assert proof['ready']['source'] == record.get('source_checkpoint')
    if source is not None:
        assert all(proof['ready'][key] for key in (
            'world_equal', 'roles_equal', 'weeks_equal', 'messages_equal', 'rng_equal', 'log_prefixes_equal'))
    finished = proof['finished']
    assert finished['day'] == 7
    assert finished['script_counts'] == dict.fromkeys(ROLES, 'once\n')
    assert finished['script_receipts'] == {role: [dict(name='once.py', classification='succeeded',
                                                     boundary_closed=True)] for role in ROLES}
    assert all(finished['usage_matches_wire'].values())
    assert all(len(ids) == len(set(ids)) for ids in finished['logical_calls'].values())
    assert all(state['weekly_status'] == 'completed' for state in finished['weeks'].values())
    assert Counter(message['kind'] for message in finished['messages']) == dict(question=1, weekly_task=2)
    assert all(message['status'] == 'delivered' for message in finished['messages'])
    operations = [row for row in finished['operations'] if row['tool'] in ('set_daily_spend', 'set_capacity_tier')]
    assert Counter((row['actor'], row['initiator'], row['tool'], row['outcome']) for row in operations) == Counter(
        [('growth', 'model', 'set_daily_spend', 'succeeded'),
         ('growth', 'model', 'set_capacity_tier', 'succeeded')])
    for role in ROLES:
        sent = [row['body']['messages'] for row in proof['wire'] if row['role'] == role]
        if sent:
            assert json.dumps(sent[-1]).count('SAVED-SCRIPT-' + role) == 1
    ceo = [row['body']['messages'] for row in proof['wire'] if row['role'] == 'ceo'][-1]
    assert json.dumps(ceo).count('SAVED-OPENING-ANSWER') == 1
    for role in ROLES[1:]:
        assert json.dumps(ceo).count('SAVED-WEEKLY-REPORT-' + role) == 1


@pytest.mark.parametrize('group', ['git', 'pf'])
@pytest.mark.parametrize('boundary', ['ask_completed', 'answers_ready'])
def test_finished_driver_replays_exact_opening_checkpoint_repeatedly_and_week_one(driver, group, boundary):
    policy, entry = driver.policy('history', group)
    _, original, first = driver.run(policy, entry)
    _finished(original, first)
    selected = Path(first['checkpoints'][boundary])
    saved = batch.read(selected / 'team.json')
    assert saved['day'] == 0 and saved['stage'] == ('ceo_opening' if boundary == 'ask_completed' else 'ceo')
    initial = Path(first['checkpoints']['ready'])
    assert batch.read(initial / 'team.json')['stage'] == 'boundary'
    for _ in range(2):
        _, record, proof = driver.run(policy, entry, checkpoint=selected, resume=True)
        _finished(record, proof, source=selected)
        assert {message['request_id'] for message in saved['messages']} <= {
            message['request_id'] for message in proof['finished']['messages']}
        with closing(load_session_db(selected / 'world.nmdb')) as world:
            references = {row[0] for row in world.execute('SELECT operation_ref FROM business_operations')}
        assert references <= {row['operation_ref'] for row in proof['finished']['operations']}
        assert not any(row['phase'] == 'ceo_opening' and row['role'] == 'growth' for row in proof['wire'])
        if boundary == 'answers_ready':
            assert [row['role'] for row in proof['wire']] == ['ceo']
        else:
            assert Counter(row['role'] for row in proof['wire']) == dict(ceo=2, growth=1, ops_finance=1)
    _, record, proof = driver.run(policy, entry, week=1, checkpoint=selected, resume=True)
    _finished(record, proof, source=initial)
    assert proof['ready']['day'] == 0 and proof['ready']['stage'] == 'boundary'
    missing = driver.root / 'unavailable-exact-archive'
    attempt, record, proof = driver.run(policy, entry, checkpoint=missing, resume=True, expected=3)
    assert record['status'] == 'paused' and record['hold']
    assert record['reason'] == 'missing_checkpoint' and record['execution_started'] is False
    assert str(missing) in json.dumps(record)
    assert proof['wire'] == [] and not proof['clients_closed']
    assert 'ready' not in proof and not (attempt / 'runtime').exists()
    assert (attempt / 'HOLD').exists()


@pytest.mark.parametrize('group', ['git', 'pf'])
def test_driver_empty_history_honors_external_checkpoint_and_explicit_sealed_d0(driver, group):
    policy, entry = driver.policy('source', group)
    _, record, original = driver.run(policy, entry)
    _finished(record, original)
    selected = Path(original['checkpoints']['answers_ready'])
    original_trees = {path: tree_hash(path, exact=True) for path in batch.attempts(entry)}
    external_policy, external_entry = driver.policy('external', group)
    assert not batch.attempts(external_entry)
    _, record, proof = driver.run(external_policy, external_entry, checkpoint=selected, resume=True)
    _finished(record, proof, source=selected)
    assert record['attempt'] == 1 and [row['role'] for row in proof['wire']] == ['ceo']
    assert {path: tree_hash(path, exact=True) for path in original_trees} == original_trees
    d0_policy, d0_entry = driver.policy('sealed-d0', group)
    _, record, proof = driver.run(d0_policy, d0_entry, checkpoint=selected, week=1, resume=True)
    _finished(record, proof)
    assert record['replay_week'] == 1 and record['attempt'] == 1
    assert proof['ready']['day'] == 0 and proof['ready']['stage'] == 'boundary'
    assert proof['wire'][0]['phase'] == 'ceo_opening'
    assert {path: tree_hash(path, exact=True) for path in original_trees} == original_trees


if __name__ == '__main__':
    raise SystemExit(_worker(sys.argv[1], sys.argv[2:]))
