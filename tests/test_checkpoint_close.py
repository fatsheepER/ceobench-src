from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import socket
import subprocess
import sys
import threading
import time

import httpx
import pytest

from saas_bench import team_batch as batch
from saas_bench.db_protection import load_session_db
from saas_bench.run_lifecycle import BusinessOutcomeUnknown, RunCancelled
from saas_bench.run_state import file_hash, tree_hash, write_json
from saas_bench.team_checkpoint import validate_snapshot
from test_preflight_team_batch import changed_manifest, frozen
from test_preflight_team_checkpoint import fast_simulator
from test_preflight_team_flow import ADVANCE, calls, completion, final, opening_response
from test_team_stop_residuals import controller_module


def rollback_usage(attempt, snapshot, restored):
    def calls(root):
        return {row['call_id'] for path in (root / 'private').glob('*/model-usage.jsonl')
            for line in path.read_text().splitlines() for row in [json.loads(line)] if row['event'] == 'request'}
    retained_prefix = calls(snapshot)
    discarded = calls(attempt / 'runtime') - retained_prefix
    result = restored['result']
    retained = {row['call_id'] for row in result['usage']['records']}
    waste = {row['call_id'] for row in result['waste']['records']}
    assert retained_prefix <= retained
    assert discarded and waste == discarded and not retained & discarded
    assert not result['pending']['records']
    assert result['waste']['roles']['ceo']['billed_units'] == 1
    return dict(retained_calls=sorted(retained), discarded_calls=sorted(discarded),
        retained_tokens=result['usage']['team']['complete']['total_tokens'],
        waste_tokens=result['waste']['team']['complete']['total_tokens'])


@pytest.mark.parametrize('host_timeout', [False, True], ids=['client_disconnect', 'host_timeout'])
@pytest.mark.parametrize('late_failure', [False, True], ids=['late_success', 'late_failure'])
def test_batch_close_keeps_late_advance_resources_until_terminal(
        frozen, tmp_path, monkeypatch, host_timeout, late_failure):
    def one_week(manifest):
        for entry in manifest['runs']:
            entry['stop_after_day'] = 7

    path = changed_manifest(frozen, tmp_path, one_week)
    entry = next(row for row in batch.read(path)['runs'] if row['group'] == 'git')
    entered, release, listeners_stopped, resources_closing, wrapper_pending = (threading.Event() for _ in range(5))
    holder, sent, clients, close_observations, step_errors, callback_errors, handler_threads = {}, [], [], [], [], [], []
    make_team = batch.make_team

    def client(role):
        def respond(request):
            sent.append(role)
            body = json.loads(request.content)
            if response := opening_response(role, body):
                return response
            message = calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final('Offline analyst answer.')
            return completion(message, body.get('stream', False))

        created = batch.checked_client(role, 'checkpoint-close', transport=httpx.MockTransport(respond))
        close = created.close
        observe = len(clients) < 3

        def observed_close():
            if observe and 'team' in holder:
                close_observations.append(dict(role=role,
                    worker_alive=holder.get('worker') is not None and holder['worker'].is_alive(),
                    advance_statuses=[row['status'] for path in
                        (holder['team'].root / 'private/operations').glob('*.json')
                        for row in [batch.read(path)] if row['kind'] == 'api:advance'],
                    marker_count=holder['team'].server.conn.execute(
                        "SELECT COUNT(*) FROM ledger WHERE note='late-close-marker'").fetchone()[0],
                    requests=holder['team'].server.public_request_state()))
                resources_closing.set()
            close()

        monkeypatch.setattr(created, 'close', observed_close)
        clients.append(created)
        return created

    def delayed_simulator(conn, config, rng):
        sim = fast_simulator(conn, config, rng)
        sim.shock_manager = batch.simulator(conn, config, rng).shock_manager
        step = sim.step_week

        def delayed_step():
            holder['worker'] = threading.current_thread()
            entered.set()
            try:
                assert release.wait(60), 'Test did not release the admitted host advancement'
                result = step()
                sim.conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES(7,'operations',-1,'late-close-marker')")
                sim.conn.commit()
                sim.save_rng_states()
                if late_failure:
                    raise RuntimeError('injected late host failure')
                return result
            except BaseException as exc:
                step_errors.append(type(exc).__name__ + ': ' + str(exc))
                raise

        sim.step_week = delayed_step
        return sim

    def observed_team(*args, **kwargs):
        team = make_team(*args, **kwargs)
        holder['team'] = team
        team.server.STEP_WEEK_TIMEOUT = .1 if host_timeout else None
        team.roles['ceo'].executor.bash_timeout = .2
        team.roles['ceo'].executor.advance_timeout = .2
        stop = team.server.stop
        finished = team.server.request_finished_callback
        finish_operation = team._finish_operation

        def observed_operation(*args, **kwargs):
            try:
                return finish_operation(*args, **kwargs)
            except RunCancelled as exc:
                if exc.reason == 'business_operation_pending':
                    wrapper_pending.set()
                raise

        def observed_stop():
            stop()
            listeners_stopped.set()

        def observed_finished():
            handler_threads.append(threading.current_thread())
            try:
                finished()
            except BaseException as exc:
                callback_errors.append(type(exc).__name__ + ': ' + str(exc))
                raise

        monkeypatch.setattr(team.server, 'stop', observed_stop)
        monkeypatch.setattr(team, '_finish_operation', observed_operation)
        team.server.request_finished_callback = observed_finished
        return team

    monkeypatch.setattr(batch, 'make_team', observed_team)
    with ThreadPoolExecutor(max_workers=1) as pool:
        preparing = pool.submit(batch.prepare, path, entry['run_id'], live=True,
            client_factory=client, simulator_factory=delayed_simulator)
        try:
            assert entered.wait(30)
            team = holder['team']
            assert wrapper_pending.wait(30), 'The role wrapper did not record the still-running public advance'
            with pytest.raises((BusinessOutcomeUnknown, RuntimeError)):
                team.checkpoint()
            if host_timeout:
                assert listeners_stopped.wait(30), 'The batch did not reach its resource cleanup path'
                resources_closing.wait(2)
            else:
                assert not listeners_stopped.wait(.2), 'The runtime passed the world owner before settlement'
            before_release = dict(requests=team.server.public_request_state(),
                prepare_returned=preparing.done(), client_closed=any(created.is_closed() for created in clients))
        finally:
            release.set()
        result = preparing.result(timeout=15)

    holder['worker'].join(5)
    assert not holder['worker'].is_alive(), 'The host advancement worker outlived cleanup'
    for handler in handler_threads:
        handler.join(5)
        assert not handler.is_alive(), 'The public request handler outlived cleanup'
    directory = batch.attempts(entry)[0]
    receipts = [batch.read(row) for row in (directory / 'runtime/private/operations').glob('*.json')]
    observation = dict(before_release=before_release,
        close_observations=close_observations, step_errors=step_errors, callback_errors=callback_errors,
        receipts=receipts, attempt=result)
    batch.write_json(tmp_path / 'close-observation.json', observation)
    if destination := os.environ.get('CEOBENCH_R228_SAMPLES'):
        batch.write_json(Path(destination) / f'close-hosttimeout-{host_timeout}-failure-{late_failure}.json', observation)
    assert close_observations and all(row['advance_statuses'] == ['failed' if late_failure else 'completed']
        and not row['requests']['active'] for row in close_observations), (
            'Batch cleanup closed shared resources before admitted host work ended')
    assert not before_release['prepare_returned'] and not before_release['client_closed']
    assert callback_errors == []
    assert step_errors == (['RuntimeError: injected late host failure'] if late_failure else [])
    assert all(row['marker_count'] == 1 for row in close_observations)
    assert team.server.public_request_state()['active'] == []
    assert team.server.public_request_state()['queued'] is False
    assert not team.server._advance_lock.locked()
    assert not team.server._role_httpds and team.server._httpd is None
    assert all(created.is_closed() for created in clients)
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        team.server.conn.execute('SELECT 1')
    assert result['status'] == 'paused' and result['hold']
    assert sent.count('ceo') == 2
    usage = [json.loads(line) for line in (team.root / 'private/ceo/model-usage.jsonl').read_text().splitlines()]
    assert sum(row['event'] == 'request' for row in usage) == 2
    assert sum(row['event'] == 'response' for row in usage) == 2
    advance = next(row for row in receipts if row['kind'] == 'api:advance')
    wrapper = next(row for row in receipts if row['kind'] == 'ceo_tool:bash')
    assert advance['status'] == ('failed' if late_failure else 'completed')
    assert advance['advance']['reached_day'] == advance['advance']['last_completed_day'] == 7
    assert team.server._advance_outcome == advance['advance']
    assert wrapper['status'] == 'failed'
    assert wrapper['finalization']['process_boundary']['descendants_closed'] is True
    assert team.roles['ceo'].executor.preserved_process is None
    with pytest.raises(RunCancelled):
        team._check_model_send()
    selected, state = validate_snapshot(result['checkpoint'])
    assert state['day'] == 0 and state['stage'] == 'ceo'
    assert result['recovery_validation']['eligible'] is True
    assert result['recovery_validation']['discard_post_checkpoint_operations'] is True
    original = {row: row.read_bytes() for row in directory.rglob('*') if row.is_file()}
    monkeypatch.setenv('CEOBENCH_RECOVERY_CHECKPOINTS', json.dumps({entry['run_id']: str(selected)}))
    restored = batch.prepare(path, entry['run_id'], resume=True, live=True, force_resume=True,
        client_factory=client, simulator_factory=fast_simulator)
    assert (restored['attempt'], restored['status'], restored['day']) == (2, 'finished', 7)
    assert restored['source_checkpoint'] == str(selected)
    assert restored['checkpoint_rollback']['day'] == 0
    assert {row: row.read_bytes() for row in original} == original
    second = batch.attempts(entry)[1]
    with closing(load_session_db(second / 'runtime/world.nmdb')) as connection:
        assert connection.execute("SELECT COUNT(*) FROM ledger WHERE note='late-close-marker'").fetchone()[0] == 0
    observation['manual_resume'] = rollback_usage(directory, selected, restored)
    if destination := os.environ.get('CEOBENCH_R228_SAMPLES'):
        batch.write_json(Path(destination) / f'close-hosttimeout-{host_timeout}-failure-{late_failure}.json', observation)


def _cleanup_worker(spec_path):
    spec = batch.read(spec_path)
    connect = socket.socket.connect
    prepare, make_team, close = batch.prepare, batch.make_team, batch.close_team
    holder = {}
    batch.ENDPOINT = spec['endpoint']
    batch.TEAM_CLOSE_SECONDS = .2
    os.environ['OPENCODE_API_KEY'] = 'OFFLINE-ONLY'

    def local_only(channel, address):
        if isinstance(address, tuple) and address[0] not in ('127.0.0.1', '::1', 'localhost'):
            raise AssertionError('Cleanup controller regression prohibits external connections')
        return connect(channel, address)

    def world(conn, config, rng):
        sim = fast_simulator(conn, config, rng)
        sim.shock_manager = batch.simulator(conn, config, rng).shock_manager

        def blocked_step():
            holder['worker'] = threading.current_thread()
            write_json(spec['entered'], dict(pid=os.getpid(), thread=threading.get_ident()))
            if not threading.Event().wait(120):
                raise RuntimeError('Controller did not terminate the blocked host worker')

        sim.step_week = blocked_step
        return sim

    def prepared(*args, **kwargs):
        return prepare(*args, **kwargs, simulator_factory=world)

    def created(*args, **kwargs):
        team = make_team(*args, **kwargs)
        team.server.STEP_WEEK_TIMEOUT = .1
        team.roles['ceo'].executor.bash_timeout = .2
        team.roles['ceo'].executor.advance_timeout = .2
        return team

    def closed(team, record=None):
        error = None
        try:
            return close(team, record)
        except BaseException as exc:
            error = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            try:
                team.server.conn.execute('SELECT 1').fetchone()
                database_open = True
            except sqlite3.ProgrammingError:
                database_open = False
            write_json(spec['closure'], dict(error=error, database_open=database_open,
                clients_closed=[role.agent.client.is_closed() for role in team.roles.values()],
                host_worker_alive=holder.get('worker') is not None and holder['worker'].is_alive(),
                requests=team.server.public_request_state(),
                step_timed_out=team.server._step_day_timed_out,
                attempt=batch.read(team.root.parent / 'attempt.json')))

    socket.socket.connect = local_only
    batch.prepare, batch.make_team, batch.close_team = prepared, created, closed
    return controller_module().worker(Path(spec['policy']), spec['run_id'], resume=False,
        parent_pid=spec['parent_pid'])


def test_controller_holds_and_reaps_host_after_resource_cleanup_expires(frozen, tmp_path, monkeypatch):
    def one_run(manifest):
        entry = next(row for row in manifest['runs'] if row['group'] == 'git')
        entry['stop_after_day'] = 7
        manifest['runs'] = [entry]

    manifest_path = changed_manifest(frozen, tmp_path, one_run)
    manifest = batch.read(manifest_path)
    entry, = manifest['runs']
    directory = tmp_path / 'controller'
    policy = directory / 'policy.json'
    identity = {key: manifest[key] for key in ('batch_id', 'source_root', 'public_dir', 'source_commit')}
    identity.update(manifest=str(manifest_path), manifest_sha256=file_hash(manifest_path),
        host_logs=str(directory / 'host-logs'), runs=[{key: entry[key] for key in ('run_id', 'output_dir')}])
    write_json(policy, dict(identity=identity))
    entered, closure = directory / 'entered.json', directory / 'closure.json'
    requests, children, units, watchdog_errors = [], [], [], []
    finished, expired = threading.Event(), threading.Event()
    controller = controller_module()
    monkeypatch.setattr(controller, 'COOPERATIVE_STOP_SECONDS', .3)
    monkeypatch.setattr(controller, 'FORCE_STOP_SECONDS', .6)
    monkeypatch.setenv('PYTHON_DOTENV_DISABLED', '1')
    monkeypatch.delenv('CEOBENCH_RECOVERY_CHECKPOINTS', raising=False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            role = self.headers['x-opencode-session'].rsplit(':', 1)[-1]
            requests.append(dict(role=role, body=body))
            response = opening_response(role, body)
            if response is None:
                message = calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final('Offline analyst answer.')
                response = completion(message, body.get('stream', False))
            self.send_response(200)
            self.send_header('Content-Type', response.headers['content-type'])
            self.send_header('Content-Length', str(len(response.content)))
            self.end_headers()
            self.wfile.write(response.content)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    serving = threading.Thread(target=server.serve_forever, name='cleanup-local-model')
    spec = directory / 'worker.json'
    write_json(spec, dict(endpoint=f'http://127.0.0.1:{server.server_port}/opencode.ai/v1',
        entered=str(entered), closure=str(closure), policy=str(policy), run_id=entry['run_id'],
        parent_pid=os.getpid()))
    popen, poll = subprocess.Popen, controller.poll

    def launch(command, **kwargs):
        if '--worker' not in command:
            return popen(command, **kwargs)
        units.append(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--unit=')))
        prefix = command[:command.index(sys.executable)]
        child = popen([*prefix, sys.executable, str(Path(__file__).resolve()), str(spec)], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(controller.subprocess, 'Popen', launch)
    monkeypatch.setattr(controller, 'poll', lambda *args: poll(*args, quota_fn=lambda: {
        window: dict(status='ok', percent=1) for window in ('rolling', 'weekly', 'monthly')}))

    def watchdog():
        deadline = time.monotonic() + 90
        while not closure.exists() and time.monotonic() < deadline:
            if finished.wait(.02):
                return
        if closure.exists() and finished.wait(10):
            return
        expired.set()
        watchdog_errors.append('Controller did not close the blocked host after cleanup finished or expired')
        for unit in list(units):
            subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=all', '--signal=SIGKILL', unit],
                capture_output=True, timeout=5)
        for child in list(children):
            if child.poll() is None:
                child.kill()

    watching = threading.Thread(target=watchdog, name='cleanup-controller-budget')
    serving.start()
    watching.start()
    try:
        result, = controller.supervise(policy)
        finished.set()
        watching.join(3)
        attempt, = batch.attempts(entry)
        record = batch.read(attempt / 'attempt.json')
        receipt = batch.read(directory / 'controller-exit.json')
        events = [json.loads(line) for line in (directory / 'controller-events.jsonl').read_text().splitlines()]
        proof = batch.read(closure) if closure.exists() else None
        sample = dict(proof=proof, attempt=record, controller=receipt, events=events,
            model_roles=[row['role'] for row in requests], watchdog_errors=watchdog_errors,
            child_exit_codes=[child.returncode for child in children])
        if destination := os.environ.get('CEOBENCH_R228_SAMPLES'):
            write_json(Path(destination) / 'cleanup-controller-expiry.json', sample)
        assert not expired.is_set(), sample
        assert len(children) == 1 and children[0].poll() is not None
        assert proof and proof['error']['type'] == 'BusinessOutcomeUnknown'
        assert proof['database_open'] and proof['clients_closed'] == [False, False, False]
        assert proof['host_worker_alive'] and proof['step_timed_out']
        assert proof['requests']['active'] == []
        assert record['status'] == result['status'] == 'paused' and record['hold']
        assert record['resource_cleanup']['status'] == 'pending'
        assert (attempt / 'HOLD').exists() and not (attempt / 'STOP').exists()
        assert not (directory / 'STOP').exists()
        assert receipt['hold'] and not receipt['operator_stopped'] and not receipt['auto_resume']
        assert receipt['children_reaped'] and receipt['memory_sampler']['closed']
        assert all(unit['scope_empty'] for unit in receipt['units'].values())
        assert controller.exit_code(receipt) == 3
        assert sum(row['event'] == 'worker_launched' for row in events) == 1
        terminal = batch.read(directory / 'controller-terminal.json')[entry['run_id']]
        assert terminal['disposition'] == 'hold' and not terminal.get('operator_stop')
        assert sum(row['role'] == 'ceo' for row in requests) == 2
        usage = [json.loads(line) for line in (attempt / 'runtime/private/ceo/model-usage.jsonl').read_text().splitlines()]
        assert sum(row['event'] == 'request' for row in usage) == sum(row['event'] == 'response' for row in usage) == 2
        snapshot, state = validate_snapshot(record['checkpoint'])
        assert state['day'] == 0 and state['stage'] == 'ceo'
        assert record['recovery_validation']['eligible']
        original = tree_hash(attempt, exact=True)
        monkeypatch.setenv('CEOBENCH_RECOVERY_CHECKPOINTS', json.dumps({entry['run_id']: str(snapshot)}))

        def resume_client(role):
            def respond(request):
                body = json.loads(request.content)
                if response := opening_response(role, body):
                    return response
                message = calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final('Offline analyst answer.')
                return completion(message, body.get('stream', False))
            return batch.checked_client(role, 'cleanup-manual-resume', transport=httpx.MockTransport(respond))

        restored = batch.prepare(manifest_path, entry['run_id'], resume=True, live=True, force_resume=True,
            client_factory=resume_client, simulator_factory=fast_simulator)
        assert (restored['attempt'], restored['status'], restored['day']) == (2, 'finished', 7)
        assert restored['source_checkpoint'] == str(snapshot)
        assert tree_hash(attempt, exact=True) == original
        sample['manual_resume'] = rollback_usage(attempt, snapshot, restored)
        if destination := os.environ.get('CEOBENCH_R228_SAMPLES'):
            write_json(Path(destination) / 'cleanup-controller-expiry.json', sample)
    finally:
        finished.set()
        watching.join(3)
        for unit in units:
            subprocess.run(['systemctl', '--user', 'stop', unit], capture_output=True, timeout=5)
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=3)
        server.shutdown()
        server.server_close()
        serving.join(3)
    assert not watching.is_alive() and not serving.is_alive()


if __name__ == '__main__':
    raise SystemExit(_cleanup_worker(sys.argv[1]))
