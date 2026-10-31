from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import pytest

from saas_bench import team_batch as batch
from saas_bench.run_lifecycle import WorkerLifecycle
from saas_bench.run_state import file_hash, tree_hash, write_json
from saas_bench.team_checkpoint import validate_snapshot
from saas_bench.team_recovery import latest_checkpoint
from test_preflight_team_batch import changed_manifest, frozen
from test_team_stop_residuals import controller_module


def _network_worker(spec_path):
    spec = batch.read(spec_path)
    controller = controller_module()
    batch.ENDPOINT = spec['endpoint']
    connect = socket.socket.connect

    def local_only(channel, address):
        if isinstance(address, tuple) and address[0] not in ('127.0.0.1', '::1', 'localhost'):
            raise AssertionError('Controller STOP test prohibits external connections')
        return connect(channel, address)

    socket.socket.connect = local_only
    sleep = WorkerLifecycle.sleep
    waits = []

    def retry_wait(lifecycle, seconds):
        waits.append(seconds)
        if spec['phase'] == 'backoff' and len(waits) == 6:
            wait = lifecycle._wake.wait

            def blocked_wait(timeout):
                write_json(spec['blocked'], dict(waits=waits))
                return wait(timeout)

            lifecycle._wake.wait = blocked_wait
            try:
                return sleep(lifecycle, seconds)
            finally:
                lifecycle._wake.wait = wait
        return sleep(lifecycle, .001)

    WorkerLifecycle.sleep = retry_wait
    close = batch.close_team

    def close_team(team, record=None):
        sockets = [*team.server.role_sockets.values(), *team.server.script_role_sockets.values()]
        clients = [role.agent.client for role in team.roles.values()]
        customer = team.server.simulator.customer_simulator
        if customer and customer.client:
            clients.append(customer.client)
        close(team, record)
        write_json(spec['closed'], dict(
            lifecycle_reason=team._lifecycle.reason,
            live_role_threads=[thread.name for thread in threading.enumerate()
                               if thread.name.startswith('team-role')],
            sockets_remaining=[str(path) for path in sockets if Path(path).exists()],
            clients_closed=[client._client.is_closed for client in clients],
            public_requests=team.server.public_request_state(),
            waits=waits))

    batch.close_team = close_team
    return controller.worker(Path(spec['policy']), spec['run_id'], resume=False,
                             parent_pid=spec['parent_pid'])


@pytest.mark.parametrize('phase', ['headers', 'backoff'])
def test_controller_stop_after_six_network_failures_preserves_attempt_and_checkpoint(
        frozen, tmp_path, monkeypatch, phase):
    monkeypatch.delenv('CEOBENCH_MODEL_REQUEST_ATTEMPTS', raising=False)
    monkeypatch.delenv('CEOBENCH_MODEL_RETRY_DELAYS', raising=False)
    controller = controller_module()
    manifest_path = changed_manifest(frozen, tmp_path, lambda manifest: None)
    manifest = batch.read(manifest_path)
    entry = next(row for row in manifest['runs'] if row['group'] == 'git')
    directory = tmp_path / 'controller'
    policy = directory / 'policy.json'
    identity = {key: manifest[key] for key in ('batch_id', 'source_root', 'public_dir', 'source_commit')}
    identity.update(manifest=str(manifest_path), manifest_sha256=file_hash(manifest_path),
                    host_logs=str(directory / 'host-logs'),
                    runs=[{key: entry[key] for key in ('run_id', 'output_dir')}])
    write_json(policy, dict(identity=identity))
    blocked, closed = directory / 'blocked.json', directory / 'closed.json'
    requests, channels, children, units, stop_errors = [], [], [], [], []
    peer_closed, finished = threading.Event(), threading.Event()
    before = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            if len(requests) > 6:
                channels.append(self.connection)
                self.connection.settimeout(15)
                write_json(blocked, dict(requests=len(requests)))
                try:
                    if self.connection.recv(1) == b'':
                        peer_closed.set()
                except ConnectionResetError:
                    peer_closed.set()
                except TimeoutError:
                    pass
                return
            payload = b'{"error":{"message":"temporary outage"}}'
            self.send_response(503)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    serving = threading.Thread(target=server.serve_forever)
    spec = directory / 'worker.json'
    write_json(spec, dict(endpoint=f'http://127.0.0.1:{server.server_port}/v1', phase=phase,
        blocked=str(blocked), closed=str(closed), policy=str(policy), run_id=entry['run_id'],
        parent_pid=os.getpid()))
    popen, check_output, poll = subprocess.Popen, subprocess.check_output, controller.poll

    def launch(command, **kwargs):
        if '--worker' not in command:
            return popen(command, **kwargs)
        units.append(next(arg.split('=', 1)[1] for arg in command if arg.startswith('--unit=')))
        prefix = command[:command.index(sys.executable)]
        child = popen([*prefix, sys.executable, str(Path(__file__).resolve()), str(spec)], **kwargs)
        children.append(child)
        return child

    def source_identity(command, **kwargs):
        if command == ['git', 'rev-parse', 'HEAD']:
            return manifest['source_commit'] + '\n'
        return check_output(command, **kwargs)

    monkeypatch.setattr(controller.subprocess, 'Popen', launch)
    monkeypatch.setattr(controller.subprocess, 'check_output', source_identity)
    monkeypatch.setattr(controller, 'poll', lambda *args: poll(*args, quota_fn=lambda: {
        window: dict(status='ok', percent=1) for window in ('rolling', 'weekly', 'monthly')}))

    def stop_when_blocked():
        try:
            deadline = time.monotonic() + 15
            while not blocked.exists():
                if finished.wait(.01) or time.monotonic() >= deadline:
                    raise AssertionError(f'Worker stopped or timed out after {len(requests)} HTTP attempts')
            attempt, = batch.attempts(entry)
            record = batch.read(attempt / 'attempt.json')
            assert record['status'] == 'running'
            checkpoint = Path(record['checkpoint'])
            before.update(attempt=attempt, checkpoint=checkpoint, checkpoint_hash=tree_hash(checkpoint, exact=True),
                          stopped_at=time.monotonic())
        except BaseException as exc:
            stop_errors.append(str(exc))
        finally:
            (directory / 'STOP').touch()
        if not finished.wait(8):
            stop_errors.append('Controller did not close after STOP within 8 seconds')
            for unit in list(units):
                subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=all', '--signal=SIGKILL', unit],
                               capture_output=True, timeout=3)
            for child in list(children):
                if child.poll() is None:
                    child.kill()

    stopping = threading.Thread(target=stop_when_blocked)
    serving.start()
    stopping.start()
    try:
        result, = controller.supervise(policy)
        finished.set()
        stopping.join(3)
        assert not stop_errors, stop_errors
        assert time.monotonic() - before['stopped_at'] < 8
        assert len(children) == 1 and children[0].returncode == 2
        assert len(requests) == (7 if phase == 'headers' else 6)
        assert all(request == requests[0] for request in requests)
        assert batch.attempts(entry) == [before['attempt']]
        record = batch.read(before['attempt'] / 'attempt.json')
        assert record['status'] == result['status'] == 'paused'
        assert record['runtime_failure'] == 'supervisor_cancelled'
        assert not record.get('hold') and not (before['attempt'] / 'HOLD').exists()
        assert record['checkpoint'] == str(before['checkpoint'])
        assert tree_hash(before['checkpoint'], exact=True) == before['checkpoint_hash']
        snapshot, state = validate_snapshot(record['checkpoint'])
        assert state['day'] == 0 and state['stage'] == 'boundary'
        assert record['recovery_validation']['eligible']
        assert latest_checkpoint(entry, week=1) == snapshot
        proof = batch.read(closed)
        assert proof['lifecycle_reason'] == 'supervisor_cancelled'
        assert proof['live_role_threads'] == proof['sockets_remaining'] == []
        assert proof['clients_closed'] and all(proof['clients_closed'])
        assert proof['public_requests']['active'] == []
        assert proof['public_requests']['queued'] is False
        assert proof['waits'] == [10, 20, 40, 80, 80, 80]
        if phase == 'headers':
            assert peer_closed.wait(1), 'STOP must close the blocked model connection'
        usage = before['attempt'] / 'runtime/private/ceo/model-usage.jsonl'
        rows = [json.loads(line) for line in usage.read_text().splitlines()]
        logical, = [row for row in rows if row['event'] == 'response']
        assert logical['model_attempts'] == len(requests)
        assert logical['cost_usd'] is None and all(value is None for value in logical['usage'].values())
        assert len({row['call_id'] for row in rows if row['event'] == 'http_request'}) == 1
        receipt = batch.read(directory / 'controller-exit.json')
        assert receipt['operator_stopped'] and receipt['children_reaped']
        assert not receipt['hold'] and not receipt['failed'] and not receipt['auto_resume']
        assert controller.exit_code(receipt) == 0
        assert receipt['memory_sampler']['closed']
        assert all(unit['scope_empty'] for unit in receipt['units'].values())
        terminal = batch.read(directory / 'controller-terminal.json')[entry['run_id']]
        assert terminal['disposition'] == 'paused' and terminal['operator_stop']
        events = [json.loads(line) for line in (directory / 'controller-events.jsonl').read_text().splitlines()]
        assert len([row for row in events if row['event'] == 'worker_launched']) == 1
        assert not any(row['event'] in ('group_hold', 'group_alarm') for row in events)
    finally:
        (directory / 'STOP').touch()
        finished.set()
        stopping.join(3)
        for unit in units:
            subprocess.run(['systemctl', '--user', 'stop', unit], capture_output=True, timeout=5)
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=3)
        for channel in channels:
            try:
                channel.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        server.shutdown()
        server.server_close()
        serving.join(3)
    assert not stopping.is_alive() and not serving.is_alive()


if __name__ == '__main__':
    raise SystemExit(_network_worker(sys.argv[1]))
