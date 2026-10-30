from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import signal
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
import uuid

import httpx
from openai import OpenAI
import pytest

from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor, get_bash_agent_tool_descriptions
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.run_lifecycle import RunCancelled, WorkerLifecycle
from saas_bench.systemd_guard import scope_command
from saas_bench import team_batch as batch
from saas_bench.run_state import write_json
from test_preflight_team_observer import observer


def controller_module():
    scripts = str(Path(__file__).resolve().parents[1] / 'scripts')
    spec = importlib.util.spec_from_file_location('team_controller_stop_test',
        Path(__file__).resolve().parents[1] / 'scripts/team_controller.py')
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, scripts)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(scripts)
    return module


def wait_until(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.01)
    assert predicate()


@pytest.mark.parametrize('failed_listener', [1, 3])
def test_listener_start_failure_closes_real_runtime_and_next_start(tmp_path, monkeypatch, failed_listener):
    from numpy.random import default_rng
    from saas_bench.api_server import NovaMindAPIServer
    from saas_bench.config import BenchmarkConfig
    from saas_bench.database import init_database
    from saas_bench.simulation import Simulator
    from saas_bench.tools import AgentTools
    from test_preflight_team_flow import final, make_team
    conn = init_database(':memory:')
    config = BenchmarkConfig(seed=123)
    simulator = Simulator(conn, config, default_rng(123))
    simulator.initialize()
    tools = AgentTools(conn, 0, tmp_path / 'host', config=config, rng=simulator.rng)
    starts, apis, clients, listeners = [], [], [], []
    thread_start, api_start = threading.Thread.start, NovaMindAPIServer.start

    def injected_start(thread):
        if getattr(thread._target, '__name__', None) == 'serve_forever':
            starts.append(thread)
            if len(starts) == failed_listener:
                raise RuntimeError('Injected listener startup failure')
            listeners.append(thread)
        return thread_start(thread)

    def remember(api):
        apis.append(api)
        return api_start(api)

    def client(role):
        result = OpenAI(api_key='OFFLINE', http_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))))
        clients.append(result)
        return result

    try:
        started = time.monotonic()
        with monkeypatch.context() as failure:
            failure.setattr(threading.Thread, 'start', injected_start)
            failure.setattr(NovaMindAPIServer, 'start', remember)
            with pytest.raises(RuntimeError, match='Injected listener startup failure'):
                MultiAgentRuntime(tmp_path / 'failed', tools, conn=conn, simulator=simulator,
                    mode='git', client_factory=client)
        assert time.monotonic() - started < 5
        assert len(apis) == 1
        assert apis[0]._httpd is None and apis[0]._role_httpds == []
        assert apis[0].public_request_state()['active'] == []
        assert all(not Path(path).exists() for path in apis[0].role_sockets.values())
        for thread in listeners:
            thread.join(1)
            assert not thread.is_alive()
    finally:
        for item in clients:
            item.close()
        conn.close()
    with make_team('git', lambda team, role, body: final('unused'), 'startup-retry') as team:
        output = team.roles['ceo'].executor.execute('bash',
            dict(command="python -c 'import novamind_api as nm; print(nm.vars.current_day)'"))
        assert output.strip() == '0'
        assert team.server.public_request_state()['mutation_epoch'] == 0


@pytest.mark.parametrize('bash_timeout', [480, 1200])
def test_analyst_failure_cancels_partner_bash_before_publication(tmp_path, monkeypatch, bash_timeout):
    marker = tmp_path / 'running'
    runtime = MultiAgentRuntime.__new__(MultiAgentRuntime)
    runtime.effective_end, runtime._pool, runtime._stage = 497, None, 'boundary'
    runtime._lifecycle = WorkerLifecycle()
    runtime.server = SimpleNamespace(tools=SimpleNamespace(current_day=0), public_dashboard=lambda: 'offline')
    runtime._set_phase = lambda phase: setattr(runtime.server, 'team_phase', phase)
    runtime._natural_end = lambda: False
    runtime._rotate_sessions = lambda: None
    runtime._script_output = lambda role: ''
    runtime._message = lambda role, text: SimpleNamespace(receiver=role)
    executor = BashAgentToolExecutor(workspace_path=tmp_path, bash_timeout=bash_timeout)
    executor.lifecycle = runtime._lifecycle
    monkeypatch.setattr(executor, '_bwrap', lambda: None)

    def analyst(role, requests, dashboard=None):
        if role == 'growth':
            wait_until(marker.exists)
            raise RuntimeError('offline analyst failure')
        executor.execute('bash', dict(command=f'touch {marker}; sleep 60'))

    runtime._run_analyst = analyst
    started = time.monotonic()
    outcome = runtime.run(stop_after_day=7)
    assert time.monotonic() - started < 5
    assert outcome.reason == 'failed'
    assert runtime.server.team_phase == 'paused'
    assert executor.last_status == 'cancelled'
    assert executor._bash_facts['boundary_closed']
    assert executor._bash_facts['process_boundary']['descendants_closed']
    assert executor._bash_facts['cancelled'] == 'analyst_failure'
    assert not executor._bash_facts['timed_out']


@pytest.mark.parametrize('phase', ['headers', 'stream', 'reuse_headers'])
@pytest.mark.parametrize('secure', [False, True])
def test_cancel_wakes_threaded_model_socket_and_preserves_unknown_usage(tmp_path, phase, secure):
    blocked, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            if self.path == '/prime':
                self.send_response(200)
                self.send_header('Content-Length', '2')
                self.end_headers()
                self.wfile.write(b'{}')
                self.wfile.flush()
                return
            if phase == 'stream':
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.flush()
            blocked.set()
            release.wait(5)
            self.close_connection = True

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    if secure:
        certificate, key = tmp_path / 'certificate.pem', tmp_path / 'key.pem'
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
            '-keyout', str(key), '-out', str(certificate), '-subj', '/CN=localhost', '-days', '1'],
            capture_output=True, check=True, timeout=5)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certificate, key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    base = f'{"https" if secure else "http"}://127.0.0.1:{server.server_port}'
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = OpenAI(api_key='offline', base_url=base, max_retries=0,
                    http_client=httpx.Client(timeout=600, verify=not secure))
    lifecycle = WorkerLifecycle()
    agent = BashAgent(get_bash_agent_tool_descriptions(), client, model='offline',
                      workspace_path=tmp_path, reasoning_effort='high')
    agent.lifecycle = lifecycle
    try:
        lifecycle.watch_client(client)
        if phase == 'reuse_headers':
            assert client._client.post(base + '/prime', content=b'{}').status_code == 200
        with ThreadPoolExecutor(max_workers=1) as pool:
            call = pool.submit(agent._request_model, 'chat', {}, lambda:
                client.chat.completions.create(model='offline', messages=[dict(role='user', content='offline')],
                                               stream=phase == 'stream').__enter__().response.read()
                if phase == 'stream' else
                client.chat.completions.create(model='offline', messages=[dict(role='user', content='offline')]))
            assert blocked.wait(3)
            lifecycle.cancel('analyst_failure')
            with pytest.raises(RunCancelled, match='analyst_failure'):
                call.result(timeout=2)
        assert agent.usage_recorder.summary['calls'] == 1
        assert agent.usage_recorder.summary['missing_cost'] == 1
        assert all(value is None for value in agent.usage_recorder.summary['known'].values())
    finally:
        release.set()
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_bound_scope_kills_detached_session_after_worker_exit(tmp_path):
    controller = controller_module()
    unit = 'ceobench-stop-test-' + uuid.uuid4().hex + '.scope'
    marker = tmp_path / 'detached.json'
    child_code = f'import json,os,pathlib,time; pathlib.Path({str(marker)!r}).write_text(json.dumps(dict(pid=os.getpid(),pgid=os.getpgrp()))); time.sleep(60)'
    worker_code = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",sys.argv[1]],start_new_session=True); time.sleep(60)'
    worker = subprocess.Popen(scope_command([sys.executable, '-c', worker_code, child_code], unit,
        memory_high=64*1024**2, memory_max=96*1024**2), start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        wait_until(marker.exists)
        child = json.loads(marker.read_text())
        bound = controller.scope_state(dict(unit=unit))
        assert bound['invocation_id'] and bound['control_group']
        assert not bound['scope_empty']
        os.killpg(worker.pid, signal.SIGKILL)
        worker.wait(timeout=3)
        os.kill(child['pid'], 0)
        controller.signal_scope(bound, signal.SIGKILL)
        worker.communicate(timeout=3)
        wait_until(lambda: controller.scope_state(bound, exit_code=worker.returncode)['scope_empty'])
        assert controller.scope_state(bound, exit_code=worker.returncode)['scope_empty']
    finally:
        subprocess.run(['systemctl', '--user', 'stop', unit], capture_output=True, timeout=5)
        if worker.poll() is None:
            worker.kill()
        worker.communicate(timeout=3)


@pytest.mark.parametrize('field', ['invocation_id', 'control_group'])
def test_scope_identity_mismatch_warns_and_keeps_stop_available(tmp_path, monkeypatch, caplog, field):
    controller = controller_module()
    cgroup = tmp_path / 'scope'
    cgroup.mkdir()
    (cgroup / 'cgroup.events').write_text('populated 1\n')
    record = dict(unit='test.scope', invocation_id='original', control_group='/original')
    inv, group = ('replacement', '/original') if field == 'invocation_id' else ('original', '/replacement')
    signals = []

    def command(argv, **kwargs):
        if argv[2] == 'kill':
            signals.append(argv)
        return SimpleNamespace(returncode=0, stdout=f'LoadState=loaded\nActiveState=active\nInvocationID={inv}\nControlGroup={group}\n', stderr='')

    monkeypatch.setattr(controller.subprocess, 'run', command)
    monkeypatch.setattr(controller, 'unit_status', lambda unit, **kwargs:
        dict(unit=unit, invocation_id=inv, result='running', errors=[]))
    controller.signal_scope(record, signal.SIGKILL)
    assert 'Worker scope identity changed' in caplog.text
    assert len(signals) == 1 and '--signal=SIGKILL' in signals[0]


def test_failed_worker_scope_is_reaped_once_and_peer_finishes(observer, tmp_path, monkeypatch):
    controller = controller_module()
    source = str(Path(__file__).resolve().parents[1])
    observer.expected['source_root'] = observer.manifest['source_root'] = source
    for entry in observer.manifest['runs']:
        entry.update(pair_id='s42-r1', group=entry['run_id'])
    policy = observer.output.parent / 'policy.json'
    write_json(policy, dict(identity=observer.expected))
    monkeypatch.setattr(controller, 'identity', observer.module.identity)
    monkeypatch.setattr(controller, 'poll', lambda *args:
        observer.module.poll(*args, quota_fn=observer.quota))
    popen = subprocess.Popen
    children, units = [], []
    marker = tmp_path / 'detached.pid'

    def launch(command, **kwargs):
        if '--worker' not in command:
            return popen(command, **kwargs)
        role = command[command.index('--worker') + 1]
        entry = batch.selected(observer.manifest, role)
        record = Path(entry['output_dir']) / 'attempt-001/attempt.json'
        unit = next(arg.split('=', 1)[1] for arg in command if arg.startswith('--unit='))
        units.append(unit)
        child_code = f'import os,pathlib,time; pathlib.Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)'
        code = '''import json, pathlib, subprocess, sys, time
role, record, child_code = sys.argv[1:]
if role == 'git':
    subprocess.Popen([sys.executable, '-c', child_code], start_new_session=True)
    time.sleep(.1)
else:
    time.sleep(.4)
p = pathlib.Path(record); p.parent.mkdir(parents=True)
p.write_text(json.dumps(dict(status='failed' if role == 'git' else 'finished', day=28)))
sys.exit(2 if role == 'git' else 0)
'''
        child = popen([*command[:command.index(sys.executable)], sys.executable, '-c', code,
                       role, str(record), child_code], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(controller.subprocess, 'Popen', launch)
    try:
        results = controller.supervise(policy)
        assert [record['status'] for record in results] == ['paused', 'finished']
        assert results[0]['hold'] is True
        assert [child.returncode for child in children] == [2, 0]
        exit_record = batch.read(policy.parent / 'controller-exit.json')
        assert exit_record['hold'] and not exit_record['failed'] and exit_record['children_reaped']
        assert controller.exit_code(exit_record) == 3
        assert all(record['scope_empty'] for record in exit_record['units'].values())
        assert not observer.hold.exists()
        if marker.exists():
            path = Path('/proc') / marker.read_text() / 'stat'
            assert not path.exists() or path.read_text().split(') ')[1][0] == 'Z'
    finally:
        for unit in units:
            subprocess.run(['systemctl', '--user', 'stop', unit], capture_output=True, timeout=5)
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=3)
