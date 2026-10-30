from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import signal
import socket
import ssl
import subprocess
import threading

import httpx
from openai import APIConnectionError, OpenAI
import pytest

from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from saas_bench.run_lifecycle import RunCancelled, WorkerLifecycle


@pytest.fixture
def duplicated_sockets(monkeypatch):
    original = socket.fromfd
    duplicates = []

    def duplicate(*args, **kwargs):
        result = original(*args, **kwargs)
        duplicates.append(result)
        return result

    monkeypatch.setattr(socket, 'fromfd', duplicate)
    return duplicates


@contextmanager
def blackhole_tls(roles, *, proxy=False, proxy_context=None):
    server = socket.socket()
    server.bind(('127.0.0.1', 0))
    server.listen()
    server.settimeout(3)
    hello, release = threading.Event(), threading.Event()
    connections = []

    def serve():
        try:
            for _ in range(roles):
                connection, _ = server.accept()
                if proxy_context:
                    connection = proxy_context.wrap_socket(connection, server_side=True)
                connections.append(connection)
                connection.settimeout(3)
                if proxy:
                    header = b''
                    while not header.endswith(b'\r\n\r\n'):
                        header += connection.recv(4096)
                    assert header.startswith(b'CONNECT model.invalid:443 ')
                    connection.sendall(b'HTTP/1.1 200 Connection established\r\n\r\n')
                if not connection.recv(4096):
                    return
            hello.set()
            release.wait(5)
        finally:
            for connection in connections:
                connection.close()

    serving = threading.Thread(target=serve)
    serving.start()
    try:
        scheme = 'http' if proxy and proxy_context is None else 'https'
        yield f'{scheme}://127.0.0.1:{server.getsockname()[1]}', hello, release
    finally:
        release.set()
        serving.join(4)
        server.close()
        assert not serving.is_alive()


@pytest.mark.parametrize('roles', [1, 2])
@pytest.mark.parametrize('signal_cancel', [False, True])
def test_cancel_wakes_model_tls_handshake_and_keeps_unknown_usage(tmp_path, duplicated_sockets, roles, signal_cancel):
    lifecycle = WorkerLifecycle()
    requests, outcomes, agents, threads = [], [], [], []
    with blackhole_tls(roles) as (base, hello, release), lifecycle:
        for role in range(roles):
            client = OpenAI(api_key='offline', base_url=base, max_retries=2,
                            http_client=httpx.Client(timeout=60, verify=False))
            client._client.event_hooks['request'].append(lambda request: requests.append(request))
            agent = BashAgent(get_bash_agent_tool_descriptions(), client, model='offline',
                              workspace_path=tmp_path / str(role), reasoning_effort='high')
            agent.lifecycle = lifecycle
            agents.append(agent)

            def invoke(agent=agent):
                try:
                    agent._request_model('chat', {}, lambda: agent.client.chat.completions.create(
                        model='offline', messages=[dict(role='user', content='offline')]))
                except BaseException as error:
                    outcomes.append(error)

            threads.append(threading.Thread(target=invoke))
        try:
            for thread in threads:
                thread.start()
            assert hello.wait(3), 'TLS ClientHello was not received'
            assert len(duplicated_sockets) == roles
            assert all(channel.fileno() >= 0 for channel in duplicated_sockets)
            if signal_cancel:
                with lifecycle._socket_lock:
                    os.kill(os.getpid(), signal.SIGUSR1)
            else:
                lifecycle.cancel('analyst_failure')
            for thread in threads:
                thread.join(.7)
            assert not any(thread.is_alive() for thread in threads), 'TLS handshake stayed active after cancellation'
            reason = 'supervisor_cancelled' if signal_cancel else 'analyst_failure'
            assert len(outcomes) == roles
            assert all(isinstance(error, RunCancelled) and str(error) == reason for error in outcomes)
            assert len(requests) == roles
            assert all(channel.fileno() == -1 for channel in duplicated_sockets)
            for agent in agents:
                assert agent.usage_recorder.summary['calls'] == 1
                assert agent.usage_recorder.summary['missing_cost'] == 1
                assert all(value is None for value in agent.usage_recorder.summary['known'].values())
        finally:
            release.set()
            for thread in threads:
                thread.join(4)
            for agent in agents:
                agent.client.close()


@pytest.mark.parametrize('secure_proxy', [False, True])
def test_model_tls_through_connect_proxy_cancels(tmp_path, duplicated_sockets, secure_proxy):
    context = None
    if secure_proxy:
        certificate, key = tmp_path / 'certificate.pem', tmp_path / 'key.pem'
        subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
            '-keyout', str(key), '-out', str(certificate), '-subj', '/CN=localhost', '-days', '1'],
            capture_output=True, check=True, timeout=5)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certificate, key)
    lifecycle, outcomes, requests = WorkerLifecycle(), [], []
    with blackhole_tls(1, proxy=True, proxy_context=context) as (proxy, hello, release):
        client = OpenAI(api_key='offline', base_url='https://model.invalid', max_retries=2,
            http_client=httpx.Client(proxy=httpx.Proxy(proxy, ssl_context=ssl._create_unverified_context() if secure_proxy else None),
                                     timeout=60, verify=False, trust_env=False))
        client._client.event_hooks['request'].append(lambda request: requests.append(request))
        agent = BashAgent(get_bash_agent_tool_descriptions(), client, model='offline',
                          workspace_path=tmp_path, reasoning_effort='high')
        agent.lifecycle = lifecycle

        def invoke():
            try:
                agent._request_model('chat', {}, lambda: client.chat.completions.create(
                    model='offline', messages=[dict(role='user', content='offline')]))
            except BaseException as error:
                outcomes.append(error)

        running = threading.Thread(target=invoke)
        running.start()
        try:
            assert hello.wait(3), 'Proxy did not receive the target TLS ClientHello'
            lifecycle.cancel('analyst_failure')
            running.join(.7)
            assert not running.is_alive(), 'Proxy TLS handshake stayed active after cancellation'
            assert len(outcomes) == 1 and isinstance(outcomes[0], RunCancelled)
            assert len(requests) == 1
            assert agent.usage_recorder.summary['missing_cost'] == 1
            assert all(channel.fileno() == -1 for channel in duplicated_sockets)
        finally:
            release.set()
            running.join(4)
            client.close()


@pytest.mark.parametrize('phase', ['failed', 'cancel_before_wrap'])
def test_tls_failure_and_model_exit_close_handshake_duplicate(duplicated_sockets, phase):
    with blackhole_tls(1) as (base, hello, release):
        client = OpenAI(api_key='offline', base_url=base, max_retries=0,
                        http_client=httpx.Client(timeout=.1, verify=False))
        lifecycle = WorkerLifecycle()
        events = []

        def request_hook(request):
            def trace(event, info):
                events.append(event)
                if phase == 'cancel_before_wrap' and event == 'connection.start_tls.started':
                    lifecycle.reason = 'before_wrap'
            request.extensions['trace'] = trace

        client._client.event_hooks['request'].append(request_hook)
        lifecycle.watch_client(client)
        try:
            expected = RunCancelled if phase == 'cancel_before_wrap' else APIConnectionError
            with pytest.raises(expected):
                with lifecycle.model_call():
                    client.chat.completions.create(model='offline', messages=[dict(role='user', content='offline')])
            assert len(duplicated_sockets) == 1
            assert duplicated_sockets[0].fileno() == -1
            assert ('connection.start_tls.failed' in events) == (phase == 'failed')
        finally:
            release.set()
            client.close()


def test_successful_tls_handshake_closes_duplicate_and_reuses_connection(tmp_path, duplicated_sockets):
    certificate, key = tmp_path / 'certificate.pem', tmp_path / 'key.pem'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes',
                    '-keyout', str(key), '-out', str(certificate), '-subj', '/CN=localhost', '-days', '1'],
                   capture_output=True, check=True, timeout=5)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            body = json.dumps(dict(id='offline', model='offline', object='chat.completion', created=0,
                                   choices=[dict(index=0, finish_reason='stop',
                                                 message=dict(role='assistant', content='ok'))])).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    server.socket = context.wrap_socket(server.socket, server_side=True)
    serving = threading.Thread(target=server.serve_forever)
    serving.start()
    client = OpenAI(api_key='offline', base_url=f'https://127.0.0.1:{server.server_port}', max_retries=0,
                    http_client=httpx.Client(timeout=3, verify=False))
    lifecycle = WorkerLifecycle()
    lifecycle.watch_client(client)
    try:
        for _ in range(2):
            with lifecycle.model_call():
                response = client.chat.completions.create(model='offline', messages=[dict(role='user', content='offline')])
                assert response.choices[0].message.content == 'ok'
                assert len(duplicated_sockets) == 1
                assert duplicated_sockets[0].fileno() == -1
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        serving.join(3)
