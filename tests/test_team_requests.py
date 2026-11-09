from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import threading

import pytest

from saas_bench.role_policy import ROLES
from saas_bench.model_usage import plain
from test_preflight_team_observer import observer
from test_preflight_team_flow import ADVANCE, calls, final, make_team, raw, register


def save_evidence(team, label):
    target = os.environ.get('CEOBENCH_REQUEST_EVIDENCE')
    if target:
        destination = Path(target)
        destination.mkdir(parents=True, exist_ok=True)
        usage = {role: [json.loads(line) for line in runtime.usage.path.read_text().splitlines()]
                 for role, runtime in team.roles.items()}
        (destination / (team.configuration['mode'] + '-' + label + '.json')).write_text(json.dumps(dict(
            requests=team.test_wire, timeline=team.test_timeline, usage=usage,
            messages=[asdict(message) for message in team.messages], failure=team.failure,
            failure_detail=getattr(team, 'failure_detail', None)), indent=2))


def assert_paired(messages):
    pending = []
    for message in messages:
        if message['role'] == 'assistant' and message.get('tool_calls'):
            assert not pending
            pending = [call['id'] for call in message['tool_calls']]
        elif message['role'] == 'tool':
            assert pending and message['tool_call_id'] == pending.pop(0)
        elif message['role'] == 'user':
            assert not pending, 'New input split an unfinished tool batch'
    assert not pending


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_all_roles_discover_requests_and_ceo_alias(mode):
    with make_team(mode, lambda *args: final('report'), 'request-tools') as team:
        for role, runtime in team.roles.items():
            names = {tool['name'] for tool in runtime.agent.tool_descriptions}
            assert 'ask_agent' in names
            assert ('ask_analyst' in names) == (role == 'ceo')


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_experts_request_idle_ceo_fifo_then_wake_handed_off_peer(mode, monkeypatch):
    counts = Counter()
    first_enqueued = threading.Event()
    both_enqueued = threading.Event()
    answers = []
    sessions = {}

    def respond(team, role, body):
        counts[role] += 1
        messages = body['messages']
        assert_paired(messages)
        session = team.roles[role].identity.session_id
        assert sessions.setdefault(role, session) == session
        rendered = json.dumps(messages)
        latest = messages[-1]['content']
        if role != 'ceo':
            if counts[role] == 1:
                if role == 'ops_finance':
                    assert first_enqueued.wait(10)
                return calls(('ask_agent', dict(role='ceo', message='QUESTION-' + role)),
                    ('write_file', dict(path='after-request.txt', content='continued')))
            if 'WAKE-' in str(latest):
                assert 'Team request' in latest
                return final('REPLY-' + latest.split('WAKE-')[1].split()[0])
            assert 'CEO-ANSWER-' + role in rendered
            assert (team.roles[role].workspace / 'after-request.txt').read_text() == 'continued'
            return final('WEEKLY-' + role)
        if 'QUESTION-growth' in str(latest):
            first_enqueued.set()
            assert both_enqueued.wait(10)
            assert all(not (team.roles[r].workspace / 'after-request.txt').exists() for r in ROLES[1:])
            answers.append('growth')
            status, denied = raw(team, 'ceo', '/next-week', {})
            assert status == 403, denied
            status, changed = raw(team, 'ceo', '/call', dict(tool='set_prices', args=dict(A=37)))
            assert status == 200 and changed['success'], changed
            return final('CEO-ANSWER-growth')
        if 'QUESTION-ops_finance' in str(latest):
            answers.append('ops_finance')
            return final('CEO-ANSWER-ops_finance')
        if 'WAKE-one' not in rendered:
            assert 'WEEKLY-growth' in rendered and 'WEEKLY-ops_finance' in rendered
            return calls(('ask_analyst', dict(role='growth', message='WAKE-one')),
                ('ask_agent', dict(role='growth', message='WAKE-two')))
        assert 'REPLY-one' in rendered and 'REPLY-two' in rendered
        return calls(('bash', dict(command=ADVANCE)),
            ('ask_agent', dict(role='growth', message='MUST-NOT-RUN')),
            ('write_file', dict(path='old-week.txt', content='wrong')))

    with make_team(mode, respond, 'request-fifo') as team:
        transition = team._transition
        def observed(message, status, reply=None):
            transition(message, status, reply)
            if message.text == 'QUESTION-ops_finance' and status == 'queued':
                both_enqueued.set()
        monkeypatch.setattr(team, '_transition', observed)
        for role in ROLES:
            register(team, role, 'once.py', "with open('script-runs.txt','a') as f: f.write('ran\\n')")
        result = team.run(stop_after_day=7)
        save_evidence(team, 'fifo')
        assert result.reason == 'observation_end', (team.failure, getattr(team, 'failure_detail', None))
        assert answers == ['growth', 'ops_finance']
        queued = sorted((m for m in team.messages if m.kind == 'question' and m.receiver == 'ceo'),
                        key=lambda m: m.enqueue_sequence)
        assert [m.sender for m in queued] == answers

        for role, runtime in team.roles.items():
            assert (runtime.workspace / 'script-runs.txt').read_text() == 'ran\n'
            captured = [json.loads(line)['body'] for line in runtime.usage.path.read_text().splitlines()
                        if json.loads(line)['event'] == 'http_request']
            assert captured == [row['body'] for row in team.test_wire if row['role'] == role]
        assert not (team.roles['ceo'].workspace / 'old-week.txt').exists()
        assert not any(message.text == 'MUST-NOT-RUN' for message in team.messages)
        requests = [message for message in team.messages if message.kind == 'question']
        assert len(requests) == 4 and all(message.status == 'delivered' for message in requests)
        assert len({message.request_id for message in requests}) == 4
        for request in requests:
            receiver = [row['body'] for row in team.test_wire if row['role'] == request.receiver]
            accepted = [body for body in receiver if any(team._request_ref(request) in str(m.get('content', ''))
                        and 'Team request' in str(m.get('content', '')) for m in body['messages'])]
            assert accepted
            assert all(sum(team._request_ref(request) in str(m.get('content', '')) and
                'Team request' in str(m.get('content', '')) for m in body['messages']) == 1 for body in accepted)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_simultaneous_requests_refuse_only_cycle_and_finish_batch_before_delivery(mode):
    barrier = threading.Barrier(2)
    counts = Counter()

    def respond(team, role, body):
        assert_paired(body['messages'])
        counts[role] += 1
        if role == 'ceo':
            assert all('WEEKLY-' + name in json.dumps(body) for name in ROLES[1:])
            return calls(('bash', dict(command=ADVANCE)))
        other = 'ops_finance' if role == 'growth' else 'growth'
        latest_user = [m['content'] for m in body['messages'] if m['role'] == 'user'][-1]
        if counts[role] == 1:
            barrier.wait(10)
            return calls(('ask_agent', dict(role=other, message='CYCLE-' + role)),
                ('write_file', dict(path='batch-tail.txt', content='tail completed')))
        if latest_user.startswith('Team request'):
            assert (team.roles[role].workspace / 'batch-tail.txt').exists()
            assert not (team.roles[other].workspace / 'batch-tail.txt').exists(), 'Sender ran a later tool while waiting'
            if body['messages'][-1]['role'] == 'user':
                return calls(('write_file', dict(path='answer-tool.txt', content='investigated')))
            return final('ANSWER-' + role)
        assert (team.roles[role].workspace / 'batch-tail.txt').exists()
        return final('WEEKLY-' + role)

    with make_team(mode, respond, 'request-cycle') as team:
        result = team.run(stop_after_day=7)
        save_evidence(team, 'cycle')
        assert result.reason == 'observation_end', (team.failure, getattr(team, 'failure_detail', None))
        questions = [message for message in team.messages if message.kind == 'question']
        assert Counter(message.status for message in questions) == {'rejected': 1, 'delivered': 1}
        assert 'Circular wait' in next(message.reply for message in questions if message.status == 'rejected')
        for role in ROLES[1:]:
            assert (team.roles[role].workspace / 'batch-tail.txt').exists()
        assert team.failure is None


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('final_before_request', [False, True])
def test_busy_receiver_finishes_response_and_batch_then_resumes_or_returns_idle(mode, final_before_request, monkeypatch):
    busy = threading.Event()
    enqueued = threading.Event()
    counts = Counter()

    def respond(team, role, body):
        counts[role] += 1
        assert_paired(body['messages'])
        if role == 'ceo':
            assert 'ORIGINAL-GROWTH-REPORT' in json.dumps(body)
            assert 'TEMPORARY-ANSWER' not in [m['content'] for m in body['messages'] if m['role'] == 'user'][-1]
            return calls(('bash', dict(command=ADVANCE)))
        if role == 'ops_finance':
            if counts[role] == 1:
                assert busy.wait(10)
                return calls(('ask_agent', dict(role='growth', message='BUSY-QUESTION')),
                    ('write_file', dict(path='sender-continued.txt', content='after answer')))
            assert 'TEMPORARY-ANSWER' in json.dumps(body)
            return final('OPS-WEEKLY')
        if counts[role] == 1:
            busy.set()
            assert enqueued.wait(10)
            if final_before_request:
                return final('ORIGINAL-GROWTH-REPORT')
            return calls(('write_file', dict(path='first.txt', content='first')),
                         ('write_file', dict(path='second.txt', content='second')))
        latest = [m['content'] for m in body['messages'] if m['role'] == 'user'][-1]
        if latest.startswith('Team request'):
            assert not (team.roles['ops_finance'].workspace / 'sender-continued.txt').exists()
            if not final_before_request:
                assert (team.roles[role].workspace / 'first.txt').exists()
                assert (team.roles[role].workspace / 'second.txt').exists()
            return final('TEMPORARY-ANSWER')
        assert not final_before_request, 'A completed weekly task was restarted'
        assert latest.startswith('The requested replies are complete. Resume')
        return final('ORIGINAL-GROWTH-REPORT')

    with make_team(mode, respond, 'request-busy') as team:
        transition = team._transition
        def observed(message, status, reply=None):
            transition(message, status, reply)
            if message.text == 'BUSY-QUESTION' and status == 'queued':
                enqueued.set()
        monkeypatch.setattr(team, '_transition', observed)
        result = team.run(stop_after_day=7)
        save_evidence(team, 'busy-' + str(final_before_request))
        assert result.reason == 'observation_end', (team.failure, getattr(team, 'failure_detail', None))
        assert counts['growth'] == (2 if final_before_request else 3)
        assert (team.roles['ops_finance'].workspace / 'sender-continued.txt').exists()


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_opening_three_role_cycle_and_unstarted_experts_keep_weekly_tasks(mode):
    opening = Counter()
    def respond(team, role, body):
        assert_paired(body['messages'])
        latest = [m['content'] for m in body['messages'] if m['role'] == 'user'][-1]
        if team.phase == 'ceo_opening':
            assert all(not (runtime.workspace / 'script-runs.txt').exists() for runtime in team.roles.values())
            opening[role] += 1
            if opening[role] == 1:
                target = dict(ceo='growth', growth='ops_finance', ops_finance='ceo')[role]
                tool = 'ask_analyst' if role == 'ceo' else 'ask_agent'
                return calls((tool, dict(role=target, message='CHAIN-' + role)))
            if role == 'ops_finance':
                assert 'Circular wait' in json.dumps(body)
            return final('OPENING-DIRECTIONS' if role == 'ceo' else 'CHAIN-ANSWER-' + role)
        if role == 'ceo':
            assert 'FORMAL-growth' in latest and 'FORMAL-ops_finance' in latest
            return calls(('bash', dict(command=ADVANCE)))
        assert 'OPENING-DIRECTIONS' in latest
        assert (team.roles[role].workspace / 'script-runs.txt').read_text() == 'ran\n'
        return final('FORMAL-' + role)

    with make_team(mode, respond, 'request-three-cycle', opening_responder=respond) as team:
        for role in ROLES:
            register(team, role, 'once.py', "with open('script-runs.txt','a') as f: f.write('ran\\n')")
        result = team.run(stop_after_day=7)
        save_evidence(team, 'three-cycle')
        assert result.reason == 'observation_end', (team.failure, getattr(team, 'failure_detail', None))
        assert opening == dict(ceo=2, growth=2, ops_finance=2)
        assert Counter(m.status for m in team.messages if m.kind == 'question') == dict(delivered=2, rejected=1)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_accepted_late_request_refuses_advance_then_retries_after_complete_batch(mode, monkeypatch):
    enqueued = threading.Event()
    closing = Counter()
    late_job = []

    def respond(team, role, body):
        assert_paired(body['messages'])
        latest = [m['content'] for m in body['messages'] if m['role'] == 'user'][-1]
        rendered = json.dumps(body)
        if role == 'growth' and 'INJECT-LATE-WORK' in latest:
            if not any(m.get('name') == 'ask_agent' for m in body['messages']):
                return calls(('ask_agent', dict(role='ceo', message='LATE-QUESTION')))
            assert 'LATE-ANSWER' in rendered
            return final('Late work completed')
        if role != 'ceo':
            return final('Original report ' + role)
        if latest.startswith('Team request'):
            assert 'LATE-QUESTION' in latest
            assert (team.roles['ceo'].workspace / 'batch-tail.txt').read_text() == 'old batch completed'
            assert team.server.tools.current_day == 0
            assert 'Team requests are pending' in rendered
            return final('LATE-ANSWER')
        closing[role] += 1
        if closing[role] == 1:
            def late():
                runtime = team.roles['growth']
                runtime.agent.add_input('INJECT-LATE-WORK', day=0)
                return team._drive_role('growth', completion='answer')
            late_job.append(team._submit_role('growth', late))
            assert enqueued.wait(10)
            return calls(('bash', dict(command=ADVANCE)),
                ('write_file', dict(path='batch-tail.txt', content='old batch completed')))
        assert 'LATE-ANSWER' in rendered
        assert not team.server._operation_failed
        assert latest.startswith('The requested replies are complete. Resume')
        return calls(('bash', dict(command=ADVANCE)),
            ('write_file', dict(path='old-suffix.txt', content='must be cancelled')))

    with make_team(mode, respond, 'request-advance') as team:
        transition = team._transition
        def observed(message, status, reply=None):
            transition(message, status, reply)
            if message.text == 'LATE-QUESTION' and status == 'queued':
                enqueued.set()
        monkeypatch.setattr(team, '_transition', observed)
        result = team.run(stop_after_day=7)
        save_evidence(team, 'advance-injected-late-job')
        assert result.reason == 'observation_end', (team.failure, getattr(team, 'failure_detail', None))
        assert late_job[0].result() == 'Late work completed'
        assert closing['ceo'] == 2
        assert not (team.roles['ceo'].workspace / 'old-suffix.txt').exists()
        assert len([row for row in team.test_timeline if row['kind'] == 'settlement']) == 1


def test_stop_wakes_sender_wait_and_closes_busy_recipient_model(tmp_path):
    import httpx
    from openai import OpenAI
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from concurrent.futures import ThreadPoolExecutor
    from test_team_stop_residuals import wait_until
    blocked, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            self.wfile.flush()
            blocked.set()
            release.wait(10)
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    listener = threading.Thread(target=server.serve_forever, daemon=True)
    listener.start()
    def respond(team, role, body):
        assert role == 'growth'
        assert blocked.wait(10)
        return calls(('ask_agent', dict(role='ops_finance', message='WAIT-FOR-OPS')))
    try:
        with make_team('git', respond, 'request-stop') as team:
            runtime = team.roles['ops_finance']
            runtime.agent.client.close()
            client = OpenAI(api_key='OFFLINE', base_url=f'http://127.0.0.1:{server.server_port}', max_retries=0,
                            http_client=httpx.Client(timeout=600))
            runtime.agent.client = client
            runtime.usage.attach(client)
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(team.run, stop_after_day=7)
                wait_until(lambda: team._workers['growth'].waiting is not None, seconds=10)
                with pytest.raises(RuntimeError, match='quiescent|requests'):
                    team.checkpoint()
                team.request_pause()
                outcome = result.result(timeout=5)
            save_evidence(team, 'stop-socket')
            assert outcome.reason == 'paused' and team.failure == 'requested_pause'
            assert not any(thread.name.startswith('team-role') and thread.is_alive() for thread in threading.enumerate())
            assert team.server.public_request_state()['active'] == []
            question = next(m for m in team.messages if m.kind == 'question')
            assert question.status == 'cancelled' and not question.input_accepted
            assert runtime.usage.summary['calls'] == 1
            assert runtime.usage.summary['missing_cost'] == 1
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        listener.join(2)


def run_controller_stop_child(entry_json, stop_path):
    import httpx
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from numpy.random import default_rng
    from openai import OpenAI
    from saas_bench.config import BenchmarkConfig
    from saas_bench.database import init_database
    from saas_bench.multi_agent import MultiAgentRuntime
    from saas_bench.tools import AgentTools
    from saas_bench import team_batch as batch
    from saas_bench.run_state import write_json
    from test_preflight_team_checkpoint import PRICING, fast_simulator
    from test_preflight_team_flow import completion
    from test_team_stop_residuals import wait_until
    entry = json.loads(entry_json)
    directory = Path(entry['output_dir']) / 'attempt-001'
    directory.mkdir(parents=True)
    blocked, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            self.wfile.flush()
            blocked.set()
            release.wait(30)
        def log_message(self, *args):
            pass

    provider = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    provider.daemon_threads = True
    listener = threading.Thread(target=provider.serve_forever, daemon=True)
    listener.start()
    conn = init_database(':memory:')
    config = BenchmarkConfig(seed=123)
    simulator = fast_simulator(conn, config, default_rng(123))
    simulator.initialize()
    tools = AgentTools(conn, 0, directory / 'host', config=config, rng=simulator.rng, seed=123)

    def client(role):
        if role == 'ops_finance':
            return OpenAI(api_key='OFFLINE', base_url=f'http://127.0.0.1:{provider.server_port}', max_retries=0,
                          http_client=httpx.Client(timeout=600))
        def respond(request):
            body = json.loads(request.content)
            if role == 'ceo':
                answer = final('Open the week and investigate.')
            else:
                assert blocked.wait(10)
                answer = calls(('ask_agent', dict(role='ops_finance', message='WAITING-UNDER-CONTROLLER')))
            return completion(answer, body.get('stream', False))
        return OpenAI(api_key='OFFLINE', base_url='https://opencode.ai/zen/go/v1', max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(respond)))

    team = MultiAgentRuntime(directory / 'runtime', tools, conn=conn, simulator=simulator, client_factory=client, mode='git', model_pricing=PRICING)
    record = dict(attempt=1, attempt_id='controller-request-stop', status='ready', day=0)
    snapshot = batch.save_checkpoint(team, directory, record, 'ready')
    assert snapshot is not None
    def request_stop():
        wait_until(lambda: blocked.is_set() and team._workers['growth'].waiting is not None
                   and team._workers['growth'].waiting.status == 'queued', seconds=10)
        write_json(directory / 'waiting-proof.json', dict(
            request=asdict(team._workers['growth'].waiting), model_blocked=True))
        Path(stop_path).write_text('operator stop after sender blocks')
    stopper = threading.Thread(target=request_stop)
    stopper.start()
    try:
        result = batch.execute(team, dict(pricing=PRICING), entry, directory, record, directory / 'STOP')
        assert result['status'] == 'paused' and result['runtime_failure'] == 'supervisor_cancelled', result
        assert result['checkpoint'] == str(snapshot)
        write_json(directory / 'scheduler-closure.json', dict(
            threads=[thread.name for thread in threading.enumerate() if thread.name.startswith('team-role')],
            requests=[asdict(message) for message in team.messages if message.kind == 'question'],
            model_calls=team.roles['ops_finance'].usage.summary['calls'],
            missing_cost=team.roles['ops_finance'].usage.summary['missing_cost'],
            active_public=team.server.public_request_state()['active'], checkpoint_exists=snapshot.is_dir()))
    finally:
        release.set()
        stopper.join(2)
        for runtime in team.roles.values():
            runtime.agent.client.close()
        team.close()
        provider.shutdown()
        provider.server_close()
        listener.join(2)
        conn.close()


def test_controller_stop_reaps_waiting_role_workers_and_preserves_initial_checkpoint(observer, monkeypatch):
    import subprocess
    import sys
    from saas_bench import team_batch as batch
    from saas_bench.run_state import write_json
    from test_team_stop_residuals import controller_module
    controller = controller_module()
    source = str(Path(__file__).resolve().parents[1])
    observer.expected['source_root'] = observer.manifest['source_root'] = source
    observer.manifest['runs'] = [observer.manifest['runs'][0]]
    entry = observer.manifest['runs'][0]
    entry.update(pair_id='stop-s123', category='engineering', seed=123, repeat=1, stop_after_day=7)
    policy = observer.output.parent / 'policy.json'
    write_json(policy, dict(identity=observer.expected))
    monkeypatch.setattr(controller, 'identity', observer.module.identity)
    monkeypatch.setattr(controller, 'poll', lambda *args: observer.module.poll(*args, quota_fn=observer.quota))
    popen = subprocess.Popen
    children = []

    def launch(command, **kwargs):
        if '--worker' not in command:
            return popen(command, **kwargs)
        environment = dict(kwargs['env'])
        environment['PYTHONPATH'] += os.pathsep + str(Path(__file__).parent)
        kwargs['env'] = environment
        code = 'from test_team_requests import run_controller_stop_child; import sys; run_controller_stop_child(*sys.argv[1:])'
        child = popen([*command[:command.index(sys.executable)], sys.executable, '-c', code,
                       json.dumps(entry), str(policy.parent / 'STOP')], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(controller.subprocess, 'Popen', launch)
    results = controller.supervise(policy)
    child_logs = {str(path.name): path.read_text() for path in observer.host_logs.glob('*.log')}
    if target := os.environ.get('CEOBENCH_REQUEST_EVIDENCE'):
        destination = Path(target)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'controller-diagnostics.json').write_text(json.dumps(dict(results=results, logs=child_logs), indent=2))
    assert len(results) == 1 and results[0].get('runtime_failure') == 'supervisor_cancelled', (results, child_logs)
    assert results[0]['status'] == 'paused' and not results[0].get('hold')
    directory = Path(entry['output_dir']) / 'attempt-001'
    proof = batch.read(directory / 'scheduler-closure.json')
    assert proof['threads'] == [] and proof['active_public'] == []
    assert proof['checkpoint_exists'] and proof['model_calls'] == proof['missing_cost'] == 1
    assert len(proof['requests']) == 1 and proof['requests'][0]['status'] == 'cancelled'
    assert batch.read(directory / 'waiting-proof.json')['request']['status'] == 'queued'
    exit_record = batch.read(policy.parent / 'controller-exit.json')
    assert exit_record['children_reaped'] and all(row['scope_empty'] for row in exit_record['units'].values())
    assert all(child.poll() == 0 for child in children)
    if target := os.environ.get('CEOBENCH_REQUEST_EVIDENCE'):
        path = Path(target)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'controller-stop.json').write_text(json.dumps(dict(
            closure=proof, attempt=results[0], controller=exit_record,
            waiting=batch.read(directory / 'waiting-proof.json')), indent=2))



@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_known_unstarted_advance_releases_reservation_and_agent_retries(mode, monkeypatch):
    import sqlite3
    from saas_bench import database
    from test_preflight_team_checkpoint import fast_simulator
    save_predictions = database.save_predictions
    saves = 0
    turns = 0

    def fail_once(*args, **kwargs):
        nonlocal saves
        saves += 1
        if saves == 1:
            raise sqlite3.OperationalError('injected prediction persistence failure before advancement')
        return save_predictions(*args, **kwargs)

    monkeypatch.setattr(database, 'save_predictions', fail_once)

    def respond(team, role, body):
        nonlocal turns
        if role != 'ceo':
            return final('WEEKLY-' + role)
        turns += 1
        assert turns <= 2, 'CEO repeated a known failed advance beyond the single retry'
        if turns == 2:
            result = body['messages'][-1]
            assert result['role'] == 'tool' and 'prediction_save_failed' in result['content']
            assert team.server.tools.current_day == 0 and not team.server._operation_failed
            assert not team.server.conn.in_transaction
            assert team.server.public_request_state()['active'] == []
            assert not team._advancing, 'A known non-advance retained the admission reservation'
            advances = [json.loads(path.read_text()) for path in
                        (team.root / 'private/operations').glob('*.json')]
            advances = [row for row in advances if row['kind'] == 'api:advance']
            assert len(advances) == 1 and advances[0]['status'] == 'failed'
            assert advances[0]['advance']['prediction_transaction'] == 'rolled_back'
            assert advances[0]['advance']['progression'] == 'not_advanced'
        return calls(('bash', dict(command=ADVANCE)))

    with make_team(mode, respond, 'request-known-advance-failure') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        result = team.run(stop_after_day=7)
        save_evidence(team, 'known-advance-failure')
        assert (result.day, result.reason) == (7, 'observation_end') and team.failure is None
        assert turns == saves == 2
        assert team.server.public_request_state()['active'] == []
        operations = [json.loads(path.read_text()) for path in (team.root / 'private/operations').glob('*.json')]
        advances = [row for row in operations if row['kind'] == 'api:advance']
        assert Counter(row['status'] for row in advances) == {'failed': 1, 'completed': 1}
        assert all(row['status'] != 'running' for row in operations)


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_terminal_advance_waits_for_owned_http_response_and_checkpoints(mode, monkeypatch):
    from saas_bench.api_server import _APIHandler
    original = _APIHandler._send_json
    delayed = threading.Event()
    def send(handler, data, status=200):
        if handler.path == '/next-week' and data.get('success'):
            delayed.set()
            threading.Event().wait(.6)
        return original(handler, data, status)
    monkeypatch.setattr(_APIHandler, '_send_json', send)
    def respond(team, role, body):
        if role == 'ceo':
            return calls(('bash', dict(command=ADVANCE)),
                         ('write_file', dict(path='old-suffix.txt', content='must not run')))
        return final('WEEKLY-' + role)
    snapshots = []
    with make_team(mode, respond, 'request-terminal', total_days=7) as team:
        from test_preflight_team_checkpoint import fast_simulator
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        team.server.simulator.initialize(resume=True)
        def checkpoint(runtime, boundary):
            if boundary == 'week_boundary':
                snapshots.append(runtime.checkpoint())
        team.checkpoint_callback = checkpoint
        result = team.run(stop_after_day=7)
        save_evidence(team, 'terminal-response')
        assert delayed.is_set() and result.reason == 'natural_end'
        assert team.failure is None, team.failure
        assert team._stage == 'boundary' and len(snapshots) == 1
        assert not (team.roles['ceo'].workspace / 'old-suffix.txt').exists()
        assert not team.roles['ceo'].agent._pending_tool_calls
        assert team.server.public_request_state()['active'] == []
        assert json.loads((snapshots[0] / 'team.json').read_text())['day'] == 7


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_queued_request_enters_next_input_while_prior_answer_keeps_ownership(mode, monkeypatch):
    active_batch, second_queued = threading.Event(), threading.Event()
    counts = Counter()
    first_second_body, sent_events = [], []
    def respond(team, role, body):
        assert_paired(body['messages'])
        counts[role] += 1
        rendered = json.dumps(body['messages'])
        if role != 'ceo':
            if counts[role] == 1:
                if role == 'ops_finance':
                    assert active_batch.wait(10)
                return calls(('ask_agent', dict(role='ceo', message='QUEUED-' + role)))
            assert 'ANSWER-' + role in rendered
            return final('WEEKLY-' + role)
        active = team._workers['ceo'].question
        if active is None:
            return calls(('bash', dict(command=ADVANCE)))
        if active.sender == 'growth' and counts[role] == 1:
            return calls(('write_file', dict(path='q1-batch.txt', content='before intake')))
        if active.sender == 'growth':
            first_second_body.append(body)
            assert 'QUEUED-ops_finance' in rendered, 'Queued input missed the next model boundary'
            if counts[role] == 2:
                assert 'Continue answering' in body['messages'][-1]['content']
                return calls(('write_file', dict(path='q1-second-batch.txt', content='still answering Q1')))
            assert counts[role] == 3
            assert sum('QUEUED-ops_finance' in str(m.get('content')) for m in body['messages']) == 1
            return final('ANSWER-growth')
        assert active.sender == 'ops_finance'
        latest = body['messages'][-1]['content']
        assert team._request_ref(active) in latest and 'QUEUED-ops_finance' not in latest
        assert sum('QUEUED-ops_finance' in str(m.get('content')) for m in body['messages']) == 1
        return final('ANSWER-ops_finance')
    with make_team(mode, respond, 'request-intake') as team:
        transition, execute = team._transition, team.roles['ceo'].executor.execute
        act = team.roles['ceo'].agent.act
        def observed_act(*args, **kwargs):
            action = act(*args, **kwargs)
            body = next(row['body'] for row in reversed(team.test_wire) if row['role'] == 'ceo')
            usage = team.roles['ceo'].usage
            sent_events.append((usage.last_request_event or 'http:' + usage.last_http_attempt_id, body))
            return action
        monkeypatch.setattr(team.roles['ceo'].agent, 'act', observed_act)
        def observed(message, status, reply=None):
            transition(message, status, reply)
            if message.text == 'QUEUED-ops_finance' and status == 'queued':
                second_queued.set()
        def blocked(tool, arguments, **kwargs):
            if tool == 'write_file' and arguments.get('path') == 'q1-batch.txt':
                active_batch.set()
                assert second_queued.wait(10)
            return execute(tool, arguments, **kwargs)
        monkeypatch.setattr(team, '_transition', observed)
        monkeypatch.setattr(team.roles['ceo'].executor, 'execute', blocked)
        result = team.run(stop_after_day=7)
        save_evidence(team, 'intake-during-answer')
        assert result.reason == 'observation_end', team.failure
        q1, q2 = sorted((m for m in team.messages if m.kind == 'question'), key=lambda m: m.enqueue_sequence)
        assert q1.status == q2.status == 'delivered'
        sent = [json.loads(line) for line in team.roles['ceo'].usage.path.read_text().splitlines()
                if json.loads(line)['event'] == 'http_request']
        first = next(row for row in sent if 'QUEUED-ops_finance' in json.dumps(row['body']))
        assert q2.received_request_event == next(event for event, body in sent_events if body == first['body'])
        assert q2.received_request_event is not None
        assert first['body'] == first_second_body[0]
        if mode == 'git':
            assert q2.received_request_event == 'http:' + first['attempt_id']


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('receiver_busy', [False, True])
def test_completed_reply_does_not_form_cycle_while_sender_receipt_is_pending(mode, receiver_busy, monkeypatch):
    first_queued, reverse_attempted = threading.Event(), threading.Event()
    receiver_started, first_completed, receipt_pending = (threading.Event() for _ in range(3))
    reverse_edges = []
    counts = Counter()
    def respond(team, role, body):
        assert_paired(body['messages'])
        counts[role] += 1
        if role == 'ceo':
            return calls(('bash', dict(command=ADVANCE)))
        active = team._workers[role].question
        if active is not None:
            return final('ANSWER-' + active.text)
        if role == 'growth':
            if not first_queued.is_set():
                if receiver_busy:
                    assert receiver_started.wait(10)
                return calls(('ask_agent', dict(role='ops_finance', message='FIRST')))
            return final('WEEKLY-growth')
        # Weekly work may start before FIRST arrives, or after its answer.
        # Temporary question turns must not determine the fake model's next action.
        if not first_completed.is_set():
            assert receiver_busy
            receiver_started.set()
            assert first_queued.wait(10)
            return calls(('write_file', dict(path='ops-root.txt', content='continue')))
        if not reverse_attempted.is_set():
            assert receipt_pending.wait(10)
            return calls(('ask_agent', dict(role='growth', message='REVERSE')))
        assert 'ANSWER-REVERSE' in json.dumps(body)
        return final('WEEKLY-ops_finance')
    with make_team(mode, respond, 'request-completed-edge') as team:
        team.test_event('test_schedule', receiver_busy=receiver_busy)
        service = team._service_inbox
        def scheduled_service(role, **kwargs):
            if role == 'ops_finance' and not receiver_busy:
                assert first_queued.wait(10)
            return service(role, **kwargs)
        monkeypatch.setattr(team, '_service_inbox', scheduled_service)
        transition = team._transition
        record = team.roles['growth'].agent.record_tool_result
        def observed(message, status, reply=None):
            transition(message, status, reply)
            if message.text == 'FIRST' and status == 'queued':
                first_queued.set()
            if message.text == 'FIRST' and status == 'completed':
                first_completed.set()
            if message.text == 'REVERSE' and status in ('queued', 'rejected'):
                waiting = team._workers['growth'].waiting
                edge = dict(sender_waiting=waiting.text if waiting else None,
                    first_status=waiting.status if waiting else None,
                    reverse_status=status, receipt_pending=receipt_pending.is_set())
                reverse_edges.append(edge)
                team.test_event('pending_reply_at_reverse_admission', **edge)
                reverse_attempted.set()
        def delayed(result):
            if 'ANSWER-FIRST' in str(result):
                receipt_pending.set()
                assert reverse_attempted.wait(10)
            return record(result)
        monkeypatch.setattr(team, '_transition', observed)
        monkeypatch.setattr(team.roles['growth'].agent, 'record_tool_result', delayed)
        result = team.run(stop_after_day=7)
        save_evidence(team, 'completed-edge-' + ('busy' if receiver_busy else 'early'))
        assert result.reason == 'observation_end', (team.failure, getattr(team, 'failure_detail', None))
        questions = [m for m in team.messages if m.kind == 'question']
        assert len(questions) == 2 and all(m.status == 'delivered' for m in questions)
        assert reverse_edges == [dict(sender_waiting='FIRST', first_status='completed',
            reverse_status='queued', receipt_pending=True)]
        assert counts['ops_finance'] == (4 if receiver_busy else 3)
        assert result.day == 7 and team.failure is None
        assert all(not worker.waiting and not worker.question and not worker.inbox
            for worker in team._workers.values())
        for role, runtime in team.roles.items():
            usage = [json.loads(line) for line in runtime.usage.path.read_text().splitlines()]
            sent = [row['body'] for row in usage if row['event'] == 'http_request']
            assert sent == [row['body'] for row in team.test_wire if row['role'] == role]
            assert all(not row.get('error') for row in usage)
            for question in (m for m in questions if m.sender == role):
                assert any('ANSWER-' + question.text in str(message.get('content', ''))
                    for body in sent for message in body['messages'] if message['role'] == 'tool')


def test_inprocess_model_reply_keeps_first_logical_request_identity(monkeypatch):
    from openai.types.chat import ChatCompletion
    from saas_bench.agents.bash_agent import agent as module
    from test_preflight_team_flow import completion
    counts = Counter()
    def respond(team, role, body):
        counts[role] += 1
        if role != 'ceo':
            return final('WEEKLY-' + role)
        if counts[role] == 1:
            return calls(('ask_agent', dict(role='growth', message='INPROCESS-QUESTION')))
        assert 'INPROCESS-ANSWER' in json.dumps(body)
        return calls(('bash', dict(command=ADVANCE)))
    with make_team('git', respond, 'request-inprocess') as team:
        stream = module._stream_chat_completion
        def local(client, request, check=lambda: None, *, require_success=False):
            if client is team.roles['growth'].agent.client:
                check()
                return ChatCompletion.model_validate(completion(final('INPROCESS-ANSWER')).json())
            return stream(client, request, check, require_success=require_success)
        monkeypatch.setattr(module, '_stream_chat_completion', local)
        result = team.run(stop_after_day=7)
        save_evidence(team, 'inprocess-reply')
        assert result.reason == 'observation_end', (team.failure, getattr(team, 'failure_detail', None))
        question = next(m for m in team.messages if m.kind == 'question')
        usage = [json.loads(line) for line in team.roles['growth'].usage.path.read_text().splitlines()]
        received = next(row for row in usage if row['event'] == 'request' and 'INPROCESS-QUESTION' in json.dumps(row))
        assert question.status == 'delivered' and question.received_request_event == 'call:' + received['call_id']
        assert not any(row['event'] == 'http_request' for row in usage)


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('exhausted', [False, True])
def test_generation_retry_intakes_queue_and_retains_first_successful_receipt(mode, exhausted, monkeypatch):
    first_reply, queued = threading.Event(), threading.Event()
    counts, receipts = Counter(), []
    def respond(team, role, body):
        assert_paired(body['messages'])
        counts[role] += 1
        if role != 'ceo':
            if counts[role] == 1:
                if role == 'ops_finance':
                    assert first_reply.wait(10)
                return calls(('ask_agent', dict(role='ceo', message='REGENERATE-' + role)))
            return final('WEEKLY-' + role)
        active = team._workers['ceo'].question
        if active is None:
            return calls(('bash', dict(command=ADVANCE)))
        if active.sender == 'growth':
            if counts[role] == 1:
                first_reply.set()
                assert queued.wait(10)
            elif not exhausted:
                assert 'REGENERATE-ops_finance' in json.dumps(body), 'Internal regeneration skipped queued intake'
            if exhausted or counts[role] == 1:
                invalid = calls(('write_file', dict(path='invalid.txt', content='must not run')))
                invalid['tool_calls'][0]['function']['arguments'] = '{'
                return invalid
        return final('ANSWER-' + active.sender)
    with make_team(mode, respond, 'request-regeneration') as team:
        transition = team._transition
        callback = team.roles['ceo'].agent.response_callback
        def observed(message, status, reply=None):
            transition(message, status, reply)
            if message.text == 'REGENERATE-ops_finance' and status == 'queued':
                queued.set()
        def response(**facts):
            usage = team.roles['ceo'].usage
            reference = usage.last_request_event or 'http:' + usage.last_http_attempt_id
            receipts.append((reference, plain(facts['messages'])))
            team.test_event('successful_response', role='ceo', reference=reference)
            if callback:
                callback(**facts)
        monkeypatch.setattr(team, '_transition', observed)
        monkeypatch.setattr(team.roles['ceo'].agent, 'response_callback', response)
        result = team.run(stop_after_day=7)
        save_evidence(team, 'generation-' + ('exhausted' if exhausted else 'corrected'))
        questions = sorted((m for m in team.messages if m.kind == 'question'), key=lambda m: m.enqueue_sequence)
        q1, q2 = questions
        first_q1 = next(reference for reference, body in receipts if 'REGENERATE-growth' in json.dumps(body))
        assert q1.received_request_event == first_q1, 'The first successful response was lost or overwritten'
        first_q2 = next(reference for reference, body in receipts if 'REGENERATE-ops_finance' in json.dumps(body))
        assert q2.received_request_event == first_q2
        bodies = [row['body'] for row in team.test_wire if row['role'] == 'ceo' and
                  'REGENERATE-growth' in json.dumps(row['body'])]
        assert 'REGENERATE-ops_finance' not in json.dumps(bodies[0])
        assert all(sum('REGENERATE-ops_finance' in str(m.get('content')) for m in body['messages']) == 1
                   for body in bodies[1:])
        assert not (team.roles['ceo'].workspace / 'invalid.txt').exists()
        if exhausted:
            assert result.reason == 'paused' and team.failure == 'model_attempt_limit'
            assert counts['ceo'] == 4 and q1.status == 'failed' and q2.status == 'cancelled'
        else:
            assert result.reason == 'observation_end', team.failure
            assert all(q.status == 'delivered' for q in questions)


@pytest.mark.parametrize('api', ['chat', 'responses', 'messages'])
def test_before_request_intake_runs_before_each_provider_serialization(tmp_path, api):
    import httpx
    from anthropic import Anthropic
    from openai import OpenAI
    from saas_bench.agents.bash_agent.agent import BashAgent, FinalText
    from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
    from test_preflight_usage import reply
    from saas_bench.model_usage import ModelUsage
    usage = ModelUsage(tmp_path / "model-usage.jsonl", "provider-intake")
    bodies, received = [], []
    def handle(request):
        bodies.append(json.loads(request.content))
        value = reply(api)
        if len(bodies) == 1:
            if api == 'chat':
                value['choices'][0].update(finish_reason='length')
            elif api == 'responses':
                value['status'] = 'incomplete'
            else:
                value['stop_reason'] = 'max_tokens'
        return httpx.Response(200, json=value)
    client = (Anthropic if api == 'messages' else OpenAI)(api_key='OFFLINE', max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    agent = BashAgent(get_bash_agent_tool_descriptions(), client, model='offline', workspace_path=tmp_path,
        reasoning_effort='high' if api == 'responses' else None, allow_final_text=True, max_output_tokens=100,
        usage_recorder=usage,
        response_callback=lambda **facts: received.append(plain(facts)))
    def intake():
        if bodies:
            agent.add_input('QUEUED-AT-REGENERATION', day=0)
    agent.before_request_callback = intake
    try:
        assert agent.act('Initial task', 0, False, {'day': 0}) == FinalText('hello')
        assert len(bodies) == len(received) == 2
        assert 'QUEUED-AT-REGENERATION' not in json.dumps(bodies[0])
        assert json.dumps(bodies[1]).count('QUEUED-AT-REGENERATION') == 1
        assert agent._llm_attempt == 2 and not agent._pending_tool_calls
    finally:
        if target := os.environ.get('CEOBENCH_REQUEST_EVIDENCE'):
            destination = Path(target)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / ('provider-intake-' + api + '.json')).write_text(json.dumps(dict(
                requests=bodies, responses=received,
                usage=[json.loads(line) for line in usage.path.read_text().splitlines()]), indent=2))
        client.close()


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_receipt_persistence_fault_does_not_regenerate_or_deliver(mode, monkeypatch):
    counts = Counter()
    def respond(team, role, body):
        counts[role] += 1
        if role == 'growth' and counts[role] == 1:
            return calls(('ask_agent', dict(role='ceo', message='PERSIST-RECEIPT')))
        return final('ANSWER')
    with make_team(mode, respond, 'request-receipt-fault') as team:
        transition = team._transition
        failed = []
        def injected(message, status, reply=None):
            if message.kind == 'question' and message.received_request_event and status == 'running' and not failed:
                failed.append(message.request_id)
                raise OSError('injected receipt persistence failure')
            return transition(message, status, reply)
        monkeypatch.setattr(team, '_transition', injected)
        result = team.run(stop_after_day=7)
        save_evidence(team, 'receipt-persistence-fault')
        assert result.reason in ('failed', 'paused') and counts['ceo'] == 1
        assert 'injected receipt persistence failure' in str(team.failure) + str(getattr(team, 'failure_detail', None))
        question = next(m for m in team.messages if m.kind == 'question')
        assert question.status == 'failed' and failed == [question.request_id]
        assert team.server.public_request_state()['active'] == []
        assert not any(t.name.startswith('team-role') and t.is_alive() for t in threading.enumerate())
