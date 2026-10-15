from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import shlex
import threading
import time

import httpx
import pytest

from saas_bench import api_server, team_batch as batch
from saas_bench.agents.bash_agent.agent import _stream_chat_completion
from saas_bench.role_policy import ROLES
from saas_bench.run_lifecycle import RunCancelled
from saas_bench.team_checkpoint import validate_recovery_source, validate_snapshot
from test_preflight_team_batch import frozen
from test_preflight_team_checkpoint import fast_simulator
from test_preflight_team_flow import ADVANCE, calls, completion, final, make_team, raw
from test_preflight_team_readonly_recovery import managed_git


def public_event(store):
    with closing(store.connect()) as conn:
        event, = conn.execute("SELECT event_id FROM requests WHERE branch=? AND json_extract(request,'$.kind')='public_http'",
            (store.identity['branch_id'],)).fetchall()
    return store.read_event(event[0])


CLIENT = """from novamind_api import _client
transport = _client.urlopen
_client.urlopen = lambda request: transport(request, timeout=.1)
try:
    _client.call('set_prices', {'A': 29})
except Exception:
    print('CLIENT-EXITED', flush=True)
"""


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('cancelled', [False, True])
def test_model_waits_for_parallel_registration_then_continues(mode, cancelled, monkeypatch):
    entered, checked, release = threading.Event(), threading.Event(), threading.Event()
    read_body = api_server._APIHandler._read_body

    def delayed_registration(handler):
        if handler.path == '/daily-scripts' and handler.role == 'ops_finance':
            entered.set()
            assert release.wait(10)
        return read_body(handler)

    monkeypatch.setattr(api_server._APIHandler, '_read_body', delayed_registration)
    with make_team(mode, lambda team, sender, body: final('Confirmed data checked.'), 'parallel-registration') as team:
        team._set_phase('analysis')
        runtime = team.roles['growth']
        check = runtime.usage.request_check
        def record_check():
            checked.set()
            check()
        runtime.usage.request_check = record_check
        payload = dict(model=batch.MODEL, messages=[dict(role='user', content='Check the current world.')],
            stream=True, stream_options={'include_usage': True})
        with ThreadPoolExecutor(max_workers=2) as pool:
            registration = pool.submit(raw, team, 'ops_finance', '/daily-scripts',
                dict(name='weekly.py', content="print('registered')\n"))
            try:
                assert entered.wait(5)
                sending = pool.submit(runtime.usage.call, 'chat', payload,
                    lambda: _stream_chat_completion(runtime.agent.client, payload))
                assert checked.wait(5)
                threading.Event().wait(.1)
                assert not sending.done(), 'A normal parallel write caused premature pause'
                assert team.test_wire == [], 'Model transport crossed an unconfirmed write'
                if cancelled:
                    team._lifecycle.cancel('requested_pause')
                    with pytest.raises(RunCancelled, match='^requested_pause$'):
                        sending.result(timeout=1)
            finally:
                release.set()
            status, result = registration.result(timeout=5)
            assert status == 200 and result['success']
            if not cancelled:
                response = sending.result(timeout=5)
                assert response.choices[0].message.content == 'Confirmed data checked.'
        assert len(team.test_wire) == int(not cancelled)
        assert team.server.get_daily_scripts('ops_finance') == {'weekly.py': "print('registered')\n"}


@pytest.mark.parametrize('managed_git', ['git', 'pf'], indirect=True)
@pytest.mark.parametrize('entrypoint', ['bash', 'public_cli'])
def test_successful_client_pending_write_pauses_real_attempt(managed_git, tmp_path, monkeypatch, entrypoint):
    team, directory, record, manifest, entry = managed_git
    entered, release = threading.Event(), threading.Event()
    read_body = api_server._APIHandler._read_body
    wire = []
    hold = tmp_path / 'HOLD'

    def delayed_body(handler):
        if handler.path == '/call':
            entered.set()
            assert release.wait(15)
        return read_body(handler)

    monkeypatch.setattr(api_server._APIHandler, '_read_body', delayed_body)
    command = ('python -c ' if entrypoint == 'bash' else './novamind-operation python-c ') + shlex.quote(CLIENT)

    def transport(request):
        body = json.loads(request.content)
        wire.append(body)
        if len(wire) == 1:
            message = calls(('bash', dict(command=command)))
        else:
            hold.touch()
            message = calls(('bash', dict(command=ADVANCE)))
        return completion(message, body.get('stream', False))

    role = team.roles['ceo']
    role.agent.client.close()
    role.agent.client = batch.checked_client('ceo', 'pending-write-test', transport=httpx.MockTransport(transport))
    role.usage.attach(role.agent.client)
    team._stage = 'ceo'
    team._week = dict(observation='Offline pending-write check.', ceo_started=False)
    team._set_phase('ceo')
    source = batch.save_checkpoint(team, directory, record, 'answers_ready')
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(batch.execute, team, manifest, entry, directory, record, hold)
        try:
            assert entered.wait(5)
            result = running.result(timeout=10)
            assert len(wire) == 1, 'A new model request crossed the unresolved business write'
            assert role.executor.last_status == 'succeeded'
            assert role.executor._bash_facts['exit_code'] == 0
            assert role.executor._bash_facts['boundary_closed'] is True
            assert result['status'] == 'paused'
            assert result['runtime_failure'] == 'business_operation_pending'
            assert result['checkpoint'] == str(source)
            wrapper = next(batch.read(path) for path in (team.root / 'private/operations').glob('*.json')
                if batch.read(path)['kind'] == 'ceo_tool:bash')
            assert wrapper['status'] == 'running'
            assert wrapper['recovery']['classification'] == 'pending_business_write'
            assert wrapper['recovery']['client_closed'] is True
            assert any(request['read_only'] is False for request in wrapper['recovery']['pending_requests'])
            assert team.server._operation_failed is False
            _, saved = validate_snapshot(source)
            validate_recovery_source(source, saved, current_root=team.root)
        finally:
            release.set()
            running.result(timeout=10)
            deadline = time.monotonic() + 5
            while team.server.public_request_state()['active'] and time.monotonic() < deadline:
                threading.Event().wait(.01)
    assert team.server.public_request_state()['active'] == []
    assert team.server._operation_failed is False
    assert team.server.conn.execute('SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] != 29
    batch.close_team(team, record)
    validate_recovery_source(source, saved, current_root=team.root)
    wrapper = next(batch.read(path) for path in (team.root / 'private/operations').glob('*.json')
        if batch.read(path)['kind'] == 'ceo_tool:bash')
    assert (wrapper['status'], wrapper['recovery']['classification']) == ('completed', 'read_only')
    assert batch.read(directory / 'attempt.json')['recovery_validation']['eligible'] is True
    def resume_client(role):
        def resume_response(request):
            body = json.loads(request.content)
            message = calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final('Existing data checked.')
            return completion(message, body.get('stream', False))
        return batch.checked_client(role, 'pending-write-resume', transport=httpx.MockTransport(resume_response))
    resumed = batch.prepare(tmp_path / 'manifest.json', entry['run_id'], resume=True, live=True,
        client_factory=resume_client, simulator_factory=fast_simulator)
    assert (resumed['status'], resumed['day'], resumed['attempt']) == ('finished', 28, 2)
    assert resumed['source_checkpoint'] == str(source)


@pytest.mark.parametrize('managed_git', ['git', 'pf'], indirect=True)
@pytest.mark.parametrize('operation', ['set_prices', 'next_week'])
def test_confirmed_late_business_write_survives_real_resume(managed_git, tmp_path, monkeypatch, operation):
    team, directory, record, manifest, entry = managed_git
    entered, release = threading.Event(), threading.Event()
    apply = api_server._TOOL_DISPATCH['set_prices']
    wire = []

    def late_write(tools, args):
        result = apply(tools, args)
        entered.set()
        assert release.wait(20)
        return result

    simulated = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
    simulated.initialize(resume=True)
    simulated.shock_manager = team.server.shock_manager
    assert simulated.restore_rng_states()
    team.server.simulator = simulated
    if operation == 'set_prices':
        monkeypatch.setitem(api_server._TOOL_DISPATCH, 'set_prices', late_write)
        code = CLIENT
    else:
        step = team.server.simulator.step_week
        def late_week():
            result = step()
            entered.set()
            assert release.wait(20)
            return result
        monkeypatch.setattr(team.server.simulator, 'step_week', late_week)
        code = CLIENT.replace("_client.call('set_prices', {'A': 29})",
            "_client.next_week({f'cash_{h}wk': {'point': 1000000, 'lower': 0, 'upper': 2000000} for h in (1,4,12,26)}, rationale='offline late week')")
    def transport(request):
        body = json.loads(request.content)
        wire.append(body)
        actions = [('bash', dict(command='python -c ' + shlex.quote(code)))]
        if operation == 'next_week':
            actions.append(('write_file', dict(path='old-week.txt', content='must be cancelled')))
        return completion(calls(*actions), body.get('stream', False))
    role = team.roles['ceo']
    role.agent.client.close()
    role.agent.client = role.usage.attach(batch.checked_client('ceo', 'confirmed-write-test', transport=httpx.MockTransport(transport)))
    team._stage = 'ceo'
    team._week = dict(observation='Offline late-write check.', ceo_started=False)
    team._set_phase('ceo')
    source = batch.save_checkpoint(team, directory, record, 'answers_ready')
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(batch.execute, team, manifest, entry, directory, record, tmp_path / 'HOLD')
        try:
            assert entered.wait(5)
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                pending = [batch.read(path) for path in (team.root / 'private/operations').glob('*.json')
                    if batch.read(path).get('recovery', {}).get('classification') == 'pending_business_write']
                if pending:
                    break
                threading.Event().wait(.01)
            assert pending
            assert len(wire) == 1
        finally:
            release.set()
        result = running.result(timeout=15)
    assert result['status'] == 'paused'
    batch.close_team(team, record)
    wrapper = next(batch.read(path) for path in (team.root / 'private/operations').glob('*.json')
        if batch.read(path)['kind'] == 'ceo_tool:bash')
    assert (wrapper['status'], wrapper['recovery']['classification']) == ('completed', 'unproven')
    assert batch.read(directory / 'attempt.json') == record
    assert record['recovery_validation']['eligible'] is True
    assert record['checkpoint'] == str(source)
    _, settled = validate_snapshot(record['checkpoint'])
    assert settled['day'] == 0 and settled['stage'] == 'ceo'
    assert not (role.workspace / 'old-week.txt').exists()
    _, saved = validate_snapshot(source)
    validate_recovery_source(source, saved, current_root=team.root)
    resumed_wire = []
    def resume_client(sender):
        def resume_response(request):
            body = json.loads(request.content)
            resumed_wire.append((sender, body))
            message = calls(('bash', dict(command=ADVANCE))) if sender == 'ceo' else final('Existing data checked.')
            return completion(message, body.get('stream', False))
        return batch.checked_client(sender, 'late-write-resume', transport=httpx.MockTransport(resume_response))
    resumed = batch.prepare(tmp_path / 'manifest.json', entry['run_id'], resume=True, live=True,
        client_factory=resume_client, simulator_factory=fast_simulator)
    assert (resumed['status'], resumed['day'], resumed['attempt']) == ('finished', 28, 2)
    assert resumed['source_checkpoint'] == record['checkpoint']
    if operation == 'next_week':
        assert any(sender != 'ceo' and 'Day 7' in json.dumps(body['messages']) for sender, body in resumed_wire)
    else:
        from saas_bench.db_protection import load_session_db
        with closing(load_session_db(Path(resumed['checkpoint']) / 'world.nmdb')) as conn:
            assert conn.execute('SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] != 29


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('role', ROLES)
@pytest.mark.parametrize('network_retry', [False, True])
def test_model_usage_blocks_pending_host_write_and_continues_after_confirmation(mode, role, network_retry, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    read_body = api_server._APIHandler._read_body

    def delayed_body(handler):
        if handler.path == '/call':
            entered.set()
            assert release.wait(15)
        return read_body(handler)

    monkeypatch.setattr(api_server._APIHandler, '_read_body', delayed_body)
    with make_team(mode, lambda team, sender, body: final('Confirmed data checked.'), 'send-guard') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        team._set_phase('ceo')
        source = team.checkpoint()
        executor = team.roles['ceo'].executor
        command = 'python -c ' + shlex.quote(CLIENT)
        sends, waits = [], []

        def client_exits_first():
            output = executor.execute('bash', dict(command=command))
            assert 'CLIENT-EXITED' in output
            assert executor.last_status == 'succeeded'
            assert entered.is_set() and not release.is_set()

        def transport(request):
            sends.append(request.read())
            if network_retry and len(sends) == 1:
                client_exits_first()
                raise httpx.ConnectError('offline network failure', request=request)
            return completion(final('Confirmed data checked.'), stream=True)

        runtime = team.roles[role]
        runtime.agent.client.close()
        runtime.agent.client = runtime.usage.attach(batch.checked_client(role, 'host-write-guard',
            transport=httpx.MockTransport(transport)))
        runtime.agent.client.max_retries = 1
        monkeypatch.setattr(runtime.usage.lifecycle, 'sleep', waits.append)
        payload = dict(model=batch.MODEL, messages=[dict(role='user', content='Check the current world.')],
            reasoning_effort='high', extra_body={'thinking': {'type': 'enabled'}},
            temperature=1.0, max_tokens=16384, stream=True, stream_options={'include_usage': True})
        try:
            if not network_retry:
                client_exits_first()
            with pytest.raises(RunCancelled, match='business_operation_pending'):
                runtime.usage.call('chat', payload, lambda: _stream_chat_completion(runtime.agent.client, payload))
            assert len(sends) == int(network_retry)
            assert waits == ([10] if network_retry else [])
            assert runtime.agent.client.max_retries == 1
            assert team.server._operation_failed is False
        finally:
            release.set()
            deadline = time.monotonic() + 5
            while team.server.public_request_state()['active'] and time.monotonic() < deadline:
                threading.Event().wait(.01)
        assert team.server.public_request_state()['active'] == []
        assert team.server.conn.execute('SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] == 29
        assert team.server._operation_failed is False
        if mode == 'pf':
            store = team.roles['ceo'].store
            event = public_event(store)
            assert event['result']['status'] == 'succeeded'
            assert event['delivery']['send_state'] == 'failed'
            assert event['client_receipt']['receive_state'] == 'transport_error'
            store.assert_healthy()
        response = runtime.usage.call('chat', payload, lambda: _stream_chat_completion(runtime.agent.client, payload))
        assert response.choices[0].message.content == 'Confirmed data checked.'
        assert len(sends) == 1 + int(network_retry)
        _, saved = validate_snapshot(source)
        validate_recovery_source(source, saved, current_root=team.root)


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('denied', [False, True])
@pytest.mark.parametrize('phase', ['host_result', 'delivery'])
def test_confirmed_public_response_allows_model_while_handler_closes(mode, denied, phase, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    delivered = api_server._APIHandler._capture_delivery

    def delayed_cleanup(handler, state):
        if handler.path == '/call' and phase == 'host_result':
            entered.set()
            assert release.wait(15)
        delivered(handler, state)
        if handler.path == '/call' and phase == 'delivery':
            entered.set()
            assert release.wait(15)

    monkeypatch.setattr(api_server._APIHandler, '_capture_delivery', delayed_cleanup)
    with make_team(mode, lambda team, role, body: final('Confirmed response checked.'), 'confirmed-response') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        team._set_phase('analysis' if denied else 'ceo')
        executor = team.roles['growth' if denied else 'ceo'].executor
        operation = None if denied else team._begin_operation('ceo_tool:bash')
        pending_role = None
        checkpoint_pool = ThreadPoolExecutor(max_workers=1)
        checkpoint_started = threading.Event()
        def checkpoint():
            checkpoint_started.set()
            return team.checkpoint()
        code = "from novamind_api import _client\ntry: _client.call('set_prices', {'A': 29})\nexcept _client.NovaMindAPIError: print('REJECTED')\nelse: print('CONFIRMED')"
        try:
            output = executor.execute('bash', dict(command='python -c ' + shlex.quote(code)))
            assert ('REJECTED' if denied else 'CONFIRMED') in output
            assert entered.is_set() and not release.is_set()
            assert executor.last_status == 'succeeded'
            assert team.server.public_request_state()['active']
            if operation:
                team._finish_operation(operation, executor)
                assert batch.read(operation)['status'] == 'completed'
            for role in ROLES:
                runtime = team.roles[role]
                payload = dict(model=batch.MODEL, messages=[dict(role='user', content='Check the confirmed response.')],
                    stream=True, stream_options={'include_usage': True})
                if mode == 'pf' and phase == 'host_result' and runtime.executor is executor:
                    with pytest.raises(RunCancelled, match='^sql_evidence_pending$'):
                        runtime.usage.call('chat', payload,
                            lambda: _stream_chat_completion(runtime.agent.client, payload))
                    assert runtime.store.fault is None
                    pending_role = runtime
                else:
                    result = runtime.usage.call('chat', payload,
                        lambda: _stream_chat_completion(runtime.agent.client, payload))
                    assert result.choices[0].message.content == 'Confirmed response checked.'
            assert len(team.test_wire) == 3 - int(pending_role is not None)
            saving = checkpoint_pool.submit(checkpoint)
            assert checkpoint_started.wait(1)
            threading.Event().wait(.2)
            assert not saving.done()
        finally:
            release.set()
            deadline = time.monotonic() + 5
            while team.server.public_request_state()['active'] and time.monotonic() < deadline:
                threading.Event().wait(.01)
            checkpoint_pool.shutdown(wait=True)
        assert saving.result().exists()
        if pending_role:
            result = pending_role.usage.call('chat', payload,
                lambda: _stream_chat_completion(pending_role.agent.client, payload))
            assert result.choices[0].message.content == 'Confirmed response checked.'
            assert len(team.test_wire) == 3
        assert team.server._operation_failed is False


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_ceo_wrapper_accepts_identical_finalization_and_rejects_conflict(mode):
    with make_team(mode, lambda team, role, body: final('unused'), 'finalization-replay') as team:
        team._set_phase('ceo')
        executor = team.roles['ceo'].executor
        operation = team._begin_operation('ceo_tool:bash')
        executor.execute('bash', dict(command='printf CONFIRMED'))
        team._finish_operation(operation, executor)
        before = operation.read_bytes()
        team._finish_operation(operation, executor)
        assert operation.read_bytes() == before
        executor.last_status = 'result_unknown'
        with pytest.raises(RuntimeError, match='Conflicting operation finalization'):
            team._finish_operation(operation, executor)
        assert operation.read_bytes() == before


def test_early_public_client_receipt_survives_bind_and_replays_safely(monkeypatch):
    from saas_bench.sql_evidence import EvidencePending
    entered, release = threading.Event(), threading.Event()
    with make_team('pf', lambda team, role, body: final('Late write checked.'), 'early-receipt') as team:
        team._set_phase('ceo')
        runtime = team.roles['ceo']
        store = runtime.store
        bind = store.bind_client

        def delayed_bind(event, token):
            entered.set()
            assert release.wait(15)
            return bind(event, token)

        monkeypatch.setattr(store, 'bind_client', delayed_bind)
        with ThreadPoolExecutor(max_workers=1) as pool:
            running = pool.submit(runtime.executor.execute, 'bash', dict(command='python -c ' + shlex.quote(CLIENT)))
            try:
                assert entered.wait(5)
                assert 'CLIENT-EXITED' in running.result(timeout=8)
                assert runtime.executor.last_status == 'succeeded'
                assert store.fault is None and not store.fault_path.exists()
                with closing(store.connect()) as conn:
                    receipt, = conn.execute('SELECT token,parent,payload FROM client_receipts').fetchall()
                    assert conn.execute('SELECT count(*) FROM client_calls').fetchone()[0] == 0
                    context = conn.execute('SELECT token FROM capture_contexts WHERE event_id=?', (receipt['parent'],)).fetchone()[0]
                callback = dict(context=context, call=receipt['token'], record=json.loads(receipt['payload']))
                assert raw(team, 'ceo', '/_capture', callback) == (200, {'accepted': True})
                conflicting = dict(callback, record=dict(callback['record'], state='received'))
                assert raw(team, 'ceo', '/_capture', conflicting) == (400, {'accepted': False})
                assert raw(team, 'growth', '/_capture', callback) == (400, {'accepted': False})
                unmatched = dict(callback, call='f' * 32)
                assert raw(team, 'ceo', '/_capture', unmatched) == (400, {'accepted': False})
                with pytest.raises(EvidencePending):
                    store.assert_healthy(settle=.02)
                assert store.fault is None and not store.fault_path.exists()
            finally:
                release.set()
                running.result(timeout=8)
                deadline = time.monotonic() + 5
                while team.server.public_request_state()['active'] and time.monotonic() < deadline:
                    threading.Event().wait(.01)
        store.assert_healthy()
        event = public_event(store)
        assert event['result']['status'] == 'succeeded'
        assert event['delivery']['send_state'] == 'failed'
        assert event['client_receipt']['receive_state'] == 'transport_error'
        assert team.server.conn.execute('SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] == 29
        before = store.read_event(event['request']['event_id'])
        assert raw(team, 'ceo', '/_capture', callback) == (200, {'accepted': True})
        assert raw(team, 'ceo', '/_capture', conflicting) == (400, {'accepted': False})
        bind(event['request']['event_id'], receipt['token'])
        assert store.read_event(event['request']['event_id']) == before
        with closing(store.connect()) as conn, conn:
            conn.execute('DROP TABLE client_receipts')
        from saas_bench.sql_evidence import SQLEvidenceStore
        SQLEvidenceStore(store.path, store.identity)
        assert raw(team, 'ceo', '/_capture', callback) == (200, {'accepted': True})
        assert raw(team, 'ceo', '/_capture', conflicting) == (400, {'accepted': False})
        assert store.read_event(event['request']['event_id']) == before
        payload = dict(model=batch.MODEL, messages=[dict(role='user', content='Use the confirmed late write.')], stream=True)
        response = runtime.usage.call('chat', payload, lambda: _stream_chat_completion(runtime.agent.client, payload))
        assert response.choices[0].message.content == 'Late write checked.'
