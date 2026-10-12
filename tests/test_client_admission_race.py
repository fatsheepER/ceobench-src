from concurrent.futures import ThreadPoolExecutor
import shlex
import threading
import time

from saas_bench import team_batch as batch
from saas_bench.agents.bash_agent.agent import _stream_chat_completion
from test_execution_capture import records
from test_pending_business_requests import CLIENT
from test_preflight_team_flow import final, make_team


def test_parent_closes_before_public_admission_finalizes_early_receipt(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    with make_team('pf', lambda team, role, body: final('Rejected write checked.'), 'admission-race') as team:
        team._set_phase('ceo')
        runtime = team.roles['ceo']
        store = runtime.store
        parent = store.parent

        def delayed_parent(token, active=True):
            if token and active:
                entered.set()
                assert release.wait(15)
            return parent(token, active=active)

        monkeypatch.setattr(store, 'parent', delayed_parent)
        with ThreadPoolExecutor(max_workers=1) as pool:
            running = pool.submit(runtime.executor.execute, 'bash', dict(command='python -c ' + shlex.quote(CLIENT)))
            try:
                assert entered.wait(5)
                assert 'CLIENT-EXITED' in running.result(timeout=8)
                assert runtime.executor.last_status == 'succeeded'
                assert store.fault is None and not store.fault_path.exists()
            finally:
                release.set()
                running.result(timeout=8)
                deadline = time.monotonic() + 5
                while team.server.public_request_state()['active'] and time.monotonic() < deadline:
                    threading.Event().wait(.01)
        assert team.server.public_request_state()['active'] == []
        store.assert_healthy()
        event = next(row for row in records(store) if row['request']['kind'] == 'public_http')
        assert event['result']['status'] == 'failed'
        assert event['result']['http_status'] == 403
        assert event['delivery']['send_state'] == 'failed'
        assert event['client_receipt']['receive_state'] == 'transport_error'
        assert team.server.conn.execute('SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] != 29
        assert team.server._operation_failed is False
        assert team.server.public_request_state()['mutation_epoch'] == 0
        assert not list((team.root / 'private/operations').glob('*.json'))
        payload = dict(model=batch.MODEL, messages=[dict(role='user', content='Use the confirmed rejection.')], stream=True)
        response = runtime.usage.call('chat', payload, lambda: _stream_chat_completion(runtime.agent.client, payload))
        assert response.choices[0].message.content == 'Rejected write checked.'
