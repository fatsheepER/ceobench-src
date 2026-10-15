import json
from pathlib import Path

import httpx
import pytest

from saas_bench import team_batch as batch
from saas_bench.run_lifecycle import WorkerLifecycle
from test_preflight_group_join import group_batch
from test_preflight_team_controller import controller
from test_preflight_team_observer import observer
from test_preflight_team_batch import changed_manifest, frozen, managed_team
from test_preflight_team_flow import completion, final


@pytest.mark.parametrize('failure', ['failed', 'oom-kill', 'paused'])
def test_failed_group_holds_once_and_only_manual_invocation_restarts(observer, controller, monkeypatch, failure, caplog):
    launches, children = group_batch(observer, controller, monkeypatch, first_status=failure)
    controller.module.supervise(controller.policy)
    assert launches.count('s101-pf') == 1 and 's102-pf' not in launches
    assert 's102-git' in launches and all(child.poll() is not None for _, child in children)
    entry = batch.selected(observer.manifest, 's101-pf')
    terminal = batch.read(controller.policy.parent / 'controller-terminal.json')['s101-pf']
    assert terminal['disposition'] == 'hold' and terminal['manual_resume_required']
    first = batch.attempts(entry)[0]
    before = (first / 'attempt.json').read_bytes()
    assert (Path(entry['output_dir']) / 'HOLD').exists()
    week = 0 if failure == 'paused' else 1
    controller.module.supervise(controller.policy, resume=True, run_id='s101-pf', week=week, force_resume=True)
    assert launches.count('s101-pf') == 2 and len(batch.attempts(entry)) == 2
    assert (first / 'attempt.json').read_bytes() == before
    assert '--force-resume' in controller.module.worker_arguments[-1]
    if week == 0:
        assert 'week' in caplog.text.lower()
        saved = batch.read(batch.attempts(entry)[-1] / 'attempt.json')
        assert any('week' in row['message'].lower() for row in saved['validation_warnings'])


@pytest.mark.parametrize('failure,requests', [('network', 5), ('invalid_reply', 4)])
def test_real_model_failure_writes_hold_without_creating_another_attempt(frozen, tmp_path, monkeypatch, failure, requests):
    path = changed_manifest(frozen, tmp_path, lambda manifest: None)
    entry = next(row for row in batch.read(path)['runs'] if row['group'] == 'git')
    sent, delays = [], []
    monkeypatch.setattr(WorkerLifecycle, 'sleep', lambda self, seconds: delays.append(seconds))
    def client(role):
        def respond(request):
            if role == 'ceo':
                sent.append(request)
                if failure == 'network':
                    raise httpx.ConnectError('offline connection refused', request=request)
            return completion(final('No tool action.'), json.loads(request.content).get('stream', False))
        return batch.checked_client(role, 'offline-hold', transport=httpx.MockTransport(respond))
    record = batch.prepare(path, entry['run_id'], live=True, client_factory=client,
        simulator_factory=batch.simulator)
    assert len(sent) == requests and record['status'] == 'paused' and record['hold'] is True
    assert len(batch.attempts(entry)) == 1
    hold = batch.read(batch.attempts(entry)[0] / 'HOLD')
    assert hold['manual_resume_required'] and hold['reason'] == record['runtime_failure']
    if failure == 'network':
        assert delays == [10, 20, 40, 80] and 'offline connection refused' in hold['detail']
    else:
        assert hold['reason'] == 'model_attempt_limit' and '4' in hold['detail']


def test_unknown_public_business_result_holds_for_manual_resume(frozen, tmp_path, monkeypatch):
    from saas_bench.multi_agent import MultiAgentRuntime
    from test_preflight_team_flow import raw
    path = changed_manifest(frozen, tmp_path, lambda manifest: None)
    entry = next(row for row in batch.read(path)['runs'] if row['group'] == 'git')
    def client(role):
        def respond(request):
            return completion(final('Analyst answer.'), json.loads(request.content).get('stream', False))
        return batch.checked_client(role, 'offline-business', transport=httpx.MockTransport(respond))
    def unknown(runtime, observation):
        original = runtime.server.tools.set_prices
        def write_then_disconnect(args):
            original(args)
            raise ConnectionError('business write completed but response was lost')
        monkeypatch.setattr(runtime.server.tools, 'set_prices', write_then_disconnect)
        operation = runtime._begin_operation('api:set_prices')
        status, _ = raw(runtime, 'ceo', '/call', dict(tool='set_prices', args={'A': 29}))
        assert status == 500
        runtime._finish_operation(operation)
    monkeypatch.setattr(MultiAgentRuntime, '_run_ceo', unknown)
    record = batch.prepare(path, entry['run_id'], live=True, client_factory=client,
        simulator_factory=batch.simulator)
    assert record['status'] == 'paused' and record['runtime_failure'] == 'business_operation_unknown'
    assert record['hold'] and len(batch.attempts(entry)) == 1
    assert 'unknown' in record['hold_detail'].lower()
    assert list((batch.attempts(entry)[0] / 'runtime/private/operations').glob('*.json'))


def test_unknown_outcome_discovered_at_final_checkpoint_holds(managed_team, tmp_path, monkeypatch):
    from saas_bench.run_state import write_json
    team, directory, record, manifest, entry = managed_team
    checkpoint = record['checkpoint']
    write_json(team.root / 'private/operations/unknown-at-stop.json',
        dict(kind='api:set_prices', status='running'))
    monkeypatch.setattr(team, 'run', lambda **options: type('Outcome', (),
        dict(day=entry['stop_after_day'], reason='observation_end'))())
    result = batch.execute(team, manifest, entry, directory, record, tmp_path / 'STOP')
    assert result['status'] == 'paused' and result['hold']
    assert result['runtime_failure'] == 'business_operation_unknown'
    assert result['checkpoint'] == checkpoint
    notice = batch.read(directory / 'HOLD')
    assert notice['manual_resume_required'] and 'unknown' in notice['detail'].lower()
