from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

import httpx
import pytest

from saas_bench import team_batch as batch
from saas_bench.team_checkpoint import validate_recovery_source, validate_snapshot
from saas_bench.run_state import file_hash, write_json
from test_preflight_team_batch import frozen
from test_preflight_team_checkpoint import fast_simulator
from test_preflight_team_flow import ADVANCE, calls, completion, final
from test_preflight_team_recovery_approval import approved_attempt
from test_preflight_team_batch import managed_team


@pytest.fixture
def managed_git(frozen, tmp_path, request):
    manifest = batch.canonical(frozen[2])
    for entry in manifest['runs']:
        entry['output_dir'] = str(tmp_path / 'runs' / entry['run_id'])
    path = tmp_path / 'manifest.json'
    write_json(path, manifest)
    write_json(tmp_path / 'freeze.json', dict(manifest_sha256=file_hash(path)))
    entry = next(row for row in manifest['runs'] if row['group'] == request.param)
    with batch.run_lock(Path(entry['output_dir'])):
        directory, record, _ = batch.register_attempt(path, manifest, entry)
        team = batch.make_team(manifest, entry, directory)
    try:
        yield team, directory, record, manifest, entry
    finally:
        batch.close_team(team)


@pytest.mark.parametrize('managed_git,interruption', [(group, case) for group in ('git', 'pf')
    for case in ('completed', 'cancelled', 'pending_query', 'pending_write', 'registered_write', 'local_write')]
    + [('pf', 'pending_capture')], indirect=['managed_git'])
def test_readonly_wrapper_stop_resumes_real_attempt(managed_git, tmp_path, monkeypatch, interruption):
    from saas_bench import api_server
    team, directory, record, manifest, entry = managed_git
    hold = tmp_path / 'HOLD'
    entered, release = threading.Event(), threading.Event()
    original = api_server.execute_query
    wire = []

    def delayed_query(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    if interruption == 'pending_query':
        monkeypatch.setattr(api_server, 'execute_query', delayed_query)
    if interruption == 'pending_capture':
        from saas_bench import execution_capture
        receive = execution_capture.receive_client
        def delayed_capture(handler):
            entered.set()
            assert release.wait(10)
            return receive(handler)
        monkeypatch.setattr(execution_capture, 'receive_client', delayed_capture)
    command = 'python -c "import novamind_api as nm; nm.query(\'SELECT 1 AS n\')"'
    blocked = interruption in ('pending_write', 'registered_write', 'local_write')
    if interruption == 'registered_write':
        command = './novamind-operation register-daily-script registered.py'
    if interruption == 'local_write':
        command = 'printf changed > changed.txt'
    if interruption == 'pending_write':
        read_body = api_server._APIHandler._read_body
        def delayed_body(handler):
            if handler.path == '/call':
                entered.set()
                assert release.wait(10)
            return read_body(handler)
        monkeypatch.setattr(api_server._APIHandler, '_read_body', delayed_body)
        command = 'python -c "from novamind_api._client import call; call(\'set_prices\', {\'A\':29})"'
    if interruption == 'cancelled':
        command += '; sleep 60'

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
    role.agent.client = batch.checked_client('ceo', 'readonly-test', transport=httpx.MockTransport(transport))
    role.usage.attach(role.agent.client)
    team._stage = 'ceo'
    team._week = dict(observation='Offline recovery check.', ceo_started=False)
    team._set_phase('ceo')
    if interruption == 'registered_write':
        (role.workspace / 'registered.py').write_text('print("registered calculation")\n')
    source = batch.save_checkpoint(team, directory, record, 'answers_ready')
    if interruption in ('pending_query', 'pending_write', 'pending_capture'):
        role.executor.bash_timeout = .5
        role.executor.bash_warn_after = .1

    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(batch.execute, team, manifest, entry, directory, record, hold)
        try:
            if interruption == 'cancelled':
                audit = role.audit.path
                for _ in range(500):
                    if audit.exists() and any(json.loads(line).get('kind') == 'public_http' and
                            json.loads(line).get('path') == '/query' for line in audit.read_text().splitlines()):
                        break
                    threading.Event().wait(.01)
                else:
                    pytest.fail('Public SQL client did not finish before cancellation')
                team.request_pause()
            elif interruption in ('pending_query', 'pending_write') and entry['group'] == 'pf':
                assert entered.wait(5)
                team.request_pause()
            result = running.result(timeout=7)
            assert result['status'] == ('failed' if interruption == 'pending_write' and entry['group'] == 'pf' else 'paused')
            assert result['checkpoint'] == str(source)
            assert result['recovery_validation']['eligible'] is not blocked
            assert role.executor._bash_facts['boundary_closed'] is True
            if interruption in ('pending_query', 'pending_write', 'pending_capture'):
                assert entered.is_set() and not release.is_set()
            if interruption in ('pending_query', 'pending_capture'):
                wrappers = [batch.read(path) for path in (team.root / 'private/operations').glob('*.json')
                    if batch.read(path)['kind'].startswith('ceo_tool:')]
                assert wrappers[-1]['recovery']['classification'] == 'pending_read_only'
                if entry['group'] == 'pf':
                    assert role.executor._pf_service is None
            _, state = validate_snapshot(source)
            for operation in (team.root / 'private/operations').glob('*.json'):
                if batch.read(operation)['status'] not in ('completed', 'interrupted'):
                    continue
                before = operation.read_bytes()
                team._finish_operation(operation, role.executor)
                assert operation.read_bytes() == before
            if blocked:
                with pytest.raises(RuntimeError, match='changed|unknown'):
                    validate_recovery_source(source, state, current_root=team.root)
            else:
                validate_recovery_source(source, state, current_root=team.root)
        finally:
            release.set()
            running.result(timeout=7)
    team.close()
    def resume_client(name):
        def resume_response(request):
            body = json.loads(request.content)
            message = calls(('bash', dict(command=ADVANCE))) if name == 'ceo' else final('Existing data checked.')
            return completion(message, body.get('stream', False))
        return batch.checked_client(name, 'resume-test', transport=httpx.MockTransport(resume_response))

    hold.unlink(missing_ok=True)
    if blocked:
        with pytest.raises(RuntimeError, match='changed|unknown'):
            batch.prepare(tmp_path / 'manifest.json', entry['run_id'], resume=True, live=True,
                client_factory=resume_client, simulator_factory=fast_simulator)
        assert batch.attempts(entry) == [directory]
        return
    resumed = batch.prepare(tmp_path / 'manifest.json', entry['run_id'], resume=True, live=True,
        client_factory=resume_client, simulator_factory=fast_simulator)
    assert resumed['status'] == 'finished' and resumed['day'] == 28
    assert resumed['attempt'] == 2 and resumed['source_checkpoint'] == str(source)
    assert batch.read(directory / 'attempt.json')['status'] == 'paused'
    validate_snapshot(resumed['checkpoint'])


@pytest.mark.parametrize('classification,kind,status,allowed,contradiction', [
    ('read_only_interrupted', 'ceo_tool:bash', 'running', True, {}),
    ('pending_read_only', 'ceo_tool:bash', 'completed', True, {}),
    ('read_only_interrupted', 'api:set_prices', 'running', False, {}),
    ('unknown_business_write', 'api:set_prices', 'running', True, {}),
    ('read_only_interrupted', 'ceo_tool:bash', 'running', False,
        dict(business_operations_before={}, recovery=dict(business_operations_after={'write.json': 'changed'}))),
    ('read_only_interrupted', 'ceo_tool:bash', 'running', False,
        dict(mutation_epoch_before=0, recovery=dict(mutation_epoch_after=1))),
    ('read_only_interrupted', 'ceo_tool:bash', 'running', False,
        dict(workspace_sha256_before='a', recovery=dict(workspace_sha256_after='b'))),
    ('read_only_interrupted', 'ceo_tool:bash', 'running', False,
        dict(recovery=dict(operation_failed=True))),
])
def test_legacy_residual_approval_is_exact_and_preserves_old_attempt(approved_attempt,
        monkeypatch, classification, kind, status, allowed, contradiction):
    from saas_bench import team_recovery as recovery
    approved = approved_attempt
    root = approved.directory / 'runtime'
    operation = root / 'private/operations/legacy-tail.json'
    write_json(operation, dict(kind=kind, status=status, day=0, **contradiction))
    resources = approved.approval_path.parent / 'resources.json'
    saved = approved.revision['attempts'][approved.entry['run_id']]
    saved['attempt_id'] = approved.previous['attempt_id']
    unit = dict(unit='readonly-test.scope', invocation_id='offline-bound-invocation',
        control_group='/user.slice/readonly-test.scope', scope_empty=True)
    write_json(resources, dict(children_reaped=True,
        units={approved.entry['run_id']: unit},
        attempts={approved.entry['run_id']: {key: saved[key] for key in ('attempt', 'attempt_id', 'attempt_sha256')}}))
    saved['residual_operations'] = [dict(root=str(root), operation=operation.name,
        sha256=file_hash(operation), classification=classification,
        reason='The inspected original worker and its children are closed. Business coverage is separately bound.',
        evidence={str(resources): file_hash(resources)}, resource_evidence=str(resources),
        worker_unit=unit['unit'], worker_invocation_id=unit['invocation_id'])]
    if classification == 'unknown_business_write':
        saved['allow_unknown_business_write'] = True
    write_json(approved.approval_path, approved.revision)
    monkeypatch.setenv('CEOBENCH_RECOVERY_APPROVAL_SHA256', file_hash(approved.approval_path))
    historical = (approved.directory / 'attempt.json').read_bytes()
    if allowed:
        assert recovery.recovery_source(approved.manifest_path, approved.entry,
            approved.previous, approved.directory) == approved.snapshot
    else:
        with pytest.raises(ValueError, match='Read-only residual approval'):
            recovery.recovery_source(approved.manifest_path, approved.entry,
                approved.previous, approved.directory)
    assert (approved.directory / 'attempt.json').read_bytes() == historical
    operation.write_text(operation.read_text() + '\n')
    with pytest.raises(ValueError, match='residual operation changed'):
        recovery.recovery_source(approved.manifest_path, approved.entry, approved.previous, approved.directory)
