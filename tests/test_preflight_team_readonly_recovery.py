from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import shlex
from types import SimpleNamespace

import httpx
import pytest

from saas_bench import team_batch as batch
from saas_bench.team_checkpoint import validate_recovery_source, validate_snapshot
from saas_bench.run_state import file_hash, tree_hash, write_json
from test_preflight_team_batch import frozen
from test_preflight_team_checkpoint import fast_simulator
from test_preflight_team_flow import ADVANCE, calls, completion, final
from test_preflight_team_recovery_approval import approved_attempt, bind_approval
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
    for case in ('completed', 'cancelled', 'cancelled_write', 'pending_query', 'pending_write', 'registered_write', 'local_write')]
    + [('pf', 'pending_capture'), ('git', 'symlink'), ('pf', 'symlink')], indirect=['managed_git'])
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
    blocked = interruption in ('pending_write', 'registered_write')
    hashed_symlinks = []
    if interruption == 'symlink':
        from saas_bench import run_state
        target = tmp_path / 'host-canary.txt'
        target.write_text('Nonsensitive host fixture.\n')
        original_hash = run_state.file_hash
        def observed_hash(path):
            if Path(path).is_symlink():
                hashed_symlinks.append(str(path))
            return original_hash(path)
        monkeypatch.setattr(run_state, 'file_hash', observed_hash)
        command = 'ln -s ' + shlex.quote(str(target)) + ' evidence-link'
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
    if interruption == 'cancelled_write':
        command = 'printf changed > changed.txt; ' + command
    if interruption in ('cancelled', 'cancelled_write'):
        command += '; sleep 60'

    def transport(request):
        body = json.loads(request.content)
        wire.append(body)
        if len(wire) == 1:
            message = calls(('bash', dict(command=command)))
        elif interruption == 'symlink' and len(wire) == 2:
            message = calls(('read_file', dict(path='MEMORY.md')))
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
            if interruption in ('cancelled', 'cancelled_write'):
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
            assert result['status'] == 'paused'
            assert result['checkpoint'] == str(source)
            assert result['recovery_validation']['eligible'] is not blocked
            if interruption == 'symlink':
                assert hashed_symlinks == []
                bash = next(batch.read(path) for path in (team.root / 'private/operations').glob('*.json')
                    if batch.read(path)['kind'] == 'ceo_tool:bash')
                assert bash['recovery']['client_closed'] is True
                assert bash['recovery']['process_boundary']['descendants_closed'] is True
            else:
                assert role.executor._bash_facts['boundary_closed'] is True
            if entry['group'] == 'pf' and interruption in ('cancelled', 'cancelled_write', 'pending_query', 'pending_write'):
                assert role.executor.last_status == 'cancelled'
                assert role.executor.evidence_store.fault is None
                assert not role.executor.evidence_store.fault_path.exists()
            if interruption in ('pending_query', 'pending_write', 'pending_capture'):
                assert entered.is_set() and not release.is_set()
            if interruption in ('pending_query', 'pending_capture'):
                wrappers = [batch.read(path) for path in (team.root / 'private/operations').glob('*.json')
                    if batch.read(path)['kind'].startswith('ceo_tool:')]
                assert wrappers[-1]['recovery']['classification'] == 'pending_read_only'
                if entry['group'] == 'pf':
                    assert role.executor._pf_service is None
            _, state = validate_snapshot(source)
            if interruption == 'symlink':
                assert hashed_symlinks == []
            if blocked:
                with pytest.raises(RuntimeError, match='changed|unknown'):
                    validate_recovery_source(source, state, current_root=team.root)
            else:
                validate_recovery_source(source, state, current_root=team.root)
        finally:
            release.set()
            running.result(timeout=7)
    if interruption == 'pending_write':
        batch.close_team(team, record)
        assert any(item.get('kind') == 'public_http' and item.get('path') == '/call' and
            item.get('http_status') == 403 for item in
            [json.loads(line) for line in role.audit.path.read_text().splitlines()])
        wrapper = next(batch.read(path) for path in (team.root / 'private/operations').glob('*.json')
            if batch.read(path)['kind'] == 'ceo_tool:bash')
        assert wrapper['recovery']['classification'] in ('read_only', 'read_only_interrupted')
        assert wrapper['recovery']['mutation_epoch_after'] == 0
        assert record['checkpoint'] == str(source)
        assert record['recovery_validation']['eligible'] is True
        assert batch.read(directory / 'attempt.json') == record
        blocked = False
    else:
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


@pytest.mark.parametrize('classification,kind,status,contradiction', [
    ('read_only_interrupted', 'ceo_tool:bash', 'running', {}),
    ('pending_read_only', 'ceo_tool:bash', 'completed', {}),
    ('read_only_interrupted', 'api:set_prices', 'running', {}),
    ('unknown_business_write', 'api:set_prices', 'running', {}),
    ('read_only_interrupted', 'ceo_tool:bash', 'running',
        dict(business_operations_before={}, recovery=dict(business_operations_after={'write.json': 'changed'}))),
    ('read_only_interrupted', 'ceo_tool:bash', 'running',
        dict(mutation_epoch_before=0, recovery=dict(mutation_epoch_after=1))),
    ('read_only_interrupted', 'ceo_tool:bash', 'running',
        dict(workspace_sha256_before='a', recovery=dict(workspace_sha256_after='b'))),
    ('read_only_interrupted', 'ceo_tool:bash', 'running',
        dict(recovery=dict(operation_failed=True))),
])
def test_legacy_residual_approval_is_exact_and_preserves_old_attempt(approved_attempt,
        monkeypatch, classification, kind, status, contradiction):
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
    with pytest.raises(ValueError, match='Read-only residual approval|Unknown business write'):
        recovery.recovery_source(approved.manifest_path, approved.entry,
            approved.previous, approved.directory)
    assert (approved.directory / 'attempt.json').read_bytes() == historical
    operation.write_text(operation.read_text() + '\n')
    with pytest.raises(ValueError, match='residual operation changed'):
        recovery.recovery_source(approved.manifest_path, approved.entry, approved.previous, approved.directory)


@pytest.mark.parametrize('classification,proof', [
    ('read_only_interrupted', False), ('read_only_interrupted', True), ('unknown_business_write', False),
])
@pytest.mark.parametrize('managed_git', ['git'], indirect=True)
def test_resume_requires_operation_proof_for_old_bash_write(managed_git,
        monkeypatch, tmp_path, classification, proof):
    team, directory, previous, manifest, entry = managed_git
    snapshot = batch.save_checkpoint(team, directory, previous, 'ready')
    previous.update(status='paused')
    write_json(directory / 'attempt.json', previous)
    _, state = validate_snapshot(snapshot)
    saved = dict(attempt=previous['attempt'], attempt_id=previous['attempt_id'],
        attempt_sha256=file_hash(directory / 'attempt.json'), checkpoint=str(snapshot),
        checkpoint_sha256=file_hash(snapshot / 'checkpoint.json'), day=state['day'],
        roles={name: role['identity'] for name, role in state['roles'].items()})
    manifest_path = tmp_path / 'manifest.json'
    approval_path, revision = bind_approval(monkeypatch, tmp_path / 'recovery', manifest_path,
        {entry['run_id']: saved})
    approved = SimpleNamespace(directory=directory, previous=previous, manifest=manifest, entry=entry,
        manifest_path=manifest_path, snapshot=snapshot, approval_path=approval_path, revision=revision)
    root = approved.directory / 'runtime'
    changed = root / 'roles/ceo/legacy-write.txt'
    changed.write_text('Confirmed workspace write after the checkpoint.\n')
    operation = root / 'private/operations/legacy-write.json'
    receipt = dict(kind='ceo_tool:bash', status='running', day=0)
    if proof:
        checksum = tree_hash(root / 'roles/ceo')
        receipt.update(wrapper_version=1, status='interrupted', business_operations_before={},
            mutation_epoch_before=0, workspace_sha256_before=checksum, recovery=dict(
                classification='read_only_interrupted', client_closed=True, operation_failed=False,
                step_day_timed_out=False, queued_requests=False, pending_requests=[],
                business_operations_after={}, mutation_epoch_after=0, workspace_sha256_after=checksum,
                process_boundary=dict(phase='closed', coverage='subreaper_wait', descendants_closed=True,
                    children=[], unknown=None, exit_code=0)))
    write_json(operation, receipt)
    saved = approved.revision['attempts'][approved.entry['run_id']]
    saved['attempt_id'] = approved.previous['attempt_id']
    resources = approved.approval_path.parent / 'resources.json'
    unit = dict(unit='readonly-test.scope', invocation_id='offline-bound-invocation',
        control_group='/user.slice/readonly-test.scope', scope_empty=True)
    write_json(resources, dict(children_reaped=True, units={approved.entry['run_id']: unit},
        attempts={approved.entry['run_id']: {key: saved[key] for key in ('attempt', 'attempt_id', 'attempt_sha256')}}))
    saved['residual_operations'] = [dict(root=str(root), operation=operation.name,
        sha256=file_hash(operation), classification=classification,
        reason='Verify operation proof separately from scope closure.',
        evidence={str(resources): file_hash(resources)}, resource_evidence=str(resources),
        worker_unit=unit['unit'], worker_invocation_id=unit['invocation_id'])]
    saved['allow_unknown_business_write'] = classification == 'unknown_business_write'
    write_json(approved.approval_path, approved.revision)
    monkeypatch.setenv('CEOBENCH_RECOVERY_APPROVAL_SHA256', file_hash(approved.approval_path))
    original = {path: path.read_bytes() for path in
        (approved.directory / 'attempt.json', operation, changed, approved.snapshot / 'checkpoint.json')}
    try:
        if proof:
            resumed = batch.prepare(approved.manifest_path, approved.entry['run_id'], resume=True,
                simulator_factory=fast_simulator)
            assert resumed['status'] == 'ready' and resumed['attempt'] == 2
            restored = batch.attempts(approved.entry)[-1] / 'runtime/roles/ceo/legacy-write.txt'
            assert not restored.exists()
            assert {path: path.read_bytes() for path in original} == original
            return
        with pytest.raises((RuntimeError, ValueError), match='Read-only residual approval|workspace changed|Unknown business write'):
            batch.prepare(approved.manifest_path, approved.entry['run_id'], resume=True,
                simulator_factory=fast_simulator)
        assert batch.attempts(approved.entry) == [approved.directory]
        destination = tmp_path / 'restore-refused'
        with pytest.raises((RuntimeError, ValueError), match='Read-only residual approval|workspace changed|Unknown business write'):
            batch.MultiAgentRuntime.restore(approved.snapshot, destination,
                client_factory=lambda role: batch.checked_client(role, 'offline-recovery', no_models=True),
                public_dir=approved.manifest['public_dir'], simulator_factory=fast_simulator)
        assert not destination.exists()
        assert {path: path.read_bytes() for path in original} == original
    finally:
        monkeypatch.delenv('CEOBENCH_RECOVERY_APPROVAL', raising=False)
        monkeypatch.delenv('CEOBENCH_RECOVERY_APPROVAL_SHA256', raising=False)


@pytest.mark.parametrize('managed_git', ['git', 'pf'], indirect=True)
def test_direct_bash_write_rolls_back_on_real_resume(managed_git, tmp_path):
    team, directory, previous, manifest, entry = managed_git
    batch.save_checkpoint(team, directory, previous, 'ready')
    original = Path(previous['checkpoint'])
    changed = team.roles['ceo'].workspace / 'direct-bash-write.txt'
    team.roles['ceo'].executor.execute('bash', {'command': 'printf preserved > direct-bash-write.txt'})
    assert changed.read_text() == 'preserved'
    previous['status'] = 'paused'
    write_json(directory / 'attempt.json', previous)
    before = (directory / 'attempt.json').read_bytes()
    resumed = batch.prepare(tmp_path / 'manifest.json', entry['run_id'], resume=True,
        simulator_factory=fast_simulator)
    assert resumed['status'] == 'ready' and resumed['attempt'] == 2
    assert resumed['source_checkpoint'] == str(original)
    restored = batch.attempts(entry)[-1] / 'runtime/roles/ceo/direct-bash-write.txt'
    assert not restored.exists() and changed.read_text() == 'preserved'
    assert (directory / 'attempt.json').read_bytes() == before


@pytest.mark.parametrize('managed_git', ['git'], indirect=True)
@pytest.mark.parametrize('change', ['missing', 'dangling_link'])
def test_real_resume_restores_missing_role_tree_and_preserves_original_records(managed_git, tmp_path, change):
    team, directory, previous, manifest, entry = managed_git
    snapshot = batch.save_checkpoint(team, directory, previous, 'ready')
    previous['status'] = 'paused'
    write_json(directory / 'attempt.json', previous)
    roles = team.root / 'roles'
    preserved = team.root / 'preserved-roles'
    original_roles = tree_hash(roles)
    roles.rename(preserved)
    if change == 'dangling_link':
        roles.symlink_to('missing-role-tree', target_is_directory=True)
    original = {path: path.read_bytes() for path in
        (directory / 'attempt.json', snapshot / 'checkpoint.json')}
    resumed = batch.prepare(tmp_path / 'manifest.json', entry['run_id'], resume=True,
        simulator_factory=fast_simulator)
    assert resumed['status'] == 'ready'
    restored = batch.attempts(entry)[-1] / 'runtime/roles'
    assert tree_hash(restored, exact=True) == tree_hash(snapshot / 'roles', exact=True)
    assert tree_hash(preserved) == original_roles
    assert {path: path.read_bytes() for path in original} == original


@pytest.mark.parametrize('managed_git', ['git'], indirect=True)
@pytest.mark.parametrize('change', ['chmod', 'empty_directory'])
def test_real_resume_rolls_back_bash_metadata_writes(managed_git, tmp_path, change):
    team, directory, previous, manifest, entry = managed_git
    snapshot = batch.save_checkpoint(team, directory, previous, 'ready')
    workspace = team.roles['ceo'].workspace
    command = 'chmod u+x MEMORY.md' if change == 'chmod' else 'mkdir confirmed-empty-directory'
    team.roles['ceo'].executor.execute('bash', {'command': command})
    memory = workspace / 'MEMORY.md'
    mode = memory.stat().st_mode
    assert memory.stat().st_mode & 0o100 if change == 'chmod' else (workspace / 'confirmed-empty-directory').is_dir()
    assert tree_hash(team.root / 'roles') == tree_hash(snapshot / 'roles')
    previous['status'] = 'paused'
    write_json(directory / 'attempt.json', previous)
    original = {path: path.read_bytes() for path in
        (directory / 'attempt.json', snapshot / 'checkpoint.json', memory)}
    resumed = batch.prepare(tmp_path / 'manifest.json', entry['run_id'], resume=True,
        simulator_factory=fast_simulator)
    assert resumed['status'] == 'ready'
    restored = batch.attempts(entry)[-1] / 'runtime/roles/ceo'
    assert (restored / 'MEMORY.md').stat().st_mode == (snapshot / 'roles/ceo/MEMORY.md').stat().st_mode
    assert not (restored / 'confirmed-empty-directory').exists()
    assert memory.stat().st_mode == mode
    assert change == 'chmod' or (workspace / 'confirmed-empty-directory').is_dir()
    assert {path: path.read_bytes() for path in original} == original
