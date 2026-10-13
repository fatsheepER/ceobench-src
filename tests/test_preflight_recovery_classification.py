import subprocess
import json
import os
import shutil
from pathlib import Path

import pytest

from saas_bench import team_batch as batch, team_recovery as recovery
from saas_bench.run_state import file_hash, tree_hash, write_json
from test_preflight_team_recovery_approval import approved_revision, bind_approval


def rebind(monkeypatch, path, record):
    write_json(path, record)
    monkeypatch.setenv('CEOBENCH_RECOVERY_APPROVAL_SHA256', file_hash(path))


def committed_source(tmp_path):
    root = tmp_path / 'git-source'
    root.mkdir()
    subprocess.run(['git', 'init', '--quiet'], cwd=root, check=True)
    target = root / 'src/saas_bench/run_lifecycle.py'
    target.parent.mkdir(parents=True)
    target.write_text('value = 1\n')
    subprocess.run(['git', 'add', 'src'], cwd=root, check=True)
    tree = subprocess.check_output(['git', 'write-tree'], cwd=root, text=True).strip()
    commit = subprocess.check_output(['git', '-c', 'user.name=Offline', '-c', 'user.email=offline@example.invalid',
        'commit-tree', tree, '-m', 'fix: offline proof'], cwd=root, text=True).strip()
    subprocess.run(['git', 'update-ref', '--no-deref', 'HEAD', commit], cwd=root, check=True)
    assert subprocess.check_output(['git', 'for-each-ref', 'refs/heads'], cwd=root) == b''
    return root, target, commit


def harness_revision(tmp_path, repair):
    old, root = tmp_path / 'original', tmp_path / 'revision'
    for directory in ('src', 'scripts'):
        shutil.copytree(batch.ROOT / directory, root / directory,
            ignore=shutil.ignore_patterns('__pycache__', '.env', '.env.*'))
    shutil.copytree(root, old)
    filename, before, after = {
        'model_send': ('src/saas_bench/multi_agent.py',
            'def _check_model_send(self):\n        deadline = time.monotonic() + 3',
            'def _check_model_send(self):\n        deadline = time.monotonic() + 0'),
        'reply': ('src/saas_bench/api_server.py',
            "api._active_requests[id(self.connection)]['outcome_known'] = True\n                api._request_lock.notify_all()",
            "api._active_requests[id(self.connection)]['outcome_known'] = True"),
        'cache': ('src/saas_bench/multi_agent.py', "PYTHONDONTWRITEBYTECODE='1', ", ''),
    }[repair]
    target = old / filename
    text = target.read_text()
    assert text.count(before) == 1
    target.write_text(text.replace(before, after))
    for source in (old, root):
        subprocess.run(['git', 'init', '--quiet'], cwd=source, check=True)
        subprocess.run(['git', 'add', 'src', 'scripts'], cwd=source, check=True)
        tree = subprocess.check_output(['git', 'write-tree'], cwd=source, text=True).strip()
        commit = subprocess.check_output(['git', '-c', 'user.name=Offline', '-c', 'user.email=offline@example.invalid',
            'commit-tree', tree, '-m', 'fix: offline harness repair'], cwd=source, text=True).strip()
        subprocess.run(['git', 'update-ref', '--no-deref', 'HEAD', commit], cwd=source, check=True)
        assert subprocess.check_output(['git', 'for-each-ref', 'refs/heads'], cwd=source) == b''
    return old, root, commit, filename


@pytest.mark.parametrize('repair', ['model_send', 'reply', 'cache'])
def test_real_approval_cli_accepts_harness_send_reply_and_cache_repairs(tmp_path, repair):
    old, root, commit, filename = harness_revision(tmp_path, repair)
    manifest = tmp_path / 'manifest.json'
    write_json(manifest, dict(source_root=str(old), source_sha256=tree_hash(old / 'src')))
    evidence = tmp_path / 'evidence.json'
    write_json(evidence, dict(offline=True, repair=repair))
    declaration = tmp_path / 'declaration.json'
    write_json(declaration, dict(fix_classification='harness_fix', fix_commit=commit,
        affected_files=[filename], reason='Repair runtime confirmation or cache hygiene without changing inputs',
        evidence=[str(evidence)]))
    original = manifest.read_bytes()
    result = subprocess.run([os.sys.executable, str(root / 'scripts/team_recovery.py'),
        '--manifest', str(manifest), '--output', str(tmp_path / 'approval'), '--declaration', str(declaration),
        '--source-root', str(root), '--approve'], cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root / 'src'), PYTHONDONTWRITEBYTECODE='1',
            PYTHON_DOTENV_DISABLED='1'), capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    saved = batch.read(receipt['path'])
    assert saved['fix_classification'] == 'harness_fix' and saved['fix_commit'] == commit
    assert saved['affected_files'] == [filename] and saved['status'] == 'approved'
    assert manifest.read_bytes() == original and not (tmp_path / 'HOLD').exists()


def test_approval_requires_declared_repair_classification(approved_revision, monkeypatch):
    path, record, manifest = approved_revision
    record.pop('fix_classification')
    rebind(monkeypatch, path, record)
    with pytest.raises(ValueError, match='classification'):
        recovery.approval(manifest)


def test_design_change_cannot_resume_same_freeze(approved_revision, monkeypatch):
    path, record, manifest = approved_revision
    record['fix_classification'] = 'design_change'
    rebind(monkeypatch, path, record)
    with pytest.raises(ValueError, match='design_change'):
        recovery.approval(manifest)


@pytest.mark.parametrize('filename', ['src/saas_bench/team_prompts.py', 'src/saas_bench/role_policy.py',
                                    'src/saas_bench/config.py', 'src/saas_bench/pf_cli.py'])
def test_harness_claim_rejects_prompt_policy_config_and_pf_interface(approved_revision, monkeypatch, filename):
    path, _, manifest = approved_revision
    old = batch.read(manifest)
    target = Path(old['source_root']) / filename
    target.write_text(target.read_text() + '\n')
    old['source_sha256'] = tree_hash(Path(old['source_root']) / 'src')
    write_json(manifest, old)
    path, _ = bind_approval(monkeypatch, path.parent, manifest)
    with pytest.raises(ValueError, match='prompts|PF help|configuration'):
        recovery.approval(manifest)


def test_harness_claim_rejects_public_help_changes(approved_revision, monkeypatch):
    path, _, manifest = approved_revision
    old = batch.read(manifest)
    target = Path(old['source_root']) / 'public/docs/tools-reference.md'
    target.write_text(target.read_text() + '\n')
    bind_approval(monkeypatch, path.parent, manifest)
    with pytest.raises(ValueError, match='PF help'):
        recovery.approval(manifest)


def test_harness_claim_rejects_frozen_parameters_in_mixed_module(approved_revision, monkeypatch):
    path, _, manifest = approved_revision
    old = batch.read(manifest)
    target = Path(old['source_root']) / 'src/saas_bench/team_batch.py'
    target.write_text(target.read_text().replace("temperature=1.0, max_tokens=16384", "temperature=0.5, max_tokens=16384", 1))
    old['source_sha256'] = tree_hash(Path(old['source_root']) / 'src')
    write_json(manifest, old)
    bind_approval(monkeypatch, path.parent, manifest)
    with pytest.raises(ValueError, match='frozen settings'):
        recovery.approval(manifest)


@pytest.mark.parametrize('filename,before,after', [
    ('src/saas_bench/agents/bash_agent/tools.py', 'Execute a bash command', 'Execute a different command'),
    ('src/saas_bench/multi_agent.py', 'CEO must advance through the weekly API', 'Changed message'),
])
def test_mixed_modules_cannot_change_model_descriptions(approved_revision, monkeypatch, filename, before, after):
    path, _, manifest = approved_revision
    old = batch.read(manifest)
    target = Path(old['source_root']) / filename
    content = target.read_text()
    if filename.endswith('multi_agent.py'):
        content = content.replace("runtime.agent.act('', 0, False, {'day': day})", "runtime.agent.act('new instruction', 0, False, {'day': day})")
    else:
        assert before in content
        content = content.replace(before, after, 1)
    target.write_text(content)
    old['source_sha256'] = tree_hash(Path(old['source_root']) / 'src')
    write_json(manifest, old)
    bind_approval(monkeypatch, path.parent, manifest)
    with pytest.raises(ValueError, match='prompts|model inputs'):
        recovery.approval(manifest)


def test_declared_affected_files_and_reason_are_required(approved_revision, monkeypatch):
    path, record, manifest = approved_revision
    record['affected_files'] = []
    rebind(monkeypatch, path, record)
    with pytest.raises(ValueError, match='affected files'):
        recovery.approval(manifest)
    record['affected_files'] = record['changed_files']
    record['reason'] = ''
    rebind(monkeypatch, path, record)
    with pytest.raises(ValueError, match='reason'):
        recovery.approval(manifest)


def test_input_source_mutation_cannot_be_declared_harness_fix(approved_revision, monkeypatch):
    path, _, manifest = approved_revision
    old = batch.read(manifest)
    target = Path(old['source_root']) / 'src/saas_bench/multi_agent.py'
    text = target.read_text()
    assert 'dashboard = self.server.public_dashboard()' in text
    target.write_text(text.replace('dashboard = self.server.public_dashboard()', "dashboard = 'Changed role instructions'"))
    old['source_sha256'] = tree_hash(Path(old['source_root']) / 'src')
    write_json(manifest, old)
    bind_approval(monkeypatch, path.parent, manifest)
    with pytest.raises(ValueError, match='prompts'):
        recovery.approval(manifest)


def test_historical_runtime_downgrade_cannot_be_declared_harness_fix(approved_revision, monkeypatch):
    path, _, manifest = approved_revision
    old = batch.read(manifest)
    target = Path(old['source_root']) / 'src/saas_bench/multi_agent.py'
    target.write_bytes(subprocess.check_output(['git', 'show', '95bcf95:src/saas_bench/multi_agent.py'], cwd=batch.ROOT))
    old['source_sha256'] = tree_hash(Path(old['source_root']) / 'src')
    write_json(manifest, old)
    bind_approval(monkeypatch, path.parent, manifest)
    with pytest.raises(ValueError, match='frozen settings|scheduling design'):
        recovery.approval(manifest)


def test_runtime_journal_fix_keeps_frozen_model_inputs(approved_revision, monkeypatch):
    path, _, manifest = approved_revision
    old = batch.read(manifest)
    target = Path(old['source_root']) / 'src/saas_bench/multi_agent.py'
    current = target.read_text()
    journaled = """                    operation = self._begin_operation('ceo_tool:' + action.tool)
                    try:
                        result = runtime.executor.execute(action.tool, action.arguments,
                            model_request_event=runtime.usage.last_request_event,
                            model_context_id=runtime.usage.context_id)
                        runtime.agent.record_tool_result(result)
                    finally:
                        self._finish_operation(operation, runtime.executor)
"""
    legacy = """                    result = runtime.executor.execute(action.tool, action.arguments,
                        model_request_event=runtime.usage.last_request_event,
                        model_context_id=runtime.usage.context_id)
                    runtime.agent.record_tool_result(result)
"""
    assert current.count(journaled) == 1
    target.write_text(current.replace(journaled, legacy))
    old['source_sha256'] = tree_hash(Path(old['source_root']) / 'src')
    write_json(manifest, old)
    bind_approval(monkeypatch, path.parent, manifest)
    assert recovery.approval(manifest)['fix_classification'] == 'harness_fix'


def test_fix_commit_verifies_real_git_contents_without_model_or_credentials(tmp_path):
    root, target, commit = committed_source(tmp_path)
    actual = recovery.source_files(root)
    recovery.verify_fix_commit(root, commit, set(actual), actual)
    with pytest.raises(ValueError, match='HEAD'):
        recovery.verify_fix_commit(root, '0' * 40, set(actual), actual)
    target.write_text('value = 2\n')
    with pytest.raises(ValueError, match='does not contain'):
        recovery.verify_fix_commit(root, commit, set(), recovery.source_files(root))


def test_residual_override_requires_classification_evidence_and_closed_scope(tmp_path):
    resource = tmp_path / 'resources.json'
    binding = dict(attempt=1, attempt_id='original', attempt_sha256='1' * 64)
    write_json(resource, dict(children_reaped=True, attempts={'run': binding},
        units={'run': dict(scope_empty=True, unit='worker.scope', invocation_id='invocation', control_group='/worker')}))
    item = dict(root=str(tmp_path), operation='operation.json', sha256='0' * 64,
        classification='unknown_business_write', reason='Explicitly accept interrupted write with retained evidence',
        evidence={str(resource): file_hash(resource)}, resource_evidence=str(resource),
        worker_unit='worker.scope', worker_invocation_id='invocation')
    saved = dict(binding, residual_operations=[item])
    record = dict(attempts={'run': saved})
    with pytest.raises(ValueError, match='explicit approval'):
        recovery.validate_residual_declarations(record)
    saved['allow_unknown_business_write'] = True
    recovery.validate_residual_declarations(record)
    item['worker_invocation_id'] = 'other-attempt'
    with pytest.raises(ValueError, match='original attempt'):
        recovery.validate_residual_declarations(record)
    item['worker_invocation_id'] = 'invocation'
    write_json(resource, dict(children_reaped=False, units={'run': dict(scope_empty=False)}))
    item['evidence'][str(resource)] = file_hash(resource)
    with pytest.raises(ValueError, match='scope closure'):
        recovery.validate_residual_declarations(record)


@pytest.mark.parametrize('classification', ['harness_fix', 'design_change'])
def test_approval_cli_preserves_manifest_and_records_explicit_declaration(tmp_path, classification):
    root, _, commit = committed_source(tmp_path)
    manifest = tmp_path / 'manifest.json'
    write_json(manifest, dict(source_root=str(root), source_sha256=tree_hash(root / 'src')))
    evidence = tmp_path / 'evidence.json'
    write_json(evidence, dict(offline=True, business_changes=False))
    declaration = tmp_path / 'declaration.json'
    write_json(declaration, dict(fix_classification=classification, fix_commit=commit, affected_files=[],
        reason='Identical committed execution revision with offline evidence', evidence=[str(evidence)]))
    before = manifest.read_bytes()
    output = tmp_path / 'approval'
    result = subprocess.run([os.sys.executable, str(batch.ROOT / 'scripts/team_recovery.py'),
        '--manifest', str(manifest), '--output', str(output), '--declaration', str(declaration),
        '--source-root', str(root), '--approve'],
        env=dict(os.environ, PYTHONPATH=str(batch.ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True, text=True, timeout=15)
    assert manifest.read_bytes() == before
    if classification == 'design_change':
        assert result.returncode != 0 and 'design_change requires a new frozen batch' in result.stderr
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        receipt = json.loads(result.stdout)
        record = batch.read(receipt['path'])
        assert receipt['status'] == record['status'] == 'approved'
        assert record['fix_classification'] == 'harness_fix' and record['fix_commit'] == commit
        assert record['affected_files'] == [] and record['evidence'] == {str(evidence): file_hash(evidence)}
        assert receipt['sha256'] == file_hash(receipt['path'])
        assert not (tmp_path / 'HOLD').exists()


@pytest.mark.parametrize('filename,before,after', [
    ('src/saas_bench/multi_agent.py', "PYTHONDONTWRITEBYTECODE='1'", "PYTHONDONTWRITEBYTECODE='0'"),
    ('src/saas_bench/multi_agent.py', "PYTHONHASHSEED='0'", "PYTHONHASHSEED='1'"),
    ('src/saas_bench/multi_agent.py', 'def _check_model_send(self):', 'def _check_model_send(self, bypass=False):'),
    ('src/saas_bench/api_server.py', 'return self._last_dashboard',
        "return 'Changed model instructions\\n' + self._last_dashboard"),
    ('src/saas_bench/api_server.py', 'response = json.dumps(data, default=str).encode()',
        "response = json.dumps({'changed': data}, default=str).encode()"),
    ('src/saas_bench/agents/bash_agent/tools.py', "stream.write(args['content'].encode())",
        "stream.write((args['content'] + 'changed').encode())"),
    ('scripts/team_controller.py', "for group in ('git', 'pf')} if independent else {}",
        "for group in ('pf', 'git')} if independent else {}"),
    ('src/saas_bench/team_batch.py', "return policy['mode']", "return 'paired'"),
    ('src/saas_bench/execution_capture.py', "return rel.split('/')[0] in HIDDEN_WORKSPACE_DIRS",
        "return True or rel.split('/')[0] in HIDDEN_WORKSPACE_DIRS"),
])
def test_real_approval_cli_rejects_mixed_module_design_changes(tmp_path, filename, before, after):
    old, root = tmp_path / 'original', tmp_path / 'revision'
    for directory in ('src', 'scripts'):
        shutil.copytree(batch.ROOT / directory, old / directory,
            ignore=shutil.ignore_patterns('__pycache__', '.env', '.env.*'))
    shutil.copytree(old, root)
    target = root / filename
    text = target.read_text()
    assert before in text
    target.write_text(text.replace(before, after, 1))
    subprocess.run(['git', 'init', '--quiet'], cwd=root, check=True)
    subprocess.run(['git', 'add', 'src', 'scripts'], cwd=root, check=True)
    tree = subprocess.check_output(['git', 'write-tree'], cwd=root, text=True).strip()
    commit = subprocess.check_output(['git', '-c', 'user.name=Offline', '-c', 'user.email=offline@example.invalid',
        'commit-tree', tree, '-m', 'fix: offline design boundary proof'], cwd=root, text=True).strip()
    subprocess.run(['git', 'update-ref', '--no-deref', 'HEAD', commit], cwd=root, check=True)
    assert subprocess.check_output(['git', 'for-each-ref', 'refs/heads'], cwd=root) == b''
    recovery.verify_fix_commit(root, commit, {filename}, recovery.source_files(root))
    manifest = tmp_path / 'manifest.json'
    write_json(manifest, dict(source_root=str(old), source_sha256=tree_hash(old / 'src')))
    evidence = tmp_path / 'evidence.json'
    write_json(evidence, dict(offline=True))
    declaration = tmp_path / 'declaration.json'
    write_json(declaration, dict(fix_classification='harness_fix', fix_commit=commit,
        affected_files=[filename], reason='Verify a mixed module design boundary', evidence=[str(evidence)]))
    original = manifest.read_bytes()
    output = tmp_path / 'approval'
    result = subprocess.run([os.sys.executable, str(root / 'scripts/team_recovery.py'),
        '--manifest', str(manifest), '--output', str(output), '--declaration', str(declaration),
        '--source-root', str(root), '--approve'], cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root / 'src'), PYTHONDONTWRITEBYTECODE='1',
            PYTHON_DOTENV_DISABLED='1'), capture_output=True, text=True, timeout=45)
    assert result.returncode != 0, result.stdout
    assert 'design' in result.stderr, result.stderr
    assert manifest.read_bytes() == original and not output.exists()


@pytest.mark.parametrize('group', ['git', 'pf'])
def test_real_harness_approval_resumes_d7_checkpoint_through_model_usage(tmp_path, group):
    from saas_bench.run_state import build_manifest
    old, root, commit, filename = harness_revision(tmp_path, 'model_send')
    for source in (old, root):
        shutil.copytree(batch.ROOT / 'public', source / 'public')
        shutil.copytree(batch.ROOT / 'public_sources', source / 'public_sources')
        write_json(source / 'public/build.json', build_manifest(source, source / 'public'))
    bootstrap = tmp_path / 'offline-recovery.py'
    bootstrap.write_text(r'''
import json, socket, sys
from pathlib import Path
import httpx
from saas_bench import team_batch as batch
from saas_bench.run_state import file_hash, write_json
sys.path.insert(0, str(Path(sys.argv[4]) / 'tests'))
from test_preflight_team_batch import PRICES
from test_preflight_team_checkpoint import fast_simulator
from test_preflight_team_flow import ADVANCE, calls, completion, final
from saas_bench.agents.bash_agent import run_test
def no_env(*args, **kwargs):
    raise AssertionError('Offline recovery cannot read credentials')
run_test.load_env_file = no_env
connect = socket.socket.connect
def local_only(sock, address):
    if isinstance(address, tuple) and address[0] not in ('127.0.0.1', '::1', 'localhost'):
        raise AssertionError('Offline recovery cannot access external network')
    return connect(sock, address)
socket.socket.connect = local_only
mode, directory, group = sys.argv[1:4]
directory = Path(directory)
manifest_path = directory / 'batch/manifest.json'
if mode == 'pause':
    batch.seal_d0(42, directory / 'd0', PRICES, price_source=PRICES['source'])
    manifest = batch.freeze_batch(directory / 'batch', {42: directory / 'd0'}, PRICES,
        price_source=PRICES['source'], category='engineering')
else:
    manifest = batch.read(manifest_path)
entry = next(row for row in manifest['runs'] if row['group'] == group)
wire = []
def client(role):
    def respond(request):
        body = json.loads(request.content)
        wire.append(dict(role=role, request=body))
        message = calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final('Use public data.')
        return completion(message, body.get('stream', False))
    return batch.checked_client(role, 'offline', transport=httpx.MockTransport(respond))
save = batch.save_checkpoint
def save_and_pause(team, attempt, record, boundary):
    snapshot = save(team, attempt, record, boundary)
    if mode == 'pause' and boundary == 'week_boundary' and team.server.tools.current_day == 7:
        (manifest_path.parent / 'HOLD').touch()
    return snapshot
batch.save_checkpoint = save_and_pause
def offline_simulator(conn, config, rng):
    fast = fast_simulator(conn, config, rng)
    original = batch.simulator(conn, config, rng)
    fast.customer_simulator = original.customer_simulator
    fast.shock_manager = original.shock_manager
    return fast
record = batch.prepare(manifest_path, entry['run_id'], resume=mode == 'resume', live=True,
    client_factory=client, simulator_factory=offline_simulator)
write_json(directory / (mode + '-receipt.json'), dict(record=record, wire=wire))
''')
    directory = tmp_path / 'offline'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHON_DOTENV_DISABLED='1',
        PYTHONHASHSEED='0', OPENCODE_API_KEY='OFFLINE-ONLY')
    def run(source, mode, **extra):
        result = subprocess.run([os.sys.executable, str(bootstrap), mode, str(directory), group, str(batch.ROOT)],
            cwd=source, env=dict(env, PYTHONPATH=str(source / 'src'), **extra),
            capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stderr
        return batch.read(directory / (mode + '-receipt.json'))
    paused = run(old, 'pause')['record']
    assert paused['status'] == 'paused' and paused['day'] == 7
    manifest_path = directory / 'batch/manifest.json'
    manifest = batch.read(manifest_path)
    entry = next(row for row in manifest['runs'] if row['group'] == group)
    first = batch.attempts(entry)[0]
    snapshot = Path(paused['checkpoint'])
    immutable = {path: path.read_bytes() for path in (manifest_path, first / 'attempt.json', snapshot / 'checkpoint.json')}
    snapshot_hash = tree_hash(snapshot, exact=True)
    evidence = directory / 'pause-receipt.json'
    declaration = tmp_path / 'declaration.json'
    write_json(declaration, dict(fix_classification='harness_fix', fix_commit=commit,
        affected_files=[filename], reason='Fix confirmation wait without changing the experiment',
        evidence=[str(evidence)], attempts={entry['run_id']: {}}))
    design = tmp_path / 'design-change.json'
    write_json(design, dict(batch.read(declaration), fix_classification='design_change'))
    refused = subprocess.run([os.sys.executable, str(root / 'scripts/team_recovery.py'),
        '--manifest', str(manifest_path), '--output', str(tmp_path / 'design-approval'),
        '--declaration', str(design), '--source-root', str(root), '--approve'], cwd=root,
        env=dict(env, PYTHONPATH=str(root / 'src')), capture_output=True, text=True, timeout=45)
    assert refused.returncode != 0 and 'design_change requires a new frozen batch' in refused.stderr
    assert not (tmp_path / 'design-approval').exists() and batch.attempts(entry) == [first]
    approved = subprocess.run([os.sys.executable, str(root / 'scripts/team_recovery.py'),
        '--manifest', str(manifest_path), '--output', str(tmp_path / 'approval'),
        '--declaration', str(declaration), '--source-root', str(root), '--approve'], cwd=root,
        env=dict(env, PYTHONPATH=str(root / 'src')), capture_output=True, text=True, timeout=45)
    assert approved.returncode == 0, approved.stderr
    receipt = json.loads(approved.stdout)
    resumed = run(root, 'resume', CEOBENCH_RECOVERY_APPROVAL=receipt['path'],
        CEOBENCH_RECOVERY_APPROVAL_SHA256=receipt['sha256'])
    record = resumed['record']
    assert record['status'] == 'finished' and record['day'] == 28 and record['attempt'] == 2
    assert record['source_checkpoint'] == str(snapshot) and resumed['wire']
    later = batch.attempts(entry)[-1]
    assert batch.read(later / 'runtime/recovery.json')['source_revision_sha256'] == receipt['sha256']
    rows = [json.loads(line) for line in (later / 'runtime/private/ceo/model-usage.jsonl').read_text().splitlines()]
    assert any(row['event'] == 'http_request' for row in rows)
    assert any(row['event'] == 'response' and row['usage']['input_tokens'] == 20 for row in rows)
    assert all(path.read_bytes() == original for path, original in immutable.items())
    assert tree_hash(snapshot, exact=True) == snapshot_hash
    assert (manifest_path.parent / 'HOLD').exists()

