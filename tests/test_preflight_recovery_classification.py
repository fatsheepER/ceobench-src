import subprocess
import json
import os
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

