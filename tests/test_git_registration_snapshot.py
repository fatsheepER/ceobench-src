"""Registration-time Git bytes through real role tools and checkpoint restore."""
import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
from saas_bench.registration_schema import cite_text
from saas_bench.text_registry import TextRegistry
from test_text_registry import workspace, call, declaration, git
from test_preflight_team_flow import make_team, final, calls, ADVANCE
from test_preflight_team_evidence import declare, write
from test_preflight_team_checkpoint import make_team as checkpoint_team, clients, fast_simulator, dispose


def sample(name, value):
    if root := os.environ.get('CEOBENCH_R226_SAMPLES'):
        path = Path(root) / (name + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2))


def tree_bytes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and not p.is_symlink()}


def saved(registry, record='r1'):
    return registry._load()['records'][record][-1]['references'][0]


def read_git(workspace, evidence):
    return subprocess.check_output(['git', '-C', str(workspace), 'show', evidence['commit'] + ':' + evidence['path']])


def test_bare_snapshot_preserves_exact_bytes_index_head_and_gc(workspace):
    registry = TextRegistry(workspace, 'git')
    raw = b'first\r\n\x00\xfflast\n'
    path = 'nested/a b\tfile.bin'
    (workspace / 'nested').mkdir()
    (workspace / path).write_bytes(raw)
    (workspace / 'staged.txt').write_text('keep staged')
    git(workspace, 'add', 'staged.txt')
    head, index = git(workspace, 'rev-parse', 'HEAD'), (workspace / '.git/index').read_bytes()
    receipt = registry.execute('create', declaration({'path': path}))
    evidence = saved(registry)['evidence']
    assert read_git(workspace, evidence) == raw
    assert evidence['commit'] in receipt
    assert git(workspace, 'ls-tree', '-r', '--name-only', evidence['commit']) == '"nested/a b\\tfile.bin"'
    for value in (b'second', b'third'):
        (workspace / path).write_bytes(value)
        assert read_git(workspace, evidence) == raw
    assert git(workspace, 'rev-parse', 'HEAD') == head
    assert (workspace / '.git/index').read_bytes() == index
    git(workspace, 'gc', '--prune=now')
    assert read_git(workspace, evidence) == raw
    call(registry, 'revise', record='r1', text='Still based on first bytes', reason='clarify')
    assert saved(registry)['evidence'] == evidence
    sample('own-snapshot', dict(receipt=receipt, evidence=evidence, raw_hex=raw.hex()))


def test_explicit_commit_and_week_label_read_requested_history(workspace):
    registry = TextRegistry(workspace, 'git')
    initial = git(workspace, 'rev-parse', 'HEAD')
    (workspace / 'evidence.json').write_text('{"n":99}')
    call(registry, 'create', **declaration({'path': 'evidence.json', 'commit': initial[:7]}))
    assert read_git(workspace, saved(registry)['evidence']) == b'{"n":7}'
    git(workspace, 'add', 'evidence.json')
    git(workspace, 'commit', '-qm', 'Week 1 (day 7) [week-1]')
    call(registry, 'create', **declaration({'path': 'evidence.json@week-1'}))
    from saas_bench.registration_evidence import git_reference
    evidence, full = git_reference(workspace, saved(registry, 'r2')['evidence'])
    assert read_git(workspace, dict(evidence, commit=full)) == b'{"n":99}'


@pytest.mark.parametrize('path', ['missing.txt', 'sessions/private.txt', '.git/config', '../secret.txt', 'alias.txt', 'linked/file.txt'])
def test_invalid_snapshot_returns_failure_without_publishing(workspace, path):
    (workspace / 'sessions').mkdir()
    (workspace / 'sessions/private.txt').write_text('private')
    (workspace / 'alias.txt').symlink_to(workspace / 'evidence.json')
    (workspace / 'linked').symlink_to(workspace / 'sessions', target_is_directory=True)
    registry = TextRegistry(workspace, 'git')
    before = git(workspace, 'for-each-ref')
    with pytest.raises((ValueError, OSError)):
        call(registry, 'create', **declaration(references=[
            {'cite': 'evidence.json'}, {'cite': path}]))
    assert not registry.path.exists()
    assert git(workspace, 'for-each-ref') == before
    call(registry, 'create', **declaration({'unknown': 'Original output was not saved'}))
    assert saved(registry)['evidence'] == {'unknown': 'Original output was not saved'}


def test_peer_snapshots_preserve_source_identity_and_do_not_modify_peer():
    with make_team('git', lambda *args: final('received'), 'r226-peer') as team:
        ceo = team.roles['ceo']
        write(team, 'ceo', 'ceo current')
        receipts, references = [], []
        for source in ('growth', 'ops_finance'):
            peer = team.roles[source]
            write(team, source, source + ' first')
            before = tree_bytes(peer.workspace)
            index = (ceo.workspace / '.git/index').read_bytes()
            head = team._git(ceo, 'rev-parse', 'HEAD')
            receipts.append(declare(team, 'ceo', source + ' basis', [source + ':same.txt']))
            ref = saved(ceo.registry, 'r' + str(len(receipts)))
            references.append(ref)
            evidence = ref['evidence']
            assert tree_bytes(peer.workspace) == before
            assert (ceo.workspace / '.git/index').read_bytes() == index
            assert team._git(ceo, 'rev-parse', 'HEAD') == head
            assert ref['snapshot_source'] == dict(owner=source, path='same.txt')
            assert evidence['owner'] == 'ceo'
            write(team, source, source + ' later')
            for reader in team.roles.values():
                command = shlex.join(['git', '-C', str(ceo.workspace), 'show', evidence['commit'] + ':' + evidence['path']])
                output = reader.executor.execute('bash', {'command': command})
                assert output == source + ' first'
            assert source + ':same.txt' in receipts[-1]
        assert references[0]['evidence']['commit'] != references[1]['evidence']['commit']
        declare(team, 'growth', 'reuse saved peer snapshot', [cite_text(references[0]['evidence'])])
        assert not list((team.root / 'private').rglob('*.sqlite'))
        sample('peer-snapshots', dict(receipts=receipts, references=references))


def test_git_snapshot_checkpoint_restores_references_and_discards_later_snapshots(tmp_path):
    original = checkpoint_team(tmp_path / 'original', 'git')
    restored = None
    try:
        write(original, 'growth', 'checkpoint first')
        declare(original, 'ceo', 'checkpoint evidence', ['growth:same.txt'])
        ref = saved(original.roles['ceo'].registry)
        snapshot = original.checkpoint()
        write(original, 'growth', 'discarded later')
        declare(original, 'ceo', 'discarded evidence', ['growth:same.txt'])
        later = saved(original.roles['ceo'].registry, 'r2')['evidence']
        restored = MultiAgentRuntime.restore(snapshot, tmp_path / 'restored',
            client_factory=clients(), simulator_factory=fast_simulator)
        assert saved(restored.roles['ceo'].registry) == ref
        assert 'r2' not in restored.roles['ceo'].registry._load()['records']
        for role in restored.roles.values():
            command = shlex.join(['git', '-C', str(restored.roles['ceo'].workspace), 'show',
                                 ref['evidence']['commit'] + ':' + ref['evidence']['path']])
            assert role.executor.execute('bash', {'command': command}) == 'checkpoint first'
        assert restored.roles['ceo'].executor.run_private(['git', '-C', str(restored.roles['ceo'].workspace),
            'cat-file', '-e', later['commit']], capture_output=True).returncode != 0
        sample('checkpoint-snapshot', dict(retained=ref, discarded=later))
    finally:
        if restored:
            dispose(restored)
        dispose(original)


def test_model_receives_snapshot_and_reads_registered_bytes_after_overwrite():
    turns = {}
    observed = {}

    def respond(team, role, body):
        turns[role] = turns.get(role, 0) + 1
        if role == 'ceo':
            return calls(('bash', {'command': ADVANCE}))
        if role == 'ops_finance':
            return final('No further action.')
        if turns[role] == 1:
            return calls(('write_file', {'path': 'basis.txt', 'content': 'registered original'}),
                         ('text_create', declaration({'path': 'basis.txt'})),
                         ('write_file', {'path': 'basis.txt', 'content': 'newer working file'}))
        if turns[role] == 2:
            receipt = next(m['content'] for m in body['messages'] if m.get('name') == 'text_create')
            ref = saved(team.roles[role].registry)
            assert cite_text(ref['evidence']) in receipt
            observed['receipt_input'] = body
            evidence = ref['evidence']
            return calls(('bash', {'command': shlex.join(['git', 'show', evidence['commit'] + ':' + evidence['path']])}))
        result = [m['content'] for m in body['messages'] if m.get('name') == 'bash'][-1]
        assert result == 'registered original'
        assert (team.roles[role].workspace / 'basis.txt').read_text() == 'newer working file'
        observed['read_input'] = body
        return final('The saved reference still contains registered original.')

    with make_team('git', respond, 'r226-model') as team:
        assert team.run(stop_after_day=7).reason == 'observation_end', team.failure
        assert observed.keys() == {'receipt_input', 'read_input'}
        sample('model-snapshot-flow', observed)


@pytest.mark.parametrize('command', ['hash-object', 'mktree', 'commit-tree', 'update-ref'])
def test_snapshot_git_failure_returns_no_success_and_next_call_continues(workspace, command):
    registry = TextRegistry(workspace, 'git')
    executor = BashAgentToolExecutor(workspace, text_registry=registry)
    run = registry.git_run

    def fail(argv, **kwargs):
        if command in argv:
            return subprocess.CompletedProcess(argv, 1, b'', b'injected Git failure')
        return run(argv, **kwargs)

    registry.git_run = fail
    error = executor.execute('text_create', declaration({'path': 'evidence.json'}))
    assert error.startswith('Error:') and 'snapshot' in error
    assert not registry.path.exists()
    registry.git_run = run
    result = executor.execute('text_create', declaration({'path': 'evidence.json'}))
    assert result.startswith('Registered r1.1')
    assert read_git(workspace, saved(registry)['evidence']) == b'{"n":7}'


def test_file_change_during_read_is_rejected_then_can_be_retried(workspace, monkeypatch):
    from saas_bench import workspace_io
    real_open = workspace_io.open_file
    target = workspace / 'evidence.json'

    @contextmanager
    def changed(root, path, mode='rb'):
        with real_open(root, path, mode) as stream:
            class Reader:
                def fileno(self):
                    return stream.fileno()

                def read(self):
                    data = stream.read()
                    target.write_bytes(b'concurrent writer changed the source')
                    return data
            yield Reader()

    registry = TextRegistry(workspace, 'git')
    with monkeypatch.context() as patch:
        patch.setattr(workspace_io, 'open_file', changed)
        with pytest.raises(ValueError, match='changed'):
            call(registry, 'create', **declaration({'path': 'evidence.json'}))
    assert not registry.path.exists()
    call(registry, 'create', **declaration({'path': 'evidence.json'}))
    assert read_git(workspace, saved(registry)['evidence']) == b'concurrent writer changed the source'


@pytest.mark.parametrize('raw', [b'', b'unchanged\r\nline\r\n'])
def test_snapshot_ignores_git_clean_filters_and_keeps_empty_files(workspace, raw):
    (workspace / '.gitattributes').write_text('*.json filter=mark text eol=lf\n')
    git(workspace, 'config', 'filter.mark.clean', 'touch filter-ran; cat')
    (workspace / 'evidence.json').write_bytes(raw)
    registry = TextRegistry(workspace, 'git')
    call(registry, 'create', **declaration({'path': 'evidence.json'}))
    assert read_git(workspace, saved(registry)['evidence']) == raw
    assert not (workspace / 'filter-ran').exists()


def test_snapshot_does_not_run_reference_transaction_hooks(workspace):
    hook = workspace / '.git/hooks/reference-transaction'
    hook.write_text('#!/bin/sh\nprintf hook > hook-ran\ngit add hook-ran\n')
    hook.chmod(0o755)
    index = (workspace / '.git/index').read_bytes()
    registry = TextRegistry(workspace, 'git')
    executor = BashAgentToolExecutor(workspace, text_registry=registry)
    result = executor.execute('text_create', declaration({'path': 'evidence.json'}))
    assert result.startswith('Registered r1.1'), result
    assert not (workspace / 'hook-ran').exists()
    assert (workspace / '.git/index').read_bytes() == index
    assert read_git(workspace, saved(registry)['evidence']) == b'{"n":7}'
