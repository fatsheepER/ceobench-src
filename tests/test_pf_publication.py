import json
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shlex
import threading

import pytest

from saas_bench.workspace_publication import WorkspacePublication, PublicationHold
from test_shared_workspace import shared_team
from test_preflight_team_flow import calls, final


@pytest.fixture
def publication(tmp_path):
    public = tmp_path / 'public'
    public.mkdir()
    (public / 'plan').write_text('one\ntwo\nthree\nfour\nfive\n')
    return WorkspacePublication(public, tmp_path / 'private')


def close(workspace, execution, **options):
    return workspace.finish(execution, closure={'boundary_closed': True}, **options)


def test_stale_read_refresh_and_explicit_fixed_version_resolution(publication):
    a = publication.begin('a')
    close(publication, a)
    b = publication.begin('b')
    (b.root / 'plan').write_text('B\n')
    close(publication, b)
    a = publication.begin('a')
    assert (a.root / 'plan').read_text() == 'B\n'
    (a.root / 'plan').write_text('A\n')
    pending = close(publication, a)
    assert pending['status'] == 'needs_confirmation'
    assert (publication.public / 'plan').read_text() == 'B\n'
    done = publication.resolve('a', a.id, against=pending['head'], revision=pending['candidate'])
    assert done['status'] == 'published'
    assert (publication.public / 'plan').read_text() == 'A\n'


def test_call_in_progress_merge_and_author_base_stays_separate(publication):
    a, b = publication.begin('a'), publication.begin('b')
    (a.root / 'plan').write_text('A\ntwo\nthree\nfour\nfive\n')
    (b.root / 'plan').write_text('one\ntwo\nthree\nfour\nB\n')
    close(publication, b)
    assert (a.root / 'plan').read_text().endswith('five\n')
    result = close(publication, a)
    assert result['status'] == 'merged'
    assert (publication.public / 'plan').read_text() == 'A\ntwo\nthree\nfour\nB\n'
    assert publication.tree(result['author_after']) != publication.tree(result['head'])
    next_call = publication.begin('a')
    assert (next_call.root / 'plan').read_text().endswith('B\n')
    assert close(publication, next_call)['status'] == 'unchanged'
    edit = publication.begin('a')
    (edit.root / 'plan').write_text('another edit\n')
    assert close(publication, edit)['status'] == 'needs_confirmation'


@pytest.mark.parametrize('operation', ['overlap', 'append', 'delete', 'binary', 'type'])
def test_true_concurrent_conflicts_keep_real_candidates(publication, operation):
    a, b = publication.begin('a'), publication.begin('b')
    (b.root / 'plan').write_text('B\n')
    if operation == 'delete':
        (a.root / 'plan').unlink()
    elif operation == 'type':
        (a.root / 'plan').unlink()
        (a.root / 'plan').mkdir()
    elif operation == 'binary':
        (a.root / 'plan').write_bytes(b'\0A')
    elif operation == 'append':
        with (a.root / 'plan').open('a') as stream:
            stream.write('A\n')
        with (b.root / 'plan').open('w') as stream:
            stream.write('one\ntwo\nthree\nfour\nfive\nB\n')
    else:
        (a.root / 'plan').write_text('A\n')
    close(publication, b)
    result = close(publication, a)
    assert result['status'] == 'conflict'
    assert result['conflicts'] == ['plan']
    assert publication.tree(result['author_after']) == publication.scan(a.root)


def test_atomic_group_old_draft_and_unrelated_call(publication):
    a, b = publication.begin('a'), publication.begin('b')
    (b.root / 'plan').write_text('B\n')
    close(publication, b)
    (a.root / 'plan').write_text('A\n')
    (a.root / 'dependent').write_text('private together')
    pending = close(publication, a)
    assert pending['status'] == 'conflict'
    assert not (publication.public / 'dependent').exists()
    next_call = publication.begin('a')
    assert (next_call.root / 'plan').read_text() == 'A\n'
    assert (next_call.root / 'dependent').read_text() == 'private together'
    (next_call.root / 'unrelated').write_text('public')
    assert close(publication, next_call)['status'] == 'published'
    assert not (publication.public / 'dependent').exists()
    edit = publication.begin('a')
    (edit.root / 'dependent').write_text('revised draft')
    result = close(publication, edit)
    assert result['status'] == 'needs_confirmation'
    assert result['paths'] == ['dependent', 'plan']
    assert not (publication.public / 'dependent').exists()
    assert json.loads(publication.get('candidates', pending['candidate']))['files']['dependent']


def test_read_only_same_bytes_and_independent_events(publication):
    before = publication.head()
    first = publication.begin('a')
    (first.root / 'plan').write_bytes((first.root / 'plan').read_bytes())
    one = close(publication, first)
    second = publication.begin('a')
    two = close(publication, second)
    assert one['status'] == two['status'] == 'unchanged'
    assert publication.head() == before
    assert one['execution'] != two['execution']
    assert one['author_after'] == two['author_after']
    assert (first.root / 'plan').stat().st_ino != (second.root / 'plan').stat().st_ino


def test_rename_delete_binary_and_mode_publish(publication):
    a = publication.begin('a')
    (a.root / 'plan').rename(a.root / 'renamed')
    (a.root / 'renamed').chmod(0o700)
    (a.root / 'binary').write_bytes(b'\0\xff\x01')
    result = close(publication, a)
    assert result['status'] == 'published'
    assert not (publication.public / 'plan').exists()
    assert (publication.public / 'binary').read_bytes() == b'\0\xff\x01'
    assert (publication.public / 'renamed').stat().st_mode & 0o777 == 0o700
    b = publication.begin('a')
    (b.root / 'binary').unlink()
    assert close(publication, b)['status'] == 'published'
    assert not (publication.public / 'binary').exists()


def test_partial_closed_command_draft_does_not_publish(publication):
    a = publication.begin('a')
    (a.root / 'plan').write_text('partial')
    result = close(publication, a, status='failed')
    assert result['status'] == 'retained_partial'
    assert (publication.public / 'plan').read_text().startswith('one')
    assert (publication.begin('a').root / 'plan').read_text() == 'partial'


def test_unclosed_execution_and_unknown_receipt_hold(publication):
    a = publication.begin('a')
    (a.root / 'plan').write_text('not closed')
    with pytest.raises(PublicationHold, match='not closed'):
        publication.finish(a, closure={'boundary_closed': False})
    with pytest.raises(PublicationHold, match='unknown'):
        publication.outcome(a.id)
    with pytest.raises(PublicationHold, match='already exists'):
        publication.begin('a', execution_id=a.id)


def test_real_bash_has_stable_tree_and_merges_after_close(monkeypatch):
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        a, b = [team.roles[r].executor for r in ('growth', 'ops_finance')]
        a.execute('write_file', dict(path='plans/merge', content='one\ntwo\nthree\nfour\nfive\n'))
        b.execute('read_file', dict(path='plans/merge'))
        started, release = threading.Event(), threading.Event()
        original = a._exec_bash
        def paused(args):
            started.set()
            assert release.wait(10)
            return original(args)
        monkeypatch.setattr(a, '_exec_bash', paused)
        command = "python -c " + shlex.quote("from pathlib import Path; p=Path('plans/merge'); assert p.read_text().endswith('five\\n'); p.write_text(p.read_text().replace('one','A')); print('BASH-CLOSED')")
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(a.execute, 'bash', dict(command=command))
            assert started.wait(10)
            b.execute('edit_file', dict(path='plans/merge', old_string='five', new_string='B'))
            release.set()
            result = pending.result(timeout=30)
        assert '[PF workspace merged]' in result, result
        assert (team.roles['ceo'].workspace / 'plans/merge').read_text() == 'A\ntwo\nthree\nfour\nB\n'
        assert a._bash_facts['process_boundary']['descendants_closed'] is True
        assert 'B' in a.execute('read_file', dict(path='plans/merge'))
        assert 'A' in b.execute('bash', dict(command='pf show shared:plans/merge --full'))


def test_multi_tool_reply_refreshes_each_tool_and_modelusage(monkeypatch):
    turns = []
    def respond(team, role, body):
        turns.append(body)
        if len(turns) == 1:
            return calls(('read_file', dict(path='plans/price')), ('read_file', dict(path='plans/price')))
        return final('both reads received')
    with shared_team('pf', respond, workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        a, b = [team.roles[r].executor for r in ('growth', 'ops_finance')]
        b.execute('write_file', dict(path='plans/price', content='INITIAL'))
        original, reads = a._exec_read_file, []
        def read(args):
            result = original(args)
            reads.append(str(result))
            if len(reads) == 1:
                b.execute('write_file', dict(path='plans/price', content='LATEST'))
            return result
        monkeypatch.setattr(a, '_exec_read_file', read)
        team._rotate_sessions()
        message = team._message('growth', 'Read twice')
        team._run_analyst('growth', [message])
        assert 'INITIAL' in reads[0] and 'LATEST' in reads[1]
        requests = [json.loads(line)['body'] for line in team.roles['growth'].usage.path.read_text().splitlines() if json.loads(line)['event'] == 'http_request']
        tools = [message['content'] for message in requests[-1]['messages'] if message['role'] == 'tool']
        assert any('INITIAL' in value for value in tools) and any('LATEST' in value for value in tools)
        assert all('PF workspace' not in value for value in tools)
        assert all(wire['role'] == 'growth' for wire in team.test_wire)
        assert 'needs_confirmation' in a.execute('write_file', dict(path='plans/price', content='my old choice'))


def test_private_draft_never_becomes_peer_current_or_exact_history():
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        a, b = [team.roles[r].executor for r in ('growth', 'ops_finance')]
        a.execute('write_file', dict(path='plans/private', content='FIRST'))
        b.execute('read_file', dict(path='plans/private'))
        a.execute('write_file', dict(path='plans/private', content='PUBLIC'))
        result = b.execute('write_file', dict(path='plans/private', content='SECRET-CANDIDATE'))
        assert 'needs_confirmation' in result
        assert 'SECRET-CANDIDATE' in b.execute('read_file', dict(path='plans/private'))
        assert 'PUBLIC' in a.execute('bash', dict(command='pf show shared:plans/private --full'))
        assert 'SECRET-CANDIDATE' not in a.execute('bash', dict(command='pf log shared:plans/private --detail'))
        store = team.roles['ops_finance'].store
        with closing(store.connect()) as conn:
            private = conn.execute("SELECT v.version_id FROM versions v JOIN requests r USING(event_id) WHERE json_extract(v.metadata,'$.workspace_candidate')=1 AND json_extract(r.request,'$.role')='ops_finance'").fetchall()
        assert private
        for row in private:
            with pytest.raises(PermissionError):
                team.roles['growth'].store.get_content(row[0])


def test_multiple_conflict_hunks_are_conflict_not_hold(publication):
    middle = ''.join(str(i) + '\n' for i in range(25))
    initial = publication.begin('seed')
    (initial.root / 'plan').write_text('first\n' + middle + 'last\n')
    close(publication, initial)
    a, b = publication.begin('a'), publication.begin('b')
    (a.root / 'plan').write_text('A\n' + middle + 'A\n')
    (b.root / 'plan').write_text('B\n' + middle + 'B\n')
    close(publication, b)
    assert close(publication, a)['status'] == 'conflict'
    independent = publication.begin('a')
    (independent.root / 'unrelated').write_text('continue')
    assert close(publication, independent)['status'] == 'published'


def test_resolve_cannot_publish_an_obsolete_candidate_revision(publication, monkeypatch):
    a, b = publication.begin('a'), publication.begin('b')
    (a.root / 'plan').write_text('A')
    (b.root / 'plan').write_text('B')
    close(publication, b)
    draft = close(publication, a)
    reached, resume = threading.Event(), threading.Event()
    def interleave(stage, execution=None):
        if stage == 'manifest_durable' and threading.current_thread().name.startswith('ThreadPoolExecutor'):
            reached.set()
            assert resume.wait(10)
    monkeypatch.setattr(publication, 'stage', interleave)
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(publication.resolve, 'a', a.id, against=draft['head'], revision=draft['candidate'])
        assert reached.wait(10)
        revised = publication.begin('a')
        (revised.root / 'plan').write_text('A revised')
        revision = close(publication, revised)
        resume.set()
        result = pending.result(timeout=10)
    assert result['status'] == 'not_published'
    assert (publication.public / 'plan').read_text() == 'B'
    assert (publication.begin('a').root / 'plan').read_text() == 'A revised'


def test_private_draft_output_and_automatic_full_output_remain_private():
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        a, b = [team.roles[r].executor for r in ('growth', 'ops_finance')]
        a.execute('write_file', dict(path='plans/private', content='initial'))
        b.execute('read_file', dict(path='plans/private'))
        a.execute('write_file', dict(path='plans/private', content='public'))
        b.execute('write_file', dict(path='plans/private', content='DRAFT-SECRET' * 20000))
        for tool, args in [('read_file', dict(path='plans/private')), ('bash', dict(command='cat plans/private'))]:
            result = b.execute(tool, args)
            assert 'DRAFT-SECRET' in result
            store = team.roles['ops_finance'].store
            with closing(store.connect()) as conn:
                rows = conn.execute("SELECT version_id FROM versions WHERE event_id=? AND json_extract(metadata,'$.layer') IN ('stdout','tool_return')", (b.last_capture_event,)).fetchall()
            assert rows
            for row in rows:
                with pytest.raises(PermissionError):
                    team.roles['growth'].store.public_content(row[0])
        assert not list((team.roles['ceo'].workspace / 'tool-outputs').glob('*'))
        import re
        path = re.search(r'full output: ([^\]]+)', result).group(1)
        assert 'DRAFT-SECRET' in b.execute('read_file', dict(path=path))


def test_deleted_public_path_is_not_presented_as_current_old_bytes():
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        a, b = [team.roles[r].executor for r in ('growth', 'ops_finance')]
        a.execute('write_file', dict(path='plans/deleted', content='historical bytes'))
        a.execute('bash', dict(command='rm plans/deleted'))
        current = b.execute('bash', dict(command='pf show shared:plans/deleted --full'))
        assert 'deleted' in current.lower() and 'historical bytes' not in current, current
        old = b.execute('bash', dict(command='pf show shared:plans/deleted@v1 --full'))
        assert 'historical bytes' in old, old


@pytest.mark.parametrize('operation', ['append', 'rename', 'delete', 'binary', 'symlink', 'directory'])
def test_real_bash_file_operations_publish_actual_tree(operation):
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        executor = team.roles['growth'].executor
        executor.execute('write_file', dict(path='plans/file', content='base\n'))
        programs = dict(append="printf 'tail\\n' >> plans/file", rename='mv plans/file plans/renamed',
                        delete='rm plans/file', binary="python -c \"from pathlib import Path; Path('plans/file').write_bytes(bytes([0,255,1]))\"",
                        symlink='rm plans/file; ln -s commercial.md plans/file', directory='rm plans/file; mkdir plans/file')
        result = executor.execute('bash', dict(command=programs[operation]))
        assert 'PF workspace published' in result, result
        assert executor._bash_facts['process_boundary']['descendants_closed'] is True
        root = team.roles['ceo'].workspace
        if operation == 'append':
            assert (root / 'plans/file').read_text() == 'base\ntail\n'
        elif operation == 'rename':
            assert not (root / 'plans/file').exists() and (root / 'plans/renamed').read_text() == 'base\n'
        elif operation == 'delete':
            assert not (root / 'plans/file').exists()
        elif operation == 'binary':
            assert (root / 'plans/file').read_bytes() == b'\0\xff\1'
        elif operation == 'symlink':
            assert (root / 'plans/file').is_symlink()
        else:
            assert (root / 'plans/file').is_dir()


def test_real_bash_same_position_append_is_conflict_and_unrelated_tool_continues(monkeypatch):
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        a, b = [team.roles[r].executor for r in ('growth', 'ops_finance')]
        a.execute('write_file', dict(path='plans/append', content='base\n'))
        ready, resume = threading.Event(), threading.Event()
        original = a._exec_bash
        def pause(args):
            ready.set()
            assert resume.wait(10)
            return original(args)
        monkeypatch.setattr(a, '_exec_bash', pause)
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(a.execute, 'bash', dict(command="printf 'A\\n' >> plans/append; printf private > plans/paired"))
            assert ready.wait(10)
            assert 'published' in b.execute('bash', dict(command="printf 'B\\n' >> plans/append"))
            resume.set()
            result = pending.result(timeout=20)
        assert 'PF workspace conflict' in result, result
        root = team.roles['ceo'].workspace
        assert (root / 'plans/append').read_text() == 'base\nB\n'
        assert not (root / 'plans/paired').exists()
        assert 'published' in a.execute('write_file', dict(path='plans/unrelated', content='okay'))
        assert not (root / 'plans/paired').exists()
        assert team.failure is None and team._lifecycle.reason is None


def test_old_draft_remains_readable_after_public_parent_type_changes(tmp_path):
    import shutil
    public = tmp_path / 'public'
    (public / 'folder').mkdir(parents=True)
    (public / 'folder/file').write_text('base')
    workspace = WorkspacePublication(public, tmp_path / 'private')
    a, b = workspace.begin('a'), workspace.begin('b')
    (a.root / 'folder/file').write_text('draft')
    (b.root / 'folder/file').write_text('public')
    close(workspace, b)
    assert close(workspace, a)['status'] == 'conflict'
    b = workspace.begin('b')
    shutil.rmtree(b.root / 'folder')
    (b.root / 'folder').write_text('new type')
    close(workspace, b)
    a = workspace.begin('a')
    assert (a.root / 'folder/file').read_text() == 'draft'
    (a.root / 'unrelated').write_text('okay')
    assert close(workspace, a)['status'] == 'published'
    assert (public / 'folder').read_text() == 'new type'


def test_rejected_tool_does_not_leave_unfinished_execution():
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        executor = team.roles['growth'].executor
        assert 'disabled' in executor.execute('text_create', dict(text='old interface'))
        assert 'Unknown tool' in executor.execute('does_not_exist', {})
        assert 'published' in executor.execute('write_file', dict(path='plans/next', content='okay'))
        with closing(team.workspace_publication.connect()) as conn:
            assert conn.execute('SELECT count(*) FROM executions WHERE outcome IS NULL').fetchone()[0] == 0


def test_unreadable_directory_after_real_bash_is_not_a_deletion():
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        executor = team.roles['growth'].executor
        executor.execute('write_file', dict(path='plans/locked/file', content='must survive'))
        head = team.workspace_publication.head()
        with pytest.raises(PublicationHold):
            executor.execute('bash', dict(command='chmod 000 plans/locked'))
        assert team.workspace_publication.head() == head
        assert (team.roles['ceo'].workspace / 'plans/locked/file').read_text() == 'must survive'
        assert team._lifecycle.reason == 'workspace_publication_unknown'
        assert executor._bash_facts['process_boundary']['descendants_closed'] is True
