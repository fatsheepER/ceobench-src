import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from contextlib import closing
import sqlite3
import tempfile

import pytest

from saas_bench.workspace_publication import WorkspacePublication, PublicationHold
from test_pf_publication import close, publication


@pytest.mark.parametrize('stage', ['blobs_durable', 'candidate_durable', 'intent_durable', 'manifest_durable', 'before_commit', 'committed', 'projection_swapped', 'evidence_durable', 'receipt'])
def test_real_bash_publication_process_death_has_one_atomic_outcome(stage):
    evidence = Path(os.environ.get('CEOBENCH_PUBLICATION_ARTIFACTS', '/home/feiyang/Documents/Codex/2026-11-07/task-4/persistence'))
    evidence.mkdir(parents=True, exist_ok=True)
    case = Path(tempfile.mkdtemp(prefix=stage + '-', dir=evidence))
    program = '''
import json, os, signal
from pathlib import Path
from test_shared_workspace import shared_team
with shared_team('pf', workspace_protocol='snapshot-v1') as team:
    team._set_phase('analysis')
    executor = team.roles['growth'].executor
    result = executor.execute('bash', dict(command="printf old > plans/a; printf old > plans/b"))
    assert 'published' in result, result
    coordinator = team.workspace_publication
    def fault(stage, execution=None):
        if stage == os.environ['PUBLICATION_STAGE']:
            Path(os.environ['PUBLICATION_CASE'], 'fault.json').write_text(json.dumps(dict(root=str(team.root), execution=execution, stage=stage)))
            os.kill(os.getpid(), signal.SIGKILL)
    coordinator.stage = fault
    executor.execute('bash', dict(command="printf new > plans/a; printf new > plans/b"))
    raise AssertionError('Fault stage did not run')
'''
    env = dict(os.environ, CEOBENCH_TEAM_ARTIFACTS=str(case / 'team'),
               PUBLICATION_CASE=str(case), PUBLICATION_STAGE=stage,
               PYTHONPATH=os.pathsep.join((str(Path.cwd() / 'tests'), str(Path.cwd() / 'src'))))
    result = subprocess.run([sys.executable, '-c', program], env=env, capture_output=True, timeout=45)
    (case / 'stdout').write_bytes(result.stdout)
    (case / 'stderr').write_bytes(result.stderr)
    assert result.returncode == -signal.SIGKILL, result.stderr.decode()
    fault = json.loads((case / 'fault.json').read_text())
    root = Path(fault['root'])
    public, private = root / 'roles/shared', root / 'private/workspace-publication'
    original = tuple((public / ('plans/' + name)).read_text() for name in ('a', 'b'))
    assert original in (('old', 'old'), ('new', 'new'))
    workspace = WorkspacePublication(public, private)
    tree = workspace.tree(workspace.head())
    actual = tuple(workspace.get('blobs', tree['plans/' + name]['blob']).decode() for name in ('a', 'b'))
    if stage in ('committed', 'projection_swapped', 'evidence_durable', 'receipt'):
        assert actual == ('new', 'new')
        one = workspace.outcome(fault['execution'])
        assert one == workspace.outcome(fault['execution'])
        assert one['head'] == workspace.head() and one['status'] == 'published'
    else:
        assert actual == ('old', 'old')
        with pytest.raises(PublicationHold, match='unknown'):
            workspace.outcome(fault['execution'])
    if stage not in ('committed', 'projection_swapped', 'evidence_durable', 'receipt'):
        with pytest.raises(PublicationHold, match='prior process'):
            workspace.begin('ops_finance')
    workspace.recover_projection()
    assert tuple((public / ('plans/' + name)).read_text() for name in ('a', 'b')) == actual
    with closing(workspace.connect()) as conn:
        assert conn.execute('SELECT count(*) FROM executions WHERE id=?', (fault['execution'],)).fetchone()[0] == 1
        assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    with pytest.raises(PublicationHold, match='already exists'):
        workspace.begin('growth', execution_id=fault['execution'])


def test_duplicate_finalize_reuses_durable_candidate_and_result(publication):
    a = publication.begin('a')
    (a.root / 'plan').write_text('one committed edit')
    result = close(publication, a)
    (a.root / 'plan').write_text('late unrelated bytes')
    assert close(publication, a) == result
    assert (publication.public / 'plan').read_text() == 'one committed edit'
    with closing(publication.connect()) as conn:
        assert conn.execute('SELECT count(*) FROM executions').fetchone()[0] == 1


def test_unknown_persistence_error_through_real_executor_holds(monkeypatch):
    from test_shared_workspace import shared_team
    from saas_bench.run_lifecycle import RunCancelled
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        executor = team.roles['growth'].executor
        def broken(stage, execution=None):
            if stage == 'before_commit':
                raise OSError('injected durable transaction failure')
        monkeypatch.setattr(team.workspace_publication, 'stage', broken)
        with pytest.raises(PublicationHold, match='transaction failure'):
            executor.execute('bash', dict(command='printf retained > plans/retained'))
        assert executor._bash_facts['process_boundary']['descendants_closed'] is True
        assert not (team.roles['ceo'].workspace / 'plans/retained').exists()
        assert team._lifecycle.reason == 'workspace_publication_unknown'
        with pytest.raises(RunCancelled):
            executor.execute('bash', dict(command='touch plans/should-not-run'))
        assert not (team.roles['ceo'].workspace / 'plans/should-not-run').exists()


def test_committed_but_unprojected_retry_repairs_without_reexecution(publication, monkeypatch):
    a = publication.begin('a')
    (a.root / 'plan').write_text('committed')
    def dropped(stage, execution=None):
        if stage == 'committed':
            raise ConnectionError('response disconnected after commit')
    monkeypatch.setattr(publication, 'stage', dropped)
    try:
        close(publication, a)
    except ConnectionError:
        pass
    monkeypatch.setattr(publication, 'stage', lambda *args: None)
    result = close(publication, a)
    assert result['status'] == 'published'
    assert (publication.public / 'plan').read_text() == 'committed'
    with pytest.raises(PublicationHold, match='Conflicting'):
        close(publication, a, status='failed')


def test_late_evidence_repair_cannot_replace_newer_public_file(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from test_shared_workspace import shared_team
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        a, b = [team.roles[r].executor for r in ('growth', 'ops_finance')]
        a.execute('write_file', dict(path='plans/order', content='seed'))
        coordinator = team.workspace_publication
        original = coordinator.publish_evidence
        recover = coordinator.recover_projections
        waiting, resume = threading.Event(), threading.Event()
        failed = []
        def interrupt(store, identity, head, files, actor):
            tree = coordinator.tree(head)
            entry = tree.get('plans/order')
            if actor == 'growth' and entry and coordinator.get('blobs', entry['blob']) == b'FIRST' and not failed:
                failed.append(identity)
                raise ConnectionError('evidence acknowledgement lost')
            return original(store, identity, head, files, actor)
        def delayed(store=None):
            if threading.current_thread().name.startswith('ThreadPoolExecutor'):
                waiting.set()
                assert resume.wait(20)
            return recover(store)
        monkeypatch.setattr(coordinator, 'publish_evidence', interrupt)
        monkeypatch.setattr(coordinator, 'recover_projections', delayed)
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(a.execute, 'write_file', dict(path='plans/order', content='FIRST'))
            assert waiting.wait(10)
            second = b.execute('write_file', dict(path='plans/order', content='SECOND'))
            resume.set()
            first = pending.result(timeout=20)
        assert 'published' in first and 'published' in second
        assert (team.roles['ceo'].workspace / 'plans/order').read_text() == 'SECOND'
        store = team.roles['growth'].store
        latest, _ = store.latest_version('plans/order', 'file_bytes', 'shared')
        assert store.public_content(latest)[1] == b'SECOND'
        assert 'SECOND' in a.execute('bash', dict(command='pf show shared:plans/order --full'))
        assert team.failure is None and team._lifecycle.reason is None
