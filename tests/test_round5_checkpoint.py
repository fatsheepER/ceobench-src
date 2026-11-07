import shutil
import os
import tempfile
import json
from pathlib import Path

from test_pf_publication import publication
from test_preflight_team_batch import frozen

import pytest

from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.team_checkpoint import CheckpointDataError
from test_preflight_team_checkpoint import clients, dispose, fast_simulator
from test_shared_workspace import shared_team


@pytest.fixture
def r5_disk(tmp_path):
    root = Path(os.environ.get('CEOBENCH_PUBLICATION_ARTIFACTS', tmp_path))
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='r5-checkpoint-', dir=root))


def test_checkpoint_rejects_unfinished_publication():
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        publication = team.workspace_publication
        execution = publication.begin('growth')
        try:
            with pytest.raises(RuntimeError, match='[Pp]ublication|[Ww]orkspace'):
                team.checkpoint()
        finally:
            publication.finish(execution, status='failed', closure={'boundary_closed': True})


@pytest.mark.parametrize('missing', ['authority', 'head_manifest', 'draft_blob', 'source_history'])
def test_restore_requires_publication_authority(tmp_path, missing):
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        if missing == 'draft_blob':
            from test_round5_workflows import make_draft
            team._set_phase('analysis')
            row, _ = make_draft(team, 'growth')
        snapshot = team.checkpoint()
        damaged = tmp_path / 'missing-authority'
        shutil.copytree(snapshot, damaged)
        private = damaged / 'private/workspace-publication'
        publication = team.workspace_publication
        if missing == 'authority':
            (private / 'authority.sqlite').unlink()
        elif missing == 'head_manifest':
            (private / 'manifests' / publication.head()).unlink()
        elif missing == 'source_history':
            import sqlite3
            with sqlite3.connect(private / 'authority.sqlite') as conn:
                conn.execute('DROP TABLE source_head')
        else:
            candidate = json.loads(publication.get('candidates', row['candidate']))
            (private / 'blobs' / candidate['files']['plans/draft.txt']['blob']).unlink()
        restored = None
        try:
            with pytest.raises(CheckpointDataError, match='[Pp]ublication|[Ww]orkspace'):
                restored = MultiAgentRuntime.restore(damaged, tmp_path / 'restored',
                    client_factory=clients(), simulator_factory=fast_simulator,
                    force_resume=True)
        finally:
            if restored is not None:
                dispose(restored)


def test_concurrent_conflicting_finalization_rejects_loser(publication, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from saas_bench.workspace_publication import PublicationHold
    from test_pf_publication import close
    execution = publication.begin('growth')
    (execution.root / 'plan').write_text('candidate')
    ready = threading.Barrier(2)
    def stage(name, identity=None):
        if name == 'candidate_durable':
            ready.wait(timeout=5)
    monkeypatch.setattr(publication, 'stage', stage)
    def finish(status):
        try:
            return close(publication, execution, status=status)
        except PublicationHold as exc:
            return exc
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(finish, ['succeeded', 'failed']))
    assert sum(isinstance(result, PublicationHold) for result in results) == 1, results
    result = next(result for result in results if isinstance(result, dict))
    with publication.connect() as conn:
        row = conn.execute('SELECT candidate,outcome FROM executions WHERE id=?', (execution.id,)).fetchone()
    assert row['candidate'] == result['candidate']
    assert json.loads(row['outcome']) == result


def test_cancelled_closed_tool_preserves_unpublished_candidate(publication):
    from saas_bench.run_lifecycle import RunCancelled, WorkerLifecycle
    from test_pf_publication import close
    publication.lifecycle = WorkerLifecycle()
    execution = publication.begin('growth')
    (execution.root / 'plan').write_text('cancelled private draft')
    publication.lifecycle.cancel('requested_pause')
    with pytest.raises(RunCancelled):
        close(publication, execution, status='cancelled')
    assert publication.outcome(execution.id)['status'] == 'retained_partial'
    assert publication.draft_rows('growth')
    assert 'cancelled' not in (publication.public / 'plan').read_text()


def test_real_driver_restores_drafts_confirmations_sources_without_replay(frozen, r5_disk, monkeypatch):
    tmp_path = r5_disk
    from test_checkpoint_driver import Driver
    from saas_bench import team_batch as batch
    harness = Driver(frozen, tmp_path, monkeypatch, worker_options={'hooks': 'round5_checkpoint_fixture'}, budget_seconds=240)
    try:
        policy, entry = harness.policy('r5-history', 'pf')
        _, original, first = harness.run(policy, entry)
        assert original['status'] == 'finished'
        selected = Path(first['checkpoints']['answers_ready'])
        saved = batch.read(selected / 'team.json')
        assert saved['configuration']['workspace_protocol'] == 'snapshot-v1'
        assert saved['workspace_publication']['head']
        for _ in range(2):
            _, record, proof = harness.run(policy, entry, checkpoint=selected, resume=True)
            assert record['status'] == 'finished' and record['day'] == 7
            assert proof['workspace_restored_exactly']
            assert proof['confirmed_set_not_blocked'] and proof['confirmed_before_revision']
            assert proof['new_revision_blocked']
            assert 'R5-RECOVER-PRIVATE' in proof['restored_private_read']
            assert 'R5-RECOVER-PRIVATE' not in proof['restored_peer_read']
            assert not proof['restored_peer_read'].startswith('Error:')
            assert all('R5-RECOVER-PRIVATE' not in result and not result.startswith('Error:')
                       and '[exit code:' not in result for result in proof['restored_public_queries'])
            assert proof['restored_source'] == first['workspace_final']['scripts']
            assert proof['publication_pending'] == 0
            assert all(proof['finished']['usage_matches_wire'].values())
            assert [row for row in proof['finished']['operations'] if row['tool'] == 'set_capacity_tier'] == [
                row for row in first['finished']['operations'] if row['tool'] == 'set_capacity_tier']
            assert not any(row['role'] != 'ceo' for row in proof['wire'])
            assert proof['finished']['script_receipts'] == first['finished']['script_receipts']
    finally:
        harness.close()


@pytest.mark.parametrize('stage', ['candidate_durable', 'committed'])
def test_real_driver_killed_publication_recovers_only_selected_checkpoint(frozen, r5_disk, monkeypatch, stage):
    tmp_path = r5_disk
    from contextlib import closing
    import sqlite3
    from saas_bench import team_batch as batch
    from test_checkpoint_driver import Driver
    harness = Driver(frozen, tmp_path, monkeypatch, worker_options={
        'hooks': 'round5_checkpoint_fixture', 'fault': 'kill', 'stage': stage}, budget_seconds=240)
    try:
        policy, entry = harness.policy('r5-kill', 'pf')
        attempt, record, _ = harness.run(policy, entry, expected=3, killed=True)
        assert (Path(entry['output_dir']) / 'HOLD').exists()
        fault = batch.read(harness.children[-1][1].with_suffix('.proof.fault.json'))
        private = Path(fault['root']) / 'private/workspace-publication'
        with closing(sqlite3.connect(private / 'authority.sqlite')) as conn:
            outcome, = conn.execute('SELECT outcome FROM executions WHERE id=?', (fault['execution'],)).fetchone()
            assert conn.execute('SELECT count(*) FROM executions WHERE id=?', (fault['execution'],)).fetchone()[0] == 1
        if stage == 'committed':
            assert json.loads(outcome)['status'] == 'published'
        else:
            assert outcome is None
        selected = Path(record['checkpoint'])
        harness.worker_options = {'hooks': 'round5_checkpoint_fixture'}
        _, resumed, proof = harness.run(policy, entry, checkpoint=selected, resume=True)
        assert resumed['status'] == 'finished'
        assert proof['workspace_restored_exactly'] and proof['publication_pending'] == 0
        assert not (Path(proof['ready']['source']) / 'roles/shared/plans/fault-a').exists()
    finally:
        harness.close()


@pytest.mark.parametrize('fault', ['stop', 'disconnect'])
def test_real_controller_publication_stop_and_lost_acknowledgement(frozen, r5_disk, monkeypatch, fault):
    tmp_path = r5_disk
    from test_checkpoint_driver import Driver
    harness = Driver(frozen, tmp_path, monkeypatch, worker_options={
        'hooks': 'round5_checkpoint_fixture', 'fault': fault,
        'stage': 'committed' if fault == 'disconnect' else 'manifest_durable'}, budget_seconds=240)
    try:
        policy, entry = harness.policy('r5-' + fault, 'pf')
        harness.worker_options['stop'] = str(policy.parent / 'STOP')
        _, record, proof = harness.run(policy, entry)
        results = [row for row in proof['publication_results'] if 'plans/fault-a' in row['paths']]
        assert len(results) == 1
        assert proof['publication_pending'] == 0
        assert all(proof['finished']['usage_matches_wire'].values())
        if fault == 'stop':
            assert record['status'] == 'paused'
            assert proof['stop_lock_closed']
            assert results[0]['status'] == 'retained_partial'
            assert record['recovery_validation']['eligible']
        else:
            assert record['status'] == 'finished' and record['day'] == 7
            assert results[0]['status'] == 'published'
            assert len({row['execution'] for row in results}) == 1
    finally:
        harness.close()



def test_pf_client_disconnect_after_resolution_commit_preserves_outcome(monkeypatch):
    import shlex
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import closing
    from test_round5_workflows import make_draft
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        row, _ = make_draft(team, 'growth', content='RESOLVED-AFTER-DISCONNECT')
        publication = team.workspace_publication
        worker = team.roles['growth'].executor
        entered, release = threading.Event(), threading.Event()
        resolved = []
        def stage(name, identity=None):
            if name != 'committed':
                return
            with closing(publication.connect()) as conn:
                is_resolution = conn.execute('SELECT 1 FROM resolutions WHERE id=?', (identity,)).fetchone()
            if is_resolution:
                resolved.append(identity)
                entered.set()
                assert release.wait(10)
        monkeypatch.setattr(publication, 'stage', stage)
        arguments = ['workspace', 'resolve', publication.draft_ref(row), '--against', publication.head()]
        program = "import socket,os,json; from pathlib import Path; s=socket.socket(socket.AF_UNIX); s.connect(os.environ['PF_SOCKET']); s.sendall((json.dumps(dict(argv=" + repr(arguments) + ",cwd=os.getcwd()))+'\\n').encode()); s.close(); Path('client.closed').write_text('closed'); print('CLIENT-CLOSED')"
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(worker.execute, 'bash', dict(command='python -c ' + shlex.quote(program)))
            try:
                assert entered.wait(10)
                from test_team_stop_residuals import wait_until
                wait_until(lambda: worker.workspace_execution and (worker.workspace_execution.root / 'client.closed').exists())
                wait_until(lambda: worker._pf_service is not None and worker._pf_service.cancelled.is_set())
            finally:
                release.set()
            result = pending.result(timeout=20)
        assert 'CLIENT-CLOSED' in result
        assert len(resolved) == 1
        assert publication.outcome(resolved[0])['status'] == 'published'
        assert not publication.draft_rows('growth')
        assert worker._pf_service is None and worker._bash_facts['boundary_closed']
        assert 'RESOLVED-AFTER-DISCONNECT' in worker.execute('read_file', dict(path='plans/draft.txt'))
        with closing(publication.connect()) as conn:
            assert conn.execute('SELECT count(*) FROM resolutions').fetchone()[0] == 1
        assert team.failure is None


def test_private_draft_read_reaches_owner_model_without_public_base():
    from test_preflight_team_flow import ADVANCE, calls, final
    from test_round5_workflows import make_draft
    seen = []
    def respond(team, role, body):
        if role == 'ceo':
            return calls(('bash', dict(command=ADVANCE)))
        if role != 'growth':
            return final('No additional action.')
        seen.append(body)
        if len(seen) == 1:
            return calls(('read_file', dict(path='plans/draft.txt')))
        return final('Private draft retained.')
    with shared_team('pf', respond, workspace_protocol='snapshot-v1') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        team._set_phase('analysis')
        make_draft(team, 'growth', content='PRIVATE-OWNER-MODEL-READ')
        from test_team_script_state import register_cli
        register_cli(team, 'growth', 'model-read.py', "import novamind_api as n\nprint(n.infrastructure.get_cost_info())\n")
        publication = team.workspace_publication
        publication.check_finish('growth', 0)
        publication.confirm_finish('growth', 0, publication.draft_signature('growth', 0))
        outcome = team.run(stop_after_day=7)
        assert len(seen) == 2, (outcome.reason, team.failure)
        assert 'PRIVATE-OWNER-MODEL-READ' in json.dumps(seen[-1])
        assert outcome.reason == 'observation_end' and team.failure is None
        assert all('PRIVATE-OWNER-MODEL-READ' not in json.dumps(row['body'])
                   for row in team.test_wire if row['role'] != 'growth')


def test_real_driver_missing_publication_authority_holds_before_model(frozen, r5_disk, monkeypatch):
    from test_checkpoint_driver import Driver
    harness = Driver(frozen, r5_disk, monkeypatch, worker_options={'hooks': 'round5_checkpoint_fixture'}, budget_seconds=240)
    try:
        policy, entry = harness.policy('r5-missing-state', 'pf')
        _, _, proof = harness.run(policy, entry)
        source = Path(proof['checkpoints']['answers_ready'])
        damaged = r5_disk / 'selected-missing-authority'
        shutil.copytree(source, damaged)
        (damaged / 'private/workspace-publication/authority.sqlite').unlink()
        _, record, proof = harness.run(policy, entry, checkpoint=damaged, resume=True, expected=3)
        assert (Path(entry['output_dir']) / 'HOLD').exists()
        assert not proof['wire'] and proof['closed'] is None
        assert record['status'] == 'paused' and not record['execution_started']
        assert 'publication authority is missing' in json.dumps(record)
    finally:
        harness.close()


@pytest.mark.parametrize('legacy', [False, True])
def test_publication_source_authority_missing_is_not_reseeded(publication, legacy):
    from contextlib import closing
    from saas_bench.workspace_publication import WorkspacePublication, PublicationHold
    from test_pf_publication import close
    execution = publication.begin('growth')
    (execution.root / 'plan').write_text('published')
    close(publication, execution)
    with closing(publication.connect()) as conn, conn:
        before = {table: conn.execute('SELECT * FROM ' + table).fetchall()
                  for table in ('executions', 'authors', 'drafts')}
        conn.execute('DROP TABLE source_head')
        conn.execute('DROP TABLE source_bases')
        if legacy:
            conn.execute('PRAGMA user_version=0')
    if not legacy:
        with pytest.raises(PublicationHold, match='source authority'):
            WorkspacePublication(publication.public, publication.private)
        return
    restored = WorkspacePublication(publication.public, publication.private)
    with closing(restored.connect()) as conn:
        assert all(conn.execute('SELECT * FROM ' + table).fetchall() == rows for table, rows in before.items())
        assert conn.execute("SELECT execution FROM source_head WHERE path='plan'").fetchone()[0] == execution.id
        sources = json.loads(conn.execute("SELECT sources FROM source_bases WHERE actor='growth'").fetchone()[0])
        assert sources and all(value is None for value in sources.values())


def test_committed_resolution_reply_survives_expired_delivery_budget(monkeypatch):
    import time
    from contextlib import closing
    from saas_bench.pf_deadline import current_budget
    from test_round5_workflows import make_draft
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        row, _ = make_draft(team, 'growth', content='COMMITTED-BEFORE-DELIVERY-DEADLINE')
        publication = team.workspace_publication
        committed = []
        def stage(name, identity=None):
            if name != 'committed':
                return
            with closing(publication.connect()) as conn:
                if conn.execute('SELECT 1 FROM resolutions WHERE id=?', (identity,)).fetchone():
                    committed.append(identity)
                    current_budget().deadline = time.monotonic() - 1
        monkeypatch.setattr(publication, 'stage', stage)
        worker = team.roles['growth'].executor
        output = worker.execute('bash', dict(command='pf workspace resolve ' + publication.draft_ref(row) + ' --against ' + publication.head()))
        assert '[PF workspace published]' in output and '[exit code:' not in output, output
        assert len(committed) == 1 and publication.outcome(committed[0])['status'] == 'published'
        assert not publication.draft_rows('growth')
        assert worker._pf_service is None and worker._bash_facts['boundary_closed']
        assert 'COMMITTED-BEFORE-DELIVERY-DEADLINE' in worker.execute('read_file', dict(path='plans/draft.txt'))
        assert team.failure is None
