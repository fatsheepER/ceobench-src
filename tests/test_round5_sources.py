import json
import shlex
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import copy

import pytest

from saas_bench import evidence_handles
from test_pf_candidates import query
from test_pf_relations import cli, details
from test_preflight_team_checkpoint import fast_simulator
from test_preflight_team_flow import ADVANCE, calls, final
from test_relation_batch import batch, create
from test_receipt_outputs import artifact
from test_round5_workflows import make_draft
from test_shared_workspace import shared_team


@pytest.mark.parametrize('tool', ['write_file', 'bash'])
def test_peer_candidate_details_do_not_disclose_private_write_arguments(tool):
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        secret = 'R5-PRIVATE-WRITE-ARGUMENT'
        owner = team.roles['growth'].executor
        if tool == 'write_file':
            make_draft(team, 'growth', content=secret)
        else:
            result = owner.execute('bash', dict(command='printf ' + shlex.quote(secret) + ' > plans/draft.txt; exit 9'))
            assert 'retained_partial' in result
        event = owner.last_capture_event
        peer = team.roles['ops_finance'].executor
        result, page = query(peer, 'pf candidates --history --role growth --detail')
        assert secret not in result and secret not in page['detail_text']
        reference = evidence_handles.public_ref(peer.evidence_store, 'event', event)
        for command in ('pf candidates --history --event ' + reference + ' --detail',
                        'pf depend ' + reference + ' --include-candidates --detail'):
            assert secret not in peer.execute('bash', dict(command=command))
        query(peer, 'pf search --text ' + secret)
        assert peer.pf_queries.last_answer['total'] == 0
        assert secret not in peer.execute('read_file', dict(path='plans/draft.txt'))
        assert secret in owner.execute('read_file', dict(path='plans/draft.txt'))


def test_public_sql_bytes_do_not_expose_private_parent_arguments():
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        secret = 'PRIVATE-OUTER-COMMAND'
        owner = team.roles['growth'].executor
        command = "./novamind-operation query 'SELECT amount FROM ledger LIMIT 1'; printf " + secret + ' > plans/private.txt; exit 7'
        result = owner.execute('bash', dict(command=command))
        assert 'amount' in result and 'retained_partial' in result
        peer = team.roles['ops_finance'].executor
        store = peer.evidence_store
        with closing(store.connect()) as conn:
            row = conn.execute("SELECT v.version_id FROM versions v JOIN requests r USING(event_id) WHERE r.query_id IS NOT NULL AND json_extract(v.metadata,'$.layer')='server_public_response' ORDER BY v.rowid DESC LIMIT 1").fetchone()
        handle = evidence_handles.index(store).name(row[0])
        assert 'amount' in cli(team, 'ops_finance', 'pf', 'show', handle, '--full')
        graph = details(team, 'ops_finance', 'depend', handle, '--include-candidates')
        assert secret not in json.dumps(graph)
        output, page = query(peer, 'pf candidates --history --role growth --detail')
        assert secret not in output and secret not in page['detail_text']
        _, page = query(peer, 'pf candidates --history --role growth --kind sql --table ledger --column amount')
        item = next(row for row in page['items'] if row['version'] == handle)
        _, detail = query(peer, 'pf more ' + page['detail_cursor'])
        assert secret not in detail['detail_text']
        _, directory = query(peer, 'pf candidates --history --role growth --sql-accesses --table ledger --column amount')
        assert any(execution['content_ref'] == item['content_ref']
                   for row in directory['items'] for execution in row['executions'])
        graph = details(team, 'ops_finance', 'rdepend', handle, '--include-candidates')
        member = next(row for row in graph['items'] if row.get('basis') == 'execution_membership')
        assert member['root_event_ref'] == item['event_ref']
        _, navigated = query(peer, member['navigation'])
        assert any(row['content_ref'] == item['content_ref'] for row in navigated['items'])
        assert secret not in json.dumps(graph)


def merge_public_versions(team, monkeypatch):
    growth = team.roles['growth'].executor
    ops = team.roles['ops_finance'].executor
    path = 'plans/merge.txt'
    growth.execute('write_file', dict(path=path, content='one\ntwo\nthree\nfour\nfive\n'))
    ops.execute('read_file', dict(path=path))
    started, release = threading.Event(), threading.Event()
    original = growth._exec_bash

    def paused(args):
        started.set()
        assert release.wait(20)
        return original(args)

    monkeypatch.setattr(growth, '_exec_bash', paused)
    program = "from pathlib import Path; p=Path('plans/merge.txt'); p.write_text(p.read_text().replace('one', 'GROWTH'))"
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(growth.execute, 'bash', dict(command='python -c ' + shlex.quote(program)))
        try:
            assert started.wait(10)
            changed = ops.execute('edit_file', dict(path=path, old_string='five', new_string='OPS'))
            assert 'published' in changed
            declaration = batch(team, 'ops_finance', [create('before-merge',
                'shared:plans/merge.txt@v2', ['shared:plans/merge.txt@v1'])], 'merge-source')
            assert declaration['saved'] == 1, declaration
        finally:
            release.set()
        result = pending.result(timeout=30)
    assert '[PF workspace merged]' in result, result
    return declaration


def test_merge_keeps_structural_sources_separate_from_declarations(monkeypatch):
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        declaration = merge_public_versions(team, monkeypatch)
        current = cli(team, 'ops_finance', 'pf', 'show', 'shared:plans/merge.txt@v3', '--full')
        assert 'GROWTH' in current and 'OPS' in current
        graph = details(team, 'ops_finance', 'depend', 'shared:plans/merge.txt@v3', '--include-candidates')
        assert graph['root']['workspace_namespace'] == 'shared'
        assert graph['root']['publisher_role'] == 'growth'
        assert graph['root']['writer_role'] == 'growth'
        assert graph['root']['producer_role'] is None
        provenance = graph['root']['publication']
        assert provenance['candidate']['sha256'] != provenance['result']['sha256']
        assert provenance['writer_event'].startswith('growth:event')
        structural = [row for row in graph['items'] if row['origin'] == 'workspace_publication']
        assert all(row['assertion'] == 'execution_observation' and row['relation'] is None for row in structural)
        assert {row['source']['ref']['cite'] for row in structural} == {
            'shared:plans/merge.txt@v1', 'shared:plans/merge.txt@v2'}
        assert not details(team, 'ops_finance', 'depend', 'shared:plans/merge.txt@v3')['items']
        old = details(team, 'ops_finance', 'depend', 'shared:plans/merge.txt@v2')
        assert any(row['relation'] == declaration['items'][0]['relation'] for row in old['items'])
        reverse = details(team, 'ops_finance', 'rdepend', 'shared:plans/merge.txt@v2', '--include-candidates')
        assert any(row['target']['ref']['cite'] == 'shared:plans/merge.txt@v3' and row['origin'] == 'workspace_publication'
                   for row in reverse['items'])
        old_bytes = cli(team, 'ops_finance', 'pf', 'show', 'shared:plans/merge.txt@v1', '--full')
        assert 'one\ntwo\nthree\nfour\nfive\n' in old_bytes
        assert 'GROWTH' not in old_bytes and 'OPS' not in old_bytes
        blame = cli(team, 'ops_finance', 'pf', 'blame', 'shared:plans/merge.txt@v3')
        assert 'v2 ' in blame and 'OPS' in blame and 'v3 ' in blame and 'GROWTH' in blame
        own = declaration['items'][0]
        denied = batch(team, 'growth', [dict(item_id='foreign-revision', op='revise', relation=own['relation'],
            expected_revision=own['revision'], reason='Not my declaration.', sources=[])], 'foreign-revision')
        assert denied['saved'] == 0


def test_legacy_exact_file_namespace_does_not_rebind_shared_path():
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        store = copy(team.roles['growth'].store)
        store.identity = {key: value for key, value in store.identity.items() if key != 'workspace_namespace'}
        event = store.begin_event('write_file', dict(path='plans/legacy.txt', content='LEGACY-BYTES'))
        version = store.version(event, 'legacy_after', b'LEGACY-BYTES', layer='file_bytes',
            object_id='plans/legacy.txt', owner_role='growth')
        store.complete(event, status='succeeded', changed_paths=['plans/legacy.txt'])
        worker = team.roles['growth'].executor
        worker.execute('write_file', dict(path='plans/legacy.txt', content='SHARED-BYTES'))
        worker.execute('write_file', dict(path='plans/shared-only.txt', content='ONLY-SHARED'))
        old = cli(team, 'ops_finance', 'pf', 'show', 'growth:plans/legacy.txt@v1', '--full')
        assert 'LEGACY-BYTES' in old and 'SHARED-BYTES' not in old
        assert store.get_content(version)[1] == b'LEGACY-BYTES'
        shared = cli(team, 'ops_finance', 'pf', 'show', 'shared:plans/legacy.txt@v1', '--full')
        assert 'SHARED-BYTES' in shared and 'LEGACY-BYTES' not in shared
        absent = team.roles['ops_finance'].executor.execute('bash', dict(command='pf show growth:plans/shared-only.txt@v1 --full'))
        assert 'ONLY-SHARED' not in absent and ('Error' in absent or 'error' in absent)


def test_published_write_binds_source_from_actual_model_context():
    turns = {}
    receipts = []

    def respond(team, role, body):
        if role == 'ceo':
            return calls(('bash', dict(command=ADVANCE)))
        if role == 'ops_finance':
            return final('No additional action.')
        turns[role] = turns.get(role, 0) + 1
        if turns[role] == 1:
            return calls(('write_file', dict(path='plans/authored.json', content='{"forecast":73}')))
        if turns[role] == 2:
            return calls(('escalate', dict(items=[create('authored-current',
                'shared:plans/decision.txt@v1', ['shared:plans/authored.json@v1'], assertion='inferred')])))
        receipts.append(next((message['content'] for message in reversed(body['messages'])
                              if message['role'] == 'tool'), None))
        return final('Finished the source declaration attempt.')

    with shared_team('pf', respond, workspace_protocol='snapshot-v1') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        team.roles['ceo'].executor.execute('write_file', dict(path='plans/decision.txt', content='decision'))
        outcome = team.run(stop_after_day=7)
        assert len(receipts) == 1, dict(receipts=receipts, turns=turns, outcome=outcome.reason, failure=team.failure)
        assert isinstance(receipts[0], str), receipts
        receipt = json.loads(receipts[0])
        assert receipt['saved'] == 1, receipt
        assert receipt['items'][0]['sources'][0]['ref']['capture']['reading_scope'] == 'authored', receipt
        assert outcome.reason == 'observation_end' and team.failure is None, dict(outcome=outcome.reason, failure=team.failure)
        assert turns['growth'] == 3, turns
        usage = [json.loads(line) for line in team.roles['growth'].usage.path.read_text().splitlines()]
        requests = [row['body'] for row in usage if row['event'] == 'http_request']
        assert requests == [row['body'] for row in team.test_wire if row['role'] == 'growth']
        assert 'authored-current' in json.dumps(requests[-1])


@pytest.mark.parametrize('history', ['private_symlink', 'unpublished_comparison_base'])
def test_publication_details_do_not_expose_private_history(monkeypatch, history):
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        owner = team.roles['growth'].executor
        publication = team.workspace_publication
        if history == 'private_symlink':
            path, version = 'plans/history.txt', 'shared:plans/history.txt@v1'
            private_value = 'PRIVATE-HISTORICAL-SYMLINK-TARGET'
            failed = owner.execute('bash', dict(command='ln -s ' + private_value + ' ' + path + '; exit 3'))
            assert 'retained_partial' in failed, failed
            updated = owner.execute('bash', dict(command='rm ' + path + '; printf replacement > ' + path))
        else:
            merge_public_versions(team, monkeypatch)
            path, version = 'plans/merge.txt', 'shared:plans/merge.txt@v4'
            with closing(publication.connect()) as conn:
                private_base = json.loads(conn.execute("SELECT bases FROM authors WHERE actor='growth'").fetchone()[0])[path]
            private_value = private_base['blob']
            with closing(owner.evidence_store.connect()) as conn:
                published = [json.loads(row[0])['workspace_entry'] for row in conn.execute(
                    "SELECT metadata FROM versions WHERE json_extract(metadata,'$.workspace_publication')=1 AND json_extract(metadata,'$.object_id')=?", (path,))]
            assert private_base not in published
            updated = owner.execute('write_file', dict(path=path, content='replacement'))
        assert 'needs_confirmation' in updated, updated
        row = publication.draft_rows('growth')[0]
        resolved = owner.execute('bash', dict(command='pf workspace resolve ' + publication.draft_ref(row) + ' --against ' + publication.head()))
        assert '[PF workspace published]' in resolved, resolved
        shown = cli(team, 'ops_finance', 'pf', 'show', version, '--full')
        assert 'replacement' in shown and private_value not in shown
        graph = details(team, 'ops_finance', 'depend', version, '--include-candidates')
        assert private_value not in json.dumps(graph['root']['publication']), graph['root']
        assert 'execution_before' not in graph['root']['publication'], graph['root']
        if history == 'private_symlink':
            assert private_value not in json.dumps(graph), graph


@pytest.mark.parametrize('execution_start', ['before_aba', 'after_aba'])
def test_publication_sources_preserve_exact_versions_across_aba(monkeypatch, execution_start):
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        growth = team.roles['growth'].executor
        ops = team.roles['ops_finance'].executor
        path = 'plans/aba.txt'
        first = growth.execute('write_file', dict(path=path, content='one\ntwo\nthree\n'))
        assert '[PF workspace published]' in first, first
        assert 'shared:plans/aba.txt@v1' in ops.execute('read_file', dict(path=path))
        started, release = threading.Event(), threading.Event()
        original = growth._exec_bash

        def paused(args):
            started.set()
            assert release.wait(20), 'ABA writer was not released'
            return original(args)

        def aba():
            second = ops.execute('edit_file', dict(path=path, old_string='one', new_string='PEER'))
            assert '[PF workspace published]' in second, second
            third = ops.execute('edit_file', dict(path=path, old_string='PEER', new_string='one'))
            assert '[PF workspace published]' in third, third

        command = 'python -c ' + shlex.quote("from pathlib import Path; p=Path('plans/aba.txt'); p.write_text(p.read_text().replace('three', 'GROWTH'))")
        if execution_start == 'before_aba':
            monkeypatch.setattr(growth, '_exec_bash', paused)
            with ThreadPoolExecutor(1) as pool:
                pending = pool.submit(growth.execute, 'bash', dict(command=command))
                try:
                    assert started.wait(10), 'Growth did not acquire its execution snapshot'
                    aba()
                finally:
                    release.set()
                fourth = pending.result(timeout=30)
        else:
            aba()
            fourth = growth.execute('bash', dict(command=command))
        assert '[PF workspace published]' in fourth, fourth
        graph = details(team, 'ops_finance', 'depend', 'shared:plans/aba.txt@v4', '--include-candidates')
        sources = {source['kind']: source['version'] for source in graph['root']['publication']['sources']}
        assert sources.get('base') == 'shared:plans/aba.txt@v1', graph['root']
        assert sources.get('current') == 'shared:plans/aba.txt@v3', graph['root']
        expected_before = 'shared:plans/aba.txt@' + ('v1' if execution_start == 'before_aba' else 'v3')
        assert sources.get('execution_before') == expected_before, graph['root']
        edges = {row['kind']: row['source']['ref']['cite'] for row in graph['items']
                 if row['origin'] == 'workspace_publication' and row['target']['ref']['cite'] == 'shared:plans/aba.txt@v4'}
        assert edges['publication_base'] == 'shared:plans/aba.txt@v1'
        assert edges['publication_current'] == 'shared:plans/aba.txt@v3'
        assert edges['publication_execution_before'] == expected_before
        for number in (1, 3):
            saved = cli(team, 'ops_finance', 'pf', 'show', 'shared:plans/aba.txt@v' + str(number), '--full')
            assert 'one\ntwo\nthree' in saved and 'GROWTH' not in saved
        newest = cli(team, 'ops_finance', 'pf', 'show', 'shared:plans/aba.txt@v4', '--full')
        assert 'one\ntwo\nGROWTH' in newest


@pytest.mark.parametrize('exit_code', [0, 1])
def test_unrelated_private_receipts_do_not_block_public_escalate_or_continuation(exit_code):
    turns, receipts = [], []
    own_secret, peer_secret = 'UNRELATED-OPS-PRIVATE-OUTPUT', 'UNRELATED-GROWTH-PRIVATE-OUTPUT'

    def respond(team, role, body):
        if role == 'ceo':
            return calls(('bash', dict(command=ADVANCE)))
        if role == 'growth':
            assert peer_secret not in json.dumps(body)
            return final('ordinary growth advice')
        assert peer_secret not in json.dumps(body)
        turns.append(body)
        if len(turns) == 1:
            return calls(('bash', dict(command='printf ' + own_secret + '; exit ' + str(exit_code))))
        output = body['messages'][-1]['content']
        if len(turns) == 2:
            assert own_secret in output
            assert ('[exit code: 1]' in output) == bool(exit_code)
            return calls(('bash', dict(command='pf log plans/report.txt && pf log plans/source.txt')))
        if len(turns) == 3:
            assert 'shared:plans/report.txt@v1' in output and 'shared:plans/source.txt@v1' in output
            return calls(('escalate', dict(items=[
                create('public', 'shared:plans/report.txt@v1', ['shared:plans/source.txt@v1']),
                create('private-target', 'shared:.git/config@v1', ['shared:plans/source.txt@v1']),
                create('private-source', 'shared:plans/report.txt@v1', ['shared:.git/config@v1']),
                create('escape', 'shared:plans/report.txt@v1', ['../outside.txt@v1']),
            ])))
        receipt = json.loads(output)
        receipts.append(receipt)
        assert own_secret not in output and peer_secret not in output
        if len(turns) == 4:
            assert (receipt['saved'], receipt['rejected']) == (1, 3), receipt
            assert receipt['items'][0]['status'] == 'saved', receipt
            assert [item['error_location'] for item in receipt['items'][1:]] == [
                'target', 'sources[0].ref', 'sources[0].ref']
            return calls(('escalate', dict(items=[create('continued',
                'shared:plans/report.txt@v1', ['shared:plans/source.txt@v1'])])))
        assert len(turns) == 5 and receipt['saved'] == 1, receipt
        return final('The public report uses the exact published source.')

    with shared_team('pf', respond, workspace_protocol='snapshot-v1') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        team._set_phase('analysis')
        growth = team.roles['growth'].executor
        growth.execute('write_file', dict(path='plans/source.txt', content='public source'))
        private = growth.execute('bash', dict(command='printf ' + peer_secret + '; exit 1'))
        assert peer_secret in private and growth._bash_facts['boundary_closed']
        peer_version = growth.last_capture_event + ':tool_return'
        ops = team.roles['ops_finance']
        with pytest.raises(PermissionError):
            ops.store.get_content(peer_version)
        ops.executor.execute('write_file', dict(path='plans/report.txt', content='public report'))
        outcome = team.run(stop_after_day=7)
        assert outcome.reason == 'observation_end', team.failure
        assert len(turns) == 5 and len(ops.registry.relations._load()['records']) == 2
        with pytest.raises(PermissionError):
            ops.store.get_content(peer_version)
        assert peer_secret not in json.dumps(team.test_wire)
        usage = [json.loads(line) for line in ops.usage.path.read_text().splitlines()]
        sent = [row['body'] for row in usage if row['event'] == 'http_request']
        assert sent == [row['body'] for row in team.test_wire if row['role'] == 'ops_finance']
        assert [json.loads(request['messages'][-1]['content']) for request in sent[-2:]] == receipts
        assert not team.server._operation_failed and ops.executor._bash_facts['boundary_closed']
        artifact('private-receipt-escalate.json', dict(exit_code=exit_code, sent=sent, receipts=receipts))
