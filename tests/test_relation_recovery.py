"""Relation receipts, publication uncertainty, and selected-checkpoint recovery."""
from collections import Counter
import json
import os
from pathlib import Path
import time

import httpx
import pytest

from saas_bench import team_batch as batch
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.relation_registry import RelationRegistry
from saas_bench.run_lifecycle import RunCancelled
from test_preflight_team_batch import changed_manifest, frozen
from test_preflight_team_checkpoint import clients, dispose, fast_simulator, make_team as checkpoint_team
from test_preflight_team_evidence import declare
from test_preflight_team_flow import ADVANCE, calls, completion, final, make_team, opening_response


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def item(label, role='growth', source=None):
    return dict(item_id=label, op='create', target={'cite': role + ':r1.1'},
        sources=[{'ref': {'cite': source or role + ':r1.1'}}],
        assertion='author_declared', reason='Keep the exact material for the handoff.')


def action(arguments, call_id):
    result = calls(('relation_batch', arguments))
    result['tool_calls'][0]['id'] = call_id
    return result


def save_sample(label, **evidence):
    if destination := os.environ.get('CEOBENCH_GATE_EVIDENCE'):
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path / ('relations-' + label + '.json')).write_text(json.dumps(evidence, ensure_ascii=False, indent=2))


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_model_receives_each_item_replays_same_call_and_corrects_new_call(mode):
    turns, receipts = Counter(), []
    arguments = {'items': [item('known'), item('invalid', source='growth:never-saved.txt'), item('missing')]}
    arguments['items'][2]['sources'] = [{'ref': {'missing': {
        'description': 'Earlier unsaved output', 'reason': 'No saved source bytes are available.'}}}]

    def respond(team, role, body):
        if role == 'ceo':
            return calls(('bash', {'command': ADVANCE}))
        if role == 'ops_finance':
            return final('No further action.')
        turns[role] += 1
        if turns[role] == 1:
            return action(arguments, 'logical-relation-call')
        receipt = json.loads([m['content'] for m in body['messages'] if m.get('name') == 'relation_batch'][-1])
        receipts.append(receipt)
        if turns[role] == 2:
            assert receipt['status'] == 'partial'
            assert [r['status'] for r in receipt['items']] == ['saved', 'rejected', 'saved']
            assert receipt['items'][2]['sources'][0]['ref']['readable'] is False
            return action(arguments, 'logical-relation-call')
        if turns[role] == 3:
            assert receipt == receipts[0]
            return action({'items': [item('corrected')]}, 'corrected-relation-call')
        assert receipt['saved'] == 1 and receipt['rejected'] == 0
        return final('The exact known source is saved; the earlier output remains explicitly missing.')

    with make_team(mode, respond, 'relations-model-recovery') as team:
        declare(team, 'growth', 'A reusable conclusion.', ['unknown: initial material'])
        outcome = team.run(stop_after_day=7)
        assert outcome.reason == 'observation_end', team.failure
        registry = team.roles['growth'].registry.relations
        state = registry._load()
        assert len(state['records']) == 3
        assert len(state['calls']) == 2
        assert len(receipts) == 3
        usage = rows(team.roles['growth'].usage.path)
        sent = [row['body'] for row in usage if row['event'] == 'http_request']
        wire = [row['body'] for row in team.test_wire if row['role'] == 'growth']
        assert sent == wire and len(sent) == 4
        from saas_bench.registration_prompt import RELATIONS
        for body in sent:
            assert RELATIONS in body['messages'][0]['content']
            assert 'pf candidates' not in body['messages'][0]['content']
            assert any(tool['function']['name'] == 'relation_batch' for tool in body['tools'])
        assert sum(len([m for m in body['messages'] if m.get('name') == 'relation_batch']) for body in sent) == 6
        save_sample(mode + '-model', wire=wire, usage=usage, receipts=receipts, state=state)


def fail_second_item(monkeypatch, registry, boundary, *, wait_for=None):
    replace, fsync = os.replace, os.fsync
    publications, pending = [], {}

    def replace_relation(source, target, *args, **kwargs):
        directory = kwargs.get('dst_dir_fd')
        ours = target == 'relations.json' and directory is not None and (
            Path(os.readlink(f'/proc/self/fd/{directory}')) == registry.workspace)
        if not ours:
            return replace(source, target, *args, **kwargs)
        publications.append(target)
        if len(publications) != 3:
            return replace(source, target, *args, **kwargs)
        if wait_for:
            deadline = time.monotonic() + 10
            while not wait_for.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert wait_for.exists(), 'The partner must actually enter its subprocess before the fault'
        if boundary == 'before_replace':
            raise OSError('relation replacement failed before publication')
        result = replace(source, target, *args, **kwargs)
        pending['directory'] = directory
        return result

    def fail_directory_sync(fd):
        if fd == pending.get('directory'):
            pending.clear()
            raise OSError('relation directory fsync failed after replacement')
        return fsync(fd)

    monkeypatch.setattr(os, 'replace', replace_relation)
    monkeypatch.setattr(os, 'fsync', fail_directory_sync)
    return publications


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('boundary', ['before_replace', 'after_replace'])
def test_uncertain_publication_keeps_prefix_and_stops_models_and_partner(mode, boundary, monkeypatch):
    turns = Counter()
    arguments = {'items': [item('first'), item('uncertain'), item('never-attempted')]}

    def respond(team, role, body):
        turns[role] += 1
        if role == 'growth':
            assert turns[role] == 1, 'The uncertain relation must not be delivered as a recoverable model error'
            return action(arguments, 'uncertain-relation-call')
        if role == 'ops_finance':
            assert turns[role] == 1, 'The partner must not request a new model response after HOLD'
            return calls(('bash', {'command':
                "python -c \"from pathlib import Path; import time; Path('partner-running').touch(); time.sleep(30)\""}))
        pytest.fail('The CEO must not start the closing model request after the analyst fault')

    with make_team(mode, respond, 'relations-publication-' + boundary) as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        declare(team, 'growth', 'A reusable conclusion.', ['unknown: initial material'])
        checkpoint = team.checkpoint()
        registry = team.roles['growth'].registry.relations
        partner = team.roles['ops_finance']
        publications = fail_second_item(monkeypatch, registry, boundary,
            wait_for=partner.workspace / 'partner-running')
        outcome = team.run(stop_after_day=7)
        assert (outcome.reason, outcome.day) == ('paused', 0), team.failure
        assert team.failure == 'relation_persistence_unknown'
        assert len(publications) == 3
        receipt = registry.last_result
        assert receipt['status'] == 'hold'
        assert (receipt['submitted'], receipt['saved'], receipt['unknown'], receipt['not_attempted']) == (3, 1, 1, 1)
        assert [r['status'] for r in receipt['items']] == ['saved', 'unknown', 'not_attempted']
        persisted = json.loads(registry.path.read_text())
        assert len(persisted['records']) == (1 if boundary == 'before_replace' else 2)
        call, = persisted['calls'].values()
        assert call['complete'] is False
        before = registry.path.read_bytes()
        rejected = team.roles['growth'].executor.execute('relation_batch', {'items': [item('after-hold')]},
                                                       tool_call_id='after-hold')
        assert rejected.startswith('Error:')
        assert registry.path.read_bytes() == before
        with pytest.raises(RunCancelled):
            team._check_model_send()
        with pytest.raises((RuntimeError, RunCancelled)):
            team.checkpoint()
        assert json.loads((team.root / 'checkpoint.json').read_text())['snapshot'] == str(checkpoint)
        assert team._pool is None
        assert all(not role.executor.preserved_process for role in team.roles.values())
        partner_audit = [r for r in rows(partner.audit.path) if r['kind'] == 'tool_execution' and r['tool'] == 'bash']
        assert partner_audit and partner_audit[-1]['process_started'] is True
        assert partner_audit[-1]['boundary_closed'] is True
        fault_audit = [r for r in rows(team.roles['growth'].audit.path) if r['kind'] == 'relation_batch_hold']
        assert fault_audit[-1]['receipt'] == receipt
        usage = {role: rows(runtime.usage.path) for role, runtime in team.roles.items()}
        for role in ('growth', 'ops_finance'):
            assert len([r for r in usage[role] if r['event'] == 'http_request']) == 1
        save_sample(mode + '-' + boundary, receipt=receipt, persisted=persisted, usage=usage,
                    fault_audit=fault_audit, partner_audit=partner_audit, checkpoint=str(checkpoint))


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_selected_checkpoint_keeps_relations_and_call_receipts_without_replaying_tail(tmp_path, mode):
    original = checkpoint_team(tmp_path / 'original', mode)
    restored = None
    try:
        declare(original, 'growth', 'A reusable conclusion.', ['unknown: initial material'])
        runtime = original.roles['growth']
        kept_args = {'items': [item('kept')]}
        kept = runtime.executor.execute('relation_batch', kept_args, tool_call_id='kept-call')
        assert json.loads(kept)['saved'] == 1
        checkpoint = original.checkpoint()
        later = runtime.executor.execute('relation_batch', {'items': [item('discarded')]}, tool_call_id='discarded-call')
        assert json.loads(later)['saved'] == 1
        original_state = runtime.registry.relations.path.read_bytes()
        restored = MultiAgentRuntime.restore(checkpoint, tmp_path / 'restored',
            client_factory=clients(), simulator_factory=fast_simulator)
        resumed = restored.roles['growth']
        state = resumed.registry.relations._load()
        assert len(state['records']) == 1 and len(state['calls']) == 1
        assert all('discarded-call' not in key for key in state['calls'])
        repeated = resumed.executor.execute('relation_batch', kept_args, tool_call_id='kept-call')
        assert json.loads(repeated) == json.loads(kept)
        history = json.loads(repeated)['items'][0]['history']
        assert history['path'] == 'relations.json'
        read = resumed.executor.execute('read_file', {'path': history['path']})
        assert 'kept' in read and 'discarded-call' not in read
        assert len(resumed.registry.relations._load()['records']) == 1
        assert restored.run(stop_after_day=7).reason == 'observation_end', restored.failure
        assert len(resumed.registry.relations._load()['records']) == 1
        assert runtime.registry.relations.path.read_bytes() == original_state
        save_sample(mode + '-restore', selected=str(checkpoint), original=json.loads(original_state),
                    restored=resumed.registry.relations._load(), repeated=json.loads(repeated))
    finally:
        if restored:
            dispose(restored)
        dispose(original)


def test_real_attempt_wrapper_persists_hold_and_manual_restore_preserves_failed_attempt(frozen, tmp_path, monkeypatch):
    path = changed_manifest(frozen, tmp_path,
        lambda manifest: [row.update(stop_after_day=7) for row in manifest['runs']])
    entry = next(row for row in batch.read(path)['runs'] if row['group'] == 'git')
    seen = Counter()
    declarations = dict(text='A reusable conclusion.', objects=[dict(kind='metric', id='cash')],
                        references=[], applies='0-', reason='Keep the conclusion.')

    def client(role):
        def respond(request):
            body = json.loads(request.content)
            if response := opening_response(role, body):
                return response
            seen[role] += 1
            if role == 'growth':
                assert seen[role] == 1
                message = calls(('text_create', declarations),
                    ('relation_batch', {'items': [item('first'), item('uncertain'), item('unattempted')]}))
            else:
                message = final('Existing material is sufficient.')
            return completion(message, body.get('stream', False))
        return batch.checked_client(role, 'relations-attempt-hold', transport=httpx.MockTransport(respond))

    publish = RelationRegistry._publish
    count = Counter()
    def fail(registry, state):
        if registry.role == 'growth':
            count[registry.workspace] += 1
            if count[registry.workspace] == 3:
                raise OSError('relation item publication unavailable')
        return publish(registry, state)

    def simulator(conn, config, rng):
        sim = fast_simulator(conn, config, rng)
        sim.shock_manager = batch.simulator(conn, config, rng).shock_manager
        return sim

    with monkeypatch.context() as failure:
        failure.setattr(RelationRegistry, '_publish', fail)
        result = batch.prepare(path, entry['run_id'], live=True, client_factory=client, simulator_factory=simulator)
    assert result['status'] == 'paused' and result['hold'] is True
    assert result['runtime_failure'] == 'relation_persistence_unknown'
    original, = batch.attempts(entry)
    record = batch.read(original / 'attempt.json')
    assert record['resource_cleanup']['status'] == 'closed'
    assert record['recovery_validation']['eligible'] is True
    hold = batch.read(original / 'HOLD')
    assert hold['manual_resume_required'] and hold['reason'] == 'relation_persistence_unknown'
    source = Path(record['checkpoint'])
    state = batch.read(original / 'runtime/roles/growth/relations.json')
    assert len(state['records']) == 1
    preserved = {name: (original / name).read_bytes() for name in
        ('attempt.json', 'HOLD', 'runtime/roles/growth/relations.json', 'runtime/private/growth/model-usage.jsonl')}
    monkeypatch.setenv('CEOBENCH_RECOVERY_CHECKPOINTS', json.dumps({entry['run_id']: str(source)}))
    resumed = batch.prepare(path, entry['run_id'], resume=True, live=True,
        client_factory=clients(), simulator_factory=simulator)
    assert resumed['status'] == 'finished' and resumed['day'] == 7
    assert resumed['source_checkpoint'] == str(source)
    assert len(batch.attempts(entry)) == 2
    assert all((original / name).read_bytes() == value for name, value in preserved.items())
    new_relations = batch.attempts(entry)[-1] / 'runtime/roles/growth/relations.json'
    assert not new_relations.exists()
    save_sample('attempt-hold-restore', first=record, hold=hold, saved_state=state, resumed=resumed)
