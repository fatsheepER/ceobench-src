"""Pipeline refusal advice through Bash, model delivery, and text registration."""
from contextlib import closing
import json
import os
from pathlib import Path
import shlex

import pytest

from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
from saas_bench.execution_capture import CapturedText, model_request, text_sources
from saas_bench.registration_evidence import HANDLES
from saas_bench.role_policy import AgentIdentity
from saas_bench.sql_evidence import SQLEvidenceStore, encoded
from saas_bench.text_registry import TextRegistry
from test_sql_evidence import identity
from test_text_registry import captured, declaration, send, workspace


@pytest.fixture
def d63(workspace, tmp_path):
    fixture = json.loads((Path(__file__).parent / 'fixtures/pf_pipeline_d63_growth.json').read_text())
    actor = AgentIdentity('test-run', 'world-1', 'growth', 'growth-agent', 'growth-session')
    store = SQLEvidenceStore(tmp_path / 'private/evidence.sqlite',
        identity('growth', capture_scope='execution') | actor.fields())
    registry = TextRegistry(workspace, 'pf', store, sim_day=lambda: 63, identity=actor)
    executor = BashAgentToolExecutor(workspace, evidence_store=store, text_registry=registry,
        identity=actor, require_sandbox=True)
    event = store.begin(encoded({'sql': fixture['sql']}), 'public-sql-v1')
    store.finish(event, 200, '', fixture['response'].encode(), {'day': 63})
    store.delivered(event, 'sent')
    key = encoded(['test-run', 'world-1', 'public-sql-v1', fixture['sql'], None]).decode()
    store.save_state(HANDLES, {'query': {key: 195}})
    assert registry.resolver.handle(event + ':public_response') == fixture['handle']
    return fixture, store, registry, executor


def advice(receipt):
    return receipt.split('```bash\n', 1)[1].split('\n```', 1)[0]


def submit(executor, store, output, arguments, context='week'):
    request = dict(messages=[dict(role='tool', content=output)])
    event = model_request(store, encoded(request), text_sources(request), 'offline', 'offline', context)
    store.complete(event, send_state='response_received')
    return executor.execute('text_create', arguments, model_request_event=event, model_context_id=context)


@pytest.mark.parametrize('pipe', ['2>&1 | tail -20', '| cat', '| head -1'])
def test_d63_pipe_receipt_read_and_identical_registration(d63, pipe):
    fixture, store, registry, executor = d63
    arguments = declaration(references=[fixture['reference']])
    result = executor.execute('bash', {'command': 'pf show ' + fixture['handle'] + ' ' + pipe})
    assert (fixture['response'] in result) is ('head' not in pipe)
    piped_request = send(store, result)
    rejected = executor.execute('text_create', arguments,
        model_request_event=piped_request, model_context_id='week')
    assert 'pf_pipeline_source_ranges_unknown' in rejected, rejected
    assert 'displayed' in rejected and 'source ranges' in rejected and 'Selected fields' in rejected
    assert not registry.path.exists()
    command = advice(rejected)
    assert command == 'pf show --full -- growth:query195@v1'
    full = executor.execute('bash', {'command': command})
    full_request = send(store, full)
    accepted = executor.execute('text_create', arguments,
        model_request_event=full_request, model_context_id='week')
    assert accepted.startswith('Registered r1.1 (active).'), accepted
    binding = store.load_state('declaration:r1.1')['references'][0]
    assert sorted(binding['selected_ranges']) == [[73, 78], [80, 99], [130, 136], [138, 142]]
    store.assert_healthy()
    if directory := os.environ.get('CEOBENCH_PF_PIPE_ARTIFACTS'):
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        suffix = 'tail' if 'tail' in pipe else 'head' if 'head' in pipe else 'cat'
        (target / ('d63-' + suffix + '-replay.json')).write_text(json.dumps(dict(
            source=fixture['source'], original_command=fixture['command'], tested_pipe=pipe,
            piped_output=str(result), rejection=str(rejected), command=command,
            full_output=str(full), registration=str(accepted), binding=binding,
            requests={kind: dict(event=event, result=store.read_event(event)['result'],
                wire=json.loads(store.get_content(event + ':wire')[1]),
                occurrences=json.loads(store.get_content(event + ':occurrences')[1]))
                for kind, event in [('piped', piped_request), ('standalone', full_request)]}),
            ensure_ascii=False, indent=2))


def test_pipeline_advice_is_limited_to_originating_request_and_target(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    (workspace / 'other.json').write_text('{"n":8}')
    executor.execute('bash', {'command': 'true'})
    result = executor.execute('bash', {'command': 'pf show evidence.json | cat'})
    arguments = declaration(references=[dict(cite='other.json', select={'path': '/n'})])
    rejected = submit(executor, store, result, arguments)
    assert rejected.startswith('Error: Evidence has not been delivered'), rejected
    assert 'pf_pipeline_source_ranges_unknown' not in rejected
    result = executor.execute('bash', {'command': 'echo next-context'})
    rejected = submit(executor, store, result, declaration(references=[dict(
        cite='evidence.json', select={'path': '/n'})]), context='next-week')
    assert rejected.startswith('Error: Evidence has not been delivered'), rejected
    assert 'pf_pipeline_source_ranges_unknown' not in rejected
    assert not registry.path.exists()
    store.assert_healthy()


def test_pipeline_advice_does_not_change_whole_reference_acceptance(workspace, tmp_path):
    store, _, executor = captured(workspace, tmp_path)
    executor.execute('bash', {'command': 'true'})
    result = executor.execute('bash', {'command': 'pf show evidence.json | cat'})
    accepted = submit(executor, store, result, declaration(references=[dict(cite='evidence.json@v1')]))
    assert accepted.startswith('Registered r1.1 (active).'), accepted
    binding = store.load_state('declaration:r1.1')['references'][0]
    assert binding['basis'] == 'explicit_handle' and binding['reading_scope'] == 'not_in_request'
    store.assert_healthy()


def test_pipeline_advice_does_not_blame_unrelated_child(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    (workspace / 'large.json').write_text('{"padding":"' + 'x' * 25000 + '","n":7}')
    executor.execute('bash', {'command': 'true'})
    result = executor.execute('bash', {'command': 'pf show large.json; pf show evidence.json | cat'})
    rejected = submit(executor, store, result, declaration(references=[dict(
        cite='large.json', select={'path': '/n'})]))
    assert rejected.startswith('Error: Selected evidence was not fully delivered'), rejected
    assert 'pf_pipeline_source_ranges_unknown' not in rejected
    assert not registry.path.exists()
    store.assert_healthy()


def test_pipeline_advice_quotes_exact_file_handle(workspace, tmp_path):
    store, _, executor = captured(workspace, tmp_path)
    path = '-facts name;$(touch injected).json'
    (workspace / path).write_text('{"n":7}')
    executor.execute('bash', {'command': 'true'})
    result = executor.execute('bash', {'command': shlex.join(['pf', 'show', '--', path]) + ' | cat'})
    arguments = declaration(references=[dict(cite=path, select={'path': '/n'})])
    rejected = submit(executor, store, result, arguments)
    command = advice(rejected)
    assert command == "pf show --full -- '-facts name;$(touch injected).json@v1'"
    full = executor.execute('bash', {'command': command})
    accepted = submit(executor, store, full, arguments)
    assert accepted.startswith('Registered r1.1 (active).'), accepted
    assert not (workspace / 'injected').exists()
    store.assert_healthy()


@pytest.mark.parametrize('missing', ['parent_sink', 'child_sink', 'read_metadata'])
def test_old_pipeline_capture_without_optional_metadata_keeps_refusal(workspace, tmp_path, missing):
    store, registry, executor = captured(workspace, tmp_path)
    executor.execute('bash', {'command': 'true'})
    result = executor.execute('bash', {'command': 'pf show evidence.json | cat'})
    parent = store.get_content(result.origins[0]['version_id'])[0]['created_by_event']
    child = store.read_event(parent)['result']['pf_calls'][0]['event_id']
    if missing == 'read_metadata':
        result = CapturedText(result, result.origins)
    request = send(store, result)
    with closing(store.connect()) as conn, conn:
        if missing == 'read_metadata':
            conn.execute('DELETE FROM versions WHERE version_id=?', (child + ':read_full',))
        else:
            event = parent if missing == 'parent_sink' else child
            facts = store.read_event(event)['result']
            facts.pop('stdout_sink')
            conn.execute('UPDATE results SET record=? WHERE event_id=?', (encoded(facts), event))
    rejected = executor.execute('text_create', declaration(references=[dict(
        cite='evidence.json', select={'path': '/n'})]), model_request_event=request, model_context_id='week')
    assert rejected.startswith('Error: Evidence has not been delivered'), rejected
    assert 'pf_pipeline_source_ranges_unknown' not in rejected
    assert not registry.path.exists()
    store.assert_healthy()


def test_pipeline_receipt_preserves_captured_read_integrity_gate(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    executor.execute('bash', {'command': 'true'})
    result = executor.execute('bash', {'command': 'pf show evidence.json | cat'})
    parent = store.get_content(result.origins[0]['version_id'])[0]['created_by_event']
    child = store.read_event(parent)['result']['pf_calls'][0]['event_id']
    request = send(store, result)
    with closing(store.connect()) as conn, conn:
        conn.execute('UPDATE blobs SET content=? WHERE content_hash=(SELECT content_hash FROM versions '
                     'WHERE version_id=?)', (b'corrupt', child + ':read_full'))
    rejected = executor.execute('text_create', declaration(references=[dict(
        cite='evidence.json', select={'path': '/n'})]), model_request_event=request, model_context_id='week')
    assert rejected == 'Error: Captured evidence is missing or corrupt; collection stopped'
    assert not registry.path.exists()
    with pytest.raises(RuntimeError, match='SQL evidence capture failed; collection stopped'):
        executor.execute('bash', {'command': 'true'})
