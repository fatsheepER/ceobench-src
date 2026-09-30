"""Shared declaration behavior using real Git, captured reads, and model requests."""
import json
import re
from pathlib import Path
import shutil
import subprocess

import pytest

from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor, get_bash_agent_tool_descriptions
from saas_bench.execution_capture import model_request, text_sources
from saas_bench.sql_evidence import SQLEvidenceStore
from saas_bench.text_registry import TextRegistry
from test_sql_evidence import identity
from test_public_sql import server
from test_preflight_integration import offline_runner, packed_public, advance


def git(ws, *args):
    return subprocess.check_output(['git', '-C', str(ws), *args], text=True).strip()


@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / 'agent_workspace'
    ws.mkdir()
    git(ws, 'init', '-q')
    git(ws, 'config', 'user.email', 'fixture@example.invalid')
    git(ws, 'config', 'user.name', 'Fixture')
    (ws / 'evidence.json').write_text('{"n":7}')
    git(ws, 'add', '.')
    git(ws, 'commit', '-qm', 'initial')
    return ws


def declaration(evidence=None, **changes):
    data = dict(text='Keep the plan', objects=[dict(kind='plan', id='B')],
                applies_at={'start_day': 7}, reason='Initial observation',
                references=[dict(evidence=evidence or {'unknown': 'No saved evidence'}, purpose='current')])
    data.update(changes)
    return data


def call(registry, op, **args):
    text = registry.execute(op, args)
    return json.loads(text) if op == 'list' else registry.last_result


RECEIPT = re.compile(r'(Registered|Revised|Retired) (r[1-9][0-9]*)\.([1-9][0-9]*) \((active|retired)\)\.')


def receipt(text):
    """id, version and status of a text receipt, e.g. 'Registered r1.1 (active).'."""
    match = RECEIPT.match(text)
    assert match, text
    return dict(id=match.group(2), version=f'{match.group(2)}.{match.group(3)}', status=match.group(4))


def send(store, text, state='response_received'):
    request = dict(messages=[dict(role='tool', content=text)])
    event = model_request(store, json.dumps(request).encode(), text_sources(request), 'call', 'attempt', 'week')
    store.complete(event, send_state=state)
    return event


def captured(workspace, tmp_path, mode='pf'):
    store = SQLEvidenceStore(tmp_path / 'private/evidence.sqlite', identity(capture_scope='execution'))
    registry = TextRegistry(workspace, mode, store, sim_day=lambda: 7)
    executor = BashAgentToolExecutor(workspace, evidence_store=store, text_registry=registry)
    return store, registry, executor


def test_git_lifecycle_pins_references_preserves_history_and_never_copies(workspace):
    registry = TextRegistry(workspace, 'git', sim_day=lambda: 7)
    head = git(workspace, 'rev-parse', 'HEAD')
    created = call(registry, 'create', **declaration({'path': 'evidence.json'}))
    assert created == dict(id='r1', version='r1.1', status='active')
    original = json.loads(registry.path.read_text())['records']['r1'][0]
    # A bare path waits for the weekly commit that closes day 7's week.
    assert original['references'][0]['evidence'] == dict(path='evidence.json', commit='week-2')
    assert head not in registry.path.read_text() and '"n":7' not in registry.path.read_text()
    assert git(workspace, 'rev-parse', 'HEAD') == head
    assert git(workspace, 'diff', '--cached', '--name-only') == ''
    (workspace / 'evidence.json').write_text('{"n":99}')
    git(workspace, 'add', 'evidence.json')
    git(workspace, 'commit', '-qm', 'new evidence')
    call(registry, 'revise', record='r1', text='Revised', reason='Correction')
    state = json.loads(registry.path.read_text())['records']['r1']
    assert state[0] == original
    assert state[1]['references'] == original['references']
    assert call(registry, 'list')['records'][0]['text'] == 'Revised'
    call(registry, 'retire', record='r1', reason='No longer useful')
    assert call(registry, 'list') == dict(records=[], next_after=None)
    assert len(json.loads(registry.path.read_text())['records']['r1']) == 3
    with pytest.raises(ValueError, match='retired'):
        call(registry, 'revise', record='r1', reason='retry')


def test_bare_path_binds_the_weekly_commit_and_week_labels_resolve_after_it(workspace):
    from saas_bench.agents.bash_agent.run_test import BashAgentRunner
    from saas_bench.registration_evidence import week_commit_subject
    registry = TextRegistry(workspace, 'git', sim_day=lambda: 20)
    (workspace / 'plan.json').write_text('{"v":1}')
    (workspace / '.gitignore').write_text('*.db\n')
    (workspace / 'cache.db').write_text('x')
    call(registry, 'create', **declaration({'path': 'plan.json'}))
    assert call(registry, 'list')['records'][0]['references'][0]['evidence'] == dict(path='plan.json', commit='week-3')
    for evidence in ({'path': 'cache.db'}, {'path': 'missing.json'}, {'path': 'plan.json@week-3'}):
        with pytest.raises(ValueError):
            call(registry, 'create', **declaration(evidence))
    (workspace / 'plan.json').write_text('{"v":2}')  # later edits in the week belong to the binding
    runner = BashAgentRunner.__new__(BashAgentRunner)
    runner.agent_workspace = workspace
    runner._commit_weeks_up_to(21)
    subjects = git(workspace, 'log', '--format=%s').splitlines()
    assert week_commit_subject('week-3') in subjects
    assert git(workspace, 'show', 'HEAD:plan.json') == '{"v":2}'
    call(registry, 'create', **declaration({'path': 'plan.json@week-3'}))
    call(registry, 'create', **declaration({'path': 'plan.json', 'commit': 'week-2'}))
    refs = [r['references'][0]['evidence'] for r in call(registry, 'list')['records']]
    assert refs[1] == dict(path='plan.json', commit='week-3') and refs[2]['commit'] == 'week-2'


def test_direct_registered_text_reference_before_commit_and_pagination(workspace):
    registry = TextRegistry(workspace, 'git')
    call(registry, 'create', **declaration())
    call(registry, 'create', **declaration({'record': 'r1'}))
    call(registry, 'revise', record='r1', text='New assumption', reason='Changed')
    page = call(registry, 'list', limit=1)
    assert page['records'][0]['version'] == 'r1.2' and page['next_after'] == 1
    next_page = call(registry, 'list', after=page['next_after'], limit=1)
    assert next_page['records'][0]['references'][0]['evidence'] == {'record': 'r1.1'}
    assert next_page['next_after'] is None
    restored = TextRegistry(workspace, 'git')
    assert call(restored, 'list') == call(registry, 'list')


def test_git_unique_short_prefix_unknown_and_no_partial_mutation(workspace):
    registry = TextRegistry(workspace, 'git')
    head = git(workspace, 'rev-parse', 'HEAD')
    call(registry, 'create', **declaration({'path': 'evidence.json@' + head[:1]}))
    before = registry.path.read_bytes()
    for evidence in ({'path': 'uncommitted.json'}, {'path': '../secret'}, {'sql': 'SELECT 1'},
                     {'path': 'evidence.json', 'commit': 'HEAD~1'}, {'record': 'r1.99'}):
        with pytest.raises(ValueError):
            call(registry, 'create', **declaration(evidence))
        assert registry.path.read_bytes() == before
    created = call(registry, 'create', **declaration({'unknown': 'File is not committed yet'}))
    assert created['id'] == 'r2'


def test_notes_and_predicate_fields_are_saved_without_evaluation(workspace):
    registry = TextRegistry(workspace, 'git')
    ref = dict(evidence={'path': 'evidence.json'}, purpose='historical_only', select={'path': '/n'},
               predicate={'type': 'threshold', 'op': '>=', 'value': 99999}, note='中🙂' * 101)
    result = call(registry, 'create', **declaration(references=[ref]))
    assert result['warnings'] == ['Note truncated to 200 characters.']
    saved = call(registry, 'list')['records'][0]['references'][0]
    assert len(saved['note']) == 200 and saved['predicate']['value'] == 99999
    for changes in ({'select': {'row': 3, 'col': 'n'}}, {'predicate': {'type': 'eval', 'code': 'x'}},
                    {'purpose': 'reverse'}, {'select': None}):
        with pytest.raises(ValueError):
            call(registry, 'create', **declaration(references=[dict(ref, **changes)]))


def test_pf_requires_actual_send_uses_delivered_version_and_preserves_binding(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    text = executor.execute('read_file', {'path': 'evidence.json'})
    with pytest.raises(ValueError, match='not been delivered'):
        call(registry, 'create', **declaration({'path': 'evidence.json'}))
    request = send(store, text)
    # Bytes from another program: captured at the next Bash boundary but never sent.
    (workspace / 'evidence.json').write_text('{"n":8}')
    executor.execute('bash', {'command': 'true'})
    result = call(registry, 'create', **declaration({'path': 'evidence.json'}))
    assert result['evidence'][0] == dict(version='evidence.json@v1', latest='evidence.json@v2', differs=True)
    binding = store.load_state('declaration:r1.1')['references'][0]
    assert binding['delivered_in'][0]['request_event'] == request
    assert store.get_content(binding['version_id'])[1] == b'{"n":7}'
    assert store.get_content(binding['latest_version_id'])[1] == b'{"n":8}'
    newest = executor.execute('read_file', {'path': 'evidence.json'})
    send(store, newest)
    call(registry, 'revise', record='r1', reason='Text only', text='Clarified')
    assert store.load_state('declaration:r1.2')['references'] == [binding]
    call(registry, 'revise', record='r1', reason='New evidence', references=declaration({'path': 'evidence.json'})['references'])
    assert store.load_state('declaration:r1.3')['references'][0]['version_id'] != binding['version_id']
    with pytest.raises(ValueError, match='or a file path'):
        call(registry, 'create', **declaration({'version': 'v999'}))


def test_pf_partial_read_selectors_and_latest_delivery_not_latest_capture(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    (workspace / 'evidence.json').write_text('{\n"n": 7,\n"hidden": 42\n}')
    text = executor.execute('read_file', {'path': 'evidence.json', 'offset': 2, 'limit': 1})
    send(store, text)
    ref = dict(evidence={'path': 'evidence.json'}, purpose='current', select={'path': '/n'})
    call(registry, 'create', **declaration(references=[ref]))
    for selection in ({'path': '/hidden'}, None):
        with pytest.raises(ValueError, match='not fully delivered'):
            call(registry, 'create', **declaration(references=[dict(ref, select=selection)]))
    # An unconfirmed HTTP attempt does not establish delivery.
    full = executor.execute('read_file', {'path': 'evidence.json'})
    send(store, full, state='unknown')
    with pytest.raises(ValueError, match='not fully delivered'):
        call(registry, 'create', **declaration({'path': 'evidence.json'}))
    send(store, full)
    call(registry, 'create', **declaration({'path': 'evidence.json'}))


def test_pf_registered_text_cites_directly_like_git_and_retirement_keeps_old_reference(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    call(registry, 'create', **declaration())
    # Design 3.2: registered texts cite each other directly in both groups, unread.
    result = call(registry, 'create', **declaration({'record': 'r1'}))
    binding = store.load_state('declaration:r2.1')['references'][0]
    assert binding['status'] == 'resolved' and binding['basis'] == 'registered_text'
    assert json.loads(store.get_content(binding['version_id'])[1])['version'] == 'r1.1'
    assert result['evidence'][0]['version'] == 'r1.1'  # a registered text's handle is its revision
    with pytest.raises(ValueError, match='whole-text'):
        call(registry, 'create', **declaration(references=[dict(evidence={'record': 'r1.1'}, purpose='current',
                                                                select={'path': '/text'})]))
    with pytest.raises(ValueError, match='Unknown registered text'):
        call(registry, 'create', **declaration({'record': 'r1.9'}))
    call(registry, 'create', **declaration({'record': 'r1.1'}))
    call(registry, 'retire', record='r1', reason='Superseded')
    assert call(registry, 'list')['records'][0]['references'][0]['evidence'] == {'record': 'r1.1'}
    assert len(json.loads(registry.path.read_text())['records']['r1']) == 2


def test_pf_files_the_model_wrote_count_as_known_but_script_outputs_do_not(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    executor.execute('write_file', {'path': 'plan.json', 'content': '{"price": 99}'})
    ref = dict(evidence={'path': 'plan.json'}, purpose='current', select={'path': '/price'},
               predicate=dict(type='threshold', op='>=', value=90))
    call(registry, 'create', **declaration(references=[ref]))
    binding = store.load_state('declaration:r1.1')['references'][0]
    assert binding['status'] == 'resolved' and binding['authored_by'] and binding['delivered_in'] == []
    assert store.get_content(binding['version_id'])[1] == b'{"price": 99}'
    # A quoted heredoc writes exactly the text in the model's command.
    executor.execute('bash', {'command': "cat > setup.py <<'EOF'\nPRICE = 9\nEOF"})
    call(registry, 'create', **declaration({'path': 'setup.py'}))
    assert store.load_state('declaration:r2.1')['references'][0]['authored_by']
    # A file computed by a program was neither read nor written by the model.
    executor.execute('bash', {'command': 'python3 -c "open(\'out.txt\', \'w\').write(str(6 * 7))"'})
    with pytest.raises(ValueError, match='neither read nor wrote'):
        call(registry, 'create', **declaration({'path': 'out.txt'}))
    # A later write by another program: the model's own earlier version stays the binding.
    (workspace / 'plan.json').write_text('{"price": 79}')
    executor.execute('bash', {'command': 'true'})
    call(registry, 'create', **declaration({'path': 'plan.json'}))
    binding = store.load_state('declaration:r3.1')['references'][0]
    assert store.get_content(binding['version_id'])[1] == b'{"price": 99}'
    assert binding['version_id'] != binding['latest_version_id']


def test_pf_frozen_old_memory_does_not_hide_a_new_authored_version(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    (workspace / 'MEMORY.md').write_text('Old plan\nOld detail\n')
    frozen = executor.execute('read_file', {'path': 'MEMORY.md', 'limit': 1})
    send(store, frozen)
    current = 'New plan\nNew detail\n'
    executor.execute('write_file', {'path': 'MEMORY.md', 'content': current})
    # Every subsequent request repeats the frozen week-start MEMORY.
    send(store, frozen)
    call(registry, 'create', **declaration({'path': 'MEMORY.md'}))
    binding = store.load_state('declaration:r1.1')['references'][0]
    assert binding['authored_by'] and store.get_content(binding['version_id'])[1] == current.encode()
    # A partial reread of these same authored bytes cannot erase that knowledge.
    send(store, executor.execute('read_file', {'path': 'MEMORY.md', 'limit': 1}))
    call(registry, 'create', **declaration({'path': 'MEMORY.md'}))
    assert store.load_state('declaration:r2.1')['references'][0]['version_id'] == binding['version_id']
    # A genuinely newer, externally changed version must still be read in full.
    (workspace / 'MEMORY.md').write_text('External plan\nUnread detail\n')
    send(store, executor.execute('read_file', {'path': 'MEMORY.md', 'limit': 1}))
    with pytest.raises(ValueError, match='not fully delivered'):
        call(registry, 'create', **declaration({'path': 'MEMORY.md'}))


def test_pf_local_edit_is_immediately_citable_by_path_and_handle(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    executor.execute('write_file', dict(path='plan.txt', content='budget 100\nprice 20\n'))
    old = executor.execute('read_file', dict(path='plan.txt', limit=1))
    edited = executor.execute('edit_file', dict(path='plan.txt', old_string='100', new_string='200'))
    assert 'wrote plan.txt@v2' in edited
    send(store, old)  # Repeated frozen context must not hide the edit.
    for evidence in ({'path': 'plan.txt'}, {'version': 'plan.txt@v2'}):
        created = call(registry, 'create', **declaration(evidence))
        binding = store.load_state('declaration:' + created['version'])['references'][0]
        assert binding['version_id'] == binding['latest_version_id']
        assert store.get_content(binding['version_id'])[1] == b'budget 200\nprice 20\n'
        assert store.read_event(binding['authored_by'])['request']['kind'] == 'edit_file'
    (workspace / 'plan.txt').write_text('external 300\nprice 20\n')
    assert executor.execute('edit_file', dict(path='plan.txt', old_string='absent', new_string='x')).startswith('Error:')
    with pytest.raises(ValueError, match='neither read nor wrote'):
        call(registry, 'create', **declaration({'version': 'plan.txt@v3'}))


@pytest.mark.parametrize('delivered', [False, True])
def test_prefix_public_state_and_returns_do_not_disclose_private_resolution(workspace, tmp_path, delivered):
    store, prefix, executor = captured(workspace, tmp_path, 'prefix')
    if delivered:
        send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    result = prefix.execute('create', declaration({'path': 'evidence.json'}))
    public = prefix.path.read_bytes()
    prefix.path.unlink()
    baseline = TextRegistry(workspace, 'git', sim_day=lambda: 7)
    assert baseline.execute('create', declaration({'path': 'evidence.json'})) == result
    assert baseline.path.read_bytes() == public
    private = store.load_state('declaration:r1.1')['references'][0]
    assert private['status'] == ('resolved' if delivered else 'unknown')
    assert b'delivered_in' not in public and b'version_id' not in public


def test_fork_inherits_prefix_bindings_but_not_sibling_versions(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path, 'prefix')
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration({'path': 'evidence.json'}))
    receipt = store.snapshot(tmp_path / 'clone.sqlite')
    branch = SQLEvidenceStore(tmp_path / 'clone.sqlite', identity('pf', capture_scope='execution',
                              parent_branch='prefix', fork_seq=receipt['cutoff']))
    clone = TextRegistry(workspace, 'pf', branch)
    call(clone, 'revise', record='r1', reason='Post-fork revision', text='New text')
    assert branch.load_state('declaration:r1.2')['references'] == store.load_state('declaration:r1.1')['references']
    inherited = branch.load_state('declaration:r1.1')['references'][0]['version_id']
    assert branch.get_content(inherited)[1] == b'{"n":7}'
    assert store.load_state('declaration:r1.2') is None


def test_storage_escape_and_tool_availability(workspace, tmp_path):
    outside = tmp_path / 'outside.json'
    outside.write_text('private')
    (workspace / 'registrations.json').symlink_to(outside)
    with pytest.raises(ValueError, match='symlink'):
        call(TextRegistry(workspace, 'git'), 'create', **declaration())
    assert outside.read_text() == 'private'
    plain = BashAgentToolExecutor(workspace)
    assert plain.execute('text_list', {}).startswith('Error: Unknown tool')
    assert len(get_bash_agent_tool_descriptions()) == 6
    assert len(get_bash_agent_tool_descriptions(True)) == 10
    bad_store = SQLEvidenceStore(workspace / 'private.sqlite', identity(capture_scope='execution'))
    with pytest.raises(ValueError, match='outside'):
        TextRegistry(workspace, 'prefix', bad_store)


def test_real_cli_query_projection_to_model_and_declared_comparison(workspace, tmp_path, server):
    from test_sql_evidence import settled
    store, registry, executor = captured(workspace, tmp_path)
    server.sql_evidence = store
    server.start()
    source = Path(__file__).parents[1] / 'src/saas_bench'
    shutil.copytree(source / 'novamind_api', workspace / 'novamind_api', ignore=shutil.ignore_patterns('__pycache__'))
    # Execute the actual CLI query handler, with only its imports adapted for a
    # standalone fixture. The normal public build compiles this same handler.
    cli = (source / '_public_cli.py').read_text().replace('from .novamind_api', 'from novamind_api')
    (workspace / 'cli_fixture.py').write_text(cli)
    executor.extra_env = {'NOVAMIND_API_PORT': str(server.port)}
    sql = "SELECT 'search' AS channel, 2 AS cost UNION ALL SELECT 'social', 4"
    import shlex
    command = 'python -c ' + shlex.quote('from cli_fixture import cmd_query; from argparse import Namespace; cmd_query(Namespace(session=None, sql=' + repr(sql) + '))')
    result = executor.execute('bash', {'command': command})
    assert json.loads(result.rsplit('\n[', 1)[0])['row_count'] == 2
    settled(server)
    send(store, result)
    ref = dict(evidence={'sql': sql}, purpose='current', predicate=dict(type='compare',
               left={'row': {'channel': 'search'}, 'col': 'cost'}, op='<',
               right={'row': {'channel': 'social'}, 'col': 'cost'}))
    registered = call(registry, 'create', **declaration(references=[ref]))
    assert re.fullmatch(r'query\d+@v1', registered['evidence'][0]['version'])
    binding = store.load_state('declaration:r1.1')['references'][0]
    assert binding['version_id'].endswith(':public_response')
    assert store.read_event(binding['version_id'].rsplit(':', 1)[0])['request']['candidate_sql'] == sql
    # Equal text from another command cannot stand in for a redirected query.
    redirected_sql = 'SELECT 456 AS n'
    cli_code = 'from cli_fixture import cmd_query; from argparse import Namespace; cmd_query(Namespace(session=None, sql=' + repr(redirected_sql) + '))'
    echo_code = 'print(' + repr(json.dumps({'columns': ['n'], 'rows': [{'n': 456}], 'row_count': 1}, indent=2)) + ')'
    output = executor.execute('bash', {'command': 'python -c ' + shlex.quote(cli_code) + ' > /dev/null; python -c ' + shlex.quote(echo_code)})
    send(store, output)
    with pytest.raises(ValueError, match='not been delivered'):
        call(registry, 'create', **declaration({'sql': redirected_sql}))
    # SDK use with no observed projection cannot turn arbitrary printed text into SQL evidence.
    hidden_sql = 'SELECT 123 AS n'
    output = executor.execute('bash', {'command': 'python -c ' + shlex.quote(
        'import novamind_api as n; n.query(' + repr(hidden_sql) + '); print("123")')})
    send(store, output)
    with pytest.raises(ValueError, match='not been delivered'):
        call(registry, 'create', **declaration({'sql': hidden_sql}))


@pytest.mark.parametrize('api', ['chat', 'responses', 'messages'])
def test_all_model_apis_receive_shared_tools_and_original_prompt_is_unchanged(workspace, api):
    import httpx
    from openai import OpenAI
    from anthropic import Anthropic
    from saas_bench.agents.bash_agent.agent import BashAgent
    from saas_bench.registration_prompt import integrate
    from test_preflight_usage import reply
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        body = reply(api)
        if api == 'messages':
            body['content'] = []
            events = [dict(type='message_start', message=body),
                dict(type='content_block_start', index=0, content_block=dict(type='tool_use', id='call1', name='text_list', input={})),
                dict(type='content_block_delta', index=0, delta=dict(type='input_json_delta', partial_json='{}')),
                dict(type='content_block_stop', index=0),
                dict(type='message_delta', delta={'stop_reason': 'tool_use', 'stop_sequence': None}, usage={'output_tokens': 2}),
                dict(type='message_stop')]
            stream = ''.join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
            return httpx.Response(200, content=stream, headers={'content-type': 'text/event-stream'})
        if api == 'chat':
            body['choices'][0]['message']['tool_calls'] = [dict(id='call1', type='function', function=dict(name='text_list', arguments='{}'))]
        else:
            body['output'] = [dict(type='function_call', id='fc1', call_id='call1', name='text_list', arguments='{}', status='completed')]
        return httpx.Response(200, json=body)
    with (Anthropic if api == 'messages' else OpenAI)(api_key='offline-only', max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        original = BashAgent(get_bash_agent_tool_descriptions(), client, workspace_path=workspace)
        enhanced = BashAgent(get_bash_agent_tool_descriptions(True), client, workspace_path=workspace,
                             text_registration=True, reasoning_effort='low' if api == 'responses' else None)
        assert enhanced.system_prompt == integrate(original.system_prompt, pf=False)
        assert enhanced.act('dashboard', 0, False, {'day': 0}).tool == 'text_list'
        request = requests[-1]
        tools = [t.get('function', t) for t in request['tools']]
        assert [t['name'] for t in tools][-4:] == ['text_create', 'text_revise', 'text_retire', 'text_list']


def test_runner_registration_mode_is_frozen_on_resume(tmp_path, monkeypatch):
    from saas_bench.agents.bash_agent.run_test import BashAgentRunner
    monkeypatch.setattr('saas_bench.agents.bash_agent.run_test.load_env_file', lambda _: {})
    monkeypatch.setattr('saas_bench.run_state.verify_build', lambda *a, **k: {})
    monkeypatch.setenv('OPENCODE_API_KEY', 'offline-only')
    first = BashAgentRunner(provider='opencode', model='test-model', workspace_base=tmp_path, text_registration='prefix')
    first._prepare_manifest()
    assert first.execution_capture
    restored = BashAgentRunner(continue_from=first.workspace_dir)
    assert restored.text_registration == 'prefix'
    restored._prepare_manifest()
    with pytest.raises(ValueError, match='registration configuration mismatch'):
        BashAgentRunner(continue_from=first.workspace_dir, text_registration='pf')
    with pytest.raises(ValueError, match='requires execution capture'):
        BashAgentRunner(workspace_base=tmp_path, text_registration='prefix', execution_capture=False)
    first.client.close()
    restored.client.close()


def test_packed_prefix_registration_restore_and_git_pf_forks(offline_runner, tmp_path):
    from saas_bench.run_state import clone_sql_run, checkpoint_directory
    runner = offline_runner(text_registration='prefix')
    runner.agent.current_day = 0
    runner._execute_tool('write_file', {'path': 'facts.json', 'content': '{"n":7}'})
    # Keep this fixture synchronous: Git's detached maintenance is an open process boundary.
    result = runner._execute_tool('bash', {'command': 'git add facts.json && git -c maintenance.auto=false -c user.name=Fixture -c user.email=fixture@example.invalid commit -m facts'})
    assert '[exit code:' not in result
    send(runner.evidence_store, runner._execute_tool('read_file', {'path': 'facts.json'}))
    assert receipt(runner._execute_tool('text_create', declaration({'path': 'facts.json'})))['version'] == 'r1.1'
    page = runner._execute_tool('text_list', {})
    send(runner.evidence_store, page)
    runner._save_checkpoint(0)
    snapshot = checkpoint_directory(runner.workspace_dir, runner._load_checkpoint())
    assert (snapshot / 'agent_workspace/registrations.json').read_bytes() == (runner.agent_workspace / 'registrations.json').read_bytes()
    runner._stop_server()
    restored = offline_runner(runner.workspace_dir)
    assert restored.text_registration == 'prefix'
    assert restored._execute_tool('text_list', {}) == page
    # Group forks start at a completed week boundary (design 4.3).
    assert advance(restored)['success']
    restored._commit_weeks_up_to(7)
    restored._save_checkpoint(7)
    children = [offline_runner(clone_sql_run(restored.workspace_dir, tmp_path / mode, mode,
                                           text_registration=mode)) for mode in ('git', 'pf')]
    for child in children:
        result = receipt(child._execute_tool('text_create', declaration({'record': 'r1.1'})))
        assert result['id'] == 'r2'
        assert ('evidence' in child.tool_executor.text_registry.last_result) == (child.text_registration == 'pf')
        history = child._execute_tool('pf_read', {'target': {'record': 'r1.1'}})
        if child.text_registration == 'pf':
            assert json.loads(history.split('\n', 1)[1])['version'] == 'r1.1'
        else:
            assert history.startswith('Error: Unknown tool')
        # The only copy of resolved evidence identities is outside the workspace; a Git
        # branch neither captures nor receives the prefix evidence.
        assert not list(child.agent_workspace.rglob('*evidence.sqlite'))
        if child.text_registration == 'git':
            assert child.evidence_store is None and not list(child.workspace_dir.rglob('sql-evidence*'))
        private = restored.workspace_dir / 'sql-evidence.sqlite'
        assert '[exit code:' in child._execute_tool('bash', {'command': 'cat ' + str(private)})
        child._save_checkpoint(7)
    assert json.loads(restored._execute_tool('text_list', {}))['records'][0]['id'] == 'r1'
    import os
    if destination := os.environ.get('CEOBENCH_REGISTRATION_ARTIFACTS'):
        output = Path(destination)
        for branch in (restored, *children):
            target = output / branch.text_registration
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(branch.agent_workspace / 'registrations.json', target / 'registrations.json')
            if branch.evidence_store:
                branch.evidence_store.snapshot(target / 'evidence.sqlite')
        (output / 'validation.json').write_text(json.dumps(dict(
            evidence_kind='constructed_offline_integration', external_model_calls=0,
            stages=['packed CLI', 'captured file read', 'model send', 'declaration',
                    'checkpoint restore', 'git fork', 'pf fork', 'private storage isolation']), indent=2))


def test_private_capture_failure_stops_after_saved_declaration(workspace, tmp_path, monkeypatch):
    store, registry, executor = captured(workspace, tmp_path, 'prefix')
    save = store.save_state
    def fail(name, value):
        if name.startswith('declaration:'):
            raise OSError('injected storage fault')
        return save(name, value)
    monkeypatch.setattr(store, 'save_state', fail)
    result = executor.execute('text_create', declaration())
    assert result.startswith('Error: Text saved')
    assert json.loads(registry.path.read_text())['records']['r1'][0]['version'] == 'r1.1'
    with pytest.raises(RuntimeError, match='capture failed'):
        executor.execute('text_create', declaration())
    with pytest.raises(RuntimeError, match='capture failed'):
        store.snapshot(tmp_path / 'invalid-checkpoint.sqlite')


def test_csv_equality_keys_json_escapes_and_nonunique_rows(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    (workspace / 'data.csv').write_text('group,n\nA,7\nB,9\n')
    send(store, executor.execute('read_file', {'path': 'data.csv', 'limit': 2}))
    ref = dict(evidence={'path': 'data.csv'}, purpose='current', select={'row': {'group': 'A'}, 'col': 'n'})
    call(registry, 'create', **declaration(references=[ref]))
    with pytest.raises(ValueError, match='not fully delivered'):
        call(registry, 'create', **declaration(references=[dict(ref, select={'row': {'group': 'B'}, 'col': 'n'})]))
    (workspace / 'evidence.json').write_text('{"a/b":{"~key":7}}')
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration(references=[dict(evidence={'path': 'evidence.json'}, purpose='current', select={'path': '/a~1b/~0key'})]))
    (workspace / 'data.csv').write_text('group,n\nA,7\nA,9\n')
    send(store, executor.execute('read_file', {'path': 'data.csv'}))
    with pytest.raises(ValueError, match='exactly one row'):
        call(registry, 'create', **declaration(references=[ref]))


def test_prefix_records_git_bytes_mismatch_without_changing_public_reference(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path, 'prefix')
    (workspace / 'evidence.json').write_text('{"n":99}')
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    head = git(workspace, 'rev-parse', 'HEAD')[:7]
    result = call(registry, 'create', **declaration({'path': 'evidence.json@' + head}))
    assert 'evidence' not in result
    binding = store.load_state('declaration:r1.1')['references'][0]
    assert not binding['git_content_matches']
    assert store.get_content(binding['version_id'])[1] == b'{"n":99}'
    assert call(registry, 'list')['records'][0]['references'][0]['evidence']['commit'] == head
    # A bare path waits for the weekly commit; privately it binds the last delivered version.
    assert 'evidence' not in call(registry, 'create', **declaration({'path': 'evidence.json'}))
    binding = store.load_state('declaration:r2.1')['references'][0]
    assert binding['status'] == 'resolved' and binding['git_week'] == 'week-2' and 'git_content_matches' not in binding
    assert store.get_content(binding['version_id'])[1] == b'{"n":99}'
    assert call(registry, 'list')['records'][1]['references'][0]['evidence']['commit'] == 'week-2'


def test_applicability_accepts_open_ended_start_and_explains_shapes(workspace):
    registry = TextRegistry(workspace, 'git', sim_day=lambda: 7)
    # Both groups repeatedly sent {"start_day": N} meaning "until revised" in the free runs.
    call(registry, 'create', **declaration(applies_at={'start_day': 7}))
    assert call(registry, 'list')['records'][0]['applies_at'] == {'start_day': 7}
    for bad in ({'end_day': 7}, {'start_day': 9, 'end_day': 7}, {'day': 7, 'start_day': 7},
                {'start_day': 7, 'unknown': 'until revised'}):
        with pytest.raises(ValueError, match='until revised or retired') as error:
            call(registry, 'create', **declaration(applies_at=bad))
        assert '"21-27"' in str(error.value) and '"unknown: <reason>"' in str(error.value)
