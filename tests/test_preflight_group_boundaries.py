"""Independent acceptance of the stage 3 experiment group boundaries."""
import json
import os
from pathlib import Path
import re
import shutil

import pytest

from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor, get_bash_agent_tool_descriptions
from saas_bench.registration_evidence import HANDLES
from test_text_registry import captured, declaration, git, send, workspace
from test_public_sql import server
from test_preflight_integration import offline_runner, packed_public, advance


ORIGINAL = json.loads((Path(__file__).parent / 'fixtures/original_bash_agent.json').read_text())


def original_prompt(days):
    # Frozen inputs from the design's original source baseline, not the current
    # prompt generator. Keep the fixture independent of shallow Git checkouts.
    sim = ORIGINAL['simulator_instructions'].replace('{tool_list}\n', '').replace('{tool_list}', '')
    years = days / 365
    return (ORIGINAL['system_template'].replace('{simulator_instructions}', sim)
            .replace('{total_days}', str(days))
            # User-approved common repair: resolve the original week placeholder.
            .replace('{total_weeks}', str((days + 6) // 7))
            .replace('{total_years}', f'{years:.0f}' if years == int(years) else f'{years:.1f}'))


def save_artifact(name, value):
    if directory := os.environ.get('CEOBENCH_BOUNDARY_ARTIFACTS'):
        target = Path(directory) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


@pytest.mark.parametrize('days,weeks', [(7, 1), (14, 2), (42, 6), (497, 71), (730, 105), (3650, 522)])
def test_original_prompt_and_tools_match_frozen_baseline(days, weeks, monkeypatch):
    monkeypatch.delenv('ORACLE_MODE', raising=False)
    agent = BashAgent.__new__(BashAgent)
    agent.total_days = days
    assert f'{days} simulated days ({weeks} weeks /' in agent._default_system_prompt()
    assert f'Maximize cash over {weeks} weeks.' in agent._default_system_prompt()
    assert agent._default_system_prompt().encode() == original_prompt(days).encode()
    assert get_bash_agent_tool_descriptions() == [dict(type='function', **t) for t in ORIGINAL['tools']]


def tool_reply(api):
    import httpx
    from test_preflight_usage import reply
    body = reply(api)
    arguments = json.dumps({'path': 'MEMORY.md'})
    if api == 'chat':
        body['choices'][0]['message']['tool_calls'] = [dict(id='call1', type='function',
            function=dict(name='read_file', arguments=arguments))]
    elif api == 'responses':
        body['output'] = [dict(type='function_call', id='fc1', call_id='call1',
                              name='read_file', arguments=arguments, status='completed')]
    else:
        body['content'] = []
        events = [dict(type='message_start', message=body),
            dict(type='content_block_start', index=0, content_block=dict(type='tool_use', id='call1', name='read_file', input={})),
            dict(type='content_block_delta', index=0, delta=dict(type='input_json_delta', partial_json=arguments)),
            dict(type='content_block_stop', index=0),
            dict(type='message_delta', delta={'stop_reason': 'tool_use', 'stop_sequence': None}, usage={'output_tokens': 2}),
            dict(type='message_stop')]
        return httpx.Response(200, content=''.join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events),
                              headers={'content-type': 'text/event-stream'})
    return httpx.Response(200, json=body)


@pytest.mark.parametrize('api', ['chat', 'responses', 'messages'])
@pytest.mark.parametrize('mode', ['off', 'git', 'prefix', 'pf'])
def test_group_tools_and_memory_in_actual_requests(workspace, tmp_path, api, mode):
    import httpx
    from openai import OpenAI
    from anthropic import Anthropic
    from saas_bench.registration_schema import registration_prompt
    from saas_bench.pf_queries import PF_PROMPT, MODELS
    from saas_bench.model_usage import ModelUsage
    store, registry, executor = captured(workspace, tmp_path, mode if mode != 'off' else 'git')
    registry.execute('create', declaration(text='DO_NOT_AUTOLOAD_REGISTRATIONS'))
    memory = '甲🙂' * 20000 + 'PRIVATE_TRUNCATED_TAIL'
    (workspace / 'MEMORY.md').write_text(' \n' + memory + '\n ')
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        return tool_reply(api)
    def system(body):
        if api == 'chat':
            return body['messages'][0]['content']
        return body['instructions'] if api == 'responses' else body['system'][0]['text']
    with (Anthropic if api == 'messages' else OpenAI)(api_key='offline-only', max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        def new_agent():
            value = BashAgent(get_bash_agent_tool_descriptions(mode != 'off', mode == 'pf'), client,
                workspace_path=workspace, total_days=42, text_registration=mode != 'off',
                reasoning_effort='low' if api == 'responses' else None,
                usage_recorder=ModelUsage(None, 'agent', evidence_store=store))
            value._snapshot_path = workspace / 'conversation.json'
            return value
        first = new_agent()
        assert first.act('dashboard', 0, False, {'day': 0}).tool == 'read_file'
        base = original_prompt(42) + (registration_prompt(pf=mode == 'pf') if mode != 'off' else '')
        if mode == 'pf':
            base += PF_PROMPT
        expected = (base + '\n\n## Your MEMORY.md (auto-loaded)\n\n'
            'The following is the contents of your MEMORY.md file. '
            'This is automatically loaded into your context at the start of every day.\n\n' + memory[:40000] +
            '\n\n--- MEMORY.md TRUNCATED ---\n' + f'Showing first 40,000 of {len(memory):,} characters. '
            'Use the read_file tool to see the full contents if needed.')
        assert system(requests[0]).encode() == expected.encode()
        definitions = [dict(t.get('function', t)) for t in requests[0]['tools']]
        for item in definitions:
            item.pop('cache_control', None)
            item.pop('type', None)
            if 'input_schema' in item:
                item['parameters'] = item.pop('input_schema')
        assert definitions[:6] == ORIGINAL['tools']
        assert [t['name'] for t in definitions[6:]] == ([] if mode == 'off' else
            ['text_create', 'text_revise', 'text_retire', 'text_list'] + (list(MODELS) if mode == 'pf' else []))
        assert 'DO_NOT_AUTOLOAD_REGISTRATIONS' not in json.dumps(requests[0])
        first.record_tool_result('completed read')
        first._save_conversation_snapshot(strict=True)
        (workspace / 'MEMORY.md').write_text('New weekly note')
        restored = new_agent()
        assert restored.load_conversation_snapshot(restored._snapshot_path)
        first.act('completed read', 0, False, {'day': 0})
        restored.act('completed read', 0, False, {'day': 0})
        assert requests[1] == requests[2]
        assert system(requests[2]) == expected
        restored.record_tool_result('completed read')
        restored.act('next dashboard', 0, False, {'day': 7})
        assert 'New weekly note' in system(requests[3])
        assert memory[:40000] not in system(requests[3])
        assert 'completed read' not in json.dumps(requests[3])
        assert 'version_id' not in restored._snapshot_path.read_text()
        store.assert_healthy()
        save_artifact(f'requests/{mode}-{api}.json', dict(api=api, mode=mode, external_calls=0,
            original_baseline=ORIGINAL['baseline'], requests=requests))


def test_checkpoint_requires_private_harness_token(server):
    import urllib.request
    import urllib.error
    server.checkpoint_token = 'test-harness-token'
    calls = []
    server.checkpoint_callback = lambda: calls.append(True) or {'success': True, 'private': 'receipt'}
    server.start()
    for token in (None, 'wrong-token', 'test-harness-token'):
        headers = {'Content-Type': 'application/json'}
        if token is not None:
            headers['X-Harness-Token'] = token
        req = urllib.request.Request(f'http://127.0.0.1:{server.port}/checkpoint',
                                     json.dumps({'expected_day': 0}).encode(), headers)
        try:
            response = urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            body = json.load(response)
            assert response.status == (200 if token == server.checkpoint_token else 403)
            assert ('private' in body) == (token == server.checkpoint_token)
    assert calls == [True]


@pytest.mark.parametrize('capture', [False, True])
def test_memory_cannot_import_private_files_through_a_symlink(workspace, tmp_path, capture):
    private = tmp_path / 'private-note'
    private.write_text('PRIVATE_EVIDENCE_MUST_NOT_REACH_MODEL')
    store, registry, executor = captured(workspace, tmp_path, 'prefix')
    if not capture:
        executor = BashAgentToolExecutor(workspace)
    # The sandbox can create a dangling link to a host-only path. The host-side
    # weekly MEMORY loader must apply the same boundary as explicit file reads.
    result = executor.execute('bash', {'command': f'ln -s {private} MEMORY.md'})
    assert '[exit code:' not in result
    value = BashAgent.__new__(BashAgent)
    value.workspace_path = workspace
    value.system_prompt = 'Original instructions'
    value.usage_recorder = type('Recorder', (), {'evidence_store': store if capture else None})()
    with pytest.raises(ValueError, match='MEMORY.md.*workspace'):
        value._get_system_prompt_with_memory()
    with store.connect() as conn:
        assert not conn.execute("SELECT 1 FROM requests WHERE request LIKE '%memory_read%'").fetchone()


@pytest.mark.parametrize('delivery', ['missing', 'same', 'changed', 'partial'])
def test_prefix_lifecycle_is_byte_identical_to_git(workspace, tmp_path, monkeypatch, delivery):
    monkeypatch.setattr('saas_bench.text_registry.now', lambda: '2026-09-25T00:00:00Z')
    from saas_bench.text_registry import TextRegistry
    store, prefix, executor = captured(workspace, tmp_path, 'prefix')
    if delivery == 'partial':
        executor.execute('write_file', {'path': 'evidence.json', 'content': '{\n"n":7,\n"other":8\n}'})
        send(store, executor.execute('read_file', {'path': 'evidence.json', 'limit': 2}))
    elif delivery != 'missing':
        send(store, executor.execute('read_file', {'path': 'evidence.json'}))
        if delivery == 'changed':
            executor.execute('write_file', {'path': 'evidence.json', 'content': '{"n":9}'})
    twin = tmp_path / 'git-twin'
    shutil.copytree(workspace, twin)
    control = TextRegistry(twin, 'git', sim_day=lambda: 7)
    operations = [
        ('create', declaration({'path': 'evidence.json'})),
        ('create', declaration({'record': 'r1.1'})),
        ('revise', dict(record='r1', text='Corrected', reason='New wording')),
        ('list', dict(limit=1)),
        ('list', dict(after=1)),
        ('retire', dict(record='r1', reason='Finished')),
        ('list', {}),
    ]
    outputs = []
    head = git(workspace, 'rev-parse', 'HEAD')
    index = (workspace / '.git/index').read_bytes()
    for op, args in operations:
        left, right = prefix.execute(op, args), control.execute(op, args)
        assert left.encode() == right.encode()
        assert prefix.path.read_bytes() == control.path.read_bytes()
        outputs.append(dict(operation=op, output=left))
    assert git(workspace, 'rev-parse', 'HEAD') == head
    assert (workspace / '.git/index').read_bytes() == index
    binding = store.load_state('declaration:r1.1')['references'][0]
    assert binding['status'] == ('unknown' if delivery in ('missing', 'partial') else 'resolved')
    if delivery in ('same', 'changed'):
        assert (binding['version_id'] != binding['latest_version_id']) == (delivery == 'changed')
    for forbidden in ('delivered_in', 'version_id', 'git_content_matches', head):
        assert forbidden not in prefix.path.read_text()
        assert forbidden not in json.dumps(outputs)
    assert not store.load_state(HANDLES)
    # Once created, common list must not consult the evidence resolver at all.
    monkeypatch.setattr(prefix.resolver, 'resolve', lambda *a: pytest.fail('list resolved dependencies'))
    assert prefix.execute('list', {}) == control.execute('list', {})
    save_artifact(f'prefix-{delivery}.json', dict(outputs=outputs, private_binding=binding,
                                                registration=json.loads(prefix.path.read_text())))


def test_git_references_never_save_cited_contents_or_evaluate_predicates(workspace, tmp_path):
    from saas_bench.text_registry import TextRegistry
    registry = TextRegistry(workspace, 'git')
    head = git(workspace, 'rev-parse', 'HEAD')
    before = set(workspace.rglob('*'))
    index = (workspace / '.git/index').read_bytes()
    for size in (1, 7, 40):
        ref = dict(evidence={'path': 'evidence.json', 'commit': head[:size]}, purpose='current',
                   select={'path': '/missing'}, predicate={'type': 'threshold', 'op': '>', 'value': 9999})
        registry.execute('create', declaration(references=[ref]))
    listing = json.loads(registry.execute('list', {}))
    assert all(r['references'][0]['evidence']['commit'] == head[:7] for r in listing['records'])
    assert set(workspace.rglob('*')) - before == {registry.path}
    assert (workspace / '.git/index').read_bytes() == index
    assert git(workspace, 'rev-parse', 'HEAD') == head
    assert '{"n":7}' not in registry.path.read_text()
    assert not re.search(r'\b[0-9a-f]{8,64}\b', registry.path.read_text())
    original = registry.path.read_bytes()
    (workspace / 'new.json').write_text('{"n":1}')
    for evidence in ({'path': 'new.json'}, {'sql': 'SELECT 1'}, {'version': 'v1'},
                     {'path': 'evidence.json', 'commit': 'f' * 40}):
        with pytest.raises(ValueError):
            registry.execute('create', declaration(evidence))
        assert registry.path.read_bytes() == original
    executor = BashAgentToolExecutor(workspace, text_registry=registry)
    error = executor.execute('text_create', declaration({'version': 'a' * 64}))
    assert error.startswith('Error:') and 'a' * 64 not in error


def test_packed_forks_keep_private_material_out_of_agent_access(offline_runner, tmp_path):
    import shlex
    from saas_bench.run_state import clone_sql_run, checkpoint_directory
    prefix = offline_runner(text_registration='prefix')
    prefix._execute_tool('write_file', {'path': 'facts.json', 'content': '{"n":7}'})
    prefix._git_commit_workspace('saved facts')
    send(prefix.evidence_store, prefix._execute_tool('read_file', {'path': 'facts.json'}))
    assert json.loads(prefix._execute_tool('text_create', declaration({'path': 'facts.json'})))['id'] == 'r1'
    prefix._execute_tool('write_file', {'path': 'facts.json', 'content': '{"n":8}'})
    # Reach the specified fork boundary: completed week, before the next model call.
    assert advance(prefix)['success']
    prefix._commit_weeks_up_to(7)
    prefix._save_checkpoint(7)
    checkpoint = prefix._load_checkpoint()
    assert checkpoint['context_boundary'] == 'new_week'
    snapshot = checkpoint_directory(prefix.workspace_dir, checkpoint)
    public_json = (snapshot / 'agent_workspace/registrations.json').read_bytes()
    saved_head = prefix._git('rev-parse', 'HEAD').stdout
    binding = prefix.evidence_store.load_state('declaration:r1.1')['references'][0]
    outputs = {}
    for mode in ('git', 'pf'):
        child = offline_runner(clone_sql_run(prefix.workspace_dir, tmp_path / mode, mode, text_registration=mode))
        assert not child.agent.conversation
        assert child.agent.current_day == -1
        assert child._git('rev-parse', 'HEAD').stdout == saved_head
        assert (child.agent_workspace / 'registrations.json').read_bytes() == public_json
        assert not list(child.agent_workspace.rglob('*.sqlite'))
        assert not re.search(r'(?i)(evidence|provenance|bindings).*\.sqlite', child._git('ls-tree', '-r', '--name-only', 'HEAD').stdout)
        probes = {}
        if mode == 'git':
            # The Git branch neither captures nor receives the prefix evidence database.
            assert child.evidence_store is None and not list(child.workspace_dir.rglob('sql-evidence*'))
        private_paths = [path for path in (child.evidence_store and child.evidence_store.path, prefix.evidence_store.path,
                         snapshot / 'sql-evidence.sqlite', child.workspace_dir / 'manifest.json') if path]
        for i, path in enumerate(private_paths):
            probes[f'bash_read_{i}'] = child._execute_tool('bash', {'command': 'cat ' + shlex.quote(str(path))})
            assert '[exit code:' in probes[f'bash_read_{i}']
            assert binding['version_id'] not in probes[f'bash_read_{i}']
            assert child._execute_tool('read_file', {'path': str(path)}).startswith('Error: Path escapes workspace')
        child._execute_tool('bash', {'command': 'ln -s ' + shlex.quote(str(private_paths[0])) + ' private-link'})
        assert child._execute_tool('read_file', {'path': 'private-link'}).startswith('Error: Path escapes workspace')
        assert 'private-link:' not in child._execute_tool('search_files', {'pattern': '.', 'glob': 'private-link'})
        child._execute_tool('bash', {'command': 'rm private-link'})
        # Requests originate in the actual Git/PF sandbox, not the trusted Runner.
        code = '''import json, os, urllib.request, urllib.error
base = 'http://127.0.0.1:' + os.environ['NOVAMIND_API_PORT']
results = {}
for path, payload in [('/checkpoint', {'expected_day': 7}), ('/pf/query', {}),
                      ('/evidence', None), ('/sql-evidence', None),
                      ('/call', {'tool': 'pf_read', 'args': {}})]:
    request = urllib.request.Request(base + path, None if payload is None else json.dumps(payload).encode(), {'Content-Type':'application/json'})
    try: response = urllib.request.urlopen(request, timeout=5)
    except urllib.error.HTTPError as exc: response = exc
    with response: results[path] = {'status': response.status, 'body': json.load(response)}
results['token_present'] = 'CEOBENCH_CHECKPOINT_TOKEN' in os.environ
print(json.dumps(results))'''
        probes['http'] = json.loads(child._execute_tool('bash', {'command': 'python -c ' + shlex.quote(code)}).rsplit('\n[', 1)[0])
        assert probes['http']['/checkpoint'] == {'status': 403, 'body': {'error': 'Harness access required'}}
        assert not probes['http']['token_present']
        for path in ('/pf/query', '/evidence', '/sql-evidence'):
            assert probes['http'][path]['status'] == 404
        assert probes['http']['/call']['status'] == 200
        assert probes['http']['/call']['body']['success'] is False
        assert 'Unknown tool: pf_read' in json.dumps(probes['http']['/call']['body'])
        history = child._execute_tool('pf_read', {'target': {'record': 'r1.1'}})
        if mode == 'pf':
            assert json.loads(history.split('\n', 1)[1])['version'] == 'r1.1'
        else:
            assert history.startswith('Error: Unknown tool')
        result = child._execute_tool('text_revise', {'record': 'r1', 'reason': 'wording', 'text': 'Revised'})
        if mode == 'pf':
            assert json.loads(result)['evidence'][0]['version'].startswith('v')
        else:
            assert 'evidence' not in json.loads(result)
        assert not re.search(r'\b[0-9a-f]{8,64}\b', result)
        assert child._checkpoint_token not in json.dumps(probes)
        child._save_checkpoint(7)
        assert not any(child._checkpoint_token in p.read_text(errors='ignore') for p in
                       child.agent_workspace.rglob('*.json'))
        outputs[mode] = dict(probes=probes, revise_result=result,
                             registration=json.loads((child.agent_workspace / 'registrations.json').read_text()))
        child._stop_server()
        restored = offline_runner(child.workspace_dir)
        assert restored._checkpoint_token != child._checkpoint_token
        restored._save_checkpoint(7)
    save_artifact('packed-forks.json', dict(external_calls=0, fork_day=7,
                  initial_registration=json.loads(public_json), branches=outputs))
