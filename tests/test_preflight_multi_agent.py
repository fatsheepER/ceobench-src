import copy
import http.client
import json
import os
from pathlib import Path
import shlex
import socket
import sqlite3
import tempfile
from types import SimpleNamespace
import urllib.error
import urllib.request

import httpx
from numpy.random import default_rng
from openai import OpenAI
import pytest

from saas_bench.agents.bash_agent.agent import Message
from saas_bench.config import BenchmarkConfig
from saas_bench.database import init_database
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.role_policy import CALL_POLICY, READABLE_ROLES, ROLES
from saas_bench.simulation import Simulator
from saas_bench.tools import AgentTools

PREDICTIONS = {f'cash_{h}wk': dict(point=100, lower=0, upper=200) for h in (1, 4, 12, 26)}
MUTATIONS = [('set_prices', {'A': 19}), ('research_market', {}),
             ('research_group', {'group_id': 'S1', 'target_level': 2}),
             ('start_research_project', {'tier': 1})]


@pytest.fixture(scope='module', params=['git', 'pf'])
def team(request):
    artifacts = os.environ.get('CEOBENCH_TEAM_ARTIFACTS')
    base = Path(artifacts) if artifacts else Path(tempfile.mkdtemp(prefix='team-acceptance-'))
    base.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=request.param + '-', dir=base))
    conn = init_database(root / 'world.db')
    config = BenchmarkConfig(seed=123)
    sim = Simulator(conn, config, default_rng(123))
    sim.initialize()
    conn.close()
    conn = sqlite3.connect(root / 'world.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    tools = AgentTools(conn, 0, root / 'host', config=config, rng=default_rng(123))
    wire = []
    def client(role):
        def respond(req):
            wire.append(dict(role=role, body=json.loads(req.content)))
            return httpx.Response(200, json=dict(id='offline', object='chat.completion', created=1,
                model='offline-model', choices=[dict(index=0, finish_reason='stop',
                    message=dict(role='assistant', content='offline receipt'))]))
        return OpenAI(api_key='PRIVATE-PROVIDER-SECRET', max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    fake_step = SimpleNamespace(step_week=lambda: SimpleNamespace(day=tools.current_day + 7))
    runtime = MultiAgentRuntime(root / 'run', tools, conn=conn, simulator=fake_step,
        client_factory=client, mode=request.param, dashboard_callback=lambda day, result: f'Day {day}',
        checkpoint_token='PRIVATE-HARNESS-SECRET')
    runtime.wire, runtime.mode = wire, request.param
    runtime.private_probe = root / 'host-credentials.txt'
    runtime.private_probe.write_text('HOST-SECRET-UNREADABLE')
    for role, r in runtime.roles.items():
        (r.workspace / 'MEMORY.md').write_text(role + '-memory')
        (r.workspace / 'data.txt').write_text(role + '-data')
        (r.workspace / 'sessions').mkdir()
        (r.workspace / 'sessions' / 'secret.txt').write_text('HIDDEN-SESSION-SECRET')
    yield runtime
    for r in runtime.roles.values():
        if r.store:
            r.store.assert_healthy(settle=3)
        r.agent.client.close()
    (root / 'provider-requests.json').write_text(json.dumps(wire, indent=2))
    runtime.close()
    conn.close()


def shell(team, role, code, cli=False):
    command = ('./novamind-operation python-c ' if cli else 'python -c ') + shlex.quote(code)
    return team.roles[role].executor.execute('bash', {'command': command})


def raw(team, role, path, body=None, method=None, headers=None):
    conn = http.client.HTTPConnection('localhost')
    conn.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.sock.connect(str(team.server.role_sockets[role]))
    conn.request(method or ('POST' if body is not None else 'GET'), path,
        body=json.dumps(body) if body is not None else None,
        headers={'Content-Type': 'application/json', **(headers or {})})
    response = conn.getresponse()
    result = response.status, json.load(response)
    conn.close()
    return result


def snapshot(team):
    api = team.server
    return api.conn.serialize(), api.tools.current_day, copy.deepcopy(api.tools.rng.bit_generator.state), api.world_state_id


def test_generated_clients_share_one_world_and_ceo_can_write(team):
    assert len({id(r.agent) for r in team.roles.values()}) == 3
    assert len({id(r.executor) for r in team.roles.values()}) == 3
    for i, gateway in enumerate(('raw', 'sdk', 'cli')):
        price = 23 + i
        if gateway == 'raw':
            status, body = raw(team, 'ceo', '/call', {'tool': 'set_prices', 'args': {'A': price}})
            assert status == 200 and body['success']
        else:
            output = shell(team, 'ceo', f'import novamind_api as n; print(n.pricing.set_prices(A={price}))', gateway == 'cli')
            assert 'updated' in output, output
        for role in ROLES:
            output = shell(team, role, "import novamind_api as n; print(n.query('SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1')['rows'])")
            assert str(price) in output, output
    assert set(CALL_POLICY) == set(__import__('saas_bench.api_server', fromlist=['_TOOL_DISPATCH'])._TOOL_DISPATCH)


@pytest.mark.parametrize('role', ROLES[1:])
@pytest.mark.parametrize('gateway', ('raw', 'sdk', 'cli'))
def test_analyst_writes_research_advance_denied_without_changes(team, role, gateway):
    before = snapshot(team)
    code = '''import json, os, http.client, socket
from novamind_api import _client
mutations = MUTATIONS
predictions = PREDICTIONS
for tool, args in mutations + [('advance', {})]:
    if GATEWAY == 'raw':
        connection = http.client.HTTPConnection('localhost')
        connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.sock.connect(os.environ['NOVAMIND_API_SOCKET'])
        body = dict(rationale='legal forecast', predictions=predictions) if tool == 'advance' else dict(tool=tool,args=args)
        body.update(actor_id='ceo', role='ceo')
        connection.request('POST', '/next-week' if tool == 'advance' else '/call', json.dumps(body), {'Content-Type':'application/json','X-Actor-Id':'ceo','X-Role':'ceo'})
        response = connection.getresponse()
        assert response.status == 403, response.read()
        print(tool, response.status)
        response.read(); connection.close()
    else:
        try:
            _client.next_week(predictions, 'legal forecast') if tool == 'advance' else _client.call(tool,args)
        except _client.NovaMindAPIError as exc:
            assert '403' in str(exc) or 'Only CEO' in str(exc), str(exc)
            print(tool, 'DENIED')
        else:
            raise AssertionError(tool + ' accepted')
print('ALL-DENIED')
'''.replace('MUTATIONS', repr(MUTATIONS)).replace('PREDICTIONS', repr(PREDICTIONS)).replace('GATEWAY', repr(gateway))
    output = shell(team, role, code, gateway == 'cli')
    assert 'ALL-DENIED' in output and '[exit code:' not in output, output
    direct_cli = team.roles[role].executor.execute('bash', {'command': './novamind-operation next-week forecast ' + '100 0 200 ' * 4})
    assert 'Only CEO' in direct_cli and '[exit code: 1]' in direct_cli, direct_cli
    assert snapshot(team) == before


@pytest.mark.parametrize('actor', ROLES)
@pytest.mark.parametrize('owner', ROLES)
def test_full_file_matrix(team, actor, owner):
    r, target = team.roles[actor], team.roles[owner].workspace
    allowed = owner in READABLE_ROLES[actor]
    marker = owner + '-data'
    variants = [str(target / 'data.txt'), os.path.relpath(target / 'data.txt', r.workspace)]
    for kind in ('absolute', 'relative'):
        link = r.workspace / ('link-' + kind + '-' + owner)
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target / 'data.txt' if kind == 'absolute' else os.path.relpath(target / 'data.txt', r.workspace))
        variants.append(link.name)
    for path in variants:
        read = r.executor.execute('read_file', {'path': path})
        search = r.executor.execute('search_files', {'path': path, 'pattern': marker})
        glob = r.executor.execute('glob_files', {'pattern': path})
        bash = r.executor.execute('bash', {'command': 'cat ' + shlex.quote(path)})
        assert (marker in read) == allowed, read
        assert (marker in search) == allowed, search
        assert (marker in bash) == allowed, bash
        assert (not glob.startswith('Error:') and 'Skipped' not in glob) == allowed, glob
        original = (target / 'data.txt').read_text()
        write = r.executor.execute('write_file', {'path': path, 'content': marker + '-changed'})
        assert (target / 'data.txt').read_text() == (marker + '-changed' if actor == owner else original)
        edit = r.executor.execute('edit_file', {'path': path, 'old_string': marker + ('-changed' if actor == owner else ''), 'new_string': marker + '-edited'})
        assert (target / 'data.txt').read_text() == (marker + '-edited' if actor == owner else original)
        bash_write = r.executor.execute('bash', {'command': 'printf ' + shlex.quote(marker + '-shell') + ' > ' + shlex.quote(path)})
        assert write.startswith('File written:') == (actor == owner), write
        assert edit.startswith('File edited:') == (actor == owner), edit
        assert ('[exit code:' not in bash_write) == (actor == owner), bash_write
        assert (target / 'data.txt').read_text() == (marker + '-shell' if actor == owner else original)
        if actor == owner:
            (target / 'data.txt').write_text(original)
    hidden = str(target / 'sessions' / 'secret.txt')
    assert 'HIDDEN-SESSION-SECRET' not in r.executor.execute('read_file', {'path': hidden})
    assert 'HIDDEN-SESSION-SECRET' not in r.executor.execute('bash', {'command': 'cat ' + shlex.quote(hidden)})


@pytest.mark.parametrize('role', ROLES)
def test_private_files_sockets_network_and_controls(team, role):
    r = team.roles[role]
    probes = [team.private_probe, team.root.parent / 'world.db', team.root / 'private',
              team.root / 'private' / 'ceo' / 'model-usage.jsonl']
    probes += [p for owner, p in team.server.role_sockets.items() if owner != role]
    for path in probes:
        assert r.executor.execute('read_file', {'path': str(path)}).startswith('Error:'), path
    code = '''import os,socket
paths = PATHS
for path in paths:
    assert not os.path.exists(path), path
for name in ('NMDB_KEY','CEOBENCH_CHECKPOINT_TOKEN','OPENAI_API_KEY'):
    assert name not in os.environ
try:
    socket.create_connection(('127.0.0.1', PORT), .1)
except OSError:
    pass
else:
    raise AssertionError('host TCP reachable')
print('PRIVATE-BOUNDARY-PASS')
'''.replace('PATHS', repr([str(p) for p in probes])).replace('PORT', str(team.server.port))
    assert 'PRIVATE-BOUNDARY-PASS' in shell(team, role, code)
    for endpoint in ('/checkpoint', '/pf-refresh', '/run-metrics'):
        assert raw(team, role, endpoint, {}, headers={'X-Harness-Token': 'PRIVATE-HARNESS-SECRET'})[0] == 403
    for endpoint in ('/call', '/next-week', '/query', '/daily-scripts', '/_capture'):
        req = urllib.request.Request(f'http://127.0.0.1:{team.server.port}{endpoint}', data=b'{}')
        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(req)
        assert denied.value.code == 403


@pytest.mark.parametrize('role', ROLES)
def test_owned_registration_memory_and_registered_scripts(team, role):
    r = team.roles[role]
    assert role + '-memory' in r.agent._get_system_prompt_with_memory()
    args = dict(text=role + '-claim', objects=[dict(kind='plan', id='A')],
        references=[dict(cite='unknown: offline')], applies='0-', reason='offline')
    assert r.executor.execute('text_create', args).startswith('Registered r1.1')
    saved = json.loads((r.workspace / 'registrations.json').read_text())['records']['r1'][0]
    assert saved['author'] == saved['role'] == role
    for key, value in r.identity.fields().items():
        assert saved[key] == value
    script = "from novamind_api import _client\nprint('OWNER-" + role + "')\n"
    if role != 'ceo':
        script += "\nfor tool,args in " + repr(MUTATIONS) + ":\n try: _client.call(tool,args)\n except _client.NovaMindAPIError: print('DENIED',tool)\n else: raise AssertionError(tool)\n"
        script += "try: _client.next_week(" + repr(PREDICTIONS) + ", 'forecast')\nexcept _client.NovaMindAPIError: print('DENIED advance')\nelse: raise AssertionError('advance')\n"
    else:
        script += "print(_client.call('set_prices',{'B':31}))\n"
    assert raw(team, role, '/daily-scripts', dict(name='same.py', content=script))[0] == 200
    (r.workspace / 'same.py').write_text("raise AssertionError('unregistered changed source')")
    before = snapshot(team)
    outputs = team.server._run_daily_scripts_internal(role)
    assert 'OWNER-' + role in outputs['same.py'] and '[exit code:' not in outputs['same.py'], outputs
    if role != 'ceo':
        assert outputs['same.py'].count('DENIED') == 5
        assert snapshot(team) == before
    listed = raw(team, role, '/daily-scripts')[1]['data']['scripts']
    assert [item['name'] for item in listed] == ['same.py']
    for owner in ROLES:
        scripts = team.server.get_daily_scripts(owner)
        if scripts:
            assert 'OWNER-' + owner in scripts['same.py']
    saved['author'] = 'forged'
    path = r.workspace / 'registrations.json'
    original = path.read_text()
    tampered = json.loads(original)
    tampered['records']['r1'][0] = saved
    path.write_text(json.dumps(tampered))
    assert r.executor.execute('text_list', {}).startswith('Error: Registration identity')
    path.write_text(original)


@pytest.mark.parametrize('role', ROLES)
def test_model_receipts_and_private_evidence_identity(team, role):
    r = team.roles[role]
    observed = r.executor.execute('read_file', {'path': 'data.txt'})
    message = Message(role='user', content=observed)
    r.agent.conversation = [message]
    sentinel = 'MODEL-ONLY-PRIVATE-' + role
    req = dict(model=sentinel, messages=[dict(role='user', content=observed)])
    r.agent._request_model('chat', req, lambda: r.agent.client.chat.completions.create(**req))
    r.agent._save_conversation_snapshot(strict=True)
    assert r.agent.load_conversation_snapshot(r.agent._snapshot_path)
    assert message.message_id in [m.message_id for m in r.agent.conversation]
    entries = [json.loads(line) for line in r.usage.path.read_text().splitlines()]
    attempts = [e for e in entries if e['event'] == 'http_request']
    assert len(attempts) == 1 and attempts[0]['call_id'] and attempts[0]['attempt_id']
    for entry in entries:
        assert all(entry[k] == v for k, v in r.identity.fields().items())
        assert entry['message_ids'] == [message.message_id]
        assert entry['context_id'] and entry['world_state_id']
    assert r.usage.summary['known']['input_tokens'] is None
    assert 'PRIVATE-PROVIDER-SECRET' not in r.usage.path.read_text()
    audit = [json.loads(line) for line in r.audit.path.read_text().splitlines()]
    assert all(e['role'] == role and e['world_state_id'] for e in audit)
    if team.mode == 'git':
        assert r.store is r.registry.store is r.executor.evidence_store is None
        assert not list((team.root / 'private').rglob('*.sqlite'))
        assert 'tool_return' not in r.audit.path.read_text()
    else:
        with r.store.connect() as conn:
            records = [json.loads(row[0]) for row in conn.execute("SELECT request FROM requests WHERE json_extract(request,'$.role')=?", (role,))]
        assert records and all(item['author'] == item['role'] == role for item in records)
        assert all(item['event_id'].split('/')[1] == role for item in records)
        model = [item for item in records if item['kind'] == 'model_request'][-1]
        assert model['request']['message_ids'] == [message.message_id]
        occurrences = json.loads(r.store.get_content(model['event_id'] + ':occurrences')[1])
        assert occurrences and all(o['reader'] == role for o in occurrences)
        denied = r.executor.execute('pf_read', {'target': {'version': model['event_id'] + ':wire'}})
        assert denied.startswith('Error:') and sentinel not in denied, denied
        r.executor.execute('pf_search', {'text': sentinel})
        assert r.executor.pf_queries.last_answer['total'] == 0
        assert not r.executor.pf_queries.last_answer['items']
        for peer in ROLES:
            if peer != role:
                with pytest.raises(KeyError):
                    r.store.read_event(r.identity.run_id + '/' + peer + '/1')


def test_ceo_research_advance_positive_and_script_removal_is_scoped(team):
    for tool, args in MUTATIONS[1:]:
        status, result = raw(team, 'ceo', '/call', dict(tool=tool, args=args))
        assert status == 200 and result['success'], result
    day = team.server.tools.current_day
    status, result = raw(team, 'ceo', '/next-week', dict(predictions=PREDICTIONS, rationale='offline valid forecast'))
    assert status == 200 and result['success'] and team.server.tools.current_day == day + 7, result
    for gateway in ('sdk', 'cli'):
        output = shell(team, 'ceo', "from novamind_api import _client; print(_client.next_week(" + repr(PREDICTIONS) + ", 'valid forecast')['day'])", gateway == 'cli')
        assert str(team.server.tools.current_day) in output and '[exit code:' not in output, output
    output = team.roles['ceo'].executor.execute('bash', {'command': './novamind-operation next-week forecast ' + '100 0 200 ' * 4})
    assert 'Day ' in output and '[exit code:' not in output, output
    for role in ROLES:
        assert raw(team, role, '/vars')[1]['current_day'] == day + 28
    before = {role: team.server.get_daily_scripts(role) for role in ROLES}
    assert raw(team, 'growth', '/daily-scripts', dict(name='same.py'), method='DELETE')[0] == 200
    assert team.server.get_daily_scripts('growth') == {}
    assert team.server.get_daily_scripts('ceo') == before['ceo']
    assert team.server.get_daily_scripts('ops_finance') == before['ops_finance']


def test_git_history_and_registration_inspection_obey_mounts(team):
    for role, r in team.roles.items():
        command = "git init -q && git -c user.name=Offline -c user.email=offline@example.invalid add data.txt && git -c user.name=Offline -c user.email=offline@example.invalid commit -qm " + shlex.quote(role + '-history-secret')
        assert '[exit code:' not in r.executor.execute('bash', {'command': command})
    for actor in ROLES:
        for owner in ROLES:
            command = 'git -c safe.directory=* -C ' + shlex.quote(str(team.roles[owner].workspace)) + ' log -1 --format=%B'
            output = team.roles[actor].executor.execute('bash', {'command': command})
            assert ((owner + '-history-secret') in output) == (owner in READABLE_ROLES[actor]), output
    growth, peer = team.roles['growth'], team.roles['ops_finance']
    full = peer.executor.run_private(['git', '-C', str(peer.workspace), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    alternate = growth.workspace / '.git/objects/info/alternates'
    alternate.write_text(str(peer.workspace / '.git/objects') + '\n')
    args = dict(text='Try hidden peer commit', objects=[dict(kind='plan', id='A')],
        references=[dict(cite='data.txt@' + full)], applies='0-', reason='offline')
    try:
        output = growth.executor.execute('text_create', args)
        assert output.startswith('Error:') and 'ops_finance-history-secret' not in output, output
        prompt = growth.agent._get_system_prompt_with_memory()
        assert 'ops_finance-history-secret' not in prompt
    finally:
        alternate.unlink()


def test_capture_parent_tokens_cannot_cross_roles(team):
    if team.mode == 'git':
        assert raw(team, 'growth', '/_capture', {})[0] == 404
        return
    ceo, growth = team.roles['ceo'].store, team.roles['growth'].store
    event = ceo.begin_event('offline_parent')
    token = ceo.context(event)
    before = snapshot(team)
    try:
        status, body = raw(team, 'growth', '/call', {'tool': 'set_prices', 'args': {'A': 100}}, headers={'X-Capture-Context': token})
        assert status == 403
        assert raw(team, 'growth', '/_capture', dict(context=token, call='forged', record={}))[0] == 400
        assert snapshot(team) == before
        growth.assert_healthy()
    finally:
        ceo.complete(event)


@pytest.mark.parametrize('operation', ('read_file', 'write_file', 'edit_file'))
def test_replaced_parent_directory_cannot_escape(team, monkeypatch, operation):
    r = team.roles['ceo']
    directory = r.workspace / ('race-' + operation)
    directory.mkdir()
    (directory / 'secret.txt').write_text('owned')
    private = team.root / 'private' / ('race-' + operation)
    private.mkdir()
    secret = private / 'secret.txt'
    secret.write_text('UNMOUNTED-RACE-SECRET')
    original = r.executor._contained
    swapped = False
    def replace_after_resolution(path, shown):
        nonlocal swapped
        result = original(path, shown)
        if result == directory / 'secret.txt' and not swapped:
            directory.rename(directory.with_suffix('.saved'))
            directory.symlink_to(private)
            swapped = True
        return result
    monkeypatch.setattr(r.executor, '_contained', replace_after_resolution)
    args = dict(path=str(directory / 'secret.txt'), content='forbidden-write',
                old_string='UNMOUNTED-RACE-SECRET', new_string='forbidden-edit')
    result = r.executor.execute(operation, args)
    assert result.startswith('Error:') and 'UNMOUNTED-RACE-SECRET' not in result, result
    assert secret.read_text() == 'UNMOUNTED-RACE-SECRET'
    directory.unlink()
    directory.with_suffix('.saved').rename(directory)


@pytest.mark.parametrize('role', ROLES)
def test_every_public_read_and_private_sql_boundary(team, role):
    before = snapshot(team)
    for tool, policy in CALL_POLICY.items():
        if policy != 'read':
            continue
        args = {'group_id': 'S1'} if tool == 'get_group_insights' else {}
        status, result = raw(team, role, '/call', dict(tool=tool, args=args))
        assert status == 200 and result['success'], (tool, result)
        ceo_status, ceo_result = raw(team, 'ceo', '/call', dict(tool=tool, args=args))
        assert ceo_status == status and ceo_result == result
    for table in ('group_insight_snapshots', '_registered_scripts', '_team_scripts'):
        status, result = raw(team, role, '/query', {'sql': 'SELECT * FROM ' + table})
        assert status == 403, result
    assert snapshot(team) == before


@pytest.mark.parametrize('missing', (True, False))
def test_team_scripts_never_fall_back_to_legacy_executor(team, missing):
    executor = team.server.role_executors.pop('growth')
    if not missing:
        team.server.role_executors['growth'] = None
    before = snapshot(team)
    try:
        with pytest.raises(RuntimeError, match='owner-bound executor'):
            team.server._run_daily_scripts_internal('growth')
        assert snapshot(team) == before
    finally:
        team.server.role_executors['growth'] = executor


def test_execution_audit_records_failure_and_timeout_without_pf_capture(team):
    r = team.roles['growth']
    failed = r.executor.execute('bash', {'command': 'exit 7'})
    assert '[exit code: 7]' in failed
    assert json.loads(r.audit.path.read_text().splitlines()[-1])['status'] == 'failed'
    timeout = r.executor.bash_timeout
    try:
        r.executor.bash_timeout = .3
        result = r.executor.execute('bash', {'command': 'sleep 30'})
        assert 'timed out' in result, result
        assert json.loads(r.audit.path.read_text().splitlines()[-1])['status'] == 'timed_out'
    finally:
        r.executor.bash_timeout = timeout


def test_public_query_audit_records_actual_http_status(team):
    import time
    for role, r in team.roles.items():
        before = len(r.audit.path.read_text().splitlines()) if r.audit.path.exists() else 0
        status, result = raw(team, role, '/query', {'sql': 'SELECT 42 AS public_value'})
        assert status == 200 and result['rows'] == [{'public_value': 42}]
        deadline = time.monotonic() + 3
        while True:
            lines = r.audit.path.read_text().splitlines()
            receipt = json.loads(lines[-1]) if lines else {}
            if len(lines) > before and receipt.get('path') == '/query' and receipt.get('http_status') == 200:
                break
            assert time.monotonic() < deadline, receipt
            time.sleep(.005)
