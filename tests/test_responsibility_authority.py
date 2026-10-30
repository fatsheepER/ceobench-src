import copy
import json
import shlex
import os
from pathlib import Path
import threading
import time

import pytest

from saas_bench.api_server import _TOOL_DISPATCH
from test_enterprise_sdk_contracts import customer
from test_preflight_team_batch import frozen
from test_preflight_team_readonly_recovery import managed_git
from test_preflight_team_flow import ADVANCE, PREDICTIONS, calls, final, make_team, raw


ALL = ('ceo', 'growth', 'ops_finance')
# Independent public contract: arguments, generated SDK module, authorized roles.
ACTIONS = {
    'get_social_posts': ({}, 'analytics', ALL),
    'get_cost_info': ({}, 'infrastructure', ALL),
    'get_market_overview': ({}, 'market', ALL),
    'get_group_insights': ({'group_id': 'S1'}, 'market', ALL),
    'list_research_projects': ({}, 'research', ALL),
    'research_market': ({}, 'market', ('ceo',)),
    'research_group': ({'group_id': 'S1', 'target_level': 2}, 'market', ('ceo',)),
    'start_research_project': ({'tier': 1}, 'research', ('ceo',)),
    'set_prices': ({'A': 19}, 'pricing', ('ceo',)),
    'send_enterprise_deal': ({'deals': [[900001, [['A', 9]]]]}, 'enterprise', ('ceo',)),
    'reject_enterprise_deal': ({'deals': [{'customer_id': 900002}]}, 'enterprise', ('ceo',)),
    'post_social_media': ({'content': 'Authority matrix business update.'}, 'marketing', ('ceo', 'growth')),
    'set_model_tiers': ({'A': 3}, 'pricing', ALL),
    'set_daily_spend': ({'operations': 123, 'development': 456}, 'marketing', ALL),
    'set_targeted_ad_spend': ({'targeted_spend': {'social_media': {'S1': 17}}}, 'marketing', ALL),
    'set_capacity_tier': ({'tier': 2}, 'infrastructure', ALL),
    'set_usage_quotas': ({'A': 101, 'B': 202, 'C': 303}, 'pricing', ALL),
    'set_targeted_ops_spend': ({'by_group': {'S1': 31}}, 'analytics', ALL),
    'set_targeted_dev_spend': ({'targeted_spend': {'S1': 41}}, 'analytics', ALL),
    'set_ads_strength': ({'global_strength': 0.7}, 'marketing', ALL),
    'set_lead_promotion': ({'global_promotion': 0.2}, 'marketing', ALL),
    'set_promotion': ({'global_promotion': 0.3}, 'pricing', ALL),
}
READS = {'get_social_posts', 'get_cost_info', 'get_market_overview',
         'get_group_insights', 'list_research_projects'}


def operations(team):
    return [json.loads(p.read_text()) for p in (team.root / 'private/operations').glob('*.json')]


def world(team):
    return (team.server.conn.serialize(), copy.deepcopy(vars(team.server.tools.config)),
            copy.deepcopy(team.server.tools.rng.bit_generator.state), team.server.world_state_id)


def prepare(team):
    customer(team.server, 900001, thread_type='churn_prevention')
    customer(team.server, 900002, thread_type='churn_prevention')


def assert_business(team):
    row = team.server.conn.execute('SELECT * FROM config_history ORDER BY day DESC LIMIT 1').fetchone()
    assert (row['tier_A'], row['spend_operations'], row['spend_development'], row['capacity_tier']) == (3, 123, 456, 2)
    assert (row['quota_A'], row['quota_B'], row['quota_C']) == (101, 202, 303)
    cfg = team.server.tools.config
    assert cfg.targeted_ad_spend == {'social_media': {'S1': 17}}
    assert cfg.targeted_ops_spend == {'S1': 31}
    assert cfg.targeted_dev_spend == {'S1': 41}
    assert (cfg.ads_strength_global, cfg.lead_promotion_global, cfg.promotion_global) == (0.7, 0.2, 0.3)


def assert_role_business(team, role, initial_price):
    conn = team.server.conn
    assert conn.execute('SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] == (19 if role == 'ceo' else initial_price)
    posts = conn.execute('SELECT content FROM agent_social_media_posts').fetchall()
    assert [row[0] for row in posts] == (['Authority matrix business update.'] if role in ('ceo', 'growth') else [])
    assert conn.execute('SELECT COUNT(*) FROM research_projects').fetchone()[0] == (1 if role == 'ceo' else 0)
    for category in ('market_research', 'group_research'):
        assert conn.execute('SELECT COUNT(*) FROM ledger WHERE category=?', (category,)).fetchone()[0] == (1 if role == 'ceo' else 0)
    for cid in (900001, 900002):
        turns = conn.execute("SELECT sender,closed FROM enterprise_turns WHERE customer_id=? AND sender='agent'", (cid,)).fetchall()
        assert [tuple(row) for row in turns] == ([('agent', int(cid == 900002))] if role == 'ceo' else [])


@pytest.mark.parametrize('mode', ('git', 'pf'))
@pytest.mark.parametrize('gateway', ('http', 'sdk', 'python-c', 'python-file', 'bash-child', 'registered'))
def test_complete_responsibility_matrix(mode, gateway):
    assert set(ACTIONS) == set(_TOOL_DISPATCH)
    for role in ('growth', 'ops_finance', 'ceo'):
        with make_team(mode, lambda *args: final('done'), 'responsibility-' + gateway + '-' + role) as team:
            prepare(team)
            team._set_phase('ceo_scripts' if gateway == 'registered' and role == 'ceo' else
                            'ceo' if role == 'ceo' else 'analysis')
            initial_price = team.server.conn.execute('SELECT price_A FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0]
            if gateway == 'http':
                for name, (args, module, allowed) in ACTIONS.items():
                    before = world(team)
                    status, receipt = raw(team, role, '/call', dict(tool=name, args=args, role='ceo', actor_id='ceo'))
                    if role not in allowed:
                        assert status == 403, (role, name, status, receipt)
                        assert world(team) == before
                    else:
                        assert status == 200 and receipt['success'], (role, name, receipt)
                        assert (world(team) != before) == (name not in READS), (role, name, receipt)
                    assert not team.server._operation_failed
            else:
                program = '''import importlib,json
from novamind_api import _client
ACTIONS = ACTION_DATA
role = ROLE_DATA
gateway = GATEWAY_DATA
for name,(args,module,allowed) in ACTIONS.items():
    before = _client.query('SELECT * FROM config_history')['rows']
    try:
        result=getattr(importlib.import_module('novamind_api.'+module),name)(**args)
    except _client.NovaMindAPIError as exc:
        assert role not in allowed and ('403' in str(exc) or 'Only CEO' in str(exc)), (name,str(exc))
        assert _client.query('SELECT * FROM config_history')['rows'] == before
    else:
        assert role in allowed, name
    print('ACTION',name,'ALLOWED' if role in allowed else 'DENIED')
try:
    advance = _client.next_week(PREDICTION_DATA,'matrix forecast')
except _client.NovaMindAPIError as exc:
    if role == 'ceo':
        assert gateway == 'registered' and 'frozen during ceo_scripts' in str(exc), str(exc)
    else:
        assert '403' in str(exc) or 'Only CEO' in str(exc), str(exc)
else:
    assert role == 'ceo' and gateway != 'registered' and advance['day'] == 7, advance
print('MATRIX-COMPLETE')
'''.replace('ACTION_DATA', repr(ACTIONS)).replace('ROLE_DATA', repr(role)).replace('GATEWAY_DATA', repr(gateway)).replace('PREDICTION_DATA', repr(PREDICTIONS))
                runtime = team.roles[role]
                if gateway == 'registered':
                    assert raw(team, role, '/daily-scripts', dict(name='matrix.py', content=program))[0] == 200
                    output = team.server._run_daily_scripts_internal(role)['matrix.py']
                else:
                    path = runtime.workspace / 'matrix.py'
                    path.write_text(program)
                    command = {
                        'sdk': 'python -c ' + shlex.quote(program),
                        'python-c': './novamind-operation python-c ' + shlex.quote(program),
                        'python-file': './novamind-operation python matrix.py',
                        'bash-child': 'bash -c ' + shlex.quote('python matrix.py'),
                    }[gateway]
                    output = runtime.executor.execute('bash', dict(command=command))
                assert 'MATRIX-COMPLETE' in output and '[exit code:' not in output, (role, gateway, output)
                assert output.count('ACTION ') == 22
            assert_business(team)
            assert_role_business(team, role, initial_price)
            created = [r for r in operations(team) if r.get('role') == role and r['kind'] != 'api:advance']
            advanced = role == 'ceo' and gateway not in ('http', 'registered')
            assert team.server.tools.current_day == (7 if advanced else 0)
            advances = [r for r in operations(team) if r['kind'] == 'api:advance']
            assert len(advances) == int(advanced)
            assert all(r['role'] == 'ceo' and r['status'] == 'completed' for r in advances)
            assert team.server.conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0] == (4 if advanced else 0)
            expected = {name for name, (_, _, allowed) in ACTIONS.items() if role in allowed and name not in READS}
            assert {r['kind'] for r in created} == {'api:' + name for name in expected}, (role, created)
            assert len(created) == len(expected)
            assert all(r['status'] == 'completed' for r in created)
            assert not team.server._operation_failed
            assert raw(team, role, '/query', dict(sql='SELECT 1 AS continued'))[0] == 200
            if role != 'ceo':
                before = world(team)
                assert raw(team, role, '/next-week', dict(predictions=PREDICTIONS, rationale='matrix'))[0] == 403
                assert world(team) == before
            team._set_phase('ceo')
            before = team.server.tools.current_day
            assert raw(team, 'ceo', '/next-week', dict(predictions=PREDICTIONS, rationale='matrix'))[0] == 200
            assert team.server.tools.current_day == before + 7


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_peer_search_glob_paths_reopen_same_name_owner(mode):
    with make_team(mode, lambda *args: final('done'), 'peer-search') as team:
        for owner in ALL:
            runtime = team.roles[owner]
            assert runtime.executor.execute('write_file', dict(path='same.txt', content=owner + '-BUSINESS')).startswith('File written:')
        for reader in ALL:
            runtime = team.roles[reader]
            for owner in ALL:
                path = str(team.roles[owner].workspace / 'same.txt')
                for tool, args in [('glob_files', dict(pattern=path)),
                                   ('search_files', dict(path=path, pattern='BUSINESS'))]:
                    result = runtime.executor.execute(tool, args)
                    returned = result.split(':1: ')[0] if tool == 'search_files' else result
                    read = runtime.executor.execute('read_file', dict(path=returned))
                    assert owner + '-BUSINESS' in read, (reader, owner, tool, result, read)
                    if mode == 'pf' and tool == 'search_files':
                        assert result.origins
                        for origin in result.origins:
                            meta, raw_bytes = runtime.store.get_content(origin['version_id'])
                            a, b = origin['request_range']
                            c, d = origin['source_range']
                            assert result[a:b] == raw_bytes.decode()[c:d]
                            if meta['layer'] == 'file_text':
                                assert meta['owner_role'] == owner
                        file_version, _ = runtime.store.latest_version('same.txt', 'file_bytes', owner)
                        assert runtime.store.get_content(file_version)[0]['owner_role'] == owner


def save_sample(name, value):
    destination = os.environ.get('CEOBENCH_R221_SAMPLES')
    if destination:
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path / (name + '.json')).write_text(json.dumps(value, indent=2))


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_expert_write_refusal_and_receipt_reach_next_model(mode):
    seen = dict.fromkeys(ALL, 0)
    program = '''from novamind_api import _client
try:
    _client.call('set_prices', {'A': 99})
except _client.NovaMindAPIError as exc:
    print('EXPECTED-REFUSAL', str(exc))
else:
    raise AssertionError('expert price accepted')
print(_client.call('set_daily_spend', {'operations': 333}))
print('EXPERT-WRITE-COMPLETE')
'''

    def responder(team, role, body):
        seen[role] += 1
        if role == 'ceo':
            return calls(('bash', dict(command=ADVANCE)))
        if seen[role] == 1:
            return calls(('bash', dict(command='python -c ' + shlex.quote(program))))
        return final(role + ' executed operations spend 333; price request refused.')

    with make_team(mode, responder, 'expert-model') as team:
        outcome = team.run(stop_after_day=7)
        assert outcome.day == 7 and outcome.reason == 'observation_end', team.failure
        assert seen == dict(ceo=1, growth=2, ops_finance=2)
        for role in ('growth', 'ops_finance'):
            body = [w['body'] for w in team.test_wire if w['role'] == role][1]
            receipts = '\n'.join(m.get('content') or '' for m in body['messages'] if m['role'] == 'tool')
            assert 'EXPECTED-REFUSAL' in receipts and 'Only CEO' in receipts
            assert 'EXPERT-WRITE-COMPLETE' in receipts and 'operations' in receipts
        assert team.server.conn.execute('SELECT spend_operations FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] == 333
        assert not team.server._operation_failed
        receipts = operations(team)
        writes = [r for r in receipts if r['kind'] == 'api:set_daily_spend']
        assert len(writes) == 2 and {r['role'] for r in writes} == {'growth', 'ops_finance'}
        assert all(r['status'] == 'completed' for r in writes)
        assert not any(r['kind'] == 'api:set_prices' for r in receipts)
        usage = {}
        for role in ('growth', 'ops_finance'):
            runtime = team.roles[role]
            usage[role] = [json.loads(line) for line in runtime.usage.path.read_text().splitlines()]
            assert len([r for r in usage[role] if r['event'] == 'http_request']) == 2
            assert runtime.executor._bash_facts['process_boundary']['descendants_closed'] is True
            assert not runtime.executor.preserved_process and getattr(runtime.executor, '_pf_service', None) is None
        save_sample('expert-model-' + mode, dict(wire=team.test_wire, operations=receipts, usage=usage))


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_expert_client_exit_waits_for_server_before_model(mode, monkeypatch):
    import http.client
    import socket
    from concurrent.futures import ThreadPoolExecutor
    from saas_bench.runtime_temp import connect_unix

    entered, release = threading.Event(), threading.Event()
    original = _TOOL_DISPATCH['set_daily_spend']

    def late_write(tools, args):
        entered.set()
        assert release.wait(10)
        return original(tools, args)

    monkeypatch.setitem(_TOOL_DISPATCH, 'set_daily_spend', late_write)
    with make_team(mode, lambda *args: final('server result received'), 'expert-late') as team:
        connection = http.client.HTTPConnection('localhost')
        connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connect_unix(connection.sock, team.server.role_sockets['growth'])
        connection.request('POST', '/call', json.dumps(dict(tool='set_daily_spend', args={'operations': 444})), {'content-type': 'application/json'})
        assert entered.wait(3)
        connection.close()
        runtime = team.roles['growth']
        request = dict(model=runtime.agent.model, messages=[dict(role='user', content='Continue after the write.')])
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(runtime.agent._request_model, 'chat', request,
                lambda: runtime.agent.client.chat.completions.create(**request))
            try:
                time.sleep(.1)
                assert not future.done() and team.test_wire == []
                assert operations(team) == [dict(kind='api:set_daily_spend', role='growth', day=0, status='running')]
            finally:
                release.set()
            future.result(timeout=5)
        assert len(team.test_wire) == 1
        assert team.server.conn.execute('SELECT spend_operations FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] == 444
        assert operations(team)[0]['status'] == 'completed'
        assert not team.server.public_request_state()['active']
        assert not team.server._operation_failed


@pytest.mark.parametrize('managed_git', ('git', 'pf'), indirect=True)
def test_unknown_expert_write_holds_real_attempt_and_preserves_checkpoint(managed_git, tmp_path, monkeypatch):
    import httpx
    from saas_bench import team_batch as batch
    from saas_bench.multi_agent import MultiAgentRuntime
    from saas_bench.run_lifecycle import BusinessOutcomeUnknown, RunCancelled
    from saas_bench.team_checkpoint import validate_snapshot
    from test_preflight_team_checkpoint import clients, dispose, fast_simulator
    from test_preflight_team_flow import completion, opening_response

    team, directory, record, manifest, entry = managed_git
    mode = entry['group']
    entry = dict(entry, stop_after_day=7)
    original = _TOOL_DISPATCH['set_daily_spend']

    def unknown_after_write(tools, args):
        original(tools, args)
        raise RuntimeError('injected missing business result')

    monkeypatch.setitem(_TOOL_DISPATCH, 'set_daily_spend', unknown_after_write)
    wire = []

    def transport(role):
        def respond(request):
            body = json.loads(request.content)
            wire.append(dict(role=role, body=body))
            if response := opening_response(role, body):
                return response
            if role == 'growth' and len([r for r in wire if r['role'] == 'growth']) == 1:
                message = calls(('bash', dict(command="python -c \"import novamind_api as n; n.marketing.set_daily_spend(operations=555)\"")))
            elif role == 'growth':
                message = final('Unexpected continuation after an unknown write.')
            elif role == 'ceo':
                message = calls(('bash', dict(command=ADVANCE)))
            else:
                message = final('No additional operations required.')
            return completion(message, body.get('stream', False))
        return httpx.MockTransport(respond)

    for role, runtime in team.roles.items():
        runtime.agent.client.close()
        runtime.agent.client = batch.checked_client(role, 'expert-unknown-test', transport=transport(role))
        runtime.usage.attach(runtime.agent.client)
    source = batch.save_checkpoint(team, directory, record, 'ready')
    original_source = (source / 'checkpoint.json').read_bytes()
    initial_spend = team.server.conn.execute('SELECT spend_operations FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0]
    result = batch.execute(team, manifest, entry, directory, record, directory / 'STOP')
    assert result['status'] == 'paused' and result['hold'] is True
    assert result['checkpoint'] == str(source)
    hold = batch.read(directory / 'HOLD')
    assert hold['manual_resume_required'] is True and hold['checkpoint'] == str(source)
    assert 'unknown' in hold['detail'].lower() and 'World advancement' not in hold['detail']
    assert team.server._operation_failed
    writes = [r for r in operations(team) if r['kind'] == 'api:set_daily_spend']
    assert writes == [dict(kind='api:set_daily_spend', role='growth', day=0, status='running')]
    assert len([r for r in wire if r['role'] == 'growth']) == 1
    assert [r['role'] for r in wire if r['role'] == 'ceo'] == ['ceo']
    with pytest.raises((RunCancelled, BusinessOutcomeUnknown)):
        runtime = team.roles['growth']
        request = dict(model=runtime.agent.model, messages=[dict(role='user', content='must wait for manual resume')])
        runtime.agent._request_model('chat', request, lambda: runtime.agent.client.chat.completions.create(**request))
    assert len([r for r in wire if r['role'] == 'growth']) == 1
    with pytest.raises(RuntimeError, match='recoverable handoff|outcome unknown'):
        team.checkpoint()
    assert source.is_dir() and (source / 'checkpoint.json').read_bytes() == original_source
    validate_snapshot(source)
    for runtime in team.roles.values():
        assert not runtime.executor.preserved_process and getattr(runtime.executor, '_pf_service', None) is None
    assert team.roles['growth'].executor._bash_facts['process_boundary']['descendants_closed'] is True
    assert not team.server.public_request_state()['active']
    restored = MultiAgentRuntime.restore(source, tmp_path / 'restored', client_factory=clients(), simulator_factory=fast_simulator)
    try:
        assert restored.server.conn.execute('SELECT spend_operations FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] == initial_spend
        assert not restored.server._operation_failed
        assert not [r for r in operations(restored) if r['kind'] == 'api:set_daily_spend']
    finally:
        dispose(restored)
    save_sample('expert-unknown-' + mode, dict(wire=wire, operations=operations(team), hold=hold,
        attempt=result, growth_usage=[json.loads(line) for line in team.roles['growth'].usage.path.read_text().splitlines()]))


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_unknown_and_frozen_requests_leave_next_valid_call_and_model_usable(mode):
    with make_team(mode, lambda *args: final('continued after refusal'), 'ordinary-refusal') as team:
        for role in ALL:
            before, records = world(team), operations(team)
            status, receipt = raw(team, role, '/call', dict(tool='not_a_public_action', args={}))
            assert status == 403 and 'Unknown tool' in receipt['error']
            assert world(team) == before and operations(team) == records
            team._set_phase('analysis' if role == 'ceo' else 'ceo')
            status, receipt = raw(team, role, '/call', dict(tool='set_daily_spend', args={'operations': 777}))
            assert status == 403 and 'frozen' in receipt['error']
            assert world(team) == before and operations(team) == records
            assert not team.server._operation_failed
            team._set_phase('ceo' if role == 'ceo' else 'analysis')
            assert raw(team, role, '/call', dict(tool='set_daily_spend', args={'operations': 777}))[1]['success']
            runtime = team.roles[role]
            request = dict(model=runtime.agent.model, messages=[dict(role='user', content='Continue after refusal.')])
            runtime.agent._request_model('chat', request, lambda: runtime.agent.client.chat.completions.create(**request))
            assert team.test_wire[-1]['role'] == role
            assert not team.server._operation_failed


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_registered_expert_writes_run_in_analysis_and_enter_owner_model(mode, monkeypatch):
    phases, holder = [], {}
    for name in ('set_ads_strength', 'set_capacity_tier'):
        original = _TOOL_DISPATCH[name]

        def observe(tools, args, name=name, original=original):
            phases.append((name, holder['team'].server.team_phase))
            return original(tools, args)

        monkeypatch.setitem(_TOOL_DISPATCH, name, observe)

    def responder(team, role, body):
        return calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final(role + ' registered business action completed.')

    with make_team(mode, responder, 'expert-registered') as team:
        holder['team'] = team
        for role in ('growth', 'ops_finance'):
            call = "n.marketing.set_ads_strength(global_strength=0.6)" if role == 'growth' else "n.infrastructure.set_capacity_tier(2)"
            script = "import novamind_api as n\nprint(" + call + ")\nprint(" + repr(role + '-REGISTERED-BUSINESS') + ")\n"
            assert raw(team, role, '/daily-scripts', dict(name='owner-script.py', content=script))[1]['success']
        assert team.run(stop_after_day=7).reason == 'observation_end', (team.failure, team.failure_detail)
        assert sorted(phases) == [('set_ads_strength', 'analysis'), ('set_capacity_tier', 'analysis')]
        ceo_requests = 0
        for item in team.test_wire:
            role, body = item['role'], item['body']
            rendered = '\n'.join(m.get('content') or '' for m in body['messages'])
            if role == 'ceo':
                ceo_requests += 1
                assert len(team.messages) == 2
                if ceo_requests == 1:
                    assert 'request_world_state_id=' not in rendered
                    assert all(message.reply not in rendered for message in team.messages)
                    continue
                for message in team.messages:
                    assert message.world_state_id != team.server.world_state_id
                    assert 'request_world_state_id=' + message.world_state_id in rendered
                    assert ' | world_state_id=' not in rendered
            else:
                assert role + '-REGISTERED-BUSINESS' in rendered
                peer = 'ops_finance' if role == 'growth' else 'growth'
                assert peer + '-REGISTERED-BUSINESS' not in rendered
                assert 'owner-script.py' in rendered
                assert ("'global': 0.6" if role == 'growth' else "'tier': 2") in rendered
        assert ceo_requests == 2
        assert team.server.tools.config.ads_strength_global == 0.6
        assert team.server.conn.execute('SELECT capacity_tier FROM config_history ORDER BY day DESC LIMIT 1').fetchone()[0] == 2
        receipts = [r for r in operations(team) if r['kind'] in ('api:set_ads_strength', 'api:set_capacity_tier')]
        assert {(r['kind'], r['role'], r['status']) for r in receipts} == {
            ('api:set_ads_strength', 'growth', 'completed'), ('api:set_capacity_tier', 'ops_finance', 'completed')}
        for role in ('growth', 'ops_finance'):
            executor = team.server.role_executors[role]
            assert executor._bash_facts['process_boundary']['descendants_closed'] is True
            assert not executor.preserved_process and getattr(executor, '_pf_service', None) is None
        save_sample('expert-registered-' + mode, dict(wire=team.test_wire, operations=receipts, phases=phases))


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_host_private_git_alternates_cannot_supply_workspace_references(mode):
    import subprocess

    with make_team(mode, lambda *args: final('done'), 'private-git') as team:
        private = team.root / 'private/host-only.git'
        subprocess.run(['git', 'init', '--bare', '-q', str(private)], check=True)

        def git(*args, text):
            return subprocess.run(['git', '--git-dir=' + str(private), '-c', 'user.name=Offline',
                '-c', 'user.email=offline@example.invalid', *args], input=text, capture_output=True,
                encoding='utf-8', check=True).stdout.strip()

        blob = git('hash-object', '-w', '--stdin', text='HOST-GIT-SECRET')
        tree = git('mktree', text='100644 blob ' + blob + '\tsecret.txt\n')
        commit = git('commit-tree', tree, text='HOST-PRIVATE-COMMIT')
        for role in ALL:
            runtime = team.roles[role]
            alternate = runtime.workspace / '.git/objects/info/alternates'
            alternate.write_text(str(private / 'objects') + '\n')
            try:
                for command in ('git show ' + commit + ':secret.txt',
                                'git --git-dir=' + shlex.quote(str(private)) + ' show ' + commit + ':secret.txt'):
                    output = runtime.executor.execute('bash', dict(command=command))
                    assert 'HOST-GIT-SECRET' not in output and '[exit code:' in output, output
                result = runtime.executor.execute('text_create', dict(text='Try host private evidence',
                    objects=[dict(kind='metric', id='cash')], references=[dict(cite='secret.txt@' + commit)],
                    applies='0-', reason='host private boundary'))
                assert result.startswith('Error:') and 'HOST-GIT-SECRET' not in result, result
            finally:
                alternate.unlink()


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_truncated_peer_search_paths_and_origins_remain_owner_correct(mode, monkeypatch):
    with make_team(mode, lambda *args: final('done'), 'peer-truncated-search') as team:
        for owner in ALL:
            output = team.roles[owner].executor.execute('write_file', dict(path='same.txt',
                content='\n'.join(owner + '-BUSINESS-' + str(i) for i in range(3))))
            assert output.startswith('File written:')
        for reader in ALL:
            runtime = team.roles[reader]
            monkeypatch.setitem(runtime.executor.runtime_environment, 'CEOBENCH_SEARCH_MATCH_LIMIT', '1')
            for owner in ALL:
                output = runtime.executor.execute('search_files', dict(path=str(team.roles[owner].workspace / 'same.txt'), pattern='BUSINESS'))
                assert 'Search truncated' in output
                path = output.split(':1: ', 1)[0]
                assert owner + '-BUSINESS-0' in runtime.executor.execute('read_file', dict(path=path))
                saved = output.split('Full searched output: ', 1)[1].split(']', 1)[0]
                full = runtime.executor.execute('read_file', dict(path=saved))
                assert path + ':3: ' + owner + '-BUSINESS-2' in full
                if mode == 'pf':
                    matches = []
                    for origin in output.origins:
                        meta, raw_bytes = runtime.store.get_content(origin['version_id'])
                        a, b = origin['request_range']
                        c, d = origin['source_range']
                        assert output[a:b] == raw_bytes.decode()[c:d]
                        if meta['layer'] == 'file_text':
                            assert meta['owner_role'] == owner
                            matches.append(output[a:b])
                    assert matches == [owner + '-BUSINESS-0']


def copy_missing_git_objects(source, destination):
    import shutil

    for path in source.rglob('*'):
        target = destination / path.relative_to(source)
        if path.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


@pytest.mark.parametrize('mode', ('git', 'pf'))
@pytest.mark.parametrize('alias', ('alternates', 'objects_symlink', 'gitdir_symlink', 'gitdir_file', 'commondir', 'loose_object_symlink', 'pack_symlink'))
def test_git_citations_reject_repository_aliases_and_keep_explicit_peer_owner(mode, alias):
    import shutil

    with make_team(mode, lambda *args: final('done'), 'git-citation-alias') as team:
        for role in ALL:
            runtime = team.roles[role]
            runtime.executor.execute('write_file', dict(path='same.txt', content=role + '-EXACT-CONTENT'))
            team._snapshot_history(runtime, role + ' business snapshot')
        growth, peer = team.roles['growth'], team.roles['ops_finance']
        commit = team._git(peer, 'rev-parse', 'HEAD')
        git_dir = growth.workspace / '.git'
        if alias == 'alternates':
            (git_dir / 'objects/info/alternates').write_text(str(peer.workspace / '.git/objects') + '\n')
        elif alias == 'objects_symlink':
            shutil.rmtree(git_dir / 'objects')
            (git_dir / 'objects').symlink_to(peer.workspace / '.git/objects')
        elif alias in ('gitdir_symlink', 'gitdir_file'):
            shutil.rmtree(git_dir)
            if alias == 'gitdir_symlink':
                git_dir.symlink_to(peer.workspace / '.git', target_is_directory=True)
            else:
                git_dir.write_text('gitdir: ' + str(peer.workspace / '.git') + '\n')
        elif alias == 'commondir':
            (git_dir / 'commondir').write_text(str(peer.workspace / '.git') + '\n')
        elif alias == 'loose_object_symlink':
            copy_missing_git_objects(peer.workspace / '.git/objects', git_dir / 'objects')
            own = git_dir / 'objects' / commit[:2] / commit[2:]
            own.unlink()
            own.symlink_to(peer.workspace / '.git/objects' / commit[:2] / commit[2:])
        else:
            team._git(peer, 'repack', '-ad')
            shutil.rmtree(git_dir / 'objects/pack')
            (git_dir / 'objects/pack').symlink_to(peer.workspace / '.git/objects/pack')
        args = dict(text='Growth cites native evidence', objects=[dict(kind='metric', id='cash')],
            references=[dict(cite='same.txt@' + commit)], applies='0-', reason='exact owner')
        rejected = growth.executor.execute('text_create', args)
        assert rejected.startswith('Error:'), (mode, alias, rejected)
        args['references'] = [dict(cite='ops_finance:same.txt@' + commit)]
        accepted = growth.executor.execute('text_create', args)
        assert accepted.startswith('Registered growth:') and 'ops_finance:same.txt@' in accepted, accepted
        reference = growth.registry._load()['records']['r1'][-1]['references'][0]
        assert reference['evidence']['owner'] == 'ops_finance'
        assert reference['evidence']['path'] == 'same.txt' and reference['evidence']['commit'] == commit[:7]
        if mode == 'pf':
            assert growth.store.load_state('declaration:r1.1')['references'][0]['git_commit'] == commit
        output = growth.executor.execute('bash', dict(command='git -c safe.directory=* -C ' + shlex.quote(str(peer.workspace)) + ' show ' + commit + ':same.txt'))
        assert 'ops_finance-EXACT-CONTENT' in output


@pytest.mark.parametrize('mode', ('git', 'pf'))
def test_team_git_citations_accept_detached_own_history(mode):
    with make_team(mode, lambda *args: final('done'), 'git-native-history') as team:
        growth = team.roles['growth']
        growth.executor.execute('write_file', dict(path='same.txt', content='Growth original'))
        team._snapshot_history(growth, 'Growth saved snapshot')
        own_commit = team._git(growth, 'rev-parse', 'HEAD')
        dangling = team._git(growth, 'commit-tree', own_commit + '^{tree}', '-m', 'Native dangling evidence')
        growth.executor.execute('write_file', dict(path='same.txt', content='Growth newer version'))
        team._snapshot_history(growth, 'Growth later snapshot')
        assert team._git(growth, 'rev-parse', 'HEAD') != own_commit
        assert dangling not in team._git(growth, 'rev-list', 'HEAD')
        assert not team._git(growth, 'for-each-ref')
        for commit in (own_commit, dangling):
            output = growth.executor.execute('text_create', dict(text='Native own repository source',
                objects=[dict(kind='metric', id='cash')], references=[dict(cite='same.txt@' + commit)],
                applies='0-', reason='detached own history'))
            assert output.startswith('Registered growth:'), output
            version = growth.registry.last_result['version']
            record = growth.registry._load()['records'][version.split('.')[0]][-1]
            assert record['references'][0]['evidence'] == dict(owner='growth', path='same.txt', commit=commit[:7])
            if mode == 'pf':
                binding = growth.store.load_state('declaration:' + version)['references'][0]
                assert binding['git_commit'] == commit and binding['git_content_matches'] is True
