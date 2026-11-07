from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import shlex

import pytest

from saas_bench.agent_capabilities import AgentCapabilities
from test_preflight_team_flow import final, make_team, raw
from test_relation_batch import batch, create
from test_responsibility_authority import ACTIONS, READS, operations, prepare, world


GROWTH_WRITES = {'set_targeted_ad_spend', 'set_ads_strength', 'set_lead_promotion',
                 'set_promotion', 'post_social_media'}
OPS_WRITES = {'set_model_tiers', 'set_daily_spend', 'set_capacity_tier',
              'set_usage_quotas', 'set_targeted_ops_spend', 'set_targeted_dev_spend'}


def shared_team(mode, label):
    return make_team(mode, lambda *args: final('unused'), label,
                     workspace_layout='shared-v1', capabilities=AgentCapabilities.round_three(mode))


def bash(runtime, command):
    return runtime.executor.execute('bash', {'command': command})


def write(runtime, path, content):
    result = runtime.executor.execute('write_file', dict(path=path, content=content))
    assert result.startswith('File written:'), result
    return result


@pytest.mark.parametrize('mode', ['git', 'pf'])
@pytest.mark.parametrize('gateway', ['http', 'sdk', 'cli', 'bash-child'])
def test_shared_directory_keeps_responsibility_and_socket_identity(mode, gateway):
    for role in ('growth', 'ops_finance', 'ceo'):
        with shared_team(mode, 'shared-authority-' + gateway + '-' + role) as team:
            prepare(team)
            team._set_phase('ceo' if role == 'ceo' else 'analysis')
            allowed = set(ACTIONS) if role == 'ceo' else READS | (GROWTH_WRITES if role == 'growth' else OPS_WRITES)
            if gateway == 'http':
                for name, (args, _, _) in ACTIONS.items():
                    before = world(team)
                    status, receipt = raw(team, role, '/call',
                        dict(tool=name, args=args, role='ceo', actor_id='ceo'))
                    if name in allowed:
                        assert status == 200 and receipt['success'], (role, name, receipt)
                    else:
                        assert status == 403 and not receipt['success'], (role, name, receipt)
                        assert world(team) == before, (role, name)
            else:
                actions = {name: (args, module, name in allowed)
                           for name, (args, module, _) in ACTIONS.items()}
                program = '''import importlib
from novamind_api import _client
for name, (args, module, allowed) in ACTION_DATA.items():
    before = _client.query('SELECT * FROM config_history')['rows']
    try:
        getattr(importlib.import_module('novamind_api.' + module), name)(**args)
    except _client.NovaMindAPIError as exc:
        assert not allowed, (name, str(exc))
        assert '403' in str(exc) or 'Only ' in str(exc), (name, str(exc))
        assert _client.query('SELECT * FROM config_history')['rows'] == before
    else:
        assert allowed, name
    print('ACTION', name, 'ALLOWED' if allowed else 'DENIED')
print('MATRIX-COMPLETE')
'''.replace('ACTION_DATA', repr(actions))
                runtime = team.roles[role]
                write(runtime, 'scripts/authority.py', program)
                command = {'sdk': 'python -c ' + shlex.quote(program),
                           'cli': './novamind-operation python-c ' + shlex.quote(program),
                           'bash-child': 'bash -c ' + shlex.quote('python scripts/authority.py')}[gateway]
                result = bash(runtime, command)
                assert 'MATRIX-COMPLETE' in result and '[exit code:' not in result, (role, result)
                assert result.count('ACTION ') == len(actions), result
            recorded = [record for record in operations(team)
                        if record.get('role') == role and record['kind'].startswith('api:')]
            assert {record['kind'] for record in recorded} == {'api:' + name for name in allowed - READS}
            assert len(recorded) == len(allowed - READS)
            assert all(record['status'] == 'completed' for record in recorded)
            assert raw(team, role, '/query', dict(sql='SELECT 1 AS continued'))[0] == 200
            assert not team.server._operation_failed
            assert team.server.tools.current_day == 0
            assert not team.test_wire


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_managed_files_and_private_authority_stay_outside_shared_write_access(mode):
    with shared_team(mode, 'shared-boundaries') as team:
        workspace = team.roles['growth'].workspace
        assert {runtime.workspace for runtime in team.roles.values()} == {workspace}
        managed = {'docs/api/managed-sentinel.txt': 'MANAGED-DOC',
                   'documents/managed-sentinel.json': '{"managed":true}'}
        for path, content in managed.items():
            (workspace / path).parent.mkdir(parents=True, exist_ok=True)
            (workspace / path).write_text(content)
        before_client = hashlib.sha256((workspace / 'novamind-operation').read_bytes()).hexdigest()
        secret = 'RESEARCHER-PRIVATE-ONLY-' + mode
        private_path = team.root / 'private/growth/researcher-secret.txt'
        private_path.write_text(secret)
        for role in ('growth', 'ops_finance'):
            runtime = team.roles[role]
            for path in (*managed, 'novamind-operation'):
                result = runtime.executor.execute('write_file', dict(path='/workspace/' + path, content='CORRUPTED'))
                assert result.startswith('Error:') and 'read-only' in result, (path, result)
                result = runtime.executor.execute('edit_file', dict(path=path, old_string='MANAGED', new_string='CORRUPTED'))
                assert result.startswith('Error:') and 'read-only' in result, (path, result)
            program = '''from pathlib import Path
for name in PATHS:
    try:
        Path('/workspace', name).write_text('CORRUPTED')
    except OSError:
        print('DENIED', name)
    else:
        raise AssertionError(name)
print('MANAGED-BOUNDARY-COMPLETE')
'''.replace('PATHS', repr([*managed, 'novamind-operation']))
            result = bash(runtime, 'python -c ' + shlex.quote(program))
            assert 'MANAGED-BOUNDARY-COMPLETE' in result and '[exit code:' not in result, result
            for path in (str(private_path), '/workspace/../../private/growth/researcher-secret.txt'):
                result = runtime.executor.execute('read_file', dict(path=path))
                assert result.startswith('Error:') and secret not in result, result
                result = runtime.executor.execute('write_file', dict(path=path, content='CORRUPTED'))
                assert result.startswith('Error:'), result
                result = bash(runtime, 'cat ' + shlex.quote(path))
                assert '[exit code:' in result and secret not in result, result
            for table in ('_team_scripts', '_registered_scripts', 'group_insight_snapshots'):
                status, response = raw(team, role, '/query', dict(sql='SELECT * FROM ' + table))
                assert status == 403 and secret not in json.dumps(response), response
            status, response = raw(team, role, '/private/growth/researcher-secret.txt')
            assert status == 404 and secret not in json.dumps(response), response
            assert raw(team, role, '/query', dict(sql='SELECT 1 AS continued'))[0] == 200
            assert 'PUBLIC-AFTER-DENIAL' in bash(runtime, 'printf PUBLIC-AFTER-DENIAL > plans/continued.txt; cat plans/continued.txt')
            if mode == 'pf':
                for path in (str(private_path), '../private/growth/researcher-secret.txt', '.git/config', 'registrations.json'):
                    result = bash(runtime, shlex.join(['pf', 'show', path, '--full']))
                    assert '[exit code: 1]' in result and secret not in result, (path, result)
        assert private_path.read_text() == secret
        assert {path: (workspace / path).read_text() for path in managed} == managed
        assert hashlib.sha256((workspace / 'novamind-operation').read_bytes()).hexdigest() == before_client
        if mode == 'git':
            result = bash(team.roles['growth'], 'git add -f . && git -c user.name=Test -c user.email=test@example.invalid commit -qm public')
            assert '[exit code:' not in result, result
            result = bash(team.roles['ops_finance'], "git grep -n RESEARCHER-PRIVATE-ONLY HEAD -- || true")
            assert secret not in result, result


@pytest.mark.parametrize('path', ['scripts/shared.py', 'plans/shared.txt'])
def test_pf_shared_file_history_separates_namespace_producer_and_collector(path):
    with shared_team('pf', 'shared-file-identity') as team:
        growth, ops = team.roles['growth'], team.roles['ops_finance']
        first = write(growth, '/workspace/' + path, 'FIRST-GROWTH-BYTES\n')
        assert 'shared:' + path + '@v1' in first, first
        second = ops.executor.execute('edit_file', dict(path=path,
            old_string='FIRST-GROWTH-BYTES', new_string='SECOND-OPS-BYTES'))
        assert 'shared:' + path + '@v2' in second, second
        for reader in (growth, ops):
            listing = bash(reader, shlex.join(['pf', 'log', '/workspace/' + path]))
            assert 'shared:' + path + '@v1' in listing and 'shared:' + path + '@v2' in listing, listing
            old = bash(reader, shlex.join(['pf', 'show', 'shared:' + path + '@v1', '--full']))
            assert 'FIRST-GROWTH-BYTES' in old and 'SECOND-OPS-BYTES' not in old, old
            current = bash(reader, shlex.join(['pf', 'show', path, '--full']))
            assert 'SECOND-OPS-BYTES' in current, current
            for obsolete in ('growth:' + path + '@v1', 'ops_finance:' + path + '@v2'):
                denied = bash(reader, shlex.join(['pf', 'show', obsolete, '--full']))
                assert '[exit code: 1]' in denied and 'legacy' in denied.lower(), denied
        saved = batch(team, 'ops_finance', [create('revision', 'shared:' + path + '@v2',
            ['shared:' + path + '@v1'])], 'shared-revision')
        assert saved['saved'] == 1, saved
        target, source = saved['items'][0]['target'], saved['items'][0]['sources'][0]['ref']
        assert (target['workspace_namespace'], target['producer_role'], target['collector_role']) == ('shared', 'ops_finance', 'ops_finance')
        assert source['workspace_namespace'] == 'shared'
        assert source['source_role'] in ('growth', None)
        assert target['storage_role'] is None and source['storage_role'] is None
        observed = growth.executor.execute('read_file', dict(path='/workspace/' + path))
        assert 'SECOND-OPS-BYTES' in observed, observed
        captured = [growth.store.get_content(origin['version_id'])[0] for origin in observed.origins]
        captured_files = [meta for meta in captured if meta['layer'] == 'file_text']
        assert captured_files
        assert all(meta['workspace_namespace'] == 'shared' and meta['collector_role'] == 'growth'
                   and 'owner_role' not in meta and meta.get('producer_role') != 'growth'
                   for meta in captured_files)
        bash_change = bash(growth, 'printf SHELL-COMPUTED > plans/from-shell.txt')
        assert 'shared:plans/from-shell.txt@v1' in bash_change, bash_change
        declaration = batch(team, 'growth', [create('unknown-author', 'shared:plans/from-shell.txt@v1',
            ['shared:' + path + '@v2'], assertion='inferred')], 'shared-shell')
        assert declaration['saved'] == 1, declaration
        target = declaration['items'][0]['target']
        assert target['producer_role'] is None and target['source_role'] is None
        assert target['collector_role'] == 'growth' and target['workspace_namespace'] == 'shared'


def test_pf_outputs_keep_actor_identity_and_parallel_relations_keep_all_authors():
    with shared_team('pf', 'shared-output-relations') as team:
        runtimes = [team.roles[role] for role in ('growth', 'ops_finance')]
        outputs = [bash(runtime, "./novamind-operation query 'SELECT 1 AS SAME_COMMAND_OUTPUT'") for runtime in runtimes]
        handles = []
        for runtime, output in zip(runtimes, outputs):
            match = re.search(r'\[pf: ([^\s|\]]+@v[1-9][0-9]*)', output)
            assert match, output
            handle = match[1]
            assert handle.startswith(runtime.identity.role + ':'), (handle, output)
            handles.append(handle)
            write(runtime, 'plans/' + runtime.identity.role + '.txt', 'PLAN-' + runtime.identity.role)
        assert handles[0] != handles[1]
        for reader in runtimes:
            for handle in handles:
                result = bash(reader, shlex.join(['pf', 'show', handle, '--full']))
                assert 'SAME_COMMAND_OUTPUT' in result and '[exit code:' not in result, result

        def declare(pair):
            runtime, handle = pair
            role = runtime.identity.role
            receipt = batch(team, role, [create('own-plan', 'shared:plans/' + role + '.txt@v1',
                [handle])], 'parallel-' + role)
            assert receipt['saved'] == 1, receipt
            return receipt['items'][0]

        with ThreadPoolExecutor(2) as pool:
            declarations = list(pool.map(declare, zip(runtimes, handles)))
        projection = json.loads((runtimes[0].workspace / 'documents/relations.json').read_text())
        records = [record for revisions in projection['records'].values() for record in revisions]
        assert {record['relation_revision'] for record in records} == {item['relation_revision'] for item in declarations}
        assert {record['author_role'] for record in records} == {'growth', 'ops_finance'}
        for reader in runtimes:
            shown = reader.executor.execute('read_file', dict(path='documents/relations.json'))
            for item in declarations:
                assert item['relation_revision'] in shown
            assert all(runtime.identity.session_id not in shown for runtime in team.roles.values())
        for runtime, item in zip(runtimes, declarations):
            role = runtime.identity.role
            receipt = batch(team, role, [dict(item_id='revision', op='revise', relation=item['relation'],
                expected_revision=item['relation_revision'], reason='Updated declared basis.', sources=[])], 'revision-' + role)
            assert receipt['saved'] == 1, receipt
        projection = json.loads((runtimes[0].workspace / 'documents/relations.json').read_text())
        assert {history[-1]['author_role'] for history in projection['records'].values()} == {'growth', 'ops_finance'}
        assert all(len(history) == 2 for history in projection['records'].values())


def test_peer_read_preserves_known_producer_without_claiming_reader_authorship():
    with shared_team('pf', 'shared-read-producer') as team:
        growth, ops = team.roles['growth'], team.roles['ops_finance']
        write(growth, 'plans/basis.txt', 'SOURCE')
        write(growth, 'plans/forecast.txt', 'KNOWN-GROWTH-PLAN')
        before = batch(team, 'growth', [create('before-read', 'shared:plans/forecast.txt@v1', ['shared:plans/basis.txt@v1'])], 'before-peer-read')
        assert before['saved'] == 1, before
        assert before['items'][0]['target']['producer_role'] == 'growth'
        result = ops.executor.execute('read_file', dict(path='plans/forecast.txt'))
        assert 'KNOWN-GROWTH-PLAN' in result
        read_versions = [ops.store.get_content(origin['version_id'])[0] for origin in result.origins]
        read_files = [meta for meta in read_versions if meta['layer'] == 'file_text']
        assert read_files and all(meta.get('producer_role') == 'growth' and meta['collector_role'] == 'ops_finance' for meta in read_files), read_files
        after = batch(team, 'growth', [create('after-read', 'shared:plans/forecast.txt@v1', ['shared:plans/basis.txt@v1'])], 'after-peer-read')
        assert after['saved'] == 1, after
        target = after['items'][0]['target']
        assert target['producer_role'] == 'growth' and target['collector_role'] in ('growth', 'ops_finance')
        reader = batch(team, 'ops_finance', [create('reader-is-not-author', 'shared:plans/forecast.txt@v1', ['shared:plans/basis.txt@v1'])], 'reader-claim')
        assert reader['saved'] == 0 and reader['items'][0]['error_code'] == 'author_mismatch', reader
        assert 'KNOWN-GROWTH-PLAN' in bash(ops, 'pf show shared:plans/forecast.txt@v1 --full')
