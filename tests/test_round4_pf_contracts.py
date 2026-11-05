"""R4-1.1 public path, endpoint-error and execution-inventory contracts."""
from collections import Counter
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pytest

from saas_bench.agent_capabilities import AgentCapabilities
from test_preflight_team_flow import ADVANCE, calls, final, make_team
from test_receipt_outputs import artifact, model_receives
from test_relation_batch import batch, create


CAPS = AgentCapabilities.round_three('pf')


def bash(role, command):
    result = role.executor.execute('bash', {'command': command})
    artifact('bash-' + str(len(command)) + '.txt', dict(command=command, result=result))
    return result


def write(role, path, body):
    result = bash(role, 'mkdir -p ' + shlex.quote(path.rsplit('/', 1)[0] if '/' in path else '.') +
                  '; printf %s ' + shlex.quote(body) + ' > ' + shlex.quote(path))
    assert '[exit code:' not in result, result


@pytest.mark.parametrize('form', ['relative', 'absolute', 'normalized'])
def test_peer_path_aliases_keep_exact_old_versions(form):
    with make_team('pf', lambda *args: final('done'), 'r4-path', capabilities=CAPS) as team:
        reader, peer = team.roles['ops_finance'], team.roles['growth']
        write(peer, 'reports/forecast.txt', 'old forecast')
        write(peer, 'reports/forecast.txt', 'new forecast')
        write(reader, 'decision.txt', 'chosen decision')
        path = ('../growth/reports/forecast.txt' if form == 'relative' else
                str(peer.workspace / 'reports/forecast.txt'))
        if form == 'normalized':
            path = '../growth/reports/../reports/./forecast.txt'
        listing = bash(reader, shlex.join(['pf', 'log', path]))
        assert 'growth:reports/forecast.txt@v2' in listing, listing
        old = bash(reader, shlex.join(['pf', 'show', path + '@v1', '--full']))
        assert 'old forecast' in old and 'new forecast' not in old, old
        latest = bash(reader, shlex.join(['pf', 'show', path, '--full']))
        assert 'new forecast' in latest, latest
        receipt = batch(team, 'ops_finance', [create('old', 'decision.txt@v1', [path + '@v1'])], 'old')
        assert receipt['saved'] == 1, receipt
        item = receipt['items'][0]
        assert item['sources'][0]['ref']['ref']['cite'] == 'growth:reports/forecast.txt@v1'
        reverse = bash(reader, shlex.join(['pf', 'rdepend', path + '@v1']))
        assert item['relation_revision'] in reverse, reverse
        artifact('alias.json', dict(receipt=receipt, listing=listing, old=old, latest=latest, reverse=reverse))


def test_role_directory_is_local_and_missing_path_offers_explicit_peer_syntax():
    with make_team('pf', lambda *args: final('done'), 'r4-local', capabilities=CAPS) as team:
        reader, peer = team.roles['ops_finance'], team.roles['growth']
        write(peer, 'source.txt', 'peer bytes')
        for command in ('pf log growth/source.txt', 'pf show growth/source.txt@v1 --full',
                        'pf rdepend growth/source.txt@v1'):
            missing = bash(reader, command)
            assert 'pf log growth:source.txt' in missing and 'registered text' not in missing, missing
            assert 'peer bytes' not in missing
        write(reader, 'decision.txt', 'own decision')
        rejected = batch(team, 'ops_finance', [create('local', 'decision.txt@v1', ['growth/source.txt@v1'])], 'local')['items'][0]
        assert rejected['status'] == 'rejected' and 'pf log growth:source.txt' in rejected['correction'], rejected
        write(reader, 'growth/source.txt', 'local bytes')
        result = bash(reader, 'pf show growth/source.txt --full')
        assert 'local bytes' in result and 'peer bytes' not in result, result
        assert 'ops_finance:growth/source.txt@v1' in result


def test_endpoint_error_reaches_model_and_corrected_item_uses_exact_peer_version():
    turns, receipts = Counter(), []

    def respond(team, role, body):
        if role == 'ceo':
            return calls(('bash', {'command': ADVANCE}))
        if role == 'growth':
            return final('Use the saved forecast.')
        turns[role] += 1
        if turns[role] == 1:
            return calls(('escalate', {'items': [
                create('bad-source', 'decision.txt@v1', ['growth:source.txt@v1', 'growth:source.txt@v99']),
                create('bad-target', 'missing.txt@v1', ['growth:source.txt@v1']),
                create('valid', 'decision.txt@v1', ['growth:source.txt@v1'])]}))
        if turns[role] == 2:
            receipt = json.loads(body['messages'][-1]['content'])
            receipts.append(receipt)
            artifact('first-rejection.json', receipt)
            assert (receipt['saved'], receipt['rejected']) == (1, 2), receipt
            source, target, valid = receipt['items']
            assert source['error_location'] == 'sources[1].ref'
            assert source['reference'] == {'cite': 'growth:source.txt@v99'}
            assert source['error_code'] == 'invalid_reference'
            assert 'or a file path' not in source['error']
            assert 'pf log growth:source.txt' in source['correction']
            assert target['error_location'] == 'target'
            assert target['reference'] == {'cite': 'ops_finance:missing.txt@v1'}
            return calls(('bash', {'command': 'pf log growth:source.txt'}))
        if turns[role] == 3:
            assert 'growth:source.txt@v1' in body['messages'][-1]['content']
            return calls(('escalate', {'items': [
                create('corrected', 'decision.txt@v1', ['../growth/source.txt@v1'])]}))
        receipt = json.loads(body['messages'][-1]['content'])
        receipts.append(receipt)
        assert receipt['saved'] == 1, receipt
        assert receipt['items'][0]['sources'][0]['ref']['ref']['cite'] == 'growth:source.txt@v1'
        return final('The decision preserves the exact old peer source.')

    with make_team('pf', respond, 'r4-error-model', capabilities=CAPS) as team:
        write(team.roles['growth'], 'source.txt', 'original source')
        write(team.roles['growth'], 'source.txt', 'changed source')
        write(team.roles['ops_finance'], 'decision.txt', 'derived decision')
        result = team.run(stop_after_day=7)
        assert result.reason == 'observation_end', team.failure
        role = team.roles['ops_finance']
        usage = [json.loads(line) for line in role.usage.path.read_text().splitlines()]
        sent = [row['body'] for row in usage if row['event'] == 'http_request']
        assert sent == [row['body'] for row in team.test_wire if row['role'] == 'ops_finance']
        state = role.registry.relations._load()
        assert len(state['records']) == 2  # No partial relation for either bad item.
        artifact('endpoint-model.json', dict(receipts=receipts, sent=sent, state=state))


def test_pf_read_and_many_writes_offer_stable_complete_capture_inventory():
    with make_team('pf', lambda *args: final('done'), 'r4-inventory', capabilities=CAPS) as team:
        role = team.roles['growth']
        write(role, 'seed.txt', 'saved source')
        command = 'pf show seed.txt --full > read.txt; for i in $(seq 1 23); do printf %s "$i" > "part-$i.txt"; done'
        output = bash(role, command)
        assert model_receives(role.store, role.workspace, output) == output
        assert 'and 18 more' in output, output
        cursor = re.search(r'captures: (pf more c[0-9]+)', output)
        assert cursor, output
        first = bash(role, cursor[1])
        page = role.executor.pf_queries.last_answer
        assert 'this execution' in first and page['next_cursor'], first
        items = list(page['items'])
        write(role, 'later.txt', 'not part of the saved execution')
        write(role, 'part-23.txt', 'later replacement')
        while page['next_cursor']:
            result = bash(role, 'pf more ' + page['next_cursor'])
            assert '[exit code:' not in result, result
            page = role.executor.pf_queries.last_answer
            items += page['items']
        files = {item['path']: item['version'] for item in items if 'path' in item}
        assert set(files) == {'read.txt', *(f'part-{i}.txt' for i in range(1, 24))}
        assert files['part-23.txt'] == 'growth:part-23.txt@v1'
        assert '23' in bash(role, 'pf show ' + files['part-23.txt'] + ' --full')
        assert bash(role, cursor[1]) == first
        artifact('inventory.json', dict(output=output, first=first, items=items))


def test_paths_reject_escape_private_symlink_and_revoked_role(monkeypatch):
    from saas_bench import role_policy

    with make_team('pf', lambda *args: final('done'), 'r4-path-authority', capabilities=CAPS) as team:
        reader, peer = team.roles['ops_finance'], team.roles['growth']
        write(peer, 'source.txt', 'public old bytes')
        write(reader, 'decision.txt', 'decision')
        (peer.workspace / 'source.txt').unlink()
        # Deletion must not invalidate real captured history.
        assert 'public old bytes' in bash(reader, 'pf show ../growth/source.txt@v1 --full')
        (peer.workspace / 'source.txt').symlink_to(team.test_root / 'world.db')
        paths = ['growth:source.txt@v1', '../growth/source.txt@v1',
                 str(peer.workspace / 'source.txt') + '@v1', '../growth/../../world.db',
                 'growth:../ceo/MEMORY.md', '../growth/.git/config', '../growth/sessions/private.json',
                 '../growth/registrations.json', str(team.test_root / 'world.db')]
        for index, path in enumerate(paths):
            output = bash(reader, shlex.join(['pf', 'show', path, '--full']))
            assert '[exit code: 1]' in output and 'public old bytes' not in output, (path, output)
            receipt = batch(team, 'ops_finance', [create('denied', 'decision.txt@v1', [path]),
                create('valid', 'decision.txt@v1', ['decision.txt@v1'])], 'denied-' + str(index))
            assert (receipt['saved'], receipt['rejected']) == (1, 1), receipt
            rejected = receipt['items'][0]
            assert rejected['error_location'] == 'sources[0].ref'
            assert str(team.test_root) not in json.dumps(rejected)
        (peer.workspace / 'source.txt').unlink()
        rights = dict(role_policy.READABLE_ROLES)
        rights['ops_finance'] = ('ops_finance',)
        monkeypatch.setattr(role_policy, 'READABLE_ROLES', rights)
        for path in ('growth:source.txt@v1', '../growth/source.txt@v1', str(peer.workspace / 'source.txt') + '@v1'):
            assert 'inaccessible' in bash(reader, shlex.join(['pf', 'show', path, '--full']))
            rejected = batch(team, 'ops_finance', [create('denied', 'decision.txt@v1', [path])], path)['items'][0]
            assert rejected['error_code'] == 'permission_denied', rejected
            assert rejected['error_location'] == 'sources[0].ref'
            assert 'reference' not in rejected


def test_current_declarations_require_exact_file_versions_and_locate_disabled_text():
    with make_team('pf', lambda *args: final('done'), 'r4-endpoint-types', capabilities=CAPS) as team:
        role = team.roles['growth']
        write(role, 'source.txt', 'captured bytes')
        malformed = create('shape', 'source.txt@v1', ['source.txt@v1'])
        malformed['sources'][0]['ref'] = {'cite': 'source.txt@v1', 'operation_ref': 'growth:1'}
        receipt = batch(team, 'growth', [create('bare', 'source.txt@v1', ['source.txt']),
            create('text', 'source.txt@v1', ['growth:r1.1']), malformed,
            create('valid', 'source.txt@v1', ['source.txt@v1'])], 'types')
        assert (receipt['saved'], receipt['rejected']) == (1, 3), receipt
        assert [item['error_code'] for item in receipt['items'][:3]] == ['ambiguous_version', 'disabled_reference', 'invalid_item']
        assert all(item['error_location'] == 'sources[0].ref' for item in receipt['items'][:3])
        assert 'pf log growth:source.txt' in receipt['items'][0]['correction']
        assert 'pf log' not in receipt['items'][1]['correction']
        assert 'text endpoints are disabled' in receipt['items'][1]['correction']
        role.store.assert_healthy()


def test_public_host_bundles_path_resolution_dependency():
    bundle = Path(__file__).parents[1] / 'public/novamind-operation'
    result = subprocess.run([sys.executable, '-I', '-c',
        'import sys; sys.path.insert(0, sys.argv[1]); '
        'from saas_bench.pf_paths import normalize; print(normalize.__module__)', str(bundle)],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'saas_bench.pf_paths'


def test_overlong_reference_rejects_one_item_and_next_public_call_continues():
    with make_team('pf', lambda *args: final('done'), 'r4-long-reference', capabilities=CAPS) as team:
        role = team.roles['growth']
        write(role, 'source.txt', 'captured bytes')
        too_long = 'a' * 300 + '@v1'
        query = bash(role, shlex.join(['pf', 'show', too_long, '--full']))
        assert 'path component is too long' in query, query
        request = [create('bad-source', 'source.txt@v1', [too_long]),
                   create('bad-target', too_long, ['source.txt@v1']),
                   create('valid', 'source.txt@v1', ['source.txt@v1'])]
        receipt = batch(team, 'growth', request, 'long-reference')
        assert (receipt['saved'], receipt['rejected'], receipt['unknown']) == (1, 2, 0), receipt
        assert [item['error_location'] for item in receipt['items'][:2]] == ['sources[0].ref', 'target']
        assert not role.registry.relations.failed
        assert 'continued' in bash(role, 'printf continued')
        role.store.assert_healthy()
        artifact('long-reference.json', receipt)
