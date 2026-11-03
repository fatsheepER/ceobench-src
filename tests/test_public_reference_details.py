"""Exact Git source references remain readable through public detail projections."""
import hashlib
import json
import subprocess

from test_pf_relations import cli, details
from test_preflight_team_evidence import declare, write
from test_preflight_team_flow import final, make_team
from test_receipt_outputs import model_receives
from test_text_registry import captured, declaration, git, workspace


def colliding_commits(workspace):
    tree = git(workspace, 'rev-parse', 'HEAD^{tree}')
    header = (f'tree {tree}\nauthor Fixture <fixture@example.invalid> 0 +0000\n'
              'committer Fixture <fixture@example.invalid> 0 +0000\n\n').encode()
    seen = {}
    for number in range(500000):
        body = header + f'collision fixture {number}\n'.encode()
        digest = hashlib.sha1(b'commit ' + str(len(body)).encode() + b'\0' + body).hexdigest()
        if digest[:7] in seen:
            pair = (seen[digest[:7]], body)
            return [subprocess.check_output(['git', '-C', str(workspace), 'hash-object', '-t', 'commit',
                '-w', '--stdin'], input=value).decode().strip() for value in pair]
        seen[digest[:7]] = body
    raise AssertionError('Deterministic Git collision fixture did not find a shared seven-character prefix')


def test_git_public_references_expand_collisions_and_pin_exact_source(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path, 'git')
    commits = colliding_commits(workspace)
    assert commits[0][:7] == commits[1][:7] and commits[0] != commits[1]
    for commit in commits:
        output = executor.execute('text_create', declaration({'path': 'evidence.json', 'commit': commit}))
        assert not output.startswith('Error:'), output
        assert commit not in output
    raw = executor.execute('text_list', {'raw': True})
    records = json.loads(raw)['records']
    short = [record['references'][0]['evidence']['commit'] for record in records]
    assert short[0] != short[1] and all(7 < len(value) < 40 for value in short)
    for value, expected in zip(short, commits):
        assert git(workspace, 'rev-parse', value + '^{commit}') == expected
        assert git(workspace, 'show', value + ':evidence.json') == '{"n":7}'
    private = json.loads(registry.path.read_text())['records']
    assert [private[f'r{index + 1}'][0]['references'][0]['evidence']['commit']
            for index in range(2)] == commits
    projected = json.loads(registry.public_path.read_text())['records']
    assert [projected[f'r{index + 1}'][0]['references'][0]['evidence']['commit']
            for index in range(2)] == short
    wire = model_receives(store, workspace, raw, 'text_list')
    assert all(commit not in wire for commit in commits)


def test_nested_relation_details_project_git_evidence_and_preserve_business_text():
    with make_team('pf', lambda *args: final('done'), 'public-reference-details') as team:
        growth, ceo = team.roles['growth'], team.roles['ceo']
        business = json.dumps({'sha256': 'a' * 64, 'session_id': 'business-session'})
        write(team, 'growth', business, 'source.json')
        full = team._snapshot_history(growth, 'Business source')
        receipt = declare(team, 'growth', business, ['growth:source.json@' + full])
        private = json.loads(growth.registry.path.read_text())['records']['r1'][0]
        assert private['references'][0]['evidence']['commit'] == full
        detail = details(team, 'ceo', 'depend', 'growth:r1.1')
        item = next(row for row in detail['items'] if row['origin'] == 'legacy_text_reference')
        evidence = item['reference']['evidence']
        assert set(evidence) == {'path', 'commit', 'owner'}
        assert 7 <= len(evidence['commit']) < 40
        assert git(growth.workspace, 'rev-parse', evidence['commit'] + '^{commit}') == full
        assert item['path'] == ['growth:r1.1', 'growth:source.json@' + evidence['commit']]
        assert item['target']['record']['text'] == business
        serialized = json.dumps(detail)
        assert full not in receipt + serialized
        assert 'a' * 64 in serialized and 'business-session' in serialized
        for key in ('version_id', 'content_hash', 'run_id', 'world_id', 'agent_id'):
            assert key not in serialized
        output = cli(team, 'ceo', 'pf', 'depend', 'growth:r1.1', '--detail')
        wire = model_receives(ceo.store, ceo.workspace, output)
        assert full not in wire and 'business-session' in wire and 'a' * 64 in wire
        reverse = details(team, 'ceo', 'rdepend', 'growth:source.json@' + evidence['commit'])
        incoming = next(row for row in reverse['items'] if row['target']['ref'].get('cite') == 'growth:r1.1')
        assert incoming['path'] == list(reversed(item['path']))
