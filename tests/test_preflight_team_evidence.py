from contextlib import closing
import json
import shlex

import pytest

from saas_bench import evidence_handles
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from saas_bench.execution_capture import CapturedText
from saas_bench.pf_read import apply_delta
from saas_bench.role_policy import READABLE_ROLES, ROLES
from saas_bench.sql_evidence import SQLEvidenceStore
from saas_bench.text_registry import TextRegistry
from test_preflight_team_flow import ADVANCE, calls, final, make_team


def write(team, role, content, path='same.txt', **extra):
    result = team.roles[role].executor.execute('write_file', dict(path=path, content=content, **extra))
    assert not result.startswith('Error:'), result
    return result


def pf(team, role, command):
    return team.roles[role].executor.execute('bash', dict(command=command))


def declare(team, role, text, cites):
    result = team.roles[role].executor.execute('text_create', dict(text=text, objects=[dict(kind='metric', id='cash')],
        references=[dict(cite=c, note='support') for c in cites], applies='0-', reason='decision'))
    assert not result.startswith('Error:'), result
    return result


def version(team, role, name='same.txt'):
    return team.roles[role].store.latest_version(name, 'file_bytes')[0]


def send(team, role, texts):
    runtime = team.roles[role]
    runtime.agent._llm_attempt = 0
    request = dict(model=runtime.agent.model, messages=[dict(role='user', content=text) for text in texts])
    runtime.agent._request_model('chat', request, lambda: runtime.agent.client.chat.completions.create(**request))
    event = runtime.usage.last_request_event
    wire = json.loads(runtime.store.get_content(event + ':wire')[1])
    assert wire == team.test_wire[-1]['body']
    ledger = json.loads(runtime.store.get_content(event + ':pf_reads')[1])
    return wire, ledger


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_independent_detached_history_handoffs_weeks_and_git_authority(mode):
    seen = {role: 0 for role in ROLES}
    def responder(team, role, body):
        seen[role] += 1
        if role != 'ceo':
            return calls(('write_file', dict(path='same.txt', content=f'{role}-{seen[role]}'))) if seen[role] % 2 else final(role + ' answer')
        if seen[role] == 1:
            return calls(('write_file', dict(path='same.txt', content='ceo-own')), ('ask_analyst', dict(role='growth', message='follow up')))
        return calls(('bash', dict(command=ADVANCE)))
    with make_team(mode, responder, 'stage3-history') as team:
        assert team.run(stop_after_day=7).day == 7
        hashes = {}
        for role, runtime in team.roles.items():
            git = lambda *args: team._git(runtime, *args)
            assert (runtime.workspace / '.git').is_dir()
            assert not git('for-each-ref')
            assert 'Week 1 (day 7) [week-1]' in git('log', '--format=%s')
            assert git('show', 'HEAD:same.txt').startswith(role)
            assert not (runtime.workspace / '.git/objects/info/alternates').exists()
            assert 'sessions/' not in git('ls-tree', '-r', '--name-only', 'HEAD')
            hashes[role] = git('rev-parse', 'HEAD')
        for message in team.messages:
            assert message.handoff_commit
            runtime = team.roles[message.receiver]
            assert team._git(runtime, 'show', message.handoff_commit + ':same.txt').startswith(message.receiver)
        assert len([r for r in team.messages if r.receiver == 'growth']) == 2
        for role, runtime in team.roles.items():
            for peer in ROLES:
                result = runtime.executor.run_private(['git', '-C', str(team.roles[peer].workspace), 'show', 'HEAD:same.txt'], capture_output=True)
                output = result.stdout.decode() + result.stderr.decode()
                assert (peer in READABLE_ROLES[role]) == (peer + '-' in output), output
                own_db = runtime.executor.run_private(['git', '-C', str(runtime.workspace), 'cat-file', '-t', hashes[peer]], capture_output=True)
                assert (own_db.returncode == 0) == (role == peer)
            sibling = next((p for p in ROLES if p not in READABLE_ROLES[role]), None)
            if sibling:
                link = runtime.workspace / 'peer-escape'
                link.symlink_to(team.roles[sibling].workspace / 'same.txt')
                assert runtime.executor.run_private(['cat', str(link)], capture_output=True).returncode != 0
                git_path = runtime.workspace / '.git/objects/info/alternates'
                git_path.write_text(str(team.roles[sibling].workspace / '.git/objects') + '\n')
                result = runtime.executor.run_private(['git', '-C', str(runtime.workspace), 'cat-file', '-t', hashes[sibling]], capture_output=True)
                assert result.returncode != 0
                git_path.unlink()


def test_pf_owner_every_reader_equal_bytes_private_layers_and_cursor_reopen():
    with make_team('pf', lambda *args: final('done'), 'stage3-acl') as team:
        for role in ROLES:
            write(team, role, 'EQUAL-CONTENT-ONCE')
            write(team, role, role + '-EXCLUSIVE-SECRET', 'private.txt')
            declare(team, role, role + ' claim', [role + ':same.txt@v1'])
        assert len({version(team, role) for role in ROLES}) == 3
        for role, runtime in team.roles.items():
            q = runtime.executor.pf_queries
            for owner in ROLES:
                accessible = owner in READABLE_ROLES[role]
                for operation, args in [('pf_read', dict(target={'path': owner + ':private.txt'})),
                                        ('pf_log', dict(target={'path': owner + ':private.txt'})),
                                        ('pf_blame', dict(target={'path': owner + ':private.txt'})),
                                        ('pf_dependencies', dict(target={'record': owner + ':r1'}, purpose='historical_only')),
                                        ('pf_dependents', dict(target={'path': owner + ':private.txt'})),
                                        ('pf_read', dict(target={'version': version(team, owner, 'private.txt')})),
                                        ('pf_read', dict(target={'version': owner + ':private.txt@v1'}))]:
                    result = runtime.executor.execute(operation, args)
                    assert (not result.startswith('Error:')) == accessible, (role, owner, operation, result)
                    if not accessible:
                        assert owner + '-EXCLUSIVE-SECRET' not in result
                matches = q.answer('pf_search', dict(text=owner + '-EXCLUSIVE-SECRET'))
                assert bool(matches['items']) == accessible
            event = runtime.store.begin_event('model_request', dict(context_id='context'))
            wire = runtime.store.version(event, 'wire', 'RAW-REQUEST-SECRET', layer='model_request_wire')
            runtime.store.complete(event)
            assert 'inaccessible' in runtime.executor.execute('pf_read', dict(target={'version': wire})).lower()
            assert q.answer('pf_search', dict(text='RAW-REQUEST-SECRET'))['total'] == 0
            reopened = SQLEvidenceStore(runtime.store.path, dict(runtime.store.identity))
            assert reopened.load_state('declaration:r1.1')['version_id'].split('/')[1] == role
            assert reopened.get_content(version(team, role))[0]['owner_role'] == role
            for layer in ('model_request_wire', 'model_source_occurrences', 'model_reconstructions', 'workspace_boundary', 'program_body'):
                ev = runtime.store.begin_event('fixture')
                raw = runtime.store.version(ev, 'secret', '[]' if layer in ('model_source_occurrences', 'model_reconstructions', 'workspace_boundary') else 'HOST-ONLY', layer=layer)
                runtime.store.complete(ev)
                for peer in ROLES:
                    refused = pf(team, peer, 'pf show ' + raw)
                    assert 'HOST-ONLY' not in refused and '[exit code: 1]' in refused
            runtime.store.assert_healthy()
        for i in range(3):
            write(team, 'ceo', 'page-' + str(i))
        growth = team.roles['growth']
        page = growth.executor.pf_queries.answer('pf_log', dict(target={'path': 'ceo:same.txt'}, limit=1))
        assert page['next_cursor']
        reopened = SQLEvidenceStore(growth.store.path, dict(growth.store.identity))
        from saas_bench.pf_queries import PFQueries
        q = PFQueries(TextRegistry(growth.workspace, 'pf', reopened, identity=growth.identity))
        rest = q.answer('pf_log', dict(cursor=page['next_cursor']))
        assert rest['items'] and all(item['owner'] == 'ceo' for item in rest['items'])
        assert team.roles['ops_finance'].executor.execute('pf_more', dict(cursor=page['next_cursor'])).startswith('Error:')


def test_ceo_copy_traversal_stops_without_ops_body_summary_or_snippet():
    with make_team('pf', lambda *args: final('done'), 'stage3-trace') as team:
        for role in ROLES[1:]:
            write(team, role, role + '-ORIGINAL-SOURCE')
            declare(team, role, role + '-ORIGINAL-CLAIM', [role + ':same.txt@v1'])
        declare(team, 'ceo', 'CEO composite decision', ['growth:r1.1', 'ops_finance:r1.1'])
        result = pf(team, 'growth', 'pf depend ceo:r1 --history --detail')
        assert 'inaccessible' in result.lower(), result
        assert 'ops_finance-ORIGINAL-SOURCE' not in result and 'ops_finance-ORIGINAL-CLAIM' not in result
        assert 'growth' in result and 'CEO composite decision' in result
        result = pf(team, 'ceo', 'pf depend ceo:r1 --history --detail')
        assert 'growth-ORIGINAL-CLAIM' in result and 'ops_finance-ORIGINAL-CLAIM' in result
        for runtime in team.roles.values():
            runtime.store.assert_healthy()


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_notes_schema_unicode_boundaries_links_reopen_and_symlink_protection(mode):
    git = {t['name']: t for t in get_bash_agent_tool_descriptions(text_registration=True)}
    pf_defs = {t['name']: t for t in get_bash_agent_tool_descriptions(text_registration=True, pf_queries=True)}
    for tool in ('bash', 'write_file', 'edit_file'):
        assert git[tool]['parameters'] == pf_defs[tool]['parameters']
        assert 'note' not in git[tool]['parameters']['required']
    with make_team(mode, lambda *args: final('done'), 'stage3-notes') as team:
        for role in ROLES:
            runtime = team.roles[role]
            for length in (199, 200, 201):
                write(team, role, role, note='汉🙂' * (length // 2) + ('汉' if length % 2 else ''))
            result = runtime.executor.execute('edit_file', dict(path='same.txt', old_string=role, new_string=role + '-edited', note='edit reason'))
            assert not result.startswith('Error:'), result
            assert 'tool output' in runtime.executor.execute('bash', dict(command="printf 'bash-owned' > bash.txt; printf 'tool output'", note='bash reason'))
            for tool, args in [('bash', dict(command='true')), ('write_file', dict(path='same.txt', content='x')), ('edit_file', dict(path='same.txt', old_string=role, new_string='x'))]:
                assert runtime.executor.execute(tool, dict(args, note=123)) == 'Error: note must be a string'
            entries = [json.loads(line) for line in (runtime.workspace / '.tool-notes.jsonl').read_text().splitlines()]
            assert [len(row['note']) for row in entries[:3]] == [199, 200, 200]
            assert all(row['author'] == row['role'] == role for row in entries)
            assert entries[-2]['files'] == ['same.txt'] and entries[-1]['files'] == ['bash.txt']
            assert len(entries[-1]['output_sha256']) == 64 and 'output' not in entries[-1]
            write(team, role, 'omitted', 'without-note.txt')
            assert len((runtime.workspace / '.tool-notes.jsonl').read_text().splitlines()) == len(entries)
            runtime.executor.execute('bash', dict(command='exit 2', note='failed command'))
            runtime.executor.execute('bash', dict(command='rm bash.txt', note='remove file'))
            added = [json.loads(line) for line in (runtime.workspace / '.tool-notes.jsonl').read_text().splitlines()][-2:]
            assert added[0]['status'] == 'failed' and added[1]['files'] == ['bash.txt']
            for viewer in ROLES:
                read_log = team.roles[viewer].executor.execute('read_file', dict(path=str(runtime.workspace / '.tool-notes.jsonl')))
                assert (not read_log.startswith('Error:')) == (role in READABLE_ROLES[viewer])
            read = runtime.executor.execute('read_file', dict(path='.tool-notes.jsonl'))
            assert 'bash reason' in read and 'edit reason' in read
            if mode == 'pf':
                reopened = SQLEvidenceStore(runtime.store.path, dict(runtime.store.identity))
                handles = evidence_handles.index(reopened)
                assert handles.note(version(team, role))[1] == 'edit reason'
            log = runtime.workspace / '.tool-notes.jsonl'
            log.unlink()
            log.symlink_to(team.root / 'private' / role / 'identity.json')
            original = (team.root / 'private' / role / 'identity.json').read_bytes()
            assert runtime.executor.execute('bash', dict(command='true', note='blocked')).startswith('Error:')
            assert (team.root / 'private' / role / 'identity.json').read_bytes() == original


def test_recipient_session_context_full_delta_unchanged_from_final_provider_wire():
    with make_team('pf', lambda *args: final('done'), 'stage3-reader') as team:
        content = ''.join(f'line {i:03d} business observation. ' + 'z' * 80 + '\n' for i in range(120))
        write(team, 'growth', content)
        first_growth = pf(team, 'growth', 'pf show growth:same.txt')
        wire, ledger = send(team, 'growth', [first_growth])
        assert ledger[0]['mode'] == 'FULL'
        first_ceo = pf(team, 'ceo', 'pf show growth:same.txt')
        wire, ledger = send(team, 'ceo', [CapturedText('growth says its source was read'), first_ceo])
        assert ledger[0]['mode'] == 'FULL'
        unchanged = pf(team, 'ceo', 'pf show growth:same.txt')
        wire, ledger = send(team, 'ceo', [first_ceo, unchanged])
        assert ledger[-1]['mode'] == 'UNCHANGED', ledger
        assert apply_delta(content, ledger[-1]['edits']) == content
        updated = content.replace('line 012', 'edit 012')
        write(team, 'growth', updated)
        delta = pf(team, 'ceo', 'pf show growth:same.txt')
        wire, ledger = send(team, 'ceo', [first_ceo, delta])
        assert ledger[-1]['mode'] == 'DELTA', ledger
        assert apply_delta(content, ledger[-1]['edits']) == updated
        assert wire['messages'][0]['content'].endswith(content)
        assert ledger[-1]['reader'] == 'ceo' and ledger[-1]['session_id'] == team.roles['ceo'].identity.session_id
        assert ledger[-1]['actual_tokens'] < ledger[-1]['full_tokens']
        assert ledger[-1]['tokenizer']['tokenizer_id'].endswith('/v41')
        fresh = pf(team, 'ceo', 'pf show growth:same.txt')
        wire, ledger = send(team, 'ceo', [fresh])
        assert ledger[-1]['mode'] == 'FULL'
        team._rotate_sessions()
        wire, ledger = send(team, 'ceo', [fresh])
        assert ledger[-1]['mode'] == 'FULL'
        assert wire['messages'][0]['content'].endswith(updated)
        for runtime in team.roles.values():
            runtime.store.assert_healthy()


def test_git_role_same_name_scripts_registrations_current_files_and_reopened_references():
    with make_team('git', lambda *args: final('done'), 'stage3-git-bind') as team:
        for role in ROLES:
            write(team, role, role + '-CURRENT')
            write(team, role, 'print(' + repr(role) + ')', 'analysis.py')
            declare(team, role, role + ' registered decision', ['same.txt'])
        ceo = team.roles['ceo']
        for peer in ROLES:
            before_commit = ceo.executor.execute('read_file', dict(path=str(team.roles[peer].workspace / 'same.txt')))
            assert peer + '-CURRENT' in before_commit
        for role, runtime in team.roles.items():
            team._snapshot_history(runtime, 'Week 1 (day 7) [week-1]')
            assert role in team._git(runtime, 'show', 'HEAD:analysis.py')
            committed = json.loads(team._git(runtime, 'show', 'HEAD:registrations.json'))
            assert committed['records']['r1'][0]['author'] == role
        receipt = declare(team, 'ceo', 'compare analyst recommendations', ['growth:same.txt@week-1', 'ops_finance:same.txt@week-1'])
        assert 'growth:same.txt@week-1' in receipt and 'ops_finance:same.txt@week-1' in receipt
        reopened = TextRegistry(ceo.workspace, 'git', identity=ceo.identity)
        reopened.git_run = ceo.executor.run_private
        refs = reopened._load()['records']['r2'][0]['references']
        assert [r['evidence']['owner'] for r in refs] == ['growth', 'ops_finance']
        write(team, 'growth', 'growth-CHANGED')
        assert 'growth:same.txt@week-1' in reopened.weekly_check(0)
        assert 'growth-CURRENT' == team._git(team.roles['growth'], 'show', 'HEAD:same.txt')
        assert not list((team.root / 'private').rglob('*.sqlite'))


def test_public_observations_canonical_numbered_nested_handles_and_late_capture():
    with make_team('pf', lambda *args: final('done'), 'stage3-public-handles') as team:
        def query(role, sql, **extra):
            command = './novamind-operation query ' + shlex.quote(sql)
            result = team.roles[role].executor.execute('bash', dict(command=command, **extra))
            assert '[exit code:' not in result, result
            store = team.roles[role].store
            with closing(store.connect()) as conn:
                row = conn.execute("SELECT v.version_id FROM versions v JOIN requests r USING(event_id) WHERE r.branch=? AND json_extract(v.metadata,'$.layer')='server_public_response' ORDER BY v.rowid DESC LIMIT 1", (role,)).fetchone()
            return row[0]
        ceo_b = query('ceo', 'SELECT 2 AS public_b')
        growth_a = query('growth', 'SELECT 1 AS public_a')
        growth_b = query('growth', 'SELECT 2 AS public_b')
        assert team.roles['growth'].store.get_content(growth_b)[0]['previous_version'] is None
        ceo, growth = team.roles['ceo'], team.roles['growth']
        producer = evidence_handles.index(growth.store)
        ha, hb = producer.name(growth_a), producer.name(growth_b)
        assert ha != hb
        consumer = evidence_handles.index(ceo.store)
        assert consumer.lookup(ha)[-1] == growth_a and consumer.lookup(hb)[-1] == growth_b
        reopened = SQLEvidenceStore(ceo.store.path, dict(ceo.store.identity))
        assert evidence_handles.index(reopened).lookup(ha)[-1] == growth_a
        assert 'public_a' in pf(team, 'ceo', 'pf show ' + ha)
        assert 'public_b' in pf(team, 'ceo', 'pf log ' + hb.split('@')[0])
        ops_public = query('ops_finance', 'SELECT 3 AS SHARED_PUBLIC', note='OPS-NOTE-PRIVATE')
        assert 'SHARED_PUBLIC' in pf(team, 'growth', 'pf show ' + ops_public)
        growth.executor.execute('write_file', dict(path='reports/marker.txt', content='directory'))
        public_from_cwd = pf(team, 'growth', 'cd reports && pf show ' + ops_public)
        assert 'SHARED_PUBLIC' in public_from_cwd and 'OPS-NOTE-PRIVATE' not in public_from_cwd
        assert growth.executor.pf_queries.answer('pf_search', dict(text='OPS-NOTE-PRIVATE'))['total'] == 0
        assert ceo.executor.pf_queries.answer('pf_search', dict(text='OPS-NOTE-PRIVATE'))['total'] > 0
        shared_name = evidence_handles.index(team.roles['ops_finance'].store).name(ops_public)
        assert 'SHARED_PUBLIC' in pf(team, 'growth', 'pf show ' + shared_name.split('@')[0])
        event = team.roles['ops_finance'].store.begin_event('dashboard_generation', dict(day=0))
        dashboard = team.roles['ops_finance'].store.version(event, 'public', 'SHARED-DASHBOARD', layer='dashboard', object_id='dashboard')
        team.roles['ops_finance'].store.complete(event)
        assert 'SHARED-DASHBOARD' in pf(team, 'growth', 'pf show ' + dashboard)
        assert 'SHARED-DASHBOARD' in pf(team, 'growth', 'pf show ops_finance:dashboard')
        write(team, 'growth', 'nested old', 'reports/same.txt')
        write(team, 'growth', 'nested new', 'reports/same.txt')
        write(team, 'ceo', 'CEO distinct nested file', 'reports/same.txt')
        assert 'nested old' in ceo.executor.execute('pf_read', dict(target={'path': 'growth:reports/same.txt@v1'}))
        assert 'nested old' in pf(team, 'ceo', 'pf show growth:reports/same.txt@v1')
        assert 'nested new' in pf(team, 'ceo', 'pf diff growth:reports/same.txt')
        assert 'growth:reports/same.txt@v1' in pf(team, 'ceo', 'pf log growth:reports/same.txt')
        late = growth.store.begin_event('write_file', dict(path='late.txt', content='LATE-BUT-READABLE'))
        ceo.executor.pf_queries.answer('pf_search', dict(text='LATE-BUT-READABLE'))
        consumer.refresh()
        late_version = growth.store.version(late, 'after', 'LATE-BUT-READABLE', layer='file_bytes', object_id='late.txt')
        growth.store.complete(late)
        assert 'LATE-BUT-READABLE' in pf(team, 'ceo', 'pf show growth:late.txt@v1')
        assert consumer.name(late_version) == 'growth:late.txt@v1'
        write(team, 'growth', 'literal ' + 'x' * 5000, 'literal.txt')
        literal = ceo.executor.execute('read_file', dict(path=str(growth.workspace / 'literal.txt')))
        saved = pf(team, 'ceo', 'pf show growth:literal.txt')
        wire, ledger = send(team, 'ceo', [literal, saved])
        assert ledger[-1]['mode'] == 'UNCHANGED'
        assert 'literal ' in wire['messages'][0]['content']
        for runtime in team.roles.values():
            runtime.store.assert_healthy()
