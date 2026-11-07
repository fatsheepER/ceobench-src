import json
import re
import shlex
from pathlib import Path

import pytest

from test_pf_publication import close
from test_preflight_team_checkpoint import fast_simulator
from test_preflight_team_flow import ADVANCE, calls, final, raw
from test_team_script_state import register_cli
from test_receipt_outputs import artifact
from test_shared_workspace import shared_team


def test_workspace_status_is_available_through_public_pf_cli():
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        executor = team.roles['growth'].executor
        output = executor.execute('bash', {'command': 'pf workspace status'})
        assert 'published_head=' in output, output
        assert 'drafts=0' in output, output
        assert executor.last_status == 'succeeded', output


def test_workspace_private_native_git_commit_survives_next_call():
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        worker = team.roles['growth'].executor
        worker.execute('write_file', dict(path='plans/git.txt', content='private git history\n'))
        output = worker.execute('bash', {'command': "git add plans/git.txt && git -c user.name=Test -c user.email=test@example.invalid commit -m 'my private commit' && git rev-parse HEAD"})
        assert 'my private commit' in output, output
        output = worker.execute('bash', {'command': 'git log -1 --format=%s'})
        assert 'my private commit' in output, output


def test_workspace_revision_resolution_and_discard_use_real_cli():
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        owner, peer = (team.roles[r].executor for r in ('growth', 'ops_finance'))
        owner.execute('write_file', dict(path='plans/value.txt', content='old\n'))
        peer.execute('read_file', dict(path='plans/value.txt'))
        owner.execute('write_file', dict(path='plans/value.txt', content='shared\n'))
        output = peer.execute('write_file', dict(path='plans/value.txt', content='mine\n'))
        assert 'VERSION CHANGED' in output and '--- plans/value.txt base@' in output, output
        publication = team.workspace_publication
        row = publication.draft_rows('ops_finance')[0]
        ref = publication.draft_ref(row)
        head = publication.head()
        output = peer.execute('bash', dict(command='pf workspace resolve ' + ref + ' --against ' + head))
        assert '[PF workspace published]' in output, output
        assert (team.roles['ceo'].workspace / 'plans/value.txt').read_text() == 'mine\n'
        output = owner.execute('write_file', dict(path='plans/value.txt', content='another draft\n'))
        row = publication.draft_rows('growth')[0]
        ref = publication.draft_ref(row)
        output = owner.execute('bash', dict(command='pf workspace discard ' + ref))
        assert 'discarded' in output, output
        assert publication.draft_rows('growth') == []
        assert 'mine' in owner.execute('read_file', dict(path='plans/value.txt'))



def make_draft(team, role, path='plans/draft.txt', content='PRIVATE-DRAFT\n'):
    worker = team.roles[role].executor
    worker.execute('write_file', dict(path=path, content='base\n'))
    publication = team.workspace_publication
    peer = publication.begin('peer-' + role, automatic=True)
    (peer.root / path).write_text('shared\n')
    close(publication, peer, store=team.roles[role].store)
    output = worker.execute('write_file', dict(path=path, content=content))
    assert 'needs_confirmation' in output, output
    return publication.draft_rows(role)[-1], output


def test_same_bash_edits_defer_then_standalone_advance_succeeds():
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        team._set_phase('ceo')
        worker = team.roles['ceo'].executor
        output = worker.execute('bash', dict(command="printf changed > plans/new.txt; " + ADVANCE))
        assert 'workspace_tool_pending' in output, output
        assert team.server.tools.current_day == 0
        assert 'retained_partial' in output
        row = team.workspace_publication.draft_rows('ceo')[0]
        solved = worker.execute('bash', dict(command='pf workspace resolve ' + team.workspace_publication.draft_ref(row) + ' --against ' + team.workspace_publication.head()))
        assert 'published' in solved
        assert (team.roles['ceo'].workspace / 'plans/new.txt').read_text() == 'changed'
        output = worker.execute('bash', dict(command=ADVANCE))
        assert team.server.tools.current_day == 7, output
        assert not team.server._operation_failed
        assert worker._bash_facts['boundary_closed']


def test_fixed_against_race_retains_candidate_without_changing_base():
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        row, _ = make_draft(team, 'growth')
        publication = team.workspace_publication
        original = json.loads(publication.get('candidates', row['candidate']))
        head = publication.head()
        peer = publication.begin('peer-growth', automatic=True)
        (peer.root / 'plans/draft.txt').write_text('newer\n')
        close(publication, peer, store=team.roles['growth'].store)
        output = team.roles['growth'].executor.execute('bash', dict(command='pf workspace resolve ' + publication.draft_ref(row) + ' --against ' + head))
        assert 'needs_confirmation' in output, output
        row = publication.draft_rows('growth')[0]
        saved = json.loads(publication.get('candidates', row['candidate']))
        assert saved['bases'] == original['bases'] and saved['public'] == original['public']
        assert (publication.public / 'plans/draft.txt').read_text() == 'newer\n'
        assert not team.server._operation_failed


def test_script_new_public_inputs_registered_source_and_old_failed_draft():
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        worker = team.roles['growth'].executor
        code = """from pathlib import Path
import adjacent
import novamind_api as nm
print('SOURCE=' + Path(__file__).read_text().splitlines()[0])
print('INPUT=' + Path('data/input.txt').read_text())
print('MODULE=' + adjacent.VALUE)
print('DAY=' + str(nm.vars.current_day))
Path('outputs/old.txt').write_text('literal old value')
raise ValueError('partial')
"""
        for path, value in [('scripts/adjacent.py', "VALUE='fresh'"), ('data/input.txt', 'one')]:
            worker.execute('write_file', dict(path=path, content=value))
        register_cli(team, 'growth', 'scripts/registered.py', code)
        first = team.server._run_daily_scripts_internal('growth')['scripts/registered.py']
        assert 'INPUT=one' in first and 'MODULE=fresh' in first and 'DAY=0' in first, first
        assert 'retained_partial' in first, first
        original = team.workspace_publication.draft_rows('growth')
        worker.execute('write_file', dict(path='data/input.txt', content='two'))
        second = team.server._run_daily_scripts_internal('growth')['scripts/registered.py']
        assert 'INPUT=two' in second and 'SOURCE=from pathlib import Path' in second, second
        assert not (team.roles['ceo'].workspace / 'outputs/old.txt').exists()
        drafts = team.workspace_publication.draft_rows('growth')
        assert len(drafts) == 2 and original[0] in drafts
        record = team.server.role_script_results['growth'][0]
        assert record['classification'] == 'script_error' and record['facts']['boundary_closed']
        assert Path(record['evidence'], 'stdout.bin').read_text().startswith('SOURCE=')
        assert not list((team.root / 'private/workspace-publication/executions').rglob('.registered-*.py'))


@pytest.mark.parametrize('failure', ['nonzero', 'timeout'])
def test_successful_api_and_failed_file_keep_separate_outcomes_without_replay(failure):
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('ceo')
        worker = team.roles['ceo'].executor
        worker.bash_timeout = 2 if failure == 'timeout' else worker.bash_timeout
        output = worker.execute('bash', dict(command="python -c \"import novamind_api as nm; print(nm.pricing.set_prices(A=21, B=41, C=81))\"; printf private > plans/partial.txt; " + ("sleep 60" if failure == "timeout" else "exit 3")))
        assert worker._bash_facts['boundary_closed']
        assert ('timed out' if failure == 'timeout' else 'exit code') in output
        assert 'retained_partial' in output, output
        assert '21' in output, output
        assert not (team.roles['ceo'].workspace / 'plans/partial.txt').exists()
        row = team.workspace_publication.draft_rows('ceo')[0]
        before = team.server.conn.execute('SELECT * FROM config_history').fetchall()
        output = worker.execute('bash', dict(command='pf workspace resolve ' + team.workspace_publication.draft_ref(row) + ' --against ' + team.workspace_publication.head()))
        assert 'published' in output, output
        assert team.server.conn.execute('SELECT * FROM config_history').fetchall() == before
        assert not team.server._operation_failed


def test_week_finish_confirmation_binds_author_day_and_revisions():
    turns = {}
    tokens = {}
    def respond(team, role, body):
        turns[role] = turns.get(role, 0) + 1
        text = json.dumps(body['messages'])
        if role == 'ops_finance':
            assert 'PRIVATE-DRAFT' not in text
            return final('ops ordinary weekly advice')
        if role == 'ceo':
            assert 'PRIVATE-DRAFT' not in text and 'Weekly finish not completed' not in text
            return calls(('bash', dict(command=ADVANCE)))
        if turns[role] == 1:
            assert 'Your unpublished drafts remain private.' in text
            return final('weekly message before draft confirmation')
        if turns[role] == 2:
            reminder = body['messages'][-1]['content']
            assert 'Weekly finish not completed' in reminder
            tokens[role] = re.search(r'--confirm-finish ([a-f0-9]+)', reminder).group(1)
            return calls(('bash', dict(command='pf workspace status --confirm-finish ' + tokens[role])))
        assert turns[role] == 3, body
        assert 'finish_confirmed' in text
        return final('EXACT-FINAL-MESSAGE')
    with shared_team('pf', respond, workspace_protocol='snapshot-v1') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        row, _ = make_draft(team, 'growth')
        result = team.run(stop_after_day=7)
        assert result.reason == 'observation_end', team.failure
        assert len(team.workspace_publication.draft_rows('growth')) == 1
        assert not team.workspace_publication.check_finish('growth', 0)
        assert team.workspace_publication.check_finish('growth', 7)
        assert team.workspace_publication.week_reminder('growth', 7)
        assert not team.workspace_publication.week_reminder('growth', 7)
        assert any(message.reply == 'EXACT-FINAL-MESSAGE' for message in team.messages)
        rows = [json.loads(line) for line in team.roles['growth'].usage.path.read_text().splitlines()]
        requests = [row['body'] for row in rows if row['event'] == 'http_request']
        assert requests and sum(m.get('content','').count('Your unpublished drafts remain private.') for m in requests[-1]['messages']) == 1
        artifact('finish-wire.json', requests)


@pytest.mark.parametrize('api', ['chat', 'responses', 'messages'])
def test_three_provider_actual_receipts_retry_confirmation_and_continue(tmp_path, monkeypatch, api):
    import httpx
    from anthropic import Anthropic
    from openai import OpenAI
    from saas_bench.agents.bash_agent.agent import BashAgent, Message, FinalText
    from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
    from saas_bench.model_usage import ModelUsage
    from test_preflight_usage import reply
    requests = []
    def respond(request):
        requests.append(json.loads(request.read()))
        if len(requests) == 1:
            return httpx.Response(503, json={'error': {'type': 'server_error', 'message': 'offline retry'}})
        body = reply(api)
        if api != 'messages':
            return httpx.Response(200, json=body)
        events = [dict(type='message_start', message=body), dict(type='message_stop')]
        payload = ''.join('event: ' + event['type'] + '\ndata: ' + json.dumps(event) + '\n\n' for event in events)
        return httpx.Response(200, content=payload, headers={'content-type': 'text/event-stream'})
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        row, receipt = make_draft(team, 'growth', content='PRIVATE-LONG-' + 'detail\n' * 2000 + 'END-PRIVATE-DIFF\n')
        assert '[diff truncated]' in receipt and 'END-PRIVATE-DIFF' not in receipt
        worker = team.roles['growth'].executor
        detail = worker.execute('bash', dict(command='pf workspace resolve ' + team.workspace_publication.draft_ref(row) + ' --full'))
        assert 'END-PRIVATE-DIFF' in detail, detail[-2000:]
        reminder = team.workspace_publication.check_finish('growth', 0)
        signature = re.search(r'--confirm-finish ([a-f0-9]+)', reminder).group(1)
        client = (Anthropic if api == 'messages' else OpenAI)(api_key='offline', max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        usage = ModelUsage(tmp_path / (api + '.jsonl'), 'growth', evidence_store=team.roles['growth'].store)
        monkeypatch.setattr(usage.lifecycle, 'sleep', lambda _: None)
        agent = BashAgent(get_bash_agent_tool_descriptions(True, True), client, model='offline',
            reasoning_effort='low' if api == 'responses' else None, allow_final_text=True,
            system_prompt=team.roles['growth'].agent.system_prompt, workspace_path=worker.workspace_path,
            usage_recorder=usage, pf=True, text_registration=False, capabilities=team.capabilities)
        agent.current_day = 0
        agent.conversation = [Message(role='system', content=agent.system_prompt),
            Message(role='assistant', content='', tool_calls=[dict(id='draft', type='function', function=dict(name='write_file', arguments='{}'))])]
        agent._pending_tool_calls = [dict(id='draft', name='write_file', arguments={})]
        agent.record_tool_result(receipt)
        agent.add_input(team.workspace_publication.week_reminder('growth', 0), day=0)
        assert not team.workspace_publication.week_reminder('growth', 0)
        agent.add_input(reminder, day=0)
        try:
            assert isinstance(agent.act('', 0, False, {'day': 0}), FinalText)
            assert requests[0] == requests[1]
            assert 'VERSION CHANGED' in json.dumps(requests[1]) and 'diff truncated' in json.dumps(requests[1])
            assert signature in json.dumps(requests[1])
            assert json.dumps(requests[1]).count('Your unpublished drafts remain private.') == 1
            confirmation = worker.execute('bash', dict(command='pf workspace status --confirm-finish ' + signature))
            agent.add_input(confirmation, day=0)
            assert isinstance(agent.act('', 0, False, {'day': 0}), FinalText)
            assert 'finish_confirmed' in json.dumps(requests[-1])
            assert not team.workspace_publication.check_finish('growth', 0)
            assert team.workspace_publication.draft_rows('growth') == [row]
            usage_rows = [json.loads(line) for line in usage.path.read_text().splitlines()]
            assert [r['body'] for r in usage_rows if r['event'] == 'http_request'] == requests
            assert not team.roles['ops_finance'].usage.path.exists()
            artifact(api + '-actual-model-requests.json', requests)
        finally:
            client.close()


@pytest.mark.parametrize('operation', ['checkout', 'reset'])
def test_private_git_objects_remain_private_after_discard_and_gc(operation):
    from test_pf_candidates import query
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        row, _ = make_draft(team, 'growth', content='CANDIDATE-ONLY-SECRET\n')
        worker = team.roles['growth'].executor
        committed = worker.execute('bash', dict(command="git add plans/draft.txt && git -c user.name=Test -c user.email=t@example.invalid commit -m private && git rev-parse HEAD && git gc"))
        commit = re.search(r'(?m)^[0-9a-f]{40}$', committed).group()
        worker.execute('bash', dict(command='pf workspace discard ' + team.workspace_publication.draft_ref(row)))
        output = worker.execute('bash', dict(command='git show ' + commit + ':plans/draft.txt'))
        assert 'CANDIDATE-ONLY-SECRET' in output
        peer = team.roles['ops_finance'].executor
        missing = peer.execute('bash', dict(command='git show ' + commit + ':plans/draft.txt'))
        assert 'CANDIDATE-ONLY-SECRET' not in missing and '[exit code:' in missing
        listing, page = query(peer, 'pf candidates --role all --history --kind stdout --text CANDIDATE-ONLY-SECRET')
        assert page['items'] == [], listing
        command = 'git checkout ' + commit + ' -- plans/draft.txt' if operation == 'checkout' else 'git reset --hard ' + commit
        changed = worker.execute('bash', dict(command=command))
        assert 'needs_confirmation' in changed, changed
        assert (team.roles['ceo'].workspace / 'plans/draft.txt').read_text() == 'shared\n'


def test_mutating_workspace_persistence_failure_holds_and_does_not_continue(monkeypatch):
    from saas_bench.workspace_publication import PublicationHold
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        row, _ = make_draft(team, 'growth')
        publication = team.workspace_publication
        original = publication.stage
        def fail(stage, execution=None):
            if stage == 'manifest_durable':
                raise OSError('publication persistence failure')
            return original(stage, execution)
        monkeypatch.setattr(publication, 'stage', fail)
        with pytest.raises(PublicationHold):
            team.roles['growth'].executor.execute('bash', dict(command='pf workspace resolve ' + publication.draft_ref(row) + ' --against ' + publication.head()))
        assert team.roles['growth'].executor.last_status == 'result_unknown'
        assert team.roles['growth'].executor._bash_facts['boundary_closed']
        assert not team.roles['growth'].executor.preserved_process
        assert publication.draft_rows('growth') == [row]
        assert team.roles['growth'].store.fault


def test_ceo_opening_does_not_finish_and_confirmed_draft_allows_advance():
    counts = {}
    def respond(team, role, body):
        counts[role] = counts.get(role, 0) + 1
        rendered = json.dumps(body)
        if role != 'ceo':
            assert 'CEO-PRIVATE' not in rendered and '--confirm-finish ' not in body['messages'][-1]['content']
            return final('ordinary advice')
        if counts[role] in (1, 3):
            return calls(('bash', dict(command=ADVANCE)))
        assert counts[role] == 2
        result = next(m['content'] for m in reversed(body['messages']) if m['role'] == 'tool')
        assert 'draft_confirmation_required' in result
        token = re.search(r'--confirm-finish ([a-f0-9]+)', result).group(1)
        return calls(('bash', dict(command='pf workspace status --confirm-finish ' + token)))
    with shared_team('pf', respond, workspace_protocol='snapshot-v1') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        make_draft(team, 'ceo', content='CEO-PRIVATE\n')
        result = team.run(stop_after_day=7)
        assert result.reason == 'observation_end', team.failure
        assert counts['ceo'] == 3
        assert team.workspace_publication.draft_rows('ceo')
        assert not team.workspace_publication.check_finish('ceo', 0)


def test_changed_draft_and_wrong_role_cannot_reuse_confirmation():
    from saas_bench.workspace_publication import WorkspacePublication
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        make_draft(team, 'growth')
        publication = team.workspace_publication
        reminder = publication.check_finish('growth', 0)
        token = re.search(r'--confirm-finish ([a-f0-9]+)', reminder).group(1)
        other = team.roles['ops_finance'].executor.execute('bash', dict(command='pf workspace status --confirm-finish ' + token))
        assert 'Draft set changed' in other
        worker = team.roles['growth'].executor
        assert 'finish_confirmed' in worker.execute('bash', dict(command='pf workspace status --confirm-finish ' + token))
        reopened = WorkspacePublication(publication.public, publication.private)
        assert not reopened.check_finish('growth', 0)
        worker.execute('write_file', dict(path='plans/draft.txt', content='revised private draft'))
        assert 'Draft set changed' in worker.execute('bash', dict(command='pf workspace status --confirm-finish ' + token))
        assert publication.check_finish('growth', 0)


def test_automatic_script_snapshot_concurrency_and_literal_replacement(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        worker = team.roles['growth'].executor
        worker.execute('write_file', dict(path='data/input.txt', content='before'))
        code = "from pathlib import Path\nprint(Path('data/input.txt').read_text())\nPath('outputs/result.txt').write_text('literal old')\n"
        register_cli(team, 'growth', 'scripts/literal.py', code)
        publication = team.workspace_publication
        ready, release = threading.Event(), threading.Event()
        original = publication.stage
        def stage(name, execution=None):
            if name == 'execution_started':
                from contextlib import closing
                with closing(publication.connect()) as conn:
                    row = conn.execute('SELECT record FROM executions WHERE id=?', (execution,)).fetchone()
                if json.loads(row[0]).get('automatic') and not ready.is_set():
                    ready.set()
                    assert release.wait(10)
            return original(name, execution)
        monkeypatch.setattr(publication, 'stage', stage)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(team.server._run_daily_scripts_internal, 'growth')
            assert ready.wait(10)
            try:
                worker.execute('write_file', dict(path='data/input.txt', content='after'))
            finally:
                release.set()
            first = future.result()['scripts/literal.py']
        assert 'before' in first and 'after' not in first, first
        peer = publication.begin('peer-literal', automatic=True)
        (peer.root / 'outputs/result.txt').write_text('newer business value')
        close(publication, peer, store=team.roles['growth'].store)
        second = team.server._run_daily_scripts_internal('growth')['scripts/literal.py']
        assert 'after' in second, second
        assert (publication.public / 'outputs/result.txt').read_text() == 'literal old'
        assert not publication.draft_rows('growth')


def test_registered_source_preparation_failure_closes_execution_and_next_script_runs(monkeypatch):
    from contextlib import contextmanager, closing
    import saas_bench.api_server as api
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        register_cli(team, 'growth', 'scripts/first.py', "print('first')")
        register_cli(team, 'growth', 'scripts/second.py', "print('second')")
        original = api.registered_script_source
        @contextmanager
        def prepare(executor, name, code):
            if name == 'scripts/first.py':
                raise api.RegisteredScriptPreparationError(OSError('temporary source could not be prepared'))
            with original(executor, name, code) as path:
                yield path
        monkeypatch.setattr(api, 'registered_script_source', prepare)
        output = team.server._run_daily_scripts_internal('growth')
        assert 'second' in output['scripts/second.py']
        assert [r['classification'] for r in team.server.role_script_results['growth']] == ['script_error', 'succeeded']
        with closing(team.workspace_publication.connect()) as conn:
            assert conn.execute('SELECT count(*) FROM executions WHERE outcome IS NULL').fetchone()[0] == 0


def test_private_git_host_symlink_is_not_dereferenced(tmp_path):
    from saas_bench.workspace_publication import PublicationHold
    host_file = tmp_path / 'host-private.txt'
    host_file.write_text('HOST-PRIVATE-BYTES')
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        worker = team.roles['growth'].executor
        with pytest.raises(PublicationHold, match='unsupported links'):
            worker.execute('bash', dict(command='ln -s ' + shlex.quote(str(host_file)) + ' .git/host-link'))
        assert not list((team.workspace_publication.private / 'git-repositories').rglob('host-link'))
        assert worker._bash_facts['boundary_closed']


@pytest.mark.parametrize('action', ['discard', 'resolve'])
def test_edit_after_resolving_inherited_draft_remains_actionable(action):
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        row, _ = make_draft(team, 'growth')
        publication = team.workspace_publication
        ref = publication.draft_ref(row)
        command = 'pf workspace ' + action + ' ' + ref
        if action == 'resolve':
            command += ' --against ' + publication.head()
        worker = team.roles['growth'].executor
        output = worker.execute('bash', dict(command=command + '; printf revised > plans/draft.txt'))
        assert 'not_published' in output, output
        rows = publication.draft_rows('growth')
        assert len(rows) == 1, output
        new_ref = publication.draft_ref(rows[0])
        assert new_ref != ref and new_ref in output
        _, candidate = publication.draft('growth', new_ref)
        assert publication.get('blobs', candidate['files']['plans/draft.txt']['blob']) == b'revised'
        assert new_ref in publication.check_finish('growth', 0)
        output = worker.execute('bash', dict(command='pf workspace resolve ' + new_ref + ' --against ' + publication.head()))
        assert 'published' in output, output
        assert (publication.public / 'plans/draft.txt').read_text() == 'revised'
        assert publication.draft_rows('growth') == []


def test_mode_only_draft_confirmation_shows_shared_and_draft_modes():
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        worker = team.roles['growth'].executor
        worker.execute('write_file', dict(path='plans/mode.txt', content='unchanged\n'))
        worker.execute('bash', dict(command='chmod 644 plans/mode.txt'))
        publication = team.workspace_publication
        peer = publication.begin('mode-peer', automatic=True)
        (peer.root / 'plans/mode.txt').chmod(0o600)
        close(publication, peer, store=team.roles['growth'].store)
        output = worker.execute('bash', dict(command='chmod 755 plans/mode.txt'))
        assert 'needs_confirmation' in output, output
        row = publication.draft_rows('growth')[0]
        output = worker.execute('bash', dict(command='pf workspace resolve ' + publication.draft_ref(row) + ' --full'))
        assert 'old mode 0644' in output, output
        assert 'new mode 0600' in output and 'new mode 0755' in output, output
        assert 'plans/mode.txt shared' in output and 'plans/mode.txt draft' in output, output


def test_private_git_root_link_is_rejected_before_host_reads(tmp_path, monkeypatch):
    from saas_bench.workspace_publication import PublicationHold
    host_git = tmp_path / 'host-git'
    host_git.mkdir()
    secret = host_git / 'private-marker'
    secret.write_text('HOST-PRIVATE')
    read_bytes = Path.read_bytes
    reads = []
    def observe(path):
        if path.resolve() == secret:
            reads.append(path)
        return read_bytes(path)
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        monkeypatch.setattr(Path, 'read_bytes', observe)
        worker = team.roles['growth'].executor
        with pytest.raises(PublicationHold):
            worker.execute('bash', dict(command='rm -rf .git; ln -s ' + shlex.quote(str(host_git)) + ' .git'))
        assert not reads
        assert worker._bash_facts['boundary_closed']


def test_private_git_sync_failure_holds_before_author_pointer_changes(monkeypatch):
    import os
    from saas_bench.workspace_publication import PublicationHold
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        worker = team.roles['growth'].executor
        fsync = os.fsync
        def fail_git(descriptor):
            if '/git-repositories/' in os.readlink('/proc/self/fd/' + str(descriptor)):
                raise OSError('private Git persistence fault')
            return fsync(descriptor)
        monkeypatch.setattr(os, 'fsync', fail_git)
        with pytest.raises(PublicationHold, match='private Git persistence fault'):
            worker.execute('bash', dict(command='git -c user.name=Test -c user.email=test@example.invalid commit --allow-empty -m private'))
        with team.workspace_publication.connect() as conn:
            assert not conn.execute('SELECT * FROM private_git').fetchall()
        assert worker._bash_facts['boundary_closed']
        assert worker.last_status == 'result_unknown'


def test_combined_redirection_cannot_bypass_advance_closure():
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team.server.simulator = fast_simulator(team.server.conn, team.server.tools.config, team.server.tools.rng)
        team._set_phase('ceo')
        worker = team.roles['ceo'].executor
        command = './novamind-operation next-week why {0,0,0} 0 0 0 0 0 0 0 0 0 &> plans/advance.txt'
        output = worker.execute('bash', dict(command=command))
        assert team.server.tools.current_day == 0, output
        row = team.workspace_publication.draft_rows('ceo')[0]
        _, candidate = team.workspace_publication.draft('ceo', team.workspace_publication.draft_ref(row))
        assert b'workspace_tool_pending' in team.workspace_publication.get('blobs', candidate['files']['plans/advance.txt']['blob'])
        worker.execute('bash', dict(command='pf workspace discard ' + team.workspace_publication.draft_ref(row)))
        output = worker.execute('bash', dict(command=ADVANCE))
        assert team.server.tools.current_day == 7, output
        assert worker._bash_facts['boundary_closed']


def test_private_split_index_never_writes_shared_git_metadata():
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        publication = team.workspace_publication
        worker = team.roles['growth'].executor
        before = {p.name for p in (publication.public / '.git').glob('sharedindex.*')}
        command = "git -c splitIndex.maxPercentChange=100 update-index --split-index; mkdir -p plans/private-index; for n in {1..80}; do printf PRIVATE > plans/private-index/SECRET-$n; done; git -c splitIndex.maxPercentChange=100 add plans/private-index; exit 3"
        output = worker.execute('bash', dict(command=command))
        assert 'retained_partial' in output, output
        output = worker.execute('bash', dict(command='git status --short'))
        assert 'SECRET-' in output, output
        after = {p.name for p in (publication.public / '.git').glob('sharedindex.*')}
        assert after == before
        peer = team.roles['ops_finance'].executor
        output = peer.execute('bash', dict(command="python -c \"from pathlib import Path; print([p.read_bytes() for p in Path('.git').glob('sharedindex.*')])\""))
        assert 'SECRET-' not in output, output
        assert not (publication.public / 'plans/private-index').exists()
