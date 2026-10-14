import json
import re

import pytest

from saas_bench import pf_cli, registration_prompt, team_prompts
from saas_bench.pf_refresh import refresh
from saas_bench.public_sql import execute_query
from saas_bench.registration_schema import PF_CREATE, PF_REVISE, tool_definitions
from saas_bench.role_policy import ROLES
from test_pf_stale import record_sql
from test_preflight_team_evidence import pf, write
from test_preflight_team_flow import calls, make_team
from test_public_sql import server
from test_text_registry import captured, send, workspace


def example(text):
    blocks = re.findall(r'```json\n(.*?)\n```', text, re.S)
    assert len(blocks) == 1
    return json.loads(blocks[0])


def commands(flow):
    section = re.search(rf'\n{flow}\. .*?(?=\n\d\. |\nOther boundaries:|\Z)', pf_cli.HELP, re.S)[0]
    return [line for block in re.findall(r'```bash\n(.*?)\n```', section, re.S)
            for line in block.splitlines()]


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_three_roles_receive_shared_rules_and_mode_specific_example(mode):
    with make_team(mode, lambda *args: calls(('text_list', {})), 'guidance-wire') as team:
        for role in ROLES:
            runtime = team.roles[role]
            assert runtime.agent.act('Synthetic dashboard', 0, False, {'day': 0}).tool == 'text_list'
            wire = team.test_wire[-1]
            assert wire['role'] == role
            if mode == 'pf':
                event = runtime.usage.last_request_event
                assert json.loads(runtime.store.get_content(event + ':wire')[1]) == wire['body']
            system = '\n'.join(m['content'] for m in wire['body']['messages'] if m['role'] == 'system')
            assert system.count(team_prompts.HANDOFF) == 1
            assert 'omitted references keep the old list' in system
            tools = {t['function']['name']: t['function'] for t in wire['body']['tools']}
            assert ('ask_analyst' in tools) == (role == 'ceo')
            for published in tool_definitions(pf=mode == 'pf', team=True):
                assert tools[published['name']] == published
            description = tools['text_create']['description']
            if mode == 'pf':
                assert system.count(team_prompts.PF_HISTORY) == 1
                assert team_prompts.GIT_HISTORY not in system
                assert description.count('Example request:') == 1
                PF_CREATE.model_validate(example(description))
            else:
                assert system.count(team_prompts.GIT_HISTORY) == 1
                assert team_prompts.PF_HISTORY not in system
                assert 'Example request:' not in description
                assert 'EXAMPLE_GROUP' not in description


def test_d0_prompt_example_runs_and_full_help_reaches_all_role_wires():
    def respond(team, role, body):
        system = '\n'.join(m['content'] for m in body['messages'] if m['role'] == 'system')
        command = re.search(r'for example `(pf[^`]*)`', system)[1]
        return calls(('bash', {'command': command}))

    with make_team('pf', respond, 'guidance-d0-help') as team:
        for role in ROLES:
            runtime = team.roles[role]
            action = runtime.agent.act('Synthetic dashboard', 0, False, {'day': 0})
            result = runtime.executor.execute(action.tool, action.arguments)
            assert result == pf_cli.HELP
            assert runtime.executor.last_status == 'succeeded'
            runtime.agent.act(result, 0, False, {'day': 0})
            wire = team.test_wire[-1]
            assert wire['role'] == role
            assert wire['body']['messages'][-1]['content'] == pf_cli.HELP
            event = runtime.usage.last_request_event
            assert json.loads(runtime.store.get_content(event + ':wire')[1]) == wire['body']
            system = wire['body']['messages'][0]['content']
            assert '`pf help` has complete examples' in system
            bash = next(t['function'] for t in wire['body']['tools'] if t['function']['name'] == 'bash')
            assert '`pf help` has complete examples' in bash['description']


def test_standalone_prompt_example_is_safe_before_any_registration(workspace, tmp_path):
    _, registry, executor = captured(workspace, tmp_path)
    command = re.search(r'for example `(pf[^`]*)`', registration_prompt.PF_HISTORY)[1]
    assert executor.execute('bash', {'command': command}) == pf_cli.HELP
    assert not registry.path.exists()
    assert '`pf help` has complete examples' in registration_prompt.PF_HISTORY
    assert 'Replace the synthetic names below with handles from actual receipts' in registration_prompt.PF_HISTORY


def test_help_peer_example_reads_exact_version_and_compares_one_object():
    with make_team('pf', lambda *args: calls(('text_list', {})), 'guidance-peer') as team:
        first = write(team, 'growth', 'earlier evidence\n', path='notes.md')
        second = write(team, 'growth', 'revised evidence\n', path='notes.md')
        show, = commands(1)
        log, diff = commands(3)
        assert 'growth:notes.md@v1' in first and 'growth:notes.md@v2' in second
        exact = pf(team, 'ceo', show)
        assert 'earlier evidence' in exact and 'revised evidence' not in exact
        latest = pf(team, 'ceo', show.replace('@v1', ''))
        assert 'revised evidence' in latest and 'growth:notes.md@v2' in latest
        history = pf(team, 'ceo', log)
        assert history.index('growth:notes.md@v2') < history.index('growth:notes.md@v1')
        changes = pf(team, 'ceo', diff)
        assert '-earlier evidence\n+revised evidence' in changes
        assert '--- growth:notes.md@v1\n+++ growth:notes.md@v2' in changes


def test_published_condition_and_review_examples(workspace, tmp_path, server):
    store, registry, executor = captured(workspace, tmp_path)
    server.sql_evidence = store
    executor.pf_queries.refresh = lambda versions, parent: refresh(server, versions, parent)
    sql = "SELECT 'EXAMPLE_GROUP' AS group_id,NULLIF(amount,-1)/100.0 AS metric FROM ledger WHERE day=0"
    server.conn.execute('UPDATE ledger SET amount=60')
    server.conn.commit()
    version = record_sql(store, sql, execute_query(server, sql))
    args = example(tool_definitions(pf=True)[0]['description'])
    assert set(args) == {'text', 'objects', 'references', 'applies', 'reason'}
    PF_CREATE.model_validate(args)
    assert args['references'][0]['cite'] == registry.resolver.handle(version)
    show_query, = commands(4)
    read = executor.execute('bash', {'command': show_query})
    send(store, read)
    assert executor.execute('text_create', args).startswith('Registered r1.1')
    show_text, check, history = commands(2)
    assert args['text'] in executor.execute('bash', {'command': show_text})
    for amount, status in ((60, 'predicate holds'), (40, 'PREDICATE FAILS'), (-1, 'cannot check')):
        server.conn.execute('UPDATE ledger SET amount=?', (amount,))
        server.conn.commit()
        before = server.conn.serialize()
        result = executor.execute('bash', {'command': check})
        assert server.conn.serialize() == before
        assert 'Checked day 7' in result and 'expanded dependencies' in result
        assert status in result, result
        assert 'query1@v1' in result
    executor.pf_queries.refresh = lambda *_: pytest.fail('Historical inspection must not refresh')
    result = executor.execute('bash', {'command': history})
    assert 'Not checked (--history lists history only)' in result
    assert 'not checked' in result and 'predicate holds' not in result and 'PREDICATE FAILS' not in result
    partial = executor.execute('bash', {'command': show_query + ' | head -1'})
    assert 'PF final delivery: partial' in partial
    send(store, partial)
    saved = registry.path.read_bytes()
    rejected = executor.execute('text_create', args)
    assert rejected.startswith('Error: PF output was displayed'), rejected
    assert 'pf_pipeline_source_ranges_unknown' in rejected and 'Selected fields cannot be registered' in rejected
    assert registry.path.read_bytes() == saved
    recovery = re.search(r'```bash\n(.*?)\n```', rejected, re.S)[1]
    assert recovery == 'pf show --full -- query1@v1'
    send(store, executor.execute('bash', {'command': recovery}))
    assert executor.execute('text_create', args).startswith('Registered r2.1')
    revise = example(pf_cli.HELP)
    PF_REVISE.model_validate(revise)
    inherited = dict(revise)
    del inherited['references']
    receipt = executor.execute('text_revise', inherited)
    assert receipt.startswith('Revised r1.2') and 'Retained references' in receipt
    records = registry._load()['records']['r1']
    assert records[-1]['references'] == records[0]['references']
    assert records[-1]['text'] == revise['text']
    assert executor.execute('text_revise', revise).startswith('Revised r1.3')
    references = registry._load()['records']['r1'][-1]['references']
    assert len(references) == 1 and references[0]['evidence'] == {'version': 'query1@v1'}
    assert references[0].get('select') is None and references[0].get('predicate') is None
    cleared = dict(revise, references=[])
    assert executor.execute('text_revise', cleared).startswith('Revised r1.4')
    assert registry._load()['records']['r1'][-1]['references'] == []


def test_help_aliases_in_bash_and_historical_parser(workspace, tmp_path):
    _, registry, executor = captured(workspace, tmp_path)
    commands = ['pf', 'pf help', 'pf --help', 'pf -h']
    for verb in ('log', 'show', 'diff', 'blame', 'depend', 'rdepend', 'search', 'more'):
        for flag in ('--help', '-h'):
            commands.extend((f'pf {verb} {flag}', f'pf {verb} {flag} missing', f'pf {verb} missing {flag}'))
    for command in commands:
        assert executor.execute('bash', {'command': command}) == pf_cli.HELP, command
        assert executor.last_status == 'succeeded'
        with pytest.raises(pf_cli.Usage) as error:
            pf_cli.parse(command, {'/workspace'})
        assert str(error.value) == pf_cli.USAGE
    assert not registry.path.exists()


@pytest.mark.parametrize('command', ['pf typo', 'pf depend', 'pf depend r1 --bogus',
    'pf search --kind', 'pf search --text term --all', 'pf more 3'])
def test_usage_errors_point_to_complete_help(workspace, tmp_path, command):
    _, registry, executor = captured(workspace, tmp_path)
    error = executor.execute('bash', {'command': command})
    assert error.startswith('[stderr]\npf')
    assert 'Run pf help for usage and complete examples.' in error
    assert error.endswith('[exit code: 2]')
    assert pf_cli.HELP not in error
    assert not registry.path.exists()


@pytest.mark.parametrize('literal', ['--help', '-h'])
def test_help_like_literal_search_values_and_paths(workspace, tmp_path, literal):
    _, _, executor = captured(workspace, tmp_path)
    executor.execute('write_file', {'path': literal, 'content': 'saved literal ' + literal})
    result = executor.execute('bash', {'command': f"pf search --text '{literal}'"})
    assert literal + '@v1' in result and 'saved literal ' + literal in result, result
    assert executor.last_status == 'succeeded'
    shown = executor.execute('bash', {'command': f"pf show -- '{literal}'"})
    assert 'saved literal ' + literal in shown and shown != pf_cli.HELP, shown
    assert pf_cli.parse_argv(['search', 'S1', '--kind', literal]) == (
        'pf_search', {'object': {'id': 'S1', 'kind': literal}, 'all': False})
    for command in (f"pf search --text '{literal}' --help", f"pf search -h --text '{literal}'"):
        assert executor.execute('bash', {'command': command}) == pf_cli.HELP
