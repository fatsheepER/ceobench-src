"""The pf command in bash, call notes, integrated prompts and the round-three receipts."""
import json
import re

import pytest

from saas_bench import pf_cli
from saas_bench.registration_prompt import git_memory_line, integrate, pf_memory_line
from saas_bench.text_registry import TextRegistry
from test_text_registry import workspace, captured, call, declaration, git, send
from test_preflight_integration import offline_runner, packed_public

ROOTS = {'/workspace'}


def test_pf_command_parsing_views_and_refusals():
    assert pf_cli.parse('ls -la', ROOTS) is None
    assert pf_cli.parse("cat > a.py <<'EOF'\npf = 3\nEOF\npython a.py", ROOTS) is None
    assert pf_cli.parse('pf log MEMORY.md', ROOTS) == ('pf_log', {'target': {'path': 'MEMORY.md'}}, None)
    assert pf_cli.parse('cd /workspace && pf show MEMORY.md@v8 2>&1 | head -20', ROOTS) == (
        'pf_read', {'target': {'version': 'MEMORY.md@v8'}, 'mode': 'content', 'full': False}, ('head', 20))
    assert pf_cli.parse('pf diff a.py.out@v1 a.py.out@v3', ROOTS)[1] == dict(
        target={'version': 'a.py.out@v3'}, mode='diff', baseline={'version': 'a.py.out@v1'})
    assert pf_cli.parse('pf diff MEMORY.md', ROOTS)[:2] == ('pf_diff', {'target': {'path': 'MEMORY.md'}})
    assert pf_cli.parse('pf depend r4 --detail', ROOTS)[1] == dict(target={'record': 'r4'}, detail=True, purpose='current')
    assert pf_cli.parse('pf rdepend query7@v2 --all', ROOTS)[1] == dict(
        target={'version': 'query7@v2'}, detail=False, current_only=False)
    assert pf_cli.parse('pf search S1 --kind customer_group', ROOTS)[1] == dict(
        object={'id': 'S1', 'kind': 'customer_group'}, all=False)
    assert pf_cli.parse('pf more c3 | tail -n 5', ROOTS) == ('pf_more', {'cursor': 'c3'}, ('tail', 5))
    for command in ('pf', 'pf help', 'pf frobnicate x', 'pf log', 'pf log a b', 'pf depend r1 --bogus',
                    'pf log MEMORY.md && ls', 'ls && pf log MEMORY.md', 'pf log MEMORY.md > out.txt',
                    'pf log MEMORY.md | grep x', 'pf more 3'):
        with pytest.raises(pf_cli.Usage) as error:
            pf_cli.parse(command, ROOTS)
        assert 'usage: pf <command>' in str(error.value), command


def test_pf_runs_in_bash_with_notes_log_diff_blame_and_search(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    first = executor.execute('write_file', {'path': 'MEMORY.md', 'content': 'a\nb\n', 'note': 'first notes'})
    assert first.endswith('\n[pf: wrote MEMORY.md@v1 | note saved]'), first
    # The note is not part of the call: repeating the command keeps its handle.
    executor.execute('bash', {'command': 'echo same', 'note': 'one'})
    again = executor.execute('bash', {'command': 'echo same', 'note': 'two'})
    assert 'cmd' not in again  # no captured evidence and no writes: no line at all
    long_note = 'x' * 250
    second = executor.execute('bash', {'command': "printf 'a\\nB\\nc\\n' > MEMORY.md", 'note': long_note})
    assert second.endswith('| wrote MEMORY.md@v2 | note saved (truncated to 200 characters)]'), second
    log = executor.execute('bash', {'command': 'cd /workspace && pf log MEMORY.md'})
    lines = log.splitlines()
    assert lines[0] == 'file MEMORY.md: 2 versions, newest first'
    assert lines[1].startswith('MEMORY.md@v2 · day 0 · +2 −1 lines · note: "' + 'x' * 147)
    assert lines[2] == 'MEMORY.md@v1 · day 0 · 2 lines · note: "first notes"'
    assert lines[3] == 'pf show MEMORY.md@v2 · pf diff MEMORY.md@v1 MEMORY.md@v2 · pf blame MEMORY.md'
    assert executor.execute('bash', {'command': 'pf log MEMORY.md | head -2'}) == '\n'.join(lines[:2])
    diff = executor.execute('bash', {'command': 'pf diff MEMORY.md'})
    assert '--- MEMORY.md@v1\n+++ MEMORY.md@v2' in diff and '-b\n+B\n+c' in diff
    assert executor.execute('bash', {'command': 'pf diff MEMORY.md@v1'}) == diff
    blame = executor.execute('bash', {'command': 'pf blame MEMORY.md'}).splitlines()
    assert blame[1:4] == ['v1 d0 | a', 'v2 d0 | B', '      | c']
    assert blame[4].startswith('Notes: v1 (day 0): "first notes" · v2 (day 0): "xxx')
    # A note is shown when its file comes back: read_file, or cat showing the whole file.
    assert executor.execute('read_file', {'path': 'MEMORY.md'}).endswith('[pf: MEMORY.md@v2 note (day 0): "' + 'x' * 200 + '"]')
    shown = executor.execute('bash', {'command': 'cat MEMORY.md'})
    assert shown.endswith('[pf: MEMORY.md@v2 note (day 0): "' + 'x' * 200 + '"]'), shown
    refused = executor.execute('bash', {'command': 'pf log MEMORY.md; ls'})
    assert refused.startswith('pf: run pf on its own') and not (workspace / 'ls').exists()
    assert executor.execute('bash', {'command': 'pf show nothing.txt'}).startswith('Error:')
    call(registry, 'create', **declaration(objects=[dict(kind='segment', id='S1')], text='S1 plan'))
    search = executor.execute('bash', {'command': 'pf search S1'})
    assert search.splitlines()[:3] == ['S1: 1 active text, 0 business writes, 0 outputs, newest first.', 'Active texts:',
                                       '  r1.1 · day 7 · text r1.1 (active): "S1 plan"']
    assert search.endswith('pf search S1 --all lists all 1 saved item.')
    texts = executor.execute('bash', {'command': 'pf log r1'})
    assert texts.splitlines()[:2] == ['text r1: 1 revision, newest first',
                                      'r1.1 · day 7 · active: "S1 plan" · reason: Initial observation']
    # The PF memory line names the loaded version, its day, its note and the history size.
    version = store.latest_version('MEMORY.md', 'file_bytes')[0]
    assert pf_memory_line(store, version) == ('[pf: MEMORY.md@v2, written day 0, note: "' + 'x' * 200 +
                                              '" | 2 versions: pf log MEMORY.md]')


def test_prompts_integrate_at_single_anchors_and_git_names_the_memory_commit(workspace):
    from saas_bench.agents.bash_agent.agent import BashAgent
    agent = BashAgent.__new__(BashAgent)
    agent.total_days = 497
    original = agent._default_system_prompt()
    for pf in (False, True):
        prompt = integrate(original, pf)
        assert 'You have 10 tools:' in prompt and 'MEMORY.md is the ONLY way' not in prompt
        assert prompt.index('## Registered Texts') < prompt.index('## Weekly Workflow')
        assert ('## File and Output History (pf)' in prompt) == pf and ('## File History (git)' in prompt) != pf
    with pytest.raises(ValueError, match='exactly once'):
        integrate(original.replace('You have 6 tools:', ''), False)
    assert git_memory_line(workspace) is None
    (workspace / 'MEMORY.md').write_text('notes')
    git(workspace, 'add', '.')
    git(workspace, 'commit', '-qm', 'Week 3 (day 21) [week-3]')
    head = git(workspace, 'rev-parse', 'HEAD')[:7]
    assert git_memory_line(workspace) == f'[git: MEMORY.md last committed in week-3 ({head}) | git log -p -- MEMORY.md]'


def test_receipts_are_text_and_the_weekly_check_skips_ended_texts(workspace, tmp_path):
    registry = TextRegistry(workspace, 'git', sim_day=lambda: 7)
    text = registry.execute('create', dict(text='Plan', objects=[dict(kind='plan', id='B')], applies='7-13',
                                           reason='r', references=[dict(cite='evidence.json', note='why')]))
    assert text == "Registered r1.1 (active).\nCited: evidence.json@week-2 (this week's closing commit)"
    assert registry.execute('revise', dict(record='r1', reason='wording', text='Plan B')) == 'Revised r1.2 (active).'
    stored = json.loads(registry.path.read_text())['records']['r1'][-1]
    assert stored['applies_at'] == {'start_day': 7, 'end_day': 13}
    assert stored['references'][0]['evidence'] == {'path': 'evidence.json', 'commit': 'week-2'}
    call(registry, 'create', **declaration({'record': 'r1.2'}, text='Depends on the plan'))
    assert registry.weekly_check(7).startswith('=== Check of your registered texts (day 7) ===\n2 active texts')
    late = registry.weekly_check(14)
    assert late.endswith('Not checked because their applies window is over: r1.2 '
                         '(text_retire them if you no longer use them).'), late
    with pytest.raises(ValueError, match='applies must be'):
        registry.execute('create', dict(text='x', objects=[dict(kind='plan', id='B')], applies='soon',
                                        reason='r', references=[]))


def test_pf_receipt_lists_this_weeks_writes_on_the_same_objects(offline_runner):
    runner = offline_runner(text_registration='pf')
    runner.agent.current_day = 0
    output = runner._execute_tool('bash', {'command': './novamind-operation python-c "import novamind_api as nm; '
                                           "nm.analytics.set_targeted_dev_spend(targeted_spend={'S1': 500})\""})
    assert '[pf: cmd' in output, output
    send(runner.evidence_store, output)
    handle = re.search(r'\[pf: (cmd\d+@v\d+)', output).group(1)
    created = runner._execute_tool('text_create', dict(
        text='S1 dev 300 this week', objects=[dict(kind='customer_group', id='S1')], applies='0-',
        reason='plan', references=[dict(cite=handle)]))
    lines = created.splitlines()
    assert lines[:2] == ['Registered r1.1 (active).', f'Cited: {handle} (day 0)']
    assert lines[2].startswith('Business writes this week touching S1: day 0 set_targeted_dev_spend(') and '500' in lines[2]
    other = runner._execute_tool('text_create', dict(text='About S2', objects=[dict(kind='customer_group', id='S2')],
                                                     applies='0-', reason='plan', references=[]))
    assert other == 'Registered r2.1 (active).'


@pytest.mark.parametrize('mode', ['prefix', 'pf'])
def test_next_weeks_memory_names_its_history(offline_runner, monkeypatch, mode):
    from test_stage5_prep import fake_weeks
    runner = offline_runner(text_registration=mode, stop_after_day=14)
    runner.agent.current_day = 0
    args = {'path': 'MEMORY.md', 'content': 'cap 272812\n'}
    if mode == 'pf':
        args['note'] = 'cap table kept here'
    runner._execute_tool('write_file', args)
    runner.agent.current_day = -1
    requests = fake_weeks(runner, monkeypatch)
    assert runner.run(verbose=False)['outcome'] == 'stopped'
    system = requests[-1]['messages'][0]['content']
    assert 'loaded into your context at the start of every week.\n' in system
    if mode == 'pf':
        assert '[pf: MEMORY.md@v1, written day 0, note: "cap table kept here" | 1 version: pf log MEMORY.md]' in system
    else:
        assert re.search(r'\[git: MEMORY\.md last committed in week-1 \([0-9a-f]{7}\) \| git log -p -- MEMORY\.md\]', system)
    assert system.endswith('\n\ncap 272812')


def test_diff_of_a_text_compares_its_revisions(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    call(registry, 'create', **declaration(text='Keep B at $99'))
    call(registry, 'revise', record='r1', reason='New price', text='Keep B at $89')
    diff = executor.execute('bash', {'command': 'pf diff r1'})
    assert '"Keep B at $99"' in diff and '"Keep B at $89"' in diff and diff.index('-') < diff.index('+')
    assert executor.execute('bash', {'command': 'pf diff r1.1'}) == diff
    assert 'pf log rN' in executor.execute('bash', {'command': 'pf blame r1'})
