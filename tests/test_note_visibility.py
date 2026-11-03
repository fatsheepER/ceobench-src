"""Per-call notes remain attached to exact public artifacts in both groups."""
import json
import re
import shlex

import pytest

from test_pf_candidates import query
from test_preflight_team_flow import final, make_team
from test_receipt_outputs import model_receives
from test_text_registry import captured, git, workspace


def test_exact_same_content_output_notes_are_distinct(workspace, tmp_path):
    store, _, executor = captured(workspace, tmp_path)
    for note in ('first calculation', 'second calculation'):
        executor.execute('bash', {'command': 'printf stable', 'note': note})
    captures = []
    for note in ('first calculation', 'second calculation'):
        output, page = query(executor, 'pf search --text ' + shlex.quote(note))
        assert page['total'] == 1, output
        item = page['items'][0]
        captures.append(item['content_ref'])
        shown = executor.execute('bash', {'command': 'pf show ' + item['content_ref'] + ' --full'})
        header = json.loads(shown.splitlines()[0])['target']
        assert header['note'][1] == note
        assert header['note_context']['role'] == 'ceo'
        assert header['note_context']['status'] == 'succeeded'
        assert note in model_receives(store, workspace, shown)
    assert len(set(captures)) == 2


def test_file_notes_follow_exact_writes_and_plain_read(workspace, tmp_path):
    store, _, executor = captured(workspace, tmp_path)
    captures = []
    for note in ('first file purpose', 'second file purpose'):
        executor.execute('write_file', {'path': 'calculation.txt', 'content': 'stable bytes', 'note': note})
        _, page = query(executor, 'pf candidates --history --kind file --text calculation.txt')
        captures.append(page['items'][0]['content_ref'])
    for capture, note in zip(captures, ('first file purpose', 'second file purpose')):
        shown = executor.execute('bash', {'command': 'pf show ' + capture + ' --full'})
        assert json.loads(shown.splitlines()[0])['target']['note'][1] == note
    for tool, args in (('read_file', {'path': 'calculation.txt'}), ('bash', {'command': 'cat calculation.txt'})):
        shown = executor.execute(tool, args)
        assert 'second file purpose' in shown and 'first file purpose' not in shown
        assert 'second file purpose' in model_receives(store, workspace, shown, tool)


def test_log_notes_keep_exact_acquisition_and_frozen_pagination(workspace, tmp_path):
    from test_pf_candidates import day
    store, _, executor = captured(workspace, tmp_path)
    day(store, 0)
    for index in range(21):
        executor.execute('write_file', {'path': 'calculation.txt', 'content': str(index),
                                       'note': f'Initial purpose {index}'})
    day(store, 7)
    executor.execute('write_file', {'path': 'calculation.txt', 'content': '20', 'note': 'Rechecked purpose'})
    shown, page = query(executor, 'pf log calculation.txt')
    latest = page['items'][0]
    assert latest['day'] == 0 and latest['note'] == (7, 'Rechecked purpose')
    assert latest['acquisitions'] == 2
    capture = latest['content_ref']
    assert capture in shown and 'note (day 7): "Rechecked purpose"' in shown
    cursor = page['next_cursor']
    assert cursor
    _, frozen = query(executor, 'pf more ' + cursor)
    day(store, 14)
    executor.execute('write_file', {'path': 'calculation.txt', 'content': '20', 'note': 'Later purpose'})
    _, current = query(executor, 'pf log calculation.txt')
    assert current['items'][0]['day'] == 0
    assert current['items'][0]['note'] == (14, 'Later purpose')
    assert current['items'][0]['acquisitions'] == 3
    assert current['items'][0]['content_ref'] != capture
    _, repeated = query(executor, 'pf more ' + cursor)
    assert repeated == frozen
    exact = executor.execute('bash', {'command': 'pf show ' + capture + ' --full'})
    assert json.loads(exact.splitlines()[0])['target']['note'] == [7, 'Rechecked purpose']
    assert 'Later purpose' not in exact and 'Rechecked purpose' in model_receives(store, workspace, exact)


def test_git_note_artifacts_are_exact_cross_role_and_keep_old_files():
    with make_team('git', lambda *args: final('done'), 'git-note-artifacts') as team:
        owner, reader = team.roles['growth'], team.roles['ceo']
        for runtime in team.roles.values():
            request = dict(model=runtime.agent.model, messages=[dict(role='system', content=runtime.agent.system_prompt), dict(role='user', content='Review source guidance.')])
            runtime.agent._llm_attempt = 0
            runtime.agent._request_model('chat', request, lambda: runtime.agent.client.chat.completions.create(**request))
            assert 'git show --abbrev-commit COMMIT' in json.dumps(team.test_wire[-1]['body'])
        artifacts = []
        for index, note in enumerate(('first calculation', 'second calculation')):
            shown = owner.executor.execute('bash', {'command': f"printf 'old bytes {index}' > analysis.txt; printf stable", 'note': note})
            match = re.search(r'tool-outputs/call[1-9][0-9]*\.txt', shown)
            assert match, shown
            artifacts.append(match[0])
        assert len(set(artifacts)) == 2
        current = reader.executor.execute('read_file', {'path': str(owner.workspace / 'analysis.txt')})
        assert 'second calculation' in current and 'first calculation' not in current
        owner.executor.execute('write_file', {'path': 'analysis.txt', 'content': 'new bytes'})
        assert 'calculation' not in reader.executor.execute('read_file', {'path': str(owner.workspace / 'analysis.txt')})
        for index, (path, note) in enumerate(zip(artifacts, ('first calculation', 'second calculation'))):
            shown = reader.executor.execute('read_file', {'path': str(owner.workspace / path)})
            assert note in shown and 'stable' in shown and 'growth' in shown
            assert ('second calculation' if note == 'first calculation' else 'first calculation') not in shown
            ref = re.search(r'growth:analysis\.txt@([0-9a-f]{7,39})', shown)
            assert ref, shown
            assert git(owner.workspace, 'show', ref[1] + ':analysis.txt') == f'old bytes {index}'
            assert note in git(owner.workspace, 'show', '--no-patch', ref[1])
            historical = reader.executor.execute('bash', {'command': 'git -C ' + shlex.quote(str(owner.workspace)) + ' show --abbrev-commit ' + ref[1]})
            assert note in historical and f'old bytes {index}' in historical
            assert note in model_receives(None, reader.workspace, historical)
            assert 'git show --abbrev-commit COMMIT' in reader.agent.system_prompt
            assert owner.identity.session_id not in shown and owner.identity.run_id not in shown
            wire = model_receives(None, reader.workspace, shown, 'read_file')
            assert note in wire
        search = reader.executor.execute('bash', {'command': 'grep -rl ' + shlex.quote('first calculation') + ' ' + shlex.quote(str(owner.workspace / 'tool-outputs'))})
        assert artifacts[0] in search and artifacts[1] not in search


def test_pf_file_note_remains_visible_to_peer():
    with make_team('pf', lambda *args: final('done'), 'pf-peer-note') as team:
        growth, ceo = team.roles['growth'], team.roles['ceo']
        growth.executor.execute('write_file', {'path': 'report.txt', 'content': 'business result', 'note': 'capacity forecast'})
        shown = ceo.executor.execute('bash', {'command': 'pf show growth:report.txt --full'})
        header = json.loads(shown.splitlines()[0])['target']
        assert header['note'][1] == 'capacity forecast'
        assert header['note_context']['role'] == 'growth'
        assert 'capacity forecast' in model_receives(ceo.store, ceo.workspace, shown)


@pytest.mark.parametrize('tool,args', [
    ('write_file', {'path': 'empty.txt', 'content': ''}),
    ('bash', {'command': 'true'}),
])
def test_empty_artifacts_remain_searchable_by_note(workspace, tmp_path, tool, args):
    store, _, executor = captured(workspace, tmp_path)
    executor.execute(tool, dict(args, note='Intentional empty result'))
    output, page = query(executor, "pf search --text 'Intentional empty result'")
    items = [item for item in page['items'] if tool == 'bash' or item['version'].startswith('empty.txt@')]
    assert len(items) == 1, output
    shown = executor.execute('bash', {'command': 'pf show ' + items[0]['content_ref'] + ' --full'})
    assert json.loads(shown.splitlines()[0])['target']['note'][1] == 'Intentional empty result'
    assert 'Intentional empty result' in model_receives(store, workspace, shown)
