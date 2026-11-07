from contextlib import closing
import json
import re
import sqlite3
from pathlib import Path
import threading

from saas_bench import team_batch as batch
from test_preflight_team_flow import ADVANCE, calls, final
from round5_checkpoint_fixture import TABLES
from test_round5_workflows import make_draft
from test_team_script_state import register_cli


def setup(spec):
    if spec.get('group_stop_on_retry'):
        from saas_bench.run_lifecycle import WorkerLifecycle
        spec['retry_blocked'] = threading.Event()
        spec['r5_retry_waits'] = []
        sleep = WorkerLifecycle.sleep
        def wait(lifecycle, seconds):
            spec['r5_retry_waits'].append(seconds)
            spec['retry_blocked'].set()
            return sleep(lifecycle, seconds)
        WorkerLifecycle.sleep = wait


def scripts(team):
    for role, runtime in team.roles.items():
        runtime.executor.execute('write_file', dict(path='memory/' + role + '.md', content='R5-MEMORY-' + role))
        register_cli(team, role, 'scripts/once-' + role + '.py',
            "import novamind_api as n\n"
            "with open('data/count-" + role + ".txt', 'a') as out: out.write('once\\n')\n"
            "print(n.infrastructure.get_cost_info())\nprint('R5-SCRIPT-" + role + "')\n")


def checkpoint(team, attempt, record, boundary, spec, proof):
    publication = getattr(team, 'workspace_publication', None)
    if boundary == 'ready' and spec.get('group_stop_on_retry') and record['run_id'] == spec['reject_run']:
        def stop():
            if spec['retry_blocked'].wait(30):
                (attempt / 'STOP').touch()
                proof['group_stop_after_retry'] = True
        spec['stop_thread'] = threading.Thread(target=stop, name='r5-independent-stop')
        spec['stop_thread'].start()
    if boundary == 'ready' and record.get('source_checkpoint'):
        source = Path(record['source_checkpoint'])
        saved = batch.read(source / 'team.json')
        proof['restored_scripts'] = team.server.get_daily_scripts('growth')
        proof['saved_scripts'] = saved['scripts']['growth']
        if publication:
            with closing(publication.connect()) as live, closing(sqlite3.connect(source / 'private/workspace-publication/authority.sqlite')) as stored:
                proof['publication_exact'] = all(
                    sorted([tuple(r) for r in live.execute('SELECT * FROM ' + table)], key=repr) ==
                    sorted(stored.execute('SELECT * FROM ' + table).fetchall(), key=repr) for table in TABLES)
            proof['past_confirmation_retained'] = publication.check_finish('growth', 21) == ''
    if boundary == 'week_boundary' and team.server.tools.current_day == 28:
        phase = team.phase
        team._set_phase('analysis')
        if publication:
            make_draft(team, 'growth', content='R5-D28-PRIVATE\n')
            publication.check_finish('growth', 21)
            publication.confirm_finish('growth', 21, publication.draft_signature('growth', 21))
            row, _ = make_draft(team, 'ops_finance', path='plans/resolved.txt', content='R5-D28-RESOLVED\n')
            result = team.roles['ops_finance'].executor.execute('bash', dict(command=
                'pf workspace resolve ' + publication.draft_ref(row) + ' --against ' + publication.head()))
            assert '[PF workspace published]' in result, result
        team._set_phase(phase)


def response(team, role, body, spec):
    if role == 'ceo':
        if team.phase == 'ceo_opening':
            if not any(m.get('name') == 'ask_analyst' for m in body['messages']):
                return calls(('ask_analyst', dict(role='growth', message='R5-OPENING-CHECK')))
            return final('R5-DIRECTIONS')
        return calls(('bash', dict(command=ADVANCE)))
    if team._workers[role].question:
        return final('R5-ANSWER-' + role)
    last = body['messages'][-1].get('content') or ''
    match = re.search(r'--confirm-finish ([a-f0-9]+)', last)
    if match:
        return calls(('bash', dict(command='pf workspace status --confirm-finish ' + match.group(1))))
    return final('R5-REPORT-' + role)


def inspect(team, record, spec, proof):
    if thread := spec.get('stop_thread'):
        thread.join(2)
        proof['stop_thread_closed'] = not thread.is_alive()
        proof['r5_retry_waits'] = spec['r5_retry_waits']
    proof['counts'] = {role: (runtime.workspace / ('data/count-' + role + '.txt')).read_text()
        if (runtime.workspace / ('data/count-' + role + '.txt')).exists() else ''
        for role, runtime in team.roles.items()}
    proof['memory'] = {role: (runtime.workspace / ('memory/' + role + '.md')).read_text()
        for role, runtime in team.roles.items()}
    publication = getattr(team, 'workspace_publication', None)
    if publication:
        proof['drafts'] = publication.draft_rows('growth')
        proof['head'] = publication.head()
