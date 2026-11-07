from contextlib import closing
import fcntl
import json
import os
from pathlib import Path
import signal
import threading

from saas_bench.agent_capabilities import AgentCapabilities
from saas_bench.multi_agent import MultiAgentRuntime
from saas_bench.run_state import write_json
from test_preflight_team_flow import ADVANCE, calls, final
from test_round5_workflows import make_draft
from test_team_script_state import register_cli


TABLES = ('head', 'authors', 'drafts', 'resolutions', 'discards', 'finish_confirmations',
          'draft_reminders', 'private_git', 'source_head', 'source_bases')


def setup(spec):
    initialize = MultiAgentRuntime.__init__
    def create(self, *args, **kwargs):
        kwargs.update(workspace_layout='shared-v1', workspace_protocol='snapshot-v1',
                      capabilities=AgentCapabilities.round_three('pf'))
        initialize(self, *args, **kwargs)
    MultiAgentRuntime.__init__ = create


def scripts(team):
    for role in team.roles:
        register_cli(team, role, 'once.py',
                     "import novamind_api as n\nprint(n.infrastructure.get_cost_info())\n"
                     + "print('R5-SCRIPT-" + role + "')\n")


def checkpoint(team, attempt, record, boundary, spec, proof):
    publication = team.workspace_publication
    if boundary == 'ready':
        source = record.get('source_checkpoint')
        if source:
            with closing(publication.connect()) as conn:
                import sqlite3
                with closing(sqlite3.connect(Path(source) / 'private/workspace-publication/authority.sqlite')) as saved:
                    proof['workspace_restored_exactly'] = all(
                        sorted([tuple(row) for row in conn.execute('SELECT * FROM ' + table)], key=repr) ==
                        sorted(saved.execute('SELECT * FROM ' + table).fetchall(), key=repr)
                        for table in TABLES)
            proof['confirmed_set_not_blocked'] = publication.check_finish('growth', 0) == ''
            proof['restored_source'] = team.server.get_daily_scripts('growth')
            phase = team.phase
            team._set_phase('analysis')
            proof['restored_private_read'] = team.roles['growth'].executor.execute(
                'read_file', dict(path='plans/draft.txt'))
            peer = team.roles['ops_finance'].executor
            proof['restored_peer_read'] = peer.execute('read_file', dict(path='plans/draft.txt'))
            proof['restored_public_queries'] = [peer.execute('bash', dict(command=command))
                for command in ('pf show shared:plans/draft.txt@v1 --full',
                                'pf show shared:plans/draft.txt@v2 --full',
                                'pf candidates --history --role growth --detail')]
            team._set_phase(phase)
        else:
            phase = team.phase
            team._set_phase('analysis')
            make_draft(team, 'growth', content='R5-RECOVER-PRIVATE\n')
            publication.week_reminder('growth', 0)
            publication.check_finish('growth', 0)
            publication.confirm_finish('growth', 0, publication.draft_signature('growth', 0))
            row, _ = make_draft(team, 'ops_finance', path='plans/resolved.txt', content='R5-RESOLVED\n')
            team.roles['ops_finance'].executor.execute('bash', dict(command=
                'pf workspace resolve ' + publication.draft_ref(row) + ' --against ' + publication.head()))
            team._set_phase(phase)
        proof['workspace_ready_head'] = publication.head()
        if spec.get('fault') and not source:
            def stage(name, execution=None):
                if name != spec.get('stage', 'manifest_durable'):
                    return
                with closing(publication.connect()) as conn:
                    row = conn.execute('SELECT candidate FROM executions WHERE id=?', (execution,)).fetchone()
                if row and row[0]:
                    candidate = json.loads(publication.get('candidates', row[0]))
                    if 'plans/fault-a' not in candidate['files']:
                        return
                elif not (publication.private / 'executions' / str(execution) / 'plans/fault-a').exists():
                    return
                write_json(Path(spec['proof']).with_suffix('.fault.json'), dict(
                    root=str(team.root), execution=execution, stage=name))
                publication.stage = lambda *args: None
                if spec['fault'] == 'kill':
                    os.kill(os.getpid(), signal.SIGKILL)
                elif spec['fault'] == 'disconnect':
                    raise ConnectionError('Lost publication acknowledgement after commit')
                else:
                    stream = (publication.private / 'publish.lock').open('a+b')
                    fcntl.flock(stream, fcntl.LOCK_EX)
                    def release_after_cancel():
                        try:
                            assert team._lifecycle._wake.wait(15)
                        finally:
                            fcntl.flock(stream, fcntl.LOCK_UN)
                            stream.close()
                    thread = threading.Thread(target=release_after_cancel, name='r5-publication-lock')
                    thread.start()
                    proof['stop_lock_thread'] = thread
                    Path(spec['stop']).touch()
            publication.stage = stage


def response(team, role, body, spec):
    rendered = json.dumps(body['messages'])
    if role == 'ceo':
        if team.phase == 'ceo_opening':
            if not any(message.get('name') == 'ask_analyst' for message in body['messages']):
                return calls(('ask_analyst', dict(role='growth', message='R5-OPENING-CHECK')))
            return final('R5-DIRECTIONS')
        return calls(('bash', dict(command=ADVANCE)))
    if team._workers[role].question:
        if spec.get('fault') and 'R5-FAULT-DONE' not in rendered:
            return calls(('bash', dict(command="printf new > plans/fault-a; printf new > plans/fault-b; echo R5-FAULT-DONE")))
        return final('R5-OPENING-ANSWER')
    if role == 'ops_finance' and 'R5-API-DONE' not in rendered:
        return calls(('bash', dict(command="./novamind-operation python-c \"import novamind_api as n; n.infrastructure.set_capacity_tier(2); print('R5-API-DONE')\"")))
    return final('R5-REPORT-' + role)


def inspect(team, record, spec, proof):
    publication = team.workspace_publication
    proof['workspace_final'] = dict(head=publication.head(), drafts=publication.draft_rows('growth'),
        scripts=team.server.get_daily_scripts('growth'))
    with closing(publication.connect()) as conn:
        proof['publication_pending'] = conn.execute('SELECT count(*) FROM executions WHERE outcome IS NULL').fetchone()[0]
        proof['publication_results'] = [json.loads(row[0]) for row in conn.execute('SELECT outcome FROM executions WHERE outcome IS NOT NULL')]
    if thread := proof.pop('stop_lock_thread', None):
        thread.join(3)
        proof['stop_lock_closed'] = not thread.is_alive()
    if not team._lifecycle.reason:
        proof['confirmed_before_revision'] = publication.check_finish('growth', 0) == ''
        team._set_phase('analysis')
        team.roles['growth'].executor.execute('write_file', dict(path='plans/draft.txt', content='R5-LATER-PRIVATE\n'))
        proof['new_revision_blocked'] = bool(publication.check_finish('growth', 0))
