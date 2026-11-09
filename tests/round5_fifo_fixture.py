from pathlib import Path
import sqlite3
import time

import round5_batch_fixture as base
from test_preflight_team_flow import calls, final


def setup(spec):
    base.setup(spec)


scripts = base.scripts
inspect = base.inspect


def checkpoint(team, attempt, record, boundary, spec, proof):
    base.checkpoint(team, attempt, record, boundary, spec, proof)
    if boundary == 'final' and record['run_id'] == spec.get('sqlite_fault_run'):
        def readonly_backup(path):
            target = attempt / 'readonly-backup.sqlite'
            with sqlite3.connect(target) as writable:
                writable.execute('CREATE TABLE existing(value)')
            with sqlite3.connect(target.as_uri() + '?mode=ro', uri=True) as readonly:
                team.server.conn.backup(readonly)
        team.public_refs.snapshot = readonly_backup
    if boundary == 'ready' and record['run_id'] == spec.get('unknown_run'):
        step = team.server.simulator.step_week
        def unknown():
            with (attempt / 'unknown-business-count.txt').open('a') as out:
                out.write('once\n')
            step()
            raise ConnectionError('Offline injected missing result after a business effect')
        team.server.simulator.step_week = unknown
    if boundary == 'ready' and record['run_id'] == spec.get('bankrupt_action_run'):
        capacity = team.server.tools.set_capacity_tier
        def spend(tier):
            result = capacity(tier)
            team.server.conn.execute('INSERT INTO ledger(day,category,amount,note) VALUES(0,?,-2000000,?)',
                ('operations', 'offline action bankruptcy'))
            team.server.conn.commit()
            return result
        team.server.tools.set_capacity_tier = spend
    if boundary == 'ready' and record['run_id'] == spec.get('wait_run'):
        root = Path(spec['claims'])
        deadline = time.monotonic() + 600
        while not list(root.glob(spec['release_run'] + '-*.json')):
            if time.monotonic() >= deadline:
                raise AssertionError('Independent FIFO failed to launch the next Git seed')
            time.sleep(.1)
        proof['released_by_next_git_seed'] = True


def response(team, role, body, spec):
    if team.roles[role].identity.run_id == spec.get('invalid_run'):
        return final('')
    if team.roles[role].identity.run_id == spec.get('bankrupt_action_run') and role == 'ops_finance':
        return calls(('bash', dict(command='./novamind-operation python-c "import novamind_api as n; print(n.infrastructure.set_capacity_tier(2))"')))
    return base.response(team, role, body, spec)
