"""Restore a real day-28 snapshot and check forks with deterministic offline model replies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from saas_bench.agents.bash_agent.run_test import BashAgentRunner
from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
from saas_bench.db_protection import load_session_db
from saas_bench.run_state import checkpoint_directory, clone_sql_run, file_hash, tree_hash, write_json

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', nargs='?', default='prefix-42', help='Legacy run suffix or a run-pointer JSON path')
parser.add_argument('--output', type=Path, default=BASE / 'engineering' / 'forks')
args = parser.parse_args()
OUT = args.output.resolve()
assert os.environ.get('PYTHONHASHSEED') == '0'
os.environ['CEOBENCH_SIMULATOR_LLM_PROVIDER'] = 'deepseek'
os.environ['CEOBENCH_SIMULATOR_LLM_MODEL'] = 'deepseek-flash'
for key in ('BOSSBENCH_LLM_REPLAY_DB', 'ORACLE_MODE', 'CEOBENCH_DASHBOARD_URL'):
    os.environ.pop(key, None)
pointer_path = Path(args.run) if args.run.endswith('.json') else BASE / ('freerun-' + args.run + '.json')
pointer = json.loads(pointer_path.read_text())
source = Path(pointer['path'])
assert pointer['group'] == 'prefix'  # The day-28 snapshot is immutable while later weeks run.
candidates = sorted(p for p in (source / 'checkpoints').iterdir()
                    if json.loads((p / 'checkpoint.json').read_text())['day'] == 28)
original = candidates[0]
cp = json.loads((original / 'checkpoint.json').read_text())
assert checkpoint_directory(source, cp) == original
original_hash = tree_hash(original)
OUT.mkdir(parents=True, exist_ok=False)
selector = OUT / 'source'
(selector / 'checkpoints').mkdir(parents=True)
shutil.copytree(original, selector / 'checkpoints' / original.name)
for name in ('manifest.json', 'checkpoint.json'):
    shutil.copy2(original / name, selector / name)
shutil.copy2(source / 'config.json', selector / 'config.json')
dirs = {'control': OUT / 'control'}
shutil.copytree(selector, dirs['control'])
for mode, name in [('git', 'git'), ('pf', 'pf'), ('pf', 'pf-isolation')]:
    dirs[name] = clone_sql_run(selector, OUT / name, 'check-' + name, text_registration=mode)

# Reuse the repository's offline transport wrapper; the production zipapp and
# recorded configuration stay fixed, and the wrapper forbids external sockets.
popen = subprocess.Popen


def launch(args, *other, **kwargs):
    if len(args) > 1 and str(args[1]).endswith('novamind-operation') and kwargs.get('env', {}).get('NOVAMIND_SERVER_MODE') == '1':
        args = [args[0], str(ROOT / 'tests/preflight_server.py'), *args[1:]]
    return popen(args, *other, **kwargs)


subprocess.Popen = launch


def state(directory):
    conn = load_session_db(directory / 'world.nmdb')
    result = {}
    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        table = row[0]
        if table.startswith('sqlite_'):
            continue
        columns = [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")') if r[1] != 'submitted_at']
        select = ','.join('"' + c + '"' for c in columns)
        result[table] = sorted([list(r) for r in conn.execute(f'SELECT {select} FROM "{table}"')], key=str)
    conn.close()
    return result


def saved(runner, day):
    runner._save_checkpoint(day)
    return checkpoint_directory(runner.workspace_dir, runner._load_checkpoint())


def same(actual, expected):
    assert actual.keys() == expected.keys()
    assert not [name for name in actual if actual[name] != expected[name]], [name for name in actual if actual[name] != expected[name]]


runners = {}
checks = dict(source_run=source.name, source_day=28, source_snapshot=str(original),
              source_commit=pointer['source_commit'], external_model_calls=0,
              method='Real checkpoint, production zipapp, tests/preflight_server.py deterministic HTTP replies; submitted_at excluded from table comparison')
try:
    baseline = state(original)
    for name, path in dirs.items():
        runner = BashAgentRunner(continue_from=path, api_key='offline-only')
        runners[name] = runner
        BashAgentToolExecutor(runner.agent_workspace, require_sandbox=True).verify_sandbox()
        runner.setup()
        runner._restore_from_checkpoint(runner._load_checkpoint())
        assert runner._get_game_status()['day'] == 28
        assert (runner.agent_workspace / 'MEMORY.md').read_bytes() == (original / 'agent_workspace/MEMORY.md').read_bytes()
        assert runner._git('rev-parse','HEAD').stdout.strip() == subprocess.check_output(['git','rev-parse','HEAD'],cwd=original / 'agent_workspace',text=True).strip()
        same(state(saved(runner,28)), baseline)
    assert len({r._server_port for r in runners.values()}) == 4
    assert not list(dirs['git'].rglob('sql-evidence*'))
    checks['restored_all_tables'] = len(baseline)
    checks['restored_rng'] = '_rng_states' in baseline
    checks['restored_workspace_memory_git_scripts'] = True
    outcomes = {}
    for name, runner in runners.items():
        outcomes[name] = runner._http_post('/next-week', dict(rationale='offline checkpoint acceptance only',
            predictions={h:dict(point=900000,lower=-100000,upper=1200000)
                         for h in ('cash_1wk','cash_4wk','cash_12wk','cash_26wk')}), timeout=180)
        assert outcomes[name]['success']
        saved(runner,35)
    expected = state(checkpoint_directory(runners['control'].workspace_dir,runners['control']._load_checkpoint()))
    for name, runner in runners.items():
        same(state(checkpoint_directory(runner.workspace_dir,runner._load_checkpoint())), expected)
        assert outcomes[name] == outcomes['control']
    checks['equal_continuation_to_day35'] = True
    checks['day35_table_counts'] = {name:len(rows) for name,rows in expected.items()}
    left,right = runners['pf'],runners['pf-isolation']
    right_evidence = file_hash(right.workspace_dir / 'sql-evidence.sqlite')
    right_memory = (right.agent_workspace / 'MEMORY.md').read_bytes()
    left._execute_tool('write_file',dict(path='MEMORY.md',content='Isolated acceptance mutation; not an online observation.'))
    assert left._http_post('/daily-scripts',dict(name='isolation_probe',content="print('isolated branch')"))['success']
    response = left._execute_tool('bash',dict(command="./novamind-operation python-c \"import novamind_api as n; print(n.pricing.set_prices(A=13.0))\""))
    assert '13.0' in response and 'Error' not in response, response
    changed = state(saved(left,35))
    assert changed['config_history'] != expected['config_history']
    assert changed['_registered_scripts'] != expected['_registered_scripts']
    assert (right.agent_workspace / 'MEMORY.md').read_bytes() == right_memory
    assert file_hash(right.workspace_dir / 'sql-evidence.sqlite') == right_evidence
    checks['right_evidence_unchanged_before_own_checkpoint'] = True
    # Saving a conversation updates this branch's agent_sources snapshot hash.
    # Check isolation before asking the untouched branch to write its own checkpoint.
    same(state(saved(right,35)),expected)
    assert tree_hash(original) == original_hash
    checks['isolated_world_scripts_memory_evidence_source'] = True
    checks['status'] = 'passed'
    write_json(OUT / 'verification.json',checks)
    print(json.dumps(checks,ensure_ascii=False,indent=2))
finally:
    for runner in runners.values():
        runner._stop_server()
        runner.client.close()
    subprocess.Popen = popen
