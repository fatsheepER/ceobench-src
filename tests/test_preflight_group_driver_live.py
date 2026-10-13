import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from saas_bench import team_batch as batch
from saas_bench.run_state import file_hash, write_json
from test_preflight_team_batch import frozen


BOOTSTRAP = r'''
import json, os, runpy, signal, socket, subprocess, sys, time
from pathlib import Path
import httpx
from saas_bench import team_batch as batch, systemd_guard
from saas_bench.run_lifecycle import RunCancelled
sys.path.insert(0, str(batch.ROOT / 'tests'))
from test_preflight_team_checkpoint import fast_simulator
from test_preflight_team_flow import ADVANCE, calls, completion, final
sys.path.insert(0, str(batch.ROOT / 'scripts'))
import team_controller, team_observer

connect = socket.socket.connect
def local_only(sock, address):
    if isinstance(address, tuple) and address[0] not in ('127.0.0.1', '::1', 'localhost'):
        raise AssertionError('Live driver regression cannot access external network')
    return connect(sock, address)
socket.socket.connect = local_only
def no_env_file(*args, **kwargs):
    raise AssertionError('Live driver regression cannot read .env')
from saas_bench.agents.bash_agent import run_test
run_test.load_env_file = no_env_file
team_observer.round5.quota_health = lambda: {
    window: dict(status='ok', percent=1) for window in ('rolling', 'weekly', 'monthly')}
poll = team_observer.poll
team_controller.poll = lambda *args, **kwargs: poll(*args,
    quota_fn=team_observer.round5.quota_health, **kwargs)

def scope_command(command, unit, **kwargs):
    return command
def unit_status(unit, **kwargs):
    code = kwargs.get('exit_code')
    return dict(unit=unit, invocation_id=kwargs.get('invocation_id') or unit,
        control_group='/offline/' + unit, scope_empty=code is not None,
        result='running' if code is None else 'success' if code == 0 else 'exit-code', exit_code=code)
systemd_guard.scope_command = team_controller.scope_command = scope_command
systemd_guard.unit_status = team_controller.unit_status = unit_status
team_controller.scope_state = lambda saved, **kwargs: unit_status(saved['unit'], **kwargs)
popen = subprocess.Popen
def launch(command, **kwargs):
    if len(command) > 1 and command[1].endswith('/scripts/team_controller.py'):
        command = [command[0], __file__, *command[1:]]
    return popen(command, **kwargs)
subprocess.Popen = launch
def signal_scope(saved, signum):
    policy = Path(sys.argv[sys.argv.index('--policy') + 1])
    events = [json.loads(line) for line in (policy.parent / 'controller-events.jsonl').read_text().splitlines()]
    pid = next(row['pid'] for row in reversed(events)
        if row['event'] == 'worker_launched' and row['unit'] == saved['unit'])
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass
    return saved
team_controller.signal_scope = signal_scope
checked = batch.checked_client
def client(role, session, **kwargs):
    def respond(request):
        body = json.loads(request.content)
        run = sys.argv[sys.argv.index('--worker') + 1]
        policy = Path(sys.argv[sys.argv.index('--policy') + 1])
        manifest = batch.read(batch.read(policy)['identity']['manifest'])
        git_first = manifest['scheduling']['queues']['git'][0]
        pf_second = batch.selected(manifest, manifest['scheduling']['queues']['pf'][1])
        if run == pf_second['run_id']:
            raise RunCancelled('requested_pause')
        if run == git_first and '--resume' not in sys.argv:
            deadline = time.monotonic() + 180
            while not batch.attempts(pf_second):
                if time.monotonic() > deadline:
                    raise AssertionError('PF did not join while Git was executing')
                time.sleep(.05)
            raise RunCancelled('requested_pause')
        if run == git_first and role == 'ceo' and any(
                message.get('role') == 'tool' for message in body['messages']):
            raise RunCancelled('requested_pause')
        command = './novamind-operation python-c "import novamind_api as nm; print(nm.vars.current_day)"'
        if run != git_first:
            command = ADVANCE
        message = calls(('bash', dict(command=command))) if role == 'ceo' else final('Use public data.')
        return completion(message, body.get('stream', False))
    return checked(role, session, **dict(kwargs, transport=httpx.MockTransport(respond)))
batch.checked_client = client
simulator = batch.simulator
def offline_simulator(conn, config, rng, *, client=None):
    fast = fast_simulator(conn, config, rng)
    original = simulator(conn, config, rng, client=client)
    fast.customer_simulator = original.customer_simulator
    fast.shock_manager = original.shock_manager
    return fast
batch.simulator = offline_simulator
target = sys.argv.pop(1)
sys.argv[0] = target
runpy.run_path(target, run_name='__main__')
'''


@pytest.mark.parametrize('two_seeds', [False, True])
def test_real_group_driver_live_and_resume_keep_worker_and_model_entries(frozen, tmp_path, two_seeds):
    base, original, _, sources = frozen
    sources = dict(sources)
    seeds = [1044, 1045] if two_seeds else original['seeds']
    if two_seeds:
        for seed in seeds:
            sources[seed] = tmp_path / f'd0-{seed}'
            batch.seal_d0(seed, sources[seed], original['pricing'],
                price_source=original['price_source'], public_dir=base / 'public')
    manifest = batch.freeze_batch(tmp_path / 'formal',
        {seed: sources[seed] for seed in seeds}, original['pricing'],
        price_source=original['price_source'], public_dir=base / 'public',
        selection_rule=original['seed_selection_rule'], excluded_seeds=(*original['excluded_debug_seeds'], 1043),
        scheduling='group_independent')
    path = tmp_path / 'formal/manifest.json'
    expected = {key: manifest[key] for key in ('batch_id', 'source_root', 'public_dir', 'source_commit')}
    expected.update(manifest=str(path), manifest_sha256=file_hash(path), host_logs=str(tmp_path / 'host-logs'),
        runs=[{key: row[key] for key in ('run_id', 'output_dir')} for row in manifest['runs']])
    policy = tmp_path / 'policy.json'
    write_json(policy, dict(identity=expected))
    bootstrap = tmp_path / 'offline-driver.py'
    bootstrap.write_text(BOOTSTRAP)
    env = dict(os.environ, OPENCODE_API_KEY='OFFLINE-ONLY', PYTHON_DOTENV_DISABLED='1',
        PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
        PYTHONPATH=os.pathsep.join(str(batch.ROOT / part) for part in ('src', 'tests')))
    command = [sys.executable, str(bootstrap), str(batch.ROOT / 'scripts/team_driver.py'),
        '--policy', str(policy), '--live']
    first = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
    assert first.returncode == 2, first.stderr
    assert not (path.parent / 'HOLD').exists()
    first_git = batch.selected(manifest, manifest['scheduling']['queues']['git'][0])
    paused = batch.read(batch.attempts(first_git)[0] / 'attempt.json')
    assert paused['status'] == 'paused' and paused['recovery_validation']['eligible']
    first_pf = batch.selected(manifest, manifest['scheduling']['queues']['pf'][0])
    second_pf = batch.selected(manifest, manifest['scheduling']['queues']['pf'][1])
    if two_seeds:
        assert first_pf['seed'] != second_pf['seed'] and set(manifest['seeds']) == {1044, 1045}
        assert all(row['seed'] != 1043 for row in manifest['runs'])
    assert batch.read(batch.attempts(first_pf)[0] / 'attempt.json')['status'] == 'finished'
    assert batch.read(batch.attempts(second_pf)[0] / 'attempt.json')['status'] == 'paused'
    assert not batch.attempts(batch.selected(manifest, manifest['scheduling']['queues']['pf'][2]))
    assert not batch.attempts(batch.selected(manifest, manifest['scheduling']['queues']['git'][1]))
    events = [json.loads(line) for line in (tmp_path / 'controller-events.jsonl').read_text().splitlines()]
    joined = next(row for row in events if row['event'] == 'worker_launched'
        and row['run_id'] == manifest['scheduling']['queues']['pf'][1])
    git_exit = next(row for row in events if row['event'] == 'worker_exited' and row['run_id'] == first_git['run_id'])
    assert events.index(joined) < events.index(git_exit)
    assert joined['launch_reason'] == 'previous_group_run_terminal'
    resumed = subprocess.run([*command, '--resume'], env=env, capture_output=True, text=True, timeout=600)
    assert resumed.returncode == 2, resumed.stderr
    assert not (path.parent / 'HOLD').exists()
    attempts = batch.attempts(first_git)
    assert len(attempts) == 2
    assert batch.read(attempts[-1] / 'attempt.json')['source_checkpoint'] == paused['checkpoint']
    terminal = batch.read(tmp_path / 'controller-terminal.json')
    assert set(terminal) == {first_pf['run_id']}
    assert all(row['unit']['scope_empty'] for row in terminal.values())
    for entry in (first_git, first_pf):
        current = batch.attempts(entry)[-1]
        record = batch.read(current / 'attempt.json')
        assert record['status'] == ('paused' if entry['group'] == 'git' else 'finished')
        assert record['day'] == (0 if entry['group'] == 'git' else 112)
        rows = [json.loads(line) for line in (current / 'runtime/private/ceo/model-usage.jsonl').read_text().splitlines()]
        assert any(row['event'] == 'http_request' for row in rows)
        assert any(row['event'] == 'response' and row.get('usage', {}).get('input_tokens') == 20
            and row.get('usage', {}).get('output_tokens') == 10 for row in rows)
    assert batch.read(tmp_path / 'controller-exit.json')['children_reaped']
    assert batch.read(tmp_path / 'driver-exit.json')['returncode'] == 2
    assert (attempts[-1] / 'runtime/roles/ceo/MEMORY.md').read_bytes() == (
        Path(paused['checkpoint']) / 'roles/ceo/MEMORY.md').read_bytes()
    tool_messages = [json.loads(line) for line in (
        attempts[-1] / 'runtime/private/ceo/model-usage.jsonl').read_text().splitlines()]
    assert any(message.get('role') == 'tool' and message.get('name') == 'bash'
        and message.get('content') == '0\n' for row in tool_messages if row['event'] == 'request'
        for message in row['request']['messages'])
