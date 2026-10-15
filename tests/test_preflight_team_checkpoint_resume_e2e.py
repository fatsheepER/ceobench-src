import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import pytest

from saas_bench import team_batch as batch
from saas_bench.run_state import build_manifest, tree_hash, write_json


BOOTSTRAP = r'''
import contextlib, json, runpy, socket, sqlite3, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import httpx
from saas_bench import team_batch as batch
from saas_bench.db_protection import load_session_db
from saas_bench.run_state import file_hash, tree_hash, write_json
sys.path.insert(0, sys.argv[4])
from test_preflight_team_batch import PRICES
from test_preflight_team_checkpoint import fast_simulator
from test_preflight_team_flow import ADVANCE, calls, completion, final
from saas_bench.agents.bash_agent import run_test

mode, directory, group = sys.argv[1:4]
directory = Path(directory)
source = batch.ROOT
manifest = directory / 'batch/manifest.json'
holder, counts = {}, {}
lock = threading.Lock()
original_client, original_simulator, original_save = batch.checked_client, batch.simulator, batch.save_checkpoint

def no_env(*args, **kwargs):
    raise AssertionError('This offline CLI test cannot read credentials')
run_test.load_env_file = no_env
connect = socket.socket.connect
def local_only(sock, address):
    if isinstance(address, tuple) and address[0] not in ('127.0.0.1', '::1', 'localhost'):
        raise AssertionError('This offline CLI test cannot connect to external services')
    return connect(sock, address)
socket.socket.connect = local_only

def files(root):
    return {str(path.relative_to(root)): (
        path.lstat().st_mode,
        ('link', path.readlink().as_posix()) if path.is_symlink() else
        ('file', file_hash(path)) if path.is_file() else ('directory', None))
        for path in [root, *sorted(root.rglob('*'))]}

def tables(conn):
    return {name: sorted([tuple(row) for row in conn.execute('SELECT * FROM "' + name + '"')], key=repr)
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_stat%'")}

def evidence_tables(path):
    with contextlib.closing(sqlite3.connect(path)) as conn:
        result = tables(conn)
        result['private_state'] = [row for row in result['private_state'] if row[0] != 'admission_paused']
        return result

def observe_ready(team, attempt, record, boundary):
    holder['team'] = team
    if mode == 'resume' and boundary == 'ready':
        snapshot = Path(json.loads((directory / 'week-boundary.json').read_text())['snapshot'])
        state = json.loads((snapshot / 'team.json').read_text())
        assert team.server.tools.current_day == state['day'] == 7
        assert files(team.root / 'roles') == files(snapshot / 'roles'), 'Resume must restore every role file, link and mode'
        for role, runtime in team.roles.items():
            assert (runtime.workspace / 'notebook.md').read_text() == role + ' boundary notes\n'
            assert not (runtime.workspace / 'draft.py').exists()
            assert (runtime.workspace / 'policy-link').readlink() == Path('weekly_policy.py')
            assert tree_hash(runtime.workspace / '.git', exact=True) == tree_hash(snapshot / 'roles' / role / '.git', exact=True)
            assert (runtime.workspace / '.git/HEAD').read_bytes() == (snapshot / 'roles' / role / '.git/HEAD').read_bytes()
        with contextlib.closing(load_session_db(snapshot / 'world.nmdb')) as world:
            assert tables(team.server.conn) == tables(world), 'World SQL must return to the selected week boundary'
        assert team.server.simulator.rng.bit_generator.state == state['tools_rng']
        assert team.server.tools.rng.bit_generator.state == state['tools_rng']
        receipt = json.loads((snapshot / 'checkpoint.json').read_text())
        for filename, cutoff in receipt['log_cutoffs'].items():
            assert (team.root / filename).read_bytes()[:cutoff] == (snapshot / filename).read_bytes()[:cutoff]
        if group == 'pf':
            restored = evidence_tables(team.root / 'private/evidence.sqlite')
            frozen = evidence_tables(snapshot / 'private/evidence.sqlite')
            assert restored == frozen, 'PF provenance, private state and all evidence rows must match the snapshot'
            for runtime in team.roles.values():
                assert runtime.store.load_state('declaration:r1.1') is not None
                assert runtime.store.load_state('declaration:r2.1') is None
            assert json.loads((directory / 'kill-ready.json').read_text())['pf_versions'] > len(restored['versions'])
            with contextlib.closing(sqlite3.connect(team.root / 'private/evidence.sqlite')) as conn:
                cutoff = conn.execute('SELECT coalesce(max(seq),0) FROM requests WHERE branch=?',
                    (state['evidence']['identity']['branch_id'],)).fetchone()[0]
                assert cutoff == state['evidence']['cutoff']
        write_json(directory / 'restored-ready.json', dict(day=7, roles=True, git_history=True,
            sql=True, rng=True, log_cutoffs=True, pf_evidence=group == 'pf'))
    snapshot = original_save(team, attempt, record, boundary)
    if mode == 'interrupt' and boundary == 'week_boundary' and team.server.tools.current_day == 7:
        write_json(directory / 'week-boundary.json', dict(snapshot=str(snapshot)))
    return snapshot
batch.save_checkpoint = observe_ready

class ModelHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        role = self.headers['x-opencode-session'].rsplit(':', 1)[-1]
        team = holder['team']
        day = team.server.tools.current_day
        with lock:
            key = role, day
            counts[key] = counts.get(key, 0) + 1
            turn = counts[key]
            with (directory / (mode + '-model-requests.jsonl')).open('a') as stream:
                stream.write(json.dumps(dict(role=role, day=day, turn=turn, body=body)) + '\n')
        if mode == 'interrupt' and day == 7 and role == 'ceo' and turn == 2:
            for name, runtime in team.roles.items():
                assert (runtime.workspace / 'notebook.md').read_text() == name + ' midweek notes\nextra notes\n'
                assert (runtime.workspace / 'draft.py').read_text() == 'LIMIT = 99\n'
                assert getattr(runtime.executor, '_pf_service', None) is None and not runtime.executor.preserved_process
                records = json.loads((runtime.workspace / 'registrations.json').read_text())['records']
                assert records['r2'][0]['text'] == 'Midweek draft'
                if group == 'pf':
                    assert runtime.store.load_state('declaration:r2.1') is not None
            requests = team.server.public_request_state()
            assert requests['active'] == [] and requests['queued'] is False
            write_json(directory / 'kill-ready.json', dict(day=day, phase=team.phase,
                role_files_written=list(team.roles), public_requests_closed=True,
                pf_versions=len(evidence_tables(team.root / 'private/evidence.sqlite')['versions']) if group == 'pf' else None))
            threading.Event().wait(60)
            return
        if mode == 'interrupt' and day in (0, 7) and turn == 1:
            midweek = day == 7
            note = role + (' midweek notes\n' if midweek else ' boundary notes\n')
            command = ("printf 'extra notes\\n' >> notebook.md; chmod 0600 weekly_policy.py; "
                "rm policy-link; ln -s draft.py policy-link") if midweek else (
                'chmod 0750 weekly_policy.py; ln -s weekly_policy.py policy-link')
            actions = [('write_file', dict(path='notebook.md', content=note)),
                ('write_file', dict(path='weekly_policy.py', content='LIMIT = 9\n' if midweek else 'LIMIT = 3\n'))]
            if midweek:
                actions.append(('write_file', dict(path='draft.py', content='LIMIT = 99\n')))
            actions.extend([('bash', dict(command=command)), ('text_create', dict(
                text='Midweek draft' if midweek else 'Boundary plan', objects=[dict(kind='metric', id='cash')],
                references=[], applies=str(day) + '-', reason='Save the weekly planning rationale'))])
            message = calls(*actions)
        else:
            message = calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final('Continue the saved weekly plan.')
        response = completion(message, body.get('stream', False))
        self.send_response(response.status_code)
        self.send_header('content-type', response.headers.get('content-type', 'application/json'))
        self.send_header('Content-Length', str(len(response.content)))
        self.end_headers()
        self.wfile.write(response.content)

server = ThreadingHTTPServer(('127.0.0.1', 0), ModelHandler)
server.daemon_threads = True
threading.Thread(target=server.serve_forever, daemon=True).start()

class LocalTransport(httpx.BaseTransport):
    def __init__(self):
        self.transport = httpx.HTTPTransport(trust_env=False)

    def handle_request(self, request):
        local = httpx.Request(request.method, 'http://127.0.0.1:' + str(server.server_port) + '/model',
            headers=request.headers, content=request.content, extensions=request.extensions)
        response = self.transport.handle_request(local)
        try:
            return httpx.Response(response.status_code, headers=response.headers, content=response.read())
        finally:
            response.close()

    def close(self):
        self.transport.close()

def client(role, session, **options):
    options['transport'] = LocalTransport()
    return original_client(role, session, **options)
batch.checked_client = client

def simulator(conn, config, rng, **options):
    fast = fast_simulator(conn, config, rng)
    original = original_simulator(conn, config, rng, **options)
    fast.customer_simulator = original.customer_simulator
    fast.shock_manager = original.shock_manager
    return fast
batch.simulator = simulator

def cli(command, *arguments):
    sys.argv = [str(source / 'scripts/team_batch.py'), command, *map(str, arguments)]
    runpy.run_path(sys.argv[0], run_name='__main__')

try:
    if mode == 'freeze':
        pricing = directory / 'prices.json'
        write_json(pricing, PRICES)
        cli('seal-d0', '--seed', '42', '--output', directory / 'd0', '--pricing', pricing,
            '--price-source', PRICES['source'])
        cli('freeze', '--d0', '42=' + str(directory / 'd0'), '--output', directory / 'batch',
            '--pricing', pricing, '--price-source', PRICES['source'], '--category', 'engineering')
    else:
        entry = next(row for row in batch.read(manifest)['runs'] if row['group'] == group)
        cli('resume' if mode == 'resume' else 'run', '--manifest', manifest, '--run-id', entry['run_id'])
finally:
    server.shutdown()
    server.server_close()
'''


def children(pid):
    result = set()
    for path in Path('/proc', str(pid), 'task').glob('*/children'):
        try:
            result.update(map(int, path.read_text().split()))
        except FileNotFoundError:
            continue
    return result


def process_group_members(group):
    members = []
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            continue
        if int(fields[2]) == group:
            members.append(int(path.parent.name))
    return members


@pytest.mark.parametrize('group', ['git', 'pf'])
def test_real_cli_manual_resume_after_midweek_sigkill_restores_selected_week_boundary(tmp_path, group):
    source = tmp_path / 'source'
    for name in ('src', 'scripts', 'public', 'public_sources'):
        shutil.copytree(batch.ROOT / name, source / name,
            ignore=shutil.ignore_patterns('__pycache__', '.env', '.env.*'))
    baseline = os.environ.get('CEOBENCH_CHECKPOINT_RESUME_BASELINE')
    if baseline:
        (source / 'src/saas_bench/team_checkpoint.py').write_bytes(subprocess.check_output(
            ['git', 'show', baseline + ':src/saas_bench/team_checkpoint.py'], cwd=batch.ROOT))
    subprocess.run(['git', 'init', '--quiet'], cwd=source, check=True)
    subprocess.run(['git', 'add', 'src', 'scripts'], cwd=source, check=True)
    tree = subprocess.check_output(['git', 'write-tree'], cwd=source, text=True).strip()
    commit = subprocess.check_output(['git', '-c', 'user.name=Offline', '-c', 'user.email=offline@example.invalid',
        'commit-tree', tree, '-m', 'fix: offline checkpoint recovery proof'], cwd=source, text=True).strip()
    subprocess.run(['git', 'update-ref', '--no-deref', 'HEAD', commit], cwd=source, check=True)
    assert subprocess.check_output(['git', 'for-each-ref', 'refs/heads'], cwd=source) == b''
    write_json(source / 'public/build.json', build_manifest(source, source / 'public'))
    directory = tmp_path / 'offline'
    directory.mkdir()
    bootstrap = tmp_path / 'offline-cli.py'
    bootstrap.write_text(BOOTSTRAP)
    env = dict(os.environ, PYTHONPATH=str(source / 'src'), PYTHONDONTWRITEBYTECODE='1',
        PYTHON_DOTENV_DISABLED='1', PYTHONHASHSEED='0', OPENCODE_API_KEY='OFFLINE-ONLY')
    for name in ('CEOBENCH_RECOVERY_CHECKPOINTS',
                 'BOSSBENCH_LLM_REPLAY_DB', 'ORACLE_MODE', 'CEOBENCH_READ_ONLY_TASK'):
        env.pop(name, None)

    def command(mode):
        return [os.sys.executable, str(bootstrap), mode, str(directory), group, str(batch.ROOT / 'tests')]

    frozen = subprocess.run(command('freeze'), cwd=source, env=env, capture_output=True, text=True, timeout=120)
    assert frozen.returncode == 0, frozen.stderr
    output = directory / 'interrupt-output.txt'
    with output.open('w') as stream:
        process = subprocess.Popen(command('interrupt'), cwd=source, env=env, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 150
            while not (directory / 'kill-ready.json').exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.05)
            assert (directory / 'kill-ready.json').exists(), output.read_text()
            assert children(process.pid) == set(), 'All role tool processes must close before the worker is killed'
            ready = batch.read(directory / 'kill-ready.json')
            assert ready['day'] == 7 and ready['phase'] == 'ceo'
            assert ready['role_files_written'] == ['ceo', 'growth', 'ops_finance']
            process.kill()
            assert process.wait(timeout=15) == -signal.SIGKILL
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
    assert process_group_members(process.pid) == [], 'Killed worker must leave no subprocesses'
    manifest_path = directory / 'batch/manifest.json'
    manifest = batch.read(manifest_path)
    entry = next(row for row in manifest['runs'] if row['group'] == group)
    first = batch.attempts(entry)[0]
    snapshot = Path(batch.read(directory / 'week-boundary.json')['snapshot'])
    state = batch.read(snapshot / 'team.json')
    assert state['day'] == 7 and state['stage'] == state['phase'] == 'boundary'
    assert tree_hash(first / 'runtime/roles', exact=True) != tree_hash(snapshot / 'roles', exact=True)
    immutable = {path: path.read_bytes() for path in (manifest_path, first / 'attempt.json')}
    source_before, snapshot_before = tree_hash(first, exact=True), tree_hash(snapshot, exact=True)
    resumed = subprocess.run(command('resume'), cwd=source,
        env=dict(env, CEOBENCH_RECOVERY_CHECKPOINTS=json.dumps({entry['run_id']: str(snapshot)})),
        capture_output=True, text=True, timeout=180)
    assert resumed.returncode == 0, resumed.stderr
    observed = batch.read(directory / 'restored-ready.json')
    assert observed == dict(day=7, roles=True, git_history=True, sql=True, rng=True,
        log_cutoffs=True, pf_evidence=group == 'pf')
    later = batch.attempts(entry)[-1]
    record = batch.read(later / 'attempt.json')
    assert record['status'] == 'finished' and record['day'] == 28 and record['attempt'] == 2
    assert record['source_checkpoint'] == str(snapshot)
    for role in ('ceo', 'growth', 'ops_finance'):
        rows = [json.loads(line) for line in (later / 'runtime/private' / role / 'model-usage.jsonl').read_text().splitlines()]
        assert any(row['event'] == 'http_request' for row in rows)
        assert any(row['event'] == 'response' and row['usage']['input_tokens'] == 20 for row in rows)
        assert (later / 'runtime/roles' / role / 'notebook.md').read_text() == role + ' boundary notes\n'
        assert not (later / 'runtime/roles' / role / 'draft.py').exists()
    requests = [json.loads(line) for line in (directory / 'resume-model-requests.jsonl').read_text().splitlines()]
    assert {row['role'] for row in requests if row['day'] == 7} == {'ceo', 'growth', 'ops_finance'}
    assert all('Midweek draft' not in json.dumps(row['body']) for row in requests if row['day'] == 7)
    assert all(path.read_bytes() == original for path, original in immutable.items())
    assert tree_hash(first, exact=True) == source_before and tree_hash(snapshot, exact=True) == snapshot_before
    print(json.dumps(dict(group=group, interrupted_day=ready['day'], resumed_day=record['day'],
        restored=observed, checkpoint_rollback=record['checkpoint_rollback'], source_preserved=True)))
