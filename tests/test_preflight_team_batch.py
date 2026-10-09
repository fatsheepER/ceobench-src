import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile

import httpx
import pytest

from saas_bench import team_batch as batch
from saas_bench.role_policy import ROLES
from saas_bench.run_state import build_manifest, file_hash, tree_hash, write_json
from saas_bench.team_checkpoint import validate_snapshot
from saas_bench.team_usage import aggregate
from test_preflight_team_flow import ADVANCE, calls, completion, final


PRICES = dict(source='offline acceptance fixture', basis='USD per 1000 tokens',
              rates={batch.MODEL: dict(input=0.001, output=0.002, cache_read=0.0001, cache_write=0.001)})


def test_live_batch_refuses_read_only_environment_before_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, 'validate_manifest', lambda path: {})
    monkeypatch.setenv('CEOBENCH_READ_ONLY_TASK', '1')
    with pytest.raises(ValueError, match='read-only'):
        batch.prepare(tmp_path / 'manifest.json', 'unused', live=True)
    assert not (tmp_path / 'runs').exists()


@pytest.fixture(scope='module')
def frozen(tmp_path_factory):
    base = Path(os.environ.get('CEOBENCH_TEAM_BATCH_ARTIFACTS', tmp_path_factory.mktemp('team-batch')))
    base.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix='frozen-', dir=base))
    public = base / 'public'
    shutil.copytree(batch.ROOT / 'public', public)
    write_json(public / 'build.json', build_manifest(batch.ROOT, public))
    sources = {}
    for seed in (101, 102, 103, 42):
        sources[seed] = base / f'd0-{seed}'
        batch.seal_d0(seed, sources[seed], PRICES, price_source=PRICES['source'], public_dir=public)
    manifest = batch.freeze_batch(base / 'formal', {seed: sources[seed] for seed in (101, 102, 103)}, PRICES,
        price_source=PRICES['source'], excluded_seeds=(42, 123), selection_rule='First unused integers from 101',
        order_seed=17, public_dir=public)
    engineering = batch.freeze_batch(base / 'engineering', {42: sources[42]}, PRICES,
        price_source=PRICES['source'], category='engineering', public_dir=public)
    return base, manifest, engineering, sources


def changed_manifest(frozen, tmp_path, change, *, reanchor=True):
    base, manifest, _, _ = frozen
    copied = copy.deepcopy(manifest)
    for entry in copied['runs']:
        entry['output_dir'] = str(tmp_path / 'runs' / entry['run_id'])
    change(copied)
    path = tmp_path / 'manifest.json'
    write_json(path, copied)
    write_json(tmp_path / 'freeze.json', dict(manifest_sha256=file_hash(path) if reanchor else 'changed'))
    return path


def test_model_free_validation_enumerates_exact_formal_pairs_and_engineering(frozen):
    base, manifest, engineering, _ = frozen
    assert len(batch.validate_manifest(base / 'formal/manifest.json')['runs']) == 12
    assert {(r['seed'], r['repeat'], r['group']) for r in manifest['runs']} == {
        (seed, repeat, group) for seed in (101, 102, 103) for repeat in (1, 2) for group in ('git', 'pf')}
    assert len({r['run_id'] for r in manifest['runs']}) == len({r['output_dir'] for r in manifest['runs']}) == 12
    assert all(r['stop_after_day'] == 112 for r in manifest['runs'])
    assert all(r['category'] == 'engineering' and r['seed'] == 42 and r['stop_after_day'] == 28 for r in engineering['runs'])
    env = dict(os.environ, PYTHONPATH=str(batch.ROOT / 'src'), PYTHONHASHSEED='0')
    result = subprocess.run([os.sys.executable, str(batch.ROOT / 'scripts/team_batch.py'), 'validate',
                             '--manifest', str(base / 'formal/manifest.json')], env=env, capture_output=True, text=True, check=True)
    assert len(json.loads(result.stdout)) == 12
    assert batch.PARAMETERS['model'] == batch.MODEL and batch.PARAMETERS['reasoning_effort'] == 'high'
    assert all('D497' in p['system'] and 'D112' not in p['system'] for prompts in manifest['prompts'].values() for p in prompts.values())


@pytest.mark.parametrize('change', [
    lambda m: m['runs'][0].update(group='prefix'),
    lambda m: m['runs'][0].update(seed=42),
    lambda m: m['runs'].append(copy.deepcopy(m['runs'][0])),
    lambda m: m['runs'][0].update(d0=m['d0']['102']['path'] if m['runs'][0]['seed'] != 102 else m['d0']['101']['path']),
    lambda m: m['parameters'].update(reasoning_effort='low'),
    lambda m: m['pricing']['rates'][batch.MODEL].update(input=99),
    lambda m: m.update(build=dict(m['build'], source_sha256='changed')),
    lambda m: m['runs'][0].update(output_dir=m['runs'][1]['output_dir']),
    lambda m: m['runs'][0].update(stop_after_day=28),
])
def test_validation_rejects_mismatch_and_duplicate(frozen, tmp_path, change):
    path = changed_manifest(frozen, tmp_path, change)
    with pytest.raises(ValueError):
        batch.validate_manifest(path)


def test_validation_rejects_legacy_scalar_timeout(frozen, tmp_path):
    path = changed_manifest(frozen, tmp_path, lambda m: m['parameters'].update(timeout_seconds=60))
    with pytest.raises(ValueError, match='Frozen model parameters'):
        batch.validate_manifest(path)


def test_manifest_anchor_rejects_prompt_change_and_original_sources_are_sealed(frozen, tmp_path):
    _, _, _, sources = frozen
    path = changed_manifest(frozen, tmp_path, lambda m: m['prompts']['git']['ceo'].update(system='modified'), reanchor=False)
    with pytest.raises(ValueError, match='manifest changed'):
        batch.validate_manifest(path)
    for source in sources.values():
        assert source.stat().st_mode & 0o222 == 0
        assert (source / 'world.nmdb').stat().st_mode & 0o222 == 0


def test_same_seed_clones_share_d0_but_have_independent_sessions_worlds_and_histories(frozen, tmp_path):
    _, manifest, _, sources = frozen
    seed = manifest['runs'][0]['seed']
    entries = [next(r for r in manifest['runs'] if r['seed'] == seed and r['group'] == g) for g in ('git', 'pf')]
    teams = [batch.make_team(manifest, entry, tmp_path / entry['group']) for entry in entries]
    before = file_hash(sources[seed] / 'world.nmdb')
    try:
        sessions = [r.identity.session_id for team in teams for r in team.roles.values()]
        assert len(set(sessions)) == 6
        assert all(asdict(team.server.tools.config) == asdict(batch.configuration(seed, PRICES)) for team in teams)
        assert teams[0].server.tools.rng.bit_generator.state == teams[1].server.tools.rng.bit_generator.state
        assert len({r.workspace for team in teams for r in team.roles.values()}) == 6
        for team in teams:
            assert len({r.registry.workspace for r in team.roles.values()}) == 3
            for rr in team.roles.values():
                assert rr.agent.client.max_retries == 2
                assert rr.agent.model == batch.MODEL and rr.agent.reasoning_effort == 'high'
                assert rr.usage.pricing == PRICES['rates']
        teams[0].server.conn.execute('UPDATE config_history SET price_A = 17')
        teams[0].server.conn.commit()
        (teams[0].roles['ceo'].workspace / 'MEMORY.md').write_text('ONLY FIRST CLONE')
        assert teams[1].server.conn.execute('SELECT price_A FROM config_history LIMIT 1').fetchone()[0] != 17
        assert teams[1].roles['ceo'].workspace.joinpath('MEMORY.md').read_text() == ''
        assert file_hash(sources[seed] / 'world.nmdb') == before
    finally:
        for team in teams:
            batch.close_team(team)


def test_duplicate_registration_hold_and_original_manifest_resume(frozen, tmp_path):
    path = changed_manifest(frozen, tmp_path, lambda m: None)
    manifest = batch.validate_manifest(path)
    entry = manifest['runs'][0]
    output = Path(entry['output_dir'])
    with batch.run_lock(output):
        first, record, source = batch.register_attempt(path, manifest, entry)
        assert source is None and record['attempt'] == 1
        with pytest.raises(ValueError, match='already registered'):
            batch.register_attempt(path, manifest, entry)
        team = batch.make_team(manifest, entry, first)
        try:
            snapshot = team.checkpoint()
        finally:
            batch.close_team(team)
        record.update(status='paused', checkpoint=str(snapshot))
        write_json(first / 'attempt.json', record)
        (tmp_path / 'HOLD').touch()
        with pytest.raises(ValueError, match='HOLD'):
            batch.register_attempt(path, manifest, entry, resume=True)
        (tmp_path / 'HOLD').unlink()
        second, later, checkpoint = batch.register_attempt(path, manifest, entry, resume=True)
        assert later['attempt'] == 2 and str(checkpoint) == record['checkpoint']
        assert file_hash(first / 'manifest.json') == file_hash(second / 'manifest.json') == file_hash(path)
        later.update(status='paused', checkpoint=str(second / 'runtime/checkpoints' / ('b' * 32)))
        write_json(second / 'attempt.json', later)
        second.joinpath('manifest.json').chmod(0o644)
        second.joinpath('manifest.json').write_text('{}')
        with pytest.raises(ValueError, match='original immutable manifest'):
            batch.register_attempt(path, manifest, entry, resume=True)


def test_checked_provider_parameters_and_receipts(tmp_path):
    from saas_bench.model_usage import ModelUsage
    requests = []
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        return completion(final('offline simulator narrative'), body.get('stream', False))
    client = batch.checked_client('simulator', 'offline', transport=httpx.MockTransport(respond))
    recorder = ModelUsage(tmp_path / 'simulator.jsonl', 'simulator', PRICES['rates'])
    recorder.expected_model = batch.MODEL
    recorder.attach(client)
    kwargs = dict(model=batch.MODEL, reasoning_effort='none', max_tokens=100, temperature=0.3,
                  messages=[dict(role='user', content='offline')],
                  extra_body={'thinking': {'type': 'disabled'}})
    try:
        recorder.call('chat', kwargs, lambda: client.chat.completions.create(**kwargs), day=0)
        assert requests[0]['thinking'] == {'type': 'disabled'} and requests[0]['reasoning_effort'] == 'none'
        assert [row['event'] for row in map(json.loads, (tmp_path / 'simulator.jsonl').read_text().splitlines())] == [
            'request', 'http_request', 'http_response', 'response']
        from openai import APIConnectionError
        client.max_retries = 0
        with pytest.raises(APIConnectionError):
            client.chat.completions.create(**dict(kwargs, reasoning_effort='high'))
        for changes in (dict(temperature=99), dict(max_tokens=999)):
            with pytest.raises(APIConnectionError):
                client.chat.completions.create(**dict(kwargs, **changes))
    finally:
        client.close()


@pytest.mark.parametrize('role', [*ROLES, 'simulator'])
def test_checked_client_uses_frozen_phase_timeouts(frozen, role):
    _, manifest, _, _ = frozen
    timeouts = []
    def respond(request):
        timeouts.append(request.extensions['timeout'])
        return completion(final('offline'), json.loads(request.content).get('stream', False))
    client = batch.checked_client(role, 'offline', transport=httpx.MockTransport(respond))
    kwargs = dict(model=batch.MODEL, reasoning_effort='high', max_tokens=16384, temperature=1.0,
                  messages=[dict(role='user', content='offline')], stream=True,
                  stream_options={'include_usage': True}, extra_body={'thinking': {'type': 'enabled'}})
    if role == 'simulator':
        kwargs.update(reasoning_effort='none', max_tokens=100, temperature=0.3,
                      extra_body={'thinking': {'type': 'disabled'}})
    try:
        with client.chat.completions.create(**kwargs) as response:
            list(response)
        expected = dict(connect=60, read=180, write=60, pool=60)
        assert timeouts == [expected]
        assert manifest['parameters']['timeout_seconds'] == expected
    finally:
        client.close()


def test_returned_model_mismatch_keeps_known_raw_receipts(tmp_path):
    from saas_bench.model_usage import ModelUsage
    def respond(request):
        response = completion(final('wrong model'))
        body = response.json()
        body['model'] = 'unexpected-model'
        return httpx.Response(200, json=body)
    client = batch.checked_client('simulator', 'offline', transport=httpx.MockTransport(respond))
    recorder = ModelUsage(tmp_path / 'mismatch.jsonl', 'simulator', PRICES['rates'])
    recorder.expected_model = batch.MODEL
    recorder.attach(client)
    kwargs = dict(model=batch.MODEL, reasoning_effort='none', max_tokens=100, temperature=0.3,
                  messages=[dict(role='user', content='offline')],
                  extra_body={'thinking': {'type': 'disabled'}})
    try:
        with pytest.raises(ValueError, match='model'):
            recorder.call('chat', kwargs, lambda: client.chat.completions.create(**kwargs), day=0)
        rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
        assert rows[-1]['event'] == 'response' and rows[-1]['usage']['input_tokens'] == 20
        assert rows[-1]['error'] is not None
        assert next(row for row in rows if row['event'] == 'http_response')['body']
    finally:
        client.close()


@pytest.mark.parametrize('group', ['git', 'pf'])
def test_offline_engineering_batch_runs_real_cli_and_pauses_resumes(frozen, tmp_path, group, monkeypatch):
    base, _, engineering, _ = frozen
    copied = copy.deepcopy(engineering)
    for entry in copied['runs']:
        entry['output_dir'] = str(tmp_path / 'runs' / entry['run_id'])
    path = tmp_path / 'manifest.json'
    write_json(path, copied)
    write_json(tmp_path / 'freeze.json', dict(manifest_sha256=file_hash(path)))
    run_id = next(entry['run_id'] for entry in copied['runs'] if entry['group'] == group)
    wire = []
    cleanup_checks = []
    prune = batch.prune_checkpoints
    def checked_prune(parent, ledger, directory):
        latest = Path(ledger['snapshots'][-1]['snapshot'])
        logs = list(parent.parent.glob('private/*/model-usage.jsonl'))
        def analysis():
            return (tree_hash(parent.parent / 'roles'), tree_hash(parent.parent / 'private'),
                tree_hash(latest), aggregate(logs, PRICES, category='engineering'))
        before = analysis()
        prune(parent, ledger, directory)
        assert analysis() == before
        cleanup_checks.append(str(latest))
    monkeypatch.setattr(batch, 'prune_checkpoints', checked_prune)
    pause_once = [True]
    def client(role):
        def respond(request):
            body = json.loads(request.content)
            wire.append(dict(role=role, body=body))
            if role == 'ceo' and pause_once[0]:
                pause_once[0] = False
                (tmp_path / 'HOLD').touch()
            message = calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final('Use existing public data.')
            return completion(message, body.get('stream', False))
        return batch.checked_client(role, 'offline', transport=httpx.MockTransport(respond))
    first = batch.prepare(path, run_id, live=True, client_factory=client, simulator_factory=batch.simulator)
    assert first['status'] == 'paused' and first['day'] == 0 and Path(first['checkpoint']).is_dir()
    assert 'Unresolved tool operations' in first['checkpoint_error']
    (tmp_path / 'HOLD').unlink()
    resumed = batch.prepare(path, run_id, live=True, resume=True, client_factory=client, simulator_factory=batch.simulator)
    assert resumed['status'] == 'finished' and resumed['day'] == 28
    assert resumed['checkpoint_reused_at_stop'] is True and resumed['runtime_phase'] == 'stopped'
    assert resumed['result']['status'] == 'reached_stop_day'
    assert first['manifest_sha256'] == resumed['manifest_sha256'] == file_hash(path)
    assert len(batch.attempts(batch.selected(copied, run_id))) == 2
    assert all(r['body']['model'] == batch.MODEL and r['body']['reasoning_effort'] == 'high' for r in wire)
    assert all('D497' in r['body']['messages'][0]['content'] and 'D28' not in r['body']['messages'][0]['content'] for r in wire)
    entry = batch.selected(copied, run_id)
    history = batch.attempts(entry)
    snapshot = Path(resumed['checkpoint'])
    snapshots = list(snapshot.parent.glob('*/checkpoint.json'))
    assert len(snapshots) == 2 and len(cleanup_checks) == 10
    states = [batch.read(p.parent / 'team.json') for p in snapshots]
    assert sum(s['day'] == 28 for s in states) == 1
    _, terminal = validate_snapshot(snapshot)
    assert terminal['day'] == 28 and terminal['stage'] == terminal['phase'] == 'boundary'
    messages = [json.loads(line) for line in (snapshot / 'private/messages.jsonl').read_text().splitlines()]
    assert [m['status'] for m in terminal['messages']].count('delivered') == 8
    assert sum(m['status'] == 'delivered' for m in messages) == 8
    logs = [p for attempt in history for p in attempt.glob('runtime/private/*/model-usage.jsonl')]
    assert aggregate(dict(engineering=logs, simulator=[]), PRICES, category='engineering') == resumed['result']['usage']
    raw = [json.loads(line) for p in logs for line in p.read_text().splitlines()]
    assert len({row['attempt_id'] for row in raw if row['event'] == 'http_request'}) == len(wire)
    if group == 'pf':
        assert (snapshot / 'private/evidence.sqlite').stat().st_size > 0
        assert terminal['evidence']
    source = Path(first['checkpoint'])
    assert batch.read(source.parent / '.retention-pins.json')[str(source)] == file_hash(source / 'checkpoint.json')
    before = tree_hash(snapshot)
    calls_before = len(wire)
    resumed['status'] = 'paused'
    write_json(history[-1] / 'attempt.json', resumed)
    terminal_resume = batch.prepare(path, run_id, live=True, resume=True,
        client_factory=client, simulator_factory=batch.simulator)
    assert terminal_resume['status'] == 'finished' and terminal_resume['day'] == 28
    assert terminal_resume['checkpoint_reused_at_stop'] is False
    assert batch.read(Path(terminal_resume['checkpoint']) / 'team.json')['phase'] == 'stopped'
    assert len(wire) == calls_before and tree_hash(snapshot) == before
    assert {key: value for key, value in terminal_resume['result']['usage'].items() if key != 'duplicates'} == {
        key: value for key, value in resumed['result']['usage'].items() if key != 'duplicates'}
    write_json(base / f'offline-batch-wire-{group}.json', wire)


def test_registration_publishes_complete_attempt_after_host_kill(frozen, tmp_path):
    path = changed_manifest(frozen, tmp_path, lambda manifest: None)
    manifest = batch.validate_manifest(path)
    entry = manifest['runs'][0]
    script = '''import os, signal, sys
from pathlib import Path
from saas_bench import team_batch as batch
path = Path(sys.argv[1])
manifest = batch.validate_manifest(path)
entry = manifest['runs'][0]
write = batch.write_json
def interrupted(path, value):
    if Path(path).name == 'attempt.json':
        os.kill(os.getpid(), signal.SIGKILL)
    write(path, value)
batch.write_json = interrupted
with batch.run_lock(Path(entry['output_dir'])):
    batch.register_attempt(path, manifest, entry)
'''
    env = dict(os.environ, PYTHONHASHSEED='0', PYTHONPATH=str(batch.ROOT / 'src'))
    with (tmp_path / 'registration-killed-host.log').open('w') as log:
        result = subprocess.run([os.sys.executable, '-c', script, str(path)],
            env=env, stdout=log, stderr=subprocess.STDOUT, timeout=30)
    assert result.returncode == -signal.SIGKILL
    assert not batch.attempts(entry)
    assert list(Path(entry['output_dir']).glob('.registration-*/manifest.json'))
    with batch.run_lock(Path(entry['output_dir'])):
        directory, record, source = batch.register_attempt(path, manifest, entry)
    assert record['attempt'] == 1 and source is None
    assert batch.read(directory / 'attempt.json') == record
    assert file_hash(directory / 'manifest.json') == file_hash(path)


@pytest.mark.parametrize('boundary', ['before_world', 'before_checkpoint'])
def test_killed_preparation_restarts_from_d0_and_preserves_failed_attempt(frozen, tmp_path, boundary):
    path = changed_manifest(frozen, tmp_path, lambda manifest: None)
    manifest = batch.validate_manifest(path)
    entry = manifest['runs'][0]
    script = '''import os, signal, sys
from saas_bench import team_batch as batch
def killed(*args, **kwargs):
    os.kill(os.getpid(), signal.SIGKILL)
if sys.argv[3] == 'before_world':
    batch.make_team = killed
else:
    batch.MultiAgentRuntime.checkpoint = killed
batch.prepare(sys.argv[1], sys.argv[2])
'''
    env = dict(os.environ, PYTHONHASHSEED='0', PYTHONPATH=str(batch.ROOT / 'src'))
    with (tmp_path / 'preparation-killed-host.log').open('w') as log:
        process = subprocess.run([os.sys.executable, '-c', script, str(path), entry['run_id'], boundary],
            env=env, stdout=log, stderr=subprocess.STDOUT, timeout=30)
    assert process.returncode == -signal.SIGKILL
    first = batch.attempts(entry)[0]
    original = batch.read(first / 'attempt.json')
    assert original['status'] == 'preparing' and original['checkpoint'] is None
    assert original['execution_started'] is False
    resumed = batch.prepare(path, entry['run_id'], resume=True)
    assert resumed['status'] == 'ready' and resumed['attempt'] == 2
    assert batch.read(Path(resumed['checkpoint']) / 'team.json')['day'] == 0
    assert batch.read(first / 'attempt.json')['reason'] == 'host_interrupted'
    assert len(batch.attempts(entry)) == 2
    assert all(file_hash(directory / 'manifest.json') == file_hash(path) for directory in batch.attempts(entry))


@pytest.mark.parametrize('started', [True, None])
def test_missing_checkpoint_after_execution_cannot_restart_from_d0(frozen, tmp_path, started):
    path = changed_manifest(frozen, tmp_path, lambda manifest: None)
    manifest = batch.validate_manifest(path)
    entry = manifest['runs'][0]
    with batch.run_lock(Path(entry['output_dir'])):
        directory, record, _ = batch.register_attempt(path, manifest, entry)
        record.update(status='failed', execution_started=started)
        batch.write_json(directory / 'attempt.json', record)
        with pytest.raises(ValueError, match='safe checkpoint'):
            batch.register_attempt(path, manifest, entry, resume=True)
    assert len(batch.attempts(entry)) == 1


def test_killed_host_and_failed_restore_retain_safe_checkpoint(frozen, tmp_path, monkeypatch):
    _, _, engineering, _ = frozen
    copied = copy.deepcopy(engineering)
    for entry in copied['runs']:
        entry['output_dir'] = str(tmp_path / 'runs' / entry['run_id'])
    path = tmp_path / 'manifest.json'
    write_json(path, copied)
    write_json(tmp_path / 'freeze.json', dict(manifest_sha256=file_hash(path)))
    entry = next(row for row in copied['runs'] if row['group'] == 'git')
    script = '''import os, signal, sys, json, httpx
from saas_bench import team_batch as batch
from test_preflight_team_flow import completion, final
def client(role):
    def respond(request):
        if role == 'ceo':
            os.kill(os.getpid(), signal.SIGKILL)
        body = json.loads(request.content)
        return completion(final('Use existing public data.'), body.get('stream', False))
    return batch.checked_client(role, 'offline', transport=httpx.MockTransport(respond))
batch.prepare(sys.argv[1], sys.argv[2], live=True, client_factory=client, simulator_factory=batch.simulator)
'''
    env = dict(os.environ, PYTHONHASHSEED='0', PYTHONPATH=os.pathsep.join(
        (str(batch.ROOT / 'src'), str(batch.ROOT / 'tests'))))
    with (tmp_path / 'killed-host.log').open('w') as log:
        process = subprocess.run([os.sys.executable, '-c', script, str(path), entry['run_id']],
            env=env, stdout=log, stderr=subprocess.STDOUT, timeout=30)
    assert process.returncode == -signal.SIGKILL
    first = batch.attempts(entry)[0]
    assert batch.read(first / 'attempt.json')['status'] == 'running'
    summary = batch.summarize(path)
    row = next(row for row in summary['runs'] if row['run_id'] == entry['run_id'])
    assert row['status'] == 'technical_failure' and row['reason'] == 'host_interrupted'
    checkpoint = batch.read(first / 'attempt.json')['checkpoint']
    def restore_failure(*args, **kwargs):
        raise ValueError('offline preparation failure')
    monkeypatch.setattr(batch.MultiAgentRuntime, 'restore', restore_failure)
    with pytest.raises(ValueError, match='preparation failure'):
        batch.prepare(path, entry['run_id'], live=True, resume=True)
    second = batch.attempts(entry)[1]
    assert batch.read(second / 'attempt.json')['checkpoint'] == checkpoint
    monkeypatch.undo()
    def client(role):
        def respond(request):
            body = json.loads(request.content)
            message = calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final('Use existing public data.')
            return completion(message, body.get('stream', False))
        return batch.checked_client(role, 'offline', transport=httpx.MockTransport(respond))
    resumed = batch.prepare(path, entry['run_id'], live=True, resume=True,
        client_factory=client, simulator_factory=batch.simulator)
    assert resumed['status'] == 'finished' and resumed['day'] == 28
    assert resumed['attempt'] == 3 and len(batch.attempts(entry)) == 3
    assert batch.read(first / 'attempt.json')['status'] == 'failed'
    assert batch.read(second / 'attempt.json')['status'] == 'failed'
    assert Path(checkpoint).is_dir()


@pytest.fixture
def managed_team(frozen, tmp_path):
    path = changed_manifest(frozen, tmp_path, lambda manifest: None)
    manifest = batch.validate_manifest(path)
    entry = next(row for row in manifest['runs'] if row['group'] == 'git')
    with batch.run_lock(Path(entry['output_dir'])):
        directory, record, _ = batch.register_attempt(path, manifest, entry)
        team = batch.make_team(manifest, entry, directory)
        try:
            record['status'] = 'ready'
            batch.save_checkpoint(team, directory, record, 'ready')
            yield team, directory, record, manifest, entry
        finally:
            batch.close_team(team)


def test_external_restore_permanently_pins_managed_source_before_reading(managed_team, tmp_path, monkeypatch):
    from saas_bench import team_checkpoint
    team, directory, record, manifest, _ = managed_team
    source = Path(record['checkpoint'])
    before = tree_hash(source)
    validate = team_checkpoint.validate_snapshot
    def pinned_validate(snapshot):
        assert batch.read(source.parent / '.retention-pins.json')[str(source)] == file_hash(source / 'checkpoint.json')
        return validate(snapshot)
    monkeypatch.setattr(team_checkpoint, 'validate_snapshot', pinned_validate)
    external_alias = tmp_path / 'external-snapshot-alias'
    external_alias.symlink_to(source, target_is_directory=True)
    restored = batch.MultiAgentRuntime.restore(external_alias, tmp_path / 'external-restore',
        client_factory=lambda role: batch.checked_client(role, 'offline', no_models=True), public_dir=manifest['public_dir'])
    batch.close_team(restored)
    manual = team.checkpoint(source.parent / 'manual-snapshot')
    source.parent.chmod(0o555)
    (source.parent / '.retention.lock').chmod(0o400)
    try:
        restored = batch.MultiAgentRuntime.restore(manual, tmp_path / 'manual-restore',
            client_factory=lambda role: batch.checked_client(role, 'offline', no_models=True), public_dir=manifest['public_dir'])
        batch.close_team(restored)
        assert str(manual) not in batch.read(source.parent / '.retention-pins.json')
    finally:
        source.parent.chmod(0o700)
        (source.parent / '.retention.lock').chmod(0o600)
    alias = source.parent / ('a' * 32)
    alias.symlink_to(manual, target_is_directory=True)
    for _ in range(4):
        batch.save_checkpoint(team, directory, record, 'week_boundary')
    ledger = batch.read(source.parent / '.retention.json')
    assert len(ledger['snapshots']) == 3 and str(source) in {item['snapshot'] for item in ledger['snapshots']}
    assert tree_hash(source) == before and manual.is_dir() and alias.is_symlink()
    assert not record.get('checkpoint_retention_errors')


@pytest.mark.parametrize('fault', ['bad_json', 'missing', 'unknown', 'outside', 'receipt',
    'candidate_symlink', 'parent_symlink', 'ledger_symlink', 'bad_pins', 'bad_attempt', 'partial_delete'])
def test_retention_errors_preserve_current_analysis_and_continue(managed_team, tmp_path, monkeypatch, fault):
    team, directory, record, manifest, entry = managed_team
    first = Path(record['checkpoint'])
    second = batch.save_checkpoint(team, directory, record, 'week_boundary')
    second_hash = tree_hash(second)
    logs = {p: file_hash(p) for p in (team.root / 'private').rglob('*.jsonl')}
    roles_hash = tree_hash(team.root / 'roles')
    parent, path = first.parent, first.parent / '.retention.json'
    ledger = batch.read(path)
    if fault == 'bad_json':
        path.write_text('{')
    elif fault == 'missing':
        path.unlink()
    elif fault == 'unknown':
        write_json(path, dict(ledger, version=99))
    elif fault == 'outside':
        ledger['snapshots'][0]['snapshot'] = str(tmp_path / 'outside')
        write_json(path, ledger)
    elif fault == 'receipt':
        (first / 'checkpoint.json').write_text('{}')
    elif fault == 'candidate_symlink':
        first.rename(tmp_path / 'moved-source')
        first.symlink_to(tmp_path / 'moved-source', target_is_directory=True)
    elif fault == 'parent_symlink':
        parent.rename(tmp_path / 'moved-parent')
        parent.symlink_to(tmp_path / 'moved-parent', target_is_directory=True)
    elif fault == 'ledger_symlink':
        path.rename(tmp_path / 'moved-ledger.json')
        path.symlink_to(tmp_path / 'moved-ledger.json')
    elif fault == 'bad_pins':
        write_json(parent / '.retention-pins.json', {str(first): 'wrong'})
    elif fault == 'bad_attempt':
        other = directory.parent / 'attempt-002'
        other.mkdir()
        (other / 'attempt.json').write_text('{')
    else:
        remove = shutil.rmtree
        def failed_remove(path, *args, **kwargs):
            if Path(path) == first:
                (first / 'world.nmdb').unlink(missing_ok=True)
                raise OSError('injected partial deletion failure')
            return remove(path, *args, **kwargs)
        monkeypatch.setattr(shutil, 'rmtree', failed_remove)
    third = batch.save_checkpoint(team, directory, record, 'week_boundary')
    assert second.is_dir() and tree_hash(second) == second_hash
    assert {p: file_hash(p) for p in logs} == logs and tree_hash(team.root / 'roles') == roles_hash
    assert third.is_dir() and record['checkpoint_retention_errors']
    assert first.exists()
    assert not record.get('checkpoint_error')
    if fault == 'bad_attempt':
        write_json(other / 'attempt.json', dict(checkpoint=None, source_checkpoint=None, wall_seconds=0))
    monkeypatch.setattr(team, 'run', lambda **options: type('Outcome', (), dict(day=0, reason='paused'))())
    team.failure = 'requested_pause'
    team._set_phase('paused')
    completed = batch.execute(team, manifest, entry, directory, record, tmp_path / 'HOLD')
    assert completed['status'] == 'paused' and completed['result']['status'] == 'paused'
    assert completed['checkpoint_reused_at_stop'] is False
    assert completed['checkpoint_retention_errors']
    assert batch.read(Path(completed['checkpoint']) / 'team.json')['phase'] == 'paused'


@pytest.mark.parametrize('publication', ['pointer', 'attempt'])
def test_checkpoint_publication_failure_preserves_previous_snapshots_and_original_error(managed_team, monkeypatch, publication):
    from saas_bench import team_checkpoint
    team, directory, record, _, _ = managed_team
    first = Path(record['checkpoint'])
    second = batch.save_checkpoint(team, directory, record, 'week_boundary')
    before = {p: tree_hash(p) for p in (first, second)}
    ledger = batch.read(first.parent / '.retention.json')
    module = team_checkpoint if publication == 'pointer' else batch
    write = module.write_json
    failed_path = team.root / 'checkpoint.json' if publication == 'pointer' else directory / 'attempt.json'
    def failed_publish(path, value):
        if Path(path) == failed_path:
            raise OSError('injected publication failure')
        write(path, value)
    monkeypatch.setattr(module, 'write_json', failed_publish)
    with pytest.raises(OSError, match='injected publication failure'):
        batch.save_checkpoint(team, directory, record, 'week_boundary')
    assert {p: tree_hash(p) for p in before} == before
    assert batch.read(first.parent / '.retention.json') == ledger
    assert len(list(first.parent.glob('*/checkpoint.json'))) == 3


@pytest.mark.parametrize('reason,failure', [('paused', 'requested_pause'), ('failed', 'offline business failure')])
def test_paused_and_failed_outcomes_still_freeze_final_state(managed_team, tmp_path, monkeypatch, reason, failure):
    team, directory, record, manifest, entry = managed_team
    previous = record['checkpoint']
    monkeypatch.setattr(team, 'run', lambda **options: type('Outcome', (), dict(day=0, reason=reason))())
    team.failure = failure
    team._set_phase('paused')
    result = batch.execute(team, manifest, entry, directory, record, tmp_path / 'HOLD')
    assert result['status'] == reason and result['checkpoint'] != previous
    assert result['checkpoint_reused_at_stop'] is False
    state = batch.read(Path(result['checkpoint']) / 'team.json')
    assert state['phase'] == 'paused' and state['failure'] == failure
