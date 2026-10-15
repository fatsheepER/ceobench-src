import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

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


@pytest.mark.parametrize('prior', [{}, {'wall_seconds': None}, {'wall_seconds': 0}, {'wall_seconds': 2.5}])
def test_wall_time_keeps_missing_attempts_separate_from_zero(tmp_path, prior):
    first, second = tmp_path / 'attempt-001', tmp_path / 'attempt-002'
    write_json(first / 'attempt.json', prior)
    write_json(second / 'attempt.json', {'wall_seconds': 3.5})
    missing = prior.get('wall_seconds') is None
    known = 3.5 + (prior.get('wall_seconds') or 0)
    assert batch.wall_time([first, second]) == dict(
        wall_seconds=None if missing else known, wall_seconds_known=known,
        wall_seconds_missing_attempts=['attempt-001'] if missing else [],
        waste_wall_seconds=None if missing else 0, waste_wall_seconds_known=0,
        waste_wall_seconds_missing_attempts=['attempt-001'] if missing else [])
    assert batch.wall_time([first, second], current=(second, 0))['wall_seconds_known'] == known - 3.5


def test_live_batch_warns_about_read_only_environment_before_registration(frozen, tmp_path, monkeypatch):
    path = changed_manifest(frozen, tmp_path, lambda manifest: None)
    entry = batch.read(path)['runs'][0]
    monkeypatch.setenv('CEOBENCH_READ_ONLY_TASK', '1')
    record = batch.prepare(path, entry['run_id'], simulator_factory=batch.simulator)
    assert record['status'] == 'ready'
    assert any('read-only' in row['message'] for row in record['validation_warnings'])


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


@pytest.mark.parametrize('fault', ['controller_capture_faults', 'runtime_failure', 'reason', 'fault_file',
    'historical_capture', 'historical_fault_file', 'pending_tail', 'simulator_tail', None])
def test_summarize_cli_preserves_metrics_without_counting_capture_failure_as_complete(frozen, tmp_path, fault):
    _, original, _, _ = frozen
    path = changed_manifest(frozen, tmp_path, lambda manifest: None)
    manifest = batch.read(path)
    pair = [entry for entry in manifest['runs'] if entry['seed'] == original['seeds'][0] and entry['repeat'] == 1]
    originals, results = {}, {}
    for entry in pair:
        directory = Path(entry['output_dir']) / 'attempt-001'
        private = directory / 'runtime/private/ceo'
        private.mkdir(parents=True)
        tokens = 20 if entry['group'] == 'pf' else 10
        response = dict(model=batch.MODEL, usage=dict(prompt_tokens=tokens,
            completion_tokens=tokens // 2, prompt_cache_hit_tokens=0,
            prompt_tokens_details={'cached_tokens': 0, 'cache_write_tokens': 0}))
        logfile = private / 'model-usage.jsonl'
        logfile.write_text('\n'.join(json.dumps(row) for row in [
            dict(event='request', call_id=entry['run_id'], role='ceo', day=112, api='chat', request={'model': batch.MODEL}),
            dict(event='response', call_id=entry['run_id'], role='ceo', api='chat', response=response,
                 error=None, cost_usd=.1)]) + '\n')
        usage = aggregate([logfile], manifest['pricing'], category='formal')
        if fault == 'simulator_tail' and entry['group'] == 'pf':
            simulator_log = private.parent / 'simulator-model-usage.jsonl'
            simulator_log.write_text('\n'.join(json.dumps(dict(row, role='simulator',
                call_id=entry['run_id'] + '-simulator')) for row in
                [json.loads(line) for line in logfile.read_text().splitlines()]) + '\n')
            usage = aggregate(dict(formal=[logfile], simulator=[simulator_log]), manifest['pricing'])
            simulator_log.write_bytes(simulator_log.read_bytes() + b'{"reply":"\xe4\xb8')
            originals[simulator_log] = simulator_log.read_bytes()
        result = dict(status='reached_d112', day=112, cash_at_advance=123456,
            mrr=654321, usage=usage, stop_reason='observation_end')
        failed = entry['group'] == 'pf' and fault != 'simulator_tail'
        record = dict(attempt=1, status='failed' if failed else 'finished', result=result,
            reason='model_attempt_limit' if failed else 'observation_end', wall_seconds=3)
        if failed and fault in ('controller_capture_faults', 'historical_capture'):
            record['controller_capture_faults'] = {str(logfile): {'terminal_fault': 'closed_partial_line'}}
        elif failed and fault in ('runtime_failure', 'reason'):
            record[fault] = 'run_evidence_fault'
        elif failed and fault in ('fault_file', 'historical_fault_file'):
            write_json(private.parent / 'evidence.fault.json', {'reason': 'permanent capture failure'})
        write_json(directory / 'attempt.json', record)
        write_json(directory / 'result.json', result)
        for name in ('attempt.json', 'result.json'):
            originals[directory / name] = (directory / name).read_bytes()
        originals[logfile] = logfile.read_bytes()
        results[entry['group']] = result
        if failed and fault == 'pending_tail':
            with logfile.open('ab') as stream:
                stream.write(b'{"event":"response","reply":"\xe4\xb8')
            originals[logfile] = logfile.read_bytes()
        if failed and fault in ('historical_capture', 'historical_fault_file'):
            resumed = directory.parent / 'attempt-002'
            resumed_private = resumed / 'runtime/private/ceo'
            resumed_private.mkdir(parents=True)
            resumed_log = resumed_private / 'model-usage.jsonl'
            resumed_log.write_text(logfile.read_text().replace(entry['run_id'], entry['run_id'] + '-resumed'))
            result = dict(result, usage=aggregate([resumed_log], manifest['pricing'], category='formal'))
            write_json(resumed / 'attempt.json', dict(attempt=2, status='finished', result=result,
                reason='observation_end', wall_seconds=3))
            write_json(resumed / 'result.json', result)
            for name in ('attempt.json', 'result.json'):
                originals[resumed / name] = (resumed / name).read_bytes()
            results['pf'] = result
    completed = subprocess.run([os.sys.executable, str(batch.ROOT / 'scripts/team_batch.py'),
        'summarize', '--manifest', str(path)], capture_output=True, text=True, check=True,
        env=dict(os.environ, PYTHON_DOTENV_DISABLED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(batch.ROOT / 'src')))
    summary = json.loads(completed.stdout)
    pf = next(row for row in summary['runs'] if row['run_id'] == next(entry['run_id'] for entry in pair if entry['group'] == 'pf'))
    comparison = next(row for row in summary['pairs']['pairs'] if row['seed'] == pair[0]['seed'] and row['repeat'] == 1)
    historical = fault in ('historical_capture', 'historical_fault_file')
    assert pf['status'] == ('reached_d112' if historical or fault == 'simulator_tail' else 'technical_failure')
    assert pf['attempt_status'] == ('finished' if historical or fault == 'simulator_tail' else 'failed')
    assert comparison['complete_d112'] == (historical or fault == 'simulator_tail')
    assert summary['pairs']['complete_d112_pairs'] == int(historical or fault == 'simulator_tail')
    permanent = fault not in (None, 'pending_tail', 'simulator_tail', 'historical_capture', 'historical_fault_file')
    assert pf['evidence_fault'] == permanent
    assert pf['day'] == 112 and pf['cash_at_advance'] == 123456 and pf['mrr'] == 654321
    if fault == 'pending_tail':
        assert pf['usage']['team']['known'] == results['pf']['usage']['team']['known']
        assert all(value is None for value in pf['usage']['team']['complete'].values())
    elif fault == 'simulator_tail':
        assert pf['usage']['team'] == results['pf']['usage']['team']
        assert pf['usage']['roles'] == results['pf']['usage']['roles']
        assert pf['usage']['resources']['formal'] == results['pf']['usage']['resources']['formal']
        assert pf['usage']['resources']['simulator']['known'] == results['pf']['usage']['resources']['simulator']['known']
        assert all(value is None for value in pf['usage']['resources']['simulator']['complete'].values())
        assert all(value is None for value in pf['usage']['all_resources']['complete'].values())
        assert pf['usage']['missing_receipts'][0]['category'] == 'simulator'
    else:
        assert {key: value for key, value in pf['usage'].items() if key != 'retained_logs'} == results['pf']['usage']
        if historical:
            assert pf['waste']['team']['known']['input_tokens'] == 20
    for metric in ('input_tokens', 'output_tokens', 'total_tokens', 'uniform_cost_usd'):
        observed = comparison['comparison'][metric]
        if permanent or fault == 'pending_tail':
            assert observed['pf_minus_git'] is None and observed['pf_over_git'] is None
            assert observed['missing_groups'] == ['pf']
        else:
            expected = results['pf']['usage']['team']['complete'][metric] - results['git']['usage']['team']['complete'][metric]
            assert observed['pf_minus_git'] == pytest.approx(expected) and observed['pf_over_git'] == 2
            assert observed['missing_groups'] == []
    assert all(path.read_bytes() == data for path, data in originals.items())
    if fault in ('pending_tail', 'simulator_tail'):
        assert summary['resources']['team']['known']['input_tokens'] == 30
        assert summary['resources']['team']['complete']['input_tokens'] == (30 if fault == 'simulator_tail' else None)
        if fault == 'simulator_tail':
            assert summary['resources']['roles']['ceo']['complete']['total_tokens'] == 45
            assert summary['resources']['resources']['simulator']['known']['total_tokens'] == 30
            assert all(value is None for value in summary['resources']['resources']['simulator']['complete'].values())
        assert next(gap for gap in summary['resources']['missing_receipts']
            if gap['reason'] == 'unterminated_log_tail')['bytes'] > 0
        assert not list(tmp_path.rglob('evidence.fault.json'))


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


def test_formal_observer_selects_pair_from_real_frozen_manifest(frozen, tmp_path, monkeypatch):
    import importlib.util
    base, manifest, _, _ = frozen
    scripts = batch.ROOT / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('team_observer', scripts / 'team_observer.py')
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    first = manifest['runs'][0]
    runs = [run for run in manifest['runs'] if run['pair_id'] == first['pair_id']]
    path = base / 'formal/manifest.json'
    expected = {key: manifest[key] for key in ('batch_id', 'source_root', 'public_dir', 'source_commit')}
    expected.update(manifest=str(path), manifest_sha256=file_hash(path), host_logs=str(tmp_path / 'host-logs'),
        runs=[{key: run[key] for key in ('run_id', 'output_dir')} for run in runs])
    Path(expected['host_logs']).mkdir()
    selected = observer.identity(path, expected)
    assert selected['runs'] == runs and len(manifest['runs']) == 12
    health = observer.poll(path, tmp_path / 'observer', expected, {},
        quota_fn=lambda: {window: dict(status='ok', percent=1) for window in ('rolling', 'weekly', 'monthly')})
    assert health['status'] == 'healthy'
    assert set(health['progress']) == {run['run_id'] for run in runs}
    assert health['runs'] == expected['runs'] and health['manifest_sha256'] == file_hash(path)


@pytest.mark.parametrize('unknown_write', [False, True])
def test_capture_fault_and_unknown_tail_restore_without_approval(frozen, tmp_path, monkeypatch, unknown_write):
    manifest = batch.canonical(frozen[2])
    for entry in manifest['runs']:
        entry['output_dir'] = str(tmp_path / 'runs' / entry['run_id'])
    path = tmp_path / 'manifest.json'
    write_json(path, manifest)
    run = next(entry for entry in manifest['runs'] if entry['group'] == 'pf')
    first = batch.prepare(path, run['run_id'], simulator_factory=batch.simulator)
    attempt = batch.attempts(run)[0]
    write_json(attempt / 'runtime/private/evidence.fault.json', dict(reason='old capture fault'))
    if unknown_write:
        write_json(attempt / 'runtime/private/operations/unknown.json', dict(kind='api:set_prices', status='running'))
    (path.parent / 'HOLD').touch()
    original = file_hash(attempt / 'attempt.json')
    resumed = batch.prepare(path, run['run_id'], simulator_factory=batch.simulator)
    assert resumed['attempt'] == 2 and resumed['source_checkpoint']
    assert file_hash(attempt / 'attempt.json') == original
    assert not (batch.attempts(run)[-1] / 'runtime/private/evidence.fault.json').exists()
    assert not (batch.attempts(run)[-1] / 'runtime/private/operations/unknown.json').exists()



@pytest.mark.parametrize('group', ['git', 'pf'])
def test_pending_tail_survives_real_prepare_resume_and_next_model_calls(frozen, tmp_path, group):
    manifest = batch.canonical(frozen[2])
    for entry in manifest['runs']:
        entry['output_dir'] = str(tmp_path / 'runs' / entry['run_id'])
    path = tmp_path / 'manifest.json'
    write_json(path, manifest)
    write_json(tmp_path / 'freeze.json', dict(manifest_sha256=file_hash(path)))
    entry = next(entry for entry in manifest['runs'] if entry['group'] == group)
    first = batch.prepare(path, entry['run_id'], simulator_factory=batch.simulator)
    directory = batch.attempts(entry)[0]
    first.update(status='failed', reason='host_interrupted')
    write_json(directory / 'attempt.json', first)
    log = directory / 'runtime/private/ceo/model-usage.jsonl'
    with log.open('ab') as stream:
        stream.write(b'{"event":"response","reply":"\xe4\xb8')
    original = log.read_bytes()
    calls_sent = []
    def client(role):
        def respond(request):
            calls_sent.append(role)
            request_body = json.loads(request.content)
            message = calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final('Existing data checked.')
            return completion(message, request_body.get('stream', False))
        return batch.checked_client(role, 'pending-tail', transport=httpx.MockTransport(respond))
    resumed = batch.prepare(path, entry['run_id'], resume=True, live=True,
        client_factory=client, simulator_factory=batch.simulator)
    assert resumed['status'] == 'finished' and resumed['attempt'] == 2 and resumed['day'] == 28
    assert calls_sent.count('ceo') >= 2
    assert log.read_bytes() == original and not list(directory.rglob('evidence.fault.json'))
    assert resumed['result']['usage']['team']['known']['input_tokens'] > 0
    assert resumed['result']['usage']['team']['complete']['input_tokens'] > 0
    assert any(gap['reason'] == 'unterminated_log_tail' for gap in resumed['result']['waste']['missing_receipts'])
    assert batch.summarize(path)['pairs']['complete_d112_pairs'] == 0


@pytest.mark.parametrize('change', [
    lambda m: m['runs'][0].update(group='prefix'),
    lambda m: m['runs'][0].update(seed=42),
    lambda m: m['runs'].append(copy.deepcopy(m['runs'][0])),
    lambda m: m['runs'][0].update(d0=m['d0']['102']['path'] if m['runs'][0]['seed'] != 102 else m['d0']['101']['path']),
    lambda m: m['runs'][0].update(output_dir=m['runs'][1]['output_dir']),
    lambda m: m['runs'][0].update(stop_after_day=28),
])
def test_validation_warns_about_mismatch_and_duplicate(frozen, tmp_path, change, caplog):
    path = changed_manifest(frozen, tmp_path, change)
    assert batch.validate_manifest(path) == batch.read(path)
    assert caplog.records


def test_manifest_keeps_sealed_d0_while_source_revisions_can_resume(frozen, tmp_path):
    _, _, _, sources = frozen
    path = changed_manifest(frozen, tmp_path, lambda m: m.update(
        source_commit='old', host_script_sha256='old', controller_sha256={}))
    assert batch.validate_manifest(path)['source_commit'] == 'old'
    for source in sources.values():
        assert source.stat().st_mode & 0o222 == 0
        assert (source / 'world.nmdb').stat().st_mode & 0o222 == 0


def test_saved_d0_survives_changed_code_defaults_prices_and_build(frozen, tmp_path, monkeypatch):
    _, _, _, sources = frozen
    hashes = {path: tree_hash(path) for path in sources.values()}
    path = changed_manifest(frozen, tmp_path, lambda manifest: manifest.update(
        build=dict(source_sha256='new'), source_sha256='new', price_source='new'))
    manifest = batch.read(path)
    manifest['pricing']['rates'][batch.MODEL]['input'] = 99
    manifest['pricing']['source'] = 'new'
    write_json(path, manifest)
    def changed_defaults(*args, **kwargs):
        raise AssertionError('Validation must use sealed D0 files, not current code defaults')
    monkeypatch.setattr(batch, 'configuration', changed_defaults)
    assert batch.validate_manifest(path)['pricing']['rates'][batch.MODEL]['input'] == 99
    monkeypatch.setattr(batch, 'MODEL', 'changed-code-default-model')
    assert batch.validate_manifest(path) == manifest
    assert {source: tree_hash(source) for source in hashes} == hashes


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
            for role, rr in team.roles.items():
                assert rr.agent.client.max_retries == 2
                assert rr.agent.model == batch.MODEL and rr.agent.reasoning_effort == 'high'
                assert rr.usage.pricing == PRICES['rates']
                assert (rr.executor.bash_timeout, rr.executor.bash_warn_after) == ((1200, None) if role == 'ceo' else (480, 120))
                script = team.server.role_executors[role]
                assert (script.bash_timeout, script.bash_warn_after) == (1200, None)
                description = next(tool['description'] for tool in rr.agent.tool_descriptions if tool['name'] == 'bash')
                assert ('480 seconds' in description and '120 seconds' in description and '300-second' in description) == (role != 'ceo')
        teams[0].server.conn.execute('UPDATE config_history SET price_A = 17')
        teams[0].server.conn.commit()
        (teams[0].roles['ceo'].workspace / 'MEMORY.md').write_text('ONLY FIRST CLONE')
        assert teams[1].server.conn.execute('SELECT price_A FROM config_history LIMIT 1').fetchone()[0] != 17
        assert teams[1].roles['ceo'].workspace.joinpath('MEMORY.md').read_text() == ''
        assert file_hash(sources[seed] / 'world.nmdb') == before
    finally:
        for team in teams:
            batch.close_team(team)


def test_manual_registration_restores_checkpoint_and_preserves_originals(frozen, tmp_path):
    path = changed_manifest(frozen, tmp_path, lambda m: None)
    manifest = batch.validate_manifest(path)
    entry = manifest['runs'][0]
    with batch.run_lock(Path(entry['output_dir'])):
        first, record, source = batch.register_attempt(path, manifest, entry)
        assert source is None and record['attempt'] == 1
        team = batch.make_team(manifest, entry, first)
        try:
            snapshot = team.checkpoint()
        finally:
            batch.close_team(team)
        record.update(status='running', checkpoint=str(snapshot))
        write_json(first / 'attempt.json', record)
        original = file_hash(first / 'attempt.json')
        (tmp_path / 'HOLD').touch()
        second, later, checkpoint = batch.register_attempt(path, manifest, entry)
        assert later['attempt'] == 2 and checkpoint == snapshot
        assert file_hash(first / 'attempt.json') == original
        assert file_hash(first / 'manifest.json') == file_hash(second / 'manifest.json') == file_hash(path)



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
        for changes in (dict(reasoning_effort='high'), dict(temperature=99), dict(max_tokens=999)):
            client.chat.completions.create(**dict(kwargs, **changes))
        assert requests[-1]['max_tokens'] == 999
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
        expected = dict(connect=60, read=600, write=60, pool=60)
        assert timeouts == [expected]
        assert manifest['parameters']['timeout_seconds'] == expected
    finally:
        client.close()


def test_returned_model_mismatch_keeps_one_raw_receipt_without_retry(tmp_path, monkeypatch):
    from saas_bench.model_usage import ModelUsage
    from saas_bench.run_lifecycle import RunCancelled
    def respond(request):
        response = completion(final('wrong model'))
        body = response.json()
        body['model'] = 'unexpected-model'
        return httpx.Response(200, json=body)
    client = batch.checked_client('simulator', 'offline', transport=httpx.MockTransport(respond))
    recorder = ModelUsage(tmp_path / 'mismatch.jsonl', 'simulator', PRICES['rates'])
    recorder.expected_model = batch.MODEL
    monkeypatch.setattr(recorder.lifecycle, 'sleep', lambda seconds: None)
    recorder.attach(client)
    kwargs = dict(model=batch.MODEL, reasoning_effort='none', max_tokens=100, temperature=0.3,
                  messages=[dict(role='user', content='offline')],
                  extra_body={'thinking': {'type': 'disabled'}})
    try:
        response = recorder.call('chat', kwargs, lambda: client.chat.completions.create(**kwargs), day=0)
        assert response.model == 'unexpected-model'
        rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
        assert rows[-1]['event'] == 'response' and rows[-1]['usage']['input_tokens'] == 20
        assert rows[-1]['error'] is None
        assert next(row for row in rows if row['event'] == 'http_response')['body']
        assert len([row for row in rows if row['event'] == 'http_request']) == 1
        assert rows[-1]['model_attempts'] == 1
    finally:
        client.close()


@pytest.mark.parametrize('group', ['git', 'pf'])
@pytest.mark.parametrize('missing_first_duration', [False, True])
def test_offline_engineering_batch_runs_real_cli_and_pauses_resumes(frozen, tmp_path, group, missing_first_duration, monkeypatch):
    base, _, engineering, _ = frozen
    copied = copy.deepcopy(engineering)
    for entry in copied['runs']:
        entry['output_dir'] = str(tmp_path / 'runs' / entry['run_id'])
    path = tmp_path / 'manifest.json'
    write_json(path, copied)
    write_json(tmp_path / 'freeze.json', dict(manifest_sha256=file_hash(path)))
    run_id = next(entry['run_id'] for entry in copied['runs'] if entry['group'] == group)
    wire = []
    publication_checks = []
    save = batch.save_checkpoint
    def checked_save(team, directory, record, boundary):
        before = {p: tree_hash(p) for p in (team.root / 'checkpoints').glob('*')
                  if p.is_dir() and (p / 'checkpoint.json').exists()}
        snapshot = save(team, directory, record, boundary)
        assert {p: tree_hash(p) for p in before} == before
        publication_checks.append(snapshot)
        return snapshot
    monkeypatch.setattr(batch, 'save_checkpoint', checked_save)
    pause_once = [True]
    def client(role):
        def respond(request):
            body = json.loads(request.content)
            wire.append(dict(role=role, body=body))
            if role == 'ceo' and pause_once[0]:
                pause_once[0] = False
                active = batch.attempts(batch.selected(copied, run_id))[-1]
                (active / 'STOP').touch()
            message = calls(('bash', dict(command=ADVANCE))) if role == 'ceo' else final('Use existing public data.')
            return completion(message, body.get('stream', False))
        return batch.checked_client(role, 'offline', transport=httpx.MockTransport(respond))
    first = batch.prepare(path, run_id, live=True, client_factory=client, simulator_factory=batch.simulator)
    assert first['status'] == 'paused' and first['day'] == 0 and Path(first['checkpoint']).is_dir()
    assert 'checkpoint_error' not in first
    if missing_first_duration:
        first.pop('wall_seconds')
        entry = batch.selected(copied, run_id)
        write_json(batch.attempts(entry)[0] / 'attempt.json', first)
    resumed = batch.prepare(path, run_id, live=True, resume=True, client_factory=client, simulator_factory=batch.simulator)
    assert resumed['status'] == 'finished' and resumed['day'] == 28
    assert resumed['checkpoint_reused_at_stop'] is True and resumed['runtime_phase'] == 'stopped'
    assert resumed['result']['status'] == 'reached_stop_day'
    assert first['manifest_sha256'] == resumed['manifest_sha256'] == file_hash(path)
    assert len(batch.attempts(batch.selected(copied, run_id))) == 2
    assert (resumed['result']['wall_seconds'] is None) == missing_first_duration
    known_seconds = resumed['wall_seconds'] + first.get('wall_seconds', 0)
    assert resumed['wall_seconds'] <= resumed['result']['wall_seconds_known'] <= known_seconds
    if not missing_first_duration:
        assert resumed['result']['waste_wall_seconds_known'] == pytest.approx(
            known_seconds - resumed['result']['wall_seconds_known'])
    assert resumed['result']['wall_seconds_missing_attempts'] == (['attempt-001'] if missing_first_duration else [])
    summarized = next(row for row in batch.summarize(path)['runs'] if row['run_id'] == run_id)
    for key in ('wall_seconds', 'wall_seconds_known', 'wall_seconds_missing_attempts'):
        assert summarized[key] == resumed['result'][key]
    assert all(r['body']['model'] == batch.MODEL and r['body']['reasoning_effort'] == 'high' for r in wire)
    assert all('D497' in r['body']['messages'][0]['content'] and 'D28' not in r['body']['messages'][0]['content'] for r in wire)
    entry = batch.selected(copied, run_id)
    history = batch.attempts(entry)
    snapshot = Path(resumed['checkpoint'])
    snapshots = list(snapshot.parent.glob('*/checkpoint.json'))
    published_here = [p for p in publication_checks if p.parent == snapshot.parent]
    assert len(snapshots) == len(published_here) and len(snapshots) > 2
    states = [batch.read(p.parent / 'team.json') for p in snapshots]
    assert sum(s['day'] == 28 for s in states) == 1
    _, terminal = validate_snapshot(snapshot)
    assert terminal['day'] == 28 and terminal['stage'] == terminal['phase'] == 'boundary'
    messages = [json.loads(line) for line in (snapshot / 'private/messages.jsonl').read_text().splitlines()]
    assert [m['status'] for m in terminal['messages']].count('delivered') == 8
    assert sum(m['status'] == 'delivered' for m in messages) == 8
    logs = [p for attempt in history for p in attempt.glob('runtime/private/*/model-usage.jsonl')]
    retained = resumed['result']['usage']['retained_logs']
    assert aggregate(retained, PRICES, category='engineering') == {
        key: value for key, value in resumed['result']['usage'].items() if key != 'retained_logs'}
    assert resumed['result']['waste']['calls'] == 1
    raw = [json.loads(line) for p in logs for line in p.read_text().splitlines()]
    assert len({row['attempt_id'] for row in raw if row['event'] == 'http_request'}) == len(wire)
    if group == 'pf':
        assert (snapshot / 'private/evidence.sqlite').stat().st_size > 0
        assert terminal['evidence']
    source = Path(first['checkpoint'])
    assert not (source.parent / '.retention-pins.json').exists()
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
    for key in ('team', 'roles', 'resources', 'all_resources'):
        assert terminal_resume['result']['usage'][key] == resumed['result']['usage'][key]
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
def test_manual_resume_after_killed_preparation_uses_d0_and_preserves_failed_attempt(frozen, tmp_path, boundary):
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
    assert batch.read(first / 'attempt.json') == original
    assert len(batch.attempts(entry)) == 2
    assert all(file_hash(directory / 'manifest.json') == file_hash(path) for directory in batch.attempts(entry))


@pytest.mark.parametrize('started', [True, None])
def test_missing_checkpoint_after_execution_warns_and_can_restart_from_d0(frozen, tmp_path, started):
    path = changed_manifest(frozen, tmp_path, lambda manifest: None)
    manifest = batch.validate_manifest(path)
    entry = manifest['runs'][0]
    with batch.run_lock(Path(entry['output_dir'])):
        directory, record, _ = batch.register_attempt(path, manifest, entry)
        record.update(status='failed', execution_started=started)
        batch.write_json(directory / 'attempt.json', record)
        _, resumed, source = batch.register_attempt(path, manifest, entry, resume=True)
        assert source is None and resumed['attempt'] == 2
    assert len(batch.attempts(entry)) == 2


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
    assert row['status'] == 'running'
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
    assert batch.read(first / 'attempt.json')['status'] == 'running'
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


def test_checkpoint_history_keeps_every_boundary_and_ceo_snapshot(managed_team):
    team, directory, record, _, _ = managed_team
    def finish_week():
        team._stage = 'ceo'
        team._set_phase('ceo')
        assert team.server.advance_week()['success']
        team._close_week()
        return batch.save_checkpoint(team, directory, record, 'week_boundary')
    def ceo_checkpoints():
        team._stage = 'ceo'
        team._week = dict(observation='Continue the current week.', ceo_started=False)
        team._set_phase('ceo')
        return [batch.save_checkpoint(team, directory, record, 'answers_ready') for _ in range(3)]
    initial = Path(record['checkpoint'])
    previous = finish_week()
    assert validate_snapshot(previous)[1]['day'] == 7
    middle = ceo_checkpoints()
    assert previous.is_dir(), 'The D7 boundary was deleted by newer CEO handoffs'
    current = finish_week()
    assert validate_snapshot(current)[1]['day'] == 14
    assert previous.is_dir()
    latest = ceo_checkpoints()
    assert current.is_dir(), 'The D14 boundary was deleted by newer CEO handoffs'
    expected = {str(initial), str(previous), str(current), *(str(path) for path in middle + latest)}
    ledger = batch.read(current.parent / '.retention.json')
    assert {item['snapshot'] for item in ledger['snapshots']} == expected
    for snapshot in expected:
        validate_snapshot(snapshot)
    assert all(role.usage.summary['calls'] == 0 for role in team.roles.values())
    assert not record.get('checkpoint_retention_errors')




def test_external_restore_preserves_managed_source_without_pin_writes(managed_team, tmp_path, monkeypatch):
    from saas_bench import team_checkpoint
    team, directory, record, manifest, _ = managed_team
    source = Path(record['checkpoint'])
    before = tree_hash(source)
    validate = team_checkpoint.validate_snapshot
    def readonly_validate(snapshot):
        assert not (source.parent / '.retention-pins.json').exists()
        return validate(snapshot)
    monkeypatch.setattr(team_checkpoint, 'validate_snapshot', readonly_validate)
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
        assert not (source.parent / '.retention-pins.json').exists()
    finally:
        source.parent.chmod(0o700)
        (source.parent / '.retention.lock').chmod(0o600)
    alias = source.parent / ('a' * 32)
    alias.symlink_to(manual, target_is_directory=True)
    for _ in range(4):
        batch.save_checkpoint(team, directory, record, 'week_boundary')
    ledger = batch.read(source.parent / '.retention.json')
    assert len(ledger['snapshots']) == 5 and str(source) in {item['snapshot'] for item in ledger['snapshots']}
    assert tree_hash(source) == before and manual.is_dir() and alias.is_symlink()
    assert not record.get('checkpoint_retention_errors')




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
    assert batch.save_checkpoint(team, directory, record, 'week_boundary') is None
    assert record['checkpoint_error'] == 'injected publication failure'
    monkeypatch.undo()
    assert {p: tree_hash(p) for p in before} == before
    assert batch.read(first.parent / '.retention.json') == ledger
    assert len(list(first.parent.glob('*/checkpoint.json'))) == 3


@pytest.mark.parametrize('reason,failure', [('paused', 'requested_pause'), ('paused', 'sql_evidence_pending'),
                                           ('failed', 'offline business failure')])
def test_paused_and_failed_outcomes_keep_previous_checkpoint(managed_team, tmp_path, monkeypatch, reason, failure):
    team, directory, record, manifest, entry = managed_team
    previous = record['checkpoint']
    monkeypatch.setattr(team, 'run', lambda **options: type('Outcome', (), dict(day=0, reason=reason))())
    team.failure = failure
    team._set_phase('paused')
    result = batch.execute(team, manifest, entry, directory, record, tmp_path / 'HOLD')
    assert result['status'] == 'paused' and result['checkpoint'] == previous
    assert result['checkpoint_reused_at_stop'] is False
    state = batch.read(Path(result['checkpoint']) / 'team.json')
    assert state['stage'] == 'boundary' and state['failure'] is None


@pytest.mark.parametrize('operation', ['covered', 'new_completed', 'unknown'])
def test_pending_sql_pauses_real_team_and_preserves_safe_recovery(frozen, tmp_path, monkeypatch, operation):
    from contextlib import closing
    from saas_bench import api_server
    from test_sql_evidence import settled
    path = changed_manifest(frozen, tmp_path, lambda manifest: None)
    manifest = batch.validate_manifest(path)
    entry = next(row for row in manifest['runs'] if row['group'] == 'pf')
    entered, release = threading.Event(), threading.Event()
    execute = api_server.execute_query
    sent = []

    def delayed_query(*args, **kwargs):
        entered.set()
        assert release.wait(15), 'query release missing'
        return execute(*args, **kwargs)

    def client_factory(role):
        def transport(request):
            sent.append(role)
            body = json.loads(request.content)
            return completion(final('Use existing public data.'), body.get('stream', False))
        return batch.checked_client(role, 'offline', transport=httpx.MockTransport(transport))

    monkeypatch.setattr(api_server, 'execute_query', delayed_query)
    with batch.run_lock(Path(entry['output_dir'])):
        directory, record, _ = batch.register_attempt(path, manifest, entry)
        team = batch.make_team(manifest, entry, directory, client_factory=client_factory)
        try:
            record['status'] = 'ready'
            batch.save_checkpoint(team, directory, record, 'ready')
            previous = record['checkpoint']
            if operation != 'covered':
                write_json(team.root / 'private/operations/post-snapshot.json',
                    dict(status='completed' if operation == 'new_completed' else 'started'))
            team._set_phase('analysis')
            growth = team.roles['growth']
            # Keep both peers pending so a confirmed handoff cannot invalidate the older checkpoint.
            for role in ('growth', 'ops_finance'):
                executor = team.roles[role].executor
                executor.bash_timeout, executor.bash_warn_after = .5, .1
                entered.clear()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    running = pool.submit(executor.execute, 'bash', dict(
                        command='python -c "import novamind_api as nm; nm.query(\'SELECT 1 AS n\')"'))
                    assert entered.wait(5)
                    output = running.result(timeout=5)
                    assert 'timed out after 0.5 seconds' in output
            result = batch.execute(team, manifest, entry, directory, record, tmp_path / 'HOLD')
            assert result['status'] == 'paused'
            assert result['reason'] == 'paused' and result['runtime_failure'] == 'sql_evidence_pending'
            assert batch.read(directory / 'attempt.json')['status'] == 'paused'
            assert result['checkpoint'] == previous
            assert 'checkpoint_error' not in result
            assert result['hold'] and (directory / 'HOLD').exists()
            assert growth.store.fault is None and not growth.store.fault_path.exists()
            assert sent == []
            with closing(growth.store.connect()) as conn:
                rows = [growth.store.read_event(row[0]) for row in conn.execute(
                    'SELECT event_id FROM requests WHERE branch=?', (growth.store.identity['branch_id'],))]
            assert any(row['request']['kind'] == 'sql_query' for row in rows)
            assert not any(row['request']['kind'] == 'model_request' for row in rows)
            release.set()
            settled(team.server)
            for role in ('growth', 'ops_finance'):
                team.roles[role].store.assert_healthy()
            if operation != 'covered':
                _, continued, source = batch.register_attempt(path, manifest, entry, resume=True)
                assert str(source) == previous and continued['attempt'] == 2
        finally:
            release.set()
            batch.close_team(team)
    if operation == 'covered':
        resumed = batch.prepare(path, entry['run_id'], resume=True, client_factory=client_factory)
        assert resumed['status'] == 'ready' and resumed['attempt'] == 2
        assert resumed['source_checkpoint'] == previous
        validate_snapshot(resumed['checkpoint'])


def test_team_creation_uses_frozen_bash_limits(frozen, tmp_path):
    _, original, _, _ = frozen
    manifest = copy.deepcopy(original)
    manifest['parameters'].update(analyst_bash_timeout=7, analyst_bash_warn_after=1)
    entry = manifest['runs'][0]
    team = batch.make_team(manifest, entry, tmp_path / 'custom-frozen-limits')
    try:
        for role in ('growth', 'ops_finance'):
            executor = team.roles[role].executor
            assert (executor.bash_timeout, executor.bash_warn_after) == (7, 1)
    finally:
        batch.close_team(team)
    manifest['parameters']['analyst_bash_scope'] = 'all_bash'
    assert batch.analyst_bash_settings(manifest) == dict(analyst_bash_timeout=7, analyst_bash_warn_after=1)
