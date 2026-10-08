"""Frozen D0 team batches with model-free manifest validation."""

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time
import uuid

import httpx
from numpy.random import default_rng
from openai import OpenAI

from .config import BenchmarkConfig, RUNTIME_CONFIG_FIELDS, SCENARIO_PACKS, ScenarioPack, normalize_runtime_config
from .database import init_database
from .db_protection import load_session_db, save_session_db
from .model_usage import load_pricing
from .multi_agent import MultiAgentRuntime
from .payload_tokens import load_counter
from .role_policy import ROLES
from .run_state import file_hash, tree_hash, verify_build, write_json
from .shocks import ShockManager
from .simulation import Simulator
from .tools import AgentTools
from .team_checkpoint import managed_snapshot, retention_lock


ROOT = Path(__file__).resolve().parents[2]
MODEL = 'deepseek-v4.1-flash'
ENDPOINT = 'https://opencode.ai/zen/go/v1/'
PARAMETERS = dict(provider='opencode', model=MODEL, reasoning_effort='high',
    temperature=1.0, max_tokens=16384, thinking='enabled', memory_characters=40000,
    total_days=500, effective_end=497, sdk_max_retries=2, logical_request_attempts=4,
    timeout_seconds=60, simulator_thinking='disabled', simulator_reasoning_effort='none')


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def configuration(seed, prices):
    config = BenchmarkConfig(seed=seed, total_days=500, initial_cash=1_000_000.,
        agent_llm_model=MODEL, agent_llm_reasoning_effort='high', model_pricing=prices['rates'],
        social_post_llm_provider='opencode', social_post_llm_model=MODEL,
        enterprise_llm_provider='opencode', enterprise_llm_model=MODEL)
    for name, value in normalize_runtime_config(
            {name: getattr(config, name) for name in RUNTIME_CONFIG_FIELDS}).items():
        setattr(config, name, value)
    return config


def check_prices(prices, source):
    if not all(isinstance(prices.get(key), str) and prices[key].strip() for key in ('source', 'basis')):
        raise ValueError('Pricing requires a source and applicable basis')
    rates = prices.get('rates', {}).get(MODEL, {})
    for field in ('input', 'output', 'cache_read', 'cache_write'):
        value = rates.get(field)
        if type(value) not in (int, float) or not 0 <= value < float('inf'):
            raise ValueError('Freeze finite nonnegative USD/1k prices for ' + field)
    if not isinstance(source, str) or source != prices['source']:
        raise ValueError('Freeze the pricing source and applicable basis')
    canonical(prices)


def checked_client(role, session, *, transport=None, no_models=False, config=None):
    def request_check(request):
        if no_models:
            raise AssertionError('Preparation must not request a model')
        body = json.loads(request.content)
        if str(request.url) != ENDPOINT + 'chat/completions' or body.get('model') != MODEL:
            raise ValueError('Frozen model endpoint or requested model changed')
        expected = 'none' if role == 'simulator' else 'high'
        thinking = 'disabled' if role == 'simulator' else 'enabled'
        if body.get('reasoning_effort') != expected or body.get('thinking') != {'type': thinking}:
            raise ValueError('Frozen reasoning or thinking changed')
        if role != 'simulator' and any(body.get(key) != value for key, value in
                dict(temperature=1.0, max_tokens=16384, stream=True, stream_options={'include_usage': True}).items()):
            raise ValueError('Frozen business request parameters changed')
        if role == 'simulator':
            cfg = config or BenchmarkConfig()
            allowed = {(cfg.social_post_llm_max_tokens, cfg.social_media_temperature),
                       (300, cfg.social_media_temperature),
                       (cfg.competitor_post_llm_max_tokens, cfg.social_media_temperature),
                       (cfg.enterprise_llm_max_tokens, cfg.enterprise_llm_temperature),
                       (150, cfg.enterprise_llm_temperature), (100, 0.3), (150, 0.9)}
            if (body.get('max_tokens'), body.get('temperature')) not in allowed:
                raise ValueError('Frozen simulator request parameters changed')

    key = 'OFFLINE-PREPARATION' if no_models or transport is not None else os.environ.get('OPENCODE_API_KEY')
    if not key:
        raise ValueError('OPENCODE_API_KEY is required only for live execution')
    return OpenAI(api_key=key, base_url=ENDPOINT, max_retries=PARAMETERS['sdk_max_retries'], timeout=PARAMETERS['timeout_seconds'],
        default_headers={'User-Agent': 'CEO-Bench/1.0', 'x-opencode-session': session + ':' + role},
        http_client=httpx.Client(transport=transport, event_hooks={'request': [request_check]}))


def simulator(conn, config, rng, *, client=None):
    customer = None
    if client is not None:
        from .customer_llm import CustomerSimulator
        customer = CustomerSimulator(client, conn, config)
    sim = Simulator(conn, config, rng, customer_simulator=customer)
    sim.shock_manager = ShockManager(conn, rng, SCENARIO_PACKS.get('default',
        ScenarioPack(name='Default', description='Balanced scenario')))
    return sim


def seal_d0(seed, destination, prices, *, price_source, root=ROOT, public_dir=None):
    """Seal one model-free initial world, including all engine random streams."""
    if type(seed) is not int or seed < 0:
        raise ValueError('Seed must be a nonnegative integer')
    check_prices(prices, price_source)
    root, destination = Path(root).resolve(), Path(destination).resolve()
    public = Path(public_dir or root / 'public').resolve()
    build = verify_build(public, root=root)
    destination.mkdir(parents=True, exist_ok=False)
    config = configuration(seed, prices)
    conn = init_database(':memory:')
    try:
        sim = simulator(conn, config, default_rng(seed))
        sim.initialize()
        sim.save_rng_states()
        save_session_db(conn, destination / 'world.nmdb')
        write_json(destination / 'configuration.json', canonical(asdict(config)))
        receipt = dict(version=1, seed=seed, day=0, world_id=uuid.uuid4().hex,
            source_sha256=tree_hash(root / 'src'), build=build,
            price_source=price_source, generation='model_free_initialization',
            usage_category='shared_d0', model_calls=0,
            files={name: file_hash(destination / name)
                   for name in ('world.nmdb', 'configuration.json')})
        write_json(destination / 'seal.json', receipt)
        with tempfile.TemporaryDirectory(prefix='verify-d0-') as temporary:
            loaded, restored, _ = open_d0(destination, Path(temporary), config)
            try:
                if restored.current_day != 0:
                    raise ValueError('Sealed source is not D0')
            finally:
                loaded.close()
    finally:
        conn.close()
    for path in [*destination.rglob('*'), destination]:
        path.chmod(path.stat().st_mode & ~0o222)
    return receipt


def verify_d0(path, *, seed, prices, price_source, build, source_sha256):
    path = Path(path).resolve()
    receipt = read(path / 'seal.json')
    if receipt.get('version') != 1 or receipt.get('day') != 0 or receipt.get('seed') != seed:
        raise ValueError('D0 group source or seed mismatch')
    if receipt['files'] != {name: file_hash(path / name) for name in ('world.nmdb', 'configuration.json')}:
        raise ValueError('Sealed D0 checksum mismatch')
    if receipt['build'] != build or receipt['source_sha256'] != source_sha256:
        raise ValueError('D0 source or build mismatch')
    if receipt['price_source'] != price_source or read(path / 'configuration.json') != canonical(asdict(configuration(seed, prices))):
        raise ValueError('D0 configuration or pricing mismatch')
    return dict(path=str(path), seal_sha256=file_hash(path / 'seal.json'), **receipt)


def open_d0(source, root, config, *, client=None):
    private = Path(root) / 'world'
    private.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(source) / 'world.nmdb', private / 'world.nmdb')
    conn = load_session_db(private / 'world.nmdb')
    rng = default_rng(config.seed)
    sim = simulator(conn, config, rng, client=client)
    try:
        sim.initialize(resume=True)
        if not sim.restore_rng_states() or sim.current_day != 0:
            raise ValueError('Sealed D0 lacks a complete day-zero RNG state')
    except BaseException:
        conn.close()
        raise
    return conn, sim, rng


def prompt_templates(runtime):
    return {role: dict(system=rr.agent.system_prompt.replace(str(runtime.root), '<RUN_ROOT>'),
                       tools=rr.agent.tool_descriptions)
            for role, rr in runtime.roles.items()}


def close_team(runtime):
    clients = [rr.agent.client for rr in runtime.roles.values()]
    customer = getattr(runtime.server.simulator, 'customer_simulator', None)
    if customer and customer.client:
        clients.append(customer.client)
    runtime.close()
    for client in clients:
        client.close()
    runtime.server.conn.close()


def make_team(manifest, entry, root, *, client_factory=None, simulator_factory=None, token_counter=None):
    root = Path(root)
    config = configuration(entry['seed'], manifest['pricing'])
    session = uuid.uuid4().hex
    factory = client_factory or (lambda role: checked_client(role, session, no_models=True))
    conn, sim, rng = open_d0(entry['d0'], root, config)
    if simulator_factory:
        sim = simulator_factory(conn, config, rng)
        sim.initialize(resume=True)
        if not sim.restore_rng_states():
            conn.close()
            raise ValueError('Simulator could not restore the sealed D0')
    tools = AgentTools(conn, 0, root / 'world' / 'host', rng=rng, config=config, seed=entry['seed'])
    try:
        return MultiAgentRuntime(root / 'runtime', tools, conn=conn, simulator=sim,
            shock_manager=sim.shock_manager,
            client_factory=factory, public_dir=manifest['public_dir'], mode=entry['group'],
            run_id=entry['run_id'], world_id=manifest['d0'][str(entry['seed'])]['world_id'],
            model=MODEL, reasoning_effort='high', total_days=500,
            model_pricing=manifest['pricing'], token_counter=token_counter)
    except BaseException:
        conn.close()
        raise


def freeze_batch(destination, d0_sources, prices, *, price_source, category='formal',
                 order_seed=0, excluded_seeds=(), selection_rule='', max_attempts=3,
                 root=ROOT, public_dir=None, token_counter=None):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    public = Path(public_dir or root / 'public').resolve()
    if category not in ('formal', 'engineering'):
        raise ValueError('Category must be formal or engineering')
    seeds = list(d0_sources)
    if any(type(seed) is not int or seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError('Seeds must be unique nonnegative integers')
    if category == 'formal' and (len(seeds) != 3 or not selection_rule.strip() or set(seeds) & set(excluded_seeds)):
        raise ValueError('Formal batches require three unused seeds and a recorded selection rule')
    if category == 'engineering' and seeds != [42]:
        raise ValueError('D28 engineering verification uses seed 42')
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError('Freeze a positive finite attempt limit')
    check_prices(prices, price_source)
    build, source = verify_build(public, root=root), tree_hash(root / 'src')
    d0 = {str(seed): verify_d0(path, seed=seed, prices=prices, price_source=price_source,
                             build=build, source_sha256=source) for seed, path in d0_sources.items()}
    destination.mkdir(parents=True, exist_ok=False)
    pairs = [(seed, repeat) for seed in seeds for repeat in range(1, 3 if category == 'formal' else 2)]
    ordering = random.Random(order_seed)
    ordering.shuffle(pairs)
    runs = []
    for seed, repeat in pairs:
        groups = ['git', 'pf']
        ordering.shuffle(groups)
        for group in groups:
            run_id = f'{category}-s{seed}-r{repeat}-{group}'
            runs.append(dict(run_id=run_id, pair_id=f's{seed}-r{repeat}', seed=seed, repeat=repeat,
                group=group, category=category, d0=d0[str(seed)]['path'],
                output_dir=str(destination / 'runs' / run_id), stop_after_day=112 if category == 'formal' else 28))
    manifest = dict(version=1, category=category, batch_id=uuid.uuid4().hex, seeds=seeds,
        excluded_debug_seeds=list(excluded_seeds), seed_selection_rule=selection_rule,
        order=dict(method='python.Random.shuffle pairs then groups', seed=order_seed),
        source_root=str(root), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        source_sha256=source, host_script_sha256=file_hash(root / 'scripts' / 'team_batch.py'),
        public_dir=str(public), build=build, parameters=PARAMETERS, pricing=prices,
        price_source=price_source, max_attempts=max_attempts, d0=d0, runs=runs)
    counter = token_counter or load_counter('opencode', MODEL)
    manifest['tokenizer'] = counter.metadata if counter else None
    prompts = {}
    for group in ('git', 'pf'):
        entry = next(run for run in runs if run['group'] == group)
        with tempfile.TemporaryDirectory(prefix='team-prompts-') as temporary:
            team = make_team(manifest, entry, Path(temporary), token_counter=counter)
            try:
                prompts[group] = prompt_templates(team)
            finally:
                close_team(team)
    manifest['prompts'] = canonical(prompts)
    write_json(destination / 'manifest.json', manifest)
    write_json(destination / 'freeze.json', dict(manifest_sha256=file_hash(destination / 'manifest.json')))
    for name in ('manifest.json', 'freeze.json'):
        (destination / name).chmod(0o444)
    validate_manifest(destination / 'manifest.json')
    return manifest


def validate_manifest(path):
    path = Path(path).resolve()
    frozen = read(path.parent / 'freeze.json')
    if file_hash(path) != frozen['manifest_sha256']:
        raise ValueError('Frozen manifest changed')
    manifest = read(path)
    if manifest.get('version') != 1 or manifest.get('parameters') != PARAMETERS:
        raise ValueError('Frozen model parameters or manifest version mismatch')
    if type(manifest.get('max_attempts')) is not int or manifest['max_attempts'] < 1:
        raise ValueError('Frozen finite attempt limit is invalid')
    root, public = Path(manifest['source_root']), Path(manifest['public_dir'])
    if verify_build(public, root=root) != manifest['build'] or tree_hash(root / 'src') != manifest['source_sha256']:
        raise ValueError('Frozen source or public build changed')
    if file_hash(root / 'scripts' / 'team_batch.py') != manifest['host_script_sha256']:
        raise ValueError('Frozen batch entry changed')
    check_prices(manifest['pricing'], manifest['price_source'])
    category, seeds = manifest['category'], manifest['seeds']
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('Frozen seeds must be nonnegative integers')
    if category == 'formal':
        if len(seeds) != 3 or len(set(seeds)) != 3 or set(seeds) & set(manifest['excluded_debug_seeds']) or not manifest['seed_selection_rule'].strip():
            raise ValueError('Formal batch seed selection mismatch')
        expected = {(seed, repeat, group) for seed in seeds for repeat in (1, 2) for group in ('git', 'pf')}
    elif category == 'engineering' and seeds == [42]:
        expected = {(42, 1, group) for group in ('git', 'pf')}
    else:
        raise ValueError('Batch category or engineering seed mismatch')
    actual, ids, outputs = set(), set(), set()
    for entry in manifest['runs']:
        key = entry['seed'], entry['repeat'], entry['group']
        run_id = f'{category}-s{entry["seed"]}-r{entry["repeat"]}-{entry["group"]}'
        output = str(path.parent / 'runs' / run_id)
        if key not in expected or key in actual or entry['run_id'] != run_id or run_id in ids or entry['output_dir'] != output or output in outputs:
            raise ValueError('Duplicate or mismatched run group, seed, repeat or output')
        if entry['category'] != category or entry['pair_id'] != f's{entry["seed"]}-r{entry["repeat"]}' or entry['stop_after_day'] != (112 if category == 'formal' else 28):
            raise ValueError('Run category, pair or observation stop mismatch')
        source = manifest['d0'][str(entry['seed'])]
        checked = verify_d0(entry['d0'], seed=entry['seed'], prices=manifest['pricing'],
            price_source=manifest['price_source'], build=manifest['build'], source_sha256=manifest['source_sha256'])
        if source != checked:
            raise ValueError('Run D0 differs from the frozen same-seed source')
        actual.add(key)
        ids.add(run_id)
        outputs.add(output)
    if actual != expected or len(manifest['runs']) != len(expected):
        raise ValueError('Frozen run list is incomplete')
    if set(manifest['prompts']) != {'git', 'pf'} or any(set(p) != set(ROLES) for p in manifest['prompts'].values()):
        raise ValueError('Frozen role prompts are incomplete')
    return manifest


def selected(manifest, run_id):
    matches = [entry for entry in manifest['runs'] if entry['run_id'] == run_id]
    if len(matches) != 1:
        raise ValueError('Run is absent from the original frozen manifest')
    return matches[0]


@contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.launch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Run is already active') from exc
        yield


def attempts(entry):
    return sorted(Path(entry['output_dir']).glob('attempt-*'))


def mark_interrupted(directory, record):
    if record['status'] in ('preparing', 'ready', 'running'):
        record.update(status='failed', reason='host_interrupted', error='Previous host stopped before finalizing the attempt')
        write_json(directory / 'attempt.json', record)
    return record


def register_attempt(path, manifest, entry, *, resume=False):
    if (path.parent / 'HOLD').exists():
        raise ValueError('Batch HOLD marker is active')
    history = attempts(entry)
    source = None
    if history:
        previous = read(history[-1] / 'attempt.json')
        if not resume:
            raise ValueError('Run already registered; use explicit resume for paused or failed attempts')
        previous = mark_interrupted(history[-1], previous)
        if previous['status'] not in ('paused', 'failed'):
            raise ValueError('Resume requires a paused or failed attempt with a safe checkpoint')
        if previous['manifest_sha256'] != file_hash(path) or file_hash(history[-1] / 'manifest.json') != file_hash(path):
            raise ValueError('Resume must retain the original immutable manifest')
        if previous.get('checkpoint'):
            source = Path(previous['checkpoint'])
            owner = next((directory for directory in history
                if source.is_relative_to(directory / 'runtime' / 'checkpoints')), None)
            if owner is None:
                raise ValueError('Resume checkpoint belongs to another run')
            if file_hash(owner / 'manifest.json') != file_hash(path):
                raise ValueError('Resume must retain the checkpoint original manifest')
            from .team_checkpoint import validate_recovery_source, validate_snapshot
            _, state = validate_snapshot(source)
            latest_root = history[-1] / 'runtime'
            # Preparation can fail before creating a runtime; no execution started there.
            current_root = latest_root if latest_root.exists() or previous.get('execution_started') is not False else None
            validate_recovery_source(source, state, current_root=current_root)
            if state['roles']['ceo']['identity']['run_id'] != entry['run_id']:
                raise ValueError('Resume checkpoint belongs to another run')
        elif previous.get('execution_started') is not False:
            raise ValueError('Resume requires a safe checkpoint once execution has started')
    elif resume:
        raise ValueError('Resume requires an original attempt')
    number = len(history) + 1
    if number > manifest['max_attempts']:
        raise ValueError('Frozen finite attempt limit reached')
    directory = Path(entry['output_dir']) / f'attempt-{number:03}'
    temporary = Path(tempfile.mkdtemp(prefix='.registration-', dir=directory.parent))
    shutil.copyfile(path, temporary / 'manifest.json')
    (temporary / 'manifest.json').chmod(0o444)
    record = dict(run_id=entry['run_id'], category=entry['category'], attempt=number,
        attempt_id=uuid.uuid4().hex, status='preparing', execution_started=False, manifest_sha256=file_hash(path),
        source_checkpoint=str(source) if source else None, checkpoint=str(source) if source else None,
        started_at=datetime.now(timezone.utc).isoformat())
    write_json(temporary / 'attempt.json', record)
    temporary.rename(directory)
    return directory, record, source


def prune_checkpoints(parent, ledger, directory):
    if (set(ledger) != {'version', 'snapshots'} or ledger['version'] != 1 or
            not isinstance(ledger['snapshots'], list) or not ledger['snapshots']):
        raise ValueError('Invalid checkpoint retention ledger')
    snapshots = {}
    for item in ledger['snapshots']:
        if not isinstance(item, dict) or set(item) != {'snapshot', 'sha256'} or item['snapshot'] in snapshots:
            raise ValueError('Invalid checkpoint retention entry')
        snapshots[item['snapshot']] = managed_snapshot(parent, item['snapshot'], item['sha256'])
    protected = {item['snapshot'] for item in ledger['snapshots'][-2:]}
    pins_path = parent / '.retention-pins.json'
    if pins_path.is_symlink():
        raise ValueError('Checkpoint retention pins cannot be a symlink')
    pins = read(pins_path) if pins_path.exists() else {}
    if not isinstance(pins, dict):
        raise ValueError('Invalid checkpoint retention pins')
    for value, checksum in pins.items():
        managed_snapshot(parent, value, checksum)
        protected.add(value)
    history = sorted(directory.parent.glob('attempt-*'))
    for attempt in history:
        if attempt.is_symlink() or (attempt / 'attempt.json').is_symlink():
            raise ValueError('Checkpoint retention attempt cannot be a symlink')
        saved = read(attempt / 'attempt.json')
        for key in ('checkpoint', 'source_checkpoint'):
            value = saved[key]
            if value is not None:
                if (not isinstance(value, str) or not Path(value).is_absolute() or str(Path(value)) != value or
                        Path(value).parent not in {p / 'runtime' / 'checkpoints' for p in history} or
                        not Path(value).is_dir() or any(p.is_symlink() for p in (Path(value), *Path(value).parents))):
                    raise ValueError('Invalid attempt checkpoint reference')
                protected.add(value)
        pointer = attempt / 'runtime' / 'checkpoint.json'
        if pointer.is_symlink() or any(p.is_symlink() for p in pointer.parents):
            raise ValueError('Checkpoint retention pointer cannot contain symlinks')
        if pointer.exists():
            saved = read(pointer)
            if set(saved) != {'snapshot', 'sha256'}:
                raise ValueError('Invalid runtime checkpoint pointer')
            value = saved['snapshot']
            if not isinstance(value, str) or not Path(value).is_absolute() or str(Path(value)) != value:
                raise ValueError('Invalid runtime checkpoint reference')
            managed_snapshot(attempt / 'runtime' / 'checkpoints', value, saved['sha256'])
            protected.add(value)
    discarded = [value for value in snapshots if value not in protected]
    for value in discarded:
        shutil.rmtree(snapshots[value])
    if discarded:
        ledger['snapshots'] = [item for item in ledger['snapshots'] if item['snapshot'] not in discarded]
        write_json(parent / '.retention.json', ledger)


def save_checkpoint(team, directory, record, boundary):
    parent = team.root / 'checkpoints'
    with ExitStack() as stack:
        lock_error = None
        try:
            created = stack.enter_context(retention_lock(parent, create=True))
        except Exception as exc:
            lock_error = exc
        snapshot = team.checkpoint()
        record.update(checkpoint=str(snapshot), boundary=boundary)
        write_json(directory / 'attempt.json', record)
        try:
            if lock_error:
                raise lock_error
            path = parent / '.retention.json'
            if path.is_symlink():
                raise ValueError('Checkpoint retention ledger cannot be a symlink')
            if path.exists():
                ledger = read(path)
                if (not isinstance(ledger, dict) or set(ledger) != {'version', 'snapshots'} or
                        ledger['version'] != 1 or not isinstance(ledger['snapshots'], list)):
                    raise ValueError('Invalid checkpoint retention ledger')
            elif created:
                ledger = dict(version=1, snapshots=[])
            else:
                raise ValueError('Checkpoint retention ledger is missing')
            ledger['snapshots'].append(dict(snapshot=str(snapshot), sha256=file_hash(snapshot / 'checkpoint.json')))
            write_json(path, ledger)
            prune_checkpoints(parent, ledger, directory)
        except Exception as exc:
            record.setdefault('checkpoint_retention_errors', []).append(dict(error_type=type(exc).__name__, error=str(exc)))
            try:
                write_json(directory / 'attempt.json', record)
            except OSError:
                pass
        return snapshot


def prepare(path, run_id, *, resume=False, live=False, client_factory=None, simulator_factory=None):
    path = Path(path).resolve()
    manifest = validate_manifest(path)
    if live and (os.environ.get('BOSSBENCH_LLM_REPLAY_DB') or os.environ.get('ORACLE_MODE') == '1'
            or os.environ.get('CEOBENCH_READ_ONLY_TASK') == '1'):
        raise ValueError('Frozen team execution cannot enable replay, oracle or read-only overrides')
    entry = selected(manifest, run_id)
    output = Path(entry['output_dir'])
    with run_lock(output):
        directory, record, source = register_attempt(path, manifest, entry, resume=resume)
        team = None
        try:
            counter = load_counter('opencode', MODEL)
            if canonical(counter.metadata if counter else None) != manifest['tokenizer']:
                raise ValueError('Frozen tokenizer changed')
            session = record['attempt_id']
            factory = client_factory or (lambda role: checked_client(role, session, no_models=not live))
            def world_factory(conn, config, rng):
                client = checked_client('simulator', session, no_models=not live, config=config)
                sim = simulator(conn, config, rng, client=client)
                sim.customer_simulator.usage_recorder.path = directory / 'runtime' / 'private' / 'simulator-model-usage.jsonl'
                sim.customer_simulator.usage_recorder.expected_model = MODEL
                return sim
            sim_factory = simulator_factory or world_factory
            if source:
                team = MultiAgentRuntime.restore(source, directory / 'runtime', client_factory=factory,
                    public_dir=manifest['public_dir'], token_counter=counter, simulator_factory=sim_factory,
                    model=MODEL, reasoning_effort='high', total_days=500, model_pricing=manifest['pricing'])
            else:
                team = make_team(manifest, entry, directory, client_factory=factory,
                                 simulator_factory=sim_factory, token_counter=counter)
            if canonical(prompt_templates(team)) != manifest['prompts'][entry['group']]:
                raise ValueError('Frozen role prompts or tools changed')
            for runtime in team.roles.values():
                runtime.usage.expected_model = MODEL
            write_json(team.root / 'private' / 'frozen-run.json', dict(entry=entry, manifest_sha256=file_hash(path)))
            record['status'] = 'ready'
            save_checkpoint(team, directory, record, 'ready')
            if not live:
                return record
            return execute(team, manifest, entry, directory, record, path.parent / 'HOLD')
        except BaseException as exc:
            record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
            write_json(directory / 'attempt.json', record)
            raise
        finally:
            if team:
                close_team(team)


def execute(team, manifest, entry, directory, record, hold):
    from .team_usage import aggregate
    from .team_results import summarize_run
    start = time.monotonic()
    record.update(status='running', execution_started=True)
    write_json(directory / 'attempt.json', record)
    last_boundary = None
    def checkpoint(runtime, boundary):
        nonlocal last_boundary
        snapshot = save_checkpoint(runtime, directory, record, boundary)
        last_boundary = dict(snapshot=str(snapshot), sha256=file_hash(snapshot / 'checkpoint.json'),
            boundary=boundary, day=runtime.server.tools.current_day, stage=runtime._stage, failure=runtime.failure)
    team.checkpoint_callback = checkpoint
    outcome = team.run(stop_after_day=entry['stop_after_day'], hold=hold)
    reason = outcome.reason
    if reason == 'paused' and team.failure not in ('hold', 'hold_during_retry', 'requested_pause', 'supervisor_cancelled'):
        reason = 'failed'
    reuse = (last_boundary is not None and reason == 'observation_end' and
        outcome.day == entry['stop_after_day'] == team.server.tools.current_day and
        last_boundary['boundary'] == 'week_boundary' and last_boundary['day'] == outcome.day and
        last_boundary['stage'] == team._stage == 'boundary' and last_boundary['failure'] is None and team.failure is None and
        record['checkpoint'] == last_boundary['snapshot'])
    if reuse:
        try:
            reuse = (read(team.root / 'checkpoint.json') == {key: last_boundary[key] for key in ('snapshot', 'sha256')} and
                read(directory / 'attempt.json').get('checkpoint') == last_boundary['snapshot'] and
                file_hash(Path(last_boundary['snapshot']) / 'checkpoint.json') == last_boundary['sha256'])
        except (OSError, ValueError):
            reuse = False
    if not reuse:
        try:
            save_checkpoint(team, directory, record, 'final')
        except (ValueError, RuntimeError) as exc:
            record['checkpoint_error'] = str(exc)
    record['checkpoint_reused_at_stop'] = reuse
    logs = {entry['category']: [], 'simulator': []}
    for previous in attempts(entry):
        logs[entry['category']].extend(previous.glob('runtime/private/*/model-usage.jsonl'))
        if (previous / 'runtime/private/simulator-model-usage.jsonl').exists():
            logs['simulator'].append(previous / 'runtime/private/simulator-model-usage.jsonl')
    usage = aggregate(logs, manifest['pricing'], category=entry['category'])
    elapsed = time.monotonic() - start
    total_elapsed = elapsed + sum(read(p / 'attempt.json').get('wall_seconds', 0)
                                 for p in attempts(entry) if p != directory)
    result = summarize_run(team.server.conn, run_id=entry['run_id'], group=entry['group'],
        seed=entry['seed'], repeat=entry['repeat'], day=outcome.day, reason=reason,
        usage=usage, attempt=record['attempt'], stop_after_day=entry['stop_after_day'],
        wall_seconds=total_elapsed)
    write_json(directory / 'result.json', result)
    record.update(status='paused' if reason == 'paused' else 'failed' if reason == 'failed' else 'finished',
        day=outcome.day, reason=reason, runtime_failure=team.failure, result=result,
        runtime_phase=team.phase,
        wall_seconds=elapsed,
        finished_at=datetime.now(timezone.utc).isoformat())
    write_json(directory / 'attempt.json', record)
    return record


def summarize(path, resource_paths=None):
    from .team_usage import aggregate
    from .team_results import summarize_pairs
    manifest = validate_manifest(path)
    rows, categorized = [], {manifest['category']: [], 'simulator': [], 'shared_d0': []}
    for entry in manifest['runs']:
        history = attempts(entry)
        if not history:
            rows.append(dict(entry, status='not_started', missing_reason='No attempt registered'))
            continue
        last = read(history[-1] / 'attempt.json')
        if last['status'] in ('preparing', 'ready', 'running'):
            try:
                with run_lock(Path(entry['output_dir'])):
                    last = mark_interrupted(history[-1], read(history[-1] / 'attempt.json'))
            except ValueError:
                pass
        result = last.get('result', {})
        row = dict(entry, **{k: v for k, v in result.items() if k not in entry})
        row.update(status=result.get('status', 'technical_failure' if last['status'] == 'failed' else last['status']), attempt_status=last['status'],
                   attempt=last['attempt'], reason=last.get('reason'))
        rows.append(row)
        for directory in history:
            categorized[manifest['category']].extend(directory.glob('runtime/private/*/model-usage.jsonl'))
            if (directory / 'runtime/private/simulator-model-usage.jsonl').exists():
                categorized['simulator'].append(directory / 'runtime/private/simulator-model-usage.jsonl')
    for source in manifest['d0'].values():
        usage = Path(source['path']) / 'model-usage.jsonl'
        if usage.exists():
            categorized['shared_d0'].append(usage)
    for category, paths in (resource_paths or {}).items():
        if category not in ('formal', 'simulator', 'shared_d0', 'engineering', 'abandoned'):
            raise ValueError('Unknown resource category')
        categorized.setdefault(category, []).extend(Path(p) for p in paths)
    resources = aggregate(categorized, manifest['pricing'], category=manifest['category'])
    return dict(category=manifest['category'], runs=rows,
                pairs=summarize_pairs(rows), resources=resources)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    seal = commands.add_parser('seal-d0')
    seal.add_argument('--seed', type=int, required=True)
    seal.add_argument('--output', type=Path, required=True)
    seal.add_argument('--pricing', type=Path, required=True)
    seal.add_argument('--price-source', required=True)
    frozen = commands.add_parser('freeze')
    frozen.add_argument('--d0', action='append', required=True, help='SEED=/absolute/sealed/path')
    frozen.add_argument('--output', type=Path, required=True)
    frozen.add_argument('--pricing', type=Path, required=True)
    frozen.add_argument('--price-source', required=True)
    frozen.add_argument('--category', choices=('formal', 'engineering'), default='formal')
    frozen.add_argument('--excluded-debug-seed', action='append', type=int, default=[])
    frozen.add_argument('--seed-selection-rule', default='')
    frozen.add_argument('--order-seed', type=int, default=0)
    frozen.add_argument('--max-attempts', type=int, default=3)
    for name in ('validate', 'start', 'run', 'resume', 'summarize'):
        sub = commands.add_parser(name)
        sub.add_argument('--manifest', type=Path, required=True)
        if name in ('start', 'run', 'resume'):
            sub.add_argument('--run-id', required=True)
        if name == 'summarize':
            sub.add_argument('--resources', type=Path, help='JSON mapping resource categories to receipt log paths')
    args = parser.parse_args(argv)
    if args.command == 'seal-d0':
        result = seal_d0(args.seed, args.output, load_pricing(args.pricing), price_source=args.price_source)
    elif args.command == 'freeze':
        sources = {}
        for item in args.d0:
            seed, value = item.split('=', 1)
            if int(seed) in sources:
                parser.error('Duplicate D0 seed')
            sources[int(seed)] = Path(value)
        result = freeze_batch(args.output, sources, load_pricing(args.pricing), price_source=args.price_source,
            category=args.category, order_seed=args.order_seed, excluded_seeds=args.excluded_debug_seed,
            selection_rule=args.seed_selection_rule, max_attempts=args.max_attempts)
    elif args.command == 'validate':
        result = validate_manifest(args.manifest)['runs']
    elif args.command == 'summarize':
        result = summarize(args.manifest, read(args.resources) if args.resources else None)
    else:
        if os.environ.get('PYTHONHASHSEED') != '0':
            parser.error('Set PYTHONHASHSEED=0 before team execution')
        result = prepare(args.manifest, args.run_id, resume=args.command == 'resume', live=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
