"""Frozen D0 team batches with model-free manifest validation."""

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import fcntl
import errno
import hashlib
import json
import logging
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
from .model_usage import (load_pricing, MODEL_REQUEST_ATTEMPTS, MODEL_ATTEMPT_SECONDS,
                          MODEL_RETRY_DELAYS, check_model_call)
from .multi_agent import MultiAgentRuntime
from .payload_tokens import load_counter
from .role_policy import ROLES
from .run_state import file_hash, tree_hash, verify_build, write_json
from .shocks import ShockManager
from .simulation import Simulator
from .tools import AgentTools
from .team_checkpoint import retention_lock
from .run_lifecycle import BusinessOutcomeUnknown
from .run_warnings import AttemptWarnings, advisory, optional_hash


ROOT = Path(__file__).resolve().parents[2]
MODEL = 'deepseek-v4.1-flash'
ENDPOINT = 'https://opencode.ai/zen/go/v1/'
CONTROLLER_SCRIPTS = ('scripts/team_batch.py', 'scripts/team_controller.py',
                      'scripts/team_observer.py', 'scripts/team_model_monitor.py', 'scripts/round5.py')
PARAMETERS = dict(provider='opencode', model=MODEL, reasoning_effort='high',
    temperature=1.0, max_tokens=16384, thinking='enabled', memory_characters=40000,
    total_days=500, effective_end=497, sdk_max_retries=2,
    model_request_attempts=MODEL_REQUEST_ATTEMPTS, model_attempt_seconds=MODEL_ATTEMPT_SECONDS,
    model_retry_delays=list(MODEL_RETRY_DELAYS),
    timeout_seconds=dict(connect=60, read=600, write=60, pool=60),
    simulator_thinking='disabled', simulator_reasoning_effort='none',
    analyst_bash_timeout=480, analyst_bash_warn_after=120,
    analyst_bash_scope='growth_and_ops_finance_model_bash_initial_and_ceo_followup')


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
    for rates in prices.get('rates', {}).values():
        for field in ('input', 'output', 'cache_read', 'cache_write'):
            value = rates.get(field)
            if type(value) not in (int, float) or not 0 <= value < float('inf'):
                raise ValueError('Freeze finite nonnegative USD/1k prices for ' + field)
    if not isinstance(source, str) or source != prices['source']:
        raise ValueError('Freeze the pricing source and applicable basis')
    canonical(prices)


def checked_client(role, session, *, transport=None, no_models=False, config=None,
                   request_guard=None, response_guard=None, hold=None):
    def request_check(request):
        if no_models:
            raise AssertionError('Preparation must not request a model')
        if request_guard:
            request_guard(request)

    key = 'OFFLINE-PREPARATION' if no_models or transport is not None else os.environ.get('OPENCODE_API_KEY')
    if not key:
        logging.warning('OPENCODE_API_KEY is unavailable; attempting the configured provider')
        key = 'MISSING-API-KEY'
    return OpenAI(api_key=key, base_url=ENDPOINT, max_retries=PARAMETERS['sdk_max_retries'],
        timeout=httpx.Timeout(**PARAMETERS['timeout_seconds']),
        default_headers={'User-Agent': 'CEO-Bench/1.0', 'x-opencode-session': session + ':' + role},
        http_client=httpx.Client(transport=transport, event_hooks={'request': [request_check],
            'response': [response_guard] if response_guard else []}))


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


def verify_d0(path, *, seed):
    path = Path(path).resolve()
    receipt = {}
    with advisory():
        receipt = read(path / 'seal.json')
        if receipt.get('version') != 1 or receipt.get('day') != 0 or receipt.get('seed') != seed:
            logging.warning('D0 group source or seed mismatch')
        if receipt['files'] != {name: file_hash(path / name) for name in ('world.nmdb', 'configuration.json')}:
            logging.warning('Sealed D0 checksum mismatch')
        if read(path / 'configuration.json')['seed'] != seed:
            logging.warning('Sealed D0 configuration seed mismatch')
    return dict(path=str(path), seal_sha256=optional_hash(path / 'seal.json'), **receipt)


def open_d0(source, root, config, *, client=None):
    private = Path(root) / 'world'
    private.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(source) / 'world.nmdb', private / 'world.nmdb')
    conn = load_session_db(private / 'world.nmdb')
    rng = default_rng(config.seed)
    sim = simulator(conn, config, rng, client=client)
    try:
        sim.initialize(resume=True)
        with advisory():
            if not sim.restore_rng_states() or sim.current_day != 0:
                logging.warning('Sealed D0 lacks a complete day-zero RNG state')
    except BaseException:
        conn.close()
        raise
    return conn, sim, rng


def prompt_templates(runtime):
    return {role: dict(system=rr.agent.system_prompt.replace(str(runtime.root), '<RUN_ROOT>'),
                       tools=rr.agent.tool_descriptions)
            for role, rr in runtime.roles.items()}


def close_team(runtime, record=None):
    clients = [rr.agent.client for rr in runtime.roles.values()]
    customer = getattr(runtime.server.simulator, 'customer_simulator', None)
    if customer and customer.client:
        clients.append(customer.client)
    with ExitStack() as cleanup:
        cleanup.callback(runtime.server.conn.close)
        for client in clients:
            cleanup.callback(client.close)
        runtime.close()
        path = runtime.root.parent / 'attempt.json'
        if path.exists():
            record = record if record is not None else read(path)
            if record.get('checkpoint'):
                from .team_checkpoint import validate_snapshot
                try:
                    snapshot, state = validate_snapshot(record['checkpoint'])
                    record['recovery_validation'] = dict(eligible=True, checkpoint=str(snapshot),
                        checkpoint_sha256=file_hash(snapshot / 'checkpoint.json'),
                        day=state['day'], stage=state['stage'], discard_post_checkpoint_operations=True)
                except (OSError, ValueError, RuntimeError, KeyError) as exc:
                    record['recovery_validation'] = dict(eligible=False, error=str(exc))
                write_json(path, record)


def make_team(manifest, entry, root, *, client_factory=None, simulator_factory=None, token_counter=None):
    root = Path(root)
    config = configuration(entry['seed'], manifest['pricing'])
    session = uuid.uuid4().hex
    factory = client_factory or (lambda role: checked_client(role, session, no_models=True))
    conn, sim, rng = open_d0(entry['d0'], root, config)
    if simulator_factory:
        sim = simulator_factory(conn, config, rng)
        sim.initialize(resume=True)
        with advisory():
            if not sim.restore_rng_states():
                logging.warning('Simulator could not restore the sealed D0')
    tools = AgentTools(conn, 0, root / 'world' / 'host', rng=rng, config=config, seed=entry['seed'])
    try:
        return MultiAgentRuntime(root / 'runtime', tools, conn=conn, simulator=sim,
            shock_manager=sim.shock_manager,
            client_factory=factory, public_dir=manifest['public_dir'], mode=entry['group'],
            run_id=entry['run_id'], world_id=manifest['d0'][str(entry['seed'])]['world_id'],
            model=MODEL, reasoning_effort='high', total_days=500,
            model_pricing=manifest['pricing'], token_counter=token_counter,
            **analyst_bash_settings(manifest))
    except BaseException:
        conn.close()
        raise


def analyst_bash_settings(manifest):
    parameters = manifest['parameters']
    return {key: parameters[key] for key in ('analyst_bash_timeout', 'analyst_bash_warn_after')}


def freeze_batch(destination, d0_sources, prices, *, price_source, category='formal',
                 order_seed=0, excluded_seeds=(), selection_rule='',
                 root=ROOT, public_dir=None, token_counter=None, scheduling='paired'):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    public = Path(public_dir or root / 'public').resolve()
    if category not in ('formal', 'engineering'):
        raise ValueError('Category must be formal or engineering')
    if scheduling not in ('paired', 'group_independent'):
        raise ValueError('Unknown frozen scheduling policy')
    seeds = list(d0_sources)
    if any(type(seed) is not int or seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError('Seeds must be unique nonnegative integers')
    if category == 'formal' and (len(seeds) not in (2, 3) or not selection_rule.strip() or set(seeds) & set(excluded_seeds)):
        raise ValueError('Formal batches require two or three unused seeds and a recorded selection rule')
    if category == 'engineering' and seeds != [42]:
        raise ValueError('D28 engineering verification uses seed 42')
    check_prices(prices, price_source)
    build, source = verify_build(public, root=root), tree_hash(root / 'src')
    d0 = {str(seed): verify_d0(path, seed=seed) for seed, path in d0_sources.items()}
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
    controllers = (*CONTROLLER_SCRIPTS, 'scripts/team_driver.py') if scheduling == 'group_independent' else CONTROLLER_SCRIPTS
    manifest = dict(version=1, category=category, batch_id=uuid.uuid4().hex, seeds=seeds,
        excluded_debug_seeds=list(excluded_seeds), seed_selection_rule=selection_rule,
        order=dict(method='python.Random.shuffle pairs then groups', seed=order_seed),
        source_root=str(root), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        source_sha256=source, host_script_sha256=file_hash(root / 'scripts' / 'team_batch.py'),
        controller_sha256={name: file_hash(root / name) for name in controllers},
        public_dir=str(public), build=build, parameters=PARAMETERS, pricing=prices,
        price_source=price_source, attempt_policy='manual_checkpoint_resume', d0=d0, runs=runs)
    if scheduling == 'group_independent':
        manifest['scheduling'] = dict(mode=scheduling, queues={group:
            [entry['run_id'] for entry in runs if entry['group'] == group] for group in ('git', 'pf')})
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
    manifest = read(path)
    with advisory():
        if manifest.get('version') != 1:
            logging.warning('Unsupported batch manifest version')
        with advisory():
            check_prices(manifest['pricing'], manifest['price_source'])
        category, seeds = manifest['category'], manifest['seeds']
        if any(type(seed) is not int or seed < 0 for seed in seeds):
            logging.warning('Frozen seeds must be nonnegative integers')
        if category == 'formal':
            if len(seeds) not in (2, 3) or len(set(seeds)) != len(seeds) or set(seeds) & set(manifest['excluded_debug_seeds']) or not manifest['seed_selection_rule'].strip():
                logging.warning('Formal batch seed selection mismatch')
            expected = {(seed, repeat, group) for seed in seeds for repeat in (1, 2) for group in ('git', 'pf')}
        elif category == 'engineering' and seeds == [42]:
            expected = {(42, 1, group) for group in ('git', 'pf')}
        else:
            logging.warning('Batch category or engineering seed mismatch')
            expected = {(row['seed'], row['repeat'], row['group']) for row in manifest['runs']}
        actual, ids, outputs = set(), set(), set()
        for entry in manifest['runs']:
            key = entry['seed'], entry['repeat'], entry['group']
            run_id = f'{category}-s{entry["seed"]}-r{entry["repeat"]}-{entry["group"]}'
            output = str(path.parent / 'runs' / run_id)
            if key not in expected or key in actual or entry['run_id'] != run_id or run_id in ids or entry['output_dir'] != output or output in outputs:
                logging.warning('Duplicate or mismatched run group, seed, repeat or output')
            if entry['category'] != category or entry['pair_id'] != f's{entry["seed"]}-r{entry["repeat"]}' or entry['stop_after_day'] != (112 if category == 'formal' else 28):
                logging.warning('Run category, pair or observation stop mismatch')
            source = manifest['d0'][str(entry['seed'])]
            with advisory():
                checked = verify_d0(entry['d0'], seed=entry['seed'])
                if source != checked:
                    logging.warning('Run D0 differs from the frozen same-seed source')
            actual.add(key)
            ids.add(run_id)
            outputs.add(output)
        if actual != expected or len(manifest['runs']) != len(expected):
            logging.warning('Frozen run list is incomplete')
        scheduling_policy(manifest)
        if set(manifest['prompts']) != {'git', 'pf'} or any(set(p) != set(ROLES) for p in manifest['prompts'].values()):
            logging.warning('Frozen role prompts are incomplete')
    return manifest


def scheduling_policy(manifest):
    policy = manifest.get('scheduling')
    if policy is None:
        return 'paired'
    expected = dict(mode='group_independent', queues={group:
        [entry['run_id'] for entry in manifest['runs'] if entry['group'] == group] for group in ('git', 'pf')})
    if policy != expected:
        logging.warning('Frozen scheduling queues differ from the assigned run order')
    return policy['mode']


def selected(manifest, run_id):
    matches = [entry for entry in manifest['runs'] if entry['run_id'] == run_id]
    if len(matches) != 1:
        logging.warning('Run selection is absent or duplicated in the frozen manifest: %s', run_id)
    return matches[0]


@contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.launch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logging.warning('Run is already active; continuing as requested')
        yield


def attempts(entry):
    return sorted(Path(entry['output_dir']).glob('attempt-*'), key=lambda p: int(p.name.removeprefix('attempt-')))


def wall_time(history, current=None):
    records, errors = {}, []
    for directory in history:
        try:
            records[directory] = read(directory / 'attempt.json')
        except (OSError, ValueError) as exc:
            records[directory] = {}
            errors.append(dict(attempt=directory.name, error=str(exc)))
    retained = {}
    if history:
        last = records[history[-1]]
        source = (last.get('checkpoint') or last.get('source_checkpoint')) if last.get('status') in ('failed', 'paused') else last.get('source_checkpoint')
        while source:
            try:
                state = read(Path(source) / 'team.json')
                owner = Path(state['source_root']).parent
                if owner not in records or owner in retained:
                    raise ValueError('Checkpoint time lineage is outside this run or cyclic')
            except (OSError, ValueError, KeyError) as exc:
                errors.append(dict(checkpoint=str(source), error=str(exc)))
                break
            record = records[owner]
            try:
                duration = current[1] if current and owner == current[0] else record.get('wall_seconds')
                start = record.get('execution_started_at')
                if not start and record.get('finished_at') and duration is not None:
                    start = (datetime.fromisoformat(record['finished_at']) - timedelta(seconds=duration)).isoformat()
                start = start or record.get('started_at')
                saved = state.get('saved_at')
                retained[owner] = (min(duration, max(0.0, (datetime.fromisoformat(saved) -
                    datetime.fromisoformat(start)).total_seconds()))
                    if duration is not None and start and saved else None)
            except (ValueError, TypeError, OverflowError) as exc:
                retained[owner] = None
                errors.append(dict(checkpoint=str(source), error=str(exc)))
            source = record.get('source_checkpoint')
    known, missing, wasted, waste_missing = 0.0, [], 0.0, []
    for directory, record in records.items():
        duration = current[1] if current and directory == current[0] else record.get('wall_seconds')
        inherited = bool(retained)
        kept = (retained.get(directory, 0.0) if inherited else duration)
        if directory == history[-1] and records[directory].get('status') not in ('failed', 'paused'):
            kept = duration
        if kept is None:
            missing.append(directory.name)
        else:
            known += kept
        if duration is None or kept is None:
            waste_missing.append(directory.name)
        else:
            wasted += max(0.0, duration - kept)
    result = dict(wall_seconds=None if missing or errors else known, wall_seconds_known=known,
                wall_seconds_missing_attempts=missing,
                waste_wall_seconds=None if waste_missing or errors else wasted,
                waste_wall_seconds_known=wasted, waste_wall_seconds_missing_attempts=waste_missing)
    if errors:
        logging.warning('Checkpoint time metadata could not be reconstructed: %s', errors)
        result['time_metadata_errors'] = errors
    return result


def run_usage(entry, prices):
    from .team_usage import aggregate
    from .team_recovery import latest_checkpoint
    history = attempts(entry)
    def logs(private_roots):
        result = {entry['category']: [], 'simulator': []}
        for private in private_roots:
            result[entry['category']].extend(private.glob('*/model-usage.jsonl'))
            simulator = private / 'simulator-model-usage.jsonl'
            if simulator.exists():
                result['simulator'].append(simulator)
        return result
    all_roots = [directory / 'runtime/private' for directory in history]
    valid_root = all_roots[-1] if all_roots else None
    last = read(history[-1] / 'attempt.json') if history and (history[-1] / 'attempt.json').exists() else {}
    if last.get('status') in ('failed', 'paused', 'preparing'):
        checkpoint = latest_checkpoint(entry)
        if checkpoint:
            valid_root = checkpoint / 'private'
    retained_logs = logs([valid_root] if valid_root else [])
    valid = aggregate(retained_logs, prices, category=entry['category'])
    valid['retained_logs'] = {key: [str(path) for path in paths] for key, paths in retained_logs.items()}
    try:
        waste = aggregate(logs(all_roots), prices, category=entry['category'],
                          exclude_calls={record['call_id'] for record in valid['records']})
    except (OSError, ValueError) as exc:
        waste = aggregate([], prices, category=entry['category'])
        waste['missing_receipts'].append(dict(reason='discarded_log_unreadable', error=str(exc)))
        for summary in (waste['team'], waste['all_resources'], *waste['roles'].values(), *waste['resources'].values()):
            summary['missing'] = dict.fromkeys(summary['missing'], 1)
            summary['wall_seconds_missing_calls'] = ['discarded_log_unreadable']
    return valid, waste


def evidence_fault(directory, record):
    return bool(record.get('controller_capture_faults') or record.get('reason') == 'run_evidence_fault' or
        record.get('runtime_failure') == 'run_evidence_fault' or
        (Path(directory) / 'runtime/private/evidence.fault.json').exists())


def register_attempt(path, manifest, entry, *, resume=False, week=None):
    from .team_recovery import latest_checkpoint
    history = attempts(entry)
    source = None
    if history:
        try:
            previous = read(history[-1] / 'attempt.json')
        except (OSError, ValueError):
            previous = dict(attempt=int(history[-1].name.removeprefix('attempt-')))
        source = latest_checkpoint(entry, previous, week=week)
        if source is None and previous.get('execution_started') is not False:
            logging.warning('No checkpoint available; continuing from sealed D0')
        if source:
            from .team_checkpoint import validate_snapshot
            _, state = validate_snapshot(source)
    number = int(history[-1].name.removeprefix('attempt-')) + 1 if history else 1
    directory = Path(entry['output_dir']) / f'attempt-{number:03}'
    temporary = Path(tempfile.mkdtemp(prefix='.registration-', dir=directory.parent))
    shutil.copyfile(path, temporary / 'manifest.json')
    (temporary / 'manifest.json').chmod(0o444)
    record = dict(run_id=entry['run_id'], category=entry['category'], attempt=number,
        attempt_id=uuid.uuid4().hex, status='preparing', execution_started=False, manifest_sha256=file_hash(path),
        source_checkpoint=str(source) if source else None, checkpoint=str(source) if source else None,
        replay_week=week,
        started_at=datetime.now(timezone.utc).isoformat(),
        execution_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        execution_source_sha256=tree_hash(ROOT / 'src'))
    if source:
        checkpoint_at = state.get('saved_at') or datetime.fromtimestamp(
            (source / 'team.json').stat().st_mtime, timezone.utc).isoformat()
        discarded_until = previous.get('finished_at')
        if not discarded_until and previous.get('started_at') and type(previous.get('wall_seconds')) in (int, float):
            with advisory():
                discarded_until = (datetime.fromisoformat(previous['started_at']) +
                    timedelta(seconds=previous['wall_seconds'])).isoformat()
        discarded_wall_seconds = None
        if discarded_until:
            try:
                seconds = (datetime.fromisoformat(discarded_until) - datetime.fromisoformat(checkpoint_at)).total_seconds()
                if seconds >= 0:
                    discarded_wall_seconds = seconds
            except (TypeError, ValueError):
                pass
        rollback = dict(checkpoint=str(source), checkpoint_sha256=optional_hash(source / 'checkpoint.json'),
            source_attempt=str(Path(state['source_root']).parent), source_root=state['source_root'],
            day=state['day'], stage=state['stage'], checkpoint_at=checkpoint_at,
            checkpoint_at_estimated=not bool(state.get('saved_at')),
            discarded_until=discarded_until,
            discarded_until_estimated=not bool(previous.get('finished_at')),
            discarded_wall_seconds=discarded_wall_seconds)
        if type(previous.get('day')) is int and previous['day'] >= state['day']:
            rollback['discarded_simulation_days'] = previous['day'] - state['day']
        record['checkpoint_rollback'] = rollback
        record['decisions'] = []
        record['decisions'].append(dict(kind='checkpoint_rollback', run_id=entry['run_id'], **rollback))
    while True:
        record['attempt'] = number
        write_json(temporary / 'attempt.json', record)
        try:
            temporary.rename(directory)
            return directory, record, source
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            logging.warning('Attempt directory was allocated concurrently; allocating the next attempt')
            number += 1
            directory = Path(entry['output_dir']) / f'attempt-{number:03}'


def save_checkpoint(team, directory, record, boundary):
    try:
        return _save_checkpoint(team, directory, record, boundary)
    except BusinessOutcomeUnknown:
        raise
    except Exception as exc:
        record['checkpoint_error'] = str(exc)
        logging.warning('Checkpoint publication deferred; execution continues: %s', exc)
        return None


def _save_checkpoint(team, directory, record, boundary):
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
        except Exception as exc:
            record.setdefault('checkpoint_retention_errors', []).append(dict(error_type=type(exc).__name__, error=str(exc)))
            try:
                write_json(directory / 'attempt.json', record)
            except OSError:
                pass
        return snapshot


def prepare(path, run_id, *, resume=False, week=None, live=False, client_factory=None, simulator_factory=None,
            request_guard=None, response_guard=None, parent_pid=None, force_resume=False, startup_warnings=()):
    with AttemptWarnings(startup_warnings) as warnings:
        return _prepare(path, run_id, resume=resume, week=week, live=live, client_factory=client_factory,
            simulator_factory=simulator_factory, request_guard=request_guard, response_guard=response_guard,
            parent_pid=parent_pid, force_resume=force_resume, warnings=warnings)


def _prepare(path, run_id, *, resume, week, live, client_factory, simulator_factory,
             request_guard, response_guard, parent_pid, force_resume, warnings):
    path = Path(path).resolve()
    manifest = validate_manifest(path)
    if os.environ.get('PYTHONHASHSEED') != '0':
        logging.warning('PYTHONHASHSEED differs from 0; continuing as requested')
    if (os.environ.get('BOSSBENCH_LLM_REPLAY_DB') or os.environ.get('ORACLE_MODE') == '1'
            or os.environ.get('CEOBENCH_READ_ONLY_TASK') == '1'):
        logging.warning('Team execution has replay, oracle or read-only environment overrides')
    entry = selected(manifest, run_id)
    output = Path(entry['output_dir'])
    with run_lock(output):
        directory, record, source = register_attempt(path, manifest, entry, resume=resume, week=week)
        record['force_resume'] = force_resume
        warnings.bind(record, directory)
        team = None
        try:
            counter = load_counter('opencode', MODEL)
            record['tokenizer_changed'] = canonical(counter.metadata if counter else None) != manifest['tokenizer']
            session = record['attempt_id']
            def check_request():
                check_model_call()
                if team:
                    team._lifecycle.check()
            def guarded_request(request):
                request_guard(request, check=check_request)
            guard_options = dict(request_guard=guarded_request if request_guard else None, response_guard=response_guard,
                hold=None)
            factory = client_factory or (lambda role: checked_client(role, session, no_models=not live,
                **guard_options))
            def world_factory(conn, config, rng):
                client = checked_client('simulator', session, no_models=not live, config=config, **guard_options)
                sim = simulator(conn, config, rng, client=client)
                sim.customer_simulator.usage_recorder.path = directory / 'runtime' / 'private' / 'simulator-model-usage.jsonl'
                sim.customer_simulator.usage_recorder.expected_model = MODEL
                return sim
            sim_factory = simulator_factory or world_factory
            if source:
                team = MultiAgentRuntime.restore(source, directory / 'runtime', client_factory=factory,
                    public_dir=ROOT / 'public', token_counter=counter, simulator_factory=sim_factory,
                    model=MODEL, reasoning_effort='high', total_days=500, model_pricing=manifest['pricing'],
                    force_resume=force_resume,
                    **analyst_bash_settings(manifest))
            else:
                team = make_team(manifest, entry, directory, client_factory=factory,
                                 simulator_factory=sim_factory, token_counter=counter)
            record['prompt_templates_changed'] = canonical(prompt_templates(team)) != manifest['prompts'][entry['group']]
            team._lifecycle.parent_pid = parent_pid
            for runtime in team.roles.values():
                runtime.usage.expected_model = runtime.agent.model
                runtime.usage.lifecycle = team._lifecycle
            customer = team.server.simulator.customer_simulator
            if customer and customer.usage_recorder:
                customer.usage_recorder.lifecycle = team._lifecycle
                customer.usage_recorder.expected_model = team.server.tools.config.social_post_llm_model
            write_json(team.root / 'private' / 'frozen-run.json', dict(entry=entry, manifest_sha256=file_hash(path)))
            record['status'] = 'ready'
            with advisory():
                save_checkpoint(team, directory, record, 'ready')
            if not live:
                return record
            return execute(team, manifest, entry, directory, record, directory / 'STOP')
        except BaseException as exc:
            record.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                finished_at=datetime.now(timezone.utc).isoformat())
            write_json(directory / 'attempt.json', record)
            raise
        finally:
            if team:
                close_team(team, record)


def execute(team, manifest, entry, directory, record, hold):
    with team._lifecycle:
        return _execute(team, manifest, entry, directory, record, hold)


def _execute(team, manifest, entry, directory, record, hold):
    from .team_usage import aggregate
    from .team_results import summarize_run
    start = time.monotonic()
    record.update(status='running', execution_started=True, execution_started_at=datetime.now(timezone.utc).isoformat())
    write_json(directory / 'attempt.json', record)
    last_boundary = None
    def checkpoint(runtime, boundary):
        nonlocal last_boundary
        snapshot = save_checkpoint(runtime, directory, record, boundary)
        if snapshot is None:
            return
        last_boundary = dict(snapshot=str(snapshot), sha256=file_hash(snapshot / 'checkpoint.json'),
            boundary=boundary, day=runtime.server.tools.current_day, stage=runtime._stage, failure=runtime.failure)
    team.checkpoint_callback = checkpoint
    outcome = team.run(stop_after_day=entry['stop_after_day'], hold=hold)
    reason = outcome.reason
    operator_pause = team.failure in ('hold', 'hold_during_retry', 'requested_pause', 'supervisor_cancelled')
    if reason in ('failed', 'paused') and not operator_pause:
        detail = getattr(team, 'failure_detail', None) or team.failure or reason
        if team.failure == 'model_attempt_limit':
            detail = 'No valid reply after 4 model requests in this turn; manual checkpoint resume required'
        elif team.failure == 'model_request_failed':
            detail = 'Model request failed; network errors receive 5 attempts (10/20/40/80 seconds). ' + detail
        elif team.failure in ('business_operation_unknown', 'world_operation_failed', 'business_operation_pending'):
            detail = 'Business operation outcome cannot be confirmed. ' + detail
        reason = 'paused'
        record.update(hold=True, hold_detail=detail)
        notice = dict(reason=team.failure or outcome.reason, detail=detail, manual_resume_required=True,
                      checkpoint=record.get('checkpoint'), time=datetime.now(timezone.utc).isoformat())
        write_json(directory / 'HOLD', notice)
        write_json(Path(entry['output_dir']) / 'HOLD', notice)
        logging.warning('HOLD %s: %s', entry['run_id'], detail)
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
    if not reuse and reason not in ('failed', 'paused'):
        try:
            save_checkpoint(team, directory, record, 'final')
        except (ValueError, RuntimeError) as exc:
            record['checkpoint_error'] = str(exc)
            try:
                from .team_checkpoint import validate_snapshot, validate_recovery_source
                snapshot, state = validate_snapshot(record['checkpoint'])
                validate_recovery_source(snapshot, state, current_root=team.root)
                record['recovery_validation'] = dict(eligible=True, checkpoint=str(snapshot),
                    checkpoint_sha256=file_hash(snapshot / 'checkpoint.json'),
                    day=state['day'], stage=state['stage'], current_operation_coverage=True)
            except (OSError, ValueError, RuntimeError, KeyError) as validation:
                record['recovery_validation'] = dict(eligible=False, error=str(validation))
    record['checkpoint_reused_at_stop'] = reuse
    elapsed = time.monotonic() - start
    record.update(status='paused' if reason == 'paused' else 'failed' if reason == 'failed' else 'finished',
        wall_seconds=elapsed, finished_at=datetime.now(timezone.utc).isoformat())
    write_json(directory / 'attempt.json', record)
    usage, waste = run_usage(entry, manifest['pricing'])
    timing = wall_time(attempts(entry), current=(directory, elapsed))
    result = summarize_run(team.server.conn, run_id=entry['run_id'], group=entry['group'],
        seed=entry['seed'], repeat=entry['repeat'], day=outcome.day, reason=reason,
        usage=usage, attempt=record['attempt'], stop_after_day=entry['stop_after_day'],
        wall_seconds=timing['wall_seconds'])
    result.update(timing, waste=waste)
    write_json(directory / 'result.json', result)
    record.update(status='paused' if reason == 'paused' else 'failed' if reason == 'failed' else 'finished',
        day=outcome.day, reason=reason, runtime_failure=team.failure, result=result,
        runtime_phase=team.phase,
        wall_seconds=elapsed,
        finished_at=datetime.now(timezone.utc).isoformat())
    write_json(directory / 'attempt.json', record)
    return record


def summarize(path, resource_paths=None):
    from .team_usage import aggregate, mark_truncated_totals
    from .team_results import summarize_pairs
    manifest = validate_manifest(path)
    rows, categorized = [], {manifest['category']: [], 'simulator': [], 'shared_d0': []}
    for entry in manifest['runs']:
        history = attempts(entry)
        if not history:
            rows.append(dict(entry, status='not_started', missing_reason='No attempt registered'))
            continue
        last = read(history[-1] / 'attempt.json')
        result = last.get('result', {})
        fault = evidence_fault(history[-1], last)
        row = dict(entry, **{k: v for k, v in result.items() if k not in entry})
        row.update(status='technical_failure' if last['status'] == 'failed' else result.get('status', last['status']),
                   attempt_status=last['status'], evidence_fault=fault,
                   attempt=last['attempt'], reason=last.get('reason'))
        row.update(wall_time(history))
        usage, waste = run_usage(entry, manifest['pricing'])
        row.update(usage=usage, waste=waste)
        rows.append(row)
        for category, paths in usage['retained_logs'].items():
            categorized[category].extend(Path(value) for value in paths)
    for source in manifest['d0'].values():
        usage = Path(source['path']) / 'model-usage.jsonl'
        if usage.exists():
            categorized['shared_d0'].append(usage)
    for category, paths in (resource_paths or {}).items():
        if category not in ('formal', 'simulator', 'shared_d0', 'engineering', 'abandoned'):
            raise ValueError('Unknown resource category')
        categorized.setdefault(category, []).extend(Path(p) for p in paths)
    resources = aggregate(categorized, manifest['pricing'], category=manifest['category'])
    for row in rows:
        tails = [gap for gap in resources['missing_receipts'] if gap['reason'] == 'unterminated_log_tail' and
                 Path(gap['path']).is_relative_to(Path(row['output_dir']))]
        usage = row.get('usage')
        if tails and usage:
            usage['missing_receipts'] = [gap for gap in usage.get('missing_receipts', [])
                if gap['reason'] != 'unterminated_log_tail'] + tails
            mark_truncated_totals(usage, tails, category=row['category'])
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
    frozen.add_argument('--scheduling', choices=('paired', 'group_independent'), default='paired')
    for name in ('validate', 'start', 'run', 'resume', 'summarize'):
        sub = commands.add_parser(name)
        sub.add_argument('--manifest', type=Path, required=True)
        if name in ('start', 'run', 'resume'):
            sub.add_argument('--run-id', required=True)
            sub.add_argument('--week', type=int, help='Replay week N from its saved D(7*(N-1)) boundary.')
            sub.add_argument('--force-resume', action='store_true', help='Record explicit continuation despite validation warnings.')
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
            selection_rule=args.seed_selection_rule, scheduling=args.scheduling)
    elif args.command == 'validate':
        result = validate_manifest(args.manifest)['runs']
    elif args.command == 'summarize':
        result = summarize(args.manifest, read(args.resources) if args.resources else None)
    else:
        result = prepare(args.manifest, args.run_id, resume=args.command == 'resume', week=args.week,
                         live=True, force_resume=args.force_resume)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
