"""Immutable, private snapshots at quiescent team handoffs."""

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import fcntl
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import httpx

from .run_state import artifact_hashes, copy_workspace, file_hash, tree_hash, write_json
from .run_warnings import advisory, optional_hash
from .run_lifecycle import BusinessOutcomeUnknown


@contextmanager
def retention_lock(parent, *, create=False):
    parent = Path(parent).absolute()
    if any(path.is_symlink() for path in (parent, *parent.parents)):
        raise ValueError('Checkpoint retention parent cannot contain symlinks')
    if create:
        parent.mkdir(parents=True, exist_ok=True)
    path = parent / '.retention.lock'
    created = False
    if create:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            created = True
        except FileExistsError:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    else:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield created


def business_operations(root):
    return {path.name: file_hash(path) for path in (Path(root) / 'private' / 'operations').glob('*.json')
        if not _wrapper(json.loads(path.read_text()))}


def _wrapper(receipt):
    kind = receipt.get('kind', '')
    return kind == 'ceo_scripts' or kind.startswith('ceo_tool:')


def recovery_operation_safe(root, receipt, *, allow_workspace_rollback=False):
    proof = receipt.get('recovery', {})
    classifications = ('read_only', 'read_only_interrupted', 'pending_read_only')
    if allow_workspace_rollback:
        classifications += ('unproven',)
    if (not _wrapper(receipt) or receipt.get('wrapper_version') != 1 or
            receipt.get('status') not in ('completed', 'interrupted') or
            proof.get('classification') not in classifications or
            proof.get('client_closed') is not True or proof.get('operation_failed') is not False or
            proof.get('step_day_timed_out') is not False or proof.get('queued_requests') is not False or
            not isinstance(proof.get('business_operations_after'), dict) or
            receipt.get('business_operations_before') != proof['business_operations_after'] or
            not isinstance(receipt.get('mutation_epoch_before'), int) or
            receipt['mutation_epoch_before'] != proof.get('mutation_epoch_after') or
            (not allow_workspace_rollback and
                (not isinstance(receipt.get('workspace_sha256_before'), str) or
                    receipt['workspace_sha256_before'] != proof.get('workspace_sha256_after'))) or
            any(request.get('read_only') is not True for request in proof.get('pending_requests', []))):
        return False
    if proof.get('subprocess') is False:
        return receipt['kind'].startswith('ceo_tool:') and receipt['kind'] != 'ceo_tool:bash'
    boundary = proof.get('process_boundary') or {}
    return (boundary.get('phase') == 'closed' and boundary.get('coverage') == 'subreaper_wait' and
        boundary.get('descendants_closed') is True and boundary.get('children') == [] and
        boundary.get('unknown', 'missing') is None and boundary.get('exit_code') is not None)


def _known_operation(root):
    for path in (Path(root) / 'private' / 'operations').glob('*.json'):
        receipt = json.loads(path.read_text())
        if receipt.get('status') == 'failed' and (
                receipt.get('kind') == 'api:advance' and receipt.get('advance', {}).get('status') == 'failed' or
                receipt.get('recovery', {}).get('classification') == 'known_advance_failure' and
                receipt.get('recovery', {}).get('client_closed') is True and receipt.get('advance_operations')):
            continue
        if receipt.get('status') != 'completed' and not recovery_operation_safe(root, receipt):
            raise BusinessOutcomeUnknown('Business operation outcome unknown; HOLD for manual checkpoint resume')


def validate_recovery_source(snapshot, state, *, current_root=None):
    """The complete snapshot replaces all later operation outcomes and evidence."""
    return dict(checkpoint=str(Path(snapshot)), day=state['day'],
        discard_post_checkpoint_operations=True)


def _validate_restored_roles(snapshot, root):
    roles, frozen_roles = Path(root) / 'roles', Path(snapshot) / 'roles'
    if not roles.is_dir() or roles.is_symlink() or not frozen_roles.is_dir() or frozen_roles.is_symlink():
        logging.warning('Restored role workspace differs from checkpoint')
    links = [{str(path.relative_to(workspace)): os.readlink(path)
        for path in workspace.rglob('*') if path.is_symlink()}
        for workspace in (roles, frozen_roles)]
    try:
        unsafe_links = any(Path(target).is_absolute() or not (roles / name).resolve().is_relative_to(roles)
            for name, target in links[0].items())
    except (OSError, RuntimeError):
        unsafe_links = True
    modes = [{str(path.relative_to(workspace)): path.lstat().st_mode
        for path in (workspace, *workspace.rglob('*'))}
        for workspace in (roles, frozen_roles)]
    if links[0] != links[1] or unsafe_links or modes[0] != modes[1] or tree_hash(roles, exact=True) != tree_hash(frozen_roles, exact=True):
        logging.warning('Restored role workspace differs from checkpoint')


def _client_configuration(client):
    return dict(endpoint=str(client.base_url.copy_with(username='', password='', query=None)),
        max_retries=client.max_retries, timeout=httpx.Timeout(client.timeout).as_dict())


def checkpoint(runtime, destination=None):
    server = runtime.server
    with runtime._checkpoint_lock, ExitStack() as stack:
        if runtime._pool is not None and not runtime._safe_point:
            raise RuntimeError('Team roles must stop at a quiescent handoff before checkpoint')
        if runtime._stage not in ('boundary', 'ceo'):
            raise RuntimeError('Team is outside a recoverable handoff')
        _known_operation(runtime.root)
        recorders = [role.usage for role in runtime.roles.values()]
        customer = getattr(server.simulator, 'customer_simulator', None)
        if customer:
            recorders.append(customer.usage_recorder)
        for recorder in recorders:
            if not recorder.call_lock.acquire(blocking=False):
                raise RuntimeError('Model request is still in flight')
            stack.callback(recorder.call_lock.release)
        for role in runtime.roles.values():
            if role.agent._pending_tool_calls:
                raise RuntimeError('Unresolved tool operations cannot be checkpointed')
            for executor in (role.executor, server.role_executors[role.identity.role]):
                if not executor._execute_lock.acquire(blocking=False):
                    raise RuntimeError('Role execution is still in flight')
                stack.callback(executor._execute_lock.release)
                if executor.preserved_process:
                    raise RuntimeError('Role process outcome is unknown')
        if not server._advance_lock.acquire(blocking=False):
            raise RuntimeError('World settlement is still in flight')
        stack.callback(server._advance_lock.release)
        def unpause():
            with server._sql_lock:
                server._sql_paused = False
                server._sql_lock.notify_all()
        stack.callback(unpause)
        with server._sql_lock:
            server._sql_paused = True
            if not server._sql_lock.wait_for(lambda: server._sql_active == 0, timeout=180):
                raise RuntimeError('World query is still in flight')
        with server._request_lock:
            def quiescent():
                requests = server.public_request_state()
                return not requests['active'] and not requests['queued']
            if not server._request_lock.wait_for(quiescent, timeout=3):
                raise RuntimeError('Public HTTP operation is still in flight')
        if not server._lock.acquire(blocking=False):
            raise RuntimeError('World API operation is still in flight')
        stack.callback(server._lock.release)
        stack.enter_context(server._request_lock)
        requests = server.public_request_state()
        if requests['active'] or requests['queued']:
            raise RuntimeError('Public HTTP operation is still in flight')
        _known_operation(runtime.root)
        if server._operation_failed or server._step_day_timed_out:
            if server._advance_outcome is not None:
                raise RuntimeError('World advancement failed or timed out; restore a previous checkpoint')
            raise BusinessOutcomeUnknown('World operation outcome unknown; HOLD for manual checkpoint resume')
        for role in runtime.roles.values():
            if role.store:
                role.store.assert_healthy(settle=3)
            role.agent._save_conversation_snapshot(strict=True)
        if server.conn is None or not hasattr(server.simulator, 'save_rng_states'):
            raise ValueError('Team checkpoint requires a persistent world and complete simulator RNG state')
        target = Path(destination).resolve() if destination else runtime.root / 'checkpoints' / uuid.uuid4().hex
        if target.is_relative_to(runtime.root / 'roles') or target.is_relative_to(runtime.root / 'private'):
            raise ValueError('Checkpoint must stay outside role workspaces and live private state')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError('Immutable checkpoint already exists')
        temporary = Path(tempfile.mkdtemp(prefix='.team-freeze-', dir=target.parent))
        try:
            _freeze(runtime, temporary)
            temporary.rename(target)
            write_json(runtime.root / 'checkpoint.json', dict(snapshot=str(target), sha256=file_hash(target / 'checkpoint.json')))
            return target
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


def validate_executor_settings(configuration, roles):
    from .agents.bash_agent.tools import validate_bash_limits
    policy_keys = ('analyst_bash_timeout', 'analyst_bash_warn_after')
    explicit = any(key in configuration for key in policy_keys)
    if explicit:
        if not all(key in configuration for key in policy_keys):
            logging.warning('Incomplete analyst Bash policy in checkpoint')
        with advisory():
            validate_bash_limits(*(configuration[key] for key in policy_keys))
    for role, saved in roles.items():
        for kind in ('executor', 'script_executor'):
            settings = saved[kind]
            actual = (settings['bash_timeout'], settings.get('bash_warn_after'))
            with advisory():
                validate_bash_limits(*actual)
            expected = ((configuration['analyst_bash_timeout'], configuration['analyst_bash_warn_after'])
                if all(key in configuration for key in policy_keys) and role != 'ceo' and kind == 'executor' else (1200, None))
            if (explicit and actual != expected) or (not explicit and actual[1] is not None):
                logging.warning('Checkpoint Bash policy drift for ' + role + ' ' + kind)


def _freeze(runtime, directory):
    from .db_protection import save_session_db
    from .execution_capture import text_sources
    server = runtime.server
    server.simulator.save_rng_states()
    if server.event_logger:
        server.event_logger.save_incremental()
    save_session_db(server.conn, directory / 'world.nmdb')
    copy_workspace(runtime.root / 'roles', directory / 'roles', exact=True)
    copy_workspace(runtime.root / 'private', directory / 'private')
    evidence = None
    if runtime.roles['ceo'].store:
        evidence = runtime.roles['ceo'].store.snapshot(directory / 'private' / 'evidence.sqlite')
        for suffix in ('-wal', '-shm'):
            (directory / 'private' / ('evidence.sqlite' + suffix)).unlink(missing_ok=True)
    roles = {name: dict(identity=role.identity.fields(), usage=role.usage.summary,
        weekly_check=role.usage.weekly_check,
        client=_client_configuration(role.agent.client),
        context_id=role.usage.context_id, last_request_event=role.usage.last_request_event,
        expected_model=getattr(role.usage, 'expected_model', None),
        total_turns=role.agent.total_turns, system_prompt=role.agent.system_prompt,
        executor=dict(bash_timeout=role.executor.bash_timeout, bash_warn_after=role.executor.bash_warn_after,
            stop_on_timeout=role.executor.stop_on_timeout),
        script_executor=dict(bash_timeout=server.role_executors[name].bash_timeout,
            bash_warn_after=server.role_executors[name].bash_warn_after,
            stop_on_timeout=server.role_executors[name].stop_on_timeout))
        for name, role in runtime.roles.items()}
    with advisory():
        validate_executor_settings(runtime.configuration, roles)
    result = server._last_day_result
    if result is not None and not is_dataclass(result):
        raise ValueError('Checkpoint requires the simulator DayResult data shape')
    state = dict(version=1, saved_at=datetime.now(timezone.utc).isoformat(),
        source_root=str(runtime.root), configuration=runtime.configuration,
        benchmark_config=asdict(server.tools.config), day=server.tools.current_day,
        seed=server.tools.seed, tools_rng=server.tools.rng.bit_generator.state,
        shared_rng=server.tools.rng is server.simulator.rng,
        stage=runtime._stage, week=runtime._week, phase=runtime.phase,
        world_state_id=server.world_state_id, roles=roles,
        messages=[asdict(message) for message in runtime.messages], failure=runtime.failure,
        scripts={role: server.get_daily_scripts(role) for role in runtime.roles},
        script_results=server.role_script_results, dashboard=server._last_dashboard,
        day_result=asdict(result) if result is not None else None,
        evidence=evidence, live_customer=server.simulator.customer_simulator is not None,
        public_artifacts=artifact_hashes(runtime._public_dir),
        source_sha256=tree_hash(Path(__file__).parent),
        sources=text_sources(dict(week=runtime._week, dashboard=server._last_dashboard,
            script_results=server.role_script_results)))
    if server.shock_manager:
        state['scenario'] = asdict(server.shock_manager.scenario)
    customer = server.simulator.customer_simulator
    if customer and getattr(customer, 'usage_recorder', None):
        usage = customer.usage_recorder
        if usage.path and usage.path.exists():
            shutil.copy2(usage.path, directory / 'private' / 'simulator-model-usage.jsonl')
        state['simulator_usage'] = dict(summary=usage.summary, context_id=usage.context_id,
            client=_client_configuration(customer.client),
            pricing=usage.pricing, expected_model=getattr(usage, 'expected_model', None))
    if server.event_logger:
        logger = server.event_logger
        shutil.copy2(logger.log_file, directory / 'private' / 'world-events.jsonl')
        state['event_logger'] = dict(metadata=asdict(logger.metadata),
            counters={key: getattr(logger, key) for key in
                ('current_day', '_event_count', '_total_llm_cost', '_missing_llm_cost')})
    write_json(directory / 'team.json', state)
    write_json(directory / 'checkpoint.json', dict(version=1, files={
        'team.json': file_hash(directory / 'team.json'), 'world.nmdb': file_hash(directory / 'world.nmdb')},
        roles_hash_policy='all_files', roles_sha256=tree_hash(directory / 'roles', exact=True),
        private_sha256=tree_hash(directory / 'private'),
        log_cutoffs={str(path.relative_to(directory)): path.stat().st_size
            for path in (directory / 'private').rglob('*.jsonl')}))


def validate_snapshot(snapshot):
    snapshot = Path(snapshot).resolve()
    with advisory():
        receipt = json.loads((snapshot / 'checkpoint.json').read_text())
        if receipt.get('version') != 1 or set(receipt.get('files', {})) != {'team.json', 'world.nmdb'}:
            logging.warning('Incomplete team checkpoint')
        if receipt.get('roles_hash_policy') not in (None, 'all_files'):
            logging.warning('Unknown role checkpoint hash policy')
        for name in ('team.json', 'world.nmdb'):
            with advisory():
                if file_hash(snapshot / name) != receipt.get('files', {}).get(name):
                    logging.warning('Team checkpoint checksum mismatch for %s', name)
        for name in ('roles', 'private'):
            with advisory():
                if tree_hash(snapshot / name, exact=name == 'roles') != receipt.get(name + '_sha256'):
                    logging.warning('Team checkpoint checksum mismatch for %s', name)
    return snapshot, json.loads((snapshot / 'team.json').read_text())


def restore(cls, snapshot, root, *, client_factory, public_dir=None, token_counter=None,
            simulator_factory=None, force_resume=False, **options):
    from numpy.random import default_rng
    from .config import BenchmarkConfig, ScenarioPack
    from .db_protection import load_session_db
    from .simulation import Simulator
    from .shocks import ShockManager
    from .tools import AgentTools
    snapshot, state = validate_snapshot(snapshot)
    root = Path(root).resolve()
    source = Path(state['source_root'])
    if root.is_relative_to(source) or source.is_relative_to(root) or root.is_relative_to(snapshot):
        logging.warning('Recovery root overlaps the source attempt or snapshot')
    if root.exists():
        logging.warning('Recovery target already exists; restoring into it')
    public = Path(public_dir or Path(__file__).resolve().parents[2] / 'public')
    source_changed = tree_hash(Path(__file__).parent) != state['source_sha256']
    public_changed = artifact_hashes(public) != state['public_artifacts']
    configuration = state['configuration']
    with advisory():
        validate_executor_settings(configuration, state['roles'])
    for key in list(options):
        if key in configuration:
            if options.pop(key) != configuration[key]:
                logging.warning('Recovery restores checkpoint setting for %s; supplied value differs', key)
        elif key in ('analyst_bash_timeout', 'analyst_bash_warn_after'):
            field = 'bash_timeout' if key == 'analyst_bash_timeout' else 'bash_warn_after'
            requested = options.pop(key)
            if any(saved['executor'].get(field) != requested for name, saved in state['roles'].items() if name != 'ceo'):
                logging.warning('Recovery restores checkpoint analyst setting for %s; supplied value differs', key)
    if state['live_customer'] and simulator_factory is None:
        logging.warning('Live simulator recovery has no customer client factory')
    root.mkdir(parents=True, exist_ok=True)
    try:
        if (snapshot / 'roles').resolve() != (root / 'roles').resolve():
            copy_workspace(snapshot / 'roles', root / 'roles', exact=True)
        with advisory():
            _validate_restored_roles(snapshot, root)
        if (snapshot / 'private').resolve() != (root / 'private').resolve():
            copy_workspace(snapshot / 'private', root / 'private', exact=True)
    except BaseException:
        logging.exception('Checkpoint workspace copy failed')
        raise
    write_json(root / 'recovery.json', dict(snapshot=str(snapshot),
        source_root=str(source), attempt_id=uuid.uuid4().hex,
        checkpoint_sha256=optional_hash(snapshot / 'checkpoint.json'),
        source_revision_changed=source_changed, public_build_changed=public_changed,
        discard_post_checkpoint_operations=True, force_resume=force_resume))
    world = root / 'world.nmdb'
    if (snapshot / 'world.nmdb').resolve() != world.resolve():
        shutil.copy2(snapshot / 'world.nmdb', world)
    conn = load_session_db(world)
    try:
        config = BenchmarkConfig(**state['benchmark_config'])
        rng = default_rng(config.seed)
        simulator = simulator_factory(conn, config, rng) if simulator_factory else Simulator(conn, config, rng)
        simulator.initialize(resume=True)
        shock_manager = None
        if 'scenario' in state:
            shock_manager = ShockManager(conn, rng, ScenarioPack(**state['scenario']))
            simulator.shock_manager = shock_manager
        with advisory():
            if not simulator.restore_rng_states() or simulator.current_day != state['day']:
                logging.warning('World RNG checkpoint is missing or differs from team day')
        tools_rng = rng if state['shared_rng'] else default_rng(state['seed'])
        with advisory():
            tools_rng.bit_generator.state = state['tools_rng']
        tools = AgentTools(conn, state['day'], root / 'private' / 'world-tools',
            rng=tools_rng, config=config, seed=state['seed'])
        runtime = cls(root, tools, conn=conn, simulator=simulator, shock_manager=shock_manager,
            client_factory=client_factory, public_dir=public, token_counter=token_counter,
            _restored_state=state, **configuration, **options)
        return runtime
    except BaseException:
        conn.close()
        raise


def _relocate_prompt(text, original, destination):
    from .execution_capture import CapturedText, slice_origins, slice_read_spans
    if isinstance(text, list):
        return [_relocate_prompt(item, original, destination) for item in text]
    if isinstance(text, dict):
        return {key: _relocate_prompt(value, original, destination) for key, value in text.items()}
    if not isinstance(text, str):
        return text
    if not isinstance(text, CapturedText):
        return text.replace(original, destination)
    parts, origins, reads, start, target = [], [], [], 0, 0
    position = text.find(original)
    while position >= 0:
        end = position + len(original)
        if not any(a < end and b > position for a, b in
                (item['request_range'] for item in [*text.origins, *text.pf_read_spans])):
            piece = text[start:position]
            parts.extend((piece, destination))
            origins.extend(slice_origins(text.origins, start, position, target))
            reads.extend(slice_read_spans(text.pf_read_spans, start, position, target))
            target += len(piece) + len(destination)
            start = end
        position = text.find(original, end)
    parts.append(text[start:])
    origins.extend(slice_origins(text.origins, start, len(text), target))
    reads.extend(slice_read_spans(text.pf_read_spans, start, len(text), target))
    return CapturedText(''.join(parts), origins, text.pf_read, reads)


def restore_runtime_state(runtime, state):
    from .execution_capture import restore_sources
    from .multi_agent import TeamMessage
    from .simulation import DayResult
    server = runtime.server
    original = state['source_root']
    old_workspace, new_workspace = 'Workspace: ' + original + '/roles/', 'Workspace: ' + str(runtime.root) + '/roles/'
    runtime._stage, runtime._week = state['stage'], state['week']
    server.team_phase = state['phase']
    server.world_state_id = state['world_state_id']
    runtime.messages = [TeamMessage(**message) for message in state['messages']]
    runtime.failure = None
    captured = dict(week=runtime._week, dashboard=state['dashboard'], script_results=state['script_results'])
    restore_sources(captured, state['sources'])
    runtime._week = captured['week']
    if 'observation' in runtime._week:
        runtime._week['observation'] = _relocate_prompt(runtime._week['observation'], old_workspace, new_workspace)
    server._last_dashboard = captured['dashboard']
    server.role_script_results = captured['script_results']
    server._last_day_result = DayResult(**state['day_result']) if state['day_result'] else None
    server._role_scripts = state['scripts']
    server._daily_scripts = server._role_scripts['ceo']
    for name, role in runtime.roles.items():
        saved = state['roles'][name]
        if _client_configuration(role.agent.client) != saved['client']:
            logging.warning('Recovery client configuration differs from checkpoint for %s', name)
        role.usage.summary = saved['usage']
        role.usage.weekly_check = saved.get('weekly_check')
        role.usage.context_id = saved['context_id']
        role.usage.last_request_event = saved['last_request_event']
        role.usage.expected_model = saved['expected_model']
        role.agent.system_prompt = saved['system_prompt'].replace(original, str(runtime.root))
        if not role.agent.load_conversation_snapshot(role.agent._snapshot_path):
            logging.warning('Cannot restore %s role conversation; continuing with available conversation', name)
        for message in role.agent.conversation:
            before, after = (original, str(runtime.root)) if message.role == 'system' else (old_workspace, new_workspace)
            message.content = _relocate_prompt(message.content, before, after)
        role.agent._last_observation = _relocate_prompt(role.agent._last_observation, old_workspace, new_workspace)
        role.agent.total_turns = saved['total_turns']
    customer = server.simulator.customer_simulator
    if 'simulator_usage' in state:
        usage = customer.usage_recorder
        saved = state['simulator_usage']
        if _client_configuration(customer.client) != saved['client']:
            logging.warning('Recovery simulator client configuration differs from checkpoint')
        usage.path = runtime.root / 'private' / 'simulator-model-usage.jsonl'
        usage.summary, usage.context_id, usage.pricing = saved['summary'], saved['context_id'], saved['pricing']
        usage.expected_model = saved['expected_model']
    if 'event_logger' in state:
        from .event_logger import EventLogger, RunMetadata
        saved = state['event_logger']
        metadata = saved['metadata']
        logger = EventLogger(metadata['run_id'], runtime.root / 'private' / 'events',
            metadata['seed'], metadata['scenario'], metadata['config'])
        logger._file.close()
        logger.log_file = runtime.root / 'private' / 'world-events.jsonl'
        logger._file = logger.log_file.open('a')
        logger.metadata = RunMetadata(**metadata)
        for key, value in saved['counters'].items():
            setattr(logger, key, value)
        server.event_logger = logger
        server.tools.set_event_logger(logger)
        server.simulator.set_event_logger(logger)
