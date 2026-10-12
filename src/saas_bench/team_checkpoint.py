"""Immutable, private snapshots at quiescent team handoffs."""

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, is_dataclass
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import httpx

from .run_state import artifact_hashes, copy_workspace, file_hash, tree_hash, write_json


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


def managed_snapshot(parent, value, checksum):
    snapshot = Path(value)
    if (not isinstance(value, str) or str(snapshot) != value or snapshot.parent != parent or
            len(snapshot.name) != 32 or any(c not in '0123456789abcdef' for c in snapshot.name) or
            snapshot.is_symlink() or snapshot.resolve() != snapshot or not snapshot.is_dir() or
            (snapshot / 'checkpoint.json').is_symlink() or
            not isinstance(checksum, str) or len(checksum) != 64 or
            any(c not in '0123456789abcdef' for c in checksum) or
            file_hash(snapshot / 'checkpoint.json') != checksum):
        raise ValueError('Invalid managed checkpoint path or receipt')
    return snapshot


def _pin_snapshot(snapshot):
    snapshot = Path(snapshot).resolve()
    parent = snapshot.parent
    signal = parent / '.retention.lock'
    if not signal.exists() and not signal.is_symlink():
        return
    with retention_lock(parent):
        ledger_path = parent / '.retention.json'
        if ledger_path.is_symlink():
            raise ValueError('Checkpoint retention ledger cannot be a symlink')
        ledger = json.loads(ledger_path.read_text())
        if (not isinstance(ledger, dict) or set(ledger) != {'version', 'snapshots'} or
                ledger['version'] != 1 or not isinstance(ledger['snapshots'], list)):
            raise ValueError('Invalid checkpoint retention ledger')
        entries = {}
        for item in ledger['snapshots']:
            if (not isinstance(item, dict) or set(item) != {'snapshot', 'sha256'} or
                    not isinstance(item['snapshot'], str) or not isinstance(item['sha256'], str) or
                    item['snapshot'] in entries):
                raise ValueError('Invalid checkpoint retention entry')
            entries[item['snapshot']] = item['sha256']
        if str(snapshot) not in entries:
            return
        checksum = entries[str(snapshot)]
        managed_snapshot(parent, str(snapshot), checksum)
        path = parent / '.retention-pins.json'
        if path.is_symlink():
            raise ValueError('Checkpoint retention pins cannot be a symlink')
        pins = json.loads(path.read_text()) if path.exists() else {}
        if not isinstance(pins, dict):
            raise ValueError('Invalid checkpoint retention pins')
        pins[str(snapshot)] = checksum
        write_json(path, pins)


def business_operations(root):
    return {path.name: file_hash(path) for path in (Path(root) / 'private' / 'operations').glob('*.json')
        if not _wrapper(json.loads(path.read_text()))}


def _wrapper(receipt):
    kind = receipt.get('kind', '')
    return kind == 'ceo_scripts' or kind.startswith('ceo_tool:')


def recovery_operation_safe(root, receipt):
    proof = receipt.get('recovery', {})
    if (not _wrapper(receipt) or receipt.get('wrapper_version') != 1 or
            receipt.get('status') not in ('completed', 'interrupted') or
            proof.get('classification') not in ('read_only', 'read_only_interrupted', 'pending_read_only') or
            proof.get('client_closed') is not True or proof.get('operation_failed') is not False or
            proof.get('step_day_timed_out') is not False or proof.get('queued_requests') is not False or
            not isinstance(proof.get('business_operations_after'), dict) or
            receipt.get('business_operations_before') != proof['business_operations_after'] or
            not isinstance(receipt.get('mutation_epoch_before'), int) or
            receipt['mutation_epoch_before'] != proof.get('mutation_epoch_after') or
            not isinstance(receipt.get('workspace_sha256_before'), str) or
            receipt['workspace_sha256_before'] != proof.get('workspace_sha256_after') or
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
        if receipt.get('status') != 'completed' and not recovery_operation_safe(root, receipt):
            raise RuntimeError('Business operation outcome unknown; automatic recovery refused')


def _approved_residuals(snapshot):
    from .team_recovery import approval
    revision = approval()
    if not revision:
        return {}
    matches = [saved for saved in revision['attempts'].values()
        if saved.get('checkpoint') == str(Path(snapshot).resolve()) and
        saved.get('checkpoint_sha256') == file_hash(Path(snapshot) / 'checkpoint.json')]
    approved = {}
    for saved in matches:
        for residual in saved.get('residual_operations', []):
            root = Path(residual['root'])
            filename = residual['operation']
            if (not root.is_absolute() or root.resolve() != root or Path(filename).name != filename or
                    not filename.endswith('.json') or not residual.get('reason', '').strip() or
                    not residual.get('evidence')):
                raise ValueError('Recovery residual requires an exact root, receipt, reason and evidence')
            operation = root / 'private' / 'operations' / filename
            if operation.is_symlink():
                raise ValueError('Recovery residual operation cannot be a symlink')
            if file_hash(operation) != residual['sha256']:
                raise ValueError('Approved residual operation changed')
            receipt = json.loads(operation.read_text())
            classification = residual['classification']
            if classification in ('read_only_interrupted', 'pending_read_only'):
                if not _wrapper(receipt):
                    raise ValueError('Read-only residual approval cannot cover a business operation')
                proof = receipt.get('recovery', {})
                contradictory = (proof.get('operation_failed') is True or proof.get('step_day_timed_out') is True or
                    any(before in receipt and after in proof and receipt[before] != proof[after]
                        for before, after in (('business_operations_before', 'business_operations_after'),
                            ('mutation_epoch_before', 'mutation_epoch_after'),
                            ('workspace_sha256_before', 'workspace_sha256_after'))))
                if contradictory:
                    raise ValueError('Read-only residual approval contradicts recorded mutations or server failure')
            elif classification != 'unknown_business_write' or saved.get('allow_unknown_business_write') is not True:
                raise ValueError('Unknown business write requires explicit recovery approval')
            for value, checksum in residual['evidence'].items():
                evidence = Path(value)
                if not evidence.is_absolute() or evidence.resolve() != evidence or file_hash(evidence) != checksum:
                    raise ValueError('Approved residual recovery evidence changed')
            key = str(root), filename
            if key in approved and approved[key] != residual:
                raise ValueError('Conflicting residual operation approvals')
            approved[key] = residual
    return approved


def validate_recovery_source(snapshot, state, *, current_root=None):
    """Reject rollback across operations that the stored world does not contain."""
    snapshot = Path(snapshot)
    approved = _approved_residuals(snapshot)
    frozen = {p.name: file_hash(p) for p in (snapshot / 'private' / 'operations').glob('*.json')
        if not recovery_operation_safe(snapshot, json.loads(p.read_text()))}
    roots = {Path(state['source_root'])}
    if current_root is not None:
        roots.add(Path(current_root))
    for root in roots:
        if not root.is_dir():
            raise RuntimeError('Recovery source is unavailable; operation coverage cannot be verified')
        current = {}
        for path in (root / 'private' / 'operations').glob('*.json'):
            receipt = json.loads(path.read_text())
            if recovery_operation_safe(root, receipt) or (str(root.resolve()), path.name) in approved:
                continue
            if receipt.get('status') != 'completed':
                raise RuntimeError('Business operation outcome unknown; automatic recovery refused')
            current[path.name] = file_hash(path)
        frozen_root = {name: checksum for name, checksum in frozen.items()
            if (str(root.resolve()), name) not in approved}
        # A completed receipt identifies a known outcome, not a snapshotted outcome.
        if current != frozen_root:
            raise RuntimeError('Business operations changed after checkpoint; recovery refused')


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
            raise RuntimeError('World operation outcome unknown; checkpoint refused')
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
            raise ValueError('Incomplete analyst Bash policy in checkpoint')
        validate_bash_limits(*(configuration[key] for key in policy_keys))
    for role, saved in roles.items():
        for kind in ('executor', 'script_executor'):
            settings = saved[kind]
            actual = (settings['bash_timeout'], settings.get('bash_warn_after'))
            validate_bash_limits(*actual)
            expected = ((configuration['analyst_bash_timeout'], configuration['analyst_bash_warn_after'])
                if explicit and role != 'ceo' and kind == 'executor' else (1200, None))
            if (explicit and actual != expected) or (not explicit and actual[1] is not None):
                raise ValueError('Checkpoint Bash policy drift for ' + role + ' ' + kind)


def _freeze(runtime, directory):
    from .db_protection import save_session_db
    from .execution_capture import text_sources
    server = runtime.server
    server.simulator.save_rng_states()
    if server.event_logger:
        server.event_logger.save_incremental()
    save_session_db(server.conn, directory / 'world.nmdb')
    copy_workspace(runtime.root / 'roles', directory / 'roles')
    copy_workspace(runtime.root / 'private', directory / 'private')
    evidence = None
    if runtime.roles['ceo'].store:
        evidence = runtime.roles['ceo'].store.snapshot(directory / 'private' / 'evidence.sqlite')
        for suffix in ('-wal', '-shm'):
            (directory / 'private' / ('evidence.sqlite' + suffix)).unlink(missing_ok=True)
    roles = {name: dict(identity=role.identity.fields(), usage=role.usage.summary,
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
    validate_executor_settings(runtime.configuration, roles)
    result = server._last_day_result
    if result is not None and not is_dataclass(result):
        raise ValueError('Checkpoint requires the simulator DayResult data shape')
    state = dict(version=1, source_root=str(runtime.root), configuration=runtime.configuration,
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
        roles_sha256=tree_hash(directory / 'roles'), private_sha256=tree_hash(directory / 'private'),
        log_cutoffs={str(path.relative_to(directory)): path.stat().st_size
            for path in (directory / 'private').rglob('*.jsonl')}))


def validate_snapshot(snapshot):
    snapshot = Path(snapshot).resolve()
    receipt = json.loads((snapshot / 'checkpoint.json').read_text())
    if receipt.get('version') != 1 or set(receipt.get('files', {})) != {'team.json', 'world.nmdb'}:
        raise ValueError('Incomplete team checkpoint')
    for name, checksum in receipt['files'].items():
        if file_hash(snapshot / name) != checksum:
            raise ValueError('Team checkpoint checksum mismatch for ' + name)
    for name in ('roles', 'private'):
        if tree_hash(snapshot / name) != receipt[name + '_sha256']:
            raise ValueError('Team checkpoint checksum mismatch for ' + name)
    state = json.loads((snapshot / 'team.json').read_text())
    _known_operation(snapshot)
    return snapshot, state


def restore(cls, snapshot, root, *, client_factory, public_dir=None, token_counter=None,
            simulator_factory=None, **options):
    from numpy.random import default_rng
    from .config import BenchmarkConfig, ScenarioPack
    from .db_protection import load_session_db
    from .simulation import Simulator
    from .shocks import ShockManager
    from .tools import AgentTools
    from .team_recovery import approval, source_revision_allowed
    revision = approval()
    migrating = revision and any(str(Path(snapshot).resolve()) == saved['checkpoint']
        for saved in revision['attempts'].values())
    if not migrating:
        _pin_snapshot(snapshot)
    snapshot, state = validate_snapshot(snapshot)
    migrated = source_revision_allowed(snapshot, state) if tree_hash(Path(__file__).parent) != state['source_sha256'] else False
    validate_recovery_source(snapshot, state)
    root = Path(root).resolve()
    source = Path(state['source_root'])
    if root.is_relative_to(source) or source.is_relative_to(root) or root.is_relative_to(snapshot):
        raise ValueError('Recovery root must be independent of the source attempt and snapshot')
    if root.exists():
        raise FileExistsError('Recovery requires a new attempt directory')
    public = Path(public_dir or Path(__file__).resolve().parents[2] / 'public')
    if artifact_hashes(public) != state['public_artifacts']:
        raise ValueError('Public build differs from frozen team configuration')
    if tree_hash(Path(__file__).parent) != state['source_sha256'] and not migrated:
        raise ValueError('Source differs from frozen team configuration')
    configuration = state['configuration']
    validate_executor_settings(configuration, state['roles'])
    for key in list(options):
        if key in configuration:
            if options.pop(key) != configuration[key]:
                raise ValueError('Recovery configuration drift for ' + key)
        elif key in ('analyst_bash_timeout', 'analyst_bash_warn_after'):
            field = 'bash_timeout' if key == 'analyst_bash_timeout' else 'bash_warn_after'
            requested = options.pop(key)
            if any(saved['executor'].get(field) != requested for name, saved in state['roles'].items() if name != 'ceo'):
                raise ValueError('Recovery configuration drift for ' + key)
    if state['live_customer'] and simulator_factory is None:
        raise ValueError('Live simulator recovery requires its frozen customer client factory')
    root.mkdir(parents=True)
    copy_workspace(snapshot / 'roles', root / 'roles')
    copy_workspace(snapshot / 'private', root / 'private')
    write_json(root / 'recovery.json', dict(snapshot=str(snapshot),
        source_root=str(source), attempt_id=uuid.uuid4().hex,
        checkpoint_sha256=file_hash(snapshot / 'checkpoint.json'),
        source_revision_sha256=os.environ.get('CEOBENCH_RECOVERY_APPROVAL_SHA256') if migrated else None))
    conn = load_session_db(snapshot / 'world.nmdb')
    try:
        config = BenchmarkConfig(**state['benchmark_config'])
        rng = default_rng(config.seed)
        simulator = simulator_factory(conn, config, rng) if simulator_factory else Simulator(conn, config, rng)
        simulator.initialize(resume=True)
        shock_manager = None
        if 'scenario' in state:
            shock_manager = ShockManager(conn, rng, ScenarioPack(**state['scenario']))
            simulator.shock_manager = shock_manager
        if not simulator.restore_rng_states() or simulator.current_day != state['day']:
            raise ValueError('World RNG checkpoint is missing or differs from team day')
        tools_rng = rng if state['shared_rng'] else default_rng(state['seed'])
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
            raise ValueError('Recovery client configuration drift for ' + name)
        role.usage.summary = saved['usage']
        role.usage.context_id = saved['context_id']
        role.usage.last_request_event = saved['last_request_event']
        role.usage.expected_model = saved['expected_model']
        role.agent.system_prompt = saved['system_prompt'].replace(original, str(runtime.root))
        if not role.agent.load_conversation_snapshot(role.agent._snapshot_path):
            raise ValueError('Cannot restore ' + name + ' role conversation')
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
            raise ValueError('Recovery simulator client configuration drift')
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
