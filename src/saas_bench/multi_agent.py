from dataclasses import asdict, dataclass, replace
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
import json
import logging
import os
from pathlib import Path
import threading
import shutil
import tempfile
import time
from types import MappingProxyType
import uuid

from .agents.bash_agent.agent import BashAgent, FinalText
from .agents.bash_agent.tools import BashAgentToolExecutor, get_bash_agent_tool_descriptions, validate_bash_limits
from .api_server import NovaMindAPIServer
from .execution_capture import join_text
from .model_usage import ModelUsage
from .role_policy import AgentIdentity, CALL_POLICY, READABLE_ROLES, ROLES, RoleAudit
from .run_state import tree_hash, write_json
from .run_lifecycle import BusinessOutcomeUnknown, RunCancelled, WorkerLifecycle
from .sql_evidence import FORMAT, SQLEvidenceStore
from .text_registry import TextRegistry
from .team_prompts import WEEKLY_TASKS, system_prompt


@dataclass
class RoleRuntime:
    identity: AgentIdentity
    workspace: Path
    agent: BashAgent
    executor: BashAgentToolExecutor
    registry: TextRegistry
    usage: ModelUsage
    store: SQLEvidenceStore | None
    audit: RoleAudit


@dataclass(frozen=True)
class RunOutcome:
    day: int
    reason: str


@dataclass
class TeamMessage:
    request_id: str
    sender: str
    receiver: str
    text: str
    day: int
    world_state_id: str
    sender_session_id: str
    receiver_session_id: str
    status: str = 'queued'
    reply: str | None = None
    handoff_commit: str | None = None


class MultiAgentRuntime:
    def __init__(self, root, tools, *, client_factory, mode='git', conn=None,
                 simulator=None, public_dir=None, run_id=None, world_id=None,
                 model="deepseek-v4.1-flash", reasoning_effort="high", total_days=500, token_counter=None,
                 model_pricing=None, _restored_state=None, analyst_bash_timeout=1200,
                 analyst_bash_warn_after=None, ceo_bash_timeout=1200, ceo_bash_warn_after=None, **server_options):
        if mode not in ('git', 'pf'):
            raise ValueError('Team supports git and pf modes')
        if type(total_days) is not int or total_days < 7:
            raise ValueError('total_days must include at least one full week')
        self.total_days = total_days
        self.configuration = dict(mode=mode, model=model, reasoning_effort=reasoning_effort,
            total_days=total_days, model_pricing=model_pricing)
        self.configuration.update(ceo_bash_timeout=ceo_bash_timeout, ceo_bash_warn_after=ceo_bash_warn_after)
        from .run_warnings import advisory
        with advisory():
            validate_bash_limits(analyst_bash_timeout, analyst_bash_warn_after)
        if not _restored_state or 'analyst_bash_timeout' in _restored_state['configuration']:
            self.configuration.update(analyst_bash_timeout=analyst_bash_timeout,
                                      analyst_bash_warn_after=analyst_bash_warn_after)
        if _restored_state:
            from .team_checkpoint import validate_executor_settings
            validate_executor_settings(_restored_state['configuration'], _restored_state['roles'])
        if mode == 'pf' and token_counter is None:
            from .payload_tokens import load_counter
            token_counter = load_counter('opencode', model)
        self.effective_end = total_days // 7 * 7
        self._lifecycle = WorkerLifecycle()
        self._message_lock = threading.Lock()
        self._operation_lock = threading.RLock()
        self._pending_operations = None
        self.messages = []
        self.failure = None
        self._pool = None
        self._checkpoint_lock = threading.RLock()
        self._safe_point = False
        self._stage = 'boundary'
        self._week = {}
        self.checkpoint_callback = None
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        private = self.root / 'private'
        private.mkdir(mode=0o700, exist_ok=bool(_restored_state))
        self.message_path = private / 'messages.jsonl'
        if _restored_state:
            identities = [_restored_state['roles'][role]['identity'] for role in ROLES]
            saved_run, saved_world = identities[0]['run_id'], identities[0]['world_id']
            if any((item['run_id'], item['world_id']) != (saved_run, saved_world) for item in identities):
                logging.warning('Restored roles disagree on run or world identity')
            if run_id not in (None, saved_run) or world_id not in (None, saved_world):
                logging.warning('Recovery run or world identity changed')
            run_id, world_id = saved_run, saved_world
        else:
            run_id, world_id = run_id or uuid.uuid4().hex, world_id or uuid.uuid4().hex
        from .runtime_temp import prepare_tmpdir
        self._temp_root = prepare_tmpdir(self.root / 'tmp')
        self._sockets = tempfile.TemporaryDirectory(prefix='ceobench-team-', dir=self._temp_root)
        public = Path(public_dir or Path(__file__).resolve().parents[2] / 'public')
        self._public_dir = public
        workspaces = {role: self.root / 'roles' / role for role in ROLES}
        sockets = {role: Path(self._sockets.name) / role / 'api.sock' for role in ROLES}
        stores, runtimes, script_executors, audits = {}, {}, {}, {}
        for role in ROLES:
            workspace = workspaces[role]
            workspace.mkdir(parents=True, exist_ok=bool(_restored_state))
            if not _restored_state:
                shutil.copytree(public / 'docs', workspace / 'docs')
                shutil.copy2(public / 'novamind-client', workspace / 'novamind-operation')
                (workspace / 'MEMORY.md').write_text('')
            identity = (AgentIdentity(**_restored_state['roles'][role]['identity']) if _restored_state else
                AgentIdentity(run_id, world_id, role, uuid.uuid4().hex, uuid.uuid4().hex))
            role_private = private / role
            role_private.mkdir(mode=0o700, exist_ok=bool(_restored_state))
            write_json(role_private / 'identity.json', identity.fields())
            store = stores[role] = SQLEvidenceStore(private / 'evidence.sqlite', dict(
                identity.fields(), branch_id=role, data_source_id=world_id, format=FORMAT,
                capture_scope='execution')) if mode == 'pf' else None
            audit = audits[role] = RoleAudit(role_private / 'audit.jsonl', identity)
            registry = TextRegistry(workspace, mode, store, sim_day=lambda: tools.current_day, identity=identity)
            env = dict(NOVAMIND_API_PORT='1', NOVAMIND_API_SOCKET='/run/novamind-api/api.sock',
                       PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(workspace / 'docs'))
            executor_options = dict(workspace_path=workspace, identity=identity, env=env,
                require_sandbox=True, evidence_store=store, audit=audit, api_socket=sockets[role],
                readable_workspaces=tuple(workspaces[r] for r in READABLE_ROLES[role]))
            settings = (dict(bash_timeout=analyst_bash_timeout, bash_warn_after=analyst_bash_warn_after) if role != 'ceo'
                        else dict(bash_timeout=ceo_bash_timeout, bash_warn_after=ceo_bash_warn_after))
            script_settings = {}
            if _restored_state:
                settings = _restored_state['roles'][role]['executor']
                script_settings = _restored_state['roles'][role]['script_executor']
            executor = BashAgentToolExecutor(**executor_options, text_registry=registry, **settings)
            script_executors[role] = BashAgentToolExecutor(**dict(executor_options, env=dict(env)), **script_settings)
            executor.lifecycle = script_executors[role].lifecycle = self._lifecycle
            executor.temp_root = script_executors[role].temp_root = self._temp_root
            usage = ModelUsage(role_private / 'model-usage.jsonl', role,
                pricing=(model_pricing or {}).get('rates', {}), evidence_store=store, identity=identity, token_counter=token_counter)
            client = client_factory(role)
            generation_policy = {}
            if _restored_state:
                generation_policy = dict(_restored_state['roles'][role].get('generation_policy', {}))
                if not generation_policy and 'runtime_environment' not in _restored_state:
                    generation_policy = dict(memory_max_chars=40000, invalid_generation_attempts=4,
                        max_output_tokens=128000 if type(client).__module__.startswith('anthropic') else 16384)
                from .runtime_options import OPTIONS
                for key, option in dict(memory_max_chars='memory_max_chars', max_output_tokens='model_max_output_tokens',
                                        invalid_generation_attempts='invalid_generation_attempts').items():
                    if OPTIONS[option][0] in _restored_state.get('_explicit_runtime_environment', []):
                        generation_policy.pop(key, None)
            agent = BashAgent(get_bash_agent_tool_descriptions(text_registration=True, pf_queries=mode == 'pf',
                ask_analyst=role == 'ceo', team=True, bash_timeout=executor.bash_timeout, bash_warn_after=executor.bash_warn_after),
                client, workspace_path=workspace, usage_recorder=usage,
                model=model, reasoning_effort=reasoning_effort, total_days=total_days,
                text_registration=True, pf=mode == 'pf', identity=identity,
                allow_final_text=role != 'ceo', **generation_policy)
            agent.system_prompt = system_prompt(agent.system_prompt, role, mode, total_days,
                self.effective_end, workspace, [workspaces[r] for r in READABLE_ROLES[role]])
            agent.lifecycle = self._lifecycle
            registry.git_run = executor.run_private
            agent.git_run = executor.run_private
            agent._snapshot_path = role_private / 'conversation.json'
            runtimes[role] = RoleRuntime(identity, workspace, agent, executor, registry, usage, store, audit)
        self.roles = MappingProxyType(runtimes)
        self.server = NovaMindAPIServer(tools, simulator=simulator, conn=conn,
            script_workspace=workspaces['ceo'], require_sandbox=True, sql_evidence=stores['ceo'],
            role_sockets=sockets, role_stores=stores, role_executors=script_executors, role_audits=audits, **server_options)
        self.server.script_check = self._check
        for runtime in self.roles.values():
            runtime.usage.request_check = self._check_model_send
        self._guard_business_operations()
        self.server.request_finished_callback = self._settle_pending_operations
        for runtime in self.roles.values():
            if runtime.executor.pf_queries:
                from .pf_refresh import refresh
                runtime.executor.pf_queries.refresh = lambda versions, parent, store=runtime.store: refresh(self.server, versions, parent, store=store)
        for store in [*stores.values(), *audits.values(), *(r.usage for r in runtimes.values())]:
            if store is not None:
                store.world_state = lambda: self.server.world_state_id
        for executor in [*(r.executor for r in runtimes.values()), *script_executors.values()]:
            executor.world_status = lambda: self.server._operation_failed or self.server._step_day_timed_out
            executor.advance_active = self.server._advance_lock.locked
            executor.authorize = self.server.authorize_team
        try:
            self.server.start()
            for runtime in self.roles.values():
                runtime.executor.verify_sandbox()
                if not _restored_state:
                    self._initialize_history(runtime)
            if _restored_state:
                from .team_checkpoint import restore_runtime_state
                restore_runtime_state(self, _restored_state)
        except BaseException:
            self.close()
            raise

    def _git(self, runtime, *args, **kwargs):
        return runtime.executor.run_private(['git', '--no-replace-objects', '-c', 'core.hooksPath=/dev/null',
            '-c', 'user.name=CEO Bench', '-c', 'user.email=ceobench@localhost', '-C', str(runtime.workspace), *args],
            capture_output=True, check=True, timeout=30, **kwargs).stdout.decode().strip()

    def _initialize_history(self, runtime):
        self._git(runtime, 'init')
        self._git(runtime, 'add', '--force', '--all', '--', '.', ':(exclude)sessions', ':(exclude)**/__pycache__/**')
        tree = self._git(runtime, 'write-tree')
        commit = self._git(runtime, 'commit-tree', tree, '-m', 'Initial workspace')
        self._git(runtime, 'update-ref', '--no-deref', 'HEAD', commit)

    def _snapshot_history(self, runtime, title):
        self._git(runtime, 'add', '--force', '--all', '--', '.', ':(exclude)sessions', ':(exclude)**/__pycache__/**')
        self._git(runtime, 'commit', '--allow-empty', '-m', title)
        return self._git(runtime, 'rev-parse', 'HEAD')

    @property
    def phase(self):
        return self.server.team_phase

    def _set_phase(self, phase):
        if phase in ('paused', 'stopped') and self.server._step_day_timed_out:
            # A timed-out settlement may still hold the world lock.
            self.server.team_phase = 'paused'
            return
        with ExitStack() as stack:
            if phase in ('analysis', 'asking', 'paused', 'stopped'):
                for executor in (self.roles['ceo'].executor, self.server.role_executors['ceo']):
                    stack.enter_context(executor._execute_lock)
                    if executor.preserved_process and phase not in ('paused', 'stopped'):
                        with self.server._lock:
                            self.server.team_phase = 'paused'
                        raise RuntimeError('CEO process boundary remains open')
            with self.server._lock:
                self.server.team_phase = phase

    def checkpoint(self, destination=None):
        from .team_checkpoint import checkpoint
        return checkpoint(self, destination)

    @classmethod
    def restore(cls, snapshot, root, *, client_factory, public_dir=None, token_counter=None,
                simulator_factory=None, **options):
        from .team_checkpoint import restore
        return restore(cls, snapshot, root, client_factory=client_factory, public_dir=public_dir,
            token_counter=token_counter, simulator_factory=simulator_factory, **options)

    def _boundary(self, name):
        self._safe_point = True
        try:
            if self.checkpoint_callback:
                self.checkpoint_callback(self, name)
        finally:
            self._safe_point = False

    def _begin_operation(self, kind):
        path = self.root / 'private' / 'operations' / (uuid.uuid4().hex + '.json')
        state = dict(kind=kind, day=self.server.tools.current_day, status='running')
        if kind == 'ceo_scripts' or kind.startswith('ceo_tool:'):
            from .team_checkpoint import business_operations
            with self.server._request_lock:
                state.update(wrapper_version=1, business_operations_before=business_operations(self.root),
                    workspace_sha256_before=self._workspace_fingerprint(),
                    mutation_epoch_before=self.server._request_mutations)
        write_json(path, state)
        return path

    def _workspace_fingerprint(self):
        if any(executor.preserved_process for executor in
                (self.roles['ceo'].executor, self.server.role_executors['ceo'])):
            return None
        workspace = self.roles['ceo'].workspace
        if workspace.is_symlink() or any(path.is_symlink() for path in workspace.rglob('*')):
            return None
        return tree_hash(workspace, exact=True)

    @staticmethod
    def _operation_finalization(executor):
        return (dict(execution_status=getattr(executor, 'last_status', None),
            process_boundary=getattr(executor, '_bash_facts', {}).get('process_boundary'),
            preserved_process=bool(executor.preserved_process),
            pf_service_open=bool(getattr(executor, '_pf_service', None))) if executor else None)

    def _finish_operation(self, path, executor=None, *, advance_outcome=None):
        with self._operation_lock:
            return self._settle_operation(path, executor, advance_outcome=advance_outcome)

    def _settle_operation(self, path, executor=None, *, advance_outcome=None):
        state = json.loads(path.read_text())
        finalization = self._operation_finalization(executor)
        if state['status'] in ('completed', 'interrupted', 'failed'):
            if 'finalization' in state and state['finalization'] != finalization:
                raise RuntimeError('Conflicting operation finalization')
            if advance_outcome is not None and state.get('advance') != advance_outcome:
                raise RuntimeError('Conflicting operation finalization')
            return
        if advance_outcome is not None:
            write_json(path, dict(state, status=advance_outcome['status'], advance=advance_outcome))
            return
        if state['kind'] == 'api:advance':
            return
        if state.get('wrapper_version') == 1 and executor:
            state['finalization'] = finalization
            from .team_checkpoint import business_operations, recovery_operation_safe
            subprocess = state['kind'] in ('ceo_tool:bash', 'ceo_scripts')
            status = getattr(executor, 'last_status', None)
            boundary = getattr(executor, '_bash_facts', {}).get('process_boundary') or {}
            not_started = getattr(executor, '_bash_facts', {}).get('process_started') is False
            startup_closed = (state['kind'] == 'ceo_scripts' and boundary.get('phase') == 'closed' and
                boundary.get('coverage') in ('preauthentication_no_target', 'no_target_start') and
                boundary.get('target_started') is False and boundary.get('direct_child_reaped') is True and
                boundary.get('streams_collected') is True and boundary.get('unknown') is None)
            client_closed = (not executor.preserved_process and not getattr(executor, '_pf_service', None) and
                (not subprocess or not_started or startup_closed or (boundary.get('phase') == 'closed' and boundary.get('coverage') == 'subreaper_wait' and
                    boundary.get('descendants_closed') is True and boundary.get('children') == [] and
                    boundary.get('unknown', 'missing') is None and boundary.get('exit_code') is not None)))
            with self.server._request_lock:
                requests = self.server.public_request_state()
                state['recovery'] = dict(
                    classification='pending_read_only' if requests['active'] else
                        'read_only_interrupted' if status == 'result_unknown' else 'read_only',
                    client_closed=client_closed,
                    subprocess=subprocess, process_boundary=boundary,
                    business_operations_after=business_operations(self.root),
                    workspace_sha256_after=self._workspace_fingerprint() if client_closed else None,
                    mutation_epoch_after=requests['mutation_epoch'],
                    pending_requests=requests['active'], queued_requests=requests['queued'],
                    operation_failed=self.server._operation_failed, step_day_timed_out=self.server._step_day_timed_out)
                state['status'] = 'interrupted' if status == 'result_unknown' else 'completed'
                state['execution_status'] = status
                changed = {name: json.loads((path.parent / name).read_text())
                    for name, sha in state['recovery']['business_operations_after'].items()
                    if state['business_operations_before'].get(name) != sha}
                advances = {name: child for name, child in changed.items() if child['kind'] == 'api:advance'}
                if requests['queued'] or any(request.get('read_only') is not True and
                        request.get('outcome_known') is not True for request in requests['active']) or any(
                        child['status'] == 'running' for child in advances.values()) or (advances and requests['active']):
                    state['status'] = 'running'
                    state['recovery']['classification'] = 'pending_business_write'
                    write_json(path, state)
                    if self._pending_operations is not None:
                        self._pending_operations.add(path)
                    raise RunCancelled('business_operation_pending')
                known_advance = (advances and all(child['status'] in ('completed', 'failed') and
                    child.get('advance', {}).get('status') == child['status'] for child in advances.values()) and
                    all(child['status'] == 'completed' or child in advances.values() for child in changed.values()))
                if (known_advance and client_closed and not requests['active'] and
                        (self.server._operation_failed or self.server._step_day_timed_out or status == 'result_unknown')):
                    state['recovery']['classification'] = 'known_advance_failure'
                    state['advance_operations'] = {name: child['advance'] for name, child in advances.items()}
                    write_json(path, dict(state, status='failed'))
                    self._lifecycle.cancel('world_operation_failed')
                    raise RunCancelled(self._lifecycle.reason)
                if (state['kind'] == 'ceo_scripts' and client_closed and not requests['active'] and
                        not requests['queued'] and not self.server._operation_failed and
                        not self.server._step_day_timed_out and
                        state['business_operations_before'] == state['recovery']['business_operations_after'] and
                        state['mutation_epoch_before'] == requests['mutation_epoch'] and
                        isinstance(state['workspace_sha256_before'], str) and
                        state['workspace_sha256_before'] == state['recovery']['workspace_sha256_after']):
                    state['recovery']['classification'] = 'read_only'
                    write_json(self.root / 'private' / 'ceo' / 'script-closures' / path.name, state)
                    path.unlink()
                    return
                if recovery_operation_safe(self.root, state):
                    write_json(path, state)
                    return
                state['recovery']['classification'] = 'unproven'
        if (self.server._operation_failed or self.server._step_day_timed_out or
                executor and (executor.preserved_process or getattr(executor, 'last_status', None) == 'result_unknown')):
            state['status'] = 'running'
            write_json(path, state)
            raise BusinessOutcomeUnknown('Business operation outcome unknown; HOLD for manual checkpoint resume')
        write_json(path, dict(state, status='completed'))

    def _guard_business_operations(self):
        server = self.server
        execute, advance = server.execute_tool, server.advance_week
        def guarded_execute(tool, args, *, role='ceo'):
            if role != 'ceo' or CALL_POLICY.get(tool) == 'read':
                return execute(tool, args, role=role)
            with server._lock:
                server.authorize_team(role, CALL_POLICY.get(tool, 'write'))
                operation = self._begin_operation('api:' + tool)
                try:
                    result = execute(tool, args, role=role)
                except BaseException:
                    server._operation_failed = True
                    server._advance_outcome = None
                    raise
                self._finish_operation(operation)
                return result
        def guarded_advance(*args, role='ceo', **kwargs):
            if role != 'ceo':
                return advance(*args, role=role, **kwargs)
            with server._lock:
                server.authorize_team(role, 'advance')
                if server._operation_failed or server._step_day_timed_out:
                    return advance(*args, role=role, **kwargs)
                operation = self._begin_operation('api:advance')
            settlement = None

            def settled(outcome):
                nonlocal settlement
                settlement = outcome
                self._finish_operation(operation, advance_outcome=outcome)
                self._settle_pending_operations()

            try:
                return advance(*args, role=role, on_settled=settled, **kwargs)
            except Exception as exc:
                if settlement is not None:
                    self._finish_operation(operation, advance_outcome=settlement)
                    raise
                old_day = json.loads(operation.read_text())['day']
                outcome = dict(status='failed', old_day=old_day, target_day=old_day + 7,
                    reached_day=old_day, last_completed_day=old_day, progression='not_advanced',
                    error=type(exc).__name__ + ': ' + str(exc))
                self._finish_operation(operation, advance_outcome=outcome)
                server._operation_failed, server._advance_outcome = True, outcome
                if isinstance(exc, PermissionError):
                    raise
                return dict(success=False, error='advance_week_failed', message=outcome['error'],
                    day=server.tools.current_day, advance=outcome)
            finally:
                self._finish_operation(operation)
        server.execute_tool, server.advance_week = guarded_execute, guarded_advance

    def _settle_pending_operations(self):
        finished = []
        with self._operation_lock:
            if self._pending_operations is None:
                self._pending_operations = {path for path in (self.root / 'private/operations').glob('*.json')
                    if json.loads(path.read_text()).get('recovery', {}).get('classification') == 'pending_business_write'}
            for path in tuple(self._pending_operations):
                state = json.loads(path.read_text())
                if (state.get('status') != 'running' or
                        state.get('recovery', {}).get('classification') != 'pending_business_write'):
                    self._pending_operations.discard(path)
                    continue
                executor = (self.server.role_executors['ceo'] if state['kind'] == 'ceo_scripts' else
                            self.roles['ceo'].executor)
                if state.get('finalization') != self._operation_finalization(executor):
                    continue
                try:
                    self._finish_operation(path, executor)
                except BusinessOutcomeUnknown:
                    pass
                except RunCancelled as exc:
                    if (str(exc) != 'business_operation_pending' and
                            json.loads(path.read_text()).get('status') not in ('completed', 'interrupted', 'failed')):
                        raise
                else:
                    finished.append(state)
                if not path.exists() or json.loads(path.read_text()).get('status') != 'running':
                    self._pending_operations.discard(path)
        return finished

    def request_pause(self):
        self._lifecycle.cancel()

    def _natural_end(self):
        if self.server._operation_failed or self.server._step_day_timed_out:
            raise RunCancelled('world_operation_failed')
        with self.server._lock:
            if self.server._operation_failed or self.server._step_day_timed_out:
                raise RunCancelled('world_operation_failed')
            from .database import get_cash
            return (self.server.tools.current_day >= self.effective_end or
                (self.server.conn is not None and get_cash(self.server.conn) < 0))

    def _check(self):
        if self._natural_end():
            raise RunCancelled('natural_end')
        self._lifecycle.check(retry=True)

    def _check_model_send(self):
        from .runtime_options import timeout_value
        timeout = timeout_value('completion_wait', 30, environment=self.server.runtime_environment)
        deadline = time.monotonic() + timeout if timeout is not None else float('inf')
        while True:
            self._lifecycle.check(retry=True)
            with self.server._request_lock:
                requests = self.server.public_request_state()
                if not requests['queued'] and not any(request.get('read_only') is not True and
                        request.get('outcome_known') is not True for request in requests['active']):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RunCancelled('business_operation_pending')
                self.server._request_lock.wait(min(remaining, .1))
        self._check()

    def _rotate_sessions(self):
        for runtime in self.roles.values():
            identity = replace(runtime.identity, session_id=uuid.uuid4().hex)
            runtime.identity = identity
            for participant in (runtime.agent, runtime.executor, runtime.registry, runtime.usage,
                                runtime.audit, self.server.role_executors[identity.role]):
                participant.identity = identity
            if runtime.store:
                runtime.store.identity.update(identity.fields())
            write_json(self.root / 'private' / identity.role / 'identity.json', identity.fields())
            runtime.agent.reset()

    def _transition(self, message, status, reply=None):
        with self._message_lock:
            if message.status == 'delivered':
                raise RuntimeError('Message was already delivered')
            message.status = status
            if reply is not None:
                message.reply = reply
            with self.message_path.open('a') as stream:
                stream.write(json.dumps(asdict(message), ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())

    def _message(self, role, text):
        if role not in ROLES[1:] or not isinstance(text, str) or not text.strip():
            raise ValueError('ask_analyst requires growth or ops_finance and a nonempty message')
        message = TeamMessage(uuid.uuid4().hex, 'ceo', role, text,
            self.server.tools.current_day, self.server.world_state_id,
            self.roles['ceo'].identity.session_id, self.roles[role].identity.session_id)
        self.messages.append(message)
        self._transition(message, 'queued')
        return message

    def _script_output(self, role):
        operation = None
        if role == 'ceo' and self.server.get_daily_scripts(role):
            operation = self._begin_operation('ceo_scripts')
        try:
            outputs = self.server._run_daily_scripts_internal(role)
        finally:
            if operation is not None:
                self._finish_operation(operation, self.server.role_executors[role])
        classifications = {record['classification'] for record in self.server.role_script_results[role]}
        if 'unknown' in classifications:
            raise RunCancelled('registered_script_outcome_unknown')
        if 'transient_infrastructure' in classifications:
            raise RunCancelled('registered_script_transient_infrastructure')
        text = ''
        for name, output in outputs.items():
            text = join_text(text, '\n' if text else '', join_text(name, '\n', output))
        return text

    def _observation(self, role, dashboard, scripts):
        runtime = self.roles[role]
        day = self.server.tools.current_day
        started = None
        if runtime.registry.mode == 'git':
            runtime.usage.weekly_check = None
            started = runtime.audit.record('git_weekly_check', day=day, status='started')
        try:
            check = runtime.executor.weekly_check(day)
        except BaseException as exc:
            if started:
                runtime.audit.record('git_weekly_check', day=day, status='failed',
                    check_id=started['event_id'], error=type(exc).__name__, detail=str(exc),
                    delivery_state='not_assembled')
            raise
        if started:
            result = runtime.audit.record('git_weekly_check', day=day, status='succeeded',
                check_id=started['event_id'], outcome='summary_ready' if check else 'not_applicable',
                summary=check, delivery_state='not_assembled')
        observation = (join_text(dashboard, '\n\nYour private registered-script output\n', scripts)
                       if scripts else dashboard)
        observation = join_text(observation, '\n\n', check) if check else observation
        if started:
            receipt = dict(check_id=started['event_id'], result_event_id=result['event_id'], day=day,
                outcome=result['outcome'], summary_chars=len(check or ''),
                delivery_state='assembled' if check else 'not_applicable')
            runtime.usage.weekly_check = dict(receipt=receipt, summary=check)
            runtime.audit.record('git_weekly_check_delivery', status='assembled', **receipt)
        return observation

    def _run_analyst(self, role, requests, dashboard=None):
        runtime = self.roles[role]
        scripts = self._script_output(role) if dashboard is not None else None
        observation = self._observation(role, dashboard, scripts) if dashboard is not None else None
        for request in requests:
            try:
                self._check()
                self._transition(request, 'running')
                answer = runtime.agent.act(join_text(observation, '\n\n', request.text) if observation is not None else request.text,
                    0, False, {'day': request.day})
                observation = None
                while not isinstance(answer, FinalText):
                    self._check()
                    if answer is None:
                        raise RuntimeError('Analyst returned no answer')
                    while answer is not None:
                        self._check()
                        result = runtime.executor.execute(answer.tool, answer.arguments,
                            model_request_event=runtime.usage.last_request_event,
                            model_context_id=runtime.usage.context_id)
                        runtime.agent.record_tool_result(result)
                        answer = runtime.agent.next_tool_action()
                    self._check()
                    answer = runtime.agent.act('', 0, False, {'day': request.day})
                self._check()
                request.handoff_commit = self._snapshot_history(runtime, f'Analyst handoff {request.request_id[:7]} (day {request.day})')
                self._transition(request, 'completed', answer.text)
            except BaseException:
                self._transition(request, 'failed')
                raise
        return requests

    def _handoff(self, request):
        header = (f'Analyst handoff | role={request.receiver} | day={request.day} | '
                  f'world_state_id={request.world_state_id} | request_id={request.request_id}\n'
                  f'Workspace: {self.roles[request.receiver].workspace}\n'
                  f'Snapshot commit: {request.handoff_commit}\n')
        return join_text(header, '\n', request.reply)

    def _parallel(self, requests, dashboard=None):
        groups = {role: [r for r in requests if r.receiver == role] for role in ROLES[1:]}
        futures = [self._pool.submit(self._run_analyst, role, group, dashboard)
                   for role, group in groups.items() if group]
        pending = set(futures)
        try:
            while pending:
                self._check()
                completed, pending = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                for future in completed:
                    future.result()
            self._check()
        except BaseException:
            self._lifecycle.cancel('analyst_failure')
            raise
        return requests

    def ask_analyst(self, role, message):
        if self.phase != 'ceo' or self._pool is None:
            raise PermissionError('Only the active CEO can ask an analyst')
        return self._ask_batch([dict(arguments=dict(role=role, message=message))])[0]

    def _ask_batch(self, calls):
        self._set_phase('asking')
        requests, results = [], []
        for call in calls:
            try:
                args = call['arguments']
                request = self._message(args.get('role'), args.get('message'))
            except (ValueError, AttributeError) as exc:
                results.append(f'Error: {exc}')
            else:
                requests.append(request)
                results.append(request)
        self._parallel(requests)
        for request in requests:
            self._transition(request, 'delivered')
        self._set_phase('ceo')
        return [self._handoff(r) if isinstance(r, TeamMessage) else r for r in results]

    def _run_ceo(self, observation):
        runtime = self.roles['ceo']
        day = self.server.tools.current_day
        self._check()
        action = runtime.agent.act('' if self._week.get('ceo_started') else observation,
            0, False, {'day': day})
        self._week['ceo_started'] = True
        while self.server.tools.current_day == day:
            self._check()
            if action is None or isinstance(action, FinalText):
                raise RuntimeError('CEO must advance through the weekly API')
            while action is not None:
                self._check()
                if action.tool == 'ask_analyst':
                    calls = []
                    for call in runtime.agent._pending_tool_calls:
                        if call['name'] != 'ask_analyst':
                            break
                        calls.append(call)
                    for result in self._ask_batch(calls):
                        runtime.agent.record_tool_result(result)
                    if not runtime.agent._pending_tool_calls:
                        self._boundary('ask_completed')
                else:
                    operation = self._begin_operation('ceo_tool:' + action.tool)
                    try:
                        result = runtime.executor.execute(action.tool, action.arguments,
                            model_request_event=runtime.usage.last_request_event,
                            model_context_id=runtime.usage.context_id)
                        runtime.agent.record_tool_result(result)
                    finally:
                        self._finish_operation(operation, runtime.executor)
                if self.server.tools.current_day != day:
                    return
                action = runtime.agent.next_tool_action()
            self._check()
            action = runtime.agent.act('', 0, False, {'day': day})

    def run(self, *, stop_after_day, hold=None):
        if type(stop_after_day) is not int or stop_after_day <= 0 or stop_after_day % 7 or stop_after_day > self.effective_end:
            raise ValueError('stop_after_day must be a positive whole-week boundary within the simulation')
        if self._pool is not None:
            raise RuntimeError('Team run is already active')
        self._lifecycle.hold = Path(hold) if hold and Path(hold).name == 'STOP' else None
        self._set_phase(self._stage)
        try:
            with self._lifecycle, ThreadPoolExecutor(max_workers=2) as pool:
                self._pool = pool
                while True:
                    day = self.server.tools.current_day
                    natural_end = self._natural_end()
                    if natural_end or day >= stop_after_day:
                        self._set_phase('stopped')
                        return RunOutcome(day, 'natural_end' if natural_end else 'observation_end')
                    self._check()
                    if self._stage == 'boundary':
                        self._rotate_sessions()
                        self._stage = 'ceo_scripts'
                        self._set_phase('ceo_scripts')
                        ceo_scripts = self._script_output('ceo')
                        self._check()
                        self._stage = 'analysis'
                        self._set_phase('analysis')
                        dashboard = self.server.public_dashboard()
                        requests = [self._message(role, WEEKLY_TASKS[role])
                                    for role in ROLES[1:]]
                        self._parallel(requests, dashboard)
                        for request in requests:
                            self._transition(request, 'delivered')
                        observation = self._observation('ceo', dashboard, ceo_scripts)
                        for request in requests:
                            observation = join_text(observation, '\n\n', self._handoff(request))
                        self._week = dict(observation=observation, ceo_started=False)
                        self._stage = 'ceo'
                        self._set_phase('ceo')
                        self._boundary('answers_ready')
                    elif self._stage != 'ceo':
                        raise RuntimeError('Team interrupted outside a recoverable boundary')
                    self._run_ceo(self._week['observation'])
                    self._close_week()
                    self._boundary('week_boundary')
        except RunCancelled as exc:
            self.failure = getattr(exc, 'reason', str(exc))
            cause = exc.__cause__
            details = [exc.detail] if getattr(exc, 'detail', None) else []
            while cause is not None:
                details.append(type(cause).__name__ + ': ' + str(cause))
                cause = cause.__cause__
            self.failure_detail = '; '.join(details) or None
            if self.failure == 'registered_script_transient_infrastructure':
                self._stage = 'boundary'
                self._week = {}
            self._set_phase('stopped' if str(exc) == 'natural_end' else 'paused')
            return RunOutcome(self.server.tools.current_day, 'natural_end' if str(exc) == 'natural_end' else 'paused')
        except BusinessOutcomeUnknown as exc:
            self.failure = 'business_operation_unknown'
            self.failure_detail = str(exc)
            self._lifecycle.cancel(self.failure)
            self._set_phase('paused')
            return RunOutcome(self.server.tools.current_day, 'paused')
        except Exception as exc:
            self.failure = f'{type(exc).__name__}: {exc}'
            self._lifecycle.cancel('runtime_failure')
            self._set_phase('paused')
            return RunOutcome(self.server.tools.current_day, 'failed')
        finally:
            self._pool = None

    def _close_week(self):
        while self.roles['ceo'].agent._pending_tool_calls:
            self.roles['ceo'].agent.record_tool_result('Cancelled because this tool batch crossed a week boundary.')
        closed_day = self.server.tools.current_day
        for runtime in self.roles.values():
            from .registration_evidence import week_commit_subject
            self._snapshot_history(runtime, week_commit_subject(f'week-{closed_day // 7}'))
        self._stage = 'boundary'
        self._week = {}
        self._set_phase('boundary')

    def close(self):
        self.server.stop()
        requests = self.server.public_request_state()
        if not requests['active'] and not requests['queued']:
            for runtime in self.roles.values():
                runtime.executor.api_socket = None
            for executor in self.server.role_executors.values():
                executor.api_socket = None
            for state in self._settle_pending_operations():
                if self._stage == 'ceo' and state['day'] < self.server.tools.current_day:
                    self._close_week()
        self._sockets.cleanup()
