from dataclasses import asdict, dataclass, field, replace
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
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
from .role_policy import AgentIdentity, CALL_POLICY, READABLE_ROLES, ROLES, RoleAudit, require_call, require_advance
from .run_state import tree_hash, write_json
from .run_lifecycle import BusinessOutcomeUnknown, RunCancelled, WorkerLifecycle
from .sql_evidence import FORMAT, SQLEvidenceStore
from .text_registry import TextRegistry
from .team_prompts import CEO_OPENING, CEO_CLOSING, WEEKLY_TASKS, system_prompt


@dataclass(frozen=True)
class RoleWeek:
    day: int
    session_id: str
    initialized: bool = False
    scripts_status: str = 'pending'
    scripts_output: str = ''
    scripts_error: str | None = None
    weekly_status: str = 'pending'
    weekly_request_id: str | None = None

    def snapshot(self):
        return dict(vars(self))


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
    week: RoleWeek | None = None


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
    kind: str = 'weekly_task'
    enqueue_sequence: int | None = None
    input_accepted: bool = False
    received_request_event: str | None = None


@dataclass
class RoleWorker:
    jobs: deque = field(default_factory=deque)
    inbox: deque = field(default_factory=deque)
    job: Future | None = None
    question: TeamMessage | None = None
    waiting: TeamMessage | None = None
    thread_id: int | None = None


class MultiAgentRuntime:
    def __init__(self, root, tools, *, client_factory, mode='git', conn=None,
                 simulator=None, public_dir=None, run_id=None, world_id=None, public_refs_path=None,
                 model="deepseek-v4.1-flash", reasoning_effort="high", total_days=500, token_counter=None,
                 model_pricing=None, _restored_state=None, analyst_bash_timeout=1200,
                 analyst_bash_warn_after=None, ceo_bash_timeout=1200, ceo_bash_warn_after=None, capabilities=None, **server_options):
        if mode not in ('git', 'pf'):
            raise ValueError('Team supports git and pf modes')
        if type(total_days) is not int or total_days < 7:
            raise ValueError('total_days must include at least one full week')
        self.total_days = total_days
        self.configuration = dict(mode=mode, model=model, reasoning_effort=reasoning_effort,
            total_days=total_days, model_pricing=model_pricing)
        from .agent_capabilities import AgentCapabilities
        self.capabilities = AgentCapabilities.resolve(capabilities)
        if capabilities is not None:
            self.configuration['capabilities'] = self.capabilities.to_dict()
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
        self._team_condition = threading.Condition()
        self._workers = {role: RoleWorker() for role in ROLES}
        self._workers_stopping = False
        self._worker_error = None
        self._advancing = False
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
        from urllib.parse import quote
        from .public_refs import PublicRefs
        saved_refs = (_restored_state or {}).get('public_refs')
        refs_root = Path(_restored_state['source_root']) if _restored_state else self.root
        refs_path = (saved_refs['path'] if saved_refs else public_refs_path or
            refs_root.parent / '.public-refs' / (quote(run_id, safe='') + '.sqlite'))
        current_restore = bool(_restored_state and not _restored_state.get('_legacy_history_unavailable'))
        if current_restore and not saved_refs:
            refs_path = self.root / 'private' / ('missing-public-refs-' + uuid.uuid4().hex + '.sqlite')
        self.public_refs = PublicRefs(refs_path,
            snapshot=self.root / (saved_refs['snapshot'] if saved_refs else 'private/public-refs.sqlite') if _restored_state else None,
            require_existing=current_restore)
        from .runtime_temp import prepare_tmpdir
        self._temp_root = prepare_tmpdir(self.root / 'tmp')
        self._sockets = tempfile.TemporaryDirectory(prefix='ceobench-team-', dir=self._temp_root)
        public = Path(public_dir or Path(__file__).resolve().parents[2] / 'public')
        self._public_dir = public
        workspaces = {role: self.root / 'roles' / role for role in ROLES}
        sockets = {role: Path(self._sockets.name) / role / 'api.sock' for role in ROLES}
        script_sockets = {role: Path(self._sockets.name) / role / 'script.sock' for role in ROLES}
        stores, runtimes, script_executors, audits = {}, {}, {}, {}
        for role in ROLES:
            workspace = workspaces[role]
            workspace.mkdir(parents=True, exist_ok=bool(_restored_state))
            if not _restored_state:
                shutil.copytree(public / 'docs', workspace / 'docs')
                shutil.copy2(public / 'novamind-client', workspace / 'novamind-operation')
                (workspace / 'MEMORY.md').write_text('')
            from .workspace_guides import write_guides
            write_guides(workspace, mode, self.capabilities)
            identity = (AgentIdentity(**_restored_state['roles'][role]['identity']) if _restored_state else
                AgentIdentity(run_id, world_id, role, uuid.uuid4().hex, uuid.uuid4().hex))
            role_private = private / role
            role_private.mkdir(mode=0o700, exist_ok=bool(_restored_state))
            write_json(role_private / 'identity.json', identity.fields())
            store = stores[role] = SQLEvidenceStore(private / 'evidence.sqlite', dict(
                identity.fields(), branch_id=role, data_source_id=world_id, format=FORMAT,
                capture_scope='execution')) if mode == 'pf' else None
            if store is not None:
                store.public_refs = self.public_refs
            audit = audits[role] = RoleAudit(role_private / 'audit.jsonl', identity)
            registry = TextRegistry(workspace, mode, store, sim_day=lambda: tools.current_day,
                identity=identity, private_root=role_private, capabilities=self.capabilities)
            registry.public_refs = self.public_refs
            env = dict(NOVAMIND_API_PORT='1', NOVAMIND_API_SOCKET='/run/novamind-api/api.sock',
                       PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(workspace / 'docs'))
            executor_options = dict(workspace_path=workspace, identity=identity, env=env,
                require_sandbox=True, evidence_store=store, audit=audit, api_socket=sockets[role], private_root=role_private,
                public_refs=self.public_refs,
                readable_workspaces=tuple(workspaces[r] for r in READABLE_ROLES[role]))
            settings = (dict(bash_timeout=analyst_bash_timeout, bash_warn_after=analyst_bash_warn_after) if role != 'ceo'
                        else dict(bash_timeout=ceo_bash_timeout, bash_warn_after=ceo_bash_warn_after))
            script_settings = {}
            if _restored_state:
                settings = _restored_state['roles'][role]['executor']
                script_settings = _restored_state['roles'][role]['script_executor']
            executor = BashAgentToolExecutor(**executor_options, text_registry=registry, capabilities=self.capabilities, **settings)
            script_executors[role] = BashAgentToolExecutor(**dict(executor_options, env=dict(env), api_socket=script_sockets[role]), **script_settings)
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
                ask_analyst=role == 'ceo', team=True, bash_timeout=executor.bash_timeout,
                bash_warn_after=executor.bash_warn_after, capabilities=self.capabilities),
                client, workspace_path=workspace, usage_recorder=usage,
                model=model, reasoning_effort=reasoning_effort, total_days=total_days,
                text_registration=True, pf=mode == 'pf', identity=identity, capabilities=self.capabilities,
                allow_final_text=role != 'ceo', **generation_policy)
            agent.system_prompt = system_prompt(agent.system_prompt, role, mode, total_days,
                self.effective_end, workspace, [workspaces[r] for r in READABLE_ROLES[role]], self.capabilities)
            agent.lifecycle = self._lifecycle
            registry.git_run = executor.run_private
            agent.git_run = executor.run_private
            agent._snapshot_path = role_private / 'conversation.json'
            runtimes[role] = RoleRuntime(identity, workspace, agent, executor, registry, usage, store, audit)
        self.roles = MappingProxyType(runtimes)
        self.server = NovaMindAPIServer(tools, simulator=simulator, conn=conn,
            script_workspace=workspaces['ceo'], require_sandbox=True, sql_evidence=stores['ceo'],
            public_refs=self.public_refs,
            role_sockets=sockets, script_role_sockets=script_sockets, role_stores=stores, role_executors=script_executors, role_audits=audits, **server_options)
        if _restored_state:
            from .business_ledger import migrate_operation_refs
            migrate_operation_refs(self.server.conn, self.public_refs)
        def operation_lookup(reference):
            from .business_ledger import read_operation
            from .relation_registry import Rejected
            deadline = time.monotonic() + self.server.QUERY_TIMEOUT_SECONDS
            while True:
                self._lifecycle.check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise Rejected('operation_unavailable', 'The public ledger is busy; retry the reference later')
                if self.server._lock.acquire(timeout=min(.1, remaining)):
                    break
            try:
                self._lifecycle.check()
                return read_operation(self.server.conn, reference)
            finally:
                self.server._lock.release()

        attempt_id = uuid.uuid4().hex
        for runtime in self.roles.values():
            runtime.registry.relations.operation_lookup = operation_lookup
            runtime.registry.relations.attempt_id = attempt_id
        self.server.team_permission = self._team_permission
        self.server.script_check = self._check_model_send
        for role, runtime in self.roles.items():
            runtime.usage.request_check = self._check_model_send
            runtime.agent.before_request_callback = lambda role=role: self._intake_questions(role)
            runtime.agent.response_callback = lambda role=role, **facts: self._record_question_receipts(role)
        self._guard_business_operations()
        self.server.request_finished_callback = self._settle_pending_operations
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
                if not _restored_state or _restored_state.get('_legacy_history_unavailable'):
                    self._initialize_history(runtime)
            if _restored_state:
                from .team_checkpoint import restore_runtime_state
                restore_runtime_state(self, _restored_state)
        except BaseException:
            try:
                self.close()
            finally:
                for runtime in self.roles.values():
                    runtime.agent.client.close()
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

    def _begin_operation(self, kind, *, role='ceo'):
        path = self.root / 'private' / 'operations' / (uuid.uuid4().hex + '.json')
        state = dict(kind=kind, role=role, day=self.server.tools.current_day, status='running')
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
        def guarded_execute(tool, args, *, role='ceo', initiator='host'):
            require_call(role, tool, initiator=initiator)
            if CALL_POLICY[tool] == 'read':
                return execute(tool, args, role=role, initiator=initiator)
            with server._lock:
                server.authorize_team(role, CALL_POLICY[tool])
                operation = self._begin_operation('api:' + tool, role=role)
                try:
                    result = execute(tool, args, role=role, initiator=initiator)
                except BaseException:
                    server._operation_failed = True
                    server._advance_outcome = None
                    raise
                self._finish_operation(operation)
                return result
        def guarded_advance(*args, role='ceo', initiator='host', **kwargs):
            require_advance(role, initiator=initiator)
            with server._lock:
                server.authorize_team(role, 'advance')
                if server._operation_failed or server._step_day_timed_out:
                    return advance(*args, role=role, initiator=initiator, **kwargs)
                with self._team_condition:
                    if self._questions_pending():
                        raise PermissionError('Team requests are pending. Finish this tool batch, answer them, then retry next-week.')
                    self._advancing = True
                operation = self._begin_operation('api:advance')
            settlement = None

            def settled(outcome):
                nonlocal settlement
                settlement = outcome
                self._finish_operation(operation, advance_outcome=outcome)
                self._settle_pending_operations()

            try:
                result = advance(*args, role=role, initiator=initiator, on_settled=settled, **kwargs)
                if (result.get('error') == 'prediction_save_failed' and settlement is not None
                        and settlement.get('progression') == 'not_advanced'
                        and not server._operation_failed and not server._step_day_timed_out):
                    with self._team_condition:
                        self._advancing = False
                        self._team_condition.notify_all()
                return result
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

    def _check_reference_history(self):
        from .public_refs import PublicReferenceUnavailable
        try:
            self.public_refs.require_available()
        except PublicReferenceUnavailable as exc:
            raise RunCancelled('public_reference_history_unavailable', str(exc)) from None

    def _check_model_send(self):
        self._check_reference_history()
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
            runtime.week = RoleWeek(self.server.tools.current_day, identity.session_id)
            self._update_role_week(identity.role)

    def _update_role_week(self, role, **changes):
        runtime = self.roles[role]
        runtime.week = replace(runtime.week, **changes)
        write_json(self.root / 'private' / role / 'week-state.json', runtime.week.snapshot())

    def _role_input(self, role, observation, *, day):
        runtime = self.roles[role]
        runtime.agent.add_input(observation, day=day)
        if runtime.week is not None and not runtime.week.initialized:
            runtime.week = replace(runtime.week, initialized=True)

    def _transition(self, message, status, reply=None):
        with self._message_lock:
            if message.status in ('delivered', 'rejected', 'cancelled', 'failed'):
                if status == message.status and (reply is None or reply == message.reply):
                    return
                raise RuntimeError('Conflicting terminal message result')
            record = replace(message, status=status, reply=message.reply if reply is None else reply)
            with self.message_path.open('a') as stream:
                stream.write(json.dumps(asdict(record), ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            message.status, message.reply = record.status, record.reply

    def _message(self, role, text, *, sender='ceo', kind='weekly_task'):
        if role not in ROLES or not isinstance(text, str) or not text.strip():
            raise ValueError('A request needs a valid role and a nonempty message')
        message = TeamMessage(uuid.uuid4().hex, sender, role, text,
            self.server.tools.current_day, self.server.world_state_id,
            self.roles[sender].identity.session_id, self.roles[role].identity.session_id, kind=kind)
        with self._message_lock:
            self.messages.append(message)
        if kind == 'weekly_task':
            self._transition(message, 'queued')
        return message

    def _request_ref(self, request):
        number = self.public_refs.ordinal('team_request', request.sender, request.request_id)
        return f'{request.sender}:request{number}'

    def _team_permission(self, role, operation):
        with self._team_condition:
            worker = self._workers[role]
            if worker.waiting is not None:
                raise PermissionError(f'{role} is waiting for a team reply')
            if worker.question is not None:
                if operation == 'advance':
                    raise PermissionError('Answer the team request without advancing the week')
                return True
            if operation == 'advance' and self._questions_pending():
                raise PermissionError('Team requests are pending. Finish this tool batch, answer them, then retry next-week.')
        return False

    def _questions_pending(self):
        return any(worker.inbox or worker.question or worker.waiting for worker in self._workers.values())

    def _request(self, sender, receiver, text):
        if receiver not in ROLES or not isinstance(text, str) or not text.strip():
            return 'Error: A request needs a valid role and a nonempty message.'
        message = self._message(receiver, text, sender=sender, kind='question')
        refusal = None
        with self._team_condition:
            owner = self._workers[sender]
            if owner.thread_id != threading.get_ident():
                raise RuntimeError('Only the role owner may send a request')
            if self._advancing or self._workers_stopping:
                refusal = 'This week is closing; the request was not accepted.'
            else:
                target = receiver
                while True:
                    if target == sender:
                        refusal = 'Circular wait: this request was refused. Continue or ask after the other request finishes.'
                        break
                    waiting = self._workers[target].waiting
                    if waiting is None or waiting.status in ('completed', 'delivered'):
                        break
                    target = waiting.receiver
            if refusal is None:
                message.status = 'admitting'
                message.enqueue_sequence = 1 + max((m.enqueue_sequence or 0 for m in self.messages), default=0)
                owner.waiting = message
                self._workers[receiver].inbox.append(message)
        if refusal is not None:
            self._transition(message, 'rejected', refusal)
            return f'Error: {refusal} request_id={self._request_ref(message)}'
        self._transition(message, 'queued')
        with self._team_condition:
            self._team_condition.notify_all()
        while True:
            self._lifecycle.check(retry=True)
            with self._team_condition:
                if self._worker_error is not None:
                    raise self._worker_error
                if message.status == 'completed':
                    return message
                self._team_condition.wait(.1)

    def _submit_role(self, role, function, *args, **kwargs):
        result = Future()
        with self._team_condition:
            self._workers[role].jobs.append((result, function, args, kwargs))
            self._team_condition.notify_all()
        return result

    def _wait_roles(self, futures):
        while not all(future.done() for future in futures):
            with self._team_condition:
                error = self._worker_error
            if error is not None:
                raise error
            self._lifecycle.check(retry=True)
            wait(futures, timeout=.1)
        if self._worker_error is not None:
            raise self._worker_error
        for future in futures:
            future.result()

    def _role_loop(self, role):
        worker = self._workers[role]
        worker.thread_id = threading.get_ident()
        try:
            while True:
                with self._team_condition:
                    while not self._workers_stopping and (
                            worker.inbox and worker.inbox[0].status == 'admitting' or
                            not worker.jobs and not worker.inbox):
                        self._team_condition.wait(.1)
                        self._lifecycle.check(retry=True)
                    if self._workers_stopping:
                        return
                    if worker.inbox:
                        job = None
                    else:
                        job = worker.jobs.popleft()
                        worker.job = job[0]
                if job is None:
                    self._service_inbox(role)
                else:
                    future, function, args, kwargs = job
                    try:
                        result = function(*args, **kwargs)
                    except BaseException as exc:
                        future.set_exception(exc)
                        raise
                    else:
                        with self._team_condition:
                            worker.job = None
                        future.set_result(result)
        except BaseException as exc:
            with self._team_condition:
                self._worker_error = self._worker_error or exc
                self._team_condition.notify_all()
            self._lifecycle.cancel(getattr(exc, 'reason', 'role_failure'))
        finally:
            with self._team_condition:
                worker.job = None
                self._team_condition.notify_all()

    def _accept_question(self, request, *, queued=False):
        active = self._workers[request.receiver].question
        if queued:
            instruction = (f'Queued after active request_id={self._request_ref(active)}. Continue answering that active request. '
                           if active else 'Queued for later. Continue your current weekly task. ')
            instruction += 'The host will activate this queued request by ID after earlier work finishes.'
        else:
            instruction = ('Answer this request in final text. You may use your tools. '
                           'This answers only the request; the host will resume your prior task afterward.')
        self._role_input(request.receiver,
            f'Team request | request_id={self._request_ref(request)} | from={request.sender} | '
            f'to={request.receiver} | day={request.day}\n{request.text}\n{instruction}', day=request.day)
        request.input_accepted = True
        self._transition(request, 'queued' if queued else 'running')
        self._update_role_week(request.receiver)

    def _intake_questions(self, role):
        if self._pool is None:
            return
        worker = self._workers[role]
        if worker.thread_id != threading.get_ident():
            raise RuntimeError('Only the role owner may accept request input')
        with self._team_condition:
            inbox = list(worker.inbox)
            while any(request.status == 'admitting' for request in inbox):
                self._lifecycle.check(retry=True)
                self._team_condition.wait(.1)
        for request in inbox:
            if not request.input_accepted:
                self._accept_question(request, queued=True)

    def _record_question_receipts(self, role):
        usage = self.roles[role].usage
        for request in self.messages:
            if (request.kind == 'question' and request.receiver == role and request.input_accepted
                    and request.received_request_event is None and request.status in ('queued', 'running')):
                request.received_request_event = (usage.last_request_event or
                    ('http:' + usage.last_http_attempt_id if usage.last_http_attempt_id else 'call:' + usage.last_call_id))
                try:
                    self._transition(request, request.status)
                except Exception as exc:
                    raise RunCancelled('team_message_evidence_failed', str(exc)) from exc

    def _service_inbox(self, role, *, resume=None):
        worker, runtime = self._workers[role], self.roles[role]
        handled = False
        while True:
            with self._team_condition:
                while worker.inbox and worker.inbox[0].status == 'admitting':
                    self._lifecycle.check(retry=True)
                    self._team_condition.wait(.1)
                if not worker.inbox:
                    break
                request = worker.inbox.popleft()
                worker.question = request
            try:
                self._check()
                if request.input_accepted:
                    self._role_input(role, f'Now answer queued request_id={self._request_ref(request)} in final text. '
                        'This answers only that request; the host will resume your prior task afterward.', day=request.day)
                    self._transition(request, 'running')
                else:
                    self._accept_question(request)
                answer = self._drive_role(role, completion='answer', question=request)
                request.handoff_commit = self._snapshot_history(runtime, f'Team reply {self._request_ref(request)} (day {request.day})')
                self._transition(request, 'completed', answer)
                handled = True
            except BaseException:
                self._transition(request, 'failed')
                raise
            finally:
                with self._team_condition:
                    worker.question = None
                    self._team_condition.notify_all()
        if handled and resume:
            self._role_input(role, 'The requested replies are complete. Resume ' + resume + '.',
                day=self.server.tools.current_day)
        return handled

    def _stop_role_workers(self):
        with self._team_condition:
            self._workers_stopping = True
            self._team_condition.notify_all()
        if self._worker_error is not None:
            self._lifecycle.cancel(getattr(self._worker_error, 'reason', 'role_failure'))

    def _automatic_scripts(self, role):
        week = self.roles[role].week
        if week is None or (week.day, week.session_id) != (
                self.server.tools.current_day, self.roles[role].identity.session_id):
            raise RuntimeError('Automatic scripts require the current role week')
        if week.scripts_status in ('succeeded', 'script_error'):
            return week.scripts_output
        if week.scripts_status != 'pending':
            raise RuntimeError('Automatic scripts already started; restore a prior checkpoint')
        expected = 'ceo_scripts' if role == 'ceo' else 'analysis'
        if self._stage != expected or 'directions' not in self._week:
            raise RuntimeError('Automatic scripts must follow CEO opening')
        self._update_role_week(role, scripts_status='running')
        try:
            output = self._script_output(role)
            classifications = {record['classification'] for record in self.server.role_script_results[role]}
            status = 'script_error' if 'script_error' in classifications else 'succeeded'
            self._update_role_week(role, scripts_status=status, scripts_output=output)
            return output
        except BaseException as exc:
            classifications = {record['classification'] for record in self.server.role_script_results[role]}
            status = 'unknown' if 'unknown' in classifications else 'interrupted'
            self._update_role_week(role, scripts_status=status, scripts_error=str(exc))
            raise

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
        text = ''
        for name, output in outputs.items():
            text = join_text(text, '\n' if text else '', join_text(name, '\n', output))
        return text

    def _observation(self, role, dashboard, scripts):
        return (join_text(dashboard, '\n\nYour private registered-script output\n', scripts)
                if scripts else dashboard)

    def _run_analyst(self, role, requests, dashboard=None):
        runtime = self.roles[role]
        scripts = self._automatic_scripts(role) if dashboard is not None else None
        observation = self._observation(role, dashboard, scripts) if dashboard is not None else None
        for request in requests:
            try:
                self._check()
                if request.kind == 'weekly_task':
                    self._update_role_week(role, weekly_status='in_progress', weekly_request_id=request.request_id)
                self._role_input(role, join_text(observation, '\n\n', request.text) if observation is not None else request.text,
                    day=request.day)
                request.input_accepted = True
                self._transition(request, 'running')
                self._update_role_week(role)
                answer = self._drive_role(role, completion='answer')
                observation = None
                request.handoff_commit = self._snapshot_history(runtime, f'Analyst handoff {self._request_ref(request)} (day {request.day})')
                self._transition(request, 'completed', answer)
                if request.kind == 'weekly_task':
                    self._update_role_week(role, weekly_status='completed')
            except BaseException:
                self._transition(request, 'failed')
                if request.kind == 'weekly_task':
                    self._update_role_week(role, weekly_status='interrupted')
                raise
        return requests

    def _handoff(self, request):
        runtime = self.roles[request.receiver]
        commit = runtime.executor.run_private(
            ['git', '-C', str(runtime.workspace), 'rev-parse', '--short=7', request.handoff_commit],
            capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        header = (f'Analyst handoff | role={request.receiver} | day={request.day} | '
                  f'request_id={self._request_ref(request)}\n'
                  f'Workspace: {runtime.workspace}\n'
                  f'Snapshot commit: {commit}\n')
        if request.kind == 'question':
            header = (f'Team reply | from={request.receiver} | to={request.sender} | day={request.day} | '
                      f'request_id={self._request_ref(request)}\n'
                      f'Workspace: {runtime.workspace}\n'
                      f'Snapshot commit: {commit}\n')
        return join_text(header, '\n', request.reply)

    def _parallel(self, requests, dashboard=None):
        groups = {role: [r for r in requests if r.receiver == role] for role in ROLES[1:]}
        futures = [self._submit_role(role, self._run_analyst, role, group, dashboard)
                   for role, group in groups.items() if group]
        self._wait_roles(futures)
        while True:
            self._check()
            with self._team_condition:
                if not self._questions_pending():
                    break
                self._team_condition.wait(.1)
        return requests

    def ask_analyst(self, role, message):
        if role not in ROLES[1:]:
            return 'Error: ask_analyst requires growth or ops_finance.'
        return self._request('ceo', role, message)

    def _run_ceo(self, *, completion='advance'):
        self._check()
        self._update_role_week('ceo', weekly_status='in_progress')
        try:
            if not self._week.get('ceo_started'):
                self._role_input('ceo', self._week['observation'], day=self.server.tools.current_day)
                self._week['ceo_started'] = True
                self._update_role_week('ceo')
            return self._drive_role('ceo', completion=completion)
        except BaseException:
            self._update_role_week('ceo', weekly_status='interrupted')
            raise

    def _drive_role(self, role, *, completion, question=None):
        runtime = self.roles[role]
        previous_policy = runtime.agent.allow_final_text
        runtime.agent.allow_final_text = completion == 'answer'
        day = self.server.tools.current_day
        try:
            while self.server.tools.current_day == day:
                self._check()
                if question is None:
                    self._service_inbox(role, resume='your current weekly task')
                action = runtime.agent.act('', 0, False, {'day': day})
                if role == 'ceo' and question is None:
                    for request in self.messages:
                        if request.status == 'completed' and request.request_id in self._week.get('weekly_request_ids', []):
                            self._transition(request, 'delivered')
                if isinstance(action, FinalText) and completion == 'answer':
                    return action.text
                if action is None or isinstance(action, FinalText):
                    raise RuntimeError('CEO must advance through the weekly API' if role == 'ceo' else 'Analyst returned no answer')
                asked = False
                while action is not None:
                    self._check()
                    if action.tool in ('ask_agent', 'ask_analyst'):
                        args = action.arguments if isinstance(action.arguments, dict) else {}
                        if action.tool == 'ask_analyst' and role != 'ceo':
                            result = 'Error: Only the CEO can use ask_analyst. Use ask_agent.'
                        elif action.tool == 'ask_analyst':
                            result = self.ask_analyst(args.get('role'), args.get('message'))
                        else:
                            result = self._request(role, args.get('role'), args.get('message'))
                        runtime.agent.record_tool_result(self._handoff(result) if isinstance(result, TeamMessage) else result)
                        if isinstance(result, TeamMessage):
                            self._transition(result, 'delivered')
                            with self._team_condition:
                                self._workers[role].waiting = None
                                self._team_condition.notify_all()
                        asked = True
                    else:
                        operation = self._begin_operation('ceo_tool:' + action.tool) if role == 'ceo' else None
                        try:
                            result = runtime.executor.execute(action.tool, action.arguments,
                                model_request_event=runtime.usage.last_request_event,
                                model_context_id=runtime.usage.context_id,
                                tool_call_id=runtime.agent._pending_tool_calls[0]['id'] if runtime.agent._pending_tool_calls else None)
                            runtime.agent.record_tool_result(result)
                        finally:
                            if operation is not None:
                                self._finish_operation(operation, runtime.executor)
                    if self.server.tools.current_day != day:
                        return
                    action = runtime.agent.next_tool_action()
                if asked and role == 'ceo' and question is None and self._stage in ('ceo_opening', 'ceo'):
                    with self._team_condition:
                        safe = not self._questions_pending() and all(
                            worker.job is None for name, worker in self._workers.items() if name != 'ceo')
                    if safe:
                        self._boundary('ask_completed')
        finally:
            runtime.agent.allow_final_text = previous_policy

    def run(self, *, stop_after_day, hold=None):
        if type(stop_after_day) is not int or stop_after_day <= 0 or stop_after_day % 7 or stop_after_day > self.effective_end:
            raise ValueError('stop_after_day must be a positive whole-week boundary within the simulation')
        if self._pool is not None:
            raise RuntimeError('Team run is already active')
        self._lifecycle.hold = Path(hold) if hold and Path(hold).name == 'STOP' else None
        self._set_phase(self._stage)
        try:
            self._check_reference_history()
            with self._lifecycle, ThreadPoolExecutor(max_workers=3, thread_name_prefix="team-role") as pool:
                self._pool = pool
                self._workers_stopping = False
                self._worker_error = None
                workers = [pool.submit(self._role_loop, role) for role in ROLES]
                try:
                    return self._run_weeks(stop_after_day)
                except BaseException as exc:
                    self._lifecycle.cancel(getattr(exc, 'reason', 'runtime_failure'))
                    raise
                finally:
                    if self._worker_error is not None or self._lifecycle.reason:
                        self._lifecycle.cancel(getattr(self._worker_error, 'reason', 'role_failure'))
                    self._stop_role_workers()
                    wait(workers)
                    for message in self.messages:
                        if message.kind == 'question' and message.status in ('admitting', 'queued', 'running'):
                            self._transition(message, 'cancelled', self._lifecycle.reason or 'Run ended')
        except RunCancelled as exc:
            self.failure = getattr(exc, 'reason', str(exc))
            cause = exc.__cause__
            details = [exc.detail] if getattr(exc, 'detail', None) else []
            while cause is not None:
                details.append(type(cause).__name__ + ': ' + str(cause))
                cause = cause.__cause__
            self.failure_detail = '; '.join(details) or None
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

    def _run_weeks(self, stop_after_day):
        while True:
            day = self.server.tools.current_day
            natural_end = self._natural_end()
            if natural_end or day >= stop_after_day:
                self._set_phase('stopped')
                return RunOutcome(day, 'natural_end' if natural_end else 'observation_end')
            self._check()
            if self._stage == 'boundary':
                with self._team_condition:
                    self._advancing = False
                self._rotate_sessions()
                self._week = dict(observation=join_text(self.server.public_dashboard(), '\n\n', CEO_OPENING),
                    ceo_started=False)
                self._stage = 'ceo_opening'
                self._set_phase('ceo_opening')
            if self._stage == 'ceo_opening':
                opening = self._submit_role('ceo', self._run_ceo, completion='answer')
                self._wait_roles([opening])
                directions = opening.result()
                self._week['directions'] = directions
                self._stage = 'ceo_scripts'
                self._set_phase('ceo_scripts')
                scripts = self._submit_role('ceo', self._automatic_scripts, 'ceo')
                self._wait_roles([scripts])
                ceo_scripts = scripts.result()
                self._check()
                self._stage = 'analysis'
                self._set_phase('analysis')
                dashboard = self.server.public_dashboard()
                requests = [self._message(role, join_text(WEEKLY_TASKS[role], '\n\nCEO directions\n', directions))
                            for role in ROLES[1:]]
                self._parallel(requests, dashboard)
                observation = join_text(self.server.public_dashboard(), '\n\n', CEO_CLOSING)
                observation = self._observation('ceo', observation, ceo_scripts)
                for request in requests:
                    observation = join_text(observation, '\n\n', self._handoff(request))
                self._week.update(observation=observation, ceo_started=False,
                    weekly_request_ids=[request.request_id for request in requests])
                self._stage = 'ceo'
                self._set_phase('ceo')
                self._boundary('answers_ready')
            elif self._stage != 'ceo':
                raise RuntimeError('Team interrupted outside a recoverable boundary')
            self._wait_roles([self._submit_role('ceo', self._run_ceo)])
            self._close_week()
            self._boundary('week_boundary')

    def _close_week(self):
        while self.roles['ceo'].agent._pending_tool_calls:
            self.roles['ceo'].agent.record_tool_result('Cancelled because this tool batch crossed a week boundary.')
        closed_day = self.server.tools.current_day
        for runtime in self.roles.values():
            from .registration_evidence import week_commit_subject
            self._snapshot_history(runtime, week_commit_subject(f'week-{closed_day // 7}'))
        if self.roles['ceo'].week is not None:
            self._update_role_week('ceo', weekly_status='completed')
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
