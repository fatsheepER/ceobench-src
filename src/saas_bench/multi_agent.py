from dataclasses import asdict, dataclass, replace
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
import json
import os
from pathlib import Path
import threading
import shutil
import tempfile
from types import MappingProxyType
import uuid

from .agents.bash_agent.agent import BashAgent, FinalText
from .agents.bash_agent.tools import BashAgentToolExecutor, get_bash_agent_tool_descriptions
from .api_server import NovaMindAPIServer
from .execution_capture import join_text
from .model_usage import ModelUsage
from .role_policy import AgentIdentity, CALL_POLICY, READABLE_ROLES, ROLES, RoleAudit
from .run_state import write_json
from .run_lifecycle import RunCancelled, WorkerLifecycle
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
                 model_pricing=None, _restored_state=None, **server_options):
        if mode not in ('git', 'pf'):
            raise ValueError('Team supports git and pf modes')
        if type(total_days) is not int or total_days < 7:
            raise ValueError('total_days must include at least one full week')
        self.total_days = total_days
        self.configuration = dict(mode=mode, model=model, reasoning_effort=reasoning_effort,
            total_days=total_days, model_pricing=model_pricing)
        if mode == 'pf' and token_counter is None:
            from .payload_tokens import load_counter
            token_counter = load_counter('opencode', model)
        self.effective_end = total_days // 7 * 7
        self._lifecycle = WorkerLifecycle()
        self._message_lock = threading.Lock()
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
                raise ValueError('Restored roles disagree on run or world identity')
            if run_id not in (None, saved_run) or world_id not in (None, saved_world):
                raise ValueError('Recovery run or world identity changed')
            run_id, world_id = saved_run, saved_world
        else:
            run_id, world_id = run_id or uuid.uuid4().hex, world_id or uuid.uuid4().hex
        self._sockets = tempfile.TemporaryDirectory(prefix='ceobench-team-')
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
                       PYTHONHASHSEED='0', PYTHONPATH=str(workspace / 'docs'))
            executor_options = dict(workspace_path=workspace, identity=identity, env=env,
                require_sandbox=True, evidence_store=store, audit=audit, api_socket=sockets[role],
                readable_workspaces=tuple(workspaces[r] for r in READABLE_ROLES[role]))
            executor = BashAgentToolExecutor(**executor_options, text_registry=registry)
            script_executors[role] = BashAgentToolExecutor(**dict(executor_options, env=dict(env)))
            usage = ModelUsage(role_private / 'model-usage.jsonl', role,
                pricing=(model_pricing or {}).get('rates', {}), evidence_store=store, identity=identity, token_counter=token_counter)
            agent = BashAgent(get_bash_agent_tool_descriptions(text_registration=True, pf_queries=mode == 'pf', ask_analyst=role == 'ceo', team=True),
                client_factory(role), workspace_path=workspace, usage_recorder=usage,
                model=model, reasoning_effort=reasoning_effort, total_days=total_days,
                text_registration=True, pf=mode == 'pf', identity=identity,
                allow_final_text=role != 'ceo')
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
        self._guard_business_operations()
        for runtime in self.roles.values():
            if runtime.executor.pf_queries:
                from .pf_refresh import refresh
                runtime.executor.pf_queries.refresh = lambda versions, parent, store=runtime.store: refresh(self.server, versions, parent, store=store)
        for store in [*stores.values(), *audits.values(), *(r.usage for r in runtimes.values())]:
            if store is not None:
                store.world_state = lambda: self.server.world_state_id
        for executor in [*(r.executor for r in runtimes.values()), *script_executors.values()]:
            executor.world_status = lambda: self.server._operation_failed or self.server._step_day_timed_out
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
        write_json(path, dict(kind=kind, day=self.server.tools.current_day, status='running'))
        return path

    def _finish_operation(self, path, executor=None):
        if (self.server._operation_failed or self.server._step_day_timed_out or
                executor and (executor.preserved_process or getattr(executor, 'last_status', None) == 'result_unknown')):
            raise RuntimeError('Business operation outcome unknown; continuation refused')
        state = json.loads(path.read_text())
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
                    raise
                self._finish_operation(operation)
                return result
        def guarded_advance(*args, role='ceo', **kwargs):
            if role != 'ceo':
                return advance(*args, role=role, **kwargs)
            with server._lock:
                server.authorize_team(role, 'advance')
                operation = self._begin_operation('api:advance')
            result = advance(*args, role=role, **kwargs)
            self._finish_operation(operation)
            return result
        server.execute_tool, server.advance_week = guarded_execute, guarded_advance

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
        if self._lifecycle.hold and self._lifecycle.hold.exists():
            self._lifecycle.cancel('hold')
        self._lifecycle.check()

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
        outputs = self.server._run_daily_scripts_internal(role)
        if operation is not None:
            self._finish_operation(operation, self.server.role_executors[role])
        if any(r['status'] != 'succeeded' for r in self.server.role_script_results[role]):
            raise RunCancelled('registered_script_failed')
        text = ''
        for name, output in outputs.items():
            text = join_text(text, '\n' if text else '', join_text(name, '\n', output))
        return text

    def _observation(self, role, dashboard, scripts):
        runtime = self.roles[role]
        check = runtime.executor.weekly_check(self.server.tools.current_day)
        observation = (join_text(dashboard, '\n\nYour private registered-script output\n', scripts)
                       if scripts else dashboard)
        return join_text(observation, '\n\n', check) if check else observation

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
                    result = runtime.executor.execute(action.tool, action.arguments,
                        model_request_event=runtime.usage.last_request_event,
                        model_context_id=runtime.usage.context_id)
                    runtime.agent.record_tool_result(result)
                    self._finish_operation(operation, runtime.executor)
                if self.server.tools.current_day != day:
                    while runtime.agent._pending_tool_calls:
                        runtime.agent.record_tool_result('Cancelled because this tool batch crossed a week boundary.')
                    return
                action = runtime.agent.next_tool_action()
            self._check()
            action = runtime.agent.act('', 0, False, {'day': day})

    def run(self, *, stop_after_day, hold=None):
        if type(stop_after_day) is not int or stop_after_day <= 0 or stop_after_day % 7 or stop_after_day > self.effective_end:
            raise ValueError('stop_after_day must be a positive whole-week boundary within the simulation')
        if self._pool is not None:
            raise RuntimeError('Team run is already active')
        self._lifecycle.hold = Path(hold) if hold else None
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
                    closed_day = self.server.tools.current_day
                    for runtime in self.roles.values():
                        from .registration_evidence import week_commit_subject
                        self._snapshot_history(runtime, week_commit_subject(f'week-{closed_day // 7}'))
                    self._stage = 'boundary'
                    self._week = {}
                    self._set_phase('boundary')
                    self._boundary('week_boundary')
        except RunCancelled as exc:
            self.failure = str(exc)
            self._set_phase('stopped' if str(exc) == 'natural_end' else 'paused')
            return RunOutcome(self.server.tools.current_day, 'natural_end' if str(exc) == 'natural_end' else 'paused')
        except Exception as exc:
            self.failure = f'{type(exc).__name__}: {exc}'
            self._lifecycle.cancel('runtime_failure')
            self._set_phase('paused')
            return RunOutcome(self.server.tools.current_day, 'failed')
        finally:
            self._pool = None

    def close(self):
        self.server.stop()
        self._sockets.cleanup()
