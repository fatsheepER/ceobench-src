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
from .model_usage import ModelUsage
from .role_policy import AgentIdentity, READABLE_ROLES, ROLES, RoleAudit
from .run_state import write_json
from .run_lifecycle import RunCancelled, WorkerLifecycle
from .sql_evidence import FORMAT, SQLEvidenceStore
from .text_registry import TextRegistry


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
                 model="deepseek-v4.1-flash", reasoning_effort="high", total_days=500, token_counter=None, **server_options):
        if mode not in ('git', 'pf'):
            raise ValueError('Team supports git and pf modes')
        if type(total_days) is not int or total_days < 7:
            raise ValueError('total_days must include at least one full week')
        self.total_days = total_days
        if mode == 'pf' and token_counter is None:
            from .payload_tokens import load_counter
            token_counter = load_counter('opencode', model)
        self.effective_end = total_days // 7 * 7
        self._lifecycle = WorkerLifecycle()
        self._message_lock = threading.Lock()
        self.messages = []
        self.failure = None
        self._pool = None
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        private = self.root / 'private'
        private.mkdir(mode=0o700)
        self.message_path = private / 'messages.jsonl'
        run_id, world_id = run_id or uuid.uuid4().hex, world_id or uuid.uuid4().hex
        self._sockets = tempfile.TemporaryDirectory(prefix='ceobench-team-')
        public = Path(public_dir or Path(__file__).resolve().parents[2] / 'public')
        workspaces = {role: self.root / 'roles' / role for role in ROLES}
        sockets = {role: Path(self._sockets.name) / role / 'api.sock' for role in ROLES}
        stores, runtimes, script_executors, audits = {}, {}, {}, {}
        for role in ROLES:
            workspace = workspaces[role]
            workspace.mkdir(parents=True)
            shutil.copytree(public / 'docs', workspace / 'docs')
            shutil.copy2(public / 'novamind-client', workspace / 'novamind-operation')
            (workspace / 'MEMORY.md').write_text('')
            identity = AgentIdentity(run_id, world_id, role, uuid.uuid4().hex, uuid.uuid4().hex)
            role_private = private / role
            role_private.mkdir(mode=0o700)
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
            usage = ModelUsage(role_private / 'model-usage.jsonl', role, evidence_store=store, identity=identity, token_counter=token_counter)
            agent = BashAgent(get_bash_agent_tool_descriptions(text_registration=True, pf_queries=mode == 'pf', ask_analyst=role == 'ceo'),
                client_factory(role), workspace_path=workspace, usage_recorder=usage,
                model=model, reasoning_effort=reasoning_effort, total_days=total_days,
                text_registration=True, pf=mode == 'pf', identity=identity,
                allow_final_text=role != 'ceo')
            if role != 'ceo':
                from .registration_prompt import REGISTERED_TEXTS, PF_HISTORY, GIT_HISTORY
                history = PF_HISTORY if mode == 'pf' else GIT_HISTORY
                cite = 'analysis.py.out@v1' if mode == 'pf' else 'notes.md'
                focus = ('demand, acquisition, conversion, retention and revenue' if role == 'growth'
                    else 'cash, costs, capacity, quality, development and cash forecasts')
                agent.system_prompt = (f'You are the {role} analyst for the NovaMind SaaS business. '
                    f'The shared business objective is to maximize final cash over about {total_days} days. '
                    f'Investigate {focus}.')
                agent.system_prompt += ('\nUse bash, read_file, search_files and glob_files to inspect public data and files. '
                    'Read-only SQL is available through novamind_api.query. Read docs/ for API and table details. '
                    'Write useful notes and MEMORY.md only in your workspace. Return a final free-text answer when finished. '
                    'Do not call business mutation, paid research or time advancement APIs. '
                    'Your advice is optional input for the CEO.\n' + REGISTERED_TEXTS.format(cite=cite) + history)
            agent.system_prompt += (f'\nThe effective simulation endpoint is D{self.effective_end}. '
                f'Your fixed role is {role}. Your workspace is {workspace}. '
                + ('You alone can change the business, purchase research, and advance time.' if role == 'ceo'
                   else 'Business changes, paid research, and time advancement belong to CEO.')
                + '\nReadable workspaces: ' + ', '.join(str(workspaces[r]) for r in READABLE_ROLES[role]))
            if role == 'ceo':
                agent.system_prompt += ('\nTwo analysts supply initial advice automatically each week. Use their existing analysis first. '
                    'Read their files, ask_analyst(role, message), investigate independently, or act as useful. '
                    'Save your artifacts and submit the weekly rationale and 12 USD cash forecasts before advancing.')
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
                self._initialize_history(runtime)
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
        outputs = self.server._run_daily_scripts_internal(role)
        if any(r['status'] != 'succeeded' for r in self.server.role_script_results[role]):
            raise RunCancelled('registered_script_failed')
        return '\n'.join(f'{name}\n{output}' for name, output in outputs.items())

    def _observation(self, role, dashboard, scripts):
        runtime = self.roles[role]
        check = runtime.executor.weekly_check(self.server.tools.current_day)
        return dashboard + ('\n\nYour private registered-script output\n' + scripts if scripts else '') + (
            '\n\n' + check if check else '')

    def _run_analyst(self, role, requests, dashboard=None):
        runtime = self.roles[role]
        scripts = self._script_output(role) if dashboard is not None else None
        observation = self._observation(role, dashboard, scripts) if dashboard is not None else None
        for request in requests:
            try:
                self._check()
                self._transition(request, 'running')
                answer = runtime.agent.act(observation + '\n\n' + request.text if observation is not None else request.text,
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
        return [r.reply if isinstance(r, TeamMessage) else r for r in results]

    def _run_ceo(self, observation):
        runtime = self.roles['ceo']
        day = self.server.tools.current_day
        self._check()
        action = runtime.agent.act(observation, 0, False, {'day': day})
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
                else:
                    result = runtime.executor.execute(action.tool, action.arguments,
                        model_request_event=runtime.usage.last_request_event,
                        model_context_id=runtime.usage.context_id)
                    runtime.agent.record_tool_result(result)
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
        self._set_phase('boundary')
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
                    self._rotate_sessions()
                    self._set_phase('ceo_scripts')
                    ceo_scripts = self._script_output('ceo')
                    self._check()
                    self._set_phase('analysis')
                    dashboard = self.server.public_dashboard()
                    requests = [self._message(role, 'Analyze the current business week and give the CEO useful advice in final prose.')
                                for role in ROLES[1:]]
                    self._parallel(requests, dashboard)
                    for request in requests:
                        self._transition(request, 'delivered')
                    self._set_phase('ceo')
                    observation = self._observation('ceo', dashboard, ceo_scripts)
                    observation += '\n\n' + '\n\n'.join(f'{r.receiver} analyst\n{r.reply}' for r in requests)
                    self._run_ceo(observation)
                    closed_day = self.server.tools.current_day
                    for runtime in self.roles.values():
                        from .registration_evidence import week_commit_subject
                        self._snapshot_history(runtime, week_commit_subject(f'week-{closed_day // 7}'))
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
