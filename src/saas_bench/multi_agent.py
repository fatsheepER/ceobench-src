from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
import uuid

from .agents.bash_agent.agent import BashAgent
from .agents.bash_agent.tools import BashAgentToolExecutor, get_bash_agent_tool_descriptions
from .api_server import NovaMindAPIServer
from .model_usage import ModelUsage
from .role_policy import AgentIdentity, READABLE_ROLES, ROLES, RoleAudit
from .run_state import write_json
from .sql_evidence import FORMAT, SQLEvidenceStore
from .text_registry import TextRegistry


@dataclass(frozen=True)
class RoleRuntime:
    identity: AgentIdentity
    workspace: Path
    agent: BashAgent
    executor: BashAgentToolExecutor
    registry: TextRegistry
    usage: ModelUsage
    store: SQLEvidenceStore | None
    audit: RoleAudit


class MultiAgentRuntime:
    def __init__(self, root, tools, *, client_factory, mode='git', conn=None,
                 simulator=None, public_dir=None, run_id=None, world_id=None,
                 model="deepseek-v4.1-flash", reasoning_effort="high", total_days=500, **server_options):
        if mode not in ('git', 'pf'):
            raise ValueError('Team supports git and pf modes')
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        private = self.root / 'private'
        private.mkdir(mode=0o700)
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
            store = stores[role] = SQLEvidenceStore(role_private / 'evidence.sqlite', dict(
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
            usage = ModelUsage(role_private / 'model-usage.jsonl', role, evidence_store=store, identity=identity)
            agent = BashAgent(get_bash_agent_tool_descriptions(text_registration=True, pf_queries=mode == 'pf'),
                client_factory(role), workspace_path=workspace, usage_recorder=usage,
                model=model, reasoning_effort=reasoning_effort, total_days=total_days,
                text_registration=True, pf=mode == 'pf', identity=identity)
            agent.system_prompt += (f'\nYour fixed role is {role}. Your workspace is {workspace}. '
                + ('You alone can change the business, purchase research, and advance time.' if role == 'ceo'
                   else 'Business changes, paid research, and time advancement belong to CEO.')
                + '\nReadable workspaces: ' + ', '.join(str(workspaces[r]) for r in READABLE_ROLES[role]))
            registry.git_run = executor.run_private
            agent.git_run = executor.run_private
            agent._snapshot_path = role_private / 'conversation.json'
            runtimes[role] = RoleRuntime(identity, workspace, agent, executor, registry, usage, store, audit)
        self.roles = MappingProxyType(runtimes)
        self.server = NovaMindAPIServer(tools, simulator=simulator, conn=conn,
            script_workspace=workspaces['ceo'], require_sandbox=True, sql_evidence=stores['ceo'],
            role_sockets=sockets, role_stores=stores, role_executors=script_executors, role_audits=audits, **server_options)
        for store in [*stores.values(), *audits.values(), *(r.usage for r in runtimes.values())]:
            if store is not None:
                store.world_state = lambda: self.server.world_state_id
        for executor in [*(r.executor for r in runtimes.values()), *script_executors.values()]:
            executor.world_status = lambda: self.server._operation_failed or self.server._step_day_timed_out
        try:
            self.server.start()
            for runtime in self.roles.values():
                runtime.executor.verify_sandbox()
        except BaseException:
            self.close()
            raise

    def close(self):
        self.server.stop()
        self._sockets.cleanup()
