"""Tool definitions and execution for the bash_agent.

The bash_agent has a small set of tools: bash (shell commands),
and file manipulation (read, write, edit, search, glob).
"""

import fnmatch
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


class NextDayTimeoutError(Exception):
    """Raised when ./novamind-operation next-week times out.

    The runner stops without publishing a checkpoint for the unknown operation.
    """
    def __init__(self, message: str, partial_stdout: str = "", partial_stderr: str = ""):
        super().__init__(message)
        self.partial_stdout = partial_stdout
        self.partial_stderr = partial_stderr


class ProcessBoundaryError(NextDayTimeoutError):
    """Process completion is unconfirmed; retain the command and server scene."""


# =========================================================================
# Tool schemas (OpenAI function-calling format)
# =========================================================================

BASH_AGENT_TOOL_DEFS = [
    {
        'name': 'bash',
        'description': (
            'Execute a bash command in the agent working directory. '
            'Use this to run ./novamind-operation CLI commands, Python scripts, '
            'and any other shell commands. The novamind_api Python library is '
            'available for import in Python scripts.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to execute',
                },
            },
            'required': ['command'],
        },
    },
    {
        'name': 'read_file',
        'description': (
            'Read the contents of a file. Returns the file content as a string. '
            'Use offset and limit for large files.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'Path to the file (relative to working directory)',
                },
                'offset': {
                    'type': 'integer',
                    'description': 'Line number to start reading from (1-indexed, optional)',
                },
                'limit': {
                    'type': 'integer',
                    'description': 'Maximum number of lines to read (optional)',
                },
            },
            'required': ['path'],
        },
    },
    {
        'name': 'write_file',
        'description': (
            'Create or overwrite a file with the given content. '
            'Use this to create new files or completely replace file contents.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'Path to the file (relative to working directory)',
                },
                'content': {
                    'type': 'string',
                    'description': 'Content to write to the file',
                },
            },
            'required': ['path', 'content'],
        },
    },
    {
        'name': 'edit_file',
        'description': (
            'Edit an existing file by replacing old_string with new_string. '
            'The old_string must appear exactly once in the file.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': 'Path to the file (relative to working directory)',
                },
                'old_string': {
                    'type': 'string',
                    'description': 'The exact string to find and replace',
                },
                'new_string': {
                    'type': 'string',
                    'description': 'The replacement string',
                },
            },
            'required': ['path', 'old_string', 'new_string'],
        },
    },
    {
        'name': 'search_files',
        'description': (
            'Search file contents using a regex pattern (like grep). '
            'Returns matching lines with file paths and line numbers.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'pattern': {
                    'type': 'string',
                    'description': 'Regular expression pattern to search for',
                },
                'path': {
                    'type': 'string',
                    'description': 'File or directory to search in (default: working directory)',
                },
                'glob': {
                    'type': 'string',
                    'description': 'Glob pattern to filter files (e.g., "*.py")',
                },
            },
            'required': ['pattern'],
        },
    },
    {
        'name': 'glob_files',
        'description': (
            'Find files matching a glob pattern. '
            'Returns a list of matching file paths.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'pattern': {
                    'type': 'string',
                    'description': 'Glob pattern (e.g., "**/*.py", "docs/*.json")',
                },
            },
            'required': ['pattern'],
        },
    },
]


# Inside the sandbox the workspace and the agent Python runtime appear at fixed paths,
# independent of the host directory layout, the run and the experiment group.
GUEST_WORKSPACE = '/workspace'
GUEST_PYTHON_ROOT = '/opt/python'
GUEST_PYTHON = GUEST_PYTHON_ROOT + '/bin/python3'
# Harness-owned files kept under the workspace (world database, session metadata with
# the full benchmark configuration, conversation snapshots). The agent never sees them.
HIDDEN_WORKSPACE_DIRS = ('sessions',)


def agent_runtime_dir() -> Path:
    configured = os.environ.get('CEOBENCH_AGENT_RUNTIME')
    return Path(configured) if configured else Path(sys.prefix).parent / 'agent-runtime'


def agent_runtime(verify=False) -> Dict[str, Any]:
    """The runtime built by scripts/build_agent_runtime.py, optionally checked byte for byte."""
    directory = agent_runtime_dir()
    try:
        record = json.loads((directory / 'runtime.json').read_text())
    except FileNotFoundError:
        raise RuntimeError(f'Agent runtime missing at {directory}; run scripts/build_agent_runtime.py') from None
    if record.get('python') != platform.python_version():
        raise RuntimeError('Agent runtime Python differs from the host interpreter; rebuild the agent runtime')
    if verify:
        from saas_bench.run_state import tree_hash
        if tree_hash(directory / 'python') != record['sha256']:
            raise RuntimeError('Agent runtime differs from its runtime.json; rebuild the agent runtime')
    return record


# PF groups deliver repeated identical reads of these tools as FULL, DELTA or UNCHANGED.
DELTA_READ_TOOLS = ('bash', 'read_file', 'search_files')


def call_identity(workspace, tool_name, args, guest_root=None):
    """One tool call's identity: the tool and its arguments, ignoring a leading `cd` into
    the workspace itself. Repeated returns of one call are versions of one object."""
    # A note explains the call; it is not part of what the call is.
    args = {k: v for k, v in args.items() if k != 'note'}
    if tool_name == 'bash':
        roots = {str(workspace), str(Path(workspace).resolve()), guest_root or str(workspace)}
        command = args.get('command', '')
        match = re.match(r'\s*cd\s+(\S+)\s*(?:&&|;|\n)\s*', command)
        if match and match.group(1).strip('\'"') in roots:
            args = dict(args, command=command[match.end():])
    return ['tool_call', tool_name, json.dumps(args, sort_keys=True, ensure_ascii=False)]


def read_identity(store, event, workspace, tool_name, args, guest_root=None):
    """The content object a repeated read is compared against (design 3.5).

    A Bash command that ran exactly one Python script file is identified by that
    script, so rerunning a report script compares with its previous output even when
    the surrounding command differs. Other calls are identified by tool and arguments,
    ignoring a leading `cd` into the workspace itself.
    """
    from contextlib import closing
    from pathlib import PurePosixPath
    if tool_name == 'bash':
        with closing(store.connect()) as conn:
            sources = [row[0] for row in conn.execute(
                "SELECT json_extract(request, '$.request.source') FROM requests "
                "WHERE json_extract(request, '$.parent_event_id') = ? "
                "AND json_extract(request, '$.kind') = 'cli_python'", (event,))]
        roots = {str(workspace), str(Path(workspace).resolve()), guest_root or str(workspace)}
        if len(sources) == 1 and sources[0] and sources[0] != 'inline':
            source = PurePosixPath(sources[0])
            for root in map(PurePosixPath, roots):
                if source.is_absolute() and source.is_relative_to(root):
                    source = source.relative_to(root)
            return ['script_output', source.as_posix()]
    return call_identity(workspace, tool_name, args, guest_root)


# PF groups may attach a note to the calls that produce outputs and files (design 3.2).
NOTE_TOOLS = ('bash', 'write_file', 'edit_file')
NOTE_LIMIT = 200
NOTE_PARAMETER = {
    'type': 'string',
    'description': ('Optional. Why you ran this or what the change is for. The note is saved with this call and its '
                    'outputs and changed files in .tool-notes.jsonl. Up to 200 characters.'),
}


def get_bash_agent_tool_descriptions(text_registration=False, pf_queries=False, ask_analyst=False) -> List[Dict[str, Any]]:
    """Get OpenAI Responses API-compatible tool descriptions for the bash agent."""
    definitions = BASH_AGENT_TOOL_DEFS
    if text_registration or pf_queries:
        # PF queries run as the `pf` command inside bash; the function tools stay those of Git.
        definitions = [dict(t, parameters=dict(t['parameters'], properties=dict(
            t['parameters']['properties'], note=NOTE_PARAMETER))) if t['name'] in NOTE_TOOLS else t
            for t in definitions]
    if text_registration:
        from saas_bench.registration_schema import tool_definitions
        definitions = definitions + tool_definitions(pf=pf_queries)
    if ask_analyst:
        definitions = definitions + [dict(name='ask_analyst', description="Ask an analyst a free-text question in this week's conversation.",
            parameters=dict(type='object', properties=dict(role=dict(type='string', enum=['growth', 'ops_finance']),
                message=dict(type='string')), required=['role', 'message'], additionalProperties=False))]
    return [
        {
            'type': 'function',
            'name': t['name'],
            'description': t['description'],
            'parameters': t['parameters'],
        }
        for t in definitions
    ]


def get_bash_agent_anthropic_tools() -> List[Dict[str, Any]]:
    """Get Anthropic API-compatible tool descriptions for the bash agent."""
    return [
        {
            'name': t['name'],
            'description': t['description'],
            'input_schema': t['parameters'],
        }
        for t in BASH_AGENT_TOOL_DEFS
    ]


# =========================================================================
# Tool execution
# =========================================================================

class BashAgentToolExecutor:
    """Executes bash_agent tools within a working directory."""

    def __init__(self, workspace_path: Path, env: Optional[Dict[str, str]] = None,
                 bash_timeout: int = 1200, require_sandbox: bool = False, stop_on_timeout: bool = False, evidence_store=None,
                 text_registry=None, pf_stale_checks=True, pf_refresh=None,
                 identity=None, readable_workspaces=(), api_socket=None, audit=None):
        """Initialize the tool executor.

        Args:
            workspace_path: Agent's working directory.
            env: Extra environment variables for bash commands.
            bash_timeout: Timeout in seconds for bash commands (default 5 min).
        """
        self.audit = audit
        self.identity = identity
        self.role = identity.role if identity else 'ceo'
        self.api_socket = Path(api_socket) if api_socket else None
        self.readable_workspaces = tuple(Path(p).resolve() for p in readable_workspaces)
        self._execute_lock = threading.RLock()
        if identity and not require_sandbox:
            raise ValueError('Team execution requires the Linux sandbox')
        self.workspace_path = Path(workspace_path).resolve()
        self.extra_env = env or {}
        self.bash_timeout = bash_timeout
        self.require_sandbox = require_sandbox
        self.stop_on_timeout = stop_on_timeout
        self.evidence_store = evidence_store
        self.text_registry = text_registry
        self.pf_queries = None
        if text_registry and text_registry.mode == 'pf':
            from saas_bench.pf_queries import PFQueries
            self.pf_queries = PFQueries(text_registry, stale_checks=pf_stale_checks, refresh=pf_refresh)
        self.capture = None
        self.world_status = None
        self.authorize = None
        self.preserved_process = None

    def weekly_check(self, day):
        """Week-start check of active registered texts, or None when nothing is registered."""
        if self.pf_queries:
            return self.pf_queries.weekly_check(day)
        if self.text_registry and self.text_registry.mode in ('git', 'prefix'):
            return self.text_registry.weekly_check(day)
        return None

    def verify_sandbox(self):
        if sys.platform != 'linux':
            raise RuntimeError('Formal runs require Linux and bubblewrap')
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        if self._bwrap() is None:
            raise RuntimeError('Formal runs require bubblewrap')
        agent_runtime()
        env = {'PATH': GUEST_PYTHON_ROOT + '/bin' + os.pathsep + os.defpath}
        command = self._build_bwrap_cmd(GUEST_PYTHON + ' -c pass', str(self.workspace_path), env)
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
        if result.returncode:
            raise RuntimeError('Bubblewrap startup failed: ' + result.stderr)

    def _bwrap(self):
        bwrap = shutil.which('bwrap')
        if not bwrap and self.require_sandbox:
            raise RuntimeError('Formal runs require bubblewrap')
        return bwrap

    @property
    def guest_root(self) -> str:
        """The workspace path as the agent sees it."""
        return GUEST_WORKSPACE if self._bwrap() and not self.identity else str(self.workspace_path)

    @property
    def python(self) -> str:
        """The interpreter path valid inside the agent's command environment."""
        return GUEST_PYTHON if self._bwrap() else sys.executable

    def execute(self, tool_name: str, args: Dict[str, Any], *, model_request_event=None, model_context_id=None) -> str:
        """Execute a tool and return the result string."""
        with self._execute_lock:
            if self.authorize:
                try:
                    self.authorize(self.role, 'executor')
                except PermissionError as exc:
                    self.last_status = 'cancelled'
                    return f'Error: {exc}'
            if self.preserved_process:
                raise ProcessBoundaryError('Previous process boundary remains open')
            return self._execute(tool_name, args, model_request_event=model_request_event, model_context_id=model_context_id)

    def _execute(self, tool_name, args, *, model_request_event=None, model_context_id=None):
        dispatch = {
            'bash': self._exec_bash,
            'read_file': self._exec_read_file,
            'write_file': self._exec_write_file,
            'edit_file': self._exec_edit_file,
            'search_files': self._exec_search_files,
            'glob_files': self._exec_glob_files,
        }
        if getattr(self, 'submit_handler', None):
            dispatch['submit_decision'] = self.submit_handler
        if self.text_registry:
            dispatch.update({f'text_{op}': lambda args, op=op: self.text_registry.execute(op, args)
                             for op in ('create', 'revise', 'retire', 'list')})
        facts = {}
        self._exit_code = None
        self._timed_out = False
        if self.pf_queries:
            from saas_bench.pf_queries import MODELS
            dispatch.update({op: lambda args, op=op: self.pf_queries.execute(op, args) for op in MODELS})
        if tool_name in NOTE_TOOLS and 'note' in args:
            if not isinstance(args['note'], str):
                return 'Error: note must be a string'
            if len(args['note']) > NOTE_LIMIT:
                args = dict(args, note=args['note'][:NOTE_LIMIT])
                facts['note_truncated'] = True
        handler = dispatch.get(tool_name)
        if handler is None:
            return f"Error: Unknown tool '{tool_name}'"
        if self.evidence_store:
            self.evidence_store.assert_healthy(quiescent=False)
        from saas_bench.execution_capture import ExecutionCapture, CURRENT_EVENT
        capture = ExecutionCapture(self.evidence_store) if self.evidence_store else None
        self.capture = capture
        previous_env = dict(self.extra_env)
        token = None
        before = None
        note_before = self._note_files() if tool_name in NOTE_TOOLS and args.get('note', '').strip() else None
        result, status = '', 'succeeded'
        try:
            if capture:
                capture.begin(tool_name, args, call=call_identity(self.workspace_path, tool_name, args, self.guest_root),
                              model_request_event=model_request_event, model_context_id=model_context_id,
                              **{k: v for k, v in facts.items() if k == 'command'})
                capture.facts.update(facts)
                token = CURRENT_EVENT.set(capture.event)
                if capture.event:
                    context = capture.safe(capture.store.context, capture.event)
                    if context:
                        self.extra_env['NOVAMIND_CAPTURE_CONTEXT'] = context
                if tool_name in ('bash', 'write_file', 'edit_file'):
                    before = capture.safe(capture.snapshot, self.workspace_path, 'before')
                if tool_name == 'bash':
                    capture.facts['capture_gaps'] = [
                        'unobserved_internal_file_reads', 'unobserved_intermediate_file_versions',
                        'unobserved_pipe_streams', 'unobserved_program_data_dependencies']
            result = handler(args)
            if tool_name != 'bash' and result.startswith('Error:'):
                status = 'failed'
            if capture:
                capture.origins.extend(getattr(result, 'origins', []))
            if self._timed_out:
                status = 'timed_out'
            elif self._exit_code:
                status = 'failed'
            if tool_name in NOTE_TOOLS and args.get('note', '').strip():
                from saas_bench.workspace_io import open_file
                note_after = self._note_files()
                note = dict(tool=tool_name, note=args['note'], author=self.role, status=status,
                            output_sha256=hashlib.sha256(str(result).encode()).hexdigest(),
                            files=sorted(path for path in note_before.keys() | note_after.keys() if note_before.get(path) != note_after.get(path)),
                            **(self.identity.fields() if self.identity else {}))
                with open_file(self.workspace_path, self.workspace_path / '.tool-notes.jsonl', 'ab') as stream:
                    stream.write((json.dumps(note, ensure_ascii=False) + '\n').encode())
        except NextDayTimeoutError:
            result = None
            status = 'result_unknown'
            raise
        except Exception as exc:
            result, status = f"Error: {exc}", 'failed'
        finally:
            if capture:
                after = None
                if before is not None:
                    after = capture.safe(capture.snapshot, self.workspace_path, 'after')
                    if after is not None:
                        capture.facts['changed_paths'] = sorted(k for k in before.keys() | after.keys()
                            if {x:v for x,v in before.get(k, {}).items() if x != 'version'} !=
                               {x:v for x,v in after.get(k, {}).items() if x != 'version'})
                # A failed or timed-out Bash command may still have delivered SQL output or
                # written files; decorate whatever this execution actually captured.
                read_key = None
                body = result
                if self.pf_queries and result is not None and status != 'result_unknown':
                    result = capture.safe(self.pf_queries.decorate, capture, result, after) or result
                    if tool_name in DELTA_READ_TOOLS and not getattr(result, 'pf_read', None):
                        read_key = capture.safe(read_identity, capture.store, capture.event,
                                                self.workspace_path, tool_name, args, self.guest_root)
                result = capture.finish(result, status, read_key=read_key, body=body,
                                        read_complete=not capture.facts.get('output_truncated') and
                                                      not capture.facts.get('pf_retrieval'))
                if status == 'result_unknown':
                    capture.store.fail('Execution outcome unknown; branch paused')
            if token is not None:
                CURRENT_EVENT.reset(token)
            self.extra_env = previous_env
            self.capture = None
            self.last_status = status
            if self.audit:
                self.audit.record('tool_execution', tool=tool_name, status=status,
                    model_request_event=model_request_event, context_id=model_context_id)
        if capture and tool_name.startswith('pf_'):
            result.pf_call = dict(operation=tool_name, arguments=args, event_id=capture.event,
                                 outcome='usage_error' if tool_name == 'pf_usage' else
                                         'succeeded' if status == 'succeeded' else 'execution_error')
        if capture and tool_name == 'bash':
            result.pf_calls = capture.facts.get('pf_calls', [])
        return result

    def _note_files(self):
        return {str(path.relative_to(self.workspace_path)): (path.stat().st_size, path.stat().st_mtime_ns)
                for path in self.workspace_path.rglob('*') if path.is_file() and not path.is_symlink()
                and path.relative_to(self.workspace_path).parts[0] not in ('.git', '.tool-notes.jsonl', *HIDDEN_WORKSPACE_DIRS)}

    def _roots(self):
        """Paths by which the agent may name its workspace, e.g. in a leading cd."""
        return {str(self.workspace_path), str(self.workspace_path.resolve()), self.guest_root}

    def _read_text(self, path, reuse=False):
        from saas_bench.execution_capture import decoded
        from saas_bench.sql_evidence import digest
        from saas_bench.workspace_io import open_file
        with open_file(self._read_root(path), path) as stream:
            raw = stream.read()
        raw_version = None
        if self.capture and reuse and self.capture.event:
            # Plain reads of unchanged bytes observe the existing version (same path@vK handle);
            # edits keep their own read version as the observed edit input.
            latest, sha = self.capture.safe(self.capture.store.latest_version,
                                            self._file_name(path), 'file_bytes', self._file_owner(path)) or (None, None)
            if latest and sha == digest(raw):
                self.capture.slots += 1
                raw_version = latest
        if self.capture and raw_version is None:
            raw_version = self.capture.file(self._file_name(path), raw, owner=self._file_owner(path))
        text = decoded(raw)
        version = self.capture.blob(f'file_{self.capture.slots}_text', text, 'file_text', derived_from=raw_version, owner_role=self._file_owner(path)) if self.capture else None
        return text, version

    def _source(self, version, text, start, end, target):
        if self.capture and version:
            from saas_bench.execution_capture import origin
            self.capture.origins.append(origin(version, text, start, end, target))

    def _read_root(self, path):
        return next(root for root in (self.workspace_path, *self.readable_workspaces)
                    if path.is_relative_to(root))

    def _file_name(self, path):
        return str(path.relative_to(self._read_root(path)))

    def _file_owner(self, path):
        return self._read_root(path).name if self.identity else self.role

    def _resolve_path(self, path_str: str, *, write=False) -> Path:
        """Resolve a workspace-relative or agent-visible absolute path, preventing escape."""
        path = self._contained(self._host_path(path_str), path_str)
        if write and not path.is_relative_to(self.workspace_path):
            raise ValueError('Only the owned workspace is writable')
        return path

    def _host_path(self, path_str: str) -> Path:
        """Map an agent-visible path onto the host workspace."""
        p = Path(path_str)
        if self.identity:
            return p if p.is_absolute() else self.workspace_path / p
        if p.is_absolute():
            p = Path(os.path.normpath(p))
            roots = [Path(self.guest_root)] + ([] if self._bwrap() else [self.workspace_path.resolve()])
            root = next((r for r in roots if p.is_relative_to(r)), None)
            if root is None:
                raise ValueError(f"Path escapes workspace: {path_str}")
            p = p.relative_to(root)
        return self.workspace_path / p

    def _contained(self, path: Path, shown: str) -> Path:
        """Resolve a host path, rejecting anything outside the agent-visible workspace."""
        resolved = path.resolve()
        # Ensure it's within workspace
        ws_resolved = self.workspace_path.resolve()
        if not any(resolved.is_relative_to(root) for root in (ws_resolved, *self.readable_workspaces)):
            raise ValueError(f"Path escapes workspace: {shown}")
        if self._hidden(resolved):
            raise ValueError(f"Path not found in workspace: {shown}")
        return resolved

    def _guest_paths(self, value: str) -> str:
        """Rewrite host workspace paths in a path-list value to their sandbox location."""
        if self.identity:
            return value
        hosts = {str(self.workspace_path), str(self.workspace_path.resolve())}
        parts = value.split(os.pathsep)
        for i, part in enumerate(parts):
            for host in hosts:
                if part == host or part.startswith(host + '/'):
                    parts[i] = GUEST_WORKSPACE + part[len(host):]
        return os.pathsep.join(parts)

    def _hidden(self, path: Path) -> bool:
        for root in (self.workspace_path, self.workspace_path.resolve(), *self.readable_workspaces):
            if path.is_relative_to(root):
                parts = path.relative_to(root).parts
                return bool(parts) and parts[0] in HIDDEN_WORKSPACE_DIRS
        return False

    # Env vars that MUST never be passed into the agent sandbox. The DB key
    # in particular is a hard secret — if the agent saw it, the .nmdb
    # encryption is meaningless.
    _FORBIDDEN_SANDBOX_ENV = frozenset({
        'NMDB_KEY',
        'CEOBENCH_CHECKPOINT_TOKEN',
    })

    @classmethod
    def _scrub_sandbox_env(cls, env: Dict[str, str]) -> Dict[str, str]:
        """Drop any env var that must never enter the bwrap sandbox."""
        return {k: v for k, v in env.items() if k not in cls._FORBIDDEN_SANDBOX_ENV}

    # Path to the sitecustomize.py that installs an `import saas_bench`
    # blocker inside the sandbox. Lives in this package so it ships with
    # the editable install.
    _SANDBOX_INIT_DIR = Path(__file__).parent / "_sandbox_init"

    def _build_bwrap_cmd(self, command: str, ws: str, env: Dict[str, str]) -> list:
        """Build a bwrap command that sandboxes bash to the workspace.

        Uses bubblewrap (bwrap) to create a filesystem namespace where:
        - The owned workspace is writable; permitted peer workspaces are
          read-only, and harness-owned `sessions/` directories appear empty
        - System binaries and libraries are read-only; the agent Python
          runtime (scripts/build_agent_runtime.py) is read-only at /opt/python
        - No host home, source, development environment, run or group paths
        - /proc/self/fd supports process substitution in a private PID namespace;
          the read-only, unlistable /proc directory excludes it from recursive scans
        - `import saas_bench` is blocked at the Python meta_path level via
          a `sitecustomize.py` ro-bound at `/opt/_sandbox_init/`
        """
        bwrap = self._bwrap()
        if not bwrap:
            return None  # Fall back to unsandboxed execution

        env = self._scrub_sandbox_env(env)

        cmd = [bwrap]

        # Read-only system paths
        for sys_path in ['/usr', '/bin', '/lib', '/lib64', '/etc',
                         '/sbin', '/usr/local']:
            if os.path.exists(sys_path):
                cmd.extend(['--ro-bind', sys_path, sys_path])

        cmd.extend(['--dev', '/dev'])
        # bwrap's /dev/fd points to /proc/self/fd. A private procfs provides
        # dynamic descriptors without exposing host processes or their roots.
        # Keep its parent unlistable so grep -r / cannot block on virtual files.
        cmd.extend(['--tmpfs', '/proc', '--proc', '/proc/.kernel',
                    '--remount-ro', '/proc/.kernel', '--symlink', '.kernel/self', '/proc/self',
                    '--chmod', '0111', '/proc', '--remount-ro', '/proc'])
        if getattr(self, '_pf_service', None):
            cmd.extend(['--ro-bind', str(self._pf_service.root), self._pf_service.guest])

        # Writable /tmp (separate from workspace, for temp files)
        cmd.extend(['--tmpfs', '/tmp'])

        # The agent's own interpreter and analysis packages. The zipapps are
        # compiled for this exact Python version (checked by agent_runtime()).
        agent_runtime()
        cmd.extend(['--ro-bind', str(agent_runtime_dir() / 'python'), GUEST_PYTHON_ROOT])

        # Sandbox init dir — contains sitecustomize.py that blocks
        # `import saas_bench` at the Python meta_path level. Mounted at a
        # fixed path inside the sandbox and prepended to PYTHONPATH so
        # site.py picks up sitecustomize on every interpreter start.
        sandbox_init_host = self._SANDBOX_INIT_DIR
        if not sandbox_init_host.is_dir():
            # The server also uses this executor from inside the zipapp.
            import pkgutil
            import tempfile
            if not hasattr(self, '_sandbox_resources'):
                self._sandbox_resources = tempfile.TemporaryDirectory(prefix='novamind-sandbox-')
                data = pkgutil.get_data('saas_bench.agents.bash_agent', '_sandbox_init/sitecustomize.py')
                if data is None:
                    raise RuntimeError('Sandbox import blocker is missing')
                (Path(self._sandbox_resources.name) / 'sitecustomize.py').write_bytes(data)
            sandbox_init_host = Path(self._sandbox_resources.name)
        sandbox_init_guest = "/opt/_sandbox_init"
        if sandbox_init_host.is_dir():
            cmd.extend(['--ro-bind', str(sandbox_init_host), sandbox_init_guest])
            existing_pp = env.get('PYTHONPATH', '')
            env['PYTHONPATH'] = (
                f"{sandbox_init_guest}:{existing_pp}" if existing_pp else sandbox_init_guest
            )

        # ORACLE MODE: ro-bind the simulator source tree so the agent can
        # read config.py, simulation.py, engine internals, and `import saas_bench`
        # works (the meta-path blocker in sitecustomize.py is also skipped when
        # ORACLE_MODE=1, see _sandbox_init/sitecustomize.py).
        if env.get('ORACLE_MODE') == '1':
            oracle_src_default = "/data/saas-bench/src"
            oracle_src = env.get('ORACLE_SOURCE_DIR', oracle_src_default)
            if os.path.isdir(oracle_src):
                cmd.extend(['--ro-bind', oracle_src, oracle_src])
                # Make sure the agent's editable install resolves: prepend the
                # source dir to PYTHONPATH so `import saas_bench` finds the
                # source even if the venv .pth file is missing.
                existing_pp = env.get('PYTHONPATH', '')
                env['PYTHONPATH'] = (
                    f"{oracle_src}:{existing_pp}" if existing_pp else oracle_src
                )

        # The agent workspace — ONLY writable directory
        cmd.extend(['--bind', ws, self.guest_root])
        mounts = [(ws, self.guest_root)]
        for peer in self.readable_workspaces:
            if peer != self.workspace_path:
                cmd.extend(['--ro-bind', str(peer), str(peer)])
                mounts.append((str(peer), str(peer)))
        for host, guest in mounts:
            for name in HIDDEN_WORKSPACE_DIRS:
                if os.path.isdir(os.path.join(host, name)):
                    cmd.extend(['--tmpfs', guest + '/' + name])
        if self.api_socket:
            cmd.extend(['--ro-bind', str(self.api_socket), '/run/novamind-api/api.sock'])
        if getattr(self, '_boundary_socket_dir', None):
            cmd.extend(['--ro-bind', self._boundary_socket_dir, '/run/novamind-boundary'])

        # Set working directory
        cmd.extend(['--chdir', self.guest_root])

        # Unshare namespaces for isolation
        cmd.extend(['--unshare-all'] + ([] if self.identity else ['--share-net']))

        # Set environment variables
        for k, v in env.items():
            cmd.extend(['--setenv', k, v])

        # The actual command
        cmd.extend(['bash', '-c', command])

        return cmd

    def run_private(self, argv, **kwargs):
        """Run host-requested Git inspection under the same filesystem authority."""
        import shlex
        env = dict(PATH=GUEST_PYTHON_ROOT + '/bin:' + os.defpath,
                   HOME=self.guest_root, GIT_CONFIG_NOSYSTEM='1', LANG='C.UTF-8')
        command = self._build_bwrap_cmd(shlex.join(argv), str(self.workspace_path), env)
        return subprocess.run(command, env=env, **kwargs)

    def _exec_bash(self, args: Dict) -> str:
        """Execute a bash command, sandboxed to the workspace directory.

        Uses bubblewrap (bwrap) to create a true filesystem sandbox where
        only the agent workspace is writable. System paths and Python are
        available read-only. Falls back to soft sandbox if bwrap unavailable.
        """
        command = args.get('command', '')
        if not command:
            return "Error: No command provided"

        from saas_bench.process_boundary import Boundary
        service = None
        if self.pf_queries:
            from saas_bench.pf_shell import ShellService
            service = self._pf_service = ShellService(self)
        boundary = Boundary(command, self.python, isolated=bool(self.identity))
        self._boundary_socket_dir = boundary.socket_dir.name if boundary.socket_dir else None
        try:
            result = self._run_bash(command, boundary)
            if service:
                self.capture.facts['pf_calls'] = service.calls
                return service.project(result)
            return result
        finally:
            boundary.close()
            self._boundary_socket_dir = None
            if service:
                service.close()
                self.capture.facts['pf_calls'] = service.calls
                self._pf_service = None

    def _run_bash(self, command, boundary):
        from saas_bench.process_boundary import BoundaryOpen
        ws = str(self.workspace_path)
        supervised_command = boundary.command

        # Build a minimal, sandboxed environment.
        # Start from scratch — do NOT inherit os.environ (which contains
        # simulator source paths, home directory, etc.)
        sandboxed = self._bwrap() is not None
        python_bin_dir = GUEST_PYTHON_ROOT + '/bin' if sandboxed else os.path.join(sys.prefix, 'bin')
        path_parts = [python_bin_dir] if sandboxed or os.path.isdir(python_bin_dir) else []
        path_parts += ['/usr/local/bin', '/usr/bin', '/bin']
        env = {
            'PATH': ':'.join(path_parts),
            'HOME': self.guest_root,
            'TMPDIR': self.guest_root,
            'LANG': os.environ.get('LANG', 'en_US.UTF-8'),
            'TERM': os.environ.get('TERM', 'xterm'),
        }
        env.update(self.extra_env)
        if getattr(self, '_pf_service', None):
            env['PATH'] = self._pf_service.guest + os.pathsep + env['PATH']
            env['PF_SOCKET'] = self._pf_service.guest + '/socket'
        env = self._scrub_sandbox_env(env)
        if sandboxed:
            env = {k: self._guest_paths(v) for k, v in env.items()}

        # Try bwrap sandbox; fall back to basic Popen if unavailable
        bwrap_cmd = self._build_bwrap_cmd(supervised_command, ws, env)

        # Use Popen so we can explicitly kill the process group on timeout.
        # subprocess.run() does NOT kill children on TimeoutExpired, leaving
        # zombie processes that can hold DB locks or resources.
        import signal
        if self.capture:
            self.capture.facts['process_started'] = False
        if bwrap_cmd:
            # CRITICAL: pass env=env so bwrap inherits a clean dict.
            # Without this, bwrap inherits the launcher's full os.environ —
            # including NMDB_KEY, which is the engine's DB encryption key.
            # bwrap's `--setenv` only adds to the inherited env; it does not
            # clear it. (Older bwrap builds don't have `--clearenv` either.)
            # 2026-04-28: this leak is how the gpt55 v3.4aa run (1267c284)
            # decrypted world.nmdb and ran UPDATE statements directly.
            proc = subprocess.Popen(
                bwrap_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                env=env,
                start_new_session=True,
            )
        else:
            proc = subprocess.Popen(
                ['bash', '-c', supervised_command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                cwd=ws,
                env=env,
                start_new_session=True,
            )
        if self.capture:
            self.capture.facts.update(process_started=True, pid=proc.pid)
            sink = os.fstat(proc.stdout.fileno())
            self.capture.facts['stdout_sink'] = [sink.st_dev, sink.st_ino]
        try:
            raw_stdout, raw_stderr = boundary.communicate(proc, self.bash_timeout)
            if self.capture:
                self.capture.facts['process_boundary'] = boundary.record
            stdout, stderr = self._streams(raw_stdout, raw_stderr, proc.returncode)

            output_parts = []
            if stdout:
                output_parts.append(stdout)
            if stderr:
                output_parts.append(f"[stderr]\n{stderr}")
            if proc.returncode != 0:
                output_parts.append(f"[exit code: {proc.returncode}]")

            output = '\n'.join(output_parts) if output_parts else "(no output)"

            # Truncate very long output (same limit as Claude Code: 30K chars)
            if self.capture:
                self.capture.origins = self._stream_origins(stdout, stderr)
            if len(output) > 30000:
                if self.capture:
                    self.capture.facts['output_truncated'] = True
                    from saas_bench.execution_capture import slice_origins
                    marker = '\n\n... (output truncated — exceeded 30,000 character limit) ...\n\n'
                    self.capture.origins = (slice_origins(self.capture.origins, 0, 15000) +
                        slice_origins(self.capture.origins, len(output) - 15000, len(output), 15000 + len(marker)))
                output = output[:15000] + "\n\n... (output truncated — exceeded 30,000 character limit) ...\n\n" + output[-15000:]

            if self.world_status and self.world_status():
                raise NextDayTimeoutError('Server operation outcome unknown', partial_stdout=stdout, partial_stderr=stderr)
            port = self.extra_env.get('NOVAMIND_API_PORT')
            if port and int(port) > 0 and not self.identity:
                import json
                import urllib.request
                with urllib.request.urlopen(f'http://127.0.0.1:{int(port)}/game-status', timeout=5) as response:
                    state = json.load(response)
                if state.get('timed_out') or state.get('operation_failed'):
                    raise NextDayTimeoutError('Server operation outcome unknown',
                                              partial_stdout=stdout, partial_stderr=stderr)

            return output

        except BoundaryOpen as exc:
            self._preserve_process(proc, exc.record)
            try:
                self._streams(exc.stdout, exc.stderr, exc.record['exit_code'], partial=True)
            except UnicodeError:
                pass  # Raw streams were retained; decoding cannot close an open execution.
            raise ProcessBoundaryError(str(exc), partial_stdout=repr(exc.stdout), partial_stderr=repr(exc.stderr)) from exc
        except subprocess.TimeoutExpired:
            self._timed_out = True
            # Kill the entire process group (bash + all children)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.kill()  # Fallback: kill the direct child
            except OSError as exc:
                self._preserve_process(proc, dict(unknown='process cleanup failed: ' + str(exc)))
                raise NextDayTimeoutError('Process cleanup failed; outcome unknown') from exc
            try:
                raw_stdout, raw_stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired as exc:
                self._preserve_process(proc, dict(unknown='second stream collection timed out'))
                self._streams(exc.output or b'', exc.stderr or b'', proc.poll(), partial=True)
                raise NextDayTimeoutError('Process cleanup did not finish; outcome unknown',
                    partial_stdout=repr(exc.output), partial_stderr=repr(exc.stderr))
            partial_stdout, partial_stderr = self._streams(raw_stdout, raw_stderr, proc.returncode)
            if self.capture:
                self.capture.facts['timed_out'] = True
            if self.stop_on_timeout or './novamind-operation next-week' in command:
                raise NextDayTimeoutError(
                    f"Tool timed out after {self.bash_timeout}s; outcome unknown",
                    partial_stdout=partial_stdout or "",
                    partial_stderr=partial_stderr or "",
                )

            # For all other commands: return partial output + timeout message
            output_parts = []
            if partial_stdout:
                output_parts.append(partial_stdout)
            if partial_stderr:
                output_parts.append(f"[stderr]\n{partial_stderr}")
            output_parts.append(f"Error: Command timed out after {self.bash_timeout} seconds")
            return '\n'.join(output_parts)

    def _preserve_process(self, proc, record):
        self.preserved_process = proc
        if self.capture:
            self.capture.facts.update(process_boundary=record, boundary_closed=False, preserved_pid=proc.pid)
            self.capture.store.fail('Process boundary unresolved; branch paused',
                                    preserve_scene=True, supervisor_pid=proc.pid, process_boundary=record)

    def _streams(self, stdout, stderr, exit_code, partial=False):
        self._exit_code = exit_code
        from saas_bench.execution_capture import decoded
        if self.capture:
            self.capture.facts['exit_code'] = exit_code
            self.capture.blob('stdout_bytes', stdout, 'stdout_bytes', extent='partial' if partial else 'full')
            self.capture.blob('stderr_bytes', stderr, 'stderr_bytes', extent='partial' if partial else 'full')
        out, err = decoded(stdout), decoded(stderr)
        if self.capture:
            self.capture.blob('stdout', out, 'stdout', derived_from=self.capture.event + ':stdout_bytes', extent='partial' if partial else 'full')
            self.capture.blob('stderr', err, 'stderr', derived_from=self.capture.event + ':stderr_bytes', extent='partial' if partial else 'full')
        return out, err

    def _stream_origins(self, stdout, stderr):
        from saas_bench.execution_capture import origin
        result = []
        if stdout:
            result.append(origin(self.capture.event + ':stdout', stdout))
            from saas_bench.execution_capture import query_projection, script_projection
            result.extend(self.capture.safe(query_projection, self.capture, stdout) or [])
            result.extend(self.capture.safe(script_projection, self.capture, stdout) or [])
        if stderr:
            result.append(origin(self.capture.event + ':stderr', stderr,
                                 target=(len(stdout) + 1 if stdout else 0) + len('[stderr]\n')))
        return result

    def _exec_read_file(self, args: Dict) -> str:
        """Read file contents."""
        path = self._resolve_path(args['path'])
        if not path.exists():
            return f"Error: File not found: {args['path']}"
        if not path.is_file():
            return f"Error: Not a file: {args['path']}"

        content, version = self._read_text(path, reuse=True)
        lines = content.split('\n')

        offset = args.get('offset', 1)
        limit = args.get('limit')
        if type(offset) is not int or offset < 1 or (limit is not None and (type(limit) is not int or limit < 1)):
            return 'Error: offset and limit must be positive integers'

        # Apply offset (1-indexed)
        start = max(0, offset - 1)
        if limit:
            end = start + limit
            lines = lines[start:end]
        else:
            lines = lines[start:]

        # Format with line numbers
        numbered = []
        source_start = sum(len(line) + 1 for line in content.split('\n')[:start])
        target = 0
        for i, line in enumerate(lines, start=start + 1):
            prefix = f'{i:6d}\t'
            # The inserted separator is the original newline between these lines.
            newline = int(i < start + len(lines))
            self._source(version, content, source_start, source_start + len(line) + newline, target + len(prefix))
            numbered.append(prefix + line)
            source_start += len(line) + 1
            target += len(prefix) + len(line) + 1

        return '\n'.join(numbered)

    def _exec_write_file(self, args: Dict) -> str:
        """Write file contents."""
        path = self._resolve_path(args['path'], write=True)
        from saas_bench.workspace_io import open_file
        with open_file(self.workspace_path, path, 'wb') as stream:
            stream.write(args['content'].encode())
        if self.capture:
            self.capture.facts['written_paths'] = [self._file_name(path)]
        return f"File written: {args['path']} ({path.stat().st_size} bytes)"

    def _exec_edit_file(self, args: Dict) -> str:
        """Edit a file by replacing old_string with new_string."""
        path = self._resolve_path(args['path'], write=True)
        if not path.exists():
            return f"Error: File not found: {args['path']}"

        content, version = self._read_text(path)
        old_str = args['old_string']
        new_str = args['new_string']

        count = content.count(old_str)
        if count == 0:
            return f"Error: old_string not found in {args['path']}"
        if count > 1:
            return f"Error: old_string found {count} times in {args['path']} (must be unique)"

        new_content = content.replace(old_str, new_str, 1)
        from saas_bench.workspace_io import open_file
        with open_file(self.workspace_path, path, 'wb') as stream:
            stream.write(new_content.encode())
        if self.capture:
            self.capture.facts['written_paths'] = [self._file_name(path)]
        return f"File edited: {args['path']}"

    def _exec_search_files(self, args: Dict) -> str:
        """Search files with regex pattern."""
        pattern = args['pattern']
        search_path = args.get('path', '.')
        glob_filter = args.get('glob', '*')

        resolved = self._resolve_path(search_path)
        if not resolved.exists():
            return f"Error: Path not found: {search_path}"

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex: {e}"

        matches = []
        if resolved.is_file():
            files = [resolved]
        else:
            files = sorted(f for f in resolved.rglob(glob_filter) if not self._hidden(f))

        target = 0
        scanned, skipped = [], []
        for fpath in files[:100]:  # Limit file count
            try:
                checked = self._contained(fpath, str(fpath))
            except ValueError:
                skipped.append(str(fpath))
                continue
            if not fpath.is_file():
                continue
            try:
                content, version = self._read_text(checked, reuse=True)
                scanned.append(str(fpath))
            except (UnicodeDecodeError, PermissionError):
                skipped.append(str(fpath))
                continue
            source_start = 0
            for i, line in enumerate(content.split('\n'), 1):
                if regex.search(line):
                    rel = self._file_name(fpath)
                    prefix = f"{rel}:{i}: "
                    self._source(version, content, source_start, source_start + len(line), target + len(prefix))
                    matches.append(prefix + line)
                    target += len(prefix) + len(line) + 1
                    if len(matches) >= 200:
                        break
                source_start += len(line) + 1
            if len(matches) >= 200:
                break

        if self.capture:
            self.capture.facts.update(scanned=scanned, skipped=skipped, candidates=len(files),
                                      source_truncated=len(files) > 100 or len(matches) >= 200)
        if len(files) > 100 or len(matches) >= 200:
            matches.append('[Search truncated: at most 100 candidates and 200 matches.]')
        if skipped:
            matches.append(f'[Skipped {len(skipped)} unreadable or out-of-workspace candidates.]')
        if not matches:
            return "No matches found."
        return '\n'.join(matches)

    def _exec_glob_files(self, args: Dict) -> str:
        """Find files matching a glob pattern."""
        pattern = args['pattern']
        root = self.workspace_path
        if self.identity:
            parts = Path(pattern).parts
            static = []
            for part in parts:
                if any(c in part for c in '*?['):
                    break
                static.append(part)
            if static:
                root = self._resolve_path(str(Path(*static)))
                pattern = str(Path(*parts[len(static):])) if len(static) < len(parts) else ''
            matches = sorted(m for m in (root.glob(pattern) if pattern else [root]) if not self._hidden(m))
        else:
            if Path(pattern).is_absolute() or '..' in Path(pattern).parts:
                return 'Error: Glob must stay within workspace'
            matches = sorted(m for m in root.glob(pattern) if not self._hidden(m))
        if not matches:
            return "No matching files."
        result = []
        for m in matches[:200]:
            try:
                self._contained(m, str(m))
                rel = self._file_name(m)
                result.append(str(rel))
            except ValueError:
                result.append('[Skipped out-of-workspace path]')
        if len(matches) > 200:
            result.append('[Glob truncated: first 200 paths.]')
        if self.capture:
            self.capture.facts.update(candidates=len(matches), source_truncated=len(matches) > 200)
        return '\n'.join(result)
