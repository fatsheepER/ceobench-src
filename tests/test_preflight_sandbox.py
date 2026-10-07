import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor


def test_formal_sandbox_rejects_missing_support(tmp_path, monkeypatch):
    executor = BashAgentToolExecutor(tmp_path, require_sandbox=True)
    monkeypatch.setattr(sys, 'platform', 'darwin')
    with pytest.raises(RuntimeError, match='Linux'):
        executor.verify_sandbox()
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr('shutil.which', lambda name: None)
    with pytest.raises(RuntimeError, match='bubblewrap'):
        executor.verify_sandbox()


@pytest.mark.parametrize('seed', ['7', '21', '0'])
def test_published_hash_seed_entry(tmp_path, seed):
    with zipfile.ZipFile(Path(__file__).parents[1] / 'public' / 'novamind-operation') as bundle:
        prefix = bundle.read('__main__.py').decode().split('if os.environ.get("NOVAMIND_SERVER_MODE")')[0]
    probe = tmp_path / 'probe.pyz'
    with zipfile.ZipFile(probe, 'w') as bundle:
        bundle.writestr('__main__.py', prefix + '\nprint(os.environ["PYTHONHASHSEED"])\n')
    result = subprocess.run([sys.executable, str(probe)], env={**os.environ, 'PYTHONHASHSEED': seed},
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == '0'


def test_file_tools_cannot_escape_to_sibling_prefix(tmp_path):
    workspace = tmp_path / 'agent'
    workspace.mkdir()
    outside = tmp_path / 'agent-other'
    outside.mkdir()
    executor = BashAgentToolExecutor(workspace)
    assert 'escapes workspace' in executor.execute('write_file', {'path': str(outside / 'bad'), 'content': 'bad'})
    assert not (outside / 'bad').exists()


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux bubblewrap acceptance runs on sheep-rog')
def test_linux_isolation_blocks_sibling_clone_and_source(tmp_path):
    workspace = tmp_path / 'agent'
    workspace.mkdir()
    sibling = tmp_path / 'other'
    sibling.write_text('private sibling')
    executor = BashAgentToolExecutor(workspace, require_sandbox=True)
    executor.verify_sandbox()
    result = executor.execute('bash', {'command': f'cat {sibling}; touch local-file; python -c "import saas_bench"'})
    assert 'private sibling' not in result
    assert 'No such file' in result and 'saas_bench' in result
    assert (workspace / 'local-file').exists()


linux_only = pytest.mark.skipif(sys.platform != 'linux', reason='Linux bubblewrap acceptance runs on sheep-rog')


def sandboxed_workspace(tmp_path):
    workspace = tmp_path / 'branches' / 'pf-42' / 'agent_workspace'
    (workspace / 'sessions' / 'abc').mkdir(parents=True)
    (workspace / 'sessions' / 'abc' / 'session.json').write_text('{"benchmark_config": {"secret": 1}}')
    (workspace / 'note.txt').write_text('marker-7c1f\n')
    executor = BashAgentToolExecutor(workspace, bash_timeout=300, require_sandbox=True)
    executor.verify_sandbox()
    return workspace, executor


def test_sandbox_requires_built_agent_runtime(tmp_path, monkeypatch):
    if sys.platform != 'linux' or not __import__('shutil').which('bwrap'):
        pytest.skip('Linux bubblewrap only')
    monkeypatch.setenv('CEOBENCH_AGENT_RUNTIME', str(tmp_path / 'missing'))
    with pytest.raises(RuntimeError, match='build_agent_runtime'):
        BashAgentToolExecutor(tmp_path, require_sandbox=True).verify_sandbox()


@linux_only
def test_sandbox_shows_only_fixed_workspace_and_agent_runtime(tmp_path):
    workspace, executor = sandboxed_workspace(tmp_path)
    result = executor.execute('bash', {'command': (
        'pwd; echo "root: $(ls / | tr "\\n" " ")"; echo "sessions: $(ls -A sessions)"; env; '
        'python -c "import sys, numpy, pandas, sklearn; print(sys.executable); print(sys.path)"; '
        'test -r /proc && echo proc-listable; python -c "import saas_bench" 2>&1 | tail -1')})
    assert result.startswith('/workspace\n')
    assert 'root: bin dev etc lib lib64 opt proc sbin tmp usr workspace ' in result
    assert 'sessions: \n' in result
    assert '/opt/python/bin/python' in result and 'blocked inside the bash_agent sandbox' in result
    assert 'proc-listable' not in result
    for host in (str(tmp_path), str(Path.home()), sys.prefix, 'pf-42'):
        assert host not in result


@linux_only
def test_recursive_root_search_finishes_inside_sandbox(tmp_path):
    workspace, executor = sandboxed_workspace(tmp_path)
    result = executor.execute('bash', {'command': 'grep -rl marker-7c1f / 2>/dev/null; echo done'})
    assert result == '/workspace/note.txt\ndone\n'


@linux_only
def test_process_substitution_and_proc_do_not_expose_host_processes(tmp_path):
    import json
    workspace, executor = sandboxed_workspace(tmp_path)
    body = {'prices': list(range(500))}
    (workspace / 'pricing.json').write_text(json.dumps(body))
    command = '''sed -n '120,400p' <(python3 -c "import json;print(json.dumps(json.load(open('pricing.json')),indent=1))")'''
    expected = '\n'.join(json.dumps(body, indent=1).splitlines()[119:400]) + '\n'
    assert executor.execute('bash', {'command': command}) == expected
    secret = tmp_path / 'host-secret'
    secret.write_text('HOST-PRIVATE-CONTENT')
    with secret.open() as stream:
        result = executor.execute('bash', {'command': (
            f'cat /proc/.kernel/{os.getpid()}/fd/{stream.fileno()}; '
            f'cat /proc/self/root{secret}; ls /proc; chmod u+r /proc; '
            'cat <(echo still-working)')})
    assert 'HOST-PRIVATE-CONTENT' not in result
    assert 'Permission denied' in result and 'Read-only file system' in result
    assert 'still-working' in result


@linux_only
def test_file_tools_follow_sandbox_paths_and_hide_sessions(tmp_path):
    workspace, executor = sandboxed_workspace(tmp_path)
    assert 'marker-7c1f' in executor.execute('read_file', {'path': '/workspace/note.txt'})
    assert 'escapes workspace' in executor.execute('read_file', {'path': str(workspace / 'note.txt')})
    assert 'not found' in executor.execute('read_file', {'path': 'sessions/abc/session.json'})
    assert 'not found' in executor.execute('write_file', {'path': '/workspace/sessions/x', 'content': 'x'})
    assert executor.execute('search_files', {'pattern': 'secret|marker'}) == 'note.txt:1: marker-7c1f'
    assert executor.execute('glob_files', {'pattern': '**/*'}) == 'note.txt'


@linux_only
def test_absolute_workspace_paths_survive_a_fork(tmp_path):
    from saas_bench.run_state import copy_workspace
    workspace, executor = sandboxed_workspace(tmp_path)
    executor.execute('write_file', {'path': 'report.py', 'content':
                                    'print(open("/workspace/note.txt").read().strip())\n'})
    command = {'command': 'cd /workspace && python /workspace/report.py'}
    before = executor.execute('bash', command)
    fork = tmp_path / 'branches' / 'git-42' / 'agent_workspace'
    copy_workspace(workspace, fork)
    after = BashAgentToolExecutor(fork, require_sandbox=True).execute('bash', command)
    assert before == after == 'marker-7c1f\n'


@linux_only
def test_repeated_read_identity_ignores_cd_into_sandbox_workspace(tmp_path):
    import sqlite3
    from saas_bench.agents.bash_agent.tools import read_identity

    class Store:
        def connect(self):
            conn = sqlite3.connect(':memory:')
            conn.execute('CREATE TABLE requests (request TEXT)')
            return conn

    workspace, executor = sandboxed_workspace(tmp_path)
    identity = [read_identity(Store(), 'e', workspace, 'bash', {'command': c}, executor.guest_root)
                for c in ('cd /workspace && cat note.txt', 'cat note.txt')]
    assert identity[0] == identity[1]
