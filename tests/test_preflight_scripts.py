from pathlib import Path
from types import SimpleNamespace
import urllib.request
import json

from saas_bench.api_server import NovaMindAPIServer


def test_registered_versions_survive_database_backup(tmp_path):
    import sqlite3
    import pytest
    tools = SimpleNamespace(workspace_path=tmp_path, current_day=0)
    conn = sqlite3.connect(':memory:')
    server = NovaMindAPIServer(tools, conn=conn)
    server.set_daily_scripts({'first': "print('A')", 'second': "print('two')"})
    clone = sqlite3.connect(':memory:')
    conn.backup(clone)
    restored = NovaMindAPIServer(tools, conn=clone)
    assert list(restored.get_daily_scripts()) == ['first', 'second']
    assert restored._run_daily_scripts_internal()['first'] == 'A\n'
    restored.set_daily_scripts({'first': "print('B')"})
    assert restored._run_daily_scripts_internal()['first'] == 'B\n'
    assert server.get_daily_scripts()['first'] == "print('A')"
    restored.set_daily_scripts({})
    assert NovaMindAPIServer(tools, conn=clone).get_daily_scripts() == {}
    conn.execute("UPDATE _registered_scripts SET content='corrupt' WHERE name='first'")
    conn.commit()
    with pytest.raises(ValueError, match='checksum'):
        NovaMindAPIServer(tools, conn=conn)


def test_registered_snapshot_executes_and_can_call_server(tmp_path):
    tools = SimpleNamespace(workspace_path=tmp_path, current_day=0)
    server = NovaMindAPIServer(tools)
    server.start()
    try:
        source = tmp_path / 'metric.py'
        source.write_text("print('A')")
        script = source.read_text() + '\nimport urllib.request, os\nprint(urllib.request.urlopen("http://127.0.0.1:" + os.environ["NOVAMIND_API_PORT"] + "/health").status)'
        request = urllib.request.Request(f'http://127.0.0.1:{server.port}/daily-scripts',
            json.dumps({'name': source.name, 'content': script}).encode(), {'Content-Type': 'application/json'})
        assert json.load(urllib.request.urlopen(request))['success']
        source.write_text("print('B')")
        assert server._run_daily_scripts_internal()['metric.py'] == 'A\n200\n'
        server.set_daily_scripts({'metric.py': source.read_text(), 'bad': 'raise ValueError("failure")'})
        outputs = server._run_daily_scripts_internal()
        assert outputs['metric.py'] == 'B\n'
        assert 'failure' in outputs['bad']
        assert len(server.last_script_results) == 2
        server.set_daily_scripts({})
        assert server._run_daily_scripts_internal() == {}
    finally:
        server.stop()


def test_script_timeout_is_visible_and_does_not_stall_following_script(tmp_path, monkeypatch):
    from saas_bench.agents.bash_agent import tools
    real_executor = tools.BashAgentToolExecutor
    def short_executor(*args, **kwargs):
        kwargs['bash_timeout'] = 0.2
        return real_executor(*args, **kwargs)
    monkeypatch.setattr(tools, 'BashAgentToolExecutor', short_executor)
    server = NovaMindAPIServer(SimpleNamespace(workspace_path=tmp_path))
    server.set_daily_scripts({'slow': "import time; print('started', flush=True); time.sleep(30)", 'after': "print('finished')"})
    outputs = server._run_daily_scripts_internal()
    assert 'timed out' in outputs['slow']
    assert 'started' in outputs['slow']
    assert outputs['after'] == 'finished\n'
