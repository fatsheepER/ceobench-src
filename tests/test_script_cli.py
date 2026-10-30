"""The public script CLI must report the server's registered snapshots."""
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import pytest

from saas_bench import _public_cli, novamind_cli
from saas_bench.api_server import NovaMindAPIServer


@pytest.fixture
def backend(tmp_path, monkeypatch):
    server = NovaMindAPIServer(SimpleNamespace(workspace_path=tmp_path, current_day=0))
    server.start()
    monkeypatch.setenv('NOVAMIND_API_PORT', str(server.port))
    monkeypatch.delenv('NOVAMIND_API_SOCKET', raising=False)
    yield server
    server.stop()


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, 'argv', ['novamind-operation', *map(str, args)])
    _public_cli.main()


def test_public_snapshot_register_list_remove(backend, tmp_path, monkeypatch, capsys):
    source = tmp_path / 'metric.py'
    source.write_text("print('version A')")
    invoke(monkeypatch, 'register-daily-script', source)
    registered = json.loads(capsys.readouterr().out)
    assert registered['success'] is True
    assert registered['data']['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_text("print('version B')")
    invoke(monkeypatch, 'list-daily-scripts')
    listing = json.loads(capsys.readouterr().out)
    assert listing['data']['scripts'][0]['sha256'] == registered['data']['sha256']
    assert backend.get_daily_scripts() == {'metric.py': "print('version A')"}
    invoke(monkeypatch, 'remove-daily-script', source.name)
    assert json.loads(capsys.readouterr().out)['data']['removed'] == source.name
    assert backend.get_daily_scripts() == {}
    assert source.read_text() == "print('version B')"
    with pytest.raises(SystemExit) as failure:
        invoke(monkeypatch, 'remove-daily-script', source.name)
    assert failure.value.code == 1
    assert json.loads(capsys.readouterr().out)['success'] is False


@pytest.mark.parametrize('command,argument', [
    ('register-daily-script', 'script_path'), ('list-daily-scripts', None),
    ('remove-daily-script', 'script_name'),
])
@pytest.mark.parametrize('entry', ['public', 'private'])
@pytest.mark.parametrize('failure', ['connection', 'rejected'])
def test_failures_never_report_success(tmp_path, monkeypatch, capsys, command, argument, entry, failure):
    source = tmp_path / 'metric.py'
    source.write_text('print(1)')
    monkeypatch.setenv('NOVAMIND_API_PORT', '1')
    monkeypatch.setenv('NOVAMIND_WORKSPACE', str(tmp_path))
    def failed(*args, **kwargs):
        if failure == 'connection':
            raise OSError('connection unavailable')
        return {'success': False, 'error': 'permission denied'}
    monkeypatch.setattr(_public_cli, '_api_call', failed)
    from saas_bench.novamind_api import _client
    for method in ('_post', '_get', '_delete'):
        monkeypatch.setattr(_client, method, failed)
    with pytest.raises(SystemExit) as error:
        if entry == 'public':
            invoke(monkeypatch, command, *([source] if argument else []))
        else:
            getattr(novamind_cli, '_cmd_' + command.replace('-', '_'))(
                SimpleNamespace(**({argument: str(source)} if argument else {})))
    assert error.value.code == 1
    assert '"success": true' not in capsys.readouterr().out
    assert source.exists()
    assert not (tmp_path / 'daily_scripts').exists()


def test_generated_script_docs_match_public_commands(monkeypatch, capsys):
    docs = novamind_cli.get_cli_docs()
    for command in ('register-daily-script', 'list-daily-scripts', 'remove-daily-script'):
        assert f'./novamind-operation {command}' in docs
        with pytest.raises(SystemExit) as result:
            invoke(monkeypatch, command, '--help')
        assert result.value.code == 0
        capsys.readouterr()
    assert 'alphabetical' not in docs
    assert 'pre-imported' not in docs
    assert 'start of each day' not in docs
    assert 'insertion order' in docs
    assert 'ask_analyst' in docs


def test_socket_only_uses_bound_role_registry(tmp_path, monkeypatch, capsys):
    from saas_bench.role_policy import ROLES
    server = NovaMindAPIServer(SimpleNamespace(workspace_path=tmp_path, current_day=0),
                              role_sockets={role: tmp_path / f'{role}.sock' for role in ROLES})
    server.start()
    monkeypatch.delenv('NOVAMIND_API_PORT', raising=False)
    monkeypatch.setenv('NOVAMIND_API_SOCKET', str(server.role_sockets['growth']))
    source = tmp_path / 'metric.py'
    source.write_text('print(1)')
    try:
        invoke(monkeypatch, 'register-daily-script', source)
        assert json.loads(capsys.readouterr().out)['success'] is True
        invoke(monkeypatch, 'list-daily-scripts')
        assert json.loads(capsys.readouterr().out)['data']['scripts'][0]['name'] == source.name
        assert server.get_daily_scripts('ceo') == {}
        invoke(monkeypatch, 'remove-daily-script', source.name)
        assert json.loads(capsys.readouterr().out)['success'] is True
        assert server.get_daily_scripts('growth') == {}
    finally:
        server.stop()


def test_standalone_session_and_source_error(backend, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('NOVAMIND_API_PORT')
    monkeypatch.setattr(_public_cli, '_sessions_dir', lambda: tmp_path)
    session = tmp_path / 'existing'
    session.mkdir()
    (session / 'session.json').write_text('{"created_at": 1}')
    (session / '.server.pid').write_text(str(os.getpid()))
    (session / '.server.port').write_text(str(backend.port))
    invoke(monkeypatch, 'list-daily-scripts')
    assert json.loads(capsys.readouterr().out)['data']['scripts'] == []
    invoke(monkeypatch, 'list-daily-scripts', '--session', 'existing')
    assert json.loads(capsys.readouterr().out)['success'] is True
    with pytest.raises(SystemExit) as failure:
        invoke(monkeypatch, 'register-daily-script', tmp_path / 'absent.py')
    assert failure.value.code == 1
    assert 'Error:' in capsys.readouterr().err
