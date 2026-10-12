import importlib.util
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest

from saas_bench.runtime_temp import bind_unix, check_tmpdir, connect_unix, prepare_tmpdir


def test_long_owned_socket_path_roundtrips_and_keeps_file_in_directory(tmp_path):
    root = prepare_tmpdir(tmp_path / ('run-' + 'x' * 120) / 'tmp')
    path = root / 'socket'
    with socket.socket(socket.AF_UNIX) as server, socket.socket(socket.AF_UNIX) as client:
        bind_unix(server, path)
        server.listen()
        connect_unix(client, path)
        connection, _ = server.accept()
        with connection:
            client.sendall(b'owned')
            assert connection.recv(5) == b'owned'
        assert path.is_socket()
    path.unlink()


@pytest.mark.parametrize('bytes_available,inodes,total', [(1, 999999, 1000000), (999999999, 1, 1000000)])
def test_storage_gate_rejects_bytes_and_inode_exhaustion(tmp_path, monkeypatch, bytes_available, inodes, total):
    monkeypatch.setattr('os.statvfs', lambda path: SimpleNamespace(f_bavail=bytes_available,
        f_frsize=1, f_favail=inodes, f_files=total))
    with pytest.raises(OSError, match='Insufficient temporary storage') as error:
        check_tmpdir(tmp_path)
    assert error.value.errno == 28


def test_btrfs_unreported_inode_count_uses_real_file_probe(tmp_path, monkeypatch):
    monkeypatch.setattr('os.statvfs', lambda path: SimpleNamespace(f_bavail=999999999,
        f_frsize=1, f_favail=0, f_files=0))
    assert check_tmpdir(tmp_path)['available_inodes'] is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('failure', ['exit', 'unreadable', 'timeout'])
def test_release_wrapper_cleans_child_temp_after_failure(tmp_path, monkeypatch, failure):
    spec = importlib.util.spec_from_file_location('release_gate', Path(__file__).parents[1] / 'scripts/run_release_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    temporary_root = prepare_tmpdir(tmp_path / 'test-storage')
    monkeypatch.setenv('TMPDIR', str(temporary_root))
    invoke = gate.subprocess.Popen
    def command(argv, **kwargs):
        command = argv[-1]
        if '-m pytest' in command:
            fixture = "import os, pathlib, sys, subprocess, time; p=pathlib.Path(os.environ['TMPDIR']); (p/'child-temp').write_text('test'); print('PARTIAL-TEST', flush=True); subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.2)']); "
            fixture += ("d=p/'sealed'; d.mkdir(); (d/'kept').write_text('frozen'); d.chmod(" +
                        ('0o000' if failure == 'unreadable' else '0o555') + "); sys.exit(4)") if failure != 'timeout' else 'time.sleep(10)'
            import shlex
            target = shlex.join([gate.sys.executable, '-c', fixture])
            args = shlex.split(command)
            args[5] = target
            argv = ['bash', '-c', shlex.join(args)]
        elif 'verify_build' in command:
            import shlex
            args = shlex.split(command)
            args[5] = shlex.join([gate.sys.executable, '-c', 'pass'])
            argv = ['bash', '-c', shlex.join(args)]
        return invoke(argv, **kwargs)
    monkeypatch.setattr(gate.subprocess, 'Popen', command)
    communicate = gate.Boundary.communicate
    def short_timeout(boundary, process, timeout, **kwargs):
        return communicate(boundary, process, .2 if failure == 'timeout' and '-m pytest' in boundary.command else timeout, **kwargs)
    monkeypatch.setattr(gate.Boundary, 'communicate', short_timeout)
    assert gate.run(tmp_path / 'gate', ['tests']) == (124 if failure == 'timeout' else 4)
    result = json.loads((tmp_path / 'gate/release-gate.json').read_text())
    assert result['temporary_cleaned'] is True
    assert result['checks'][-1]['exit_code'] == (124 if failure == 'timeout' else 4)
    assert result['checks'][-1]['process_boundary']['descendants_closed'] is True
    assert Path(result['tmpdir']) == temporary_root
    assert list(temporary_root.iterdir()) == []
    assert (tmp_path / 'gate/check-1-stdout.log').read_bytes() == b'PARTIAL-TEST\n'
