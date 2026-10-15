import threading
from types import SimpleNamespace

import pytest

from saas_bench.api_server import NovaMindAPIServer
from saas_bench.db_protection import AsyncSaver
from saas_bench.run_state import checkpoint_directory, copy_workspace


def test_async_failure_is_not_a_successful_drain(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('injected disk failure')
    monkeypatch.setattr('saas_bench.db_protection.encrypt_plain_atomic', fail)
    saver = AsyncSaver(tmp_path / 'world.nmdb', key='test-only')
    plain = tmp_path / 'plain'
    plain.write_text('data')
    saver.submit(plain)
    try:
        with pytest.raises(RuntimeError, match='save failed'):
            saver.drain(timeout=2)
    finally:
        with pytest.raises(RuntimeError, match='save failed'):
            saver.shutdown()
        assert not saver._thread.is_alive()


def test_async_drain_waits_and_times_out(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    def delayed(*args, **kwargs):
        started.set()
        release.wait(2)
    monkeypatch.setattr('saas_bench.db_protection.encrypt_plain_atomic', delayed)
    saver = AsyncSaver(tmp_path / 'world.nmdb', key='test-only')
    plain = tmp_path / 'plain'
    plain.write_text('data')
    saver.submit(plain)
    try:
        assert started.wait(2)
        assert not saver.drain(timeout=0.01)
        release.set()
        assert saver.drain(timeout=2)
    finally:
        release.set()
        saver.shutdown()


def test_checkpoint_rejects_inflight_unknown_and_wrong_day(tmp_path):
    server = NovaMindAPIServer(SimpleNamespace(workspace_path=tmp_path, current_day=7))
    server.checkpoint_callback = lambda: {'success': True, 'day': 7}
    assert server.checkpoint(7)['day'] == 7
    with pytest.raises(ValueError, match='day mismatch'):
        server.checkpoint(0)
    with server._advance_lock:
        with pytest.raises(RuntimeError, match='in-flight'):
            server.checkpoint(7)
    server._operation_failed = True
    with pytest.raises(RuntimeError, match='unknown'):
        server.checkpoint(7)


def test_legacy_is_rejected_and_external_symlink_is_warned(tmp_path, caplog):
    with pytest.raises(ValueError, match='legacy'):
        checkpoint_directory(tmp_path, {'day': 7})
    workspace = tmp_path / 'source'
    workspace.mkdir()
    (workspace / 'escape').symlink_to(tmp_path)
    copy_workspace(workspace, tmp_path / 'copy')
    assert 'Workspace symlink escapes the checkpoint' in caplog.text
    assert (tmp_path / 'copy/escape').is_symlink()
    assert (tmp_path / 'copy/escape').resolve() == tmp_path


def test_checkpoint_copy_omits_only_current_session_world(tmp_path):
    workspace = tmp_path / 'workspace'
    current = workspace / 'sessions' / 'current'
    other = workspace / 'sessions' / 'other'
    current.mkdir(parents=True)
    other.mkdir(parents=True)
    (current / 'world.nmdb').write_bytes(b'active')
    (current / 'session.json').write_text('{}')
    (other / 'world.nmdb').write_bytes(b'other')
    (workspace / 'world.nmdb').write_bytes(b'workspace')
    (workspace / 'saved.pid').write_text('123')
    (workspace / '.server.port').write_text('456')
    (workspace / 'compiled.pyc').write_bytes(b'cached')
    (workspace / '__pycache__').mkdir()
    (workspace / '__pycache__/cached.pyc').write_bytes(b'cached')

    snapshot = tmp_path / 'snapshot'
    copy_workspace(workspace, snapshot, omit_session_world='current')

    assert (current / 'world.nmdb').read_bytes() == b'active'
    assert not (snapshot / 'sessions' / 'current' / 'world.nmdb').exists()
    assert (snapshot / 'sessions' / 'current' / 'session.json').read_text() == '{}'
    assert (snapshot / 'sessions' / 'other' / 'world.nmdb').read_bytes() == b'other'
    assert (snapshot / 'world.nmdb').read_bytes() == b'workspace'
    assert not (snapshot / 'saved.pid').exists()
    assert not (snapshot / '.server.port').exists()
    assert not (snapshot / 'compiled.pyc').exists()
    assert not (snapshot / '__pycache__').exists()
