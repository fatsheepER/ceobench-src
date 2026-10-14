import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'src/saas_bench/host_memory.py'
spec = importlib.util.spec_from_file_location('host_memory_test', SOURCE)
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


@pytest.fixture
def proc(tmp_path):
    (tmp_path / 'meminfo').write_text('MemTotal: 1000 kB\nMemAvailable: 200 kB\n'
        'SwapTotal: 300 kB\nSwapFree: 100 kB\n')
    (tmp_path / 'pressure').mkdir()
    (tmp_path / 'pressure/memory').write_text(
        'some avg10=0.10 avg60=0.20 avg300=0.30 total=10\n'
        'full avg10=0.01 avg60=0.02 avg300=0.03 total=2\n')
    return tmp_path


def process(root, pid, rss, *, name='worker (name)'):
    directory = root / str(pid)
    directory.mkdir()
    fields = ['S'] + ['0'] * 18 + [str(pid * 10)]
    (directory / 'stat').write_text(f'{pid} ({name}) ' + ' '.join(fields))
    (directory / 'status').write_text(f'Name:\t{name}\nVmRSS:\t{rss} kB\n')
    (directory / 'cgroup').write_text(f'0::/test/{pid}\n')
    return directory


def test_sample_has_bytes_pressure_and_only_top_ten(proc):
    for pid in range(1, 14):
        process(proc, pid, pid * 2)
    row = host.sample_memory(proc)
    assert row['memory'] == dict(mem_total_bytes=1024000, mem_available_bytes=204800,
        swap_total_bytes=307200, swap_free_bytes=102400)
    assert row['psi']['full'] == dict(avg10=0.01, avg60=0.02, avg300=0.03, total_us=2)
    assert [item['pid'] for item in row['top_processes']] == list(range(13, 3, -1))
    assert row['top_processes'][0] == dict(pid=13, comm='worker (name)', rss_bytes=26624,
        starttime_ticks=130, cgroup='/test/13')
    assert row['processes_seen'] == 13 and row['top_processes_complete']
    assert not any(row['skipped'].values()) and row['errors'] == []


def test_process_races_are_partial_and_reused_identity_is_discarded(proc, monkeypatch):
    process(proc, 1, 10)
    process(proc, 2, 20)
    (proc / '3').mkdir()
    original = host._stat
    visits = {}

    def stat(path, pid):
        start, name = original(path, pid)
        visits[pid] = visits.get(pid, 0) + 1
        return start + (pid == 2 and visits[pid] > 1), name

    monkeypatch.setattr(host, '_stat', stat)
    row = host.sample_memory(proc)
    assert [item['pid'] for item in row['top_processes']] == [1]
    assert row['skipped']['reused'] == row['skipped']['disappeared'] == 1


def test_denied_and_malformed_processes_are_counted(proc, monkeypatch):
    denied = process(proc, 1, 10)
    malformed = process(proc, 2, 20)
    (malformed / 'status').write_text('Name: worker\n')
    original = host._text

    def text(path):
        if path.parent == denied:
            raise PermissionError('denied')
        return original(path)

    monkeypatch.setattr(host, '_text', text)
    row = host.sample_memory(proc)
    assert row['top_processes'] == []
    assert row['skipped']['denied'] == row['skipped']['malformed'] == 1


@pytest.mark.parametrize('field', ['meminfo', 'pressure/memory'])
def test_essential_fields_fail_instead_of_becoming_zero(proc, field):
    (proc / field).write_text('missing: invalid\n')
    with pytest.raises((KeyError, ValueError)):
        host.sample_memory(proc)


def test_scan_deadline_marks_incomplete_top_list(proc, monkeypatch):
    process(proc, 1, 10)
    clock = iter((0, 6))
    monkeypatch.setattr(host, 'time', SimpleNamespace(monotonic=lambda: next(clock)))
    row = host.sample_memory(proc)
    assert not row['top_processes_complete']
    assert row['processes_seen'] == 0 and row['top_processes'] == []


def test_scan_checks_deadline_between_process_files(proc, monkeypatch):
    process(proc, 1, 10)
    clock = iter((0, 0, 0, 6))
    monkeypatch.setattr(host, 'time', SimpleNamespace(monotonic=lambda: next(clock)))
    row = host.sample_memory(proc)
    assert not row['top_processes_complete']
    assert row['processes_seen'] == 1 and row['top_processes'] == []


def test_stopped_sampler_cannot_start_and_leak_a_new_child(tmp_path):
    sampler = host.MemorySampler(tmp_path / 'samples.jsonl')
    assert sampler.stop()['closed']
    with pytest.raises(RuntimeError, match='already started'):
        sampler.start()
    assert sampler.child is None


def test_real_start_proves_current_durable_row_and_stop_reaps(tmp_path):
    path = tmp_path / 'samples.jsonl'
    path.write_text('{"sequence":999,"parent_pid":0}\n')
    sampler = host.MemorySampler(path)
    try:
        proof = sampler.start()
        row = json.loads(path.read_text().splitlines()[-1])
        assert proof['sequence'] == row['sequence'] == 1
        assert proof['pid'] == row['collector_pid'] == sampler.child.pid
        assert proof['parent_pid'] == row['parent_pid'] == os.getpid()
        assert proof['monotonic'] == row['monotonic']
        assert sampler.check() == proof
    finally:
        receipt = sampler.stop()
    assert receipt == dict(pid=sampler.child.pid, exit_code=0, closed=True,
        forced=False, unexpected_exit=False, samples_path=str(path))
    assert sampler.stop() == receipt
    with pytest.raises(ChildProcessError):
        os.waitpid(sampler.child.pid, os.WNOHANG)


def test_live_child_stale_heartbeat_is_refused(tmp_path, monkeypatch):
    sampler = host.MemorySampler(tmp_path / 'samples.jsonl')
    try:
        proof = sampler.start()
        monkeypatch.setattr(host, 'time', SimpleNamespace(
            monotonic=lambda: proof['monotonic'] + host.STALE_SECONDS + 1))
        with pytest.raises(RuntimeError, match='stale'):
            sampler.check()
    finally:
        assert sampler.stop()['closed']


def test_dead_child_is_refused_even_with_fresh_heartbeat(tmp_path):
    sampler = host.MemorySampler(tmp_path / 'samples.jsonl')
    try:
        sampler.start()
        sampler.child.kill()
        sampler.child.wait(timeout=5)
        with pytest.raises(RuntimeError, match='exited'):
            sampler.check()
    finally:
        assert sampler.stop()['closed']


def test_stopped_child_is_killed_and_reaped_with_bounded_stop(tmp_path, monkeypatch):
    sampler = host.MemorySampler(tmp_path / 'samples.jsonl')
    sampler.start()
    os.kill(sampler.child.pid, signal.SIGSTOP)
    monkeypatch.setattr(host, 'STOP_SECONDS', 0.1)
    started = time.monotonic()
    receipt = sampler.stop()
    assert receipt['forced'] and receipt['closed'] and receipt['exit_code'] == -signal.SIGKILL
    assert time.monotonic() - started < 2


@pytest.mark.parametrize('payload, message', [('', 'timed out'),
    ('x' * 4097, 'exceeds bound'), ('{}\n', 'Invalid')])
def test_unready_or_invalid_child_is_closed(tmp_path, monkeypatch, payload, message):
    real_popen = subprocess.Popen

    def popen(argv, **kwargs):
        code = f'import sys,time; sys.stdout.write({payload!r}); sys.stdout.flush(); time.sleep(30)'
        return real_popen([sys.executable, '-c', code], **kwargs)

    monkeypatch.setattr(host, 'subprocess', SimpleNamespace(Popen=popen,
        PIPE=subprocess.PIPE, DEVNULL=subprocess.DEVNULL, TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(host, 'START_SECONDS', 0.15)
    sampler = host.MemorySampler(tmp_path / 'samples.jsonl')
    with pytest.raises(RuntimeError, match=message):
        sampler.start()
    assert sampler.stop()['closed']


def test_fixed_cadence_runs_in_child_without_controller_polling(tmp_path):
    code = f'''import importlib.util,os
spec=importlib.util.spec_from_file_location('sampler', {str(SOURCE)!r})
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
module.INTERVAL=0.05
module.main([{str(tmp_path / 'cadence.jsonl')!r}, '--parent-pid', str(os.getppid())])
'''
    child = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE)
    try:
        rows = []
        for _ in range(2):
            assert select.select([child.stdout], [], [], 5)[0]
            rows.append(json.loads(child.stdout.readline()))
        assert [row['sequence'] for row in rows] == [1, 2]
        assert rows[1]['monotonic'] > rows[0]['monotonic']
    finally:
        child.terminate()
        child.wait(timeout=5)
        child.stdout.close()


def test_parent_exit_wakes_sampler_and_orphan_is_reaped(tmp_path):
    owner = f'''import importlib.util,json,os
spec=importlib.util.spec_from_file_location('sampler', {str(SOURCE)!r})
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
sampler=module.MemorySampler({str(tmp_path / 'orphan.jsonl')!r})
proof=sampler.start()
print(json.dumps(proof), flush=True)
os._exit(0)
'''
    # A separate subreaper owns this test's orphan without changing pytest's process.
    supervisor = f'''import ctypes,json,os,subprocess,sys
assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
owner=subprocess.Popen([sys.executable, '-c', {owner!r}], stdout=subprocess.PIPE)
proof=json.loads(owner.stdout.readline()); owner.wait(timeout=5)
pid,status=os.waitpid(proof['pid'], 0)
print(json.dumps(dict(pid=pid, status=status)), flush=True)
'''
    started = time.monotonic()
    output = subprocess.run([sys.executable, '-c', supervisor], capture_output=True,
        text=True, check=True, timeout=10)
    assert json.loads(output.stdout)['status'] == 0
    assert time.monotonic() - started < 10
