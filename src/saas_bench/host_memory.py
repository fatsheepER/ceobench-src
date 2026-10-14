"""Host memory samples, collected outside the model workers."""

import argparse
from datetime import datetime, timezone
import heapq
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


INTERVAL = 60
STALE_SECONDS = 125
START_SECONDS = 5
STOP_SECONDS = 5
SAMPLE_SECONDS = 5


def _text(path):
    with path.open() as stream:
        value = stream.read(65537)
    if len(value) > 65536:
        raise ValueError('Oversized proc field')
    return value


def _kilobytes(value):
    fields = value.split()
    if len(fields) != 2 or fields[1] != 'kB':
        raise ValueError('Invalid proc byte count')
    count = int(fields[0])
    if count < 0:
        raise ValueError('Negative proc byte count')
    return count * 1024


def _stat(value, pid):
    opening, closing = value.index('('), value.rindex(')')
    if int(value[:opening].strip()) != pid:
        raise ValueError('Process identity changed')
    start = int(value[closing + 1:].split()[19])
    return start, value[opening + 1:closing]


def sample_memory(proc_root=Path('/proc')):
    root = Path(proc_root)
    deadline = time.monotonic() + SAMPLE_SECONDS
    fields = dict(line.split(':', 1) for line in _text(root / 'meminfo').splitlines())
    memory = {key: _kilobytes(fields[name]) for key, name in (
        ('mem_total_bytes', 'MemTotal'), ('mem_available_bytes', 'MemAvailable'),
        ('swap_total_bytes', 'SwapTotal'), ('swap_free_bytes', 'SwapFree'))}
    psi = {}
    for line in _text(root / 'pressure/memory').splitlines():
        kind, *values = line.split()
        values = dict(field.split('=', 1) for field in values)
        row = {name: float(values[name]) for name in ('avg10', 'avg60', 'avg300')}
        row['total_us'] = int(values['total'])
        if any(not math.isfinite(value) or value < 0 for value in row.values()):
            raise ValueError('Invalid memory pressure')
        psi[kind] = row
    if set(psi) != {'some', 'full'}:
        raise ValueError('Incomplete memory pressure')
    leaders, seen = [], 0
    skipped = dict(disappeared=0, denied=0, malformed=0, reused=0)
    complete = True
    with os.scandir(root) as entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            if time.monotonic() >= deadline:
                complete = False
                break
            seen += 1
            pid, directory = int(entry.name), Path(entry.path)
            try:
                texts = []
                for filename in ('stat', 'status', 'cgroup', 'stat'):
                    if time.monotonic() >= deadline:
                        complete = False
                        break
                    texts.append(_text(directory / filename))
                if not complete:
                    break
                start, name = _stat(texts[0], pid)
                status = dict(line.split(':', 1) for line in texts[1].splitlines())
                rss = _kilobytes(status['VmRSS'])
                groups = texts[2].splitlines()
                group = next(line[3:] for line in groups if line.startswith('0::'))
                if _stat(texts[3], pid)[0] != start:
                    skipped['reused'] += 1
                    continue
                row = (rss, pid, start, name, group)
                if len(leaders) < 10:
                    heapq.heappush(leaders, row)
                elif row > leaders[0]:
                    heapq.heapreplace(leaders, row)
            except FileNotFoundError:
                skipped['disappeared'] += 1
            except PermissionError:
                skipped['denied'] += 1
            except (OSError, ValueError, KeyError, IndexError, StopIteration):
                skipped['malformed'] += 1
    return dict(memory=memory, psi=psi, processes_seen=seen, skipped=skipped,
        top_processes_complete=complete and not any(skipped.values()), errors=[], top_processes=[
            dict(pid=pid, comm=name, rss_bytes=rss, starttime_ticks=start, cgroup=group)
            for rss, pid, start, name, group in sorted(leaders, reverse=True)])


class MemorySampler:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.child = None
        self._buffer = b''
        self._heartbeat = None
        self._receipt = None

    def _drain(self):
        for _ in range(16):
            try:
                data = os.read(self.child.stdout.fileno(), 4096)
            except BlockingIOError:
                return
            if not data:
                return
            self._buffer += data
            if len(self._buffer) > 4096:
                raise RuntimeError('Memory sampler heartbeat exceeds bound')
            while b'\n' in self._buffer:
                line, self._buffer = self._buffer.split(b'\n', 1)
                try:
                    row = json.loads(line)
                    valid = (set(row) == {'sequence', 'monotonic', 'pid', 'parent_pid'} and
                        type(row['sequence']) is int and row['sequence'] > 0 and
                        type(row['monotonic']) in (float, int) and math.isfinite(row['monotonic']) and
                        self._started <= row['monotonic'] <= time.monotonic() and
                        row['pid'] == self.child.pid and row['parent_pid'] == os.getpid() and
                        (self._heartbeat is None or row['sequence'] > self._heartbeat['sequence']))
                except (ValueError, KeyError, TypeError):
                    valid = False
                if not valid:
                    raise RuntimeError('Invalid memory sampler heartbeat')
                self._heartbeat = row
        raise RuntimeError('Memory sampler heartbeat stream exceeds bound')

    def start(self):
        if self.child is not None or self._receipt is not None:
            raise RuntimeError('Memory sampler already started')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._started = time.monotonic()
        self.child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
            str(self.path), '--parent-pid', str(os.getpid())], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE)
        os.set_blocking(self.child.stdout.fileno(), False)
        deadline = self._started + START_SECONDS
        try:
            while self._heartbeat is None:
                self._drain()
                if self.child.poll() is not None:
                    raise RuntimeError('Memory sampler exited before first sample')
                if self._heartbeat is not None:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError('Memory sampler first sample timed out')
                select.select([self.child.stdout], [], [], remaining)
            return self.check()
        except BaseException:
            self.stop()
            raise

    def check(self):
        if self.child is None or self._receipt is not None:
            raise RuntimeError('Memory sampler is not running')
        self._drain()
        if self.child.poll() is not None:
            raise RuntimeError('Memory sampler exited')
        if self._heartbeat is None or time.monotonic() - self._heartbeat['monotonic'] > STALE_SECONDS:
            raise RuntimeError('Memory sampler heartbeat is stale')
        return dict(self._heartbeat)

    def stop(self):
        if self._receipt is not None:
            return dict(self._receipt)
        forced = False
        unexpected_exit = self.child is not None and self.child.poll() is not None
        if self.child is not None:
            if self.child.poll() is None:
                try:
                    self.child.terminate()
                except ProcessLookupError:
                    pass
                try:
                    self.child.wait(timeout=STOP_SECONDS)
                except subprocess.TimeoutExpired:
                    forced = True
                    try:
                        self.child.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        self.child.wait(timeout=STOP_SECONDS)
                    except subprocess.TimeoutExpired:
                        pass
            self.child.stdout.close()
        self._receipt = dict(pid=self.child.pid if self.child else None,
            exit_code=self.child.returncode if self.child else None,
            closed=self.child is None or self.child.returncode is not None,
            forced=forced, unexpected_exit=unexpected_exit, samples_path=str(self.path))
        return dict(self._receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--parent-pid', type=int, required=True)
    args = parser.parse_args(argv)
    if args.parent_pid != os.getppid():
        raise RuntimeError('Memory sampler parent changed before startup')
    parent = os.pidfd_open(args.parent_pid)

    def terminate(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    sequence, scheduled = 0, time.monotonic()
    try:
        with args.path.open('a') as stream:
            while not select.select([parent], [], [], 0)[0]:
                row = sample_memory()
                sequence += 1
                stamp = time.monotonic()
                skipped = max(0, int((stamp - scheduled) // INTERVAL))
                row.update(version=1, sequence=sequence,
                    time=datetime.now(timezone.utc).isoformat(), monotonic=stamp,
                    collector_pid=os.getpid(), parent_pid=args.parent_pid, skipped_intervals=skipped)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                print(json.dumps(dict(sequence=sequence, monotonic=stamp,
                    pid=os.getpid(), parent_pid=args.parent_pid)), flush=True)
                scheduled += (skipped + 1) * INTERVAL
                if select.select([parent], [], [], max(0, scheduled - time.monotonic()))[0]:
                    break
    finally:
        os.close(parent)


if __name__ == '__main__':
    main()
