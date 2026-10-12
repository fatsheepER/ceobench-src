#!/usr/bin/env python3
"""Supervise one frozen git/PF pair. Recovery requires explicit --resume."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from saas_bench import team_batch as batch
from saas_bench.run_lifecycle import RunCancelled
from saas_bench.run_state import write_json
from saas_bench.team_recovery import execution_root, hold_path, recovery_source
from saas_bench.systemd_guard import memory_bytes, scope_command, unit_status
from team_observer import identity, poll


COOPERATIVE_STOP_SECONDS = 660
FORCE_STOP_SECONDS = COOPERATIVE_STOP_SECONDS + 30


def scope_state(saved, *, exit_code=None):
    unit = saved['unit']
    record = unit_status(unit, invocation_id=saved.get('invocation_id'), exit_code=exit_code)
    response = subprocess.run(['systemctl', '--user', 'show', unit,
        '--property=LoadState,ActiveState,InvocationID,ControlGroup'],
        capture_output=True, text=True, timeout=10)
    if response.returncode:
        raise RuntimeError('Cannot observe worker scope: ' + response.stderr.strip())
    properties = dict(line.split('=', 1) for line in response.stdout.splitlines() if '=' in line)
    invocation, group = properties.get('InvocationID'), properties.get('ControlGroup')
    if ((invocation and saved.get('invocation_id') and invocation != saved['invocation_id']) or
            (group and saved.get('control_group') and group != saved['control_group'])):
        raise RuntimeError('Worker scope identity changed; cleanup refused')
    if invocation and record.get('invocation_id') not in (None, invocation):
        raise RuntimeError('Worker scope observations disagree on invocation; cleanup refused')
    record['invocation_id'] = invocation or saved.get('invocation_id') or record.get('invocation_id')
    record['control_group'] = group or saved.get('control_group')
    active = properties.get('ActiveState') in ('active', 'activating', 'deactivating')
    if active and (not invocation or not group):
        raise RuntimeError('Active worker scope has no invocation/cgroup binding')
    if record['control_group']:
        root = Path('/sys/fs/cgroup')
        path = root / record['control_group'].lstrip('/')
        if not path.is_relative_to(root) or '..' in path.parts:
            raise RuntimeError('Invalid worker cgroup path')
        try:
            events = dict(line.split() for line in (path / 'cgroup.events').read_text().splitlines())
            record['scope_empty'] = events['populated'] == '0'
        except FileNotFoundError:
            if active:
                raise RuntimeError('Active worker cgroup is unavailable') from None
            record['scope_empty'] = True
    else:
        record['scope_empty'] = not active and bool(record['invocation_id'])
    return record


def signal_scope(saved, signum):
    current = scope_state(saved)
    if current['scope_empty']:
        return current
    if not saved.get('invocation_id') or not saved.get('control_group'):
        raise RuntimeError('Cannot signal an unbound worker scope')
    response = subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=all',
        '--signal=' + signal.Signals(signum).name, saved['unit']],
        capture_output=True, text=True, timeout=10)
    if response.returncode:
        current = scope_state(saved)
        if not current['scope_empty']:
            raise RuntimeError('Worker scope signal failed: ' + response.stderr.strip())
    return current


class RequestGate:
    def __init__(self, policy, *, run_id=None, parent_pid=None):
        self.directory = Path(policy).resolve().parent
        self.expected = batch.read(policy)['identity']
        self.hold = hold_path(self.expected['manifest'])
        self.parent_pid = parent_pid
        self.entry = None
        if run_id:
            self.entry = batch.selected(self.manifest(), run_id)

    def stop(self, reason):
        self.hold.touch()
        raise RunCancelled(reason)

    def check_stop(self):
        if any(path.exists() for path in (self.hold, self.directory / 'STOP',
                                         self.directory / 'QUOTA_EXHAUSTED')):
            self.stop('supervisor_cancelled')
        if self.entry and run_stopped(self.entry):
            raise RunCancelled('run_evidence_fault')
        if self.parent_pid is not None and os.getppid() != self.parent_pid:
            raise RunCancelled('supervisor_exited')

    def manifest(self):
        try:
            manifest = identity(self.expected['manifest'], self.expected)
            if manifest.get('controller_sha256') and Path(__file__).resolve() != execution_root(self.expected['manifest'], manifest) / 'scripts/team_controller.py':
                raise ValueError('Controller is outside the frozen source root')
            return manifest
        except (OSError, ValueError, KeyError, TypeError):
            self.stop('frozen_identity_changed')

    def __call__(self, request, *, check=lambda: None):
        while True:
            check()
            self.check_stop()
            try:
                health = batch.read(self.directory / 'observer/health.json')
            except (OSError, ValueError):
                time.sleep(.25)
                continue
            if not isinstance(health, dict):
                self.stop('frozen_identity_changed')
            for key in ('manifest', 'manifest_sha256', 'batch_id'):
                if health.get(key) != self.expected[key]:
                    self.stop('frozen_identity_changed')
            if self.expected.get('runs') is not None and health.get('runs') != self.expected['runs']:
                self.stop('frozen_identity_changed')
            status = health.get('status')
            if status not in ('healthy', 'pending'):
                self.stop('supervisor_cancelled')
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(health['time'])).total_seconds()
            except (KeyError, TypeError, ValueError):
                self.stop('frozen_identity_changed')
            if status == 'healthy' and 0 <= age <= 330:
                return
            time.sleep(.25)

    def response(self, response):
        if response.status_code != 429:
            return
        response.read()
        if any(word in response.text.lower() for word in (
                'usage limit', 'quota', 'weekly limit', 'monthly limit', 'rolling limit', 'subscription limit')):
            (self.directory / 'QUOTA_EXHAUSTED').touch()
            self.hold.touch()


def latest(entry):
    history = batch.attempts(entry)
    return batch.read(history[-1] / 'attempt.json') if history else None


def run_stopped(entry):
    history = batch.attempts(entry)
    return bool(history and (history[-1] / 'STOP').exists())


def worker(policy, run_id, *, resume, parent_pid):
    gate = RequestGate(policy, run_id=run_id, parent_pid=parent_pid)
    manifest = gate.manifest()
    batch.selected(manifest, run_id)
    from saas_bench.agents.bash_agent.run_test import load_env_file
    key = os.environ.get('OPENCODE_API_KEY') or load_env_file(Path(manifest['source_root']) / '.env').get('OPENCODE_API_KEY')
    if not key:
        raise ValueError('Existing OPENCODE_API_KEY is unavailable')
    os.environ['OPENCODE_API_KEY'] = key
    for name in ('BOSSBENCH_LLM_REPLAY_DB', 'ORACLE_MODE', 'CEOBENCH_READ_ONLY_TASK',
                 'CEOBENCH_TEST_PUBLIC', 'CEOBENCH_PUBLIC_DIR', 'NOVAMIND_PUBLIC_DIR',
                 'CEOBENCH_DASHBOARD_URL', 'CEOBENCH_SIMULATOR_USAGE_LOG', 'CEOBENCH_MODEL_SESSION'):
        os.environ.pop(name, None)
    os.environ['CEOBENCH_SIMULATOR_LLM_PROVIDER'] = 'opencode'
    os.environ['CEOBENCH_SIMULATOR_LLM_MODEL'] = batch.MODEL
    record = batch.prepare(gate.expected['manifest'], run_id, resume=resume, live=True,
        request_guard=gate, response_guard=gate.response, parent_pid=parent_pid)
    print(json.dumps({key: record.get(key) for key in ('run_id', 'attempt', 'status', 'day', 'reason')}), flush=True)
    return 0 if record['status'] == 'finished' else 2


def supervise(policy, *, resume=False, memory_high=2 * 1024**3, memory_max=3 * 1024**3, memory_swap_max=0):
    scope_command([], 'ceobench-validation.scope', memory_high=memory_high,
                  memory_max=memory_max, memory_swap_max=memory_swap_max)
    policy = Path(policy).resolve()
    gate = RequestGate(policy)
    manifest = gate.manifest()
    directory = policy.parent
    host_logs = Path(gate.expected['host_logs'])
    if not host_logs.is_absolute() or host_logs.resolve() != host_logs:
        raise ValueError('Host logs must have a canonical absolute path')
    host_logs.mkdir(parents=True, exist_ok=True)
    entries = []
    for entry in manifest['runs']:
        saved = latest(entry)
        if resume and saved and saved['status'] == 'finished':
            continue
        if bool(saved) != resume:
            raise ValueError('Use fresh execution or explicit --resume for existing attempts')
        audited = recovery_source(gate.expected['manifest'], entry, saved, batch.attempts(entry)[-1]) if saved and resume else None
        if saved and not audited and saved['status'] not in ('paused', 'failed'):
            raise ValueError('Explicit resume requires a paused or failed attempt')
        entries.append(entry)
    statefile = directory / 'observer/state.json'
    state = batch.read(statefile) if statefile.exists() else {}
    children, streams = [], []
    units = {}
    stopped_at, signaled, exited = {}, set(), set()

    def event(kind, **fields):
        with (directory / 'controller-events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), event=kind, **fields)) + '\n')

    def stopped():
        return any(path.exists() for path in (gate.hold, directory / 'STOP', directory / 'QUOTA_EXHAUSTED'))

    def observe_unit(entry, child):
        saved = units[entry['run_id']]
        if saved.get('cleanup_error'):
            return saved
        try:
            record = scope_state(saved, exit_code=child.returncode)
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError, KeyError) as exc:
            record = dict(saved, scope_empty=False, cleanup_error=str(exc))
            gate.hold.touch()
            event('scope_cleanup_refused', run_id=entry['run_id'], unit=record)
        units[entry['run_id']] = record
        write_json(directory / 'controller-units.json', units)
        return record

    def workers_running():
        return any(child.poll() is None or (not observe_unit(entry, child)['scope_empty'] and
                   not units[entry['run_id']].get('cleanup_error')) for entry, child, _ in children)

    def stop_children(*, global_stop=False):
        if global_stop:
            gate.hold.touch()
        for entry, child, ready in children:
            unit = observe_unit(entry, child)
            if unit.get('cleanup_error'):
                if child.poll() is None:
                    child.kill()
                continue
            residual = child.poll() is not None and not unit['scope_empty']
            if unit['scope_empty'] or not (residual or global_stop or run_stopped(entry)):
                continue
            elapsed = time.monotonic() - stopped_at.setdefault(child.pid, time.monotonic())
            try:
                if child.poll() is None and ready.exists() and child.pid not in signaled:
                    os.kill(child.pid, signal.SIGUSR1)
                    signaled.add(child.pid)
                if elapsed > FORCE_STOP_SECONDS or residual and elapsed > 30:
                    signal_scope(unit, signal.SIGKILL)
                elif elapsed > COOPERATIVE_STOP_SECONDS or residual:
                    signal_scope(unit, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                unit.update(cleanup_error=str(exc), scope_empty=False)
                gate.hold.touch()
                event('scope_cleanup_refused', run_id=entry['run_id'], unit=unit)
                if child.poll() is None:
                    child.kill()

    with (directory / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handlers = {sig: signal.signal(sig, lambda *_: gate.hold.touch())
                    for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            health = poll(gate.expected['manifest'], directory / 'observer', gate.expected, state)
            last_poll = time.monotonic()
            while health['status'] == 'pending' and not stopped():
                time.sleep(1)
                if time.monotonic() - last_poll >= 300:
                    health = poll(gate.expected['manifest'], directory / 'observer', gate.expected, state)
                    last_poll = time.monotonic()
            if health['status'] == 'hold' or stopped():
                gate.hold.touch()
                return []
            for entry in entries:
                if stopped():
                    break
                mode = 'resume' if resume else 'run'
                ready = host_logs / f'ready-{entry["run_id"]}-{mode}.json'
                ready.unlink(missing_ok=True)
                env = dict(os.environ, PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1',
                    PYTHONPATH=str(execution_root(gate.expected['manifest'], manifest) / 'src'),
                    CEOBENCH_LIFECYCLE_READY=str(ready))
                stream = (host_logs / f'{entry["run_id"]}-{mode}.log').open('a')
                streams.append(stream)
                command = [sys.executable, str(Path(__file__).resolve()), '--policy', str(policy),
                    '--live', '--worker', entry['run_id'], '--parent-pid', str(os.getpid())]
                if resume:
                    command.append('--resume')
                unit = 'ceobench-worker-' + uuid.uuid4().hex + '.scope'
                command = scope_command(command, unit, memory_high=memory_high,
                    memory_max=memory_max, memory_swap_max=memory_swap_max)
                units[entry['run_id']] = dict(unit=unit)
                write_json(directory / 'controller-units.json', units)
                child = subprocess.Popen(command, cwd=manifest['source_root'], env=env,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                children.append((entry, child, ready))
                deadline = time.monotonic() + 5
                while not observe_unit(entry, child).get('invocation_id') and child.poll() is None:
                    if units[entry['run_id']].get('cleanup_error'):
                        raise RuntimeError(units[entry['run_id']]['cleanup_error'])
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Worker scope invocation did not become observable')
                    time.sleep(.01)
                event('worker_launched', run_id=entry['run_id'], pid=child.pid, mode=mode, unit=unit,
                    memory_high=memory_high, memory_max=memory_max, memory_swap_max=memory_swap_max)
            while workers_running():
                if time.monotonic() - last_poll >= 300:
                    poll(gate.expected['manifest'], directory / 'observer', gate.expected, state)
                    last_poll = time.monotonic()
                for entry, child, _ in children:
                    if child.poll() is not None and child.pid not in exited:
                        exited.add(child.pid)
                        record = latest(entry) or {}
                        unit = observe_unit(entry, child)
                        event('worker_exited', run_id=entry['run_id'], exit_code=child.returncode,
                            status=record.get('status'), day=record.get('day'), unit=unit)
                stop_children(global_stop=stopped())
                time.sleep(1)
            for entry, child, _ in children:
                child.wait()
                record = latest(entry) or {}
                unit = observe_unit(entry, child)
                event('worker_reaped', run_id=entry['run_id'], exit_code=child.returncode,
                    status=record.get('status'), day=record.get('day'), unit=unit)
            if stopped():
                gate.hold.touch()
            health = poll(gate.expected['manifest'], directory / 'observer', gate.expected, state)
            for run_id, observed in health.get('units', {}).items():
                units[run_id] = dict(units.get(run_id, {}), **observed)
            return [dict(latest(entry) or {}, status='failed', reason='oom-kill')
                    if units[entry['run_id']]['result'] == 'oom-kill' else latest(entry)
                    for entry, _, _ in children]
        except BaseException:
            gate.hold.touch()
            raise
        finally:
            while workers_running():
                stop_children(global_stop=True)
                time.sleep(1)
            for _, child, _ in children:
                child.wait()
            for stream in streams:
                stream.close()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            write_json(directory / 'controller-exit.json', dict(hold=gate.hold.exists(),
                failed=gate.hold.exists() or any(record.get('result') != 'success' or record.get('cleanup_error')
                    for record in units.values()) or
                    any(child.returncode != 0 or
                    (latest(entry) or {}).get('status') != 'finished' for entry, child, _ in children),
                children_reaped=all(child.returncode is not None and units[entry['run_id']].get('scope_empty')
                    for entry, child, _ in children),
                units=units, batch_id=manifest['batch_id'], runs=[entry['run_id'] for entry in manifest['runs']]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True,
        help='Policy identity anchors the manifest and selected git/PF run IDs and output paths.')
    parser.add_argument('--live', action='store_true', help='Explicitly enable model requests.')
    parser.add_argument('--resume', action='store_true', help='Explicitly resume paused or failed runs; skip finished runs.')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--parent-pid', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--memory-high', type=memory_bytes, default=2 * 1024**3,
        help='Per-worker cgroup MemoryHigh in bytes or K/M/G/T (default 2G).')
    parser.add_argument('--memory-max', type=memory_bytes, default=3 * 1024**3,
        help='Per-worker cgroup MemoryMax (default 3G). Each group gets a separate scope.')
    parser.add_argument('--memory-swap-max', type=memory_bytes, default=0,
        help='Per-worker cgroup MemorySwapMax (default 0).')
    parser.add_argument('--bash-memory-max', type=memory_bytes,
        help='Per-process Bash address-space limit; default 1G or CEOBENCH_BASH_MEMORY_MAX_BYTES.')
    args = parser.parse_args(argv)
    if not args.live:
        parser.error('Execution requires explicit --live')
    if args.bash_memory_max is not None:
        from saas_bench.process_boundary import command_memory_limit
        os.environ['CEOBENCH_BASH_MEMORY_MAX_BYTES'] = str(command_memory_limit(args.bash_memory_max))
    if args.worker:
        if args.parent_pid is None:
            parser.error('Workers require their controller parent PID')
        return worker(args.policy, args.worker, resume=args.resume, parent_pid=args.parent_pid)
    supervise(args.policy, resume=args.resume, memory_high=args.memory_high,
        memory_max=args.memory_max, memory_swap_max=args.memory_swap_max)
    return 2 if batch.read(args.policy.resolve().parent / 'controller-exit.json')['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
