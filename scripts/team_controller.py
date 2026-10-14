#!/usr/bin/env python3
"""Supervise independent group workers and automatically restore failed attempts."""

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
from saas_bench.host_memory import MemorySampler
from saas_bench.team_recovery import execution_root
from saas_bench.systemd_guard import check_worker_cgroup, memory_bytes, scope_command, unit_status, verify_worker_scope
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
        self.hold = self.directory / 'HOLD'
        self.parent_pid = parent_pid
        self.scope = None
        self.entry = None
        if run_id:
            self.entry = batch.selected(self.manifest(), run_id)

    def stop(self, reason):
        raise RunCancelled(reason)

    def check_stop(self):
        if self.scope is not None:
            try:
                changes = check_worker_cgroup(self.scope)
            except (OSError, ValueError, RuntimeError) as exc:
                changes = [type(exc).__name__ + ': ' + str(exc)]
            if changes and changes != getattr(self, '_scope_changes', None):
                self._scope_changes = changes
                path = Path(self.expected['host_logs']) / 'worker-scope-observations.jsonl'
                try:
                    with path.open('a') as stream:
                        stream.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),
                            run_id=(self.entry or {}).get('run_id'), observations=changes)) + '\n')
                except OSError as exc:
                    print('Worker scope observation unavailable: ' + str(exc), file=sys.stderr)
        if (self.directory / 'STOP').exists():
            self.stop('supervisor_cancelled')
        if self.entry and run_stopped(self.entry):
            raise RunCancelled('run_evidence_fault')
        if self.parent_pid is not None and os.getppid() != self.parent_pid:
            raise RunCancelled('supervisor_exited')

    def manifest(self):
        return identity(self.expected['manifest'], self.expected)

    def __call__(self, request, *, check=lambda: None):
        check()
        self.check_stop()

    def response(self, response):
        if response.status_code != 429:
            return
        response.read()
        if any(word in response.text.lower() for word in (
                'usage limit', 'quota', 'weekly limit', 'monthly limit', 'rolling limit', 'subscription limit')):
            with (self.directory / 'quota-observations.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),
                    status=response.status_code, run_id=(self.entry or {}).get('run_id'))) + '\n')


def latest(entry):
    history = batch.attempts(entry)
    return batch.read(history[-1] / 'attempt.json') if history else None


def run_stopped(entry):
    history = batch.attempts(entry)
    return bool(history and (history[-1] / 'STOP').exists())


def retry_position(entry):
    from saas_bench.team_recovery import latest_checkpoint
    from saas_bench.team_checkpoint import validate_snapshot
    checkpoint = latest_checkpoint(entry, latest(entry))
    if checkpoint is None:
        return [None, None], None
    _, state = validate_snapshot(checkpoint)
    return [state['day'], state['stage']], str(checkpoint)


def record_failure(failures, run_id, position):
    previous = failures.get(run_id, {})
    count = previous.get('count', 0) + 1 if previous.get('position') == position else 1
    failures[run_id] = dict(position=position, count=count)
    return count


def worker(policy, run_id, *, resume, parent_pid, week=None):
    gate = RequestGate(policy, run_id=run_id, parent_pid=parent_pid)
    launch_id = os.environ.get("CEOBENCH_WORKER_LAUNCH_ID", uuid.uuid4().hex)
    scope_receipt = Path(gate.expected["host_logs"]) / f"worker-scope-{run_id}-{launch_id}.json"
    try:
        gate.scope = verify_worker_scope(json.loads(os.environ.pop('CEOBENCH_WORKER_SCOPE_POLICY', 'null')))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        write_json(scope_receipt, dict(verified=False, error_type=type(exc).__name__, error=str(exc)))
        gate.stop('worker_scope_invalid')
    write_json(scope_receipt, dict(verified=True, **gate.scope))
    manifest = gate.manifest()
    entry = batch.selected(manifest, run_id)
    from saas_bench.runtime_temp import prepare_tmpdir
    import tempfile
    os.environ['TMPDIR'] = str(prepare_tmpdir(Path(gate.expected['host_logs']) / 'tmp' / launch_id))
    tempfile.tempdir = os.environ['TMPDIR']
    key = os.environ.get('OPENCODE_API_KEY')
    if not key:
        raise ValueError('Existing OPENCODE_API_KEY is unavailable')
    os.environ['OPENCODE_API_KEY'] = key
    for name in ('BOSSBENCH_LLM_REPLAY_DB', 'ORACLE_MODE', 'CEOBENCH_READ_ONLY_TASK',
                 'CEOBENCH_TEST_PUBLIC', 'CEOBENCH_PUBLIC_DIR', 'NOVAMIND_PUBLIC_DIR',
                 'CEOBENCH_DASHBOARD_URL', 'CEOBENCH_SIMULATOR_USAGE_LOG', 'CEOBENCH_MODEL_SESSION'):
        os.environ.pop(name, None)
    os.environ['CEOBENCH_SIMULATOR_LLM_PROVIDER'] = 'opencode'
    os.environ['CEOBENCH_SIMULATOR_LLM_MODEL'] = batch.MODEL
    options = dict(resume=resume, live=True, request_guard=gate,
                   response_guard=gate.response, parent_pid=parent_pid)
    if week is not None:
        options['week'] = week
    record = batch.prepare(gate.expected['manifest'], run_id, **options)
    print(json.dumps({key: record.get(key) for key in ('run_id', 'attempt', 'status', 'day', 'reason')}), flush=True)
    return 0 if record['status'] == 'finished' else 2


def supervise(policy, *, resume=False, memory_high=16 * 1024**3, memory_max=20 * 1024**3,
              memory_swap_max=0, week=None, run_id=None):
    if week is not None and (type(week) is not int or week < 1):
        raise ValueError('Week must be a positive integer; week 1 starts at day 0')
    scope_command([], 'ceobench-validation.scope', memory_high=memory_high,
                  memory_max=memory_max, memory_swap_max=memory_swap_max)
    policy = Path(policy).resolve()
    gate = RequestGate(policy)
    manifest = gate.manifest()
    if run_id is not None:
        manifest = dict(manifest, runs=[batch.selected(manifest, run_id)])
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=execution_root(gate.expected['manifest'], manifest), text=True).strip()
    directory = policy.parent
    host_logs = Path(gate.expected['host_logs'])
    if not host_logs.is_absolute() or host_logs.resolve() != host_logs:
        raise ValueError('Host logs must have a canonical absolute path')
    host_logs.mkdir(parents=True, exist_ok=True)
    queues = {group: [entry for entry in manifest['runs'] if entry['group'] == group]
              for group in ('git', 'pf')}
    events_path = directory / 'controller-events.jsonl'
    failures, terminal, children, streams = {}, {}, [], []
    units_path = directory / 'controller-units.json'
    units = batch.read(units_path) if units_path.exists() else {}
    launch_ordinal = 0
    launched_runs = set()
    if events_path.exists():
        for line in events_path.read_text().splitlines():
            row = json.loads(line)
            launch_ordinal = max(launch_ordinal, row.get('launch_ordinal', 0))
            if row.get('event') == 'recovery_failure':
                failures[row['run_id']] = dict(position=row['position'], count=row['count'])
    state = {}
    stop_requested = False
    sampler = MemorySampler(host_logs / 'host-memory.jsonl')
    sampler_receipt = dict(closed=True, skipped=True)
    sampler_active = False
    last_poll = 0

    def event(kind, **fields):
        with events_path.open('a') as stream:
            stream.write(json.dumps(dict(fields, time=datetime.now(timezone.utc).isoformat(), event=kind)) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def alarm(entry, reason, **details):
        record = dict(run_id=entry['run_id'], group=entry['group'], reason=reason,
                      time=datetime.now(timezone.utc).isoformat(), **details)
        write_json(directory / ('alarm-' + entry['run_id'] + '.json'), record)
        terminal[entry['run_id']] = dict(disposition='alarm', **record)
        event('group_alarm', **record)

    def observe(slot):
        try:
            observed = scope_state(slot['unit'], exit_code=slot['child'].returncode)
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError, KeyError) as exc:
            observed = dict(slot['unit'], scope_empty=False, cleanup_error=str(exc))
        slot['unit'] = observed
        run_id = slot['entry']['run_id']
        if units.get(run_id) != observed:
            units[run_id] = observed
            write_json(directory / 'controller-units.json', units)
        return observed

    def close_previous(entry):
        saved = units.get(entry['run_id'])
        if not saved or saved.get('scope_empty'):
            return
        deadline = time.monotonic() + 45
        signaled = False
        while True:
            observed = scope_state(saved, exit_code=saved.get('exit_code'))
            units[entry['run_id']] = observed
            write_json(directory / 'controller-units.json', units)
            if observed.get('scope_empty'):
                event('previous_scope_closed', run_id=entry['run_id'], unit=observed)
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError('Previous worker scope did not close')
            if not signaled or remaining < 15:
                signal_scope(observed, signal.SIGTERM if not signaled else signal.SIGKILL)
                signaled = True
            time.sleep(.25)

    def launch(entry, predecessor=None):
        nonlocal launch_ordinal
        close_previous(entry)
        history = batch.attempts(entry)
        mode = 'resume' if history else 'run'
        launch_ordinal += 1
        launch_id = str(launch_ordinal) + '-' + uuid.uuid4().hex[:8]
        ready = host_logs / f'ready-{entry["run_id"]}-{launch_id}.json'
        from saas_bench.runtime_temp import prepare_tmpdir
        temporary = prepare_tmpdir(host_logs / 'tmp' / launch_id)
        root = execution_root(gate.expected['manifest'], manifest)
        env = dict(os.environ, TMPDIR=str(temporary), PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1',
            PYTHONPATH=str(root / 'src'), CEOBENCH_LIFECYCLE_READY=str(ready),
            CEOBENCH_WORKER_LAUNCH_ID=launch_id)
        stream = (host_logs / f'{entry["run_id"]}-{mode}-{launch_id}.log').open('a')
        streams.append(stream)
        command = [sys.executable, str(Path(__file__).resolve()), '--policy', str(policy),
            '--live', '--worker', entry['run_id'], '--parent-pid', str(os.getpid())]
        if history:
            command.append('--resume')
        if week is not None and entry['run_id'] not in launched_runs:
            command.extend(['--week', str(week)])
        unit = 'ceobench-worker-' + uuid.uuid4().hex + '.scope'
        env['CEOBENCH_WORKER_SCOPE_POLICY'] = json.dumps(dict(unit=unit,
            memory_high=memory_high, memory_max=memory_max, memory_swap_max=memory_swap_max))
        command = scope_command(command, unit, memory_high=memory_high,
            memory_max=memory_max, memory_swap_max=memory_swap_max)
        event('worker_launch_intent', run_id=entry['run_id'], mode=mode, unit=unit,
            launch_ordinal=launch_ordinal, group=entry['group'], pair_id=entry.get('pair_id'),
            launch_reason='checkpoint_recovery' if history else 'previous_group_run_terminal' if predecessor else 'initial_group_slot',
            predecessor_run_id=predecessor, source_commit=source_commit,
            frozen_source_commit=manifest.get('source_commit'))
        child = subprocess.Popen(command, cwd=root, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        launched_runs.add(entry['run_id'])
        slot = dict(entry=entry, child=child, ready=ready, unit=dict(unit=unit),
                    closed=False, stopping_at=None, signaled=False)
        children.append(slot)
        event('worker_launched', run_id=entry['run_id'], pid=child.pid, mode=mode,
            launch_ordinal=launch_ordinal, unit=unit, memory_high=memory_high,
            memory_max=memory_max, memory_swap_max=memory_swap_max, scope=observe(slot))
        return slot

    def failed(entry, *, reason, unit=None):
        try:
            position, checkpoint = retry_position(entry)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            position, checkpoint = [None, None], None
            reason += '; checkpoint: ' + str(exc)
        count = record_failure(failures, entry['run_id'], position)
        event('recovery_failure', run_id=entry['run_id'], group=entry['group'], position=position,
              checkpoint=checkpoint, count=count, reason=reason, unit=unit)
        if count >= 3:
            alarm(entry, 'repeated_failure', position=position, failures=count, failure_reason=reason)
            return None
        try:
            return launch(entry)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            return failed(entry, reason=type(exc).__name__ + ': ' + str(exc))

    def next_entry(group, predecessor=None):
        while queues[group]:
            entry = queues[group].pop(0)
            saved = latest(entry)
            try:
                close_previous(entry)
                if week is None and saved and saved.get('status') == 'finished' and units.get(entry['run_id'], {}).get('result') != 'oom-kill':
                    terminal[entry['run_id']] = dict(disposition='terminal', status='finished',
                                                    attempt=saved.get('attempt'))
                    continue
                return launch(entry, predecessor)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                return failed(entry, reason=type(exc).__name__ + ': ' + str(exc))
        return None

    def cleanup(slot, *, requested=False):
        observed = observe(slot)
        child = slot['child']
        residual = child.poll() is not None and not observed.get('scope_empty')
        if observed.get('scope_empty'):
            return True
        if not (requested or residual):
            return False
        if slot['stopping_at'] is None:
            slot['stopping_at'] = time.monotonic()
        elapsed = time.monotonic() - slot['stopping_at']
        try:
            if child.poll() is None and slot['ready'].exists() and not slot['signaled']:
                os.kill(child.pid, signal.SIGUSR1)
                slot['signaled'] = True
            if elapsed > FORCE_STOP_SECONDS or residual and elapsed > 30:
                signal_scope(observed, signal.SIGKILL)
            elif elapsed > COOPERATIVE_STOP_SECONDS or residual:
                signal_scope(observed, signal.SIGTERM)
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            if elapsed > FORCE_STOP_SECONDS:
                alarm(slot['entry'], 'resources_not_closed', error=str(exc), unit=observed)
                slot['closed'] = True
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=10)
                return False
        if elapsed > FORCE_STOP_SECONDS + 30:
            alarm(slot['entry'], 'resources_not_closed', unit=observed)
            slot['closed'] = True
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
        return observe(slot).get('scope_empty', False)

    def telemetry():
        nonlocal last_poll, sampler_active, sampler_receipt
        if sampler_active:
            try:
                sampler.check()
            except (OSError, ValueError, RuntimeError) as exc:
                event('memory_sampler_warning', error=str(exc))
                sampler_receipt = sampler.stop()
                sampler_active = False
        if time.monotonic() - last_poll >= 300:
            try:
                poll(gate.expected['manifest'], directory / 'observer', gate.expected, state)
            except Exception as exc:
                event('observer_warning', error_type=type(exc).__name__, error=str(exc))
            last_poll = time.monotonic()

    with (directory / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def request_stop(*_):
            nonlocal stop_requested
            stop_requested = True
        handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        active = {}
        try:
            try:
                event('memory_sampler_started', sample=sampler.start())
                sampler_active = True
            except (OSError, ValueError, RuntimeError) as exc:
                event('memory_sampler_warning', error=str(exc))
            for group in queues:
                active[group] = next_entry(group)
            while any(slot is not None for slot in active.values()):
                telemetry()
                stopping = stop_requested or (directory / 'STOP').exists()
                for group, slot in list(active.items()):
                    if slot is None:
                        continue
                    child, entry = slot['child'], slot['entry']
                    child.poll()
                    own_stop = run_stopped(entry)
                    closed = cleanup(slot, requested=stopping or own_stop)
                    if slot['closed']:
                        active[group] = None
                        continue
                    if child.poll() is None or not closed:
                        continue
                    child.wait()
                    slot['closed'] = True
                    saved = latest(entry) or {}
                    event('worker_reaped', run_id=entry['run_id'], exit_code=child.returncode,
                        status=saved.get('status'), day=saved.get('day'), unit=slot['unit'])
                    if stopping or own_stop or saved.get('runtime_failure') in ('requested_pause', 'supervisor_cancelled'):
                        active[group] = None
                    elif saved.get('status') == 'finished' and child.returncode == 0 and slot['unit'].get('result') != 'oom-kill':
                        terminal[entry['run_id']] = dict(disposition='terminal', status='finished',
                            unit=slot['unit'], attempt=saved.get('attempt'))
                        event('group_run_terminal', run_id=entry['run_id'], proof=terminal[entry['run_id']])
                        active[group] = next_entry(group, predecessor=entry['run_id'])
                    else:
                        active[group] = failed(entry, reason=str(saved.get('runtime_failure') or
                            saved.get('reason') or 'worker_exit:' + str(child.returncode)), unit=slot['unit'])
                    write_json(directory / 'controller-terminal.json', terminal)
                time.sleep(1)
            return [latest(entry) for entry in manifest['runs']]
        finally:
            for slot in children:
                if slot['closed']:
                    continue
                while not cleanup(slot, requested=True) and not slot['closed']:
                    time.sleep(1)
                slot['child'].wait(timeout=10)
            for stream in streams:
                stream.close()
            if sampler_active:
                sampler_receipt = sampler.stop()
            try:
                poll(gate.expected['manifest'], directory / 'observer', gate.expected, state)
            except Exception as exc:
                event('observer_warning', error_type=type(exc).__name__, error=str(exc))
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            write_json(directory / 'controller-terminal.json', terminal)
            write_json(directory / 'controller-exit.json', dict(hold=False,
                failed=any(record.get('disposition') == 'alarm' for record in terminal.values()) or
                    any((latest(entry) or {}).get('status') != 'finished' for entry in manifest['runs']),
                children_reaped=all(slot['child'].poll() is not None and slot['unit'].get('scope_empty')
                                    for slot in children) and all(unit.get('scope_empty') for unit in units.values()),
                memory_sampler=sampler_receipt, units=units,
                batch_id=manifest['batch_id'], runs=[entry['run_id'] for entry in manifest['runs']],
                auto_resume=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True,
        help='Policy identity anchors the manifest and selected git/PF run IDs and output paths.')
    parser.add_argument('--live', action='store_true', help='Explicitly enable model requests.')
    parser.add_argument('--resume', action='store_true', help='Compatibility flag; existing attempts resume automatically.')
    parser.add_argument('--run-id', help='Run only this group run from the policy selection.')
    parser.add_argument('--week', type=int, help='Replay a saved week boundary; week 1 starts at day 0.')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--parent-pid', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--memory-high', type=memory_bytes, default=16 * 1024**3,
        help='Per-worker cgroup MemoryHigh in bytes or K/M/G/T (default 16G).')
    parser.add_argument('--memory-max', type=memory_bytes, default=20 * 1024**3,
        help='Per-worker cgroup MemoryMax (default 20G). Each group gets a separate scope.')
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
        return worker(args.policy, args.worker, resume=args.resume, parent_pid=args.parent_pid, week=args.week)
    options = dict(resume=args.resume, memory_high=args.memory_high,
        memory_max=args.memory_max, memory_swap_max=args.memory_swap_max)
    if args.week is not None:
        options['week'] = args.week
    if args.run_id is not None:
        options['run_id'] = args.run_id
    supervise(args.policy, **options)
    return 2 if batch.read(args.policy.resolve().parent / 'controller-exit.json')['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
