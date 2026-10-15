#!/usr/bin/env python3
"""Run selected Git/PF groups independently with automatic checkpoint recovery."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import logging
from pathlib import Path

from saas_bench import team_batch as batch
from saas_bench.run_state import write_json
from saas_bench.run_warnings import AttemptWarnings
from saas_bench.systemd_guard import memory_bytes
import team_controller as controller
from team_observer import identity


def verify_controls(policy):
    policy = Path(policy).resolve()
    config = batch.read(policy)
    expected = config['identity']
    manifest = identity(expected['manifest'], expected)
    return config, manifest


def main(argv=None):
    with AttemptWarnings() as warnings:
        return _main(argv, warnings)


def _main(argv, warnings):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--live', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Compatibility flag; recovery is automatic.')
    parser.add_argument('--force-resume', action='store_true', help='Record explicit continuation despite validation warnings.')
    parser.add_argument('--run-id', help='Run only this group run from the policy selection.')
    parser.add_argument('--week', type=int, help='Replay a saved week boundary; week 1 starts at day 0.')
    parser.add_argument('--memory-high', type=memory_bytes, default=16 * 1024**3)
    parser.add_argument('--memory-max', type=memory_bytes, default=20 * 1024**3)
    parser.add_argument('--memory-swap-max', type=memory_bytes, default=0)
    args = parser.parse_args(argv)
    if args.resume and not args.live:
        parser.error('--resume requires --live')
    config, manifest = verify_controls(args.policy)
    if args.week is not None and args.week < 1:
        parser.error('--week must be positive; week 1 starts at day 0')
    if args.run_id is not None:
        manifest = dict(manifest, runs=[batch.selected(manifest, args.run_id)])
    if args.check:
        print(json.dumps(dict(status='validated', batch_id=manifest['batch_id'],
            scheduling='group_independent', runs=[row['run_id'] for row in manifest['runs']],
            memory_high=args.memory_high, memory_max=args.memory_max,
            model_requests=0, credential_read=False, new_attempts=0)))
        return 0
    directory = args.policy.resolve().parent
    outcome = 2
    with (directory / 'driver.lock').open('a') as lock:
        from saas_bench.run_warnings import advisory
        with advisory():
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            options = dict(resume=args.resume, force_resume=args.force_resume, startup_warnings=warnings.rows, memory_high=args.memory_high,
                memory_max=args.memory_max, memory_swap_max=args.memory_swap_max)
            if args.week is not None:
                options['week'] = args.week
            if args.run_id is not None:
                options['run_id'] = args.run_id
            controller.supervise(args.policy, **options)
            exit_record = batch.read(directory / 'controller-exit.json')
            terminal = batch.read(directory / 'controller-terminal.json') if (directory / 'controller-terminal.json').exists() else {}
            outcome = 0 if (exit_record['children_reaped'] and not exit_record['hold'] and
                not exit_record['failed'] and (exit_record.get('operator_stopped') or
                    set(terminal) == {entry['run_id'] for entry in manifest['runs']})) else 2
            return outcome
        finally:
            write_json(directory / 'driver-exit.json', dict(time=datetime.now(timezone.utc).isoformat(),
                returncode=outcome, batch_id=manifest['batch_id'], auto_resume=True))


if __name__ == '__main__':
    raise SystemExit(main())
