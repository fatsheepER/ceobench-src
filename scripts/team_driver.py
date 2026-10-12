#!/usr/bin/env python3
"""Reusable core for the existing launch_continuation/live_watch/pause_at_week wrappers.

Freeze with team_batch.py freeze --scheduling group_independent. The driver policy
uses the existing identity anchors, with runs listing every frozen run in order.
Call this source script with --policy PATH --check, or --live [--resume]; launch
wrappers keep their credential/quota/window gate and invoke this core in place of
the pair loop. The observer and controller files live next to that policy. Watch
and week-pause wrappers discover the two current workers from controller-units
and worker_launched events; their run IDs may now belong to different seeds.
The existing retention-lock/ready-pid/single-SIGUSR1 mechanism is unchanged.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

from saas_bench import team_batch as batch
from saas_bench.run_state import file_hash, write_json
from saas_bench.team_recovery import execution_root
import team_controller as controller
from team_observer import identity


def verify_controls(policy):
    policy = Path(policy).resolve()
    config = batch.read(policy)
    expected = config['identity']
    manifest = identity(expected['manifest'], expected)
    if batch.scheduling_policy(manifest) != 'group_independent':
        raise ValueError('Group driver requires an explicitly frozen group_independent batch')
    if Path(__file__).resolve() != execution_root(expected['manifest'], manifest) / 'scripts/team_driver.py':
        raise ValueError('Driver is outside the frozen source root')
    if config.get('control_freeze'):
        freeze = Path(config['control_freeze'])
        if not freeze.is_absolute() or file_hash(freeze) != config['control_freeze_sha256']:
            raise ValueError('Driver control freeze changed')
        for name, checksum in batch.read(freeze)['sha256'].items():
            if file_hash(freeze.parent / name) != checksum:
                raise ValueError('Frozen control changed: ' + name)
    return config, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--live', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.resume and not args.live:
        parser.error('--resume requires --live')
    config, manifest = verify_controls(args.policy)
    if args.check:
        print(json.dumps(dict(status='validated', batch_id=manifest['batch_id'],
            scheduling=manifest['scheduling'], model_requests=0, credential_read=False, new_attempts=0)))
        return 0
    directory = args.policy.resolve().parent
    outcome = 2
    with (directory / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            controller.supervise(args.policy, resume=args.resume)
            exit_record = batch.read(directory / 'controller-exit.json')
            terminal = batch.read(directory / 'controller-terminal.json') if (directory / 'controller-terminal.json').exists() else {}
            outcome = 0 if (exit_record['children_reaped'] and not exit_record['hold'] and
                not exit_record['failed'] and set(terminal) == {entry['run_id'] for entry in manifest['runs']}) else 2
            return outcome
        finally:
            write_json(directory / 'driver-exit.json', dict(time=datetime.now(timezone.utc).isoformat(),
                returncode=outcome, batch_id=manifest['batch_id'], auto_resume=False))


if __name__ == '__main__':
    raise SystemExit(main())
