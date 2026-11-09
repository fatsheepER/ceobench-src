#!/usr/bin/env python3
"""Prepare and check frozen shared-workspace D56 runs without starting workers."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from saas_bench import team_batch as batch
from saas_bench.run_state import file_hash, write_json
from saas_bench.team_round5 import LAUNCHER, PROFILE, PROFILE_SEEDS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--check', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--d0-source-manifest', type=Path)
    parser.add_argument('--pricing', type=Path)
    parser.add_argument('--profile', choices=tuple(PROFILE_SEEDS), default=PROFILE)
    args = parser.parse_args(argv)
    root = args.output.resolve()
    if args.prepare:
        if not args.d0_source_manifest or not args.pricing:
            parser.error('--prepare requires --d0-source-manifest and --pricing')
        source = batch.read(args.d0_source_manifest)
        prices = batch.load_pricing(args.pricing)
        sources = {seed: Path(source['d0'][str(seed)]['path']) for seed in PROFILE_SEEDS[args.profile]}
        if root.exists():
            parser.error('Preparation output must be a new directory')
        batch.freeze_batch(root / 'batch', sources, prices, price_source=prices['source'], profile=args.profile)
        policy = batch.create_policy(root / 'batch/manifest.json', root / 'control')['policy']
        command = [sys.executable, str(batch.ROOT / 'scripts/team_driver.py'), '--policy', policy]
        write_json(root / 'launch.json', dict(launcher=LAUNCHER, profile=args.profile,
            source_manifest=str(args.d0_source_manifest.resolve()),
            source_manifest_sha256=file_hash(args.d0_source_manifest),
            policy=policy, check=shlex.join(command + ['--check']),
            live=shlex.join(command + ['--live']),
            resume=shlex.join(command + ['--live', '--resume', '--run-id', '<run-id>', '--week', '<week>']),
            model_requests=0, business_operations=0, live_started=False))
    launch = batch.read(root / 'launch.json')
    if launch['launcher'] != LAUNCHER or launch['profile'] not in PROFILE_SEEDS:
        raise ValueError('Unexpected fifth-round launcher identity')
    result = subprocess.run([sys.executable, str(batch.ROOT / 'scripts/team_driver.py'),
        '--policy', launch['policy'], '--check'], check=True, capture_output=True, text=True)
    check = json.loads(result.stdout)
    print(json.dumps(dict(status='offline_validated', batches=[check], model_requests=0,
        business_operations=0, live_started=False, pending=check['contract_check']['pending']),
        ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
