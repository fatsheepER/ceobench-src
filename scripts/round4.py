#!/usr/bin/env python3
"""Prepare/check three independent D56 batches; never run a model or worker."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from saas_bench import team_batch as batch
from saas_bench.run_state import file_hash, write_json
from saas_bench.team_models import ROUND_FOUR_PROFILES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--check', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--d0-source-manifest', type=Path)
    parser.add_argument('--pricing', type=Path)
    args = parser.parse_args(argv)
    root = args.output.resolve()
    if args.prepare:
        if not args.d0_source_manifest or not args.pricing:
            parser.error('--prepare requires --d0-source-manifest and --pricing')
        source = batch.read(args.d0_source_manifest)
        prices = batch.load_pricing(args.pricing)
        # Resolve and verify both before creating any batch. Never regenerate D0.
        sources = {seed: Path(source['d0'][str(seed)]['path']) for seed in (1048, 1049)}
        for seed, path in sources.items():
            batch.verify_d0(path, seed=seed)
        if root.exists():
            parser.error('Preparation output must be a new directory')
        root.mkdir(parents=True)
        commands = []
        for profile in ROUND_FOUR_PROFILES:
            selected = sources if profile == 'r4-d56' else {1048: sources[1048]}
            directory = root / profile
            batch.freeze_batch(directory / 'batch', selected, prices, price_source=prices['source'], profile=profile)
            control = batch.create_policy(directory / 'batch/manifest.json', directory / 'control')
            policy = control['policy']
            command = [sys.executable, str(batch.ROOT / 'scripts/team_driver.py'), '--policy', policy]
            commands.append(dict(profile=profile, policy=policy, check=shlex.join(command + ['--check']),
                live=shlex.join(command + ['--live']), resume=shlex.join(command + ['--live', '--resume', '--run-id', '<run-id>', '--week', '<week>'])))
        write_json(root / 'launch.json', dict(source_manifest=str(args.d0_source_manifest.resolve()),
            source_manifest_sha256=file_hash(args.d0_source_manifest), order=commands,
            paid_calls=0, live_started=False,
            pending=['Final unified release gate and post-gate --check',
                     'OpenCode Go max forwarding/availability; no fallback to high',
                     'OpenAI credential/access check by operator; no credit or cost calculation for Sol']))
    launch = batch.read(root / 'launch.json')
    checks = []
    for command in launch['order']:
        result = subprocess.run([sys.executable, str(batch.ROOT / 'scripts/team_driver.py'),
            '--policy', command['policy'], '--check'], check=True, capture_output=True, text=True)
        checks.append(json.loads(result.stdout))
    print(json.dumps(dict(status='offline_validated', batches=checks, model_requests=0, live_started=False,
                         pending=launch['pending']), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
