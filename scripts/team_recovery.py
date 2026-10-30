#!/usr/bin/env python3
"""Show the latest complete recovery checkpoint for each selected run."""
import argparse
import json
from pathlib import Path

from saas_bench.team_recovery import latest_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--week', type=int, help='Restore point for this week; week 1 starts at day 0.')
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    entries = [entry for entry in manifest['runs'] if args.run_id in (None, entry['run_id'])]
    if not entries:
        parser.error('Run is absent from the manifest')
    result = []
    for entry in entries:
        snapshot = latest_checkpoint(entry, week=args.week)
        state = json.loads((snapshot / 'team.json').read_text()) if snapshot else {}
        result.append(dict(run_id=entry['run_id'], checkpoint=str(snapshot) if snapshot else None,
            day=state.get('day'), stage=state.get('stage'), approval_required=False))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
