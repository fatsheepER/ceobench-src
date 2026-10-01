"""Preserve an interrupted run and recover its last complete checkpoint in a new directory."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from saas_bench.run_state import checkpoint_directory, recover_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path, nargs='?')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    checkpoint = json.loads((args.source / 'checkpoint.json').read_text())
    checkpoint_directory(args.source, checkpoint)
    destination = args.destination
    if destination is None:
        import uuid
        destination = args.source.with_name(args.source.name + '-recovery-' + uuid.uuid4().hex[:7])
    print(json.dumps(dict(source=str(args.source), destination=str(destination), day=checkpoint['day'],
                          dry_run=args.dry_run), indent=2), flush=True)
    if args.dry_run:
        return
    recover_run(args.source, destination)
    if args.resume:
        subprocess.run([sys.executable, '-m', 'saas_bench.agents.bash_agent.run_test',
                        '--continue-from', str(destination)], check=True)


if __name__ == '__main__':
    main()
