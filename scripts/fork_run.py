"""Fork a stopped common prefix into a Git or PF branch run directory.

Example (prefix stopped with --stop-after-day 28):
    python scripts/fork_run.py runs/run_abcd1234 runs/pf-1 --mode pf --branch pf-1
    python -m saas_bench.agents.bash_agent.run_test --continue-from runs/pf-1 --stop-after-day 70
"""
import argparse
import json
from pathlib import Path

from saas_bench.run_state import clone_sql_run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=Path, help='Prefix run directory; its latest checkpoint is the fork snapshot')
    parser.add_argument('destination', type=Path, help='New branch run directory (must not exist)')
    parser.add_argument('--mode', choices=['git', 'pf'], required=True)
    parser.add_argument('--branch', required=True, help='Evidence branch ID, e.g. pf-1')
    parser.add_argument('--stale-checks', action=argparse.BooleanOptionalAction, default=None,
                        help='PF only: automatic stale checks (default on; off is the offline ablation)')
    args = parser.parse_args()
    destination = clone_sql_run(args.source, args.destination, args.branch,
                                text_registration=args.mode, pf_stale_checks=args.stale_checks)
    checkpoint = json.loads((destination / 'checkpoint.json').read_text())
    print(json.dumps(dict(destination=str(destination), mode=args.mode, branch=args.branch,
                          fork_day=checkpoint['day'], snapshot_id=checkpoint['snapshot_id']), indent=2))


if __name__ == '__main__':
    main()
