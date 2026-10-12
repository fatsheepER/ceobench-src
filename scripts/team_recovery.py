#!/usr/bin/env python3
"""Create a fingerprint-bound continuation approval from an explicit declaration."""

import argparse
import json
from pathlib import Path

from saas_bench.team_recovery import create_approval


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, help='Committed execution source; defaults to this checkout.')
    parser.add_argument('--declaration', type=Path, required=True,
        help='JSON with fix_classification, fix_commit, affected_files, reason, evidence and optional attempts.')
    parser.add_argument('--approve', action='store_true', help='Explicitly approve this declaration; default is pending.')
    args = parser.parse_args(argv)
    if args.declaration.name == '.env' or args.declaration.name.startswith('.env.'):
        parser.error('Credentials cannot be a repair declaration')
    declaration = json.loads(args.declaration.read_text())
    print(json.dumps(create_approval(args.manifest, args.output, declaration,
        source_root=args.source_root, approved=args.approve)))


if __name__ == '__main__':
    main()
