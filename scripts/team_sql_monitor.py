"""Host SQL and frozen-source checks for an existing engineering team batch."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import round5
from saas_bench.run_state import tree_hash, verify_build, write_json


ROOT = Path(__file__).resolve().parents[1]


def poll(output, state):
    output = Path(output)
    offsets = state.setdefault('offsets', {})
    known = set(state.get('known_fault_paths', []))
    checks, failures, queries = [], [], 0
    quota = None
    try:
        verify_build(ROOT / 'public', root=ROOT)
        manifest = json.loads((output / 'engineering' / 'manifest.json').read_text())
        if tree_hash(ROOT / 'src') != manifest['source_sha256']:
            checks.append('source_changed')
    except (OSError, ValueError, KeyError) as exc:
        checks.append('build_or_source_check:' + type(exc).__name__)
    try:
        quota = round5.quota_health()
        if any(row.get('status') != 'ok' or row.get('percent', 100) >= 95 for row in quota.values()):
            checks.append('go_quota_near_limit')
    except Exception as exc:
        checks.append('quota_read_failed:' + type(exc).__name__)
    fields = ('day', 'snapshot_ref', 'failure_stage', 'error_type', 'sqlite_errorcode', 'sqlite_errorname')
    for path in sorted((output / 'host-logs').glob('*.log')):
        try:
            offset = offsets.get(str(path), 0)
            rows, offsets[str(path)] = round5.read_increment(path, offset, prefix=b'[public_sql] ')
        except OSError:
            continue
        queries += len(rows)
        failures.extend(dict(path=str(path), **{key: row[key] for key in fields if key in row})
                        for row in rows if row.get('error_kind') == 'service')
    faults = sorted(str(path) for path in (output / 'engineering' / 'runs').glob(
        '*/attempt-*/runtime/private/evidence.fault.json'))
    new_faults = [path for path in faults if path not in known]
    state['known_fault_paths'] = sorted(known | set(faults))
    alerts = [check for check in checks if not check.startswith('quota_read_failed:')]
    new_checks = [check for check in alerts if check not in state.get('active_check_alerts', [])]
    state['active_check_alerts'] = alerts
    at = datetime.now(timezone.utc).isoformat()
    hold = output / 'engineering' / 'HOLD'
    if failures or faults or alerts:
        hold.touch()
    if failures or new_faults or new_checks:
        event = dict(time=at, event='new_service_or_freeze_fault_hold', errors=failures,
                     fault_paths=new_faults, checks=new_checks)
        with (output / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')
    health = dict(time=at, quota=quota, freeze_checks=checks, sql_events_since_last_check=queries,
                  service_failures=failures, fault_paths=faults, new_fault_paths=new_faults,
                  offsets=offsets, known_fault_paths=state['known_fault_paths'],
                  active_check_alerts=alerts,
                  status='hold' if hold.exists() else 'no_new_service_faults',
                  coverage='public_sql worker stderr service errors and private evidence faults; '
                           'ordinary SQL syntax and permission errors excluded')
    write_json(output / 'sql-health.json', health)
    with (output / 'sql-health.jsonl').open('a') as stream:
        stream.write(json.dumps(health) + '\n')
    return health


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    duration = parser.add_mutually_exclusive_group(required=True)
    duration.add_argument('--once', action='store_true')
    duration.add_argument('--until', help='ISO 8601 deadline with a timezone')
    args = parser.parse_args(argv)
    end = None
    if args.until:
        try:
            until = datetime.fromisoformat(args.until)
            if until.tzinfo is None:
                raise ValueError('deadline needs a timezone')
            end = until.timestamp()
        except ValueError as exc:
            parser.error(str(exc))
    output = args.output_dir.resolve()
    path = output / 'sql-health.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    state = dict(offsets=previous.get('offsets', {}),
                 known_fault_paths=previous.get('known_fault_paths', []),
                 active_check_alerts=previous.get('active_check_alerts', []))
    while args.once or time.time() < end:
        health = poll(output, state)
        if args.once:
            print(json.dumps(health, indent=2))
            return
        time.sleep(max(0, min(300, end - time.time())))


if __name__ == '__main__':
    main()
