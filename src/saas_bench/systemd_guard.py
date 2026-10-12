"""Memory-isolated worker scopes and invocation-bound unit observations."""
import argparse
import json
import re
import subprocess
import time

# Stable systemd OOM event IDs from sd-messages.h.
OOM_MESSAGE_IDS = {'d989611b15e44c9dbf31e3c81256e4ed', 'fe6faa94e7774663a0da52717891d8ef'}


def memory_bytes(value):
    match = re.fullmatch(r'(\d+)([KMGT]?)', value, re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError('Use bytes or an integer with K, M, G or T; unlimited memory is not allowed')
    return int(match[1]) * 1024**('KMGT'.index(match[2].upper()) + 1 if match[2] else 0)


def scope_command(command, unit, *, memory_high=2 * 1024**3, memory_max=3 * 1024**3, memory_swap_max=0):
    if not re.fullmatch(r'[A-Za-z0-9_.@:-]+\.scope', unit):
        raise ValueError('Invalid worker scope name')
    if any(type(value) is not int for value in (memory_high, memory_max, memory_swap_max)) or not (
            0 < memory_high <= memory_max and memory_swap_max >= 0):
        raise ValueError('Require 0 < MemoryHigh <= MemoryMax and MemorySwapMax >= 0')
    return ['systemd-run', '--user', '--scope', '--quiet', '--unit=' + unit,
        '--property=MemoryHigh=' + str(memory_high), '--property=MemoryMax=' + str(memory_max),
        '--property=MemorySwapMax=' + str(memory_swap_max), '--property=OOMPolicy=kill', *command]


def unit_status(unit, *, invocation_id=None, exit_code=None):
    if not re.fullmatch(r'[A-Za-z0-9_.@:-]+\.(?:scope|service)', unit):
        raise ValueError('Invalid unit name')
    properties, rows, errors = {}, [], []
    commands = [
        ['systemctl', '--user', 'show', unit, '--property=LoadState,ActiveState,Result,InvocationID'],
        ['journalctl', '--user', '--no-pager', '--output=json', '-n', '200',
         '--output-fields=USER_UNIT,UNIT,USER_INVOCATION_ID,INVOCATION_ID,_SYSTEMD_INVOCATION_ID,UNIT_RESULT,RESULT,MESSAGE_ID,MESSAGE',
         'USER_UNIT=' + unit, '+', 'UNIT=' + unit],
    ]
    for index, command in enumerate(commands):
        try:
            response = subprocess.run(command, capture_output=True, text=True, timeout=10)
            if response.returncode:
                errors.append(command[0] + ':' + response.stderr.strip())
                continue
            if index == 0:
                properties = dict(line.split('=', 1) for line in response.stdout.splitlines() if '=' in line)
            else:
                rows = [json.loads(line) for line in response.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            errors.append(command[0] + ':' + str(exc))
    def row_invocation(row):
        return row.get('USER_INVOCATION_ID') or row.get('INVOCATION_ID') or row.get('_SYSTEMD_INVOCATION_ID')
    invocation_id = invocation_id or properties.get('InvocationID') or next(
        (row_invocation(row) for row in reversed(rows) if row_invocation(row)), None)
    rows = [row for row in rows if invocation_id and row_invocation(row) == invocation_id]
    loaded = bool(invocation_id) and properties.get('LoadState') == 'loaded' and properties.get('InvocationID') == invocation_id
    result, source = 'unknown', 'missing_evidence'
    if loaded:
        result = 'running' if properties.get('ActiveState') in ('active', 'activating', 'deactivating') else properties.get('Result', 'unknown')
        source = 'systemctl'
    for row in rows:
        if (row.get('USER_UNIT') or row.get('UNIT')) != unit:
            continue
        terminal = row.get('UNIT_RESULT') or row.get('RESULT')
        message = row.get('MESSAGE', '')
        failed = re.search(r"Failed with result '([^']+)'", message)
        terminal = terminal or (failed[1] if failed else None)
        if terminal == 'oom-kill' or row.get('MESSAGE_ID') in OOM_MESSAGE_IDS:
            result, source = 'oom-kill', 'journal'
            break
        if terminal:
            result, source = terminal, 'journal'
    # An unloaded unit's synthetic Result=success has no invocation evidence.
    if result == 'unknown' and exit_code is not None and rows and not errors:
        result = 'success' if exit_code == 0 else 'exit-code'
        source = 'command_exit_and_invocation_journal'
    if exit_code not in (None, 0) and result == 'success':
        result, source = 'exit-code', 'command_exit'
    return dict(unit=unit, invocation_id=invocation_id, result=result, source=source,
        load_state=properties.get('LoadState'), active_state=properties.get('ActiveState'),
        exit_code=exit_code, errors=errors)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--invocation-id')
    parser.add_argument('--exit-code', type=int)
    parser.add_argument('--watch', action='store_true', help='Print an observation every five seconds until the unit stops.')
    args = parser.parse_args(argv)
    while True:
        record = unit_status(args.unit, invocation_id=args.invocation_id, exit_code=args.exit_code)
        print(json.dumps(record), flush=True)
        if not args.watch or record['result'] != 'running':
            return 0 if record['result'] in ('success', 'running') else 2
        args.invocation_id = record['invocation_id']
        time.sleep(5)


if __name__ == '__main__':
    raise SystemExit(main())
