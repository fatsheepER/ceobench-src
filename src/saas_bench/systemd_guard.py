"""Memory-isolated worker scopes and invocation-bound unit observations."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

# Stable systemd OOM event IDs from sd-messages.h.
OOM_MESSAGE_IDS = {'d989611b15e44c9dbf31e3c81256e4ed', 'fe6faa94e7774663a0da52717891d8ef'}
WORKER_SLICE = 'ceobench.slice'


def memory_bytes(value):
    if value.lower() in ('max', 'infinity'):
        return 'max'
    match = re.fullmatch(r'(\d+)([KMGT]?)', value, re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError('Use bytes, an integer with K, M, G or T, or max/infinity')
    return int(match[1]) * 1024**('KMGT'.index(match[2].upper()) + 1 if match[2] else 0)


def scope_command(command, unit, *, memory_high=16 * 1024**3, memory_max=20 * 1024**3, memory_swap_max=0):
    if not re.fullmatch(r'[A-Za-z0-9_.@:-]+\.scope', unit):
        raise ValueError('Invalid worker scope name')
    if (any(value != 'max' and (type(value) is not int or value < 0)
            for value in (memory_high, memory_max, memory_swap_max)) or memory_high == 0 or memory_max == 0 or
            type(memory_high) is int and type(memory_max) is int and memory_high > memory_max):
        raise ValueError('Require positive MemoryHigh/MemoryMax or max, MemoryHigh <= MemoryMax, and nonnegative MemorySwapMax')
    return ['systemd-run', '--user', '--scope', '--quiet', '--unit=' + unit,
        '--slice=' + WORKER_SLICE, '--property=ManagedOOMPreference=omit',
        *['--property=' + name + '=' + ('infinity' if value == 'max' else str(value))
          for name, value in (('MemoryHigh', memory_high), ('MemoryMax', memory_max), ('MemorySwapMax', memory_swap_max))],
        '--property=OOMPolicy=kill', *command]


def check_worker_cgroup(proof):
    group = Path('/sys/fs/cgroup') / proof['control_group'].lstrip('/')
    own = Path('/proc/self/cgroup').read_text().strip().split('0::')[-1]
    changes = []
    if own != proof['control_group']:
        changes.append('Worker moved outside its original cgroup')
    for name, expected in (('memory.high', proof['memory_high']), ('memory.max', proof['memory_max']),
                           ('memory.swap.max', proof['memory_swap_max']), ('memory.oom.group', 1)):
        if (group / name).read_text().strip() != str(expected):
            changes.append('Worker cgroup limit changed: ' + name)
    if os.getxattr(group, 'user.oomd_omit') != b'1':
        changes.append('Worker cgroup lacks effective oomd omit')
    return changes


def verify_worker_scope(policy):
    if (not isinstance(policy, dict) or set(policy) != {'unit', 'memory_high', 'memory_max', 'memory_swap_max'} or
            not isinstance(policy.get('unit'), str)):
        raise ValueError('Missing worker scope policy')
    scope_command([], policy['unit'], **{key: policy[key] for key in ('memory_high', 'memory_max', 'memory_swap_max')})
    deadline = time.monotonic() + 5
    def show(units, *, user):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError('Worker scope verification deadline expired')
        command = ['systemctl', *(['--user'] if user else []), 'show',
            '--property=Id,InvocationID,Slice,ControlGroup,MemoryHigh,MemoryMax,MemorySwapMax,OOMPolicy,ManagedOOMPreference,ManagedOOMMemoryPressure,ManagedOOMSwap',
            '--', *units]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=remaining)
        values = [dict(line.split('=', 1) for line in block.splitlines() if '=' in line)
                  for block in result.stdout.strip().split('\n\n')]
        records = {record.get('Id'): record for record in values}
        if set(records) != set(units):
            raise RuntimeError('Worker scope properties are incomplete')
        return records
    observed = show([policy['unit']], user=True)[policy['unit']]
    group = f'/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/{WORKER_SLICE}/{policy["unit"]}'
    expected = dict(Slice=WORKER_SLICE, ControlGroup=group, OOMPolicy='kill', ManagedOOMPreference='omit',
        ManagedOOMMemoryPressure='auto', ManagedOOMSwap='auto',
        **{name: 'infinity' if policy[key] == 'max' else str(policy[key]) for name, key in (
            ('MemoryHigh', 'memory_high'), ('MemoryMax', 'memory_max'), ('MemorySwapMax', 'memory_swap_max'))})
    if not observed.get('InvocationID') or not observed.get('ControlGroup'):
        raise RuntimeError('Worker scope has no invocation/cgroup binding')
    changes = [key + ' differs from requested policy' for key, value in expected.items()
               if observed.get(key) != value]
    ancestors = {'user:' + name: record for name, record in show([WORKER_SLICE, '-.slice'], user=True).items()}
    ancestors.update({'system:' + name: record for name, record in show(
        ['-.slice', 'user.slice', f'user-{os.getuid()}.slice', f'user@{os.getuid()}.service'], user=False).items()})
    if any(record.get(key) != 'auto' for record in ancestors.values()
           for key in ('ManagedOOMMemoryPressure', 'ManagedOOMSwap')):
        changes.append('Worker has an oomd-monitored ancestor')
    proof = dict(policy, pid=os.getpid(), invocation_id=observed['InvocationID'],
        control_group=observed['ControlGroup'], slice=observed.get('Slice'),
        ancestors=ancestors, checked_at=time.time(), observations=changes)
    try:
        proof['observations'].extend(check_worker_cgroup(proof))
    except (OSError, ValueError, RuntimeError) as exc:
        proof['observations'].append(type(exc).__name__ + ': ' + str(exc))
    return proof


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
