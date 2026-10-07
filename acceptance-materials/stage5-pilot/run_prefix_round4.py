"""Run only the four-week prefix; inspect metadata every five minutes."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from saas_bench.run_state import write_json

BASE = Path(__file__).resolve().parent
OUT = BASE / 'round4'
POINTER = OUT / 'long-prefix-42.json'


def outcome(record, returncode):
    if returncode is None:
        return 'running'
    result = record.get('result', {})
    if returncode == 0 and record.get('reached_stop') and result.get('days_run') == 28:
        return 'scheduled_pause'
    if returncode == 0 and result.get('outcome') in ('bankrupt', 'completed'):
        return 'natural_end'
    return 'unexpected_exit'


def alerts(status, idle, free, total, faults, errors):
    result = list(faults)
    if status == 'unexpected_exit':
        result.append('unexpected_exit')
    if status == 'running' and idle >= 1800:
        result.append('no_activity_30_minutes: diagnose before any restart')
    if free < max(2 * 1024**3, total * .01):
        result.append('low_disk_space')
    if errors >= 3:
        result.append('repeated_service_errors')
    return result


def main():
    if '--self-check' in sys.argv:
        assert outcome({}, None) == 'running'
        assert outcome({'reached_stop': True, 'result': {'days_run': 28}}, 0) == 'scheduled_pause'
        assert outcome({'result': {'outcome': 'bankrupt'}}, 0) == 'natural_end'
        assert outcome({}, 1) == 'unexpected_exit'
        assert not alerts('running', 300, 10 * 1024**3, 100 * 1024**3, [], 0)
        assert 'no_activity_30_minutes: diagnose before any restart' in alerts('running', 1800, 10 * 1024**3, 100 * 1024**3, [], 0)
        assert 'repeated_service_errors' in alerts('running', 0, 10 * 1024**3, 100 * 1024**3, [], 3)
        assert alerts('running', 0, 0, 100 * 1024**3, ['capture_fault'], 0) == ['capture_fault', 'low_disk_space']
        print('health self-check passed')
        return
    assert os.environ.get('PYTHONHASHSEED') == '0'
    if POINTER.exists():
        raise RuntimeError('Prefix already registered; do not start another automatically')
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    logpath = OUT / 'prefix-42.log'
    with logpath.open('xb') as log:
        process = subprocess.Popen([sys.executable, '-u', str(BASE / 'run_round4.py'), 'prefix'],
                                   cwd=BASE.parents[1], stdout=log, stderr=subprocess.STDOUT)
        write_json(OUT / 'prefix-process.json', {'pid': process.pid, 'supervisor_pid': os.getpid(),
                   'started_at': datetime.now(timezone.utc).isoformat(), 'pointer': str(POINTER)})
        previous_alerts = []
        offset = 0
        while True:
            returncode = process.poll()
            record = json.loads(POINTER.read_text()) if POINTER.exists() else {}
            run = Path(record['path']) if record.get('path') else None
            checkpoint = json.loads((run / 'checkpoint.json').read_text()) if run and (run / 'checkpoint.json').exists() else {}
            paths = [logpath] + (list((run / 'logs').glob('*')) if run else [])
            idle = max(0, time.time() - max([started] + [p.stat().st_mtime for p in paths if p.is_file()]))
            faults = [p.name for p in (run / 'branch_stop.json', run / 'sql-evidence.fault.json') if p.exists()] if run else []
            free = shutil.disk_usage(OUT)
            with logpath.open('rb') as stream:
                stream.seek(offset)
                # Only scan new bytes locally; model output is never printed or analyzed.
                errors = sum(b'Server error (' in line or b'LLM call error' in line for line in stream)
                offset = stream.tell()
            status = outcome(record, returncode)
            warnings = alerts(status, idle, free.free, free.total, faults, errors)
            summary = {'at': datetime.now(timezone.utc).isoformat(), 'status': status,
                       'day': checkpoint.get('day', 0), 'context_boundary': checkpoint.get('context_boundary'),
                       'pid': process.pid, 'returncode': returncode, 'idle_seconds': round(idle),
                       'disk_free_bytes': free.free, 'service_error_lines': errors,
                       'snapshot_id': checkpoint.get('snapshot_id'), 'alerts': warnings}
            write_json(OUT / 'prefix-health.json', summary)
            with (OUT / 'prefix-health.jsonl').open('a') as health:
                health.write(json.dumps(summary) + '\n')
            if warnings and warnings != previous_alerts:
                print(json.dumps(summary), flush=True)
            previous_alerts = warnings
            if returncode is not None:
                print(json.dumps(summary), flush=True)
                break
            try:
                process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                pass
    if status == 'scheduled_pause':
        assert checkpoint['day'] == 28 and checkpoint['context_boundary'] == 'new_week'
        with (OUT / 'prefix-fork-acceptance.log').open('xb') as log:
            checked = subprocess.run([sys.executable, str(BASE / 'check_forks_round4.py'), str(POINTER),
                                     '--output', str(OUT / 'engineering' / 'forks-28')],
                                    cwd=BASE.parents[1], stdout=log, stderr=subprocess.STDOUT)
        write_json(OUT / 'prefix-acceptance-status.json', {'status': 'passed' if checked.returncode == 0 else 'failed',
                   'returncode': checked.returncode, 'path': str(OUT / 'engineering' / 'forks-28' / 'verification.json')})
        print('D28 offline restore/isolation acceptance exit code:', checked.returncode, flush=True)
        if checked.returncode:
            raise SystemExit(checked.returncode)
    raise SystemExit(returncode)


if __name__ == '__main__':
    main()
