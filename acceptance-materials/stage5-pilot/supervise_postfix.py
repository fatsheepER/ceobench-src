"""Frozen seed-42 pilot. Local five-minute checks; only gates/errors reach stdout."""
from collections import Counter
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from tempfile import NamedTemporaryFile
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_prefix_round4 import alerts
from saas_bench.run_state import checkpoint_directory, clone_sql_run, file_hash, verify_build, write_json
from scripts.analyze_pf_run import analyze

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OUT = BASE / 'round4-postfix'
HOLD = OUT / 'hold.json'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def lines(path):
    if path.exists():
        with path.open() as stream:
            return [json.loads(line) for line in stream if line.strip()]
    return []


def last_week(run):
    path = run / 'weekly.jsonl'
    if not path.exists():
        return {}
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 32768))
        tail = stream.read().splitlines()
    for line in reversed(tail):
        try:
            return json.loads(line)
        except ValueError:
            pass
    return {}


def notify(event, **values):
    record = dict(at=datetime.now(timezone.utc).isoformat(), event=event, **values)
    with (OUT / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


def service_error(line, *, server=False):
    return line.startswith((b'OpenAI LLM call error', b'Anthropic LLM call error')) or \
        server and line.startswith(b'Traceback')


def launch(mode, stop, source=None):
    verify_build(ROOT / 'public', root=ROOT)
    frozen = read(OUT / 'frozen.json')
    for name, checksum in frozen['host_scripts'].items():
        assert file_hash(Path(name)) == checksum, 'Frozen host script changed: ' + name
    pointer = OUT / (f'long-prefix-42.json' if mode == 'prefix' else f'long-{mode}-42-to{stop}.json')
    logpath = OUT / f'{mode}-to{stop}.log'
    argv = [sys.executable, '-u', str(BASE / 'run_round4.py'), mode,
            '--output-dir', str(OUT), '--pricing-file', str(OUT / 'pricing.json'), '--pause-file', str(HOLD)]
    if source:
        argv += ['--continue-from', str(source), '--stop-after-day', str(stop)]
    with logpath.open('xb') as log:
        process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    return dict(mode=mode, process=process, pid=process.pid, pointer=pointer, log=logpath,
                started=time.time(), offsets={}, previous_alerts=[])


def integrity(run, mode, milestone=None):
    cp = read(run / 'checkpoint.json')
    snapshot = checkpoint_directory(run, cp)
    manifest = read(run / 'manifest.json')
    assert manifest['text_registration'] == mode
    assert not any((run / n).exists() for n in ('branch_stop.json', 'sql-evidence.fault.json'))
    usage = dict(agent=cp['usage'], simulator=read(snapshot / 'server_state.json')['usage'])
    for role in usage:
        assert usage[role]['missing_cost'] == 0 and usage[role]['known_cost_usd'] is not None
    if mode == 'pf':
        with sqlite3.connect((snapshot / 'sql-evidence.sqlite').as_uri() + '?mode=ro&immutable=1', uri=True) as conn:
            assert conn.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
            assert conn.execute('SELECT count(*) FROM requests WHERE event_id NOT IN (SELECT event_id FROM results)').fetchone()[0] == 0
            assert conn.execute('SELECT count(*) FROM client_calls WHERE received IS NULL').fetchone()[0] == 0
    elif mode == 'git':
        assert not list(run.rglob('sql-evidence*'))
    result = dict(status='passed', mode=mode, day=cp['day'], milestone=milestone,
                  snapshot_id=cp['snapshot_id'], usage=usage)
    write_json(OUT / f'integrity-{mode}-D{milestone or cp["day"]}.json', result)
    return result


def supervise(jobs, stop):
    write_json(OUT / f'processes-to{stop}.json', dict(supervisor_pid=os.getpid(), jobs=[
        {k: str(v) if isinstance(v, Path) else v for k, v in job.items() if k in ('mode','pid','pointer','log')}
        for job in jobs]))
    milestones = {job['mode']: [210, 280, 350, 420] for job in jobs}
    previous = None
    while True:
        summaries = []
        for job in jobs:
            rc = job['process'].poll()
            record = read(job['pointer'], {})
            run = Path(record['path']) if record.get('path') else None
            cp = read(run / 'checkpoint.json', {}) if run else {}
            weekly = last_week(run) if run else {}
            state = 'running' if rc is None else {'stopped':'scheduled_pause','completed':'natural_end','bankrupt':'natural_end'}.get(record.get('status'),'unexpected_exit')
            paths = [job['log']] + (list((run / 'logs').glob('*')) if run else [])
            if run: paths += [run / 'weekly.jsonl', run / 'operation.json', run / 'checkpoint.json']
            idle = time.time() - max([job['started']] + [p.stat().st_mtime for p in paths if p.is_file()])
            faults = [n for n in ('branch_stop.json','sql-evidence.fault.json') if (run / n).exists()] if run else []
            errors = 0
            for path in [job['log']] + ([run / 'logs/api_server_stderr.log'] if run else []):
                if path.exists():
                    with path.open('rb') as stream:
                        stream.seek(job['offsets'].get(str(path), 0))
                        errors += sum(service_error(line, server=path.name == 'api_server_stderr.log') for line in stream)
                        job['offsets'][str(path)] = stream.tell()
            disk = shutil.disk_usage(OUT)
            warnings = alerts(state, idle, disk.free, disk.total, faults, errors)
            summary = dict(mode=job['mode'], status=state, day=weekly.get('day',cp.get('day',0)),
                           checkpoint_day=cp.get('day'), idle_seconds=round(idle), pid=job['pid'],
                           disk_free_bytes=disk.free, service_error_lines=errors, alerts=warnings)
            if warnings and warnings != job['previous_alerts']:
                notify('attention', **summary)
                # A timeout only requests diagnosis. Never kill or replay a business write.
                if faults or state == 'unexpected_exit' or 'low_disk_space' in warnings:
                    write_json(HOLD, summary)
            job['previous_alerts'] = warnings
            while stop == 497 and milestones[job['mode']] and cp.get('day',0) >= milestones[job['mode']][0]:
                milestone = milestones[job['mode']].pop(0)
                try:
                    integrity(run, job['mode'], milestone)
                    notify('integrity_passed', mode=job['mode'], milestone=milestone, day=cp['day'])
                except Exception as exc:
                    write_json(HOLD, dict(error=str(exc), milestone=milestone))
                    notify('integrity_failed', mode=job['mode'], milestone=milestone, error=str(exc))
            summaries.append(summary)
        health = dict(at=datetime.now(timezone.utc).isoformat(), stage_end=stop, branches=summaries)
        write_json(OUT / 'health.json', health)
        with (OUT / 'health.jsonl').open('a') as stream: stream.write(json.dumps(health) + '\n')
        if all(job['process'].poll() is not None for job in jobs):
            notify('stage_finished', stage_end=stop, branches=summaries)
            return [read(job['pointer'], {}) for job in jobs]
        # Ordinary Python polling, no model calls and no reasoning/MEMORY reads.
        time.sleep(300)


def command(argv, name):
    with (OUT / f'{name}.log').open('xb') as log:
        completed = subprocess.run([sys.executable, *argv], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'{name} failed; inspect {name}.log')


def adoption(weeks):
    calls = [(w['day'], c) for w in weeks for c in w['pf_calls']
             if c['command'] not in ('help','usage')]
    successful = [(d,c) for d,c in calls if c['outcome']=='succeeded']
    errors = sum(c['outcome']!='succeeded' for w in weeks for c in w['pf_calls'])
    return dict(successes=len(successful), successful_weeks=len({d for d,c in successful}), errors=errors)


def branch_audit(pointer):
    record = read(pointer)
    start = record.get('start_day')
    record['start_day'] = 28  # Report the whole branch, including earlier launch stages.
    with NamedTemporaryFile(mode='w', suffix='.json') as temp:
        json.dump(record, temp); temp.flush()
        audit = analyze(Path(temp.name))
    audit['launch_stage_start_day'] = start
    return audit


def audit112(records):
    errors = []
    audits = []
    for record in records:
        run = Path(record['path']); mode = record['group']
        integrity(run, mode)
        audit = branch_audit(next(p for p in OUT.glob(f'long-{mode}-42-to112.json')))
        audits.append(audit)
        tools = [t for p in (run/'logs').glob('tool_results_*.jsonl') for t in lines(p)
                 if t['day']>=28 and not t['tool'].startswith('_')]
        batches = audit['tool_batch_accounting']
        errors.extend(mode + ': ' + error for error in batches['errors'])
        if any(c.get('capture_match') is False for w in audit['weeks'] for c in w['pf_calls']):
            errors.append(mode + ': PF capture mismatch')
        if any(w['clock_in_results'] for w in audit['weeks']):
            errors.append(mode + ': host clock reached model')
        # Record ordinary corrected tool errors; they do not establish a broken harness.
        audit['tool_error_counts'] = dict(Counter(t['tool'] for t in tools if (t.get('result') or '').startswith('Error:')))
        audit['tool_batch_ids_match'] = batches['status'] == 'passed'
        write_json(OUT/f'audit-{mode}-D112.json', audit)
    pf = next((a for a in audits if a['group']=='pf'), None)
    current = adoption([w for w in pf['weeks'] if 28<=w['day']<112]) if pf else dict(successes=0,successful_weeks=0,errors=0)
    baseline = adoption(read(OUT/'baseline-pf-first12.json')['weeks'])
    gate = dict(day=112, decision='awaiting_agent_review', current=current, baseline=baseline,
                engineering_errors=errors, rule=read(OUT/'frozen.json')['D112_rule'])
    write_json(OUT/'gate-D112.json', gate)
    notify('D112_ready_for_agent_review', **gate)
    return gate


def main():
    global OUT, HOLD
    if '--self-check' in sys.argv:
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from run_round4 import install_week_pause
        assert not service_error('      → Traceback (most recent call last):'.encode())
        assert not service_error(b'Traceback (most recent call last):')
        assert service_error(b'Traceback (most recent call last):', server=True)
        assert service_error(b'OpenAI LLM call error (retryable=True, status=503): unavailable')
        with TemporaryDirectory() as temp:
            runner=SimpleNamespace(total_days=497,stop_after_day=112,workspace_dir=Path(temp),_weekly_record=lambda s:None)
            hold=Path(temp)/'hold.json';install_week_pause(runner,hold)
            runner._weekly_record({'day':42});assert runner.stop_after_day==112
            hold.touch();runner._weekly_record({'day':49});assert runner.stop_after_day==49
            runner.stop_after_day=None;runner._weekly_record({'day':497});assert runner.stop_after_day is None
        print('supervisor self-check passed');return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prefix','fork','to112','to497'),required=True)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--acceptance-name',default='D28-acceptance')
    parser.add_argument('--acceptance-dir',default='engineering/forks-28')
    args=parser.parse_args()
    OUT = args.output_dir.resolve()
    HOLD = OUT / 'hold.json'
    assert os.environ.get('PYTHONHASHSEED')=='0'
    assert not HOLD.exists()
    if args.stage in ('prefix','fork'):
        if args.stage=='prefix':
            prefix=supervise([launch('prefix',28)],28)[0]
        else:
            prefix=read(OUT/'long-prefix-42.json')
        if prefix.get('status')=='bankrupt':
            notify('prefix_bankrupt',result=prefix['result']);return
        assert prefix.get('reached_stop') and prefix['result']['days_run']==28
        command([str(BASE/'check_forks_round4.py'),str(OUT/'long-prefix-42.json'),'--output',str(OUT/args.acceptance_dir)],args.acceptance_name)
        dirs={mode:clone_sql_run(prefix['path'],OUT/'branches'/mode,'postfix-'+mode+'-42',text_registration=mode) for mode in ('git','pf')}
        records=supervise([launch(mode,35,dirs[mode]) for mode in dirs],35)
        if HOLD.exists():return
        if all(r.get('reached_stop') for r in records):
            command([str(BASE/'check_gate_round4.py'),'--output-dir',str(OUT)],'D35-acceptance')
        else:
            for r in records:
                assert r.get('status')=='bankrupt' or r.get('reached_stop')
                integrity(Path(r['path']),r['group'])
        notify('D35_ready_for_agent_review')
        return
    day=35 if args.stage=='to112' else 112
    review=read(OUT/f'review-D{day}.json',{})
    assert review.get('decision')=='continue','Execution agent review required before continuation'
    records=[read(OUT/f'long-{mode}-42-to{day}.json',{}) for mode in ('git','pf')]
    active={r['group']:Path(r['path']) for r in records if r.get('reached_stop')}
    if not active:return
    end=112 if args.stage=='to112' else 497
    records=supervise([launch(mode,end,path) for mode,path in active.items()],end)
    if HOLD.exists():return
    if end==112:
        assert all(r.get('status')=='bankrupt' or r.get('reached_stop') for r in records)
        audit112(records)
        return
    for r in records:
        assert r.get('status') in ('completed','bankrupt')
        integrity(Path(r['path']),r['group'])
        command([str(ROOT/'scripts/score_predictions.py'),r['path'],'--submit-start','28','--outcome',r['status']],f'score-{r["group"]}')
        shutil.copy2(Path(r['path'])/'prediction_scores.json',OUT/f'prediction-scores-{r["group"]}.json')
        write_json(OUT/f'audit-{r["group"]}-final.json',branch_audit(OUT/f'long-{r["group"]}-42-to497.json'))
    notify('finished',results=[r['result'] for r in records])


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if '--self-check' not in sys.argv:
            write_json(HOLD,dict(error_type=type(exc).__name__,error=str(exc)))
            notify('supervisor_failed',error_type=type(exc).__name__,error=str(exc))
        raise
