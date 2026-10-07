"""Replay fixed historical refresh receipts; measure indexing/checking without model or SQL calls.

The original run is opened read-only. A temporary metadata-only ledger receives new
audit records; original blobs are verified lazily through the read-only connection.
"""
import argparse
from contextlib import closing
import copy
import json
from pathlib import Path
import sqlite3
import statistics
import subprocess
import tempfile
import time
import types

from saas_bench.pf_queries import PFQueries
from saas_bench.sql_evidence import SQLEvidenceStore
from saas_bench.text_registry import TextRegistry


def readonly(path):
    connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def sample(run, day, temporary):
    source = run / 'sql-evidence.sqlite'
    identity = json.loads((run / 'manifest.json').read_text())['sql_evidence']
    clean_identity = {k: v for k, v in identity.items() if k not in ('parent_branch', 'fork_seq', 'source_manifest_sha256')}
    store = SQLEvidenceStore(temporary / 'replay.sqlite', clean_identity)
    with closing(readonly(source)) as conn:
        parent = conn.execute("SELECT rowid,event_id FROM requests WHERE json_extract(request,'$.kind')='pf_weekly_check' "
                              "AND json_extract(request,'$.request.day')=?", (day,)).fetchone()
        if parent is None:
            raise ValueError(f'No recorded weekly check for day {day}')
        end = conn.execute("SELECT min(rowid)-1 FROM requests WHERE rowid>? "
                           "AND json_extract(request,'$.kind')='model_request'", (parent['rowid'],)).fetchone()[0]
        if end is None:
            end = conn.execute('SELECT max(rowid) FROM requests').fetchone()[0]
        receipts = conn.execute("SELECT r.event_id,s.record FROM requests r JOIN results s USING(event_id) "
                                "WHERE json_extract(r.request,'$.parent_event_id')=?", (parent['event_id'],)).fetchall()
    with closing(sqlite3.connect(store.path, uri=True)) as conn, conn:
        conn.execute('ATTACH DATABASE ? AS original', (source.resolve().as_uri() + '?mode=ro',))
        conn.execute('DELETE FROM branches')
        conn.execute('INSERT INTO branches SELECT * FROM original.branches')
        conn.execute('INSERT INTO queries SELECT * FROM original.queries')
        conn.execute('INSERT INTO requests SELECT * FROM original.requests WHERE rowid<=?', (end,))
        for table in ('results', 'deliveries', 'versions', 'client_calls'):
            conn.execute(f'INSERT INTO {table} SELECT o.* FROM original.{table} o JOIN requests r USING(event_id)')
    store.identity = identity
    original = copy.copy(store)
    original.connect = lambda: readonly(source)
    local_content = store.get_content
    def content(version):
        with closing(store.connect()) as conn:
            local = conn.execute('SELECT 1 FROM versions v JOIN blobs b ON v.content_hash=b.content_hash WHERE version_id=?',
                                 (version,)).fetchone()
        return local_content(version) if local else original.get_content(version)
    store.get_content = content
    snapshot = next(p.parent for p in (run / 'checkpoints').glob('*/checkpoint.json')
                    if json.loads(p.read_text())['day'] == day)
    registry = TextRegistry(snapshot / 'agent_workspace', 'pf', store, sim_day=lambda: day)
    query = PFQueries(registry)
    mapped, by_key = {}, {}
    query._index(None)
    for receipt in receipts:
        record = json.loads(receipt['record'])
        version = record.get('refresh_of')
        if version:
            result = receipt['event_id'] + ':public_response'
            mapped[version] = result
            by_key[query._key(version)] = result
    refresh_calls = []
    def replay(versions, parent):
        refresh_calls.append(len(versions))
        query.replay_missing = [v for v in versions if v not in mapped and query._key(v) not in by_key]
        if query.replay_missing:
            raise ValueError('Recorded refresh receipts do not cover this replay')
        return {v: mapped[v] if v in mapped else by_key[query._key(v)] for v in versions}
    query.refresh = replay
    return query, registry, refresh_calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--days', type=int, nargs='+', default=[35, 245, 490])
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    legacy = types.ModuleType('saas_bench._baseline_pf_queries')
    legacy.__package__ = 'saas_bench'
    exec(compile(subprocess.check_output(['git', 'show', '2f9969a:src/saas_bench/pf_queries.py']),
                 '<baseline-pf-queries>', 'exec'), legacy.__dict__)
    results = []
    for day in args.days:
        with tempfile.TemporaryDirectory(prefix='pf-replay-') as folder:
            query, registry, calls = sample(args.run, day, Path(folder))
            phases = {}
            for name, factory in [('baseline_index', lambda: legacy.PFQueries(registry)),
                                  ('cold_index', lambda: PFQueries(registry)),
                                  ('warm_index', lambda: query)]:
                values = []
                for _ in range(args.repeats):
                    instance = factory()
                    started = time.perf_counter()
                    if name == 'baseline_index':
                        instance._index(None)
                    else:
                        with registry.resolver.cached():
                            instance._index(None)
                    values.append(time.perf_counter()-started)
                phases[name] = dict(median=statistics.median(values), p95=max(values), samples=values)
            checks = [[], []]
            for _ in range(args.repeats):
                query.store.save_state('pf_review', dict(pending={}, ended=[]))
                query.store.save_state('pf_refresh_failures', {})
                checking = PFQueries(registry)
                checking.refresh = query.refresh
                with registry.resolver.cached():
                    checking._index(None)
                for phase in checks:
                    calls.clear()
                    started = time.perf_counter()
                    text = checking.weekly_check(day)
                    assert not getattr(query, 'replay_missing', []), query.replay_missing
                    phase.append(dict(seconds=time.perf_counter()-started, refreshed_sources=sum(calls),
                        chars=len(text or ''), pending=len((query.store.load_state('pf_review') or {}).get('pending', {}))))
            checks = [dict(median=statistics.median(s['seconds'] for s in samples),
                           p95=max(s['seconds'] for s in samples), samples=samples) for samples in checks]
            result = dict(day=day, phases=phases, fixed_receipt_checks=checks)
            results.append(result)
            print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(method='read-only historical receipts; excludes live query/model latency',
        baseline_commit='2f9969a', source_commit=subprocess.check_output(['git', 'rev-parse', '--short=7', 'HEAD'], text=True).strip(),
        repeats=args.repeats, p95_method='maximum of small repeated sample', results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
