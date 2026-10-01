"""Audit PF/Git use and weekly check delivery from run-pointer JSON files (no model calls)."""
import argparse
import collections
from contextlib import closing
import glob
import json
from pathlib import Path
import re
import sqlite3

from saas_bench import pf_cli

CLOCK = re.compile(r'20\d\d-\d\d-\d\dT\d\d:\d\d')
RECORD = re.compile(r'\br\d+(?:\.\d+)?\b')
HISTORY = re.compile(r'\bgit\s+(log|diff|show)\b')
CHECK = re.compile(r'^=== (?:Weekly check|Check of your registered texts)\b', re.M)
VERBS = dict(pf_log='log', pf_read='show', pf_diff='diff', pf_blame='blame',
             pf_dependencies='depend', pf_dependents='rdepend', pf_search='search', pf_more='more',
             pf_usage='usage', pf_help='help')


def pf_call(tool, roots, captured):
    """Prefer execution metadata; use the real CLI parser for logs made before it existed."""
    audit = tool.get('pf_call')
    if not audit:
        operation, args = tool['tool'], tool.get('arguments') or {}
        if operation == 'bash':
            try:
                parsed = pf_cli.parse(args.get('command', ''), roots)
            except pf_cli.Usage:
                parsed = ('pf_usage', {}, None)
            if not parsed:
                return None
            operation, args, _ = parsed
        if operation not in VERBS:
            return None
        audit = dict(operation=operation, arguments=args, event_id=None,
                     outcome='usage_error' if operation == 'pf_usage' else
                             'execution_error' if (tool.get('result') or '').startswith('Error:') else 'succeeded')
    audit = dict(audit)
    verb = VERBS[audit['operation']]
    if audit['operation'] == 'pf_read':
        verb = dict(history='log', diff='diff', content='show')[audit['arguments'].get('mode', 'content')]
    event = captured.get(audit['event_id'])
    match = None if not audit['event_id'] else bool(event and
        event['kind'] == audit['operation'] and event['arguments'] == audit['arguments'] and
        event['status'] == ('succeeded' if audit['outcome'] == 'succeeded' else 'failed') and
        (tool['tool'] != 'bash' or event['command'] == tool['arguments'].get('command')))
    return dict(audit, command=verb, tool=tool['tool'], chars=len(tool.get('result') or ''),
                error=audit['outcome'] != 'succeeded', capture_match=match)


def pf_calls(tool, roots, captured):
    """New Bash logs contain one receipt per subprocess; retain old single-call logs."""
    if 'pf_calls' in tool:
        return [pf_call(dict(tool, pf_call=call), roots, captured) for call in tool['pf_calls']]
    call = pf_call(tool, roots, captured)
    return [call] if call else []


def response_calls(response):
    """Count only this response, never the historical calls in its input messages."""
    if response.get('choices'):
        return response['choices'][0]['message'].get('tool_calls') or []
    return [item for item in response.get('output', response.get('content', []))
            if item.get('type') in ('function_call', 'tool_use')]


def refresh_events(conn):
    """Classify child request receipts, including failures absent from rendered digests."""
    rows = conn.execute('''SELECT r.event_id,r.request,s.record,p.request FROM requests r
        LEFT JOIN results s USING(event_id)
        JOIN requests p ON p.event_id=json_extract(r.request,'$.parent_event_id')
        WHERE json_extract(p.request,'$.kind') IN ('pf_dependencies','pf_weekly_check')
        AND json_extract(r.request,'$.kind') IN ('sql_query','public_http')''')
    for event, request, result, parent in rows:
        req, res, p = json.loads(request), json.loads(result) if result else {}, json.loads(parent)
        execution = res.get('execution') or res
        http = res.get('http_status')
        attempted = execution.get('attempted', True)
        outcome = ('result_unknown' if not result else 'not_attempted' if not attempted else
                   'timeout' if http == 504 else 'succeeded' if res.get('status') == 'succeeded' else 'error')
        yield dict(event_id=event, parent_event_id=p['event_id'] if 'event_id' in p else
                   req.get('parent_event_id'), day=(p.get('request') or {}).get('day', execution.get('day')),
                   kind=req['kind'], status=res.get('status'), http_status=http, outcome=outcome,
                   attempted=attempted, reason=execution.get('refresh_error'))


def text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(text(c.get('text') or c.get('content') or '') if isinstance(c, dict) else str(c) for c in content)
    return str(content or '')


def analyze(pointer):
    record = json.loads(pointer.read_text())
    run = Path(record['path'])
    start = record['start_day']
    captured, refreshes = {}, []
    evidence = run / 'sql-evidence.sqlite'
    if evidence.exists():
        with closing(sqlite3.connect(evidence.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
            refreshes = list(refresh_events(conn))
            for event, request, result in conn.execute('''SELECT r.event_id,r.request,s.record
                    FROM requests r LEFT JOIN results s USING(event_id)
                    WHERE json_extract(r.request,'$.kind') LIKE 'pf_%' '''):
                req, res = json.loads(request), json.loads(result) if result else {}
                captured[event] = dict(kind=req['kind'], arguments=req.get('request'),
                                       command=req.get('command'), status=res.get('status'))
    roots = {'/workspace', str(run / 'agent_workspace'), str((run / 'agent_workspace').resolve())}
    requests = [json.loads(l) for l in open(run / 'logs' / 'agent_requests.jsonl')]
    requests = [r for r in requests if r.get('event') == 'request' and r['day'] >= start]
    tools = []
    for name in glob.glob(str(run / 'logs' / 'tool_results_*.jsonl')):
        tools += [json.loads(l) for l in open(name)]
    tools = [t for t in tools if t['day'] >= start]
    weeks = {}
    for r in requests:
        weeks.setdefault(r['day'], r)
    out = dict(group=record['group'], run_id=record['run_id'], status=record['status'],
               start_day=start, days_run=(record.get('result') or {}).get('days_run'), weeks=[])
    batches = collections.defaultdict(list)
    for name in glob.glob(str(run / 'logs' / 'raw_responses_*.jsonl')):
        with open(name) as stream:
            for line in stream:
                response = json.loads(line)
                batches[response['day']].append(len(response_calls(response['raw_response'])))
    for day, first in sorted(weeks.items()):
        opening = text(first['request']['messages'][1]['content'])
        check = CHECK.search(opening)
        digest = opening[check.start():] if check else None
        flagged = re.findall(r'^(r\d+\.\d+) \(day', digest or '', re.M)
        flagged += [record for line in (digest or '').splitlines() if line.startswith('Also changed:')
                    for record in RECORD.findall(line)]
        day_tools = [t for t in tools if t['day'] == day]
        reasoning = ' '.join(t['result'] or '' for t in day_tools if t['tool'] == '_reasoning')
        calls = [t for t in day_tools if not t['tool'].startswith('_')]
        pf = [item for t in calls for item in pf_calls(t, roots, captured)]
        day_refreshes = [r for r in refreshes if r['day'] == day]
        commands = ' '.join((t['arguments'] or {}).get('command', '') for t in calls if t['tool'] == 'bash')
        mentioned = sorted({m for m in RECORD.findall(reasoning) if any(m.split('.')[0] == f.split('.')[0] for f in flagged)})
        out['weeks'].append(dict(
            day=day, requests=sum(1 for r in requests if r['day'] == day),
            model_response_tool_counts=batches[day],
            multiple_call_responses=sum(n > 1 for n in batches[day]),
            requested_tool_calls=sum(batches[day]),
            digest_chars=len(digest) if digest else 0, flagged=flagged,
            digest_mentioned_in_reasoning=bool(re.search(r'weekly check|check of (?:your|my) registered texts', reasoning, re.I)),
            flagged_ids_in_reasoning=mentioned,
            tool_counts=dict(collections.Counter(t['tool'] for t in calls)),
            pf_calls=pf,
            pf_counts=dict(collections.Counter(item['command'] for item in pf)),
            pf_outcomes=dict(collections.Counter(item['outcome'] for item in pf)),
            note_calls=dict(collections.Counter(t['tool'] for t in calls if
                isinstance((t.get('arguments') or {}).get('note'), str) and t['arguments']['note'].strip())),
            history_commands=HISTORY.findall(commands),
            revisions=[t['arguments'].get('record') for t in calls if t['tool'] in ('text_revise', 'text_retire')],
            tips=sum('Business writes this week' in (t['result'] or '') or '"tip"' in (t['result'] or '')
                     for t in calls if t['tool'] in ('text_create', 'text_revise')),
            refresh_events=day_refreshes,
            refresh_outcomes=dict(collections.Counter(r['outcome'] for r in day_refreshes)),
            refresh_errors=sum(r['outcome'] in ('error', 'timeout') for r in day_refreshes),
            clock_in_results=sum(bool(CLOCK.search(t['result'] or '')) for t in calls
                                 if t.get('pf_call') or t['tool'].startswith(('pf_', 'text_'))) + bool(CLOCK.search(digest or '')),
            digest=digest))
    registrations = run / 'agent_workspace' / 'registrations.json'
    if registrations.exists():
        state = json.loads(registrations.read_text())['records']
        refs = [ref for revs in state.values() for ref in revs[-1]['references']]
        out['registrations'] = dict(texts=len(state), references=len(refs),
                                    predicates=sum(bool(r.get('predicate')) for r in refs),
                                    clock_fields=sum('registered_at' in rev for revs in state.values() for rev in revs))
    usage = json.loads((run / 'usage_summary.json').read_text())
    # Cumulative over the lineage: a branch includes its prefix.
    out['usage'] = {role: dict(calls=usage[role]['calls'], input=usage[role]['known']['input_tokens'],
                               cached=usage[role]['known']['cached_tokens'], output=usage[role]['known']['output_tokens'],
                               usd=round(usage[role]['known_cost_usd'], 6), missing_cost=usage[role]['missing_cost'])
                    for role in ('agent', 'simulator') if role in usage}
    timing = []
    for name in glob.glob(str(run / 'logs' / 'timing_*.jsonl')):
        timing += [json.loads(l) for l in open(name)]
    out['weekly_check_seconds'] = {t['day']: t['elapsed_s'] for t in timing if t.get('event') == 'weekly_check'}
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pointers', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = json.dumps([analyze(p) for p in args.pointers], ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + '\n')
    else:
        print(output)
