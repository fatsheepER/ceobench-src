"""Agent-facing text for PF queries and the weekly check: one line per item, one detail level."""
from collections import Counter
import difflib
import json
import re


def short(text, limit=100):
    text = ' '.join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + '...'


_HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\2[ \t]*(?:\n|$)", re.S)


def command_summary(command, limit=100):
    """A command without heredoc bodies or a leading cd, e.g. the script it runs."""
    command = _HEREDOC.sub(lambda m: m.group(0).split('\n', 1)[0] + '\n', command)
    parts = [p.strip() for p in re.split(r'\n|&&|;', command) if p.strip()]
    parts = [p for p in parts if not re.fullmatch(r'cd(\s+\S+)?(\s+2>/dev/null)?(\s*\|\|\s*cd\s+\S+)?', p)]
    runs = [p for p in parts if re.search(r'\bpython\b|novamind-operation', p)]
    return short(' ; '.join(runs or parts) or command, limit)


def call_text(parsed, limit=80):
    args = json.dumps(parsed.get('args') or {}, ensure_ascii=False, separators=(',', ':'), sort_keys=True)
    return parsed.get('tool', '?') + ('(' + short(args[1:-1], limit) + ')' if args != '{}' else '()')


def label(layer, kind, request, query_sql=None, object_id=None, classification=None, record=None):
    """What a saved version is, in words the agent used when producing it."""
    request = request or {}
    if record is not None:
        return 'text ' + record['version']
    if layer in ('file_bytes', 'file_text'):
        return 'file ' + (object_id or '?')
    if layer == 'server_public_response':
        if query_sql is not None:
            return 'SQL: ' + short(query_sql, 110)
        if request.get('path') == '/vars':
            return 'read current_day'
        parsed = request.get('parsed') or {}
        if parsed.get('tool'):
            return ('write ' if classification == 'write_receipt' else 'read ') + call_text(parsed)
        return (request.get('method', '') + ' ' + request.get('path', '')).strip() or 'public response'
    if layer == 'dashboard':
        return 'dashboard'
    if layer == 'registered_script':
        return 'weekly script ' + (object_id or '?')
    command = request.get('command')
    origin = ('`' + command_summary(command) + '`' if command else
              'python ' + request['source'] if request.get('source') else
              request.get('name') or kind)
    if layer == 'tool_return':
        return 'output of ' + origin
    if layer in ('stdout', 'stderr'):
        return layer + ' of ' + origin
    if layer == 'executed_code':
        return 'code run by ' + origin
    if layer == 'query_model_projection':
        return 'query result as shown'
    return layer


# --- change summaries ------------------------------------------------------------

def _scalar(value):
    return value is None or isinstance(value, (str, int, float, bool))


def number(value):
    """40.0 -> 40; other values unchanged."""
    return int(value) if isinstance(value, float) and value.is_integer() else value


def _fmt(value):
    return json.dumps(value, ensure_ascii=False) if isinstance(value, str) else str(number(value))


def _row_key(row):
    return tuple((k, v) for k, v in sorted(row.items()) if isinstance(v, str) or isinstance(v, bool) or v is None)


def rows_change(old_rows, new_rows):
    """e.g. '2 of 6 rows differ; group_id="S2" status="active": n 58→71'."""
    old_count, new_count = Counter(json.dumps(r, sort_keys=True) for r in old_rows), \
        Counter(json.dumps(r, sort_keys=True) for r in new_rows)
    differing = max(sum((old_count - new_count).values()), sum((new_count - old_count).values()))
    head = (f'rows {len(old_rows)}→{len(new_rows)}, ' if len(old_rows) != len(new_rows) else '') + \
        f'{differing} of {max(len(old_rows), len(new_rows))} rows differ'
    old_keys, new_keys = Counter(map(_row_key, old_rows)), Counter(map(_row_key, new_rows))
    by_key = {_row_key(r): r for r in old_rows if old_keys[_row_key(r)] == 1}
    examples = []
    for row in new_rows:
        key = _row_key(row)
        before = by_key.get(key) if new_keys[key] == 1 else None
        if before is None or before == row:
            continue
        cells = [f'{k} {_fmt(before.get(k))}→{_fmt(v)}' for k, v in row.items() if before.get(k) != v]
        name = ' '.join(f'{k}={_fmt(v)}' for k, v in key) or 'row'
        examples.append(name + ': ' + ', '.join(cells))
        if len(examples) == 2:
            break
    return head + ('; ' + '; '.join(short(e, 90) for e in examples) if examples else '')


def _leaves(value, path=''):
    if _scalar(value):
        yield path or '/', value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _leaves(value[key], f'{path}.{key}' if path else str(key))
    else:
        for i, item in enumerate(value):
            yield from _leaves(item, f'{path}[{i}]')


def json_change(old, new):
    before, after = dict(_leaves(old)), dict(_leaves(new))
    changed = [k for k in after if k in before and before[k] != after[k]]
    added, removed = len(after.keys() - before.keys()), len(before.keys() - after.keys())
    parts = [f'{len(changed)} values differ'] if changed else []
    if added or removed:
        parts.append(f'+{added} −{removed} fields')
    examples = [f'{k} {_fmt(before[k])}→{_fmt(after[k])}' for k in changed[:2]]
    return ', '.join(parts or ['changed']) + ('; ' + '; '.join(short(e, 90) for e in examples) if examples else '')


def text_change(old, new):
    added = removed = 0
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm='', n=0):
        if line.startswith('+') and not line.startswith('+++'):
            added += 1
        elif line.startswith('-') and not line.startswith('---'):
            removed += 1
    return f'+{added} −{removed} lines'


def change(kind, old_raw, new_raw):
    """Short description of how new_raw differs from old_raw for one evidence kind."""
    try:
        if kind == 'query':
            old, new = json.loads(old_raw), json.loads(new_raw)
            if not new.get('success', True):
                return 'the query now fails'
            return rows_change(old.get('rows') or [], new.get('rows') or [])
        if kind == 'public':
            old, new = json.loads(old_raw), json.loads(new_raw)
            return json_change(old.get('data', old), new.get('data', new))
        text_old, text_new = old_raw.decode('utf-8'), new_raw.decode('utf-8')
    except (ValueError, UnicodeDecodeError, AttributeError, TypeError):
        return 'changed'
    if kind == 'json':
        try:
            return json_change(json.loads(text_old), json.loads(text_new))
        except ValueError:
            pass
    return text_change(text_old, text_new)


# --- page rendering --------------------------------------------------------------

def item_line(item):
    """query7@v2 · day 7 · what (flags)."""
    day = f"day {item['day']}" if item.get('day') is not None else 'day ?'
    if item.get('record'):
        state = item['text_status'] + ('' if item.get('current_revision', True) else ', superseded')
        what = f"text {item['record']} ({state}): \"{short(item.get('text', ''), 90)}\""
    else:
        what = item['what']
        if item.get('rows') is not None:
            what = what.replace('SQL: ', f"SQL ({item['rows']} rows): ", 1)
    flags = [item['status']] if item.get('status') not in (None, 'succeeded') else []
    flags += ['truncated'] if item.get('truncated') else []
    flags += ['partial capture'] if item.get('extent', 'full') != 'full' else []
    if 'snapshot_day' in item:
        flags += [f"survey day {item['snapshot_day']}; acquired day {item.get('day')}; checked day {item['checked_day']}",
                  item['refresh_hint']]
    flags += [f"unchanged since day {item['unchanged_since_day']}"] if item.get('unchanged_since_day') is not None else []
    flags += [f"same content as {item['same_content_as']}"] if item.get('same_content_as') else []
    return f"{item['version']} · {day} · {what}" + (f" ({', '.join(flags)})" if flags else '')


def item_detail(item, wanted=None):
    parts = []
    if item.get('about'):
        parts.append('about ' + item['about'])
    reads = item.get('model_reads') or {}
    if not item.get('record'):  # Registered texts are the agent's own writing.
        parts.append('seen by you: ' + ('never' if not reads.get('count') else
                                        ('in full' if reads.get('full') else 'partly') +
                                        (f", last day {reads['last_day']}" if reads.get('last_day') is not None else '')))
    if wanted and item.get('objects'):
        hit = next((o for o in item['objects'] if o['kind'] == wanted.get('kind', o['kind']) and str(o['id']) == wanted['id']), {})
        others = len({(o['kind'], str(o['id'])) for o in item['objects']}) - 1
        where = 'declared' if hit.get('basis') == 'agent_declaration' else 'at ' + str(hit.get('field') or hit.get('basis'))
        parts.append(f"{wanted['id']} {where}" + (f' (+{others} other objects)' if others > 0 else ''))
    if item.get('applies_at'):
        parts.append('applies ' + json.dumps(item['applies_at'], separators=(',', ':')))
    if item.get('reason') and item.get('record'):
        parts.append('reason: ' + short(item['reason'], 80))
    if item.get('previous_version'):
        parts.append('previous ' + item['previous_version'])
    if item.get('capture_gaps'):
        parts.append('not observed: ' + ', '.join(g.replace('unobserved_', '').replace('_', ' ')
                                                   for g in item['capture_gaps']))
    return '    ' + ' · '.join(parts)


def _more(page, what):
    if page.get('next_cursor'):
        return f"{page['remaining']} more {what}: pf more {page['next_cursor']}."
    return ''


def render_list(page, title, wanted=None):
    items, detail = page['items'], page.get('detail')
    start = page['total'] - page['remaining'] - len(items) + 1
    lines = [f"{title}: {page['total']} saved, newest first" +
             (f"; showing {start}–{start + len(items) - 1}." if items else '.')]
    for item in items:
        lines.append(item_line(item))
        if detail:
            lines.append(item_detail(item, wanted))
    tail = ' '.join(filter(None, [_more(page, 'older'), f"Read one: pf show {items[0]['version']}." if items else '']))
    return '\n'.join(lines + ([tail] if tail else []))


def log_line(item):
    """MEMORY.md@v9 · day 49 · +44 −48 lines · note (day 49): "..."."""
    if item.get('record'):
        return (f"{item['record']} · day {item['day']} · {item['text_status']}: \"{short(item.get('text', ''), 90)}\""
                + (f" · reason: {short(item['reason'], 80)}" if item.get('reason') else ''))
    parts = [item['version'], f"day {item['day']}"]
    if item.get('rows') is not None:
        parts.append(plural(item['rows'], 'row'))
    if item.get('size'):
        parts.append(item['size'])
    flags = [item['status']] if item.get('status') not in (None, 'succeeded') else []
    flags += ['truncated'] if item.get('truncated') else []
    flags += [f"same content as {item['same_content_as']}"] if item.get('same_content_as') else []
    if flags:
        parts.append(', '.join(flags))
    if item.get('note'):
        parts.append(f"note: \"{short(item['note'][1], 150)}\"")
    return ' · '.join(parts)


def render_log(page):
    items = page['items']
    root = page['root']
    what = f"text {root['record'].split('.')[0]}" if root.get('record') else root['what']
    noun = 'revision' if page.get('kind') == 'record' else 'version'
    lines = [f"{what}: {plural(page['total'], noun)}, newest first"]
    lines += [log_line(item) for item in items]
    tail = [_more(page, 'older')]
    if not root.get('record') and items:
        newest = items[0]['version']
        tail.append(f'pf show {newest}' + (f" · pf diff {items[1]['version']} {newest}" if len(items) > 1 else '')
                    + (f" · pf blame {newest.rsplit('@', 1)[0]}" if root.get('layer') == 'file_bytes' else ''))
    return '\n'.join(lines + [' '.join(filter(None, tail))])


def render_search(page):
    wanted = page['object']
    counts = ', '.join(plural(n, noun.lower().rstrip('s')) for noun, _, n in page['sections'])
    lines = [f"{wanted['id']}: {counts}, newest first."]
    for title, items, total in page['sections']:
        if not items:
            continue
        lines.append(f'{title}:' + (f' (latest {len(items)} of {total})' if total > len(items) else ''))
        for item in items:
            lines.append('  ' + item_line(item))
    lines.append(f"pf search {wanted['id']} --all lists all {plural(page['total'], 'saved item')}.")
    return '\n'.join(lines)


def status(check, row):
    """One phrase for one dependency's check result."""
    if row.get('historical_only'):
        return 'historical only, not checked'
    if check is None:
        return 'not checked'
    if check.get('reason') == 'write_receipt':
        return 'write receipt, not compared'
    now = f" (now {check['current_version']})" if check.get('version_changed') and check.get('current_version') else ''
    if check['predicate_result'] == 'fails':
        return f'PREDICATE FAILS{now}: ' + check.get('summary', '')
    if check['predicate_result'] == 'holds':
        return (f'predicate holds{now}: ' if check['version_changed'] else 'unchanged; predicate holds: ') + check.get('summary', '')
    if check['predicate_result'] == 'cannot_check':
        return 'cannot check: ' + str(check.get('reason'))
    if check['version_changed']:
        return f'changed{now}: ' + check.get('summary', 'changed')
    if check['affected']:
        return 'unchanged itself; ' + check.get('sources', 'a source changed')
    return 'unchanged'


def dependency_line(row, relation=True):
    target = row['target']
    what = item_line(target) if target else 'evidence unavailable'
    if target is None:
        what += f" ({row.get('reason') or row.get('unexpanded')})"
    head = (row['relation'] + ' ' if relation else '') + what
    extras = [status(row.get('check'), row)]
    if row.get('predicate'):
        extras.append('predicate ' + json.dumps({k: number(v) for k, v in row['predicate'].items()},
                                                 separators=(',', ':'), sort_keys=True))
    if row.get('note'):
        extras.append('note: ' + short(row['note'], 120))
    if row.get('unexpanded') == 'depth_limit':
        extras.append('depth limit; dependencies beyond this item were not expanded')
    return head + ' — ' + ' — '.join(extras)


def render_dependencies(page):
    root, detail = page['root'], page.get('detail')
    checked = page.get('stale_check') == 'performed'
    lines = [item_line(root),
             (f"Checked day {page.get('checked_day')}, against the current world within the expanded dependencies; advisory." if checked else
              'Not checked (--history lists history only).')]
    if page.get('unexpanded'):
        lines.append('Incomplete: depth limit reached at ' + ', '.join(page['unexpanded']) +
                     '. Continue with ' + '; '.join('pf depend ' + v for v in page['unexpanded']) + '.')
    if not page['items']:
        lines.append('No recorded dependencies.' if detail else
                     f"No declared references. pf depend {root['version']} --detail traces what produced it.")
    for i, row in enumerate(page['items'], 1):
        indent = '  ' * (row['depth'] - 1) if detail else ''
        lines.append(f'{indent}{i}. ' + dependency_line(row))
    tail = [_more(page, 'items')]
    if not detail:
        tail.append(f"pf depend {root['version']} --detail lists the expanded queries, reads and files.")
    # Name a pair that was just compared, so the example is directly usable.
    pair = next(((row['target']['version'], row['check']['current_version']) for row in page['items']
                 if row.get('target') and (row.get('check') or {}).get('current_version') not in
                 (None, row['target']['version'])), None)
    if pair:
        tail.append('See what changed: pf diff %s %s.' % pair)
    return '\n'.join(lines + [' '.join(filter(None, tail))])


def render_dependents(page):
    root, detail = page['root'], page.get('detail')
    lines = ['Referrers of ' + item_line(root)]
    if not page['items']:
        lines.append('No active registered text cites it.')
    for i, row in enumerate(page['items'], 1):
        source = row['source']
        via = row['path'][1:-1]
        how = 'cites it directly' if not via else 'via ' + ' → '.join(reversed(via))
        extras = [how] + (['historical only'] if row.get('historical_only') else []) + \
                 (['older revision, leads to a current one'] if row.get('traversal_only') else [])
        if row.get('note'):
            extras.append('note: ' + short(row['note'], 120))
        lines.append(f'{i}. ' + item_line(source) + ' — ' + ' — '.join(extras))
    tail = [_more(page, 'referrers')]
    if not detail:
        tail.append(f"pf rdepend {root['version']} --detail adds scripts, files and outputs derived from it.")
    return '\n'.join(lines + [' '.join(filter(None, tail))])


def plural(n, noun):
    return f'{n} {noun}' + ('' if n == 1 else 's')


def render_weekly(entries, day, pf, ended=(), underlying=(), condition_only=0, pending=0):
    """Week-start check: texts whose directly cited evidence changed, then short summaries.

    Only what the agent itself cited is itemized; changes in the queries behind a cited
    output are normal week-to-week data changes and share one line.
    """
    flagged = [e for e in entries if e['changed']]
    lines = [f'=== Check of your registered texts (day {day}) ===']
    checked = plural(len(entries), 'active text') + ' checked'
    if not flagged and pending:
        lines.append(f'{checked}; {condition_only} cover continuing conditions only. No new direct issues in these checks.')
    elif not flagged:
        lines.append(checked + ': cited files and texts are unchanged' + (' and no predicate fails.' if pf else '.'))
    else:
        lines.append(f'{checked}; {len(flagged)} with changed evidence:')
    for entry in flagged[:8]:
        text = entry['text']
        lines.append(f"{text['record']} (day {text['day']}): \"{short(text['text'], 90)}\"")
        for row in entry['changed'][:3]:
            lines.append('  - ' + (row if isinstance(row, str) else short(dependency_line(row, relation=False), 320)))
        if len(entry['changed']) > 3:
            lines.append(f"  - and {len(entry['changed']) - 3} more")
    if len(flagged) > 8:
        lines.append('Also changed: ' + ', '.join(e['text']['record'] for e in flagged[8:]))
    if underlying:
        lines.append(f"The queries behind outputs cited by {', '.join(underlying)} now return different rows; "
                     f"pf depend {underlying[0].split('.')[0]} shows which.")
    if ended:
        lines.append(f"Not checked because their applies window is over: {', '.join(ended)} "
                     '(text_retire them if you no longer use them).')
    if flagged:
        lines.append('Details: pf depend rN.' if pf else 'Details: run the git diff shown on each line.')
    return '\n'.join(lines)
