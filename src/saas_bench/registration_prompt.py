"""System prompt of the registration groups (common prefix, Git and PF).

The original prompt stays the single source. Registration and file-history text is
integrated at fixed anchors, each of which must occur exactly once in the rendered
original prompt, so a change to the original template fails loudly instead of silently
dropping an edit. The original group never passes through here.
"""

REGISTERED_TEXTS = '''## Registered Texts

Use `text_create` to register a hypothesis, forecast, plan, conclusion or
counterevidence you expect to rely on in later weeks, citing the evidence it
rests on. Texts get IDs r1, r2, ...; revisions are r4.1, r4.2. They persist in
`registrations.json` but are not loaded automatically: keep useful IDs in
MEMORY.md and review texts with `text_list`. Registering is optional and never
blocks business actions.

Each reference cites one piece of evidence and may carry a short note on what
you use it for. Its purpose is "current" (default) when the text relies on it
now, or "historical_only" when it only records a past fact. `applies` gives the
simulated days the text is about: "49", "49-55", "49-" (from day 49 until
revised or retired) or "unknown: <reason>". Each week starts with a check of
your active texts, shown after the dashboard.

Example, shortened from an actual week-7 finding:

```
text_create {{"text": "S1 leads are linear in ad spend: ~165 leads per $1000/day from $2.9K to $8K/day, no saturation yet.",
             "objects": [{{"kind": "customer_group", "id": "S1"}}],
             "references": [{{"cite": "{cite}", "note": "daily leads & subs by start_day d29-49"}}],
             "applies": "49-", "reason": "ad spend test result"}}
```

The dashboard reflects the last weekly advance; after changing settings within
a week, query the current settings instead.

'''

GIT_HISTORY = '''## File History (git)

Your working directory is a Git repository. The harness commits it at the end
of every week with the message "Week N (day 7N) [week-N]". In `text_create`,
cite a file by its path (the version this week's closing commit will store, so
later edits this week are included), an earlier commit as
`notes/observations.md@week-6` or `notes/observations.md@3f9a2c1`, another text
as `r4.2`, or `unknown: <reason>`. The weekly check compares the files and texts
your active texts cite with their current versions.

Example: MEMORY.md is rewritten almost every week. Before relying on something
you remember from earlier weeks, `git log -p -2 -- MEMORY.md` shows what the
last rewrites removed.

'''

PF_HISTORY = '''## File and Output History (pf)

PF records every version of your workspace files and every command output,
including outputs you never saved. After a command, a line such as
`[pf: scripts/an_w7d.py.out@v1 | wrote scripts/an_w7d.py@v1]` names them:
NAME@vK is version K of that file or output, and K rises only when the content
changes. In `text_create`, cite such a handle, a file path (the version you last
saw or wrote), a committed file (`MEMORY.md@week-6`), another text (`r4.2`) or
`unknown: <reason>`. The weekly check reports cited files and texts that changed
and predicates that no longer hold.

After an ordinary reference needs review, the weekly check retains its first
finding and last verification day instead of rerunning it each week. Revise the
text or run `pf depend rN --detail` to review it; incomplete scopes leave uncovered
issues pending. Explicit selected values and predicates keep being checked.
Repeated timeouts retry after 1, 2, 4, then 8 simulated weeks; `pf depend` retries
immediately. Fix invalid SQL or replace its reference before relying on it.

Group insights show their survey's `snapshot_day`, separately from the day you
retrieve them. Only a completed `research_group` survey updates that date.

`bash`, `write_file` and `edit_file` take an optional `note`: why you ran the
command or what the change is for. PF keeps it with the outputs and files of
that call and shows it when you see them again, e.g. above MEMORY.md next week.

Run `pf` in bash; commands, loops, pipes and redirects follow normal Bash rules.
Citing a whole captured object does not require reading all of it. PF records
what you actually read separately; selected fields must have reached you.
Compact DELTA/UNCHANGED returns name the exact baseline and a `pf show ... --full`
command. Use that command if you need the saved full output.

Commands:

```
pf log MEMORY.md           versions of a file, an output (scripts/an_w7d.py.out) or a text (r4)
pf show MEMORY.md@v8       one saved version
pf diff MEMORY.md          what the latest version changed; or pf diff MEMORY.md@v6 MEMORY.md@v8
pf blame MEMORY.md         the version and day each line was written
pf depend r4               what r4 cites; reruns the queries behind it and reports changes
pf rdepend scripts/an_w7d.py.out@v1   registered texts that cite it
pf search S1               your texts, business writes and latest outputs about S1 (or t10_2, ...)
```

Example: MEMORY.md is rewritten almost every week. Before relying on something
you remember from earlier weeks, `pf diff MEMORY.md` shows what the last rewrite
removed.

'''

GLOB_ROW = '| `glob_files` | Find files by pattern (like `*.py`, `docs/**/*.json`) |'
WEEKLY_SCRIPTS = '3. **Weekly scripts** — registered via `novamind register-daily-script`'


def edits(pf):
    """(original, replacement) pairs; every original must occur exactly once."""
    history = PF_HISTORY if pf else GIT_HISTORY
    cite = 'scripts/an_w7d.py.out@v1' if pf else 'notes/observations.md'
    return [
        ('You have 6 tools:', 'You have 10 tools:'),
        (GLOB_ROW, GLOB_ROW + '\n| `text_create`, `text_revise`, `text_retire`, `text_list` | Register, revise, '
                   'retire and list texts you will rely on in later weeks (see Registered Texts) |' +
                   ('\n\n`bash` also runs `pf`, which shows the history of your files and command outputs '
                    '(see File and Output History).' if pf else '')),
        (WEEKLY_SCRIPTS, WEEKLY_SCRIPTS + '\n4. **Registered texts** — saved by `text_create` in '
                         '`registrations.json`; not loaded automatically (use `text_list`)\n' +
                         ('5. **PF history** — every version of your files and command outputs, with your '
                          'notes (use `pf`)' if pf else
                          '5. **Git history** — the harness commits your working directory at the end of '
                          'every week (use `git log`)')),
        ('MEMORY.md is the ONLY way to carry knowledge forward.',
         'MEMORY.md is the only file loaded into your context automatically.'),
        ('## Weekly Workflow', REGISTERED_TEXTS.format(cite=cite) + history + '## Weekly Workflow'),
        ('1. **Read the dashboard** (automatically shown at start of week)',
         '1. **Read the dashboard** (automatically shown at start of week, followed by the check of your '
         'registered texts)'),
        ('2. **Recall context** — read your notes/files from previous weeks',
         '2. **Recall context** — read your notes/files from previous weeks (' +
         ('`pf log` and `pf diff` show how they changed)' if pf else '`git log -p` shows how they changed)')),
        ('5. **Save what matters** — update your files with observations, decisions, learnings',
         '5. **Save what matters** — update your files with observations, decisions, learnings, and '
         'register or revise the texts you will rely on later'),
    ]


def integrate(prompt, pf=False):
    """The original system prompt with the registration group's sections in place."""
    for old, new in edits(pf):
        if prompt.count(old) != 1:
            raise ValueError('Registration prompt anchor must occur exactly once: ' + old[:60])
        prompt = prompt.replace(old, new)
    return prompt


# The weekly MEMORY header of the registration groups; the original group keeps
# "at the start of every day" word for word.
MEMORY_HEADER = ('\n\n## Your MEMORY.md (auto-loaded)\n\n'
                 'The following is the contents of your MEMORY.md file. '
                 'This is automatically loaded into your context at the start of every week.\n')


def git_memory_line(workspace):
    """[git: MEMORY.md last committed in week-7 (3f9a2c1) | ...], or None before any commit."""
    import re
    import subprocess
    try:
        result = subprocess.run(['git', '-C', str(workspace), 'log', '-1', '--format=%H %s', '--', 'MEMORY.md'],
                                capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode or not result.stdout.strip():
        return None
    full, subject = (result.stdout.strip().split(' ', 1) + [''])[:2]
    week = re.search(r'\[(week-[1-9][0-9]*)\]$', subject)
    where = f'{week.group(1)} ({full[:7]})' if week else full[:7]
    return f'[git: MEMORY.md last committed in {where} | git log -p -- MEMORY.md]'


def pf_memory_line(store, version):
    """[pf: MEMORY.md@v9, written day 49, note: "..." | 9 versions: pf log MEMORY.md]."""
    from . import evidence_handles
    from .pf_render import plural, short
    handles = evidence_handles.index(store)
    group = handles.group(version)
    if group is None:
        return None
    head = f"{handles.name(version)}, written day {group['day']}"
    if note := handles.note(version):
        head += f', note: "{short(note[1], 200)}"'
    count = len(handles.groups[group['key']])
    return f"[pf: {head} | {plural(count, 'version')}: pf log MEMORY.md]"
