"""System prompt of the registration groups (common prefix, Git and PF).

The original prompt stays the single source. Registration and file-history text is
integrated at fixed anchors, each of which must occur exactly once in the rendered
original prompt, so a change to the original template fails loudly instead of silently
dropping an edit. The original group never passes through here.
"""

RELATIONS = '''Use full handoffs and focused questions to coordinate work. When a shared setting
may be changed by more than one role, agree who will make the final change and
check its current value before replacing it. Your normal write permissions remain.

In your handoff, distinguish actions actually completed, proposals, withdrawals,
observed facts and estimates. Give the relevant day, coverage, units, assumptions,
locatable evidence and unresolved questions. Brief free prose is enough.

Register sources selectively for important judgments, files or actions that will
be reused or handed off. Use relation_batch for several targets at once. Known
sources can be submitted directly; no candidate lookup or predicate is required.
Copy exact versions and operation references from receipts. Describe missing
material explicitly. A saved relation is your declaration, not proof of validity.
Operation references use the executing role and a number, such as growth:12.
Use them in relation_batch operation_ref endpoints or business ledger SQL.
Text references[].cite accepts files, outputs and texts, not operations.
Review each item result. Revisions preserve history; a supplied sources list
replaces the whole list. Registration never blocks handoff or business progress.
Public relations and revisions are readable in documents/relations.json; use ordinary file tools
or your own scripts to search them. Use author_declared only for your own target;
use inferred for another or unknown author. Revise with the returned relation ID
and expected_revision. Missing unsaved output stays missing.
'''

REGISTERED_TEXTS = '''## Registered Texts

Use `text_create` to register a hypothesis, forecast, plan, conclusion or
counterevidence you expect to rely on in later weeks, citing the evidence it
rests on. Texts get IDs r1, r2, ...; revisions are r4.1, r4.2. They persist in
the public view `documents/registrations.json` but are not loaded automatically: keep useful IDs in
MEMORY.md and review texts with `text_list`. Registering is optional and never
blocks business actions.

Each reference cites one piece of evidence and may carry a short note on what
you use it for. Its purpose is "current" (default) when the text relies on it
now, or "historical_only" when it only records a past fact. `applies` gives the
simulated days the text is about: "49", "49-55", "49-" (from day 49 until
revised or retired) or "unknown: <reason>". Review your judgments using current
evidence when needed.

`references[].cite` accepts the evidence forms listed below. To link a concrete
business operation, copy its `operation_ref` into a `relation_batch` endpoint;
operation references are not text citations.

The dashboard reflects the last weekly advance; after changing settings within
a week, query the current settings instead.

'''

GIT_CALL_NOTES = '''
A call with a note saves its full output and changed-file snapshots in the returned
`tool-outputs/callN.txt` artifact. Use ordinary file search to find its note, and
`git show --abbrev-commit COMMIT` to inspect the note and saved file changes.
Use `git show COMMIT:PATH` for exact file bytes.
'''

GIT_HISTORY = '''## File History (git)

Your working directory is a Git repository. The harness commits it at the end
of every week with the message "Week N (day 7N) [week-N]". In `text_create`,
cite a saved file by its path to snapshot its current bytes immediately, or an earlier commit as
`notes/observations.md@week-6` or `notes/observations.md@3f9a2c1`, another text
as `r4.2`, or `unknown: <reason>`. Later edits do not change a saved snapshot.
Peer paths use `ROLE:PATH`; their snapshots are stored in your repository and
the receipt names the source. Read `PATH@COMMIT` with `git show COMMIT:PATH`,
or `git -C ROLE_WORKSPACE show COMMIT:PATH` for another role's repository.
Only named saved files are snapshotted; save SQL or command output before citing it.


Example: MEMORY.md is rewritten almost every week. Before relying on something
you remember from earlier weeks, `git log -p -2 -- MEMORY.md` shows what the
last rewrites removed.

'''

GIT_HISTORY += GIT_CALL_NOTES

PF_CAPTURE_BOUNDARY = '''Workspace files are captured before and after the outer Bash call. Files created
or changed inside that call are not refreshed in PF until it ends. Finish the
writing call, then use its returned file handles in a later Bash call. To inspect
current bytes inside the same Bash call, use ordinary file commands.

First Bash call: run your script to write `reports/w4_calc.out`, then wait for the
call to finish and return its PF receipt. Second Bash call: run
`pf log reports/w4_calc.out`, or `pf show HANDLE --full` with the returned handle.
These are two separate tool calls.
'''

PF_HISTORY = '''## File and Output History (pf)

PF saves captured workspace file versions and command outputs, including outputs
you never saved. Internal reads, intermediate writes and pipe streams may be missing;
capture details report gaps. `pf candidates` finds sources across calls, defaulting
to your role and current simulated week, 20 per page. Its header states the scope
and cutoff. Use `--history`, `--role all` or explicit day/kind/text/event filters to
broaden or narrow it, and the returned `pf more` cursors for continuation or details.
SQL `--table TABLE --column COLUMN` filters authorization observations, including
possible view expansion; `--sql-accesses` lists observed pairs, not adopted columns.
Content versions and execution events are separate. Saved, delivered and declared
used are different; candidates are observations, not proof of use. After a command, a line such as
`[pf: scripts/an_w7d.py.out@v1 | wrote scripts/an_w7d.py@v1]` names them:
NAME@vK is version K of that file or output, and K rises only when the content
changes. In `text_create`, cite such a handle, a file path (the version you last
saw or wrote), a committed file (`MEMORY.md@week-6`), another text (`r4.2`) or
`unknown: <reason>`. Source navigation reads saved versions and relationships.
Run current public queries or recompute your calculations when needed.
Optional select/predicate fields store your declared condition without evaluating it.
Run `pf show HANDLE --full` alone, without a pipe, before registering selected fields.

Group insights show their survey's `snapshot_day`, separately from the day you
retrieve them. Only a completed `research_group` survey updates that date.

`bash`, `write_file` and `edit_file` take an optional `note`: why you ran the
command or what the change is for. PF keeps it with the outputs and files of
that call and shows it when you see them again, e.g. above MEMORY.md next week.

Run `pf` directly in Bash, for example `pf`. It is a separate command;
never use `novamind-operation pf`. `pf help` has complete examples.
Loops, pipes and redirects follow normal Bash rules.
Citing a whole captured object does not require reading all of it. PF records
what you actually read separately; selected fields must have reached you.
Compact DELTA/UNCHANGED returns name the exact baseline and a `pf show ... --full`
command. Use that command if you need the saved full output.

Replace the synthetic names below with handles from actual receipts.
Commands:

```
pf log MEMORY.md           versions of a file, an output (scripts/an_w7d.py.out) or a text (r4)
pf show MEMORY.md@v8       one saved version
pf diff MEMORY.md          what the latest version changed; or pf diff MEMORY.md@v6 MEMORY.md@v8
pf blame MEMORY.md         the version and day each line was written
pf depend r4               saved sources and relationships behind r4
pf rdepend scripts/an_w7d.py.out@v1   saved declarations that cite it
pf search ACTUAL_ID        your texts, business writes and latest outputs about that object
pf search --text "KEYWORD" literal text search across captured history
```

Example: MEMORY.md is rewritten almost every week. Before relying on something
you remember from earlier weeks, `pf diff MEMORY.md` shows what the last rewrite
removed.

''' + PF_CAPTURE_BOUNDARY

GLOB_ROW = '| `glob_files` | Find files by pattern (like `*.py`, `docs/**/*.json`) |'
WEEKLY_SCRIPTS = '3. **Weekly scripts** — registered via `./novamind-operation register-daily-script`'


def edits(pf):
    """(original, replacement) pairs; every original must occur exactly once."""
    history = PF_HISTORY if pf else GIT_HISTORY
    return [
        ('You have 6 tools:', 'You have 10 tools:'),
        (GLOB_ROW, GLOB_ROW + '\n| `text_create`, `text_revise`, `text_retire`, `text_list` | Register, revise, '
                   'retire and list texts you will rely on in later weeks (see Registered Texts) |' +
                   ('\n\n`bash` also runs `pf`, which shows the history of your files and command outputs '
                    '(see File and Output History).' if pf else '')),
        (WEEKLY_SCRIPTS, WEEKLY_SCRIPTS + '\n4. **Registered texts** — saved by `text_create` in '
                         '`documents/registrations.json`; not loaded automatically (use `text_list`)\n' +
                         ('5. **PF history** — every version of your files and command outputs, with your '
                          'notes (use `pf`)' if pf else
                          '5. **Git history** — the harness commits your working directory at the end of '
                          'every week (use `git log`)')),
        ('MEMORY.md is the ONLY way to carry knowledge forward.',
         'MEMORY.md is the only file loaded into your context automatically.'),
        ('## Weekly Workflow', REGISTERED_TEXTS + history + '## Weekly Workflow'),
        ('2. **Recall context** — read your notes/files from previous weeks',
         '2. **Recall context** — read your notes/files from previous weeks (' +
         ('`pf log` and `pf diff` show how they changed)' if pf else '`git log -p` shows how they changed)')),
        ('5. **Save what matters** — update your files with observations, decisions, learnings',
         '5. **Save what matters** — update your files with observations, decisions, learnings, and '
         'register or revise the texts you will rely on later'),
    ]


def capability_history(pf, capabilities):
    if capabilities.text_tools:
        history = PF_HISTORY if pf else GIT_HISTORY
        if not pf and (not capabilities.tool_notes or not capabilities.git_extra_saves):
            history = history.replace(GIT_CALL_NOTES, '')
        return history
    if pf:
        history = '''## File and Output History (pf)

PF saves captured workspace file versions and command outputs, including unsaved
outputs. Run `pf` directly in Bash. Use `pf help` for syntax, `pf search` to find
saved material, `pf log` for versions, and `pf show HANDLE --full` for exact content.
`pf diff` and `pf blame` inspect changes. `pf depend` and `pf rdepend` read saved
source relationships; `pf candidates` finds captured sources. Saved, delivered
and declared used are different. Query current business facts or recompute when needed.
Use actual returned handles and follow returned pagination cursors.

'''
        if capabilities.tool_notes:
            history += 'The optional note on bash, write_file and edit_file records the reason with the exact captured output and file version.\n\n'
        return history + PF_CAPTURE_BOUNDARY
    history = '''## File History (git)

Your workspace is a Git repository. The host commits initialization, handoffs,
answers and weekly boundaries. Save useful command results in files yourself.
Use ordinary files, MEMORY.md and Git for durable work. `git log`, `git show`,
`git diff` and `git blame` inspect saved history; you may make your own commits.
Use the published handoff workspace, commit and path to read another role's evidence.

'''
    if capabilities.tool_notes and capabilities.git_extra_saves:
        history += GIT_CALL_NOTES
    return history


def integrate(prompt, pf=False, capabilities=None):
    """The original system prompt with the registration group's sections in place."""
    if capabilities is not None and not capabilities.text_tools:
        return prompt.replace('You have 6 tools:', 'Use the available workspace tools:').replace(
            'MEMORY.md is the ONLY way to carry knowledge forward.',
            'MEMORY.md is the only file loaded into your context automatically.').replace(
                '## Weekly Workflow', capability_history(pf, capabilities) + '## Weekly Workflow')
    for old, new in edits(pf):
        if prompt.count(old) != 1:
            raise ValueError('Registration prompt anchor must occur exactly once: ' + old[:60])
        prompt = prompt.replace(old, new)
    if capabilities is not None:
        prompt = prompt.replace(PF_HISTORY if pf else GIT_HISTORY, capability_history(pf, capabilities))
    return prompt


# The weekly MEMORY header of the registration groups; the original group keeps
# "at the start of every day" word for word.
MEMORY_HEADER = ('\n\n## Your MEMORY.md (auto-loaded)\n\n'
                 'The following is the contents of your MEMORY.md file. '
                 'This is automatically loaded into your context at the start of every week.\n')


def git_memory_line(workspace, run=None):
    """[git: MEMORY.md last committed in week-7 (3f9a2c1) | ...], or None before any commit."""
    import re
    import subprocess
    try:
        result = (run or subprocess.run)(['git', '-C', str(workspace), 'log', '-1', '--format=%H %s', '--', 'MEMORY.md'],
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
    head = f"{handles.name(version)}, written day {group['day'] if group['day'] is not None else 'unknown'}"
    if note := handles.note(version):
        head += f', note: "{short(note[1], 200)}"'
    count = len(handles.groups[group['key']])
    return f"[pf: {head} | {plural(count, 'version')}: pf log MEMORY.md]"
