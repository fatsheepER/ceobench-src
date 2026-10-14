"""Role instructions and weekly tasks for the Git and PF teams."""

ROLE_BRIEFS = {
    'growth': '''Investigate acquisition, conversion, retention and revenue by customer group and channel.
Evaluate prices, promotions, advertising and pending enterprise deals. Recommend
concrete actions with resource needs, expected effects and assumptions. Flag costs
or capacity needs for the CEO to coordinate with OpsFinance. Recommend paid
research when needed; only the CEO can purchase it.''',
    'ops_finance': '''Investigate cash, actual cash flows, compute costs, capacity, service quality,
operations and development spending. Provide a financial baseline under current
settings, constraints and risks. Keep MRR separate from actual cash receipts.
Forecast cash in USD at +7, +28, +84 and +182 days, each with a point estimate and
95% lower and upper bounds, 12 numbers in horizon order with lower <= point <= upper.
State assumptions and revise the baseline when the CEO supplies candidate actions.
Your initial report cannot incorporate Growth's concurrent, unseen recommendations.''',
}

WEEKLY_TASKS = {
    'growth': 'Assess this week\'s commercial changes by group and channel, including pending enterprise deals. '
              'Give the CEO actionable growth recommendations, their resource needs and supporting evidence. '
              'Keep unchanged findings brief.',
    'ops_finance': 'Assess this week\'s cash flows, costs, capacity and quality. Give the CEO constraints, risks '
                   'and a baseline under current settings, with 12 USD cash forecast numbers for +7, +28, +84 '
                   'and +182 days. Keep unchanged findings brief.',
}

HANDOFF = '''For conclusions that affect a decision, distinguish observed facts from estimates.
Include data coverage and units, assumptions, specific recommendations and their
conditions, locatable supporting evidence, and missing information. Use concise
free prose. A brief update is enough when nothing material changes.
Save reusable scripts, results and MEMORY.md. Register only judgments you expect
to rely on later, not every temporary number. Share registered texts as exact
role-qualified versions such as growth:r16.1. Bare r16 is local; growth:r16 is
the latest revision. Keep local IDs for text_revise and text_retire.
When revising, omitted references keep the old list; supplied references replace
the whole list, and [] clears it. After changing a conclusion, decide which exact
evidence versions it still needs and supply that complete list if it changes.
Analyst handoffs include a host header with author, day, world state, request ID,
workspace and snapshot commit. The host creates that commit after the analyst's
final answer. Report relative file paths; do not guess the commit.
Analyst follow-ups address new questions, needed evidence or corrections without
repeating the whole report. They continue this week's conversation without a
fresh dashboard or script run. Query current settings when changes since the
initial report affect the answer.
'''

PF_CEO_HISTORY = '''## PF evidence review

Use the handoffs first. Use PF when a missing fact, conflicting recommendation, source change, or reused decision needs evidence. Run `pf` directly in Bash, never `novamind-operation pf`. PF captures files and command outputs, even unsaved output.
All handles and cursors below illustrate formats only. Replace them with actual role-qualified receipts.

- show: To inspect evidence behind advice, use `pf show growth:r4.2 --full` after that text exists, or show its cited file/output handle. This reads saved content.
- log: To locate an earlier assumption or receipt, use `pf log MEMORY.md` once MEMORY has been captured; for a peer, use its returned role-qualified name. If you already have the exact version, go straight to show.
- diff: To review a changed plan or reconcile revisions, use `pf diff MEMORY.md` after two content versions exist, or `pf diff growth:notes.md@v1 growth:notes.md@v2` using versions returned by log.
- rdepend: Before changing a budget, price, or source assumption, use `pf rdepend query1@v1 --detail` on an existing cited source to find registered judgments to review. It follows declared references, not every affected business decision.
- depend: Before reusing a stale or pending judgment, use `pf depend growth:r4.2 --detail` after reading that existing revision. Read the checked day, source window, expanded scope, and each result. It replays permitted SQL and public read-only APIs, never analyst scripts. Fixed-date queries keep their dates; saved calculations need an explicit recomputation.
- search: If you cannot locate a past rationale, try `pf search --text "KEYWORD"`. For a known business object, use `pf search ACTUAL_ID`, replacing ACTUAL_ID with its real ID. Open the relevant returned versions.
- blame: If a disputed assumption appears in a saved file, use `pf blame MEMORY.md` once captured to locate its introducing version and day, then inspect that evidence.
- more: If a needed result continues, use `pf more c2` with its returned cursor. This continues the same snapshot.
- help: At startup or when syntax is unclear, run `pf help` for examples.

After a material change, retain, revise, or retire your judgment and pass corrections to the responsible analyst. `--history` on depend inspects lineage without a current check. Zero checked sources, cannot_check, or an unrecomputed artifact leave the relevant issue open. A passing selected condition covers that condition only.
Group insights retain their survey snapshot_day until a new survey completes; retrieving or checking them does not refresh the survey.
When registering selected fields, run `pf show HANDLE --full` alone, without a pipe.
You can also register selected conditions for reusable decisions. Weekly checks test them within their applies window and flag failures or unavailable evidence.
'''

PF_ANALYST_HISTORY = '''## PF evidence work

Use PF to recover evidence, check reused assumptions, and support a handoff; run new public queries or your own read-only calculations when current evidence is missing. Run `pf` directly in Bash, never `novamind-operation pf`. PF captures files and command outputs, even unsaved output.
All handles and cursors below illustrate formats only. Replace them with actual role-qualified receipts.

- show: To reuse a saved result or inspect a CEO citation, use `pf show query1@v1 --full` after that output exists, substituting its returned handle. A saved result retains its original date and coverage.
- depend: Before carrying a pending or old judgment into a new report, use `pf depend r4.2 --detail` after that revision exists. Read the checked day, source window, expanded scope, and each result. It replays permitted SQL and public read-only APIs, never analyst scripts. Fixed-date queries keep their dates. Rerun your own calculations when their inputs changed.
- search: To recover a prior analysis without its filename, try `pf search --text "KEYWORD"`. For a known customer group, deal, or project, use `pf search ACTUAL_ID` with its real ID, then show relevant returned versions.
- blame: To trace a carried-forward assumption or a changed calculation, use `pf blame MEMORY.md` once captured, then inspect its introducing version and day.
- log: To find a missing output version, use `pf log notes.md` after that file has been captured, or log the output name from its receipt. Use an already supplied exact handle directly.
- diff: To see whether new evidence changes the finding, use `pf diff notes.md` after two content versions exist, or `pf diff notes.md@v1 notes.md@v2` using actual versions returned by log.
- rdepend: Before replacing an analysis source, use `pf rdepend query1@v1 --detail` on that existing source to find registered judgments that need review. It follows declared references, not every affected conclusion.
- more: If a needed result continues, use `pf more c2` with its returned cursor. This continues the same snapshot.
- help: At startup or when syntax is unclear, run `pf help` for examples.

Use `--history` on depend for lineage without a current check. Zero checked sources, cannot_check, or an unrecomputed artifact leave the relevant issue open. Register reusable judgments with exact evidence and intended applies days. Use select/predicate when a specific observed value or condition matters; a passing condition covers only that scope. Revise or retire outdated judgments and hand the CEO the exact resulting version.
When registering selected fields, run `pf show HANDLE --full` alone, without a pipe.
Register a condition when a reusable judgment depends on a specific observed value or threshold. Within its applies window, weekly checks keep testing that condition and automatically flag failures or unavailable evidence, with the checked scope and day.
'''

GIT_CEO_HISTORY = '''## Git evidence review

Use the handoffs first. Use Git when a missing fact, conflicting recommendation, source change, or reused decision needs evidence. Run Git in Bash. Use the actual readable workspace from the handoff for `git -C`.
Below, WORKSPACE, COMMIT, OLD, NEW, and PATH illustrate formats only; replace them with returned workspaces, commits, and saved paths. All commands stay within the readable workspaces listed above.

- show: To inspect evidence behind advice, use `git -C WORKSPACE show COMMIT:PATH` after the host has published that handoff snapshot. Use text_list for your active latest texts; read a readable role's registrations.json for an exact older or peer revision.
- log: To locate an earlier assumption or receipt, use `git log --oneline -- MEMORY.md` once MEMORY has been committed. For a peer, use `git -C WORKSPACE log --oneline -- PATH`. Use an already supplied exact snapshot directly.
- diff: To review a changed plan or reconcile revisions, use `git -C WORKSPACE diff OLD NEW -- PATH` after both commits exist. `git diff -- MEMORY.md` instead compares current edits with the index.
- Find affected judgments: Before changing a budget, price, or source assumption, inspect references in accessible registrations.json files. Use `grep -nF -- 'ACTUAL_SOURCE' registrations.json` to locate candidates, then read their complete references. Replace ACTUAL_SOURCE with the cited path or text ID. Text matches do not enumerate a dependency graph.
- Check dependencies: Before reusing a stale judgment, read its cited snapshots and inspect current sources. Rerun the relevant public read-only query or ask its author to recompute a saved calculation. Git comparison does not execute queries. Keep fixed-date windows explicit and inspect each result.
- search: If you cannot locate a past rationale, use `git log -S'KEYWORD' -p -- MEMORY.md` once history exists. Use `grep -nF -- 'ACTUAL_ID' MEMORY.md` for current notes, replacing ACTUAL_ID with the real business ID. Search the relevant saved files, then open the matching snapshots.
- blame: If a disputed assumption appears in a committed file, use `git blame -- MEMORY.md`, or `git -C WORKSPACE blame COMMIT -- PATH`, then inspect the introducing commit.
- Continue reading: If a needed result was truncated, request the missing read_file range or read a narrower git show range with `sed -n '1,120p'`. Preserve the commit and path while paging.
- help: At startup or when syntax is unclear, run `git help -a` or `git log -h`.

After a material change, retain, revise, or retire your judgment and pass corrections to the responsible analyst. Cite handoff files as ROLE:PATH@COMMIT using actual values; a bare path binds the weekly closing commit and includes later edits that week. Save useful outputs to files before handoff so they enter the snapshot. A changed file, an unchanged saved calculation, or a weekly notice requires you to assess whether the judgment still holds.
Group insights retain their survey snapshot_day until a new survey completes; retrieving or checking them does not refresh the survey.
'''

GIT_ANALYST_HISTORY = '''## Git evidence work

Use Git to recover evidence, check reused assumptions, and support a handoff; run new public queries or your own read-only calculations when current evidence is missing. Run Git in Bash. Use the actual readable workspace from a CEO handoff for `git -C`.
Below, WORKSPACE, COMMIT, OLD, NEW, and PATH illustrate formats only; replace them with returned workspaces, commits, and saved paths. Read only the workspaces listed above.

- show: To reuse a saved result or inspect a CEO citation, use `git show COMMIT:PATH` for your committed evidence, or `git -C WORKSPACE show COMMIT:PATH` for a published readable snapshot. Saved output retains its original date and coverage.
- Check dependencies: Before carrying an old judgment into a new report, use text_list for your active latest text, or read a readable role's registrations.json for an exact older or CEO revision, inspect their committed versions, and rerun the relevant public read-only queries or your calculations. Git does not refresh sources. Keep fixed-date windows explicit and inspect each result.
- search: To recover prior analysis without its filename, use `git log -S'KEYWORD' -p -- MEMORY.md` once history exists. For a real group, deal, or project ID, use `grep -nF -- 'ACTUAL_ID' MEMORY.md` and search the relevant saved files, then open matching snapshots.
- blame: To trace a carried-forward assumption or changed calculation, use `git blame -- MEMORY.md` once committed, then inspect its introducing commit and saved evidence.
- log: To find a missing output version, use `git log --oneline -- PATH` once the output file has been committed. Use an already supplied exact snapshot directly. Shell output is available later only if saved to a file.
- diff: To see whether new evidence changes the finding, use `git diff OLD NEW -- PATH` after both commits exist. `git diff -- PATH` instead compares current edits with the index.
- Find affected judgments: Before replacing an analysis source, inspect references in your accessible registrations.json files. Use `grep -nF -- 'ACTUAL_SOURCE' registrations.json` to locate candidates, then read their complete references. Replace ACTUAL_SOURCE with the cited path or text ID. Text matches do not enumerate a dependency graph.
- Continue reading: If a needed result was truncated, request the missing read_file range or read a narrower git show range with `sed -n '1,120p'`. Preserve the commit and path while paging.
- help: At startup or when syntax is unclear, run `git help -a` or `git log -h`.

Register reusable judgments with exact evidence and intended applies days. Git weekly checks compare whole cited files and texts; select/predicate are stored without condition evaluation. Recompute values and assess conditions yourself. Revise or retire outdated judgments and supply the complete intended reference list when it changes. Save reusable scripts and outputs, give relative paths, and let the host publish the handoff commit. Share the exact role-qualified text revision or ROLE:PATH@COMMIT from the receipt.
'''

CEO_DECISIONS = '''## Team decisions

Start each week with the Growth and OpsFinance handoffs. Use their existing
evidence where it supports the decision. For a specific gap, open the evidence,
ask the responsible analyst, or verify it yourself. Avoid repeating a full
investigation already covered by the handoffs. Independent public queries,
scripts and the history tools for your mode remain available for uncovered issues,
material conflicts, necessary calculations and checks around execution.
Coordinate recommendations and resources, choose and execute business actions,
and submit the final cash forecasts before advancing. In your rationale and
MEMORY.md, briefly explain material advice you adopted, changed or rejected and
why, including important disagreements and evidence. Register decisions only
when you expect to rely on them later.
Use ask_analyst(role, message) for focused questions. Include relevant references,
candidate actions and settings already changed this week. The analysts cannot
read each other's workspaces; pass relevant findings when coordinating them.
Registered scripts and their printed output belong to their registering role.
Your scripts can change the business; analyst scripts have read-only access.
ask_analyst follow-ups do not rerun registered scripts.

'''

ANALYST_CONTEXT = '''The business sells individual and enterprise subscriptions. Payments arrive on
30-day billing cycles. Prices, promotions, model tiers and service quality affect
conversion and retention. Leads rejected on arrival are lost. Enterprise threads
can expire within a week. Advertising is allocated by channel and customer group;
operations, development, compute and capacity also consume cash. Quality, outages,
research delays and competitors affect future demand and costs. Use the public
API and table docs for mechanics and current values.

Your initialized session advances in seven-day steps, controlled by the CEO.
Each week resets your conversation and loads your own MEMORY.md, the dashboard,
registered-text checks and private registered-script outputs. Files persist.
The dashboard describes the last advance; settings may have changed since then.
Group insights retain their survey snapshot_day until a new survey completes.
Use ./novamind-operation python or python-c with novamind_api for public read-only
queries and APIs. Read docs/api, docs/tables and docs/cli.md for details.
SQL results are limited to 5,000 rows. Check truncated and warning before using
returned rows to describe the full population. Use SQL aggregates for population
totals. An explicit LIMIT narrows the requested result, even without a truncation
warning. row_count counts returned rows, not all matching records.
Write only in your workspace. You may register read-only scripts for the start of
each week. Manage your registered snapshots with:

- ./novamind-operation register-daily-script <path>
- ./novamind-operation list-daily-scripts
- ./novamind-operation remove-daily-script <filename>

The legacy daily names mean weekly execution. Registration snapshots the source;
re-register after editing.
Scripts run in insertion order, must import novamind_api explicitly, and deliver
printed output only to their owner. ask_analyst follow-ups do not rerun them.
Only the CEO can change the business, buy research or advance time.
Return your advice in final text when finished.

Use text_create, text_revise, text_retire and text_list for lasting judgments.
Records persist but are not auto-loaded; keep useful references in MEMORY.md.
Cite supporting files, exact text versions or, in PF, returned output handles.
If evidence is missing, cite unknown: <reason>. State applicability in simulated
days such as 49, 49-55 or 49-. Registration is optional.
'''


def system_prompt(base, role, mode, total_days, effective_end, workspace, readable):
    if role == 'ceo':
        from . import registration_prompt
        old_history = registration_prompt.PF_HISTORY if mode == 'pf' else registration_prompt.GIT_HISTORY
        history = PF_CEO_HISTORY if mode == 'pf' else GIT_CEO_HISTORY
        if base.count(old_history) != 1:
            raise ValueError('Team history anchor must occur exactly once')
        base = base.replace(old_history, history)
        start, end = '## Weekly Workflow\n', '**CRITICAL:** `next-week` now requires'
        if base.count(start) != 1 or base.count(end) != 1:
            raise ValueError('Team workflow anchors must occur exactly once')
        before, workflow = base.split(start)
        _, forecasts = workflow.split(end)
        prompt = (before + CEO_DECISIONS + end + forecasts).replace(
            'You have 10 tools:', 'Use these workspace tools and ask_analyst:')
    else:
        prompt = (f'You are the {role} analyst for the NovaMind SaaS business.\n'
                  + ROLE_BRIEFS[role] + '\n\n' + ANALYST_CONTEXT)
    prompt += (f'\nThe shared objective is to maximize final cash over about {total_days} days. '
               f'The effective simulation endpoint is D{effective_end}.\n'
               f'Your fixed role is {role}. Your workspace is {workspace}.\n'
               'Readable workspaces: ' + ', '.join(map(str, readable)) + '\n\n')
    if role == 'ceo':
        prompt += 'You alone can change the business, purchase research and advance time.\n'
    history = '' if role == 'ceo' else (PF_ANALYST_HISTORY if mode == 'pf' else GIT_ANALYST_HISTORY)
    return prompt + HANDOFF + '\n' + history
