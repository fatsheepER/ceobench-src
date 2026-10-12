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

GIT_HISTORY = '''Read saved handoff files with git -C <workspace> show <handoff_commit>:<path>.
Register that exact snapshot as growth:<path>@<handoff_commit>, using the actual
role, path and commit. A bare file citation binds this week's closing commit and
includes later edits in the same week. The host commits analyst workspaces after
each handoff and every workspace at week close. Use git log or git diff to review
past changes.
'''

PF_HISTORY = '''Run PF directly in Bash, for example `pf depend r1`. PF is a separate command;
never use `novamind-operation pf`.
Choose PF operations for the evidence question:
- For exact peer evidence, copy the returned role and version into pf show.
- Before reusing an old conclusion, read its saved version; use pf depend <handle> --detail
  when its dependencies need a current check. Fresh handoffs need no automatic refresh.
- To understand changed evidence, use pf log and pf diff, then decide whether
  the conclusion and its complete reference list need revision.
- When only a selected value or condition matters, use select and predicate in
  text_create instead of treating every content change as a broken condition.
show reads saved content. depend replays permitted SQL and public read-only API
calls, never analyst scripts.
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
    return prompt + HANDOFF + '\n' + (PF_HISTORY if mode == 'pf' else GIT_HISTORY)
