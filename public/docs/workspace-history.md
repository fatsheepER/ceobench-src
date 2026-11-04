# Saved business work\n\nUse the section for your run mode. Both modes keep ordinary files, MEMORY.md, SQL, full handoffs, direct questions and automatic observation scripts. Source declarations are optional PF tools in third-round runs.\n\n## File History (git)

Your workspace is a Git repository. The host commits initialization, handoffs,
answers and weekly boundaries. Save useful command results in files yourself;
you may make your own commits. Use `git log --oneline -- PATH` to locate versions,
`git show COMMIT:PATH` to read saved bytes, `git diff OLD NEW -- PATH` to compare,
and `git blame COMMIT -- PATH` to locate a line's introduction. Search saved reasoning
with `git log -S'KEYWORD' -p -- PATH` or ordinary file search.
PATH, COMMIT, OLD, NEW and KEYWORD are placeholders. Copy actual paths and commits
from your Git log or the published handoff. Read peer evidence with
`git -C WORKSPACE show COMMIT:PATH`, using its readable handoff workspace and commit.
Save outputs needed later before handoff. Unsaved shell output is not a saved file.
For long results, read narrower ranges while keeping the same commit and path.
Git comparisons do not refresh sources; query current facts or recompute when needed.

Save judgments, assumptions, plans and change reasons in ordinary files.
Keep useful paths in MEMORY.md; only MEMORY.md is loaded automatically each week.
Use current queries or recompute when saved evidence no longer covers the decision.

## File and Output History (pf)

Run `pf` directly in Bash, never `novamind-operation pf`. PF captures file versions
and command outputs, including unsaved output. Internal reads, intermediate writes
and pipe streams may be missing. A saved result keeps its original day and coverage.
Group insights keep their survey snapshot_day until a new survey completes.

Use these paths when saved evidence would help your business work:
- Recover a calculation, assumption or output: `pf search --text "KEYWORD"`, then
  `pf log OBJECT` and `pf show HANDLE --full`. For a known business ID use
  `pf search ACTUAL_ID`. Go straight to show when the exact version is known.
- Check what changed in a report or forecast: `pf diff OBJECT` compares the last
  two versions; `pf diff OLD_HANDLE NEW_HANDLE` compares selected versions of the
  same object. `pf blame FILE` locates the version and day that introduced each line.
- Trace a report or business operation's saved basis: `pf depend TARGET --detail`.
  After a source changes, `pf rdepend SOURCE_HANDLE --detail` finds saved declarations
  citing it. Read the affected material and current facts or recompute as needed.
  These commands navigate relationships; they do not replay queries, evaluate
  conditions, certify validity or infer business impact. They default to one hop
  and current declaration revisions across visible roles and saved days.
  `--all-revisions` includes old/withdrawn declarations; `--include-candidates`
  adds automatic observations; `--depth N` explicitly expands more hops.
- If the source location is unknown, `pf candidates` finds captures across calls:
  your initiating role, current simulated week, 20 per page. Broaden explicitly
  with `--role all` or `--history`; filter with `--kind sql` or `--text "KEYWORD"`.
  Known exact sources need no candidates query. Saved, model-delivered and
  declared-used are different; a candidate summary does not deliver the source body.
- Follow returned `pf more CURSOR` commands for continuation or complete details.
  They preserve the query's scope and cutoff. Read missing-source descriptions and
  capture gaps; missing unsaved output stays missing. `pf help` gives full syntax.

Uppercase names above are placeholders, not literal handles. Get real file/output
handles from tool receipts or `pf log report.md` after saving that file; get unknown
locations from search/candidates. Copy the exact returned role-qualified version
for cross-role evidence (for example growth:report.md@v1, also a placeholder).
A bare name reads the latest version. Operation targets use the operation_ref from
the business receipt or business_operations SQL. Source role and storage role may differ.
Compact DELTA/UNCHANGED replies name a baseline and a full-read command; use that
command when needed. Do not treat a partial read as the complete source.

Save judgments, assumptions, plans and change reasons in ordinary files.
Keep useful paths in MEMORY.md; only MEMORY.md is loaded automatically each week.
Use current queries or recompute when saved evidence no longer covers the decision.

The optional note on bash, write_file and edit_file records why you ran the
call or changed the file. It stays with that exact captured version; it is
not a dependency declaration. Save longer reasoning in ordinary files.

Workspace files are captured before and after the outer Bash call. Files created
or changed inside that call are not refreshed in PF until it ends. Finish the
writing call, then use its returned file handles in a later Bash call. To inspect
current bytes inside the same Bash call, use ordinary file commands.

First Bash call: run your script to write `reports/w4_calc.out`, then wait for the
call to finish and return its PF receipt. Second Bash call: run
`pf log reports/w4_calc.out`, or `pf show HANDLE --full` with the returned handle.
These are two separate tool calls.

## Optional source declarations

Use the model tool `escalate` to declare sources for selected reusable files,
captured outputs or business operations. It is not a Bash/PF command or SDK method.
Use known exact versions directly. There is no weekly registration requirement,
quantity target, or declaration prerequisite for handoffs or business actions.
Author declarations, others' inferences and automatic capture observations differ.

In these JSON examples, TARGET_HANDLE and SOURCE_HANDLE are placeholders for exact
saved references obtained as described above; OPERATION_REF comes from a business
receipt. The example batch declares the saved report's source and the report used
for your operation. Use author_declared only for targets produced by your role;
use inferred for another or unknown author. Submit this as an escalate tool call:

```json
{"items":[{"item_id":"report","op":"create","target":{"cite":"TARGET_HANDLE"},"sources":[{"ref":{"cite":"SOURCE_HANDLE"}}],"assertion":"author_declared","reason":"This calculation supports the report."},{"item_id":"action","op":"create","target":{"operation_ref":"OPERATION_REF"},"sources":[{"ref":{"cite":"TARGET_HANDLE"}}],"assertion":"author_declared","reason":"This report informed the action."}]}
```

Inspect every item receipt, including partial failures. Copy its relation and
relation_revision into RELATION_ID and RELATION_REVISION below. Replace the complete source set:

```json
{"items":[{"item_id":"revise-report","op":"revise","relation":"RELATION_ID","expected_revision":"RELATION_REVISION","sources":[{"ref":{"cite":"SOURCE_HANDLE"}},{"ref":{"missing":{"description":"Earlier calculation","reason":"No saved or captured version is available."}}}],"reason":"Keep the reviewed source and make the evidence gap explicit."}]}
```

Omitted revision fields retain their values; sources replaces the whole list and
[] clears it. Missing targets are invalid; use missing descriptions for unavailable
sources, never invent a handle. To withdraw, use op=retire with item_id, relation,
expected_revision and reason. Revisions preserve history. Saved does not mean
complete evidence, business success or current validity. Public declarations and
their revisions are readable in documents/relations.json.

