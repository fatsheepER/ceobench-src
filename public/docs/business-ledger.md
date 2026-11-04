# Query business operations

Git and PF share the same world ledger. Every role can query `business_operations`, `business_effects`, and `business_ledger_metadata` with `novamind_api.query` or `./novamind-operation query`.

Each authorized public business call has a stable `operation_ref` such as `growth:1`, `ops_finance:1`, or `ceo:1`. Copy the returned reference; its number is assigned by the host and increases within the producing role across weeks. The host binds `actor` and `initiator`. In multi-agent runs, model calls have `initiator = 'model'`; registered scripts have `initiator = 'registered_script'`, including raw HTTP and nested clients. Direct host calls have `initiator = 'host'`. Caller-supplied headers cannot change these fields. Permission and phase refusals before dispatch remain in the private host audit. Business validation rejections after dispatch have an operation and no effects.

`parameters_json` preserves the public request. `result_json` preserves the complete tool result, including every enterprise item when the batch fails or succeeds partially. For successful `next_week`, it stores the canonical business terminal result (`success`, `day`, `advance`); dashboard rendering and response delivery remain separate. Confirmed terminal results are immutable: repeating the same finalization is accepted, and a conflicting finalization fails. Business receipts add short `operation` metadata with the exact `operation_ref`, admission `operation_seq`, recorded `outcome`, and committed `effect_seqs`. The SDK includes it in returned result data and in business rejection exceptions. The CLI prints it with `next-week` results. Permission and phase refusals before dispatch have no operation reference. An unfinished receipt uses `effect_seqs: null`; inspect the public ledger after work settles. Full business results remain in the ledger.

`operation_seq` orders admission on the retained world route. `effect_seq` orders actual committed transactions. Multiple effects from one commit share `effect_seq` and have distinct zero-based `item_index` values. `effect_ref` combines the operation reference with its local effect index, for example `growth:1/effect:0`. Same-value writes are new operations and new effects. Runtime configuration effects record full before/after scopes, including mapping replacement and clearing. Row effects contain only explicitly public columns. Precise hidden insight snapshots, relationship values and enterprise reply scheduling are excluded.

## Read fixed pages

Capture both high-water marks in one query. Substitute the returned integers for `OP_HIGH` and `EFFECT_HIGH` below.

```sql
SELECT
  (SELECT COALESCE(MAX(operation_seq), 0) FROM business_operations) AS operation_high,
  (SELECT COALESCE(MAX(effect_seq), 0) FROM business_effects) AS effect_high;
```

Read operation pages. Start `OP_LAST` at 0 and use the last returned `operation_seq` for the next page.

```sql
SELECT * FROM business_operations
WHERE operation_seq > OP_LAST AND operation_seq <= OP_HIGH
ORDER BY operation_seq LIMIT 100;
```

Read effect pages. Start `EFFECT_LAST` at 0 and `ITEM_LAST` at -1. Use the last returned pair for the next page. Keep the original high-water marks for the entire scan.

```sql
SELECT * FROM business_effects
WHERE effect_seq <= EFFECT_HIGH
  AND (effect_seq > EFFECT_LAST
       OR (effect_seq = EFFECT_LAST AND item_index > ITEM_LAST))
ORDER BY effect_seq, item_index LIMIT 100;
```

The high-water marks fix membership, not the final status of unfinished operations. An admitted operation can gain effects and a result later. Query it again by its stable reference after it settles.

```sql
SELECT * FROM business_operations WHERE operation_ref = 'growth:1';
SELECT * FROM business_effects WHERE operation_ref = 'growth:1'
ORDER BY effect_seq, item_index;
```

The example reference must be replaced with the actual receipt. Use `{operation_ref: "growth:1"}` as an `escalate` endpoint and `pf depend growth:1` to query saved sources in PF. This optional declaration workflow is available only when source declarations are enabled; Git users can save the operation reference and reasoning in ordinary files.

`escalate` is a model tool, not a business SDK method or a `pf` subcommand. It accepts
`{"items": [...]}`; each item independently creates, revises, or retires one source
declaration. Inspect every item receipt, including partial failures. For example:

```json
{"items":[{"item_id":"capacity-basis","op":"create","target":{"operation_ref":"growth:1"},"sources":[{"ref":{"cite":"growth:capacity.md@v1"}}],"assertion":"author_declared","reason":"This saved calculation informed the capacity change."}]}
```

The operation and file handles above are placeholders: copy the actual operation
reference from its business receipt and the exact file version from a PF receipt
or `pf log`. Use `assertion: "inferred"` for another role's target. `revise` and
`retire` take the returned `relation`, `expected_revision`, an `item_id` and a
`reason`; a supplied `sources` list replaces the complete list. A missing source
uses `{"ref":{"missing":{"description":"...","reason":"..."}}}`. A saved declaration
does not certify business success, completeness, or current validity. Availability
depends on the run's capabilities; third-round Git disables declarations and
third-round PF permits non-text targets and sources. The old model tool name
`relation_batch` returns a rename error when declarations are enabled; both names
return a disabled error when that capability is off. Neither old-name path executes
a declaration. Historical relation fields and revisions
retain their existing names and remain readable.


For a large result, request its `result_json` or narrower columns separately and reduce the page size to fit the normal public SQL response limits. `LIMIT` is explicit; continue until a page is empty. No ledger result is broadcast automatically.

## Interpret incomplete work

`outcome` is `succeeded`, `rejected`, `partial`, `unknown`, or `unfinished`. `unfinished` means there is no confirmed final result. It does not mean zero effects, and it does not prove that a worker is still running. Inspect committed effects independently. An unexpected exception on a clean connection can record a sanitized unknown result. A dirty exception leaves the pending transaction untouched and the operation unfinished. Public SQL remains unavailable while the world is failed or has an unfinished transaction. The original attempt preserves the fault scene for host inspection.

`next_week` includes separately committed predictions and actual simulator commit progress. Each `advance_progress` effect reports the reached day and the last completed day available from the committed simulator checkpoint state. A null completed day means that evidence is unavailable. These are transaction progress summaries, not a row-by-row simulation audit. A partial day may commit changes before that day is complete. Server completion, response delivery and background resource closure remain separate facts. A late completion settles the original operation.

The ledger travels with the world database through checkpoints. Restoring a checkpoint keeps its earlier records and discards later records in the new attempt. The abandoned attempt retains its old records. New tails use new role numbers. A number allocated after the selected checkpoint remains reserved for its original operation and is never assigned to another operation, even though that discarded operation is absent from the restored ledger. Admission and effect sequence numbers can repeat after rollback; use operation_ref as the stable identity. `business_ledger_metadata.inception_day` identifies when the ledger was added to an older world; earlier operations are not reconstructed. Existing checkpoint recovery applies to runtime configuration. SQLite and Python memory do not have process-crash atomicity.
