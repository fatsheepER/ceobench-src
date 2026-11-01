# Query business operations

Git and PF share the same world ledger. Every role can query `business_operations`, `business_effects`, and `business_ledger_metadata` with `novamind_api.query` or `./novamind-operation query`.

Each authorized public business call has a random `operation_ref`. The host binds `actor` and `initiator`. In multi-agent runs, model calls have `initiator = 'model'`; registered scripts have `initiator = 'registered_script'`, including raw HTTP and nested clients. Direct host calls have `initiator = 'host'`. Caller-supplied headers cannot change these fields. Permission and phase refusals before dispatch remain in the private host audit. Business validation rejections after dispatch have an operation and no effects.

`parameters_json` preserves the public request. `result_json` preserves the complete tool result, including every enterprise item when the batch fails or succeeds partially. For successful `next_week`, it stores the canonical business terminal result (`success`, `day`, `advance`); dashboard rendering and response delivery remain separate. Confirmed terminal results are immutable: repeating the same finalization is accepted, and a conflicting finalization fails. Business receipts add short `operation` metadata with the exact `operation_ref`, admission `operation_seq`, recorded `outcome`, and committed `effect_seqs`. The SDK includes it in returned result data and in business rejection exceptions. The CLI prints it with `next-week` results. Permission and phase refusals before dispatch have no operation reference. An unfinished receipt uses `effect_seqs: null`; inspect the public ledger after work settles. Full business results remain in the ledger.

`operation_seq` orders admission on the retained world route. `effect_seq` orders actual committed transactions. Multiple effects from one commit share `effect_seq` and have distinct zero-based `item_index` values. `effect_ref` combines a random operation reference with its local effect index. Same-value writes are new operations and new effects. Runtime configuration effects record full before/after scopes, including mapping replacement and clearing. Row effects contain only explicitly public columns. Precise hidden insight snapshots, relationship values and enterprise reply scheduling are excluded.

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
SELECT * FROM business_operations WHERE operation_ref = 'op:REPLACE_ME';
SELECT * FROM business_effects WHERE operation_ref = 'op:REPLACE_ME'
ORDER BY effect_seq, item_index;
```

For a large result, request its `result_json` or narrower columns separately and reduce the page size to fit the normal public SQL response limits. `LIMIT` is explicit; continue until a page is empty. No ledger result is broadcast automatically.

## Interpret incomplete work

`outcome` is `succeeded`, `rejected`, `partial`, `unknown`, or `unfinished`. `unfinished` means there is no confirmed final result. It does not mean zero effects, and it does not prove that a worker is still running. Inspect committed effects independently. An unexpected exception on a clean connection can record a sanitized unknown result. A dirty exception leaves the pending transaction untouched and the operation unfinished. Public SQL remains unavailable while the world is failed or has an unfinished transaction. The original attempt preserves the fault scene for host inspection.

`next_week` includes separately committed predictions and actual simulator commit progress. Each `advance_progress` effect reports the reached day and the last completed day available from the committed simulator checkpoint state. A null completed day means that evidence is unavailable. These are transaction progress summaries, not a row-by-row simulation audit. A partial day may commit changes before that day is complete. Server completion, response delivery and background resource closure remain separate facts. A late completion settles the original operation.

The ledger travels with the world database through checkpoints. Restoring a checkpoint keeps its earlier records and discards later records in the new attempt. The abandoned attempt retains its old records. New tails use new random references even if enumeration numbers repeat. `business_ledger_metadata.inception_day` identifies when the ledger was added to an older world; earlier operations are not reconstructed. Existing checkpoint recovery applies to runtime configuration. SQLite and Python memory do not have process-crash atomicity.
