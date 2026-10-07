# NovaMind CLI Reference

## `novamind-operation`

The primary CLI for interacting with the NovaMind SaaS simulator.

### Session Management

#### `novamind-operation new-session`
Create a new simulation session.

```bash
novamind-operation new-session [--days 365] [--seed 42] [--cash 1000000]
```

**Options:**
- `--days`: Total simulation days (default: 365)
- `--seed`: Random seed for reproducibility (default: 42)
- `--cash`: Initial cash balance (default: 1,000,000)

**Returns:** JSON with `session_id`, `seed`, `total_days`, `initial_cash`, `workspace` path.

#### `novamind-operation list-sessions`
List all existing sessions.

```bash
novamind-operation list-sessions
```

#### `novamind-operation status [--session ID]`
Get the current status of a session.

```bash
novamind-operation status
novamind-operation status --session abc123def456
```

#### `novamind-operation stop [--session ID]`
Stop the simulation server for a session.

```bash
novamind-operation stop
```

---

### Simulation Control

#### `novamind-operation next-week <rationale> <c1_pt> <c1_lo> <c1_hi> <c4_pt> <c4_lo> <c4_hi> <c12_pt> <c12_lo> <c12_hi> <c26_pt> <c26_lo> <c26_hi> [--session ID]`
Advance the simulation by one week (7 days). Requires a non-empty rationale string and 12 cash forecast values in USD.

For 20.7 million USD, submit `20700000` or `20.7e6`.

```bash
novamind-operation next-week \
    "Holding prices and raising enterprise ad spend" \
    1050000 1000000 1100000 \
    1200000 1050000 1400000 \
    1800000 1400000 2300000 \
    3000000 2000000 4500000
```

**Arguments:**
- `rationale`: Strategic reasoning for this week's actions, as a non-empty quoted string.
- `c1_pt`, `c1_lo`, `c1_hi`: Cash point estimate, 95% CI lower bound, and upper bound in USD, +7 days.
- `c4_pt`, `c4_lo`, `c4_hi`: Cash point estimate, 95% CI lower bound, and upper bound in USD, +28 days.
- `c12_pt`, `c12_lo`, `c12_hi`: Cash point estimate, 95% CI lower bound, and upper bound in USD, +84 days.
- `c26_pt`, `c26_lo`, `c26_hi`: Cash point estimate, 95% CI lower bound, and upper bound in USD, +182 days.

Each horizon requires `lower <= point <= upper`. Predictions are stored in the `predictions` table at submission time and scored on point percent error `(point - actual) / actual`, CI coverage, and sharpness when actual cash is known.

**Output:** The weekly dashboard showing cash, subscribers, MRR, this week's metrics, current config, product quality, inbox notifications, and submitted cash forecasts in USD.

---

### Code Execution

#### `novamind-operation python <script.py> [--session ID]`
Execute a Python script in the simulation environment with `novamind_api` available.

```bash
novamind-operation python my_strategy.py
```

The script runs with `novamind_api` importable. Example script:
```python
import novamind_api as nm

# Set prices
nm.pricing.set_prices(A=25, B=69, C=179)

# Check current day
print(f"Day: {nm.vars.current_day}")

# Query data
result = nm.query("SELECT COUNT(*) as n FROM subscriptions WHERE status='active'")
print(f"Active subscribers: {result['rows'][0]['n']}")
```

#### `novamind-operation python-c "<code>" [--session ID]`
Execute inline Python code.

```bash
novamind-operation python-c "import novamind_api as nm; nm.pricing.set_prices(A=29.99)"
```

---

### Database Queries

#### `novamind-operation query "<SQL>" [--session ID]`
Execute a read-only SQL query against the simulation database.

```bash
novamind-operation query "SELECT * FROM subscriptions WHERE status='active' LIMIT 10"
novamind-operation query "SELECT group_id, COUNT(*) as n FROM subscriptions WHERE status='active' GROUP BY group_id"
```

**Restrictions:**
- Read-only (SELECT only) — no INSERT/UPDATE/DELETE
- Schema introspection blocked (no PRAGMA, sqlite_master)
- Some internal tables and columns are hidden
- Results capped at 5,000 rows

See `docs/tables-reference.md` for available tables and columns.

---

### History

#### `novamind-operation history [--tail N] [--session ID]`
View the action history for a session.

```bash
novamind-operation history
novamind-operation history --tail 100
```

Shows recent tool calls, queries, next-day advancements, and Python executions.

---

### Session ID

All commands accept `--session <id>` to target a specific session. If omitted, the most recently created session is used.

```bash
# These are equivalent (both use latest session):
novamind-operation next-day
novamind-operation next-day --session <latest-id>

# Target a specific session:
novamind-operation next-day --session abc123def456
```
