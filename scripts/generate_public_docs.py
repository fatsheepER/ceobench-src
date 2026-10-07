#!/usr/bin/env python3
"""Generate static documentation for the public NovaMind Bench repo.

Renders TABLE_DOCS, TOOL_DOCS, CLI docs, and simulator instructions
into the public/ directory structure.

Usage:
    cd projects/saas-bench
    uv run python scripts/generate_public_docs.py [--output public/docs]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from saas_bench.tools import TOOL_DOCS, get_tool_summary_table
from saas_bench.database import TABLE_DOCS
from saas_bench.docs_generator import (
    render_api_docs,
    render_table_docs,
    _EXCLUDED_TOOLS,
    _TOOL_TO_MODULE,
)


def render_tools_reference(output_path: Path):
    """Render TOOL_DOCS as a comprehensive markdown reference."""
    lines = [
        "# NovaMind Tools Reference",
        "",
        "Complete reference for all available tools. Call them via the Python API",
        "(`import novamind_api as nm`) — see `docs/novamind_api/` for the SDK source",
        "and run scripts with `./novamind-operation python <script.py>` or",
        "`./novamind-operation python-c \"<inline code>\"`.",
        "",
        "## Tool Summary",
        "",
        get_tool_summary_table(),
        "",
        "---",
        "",
        "## Detailed Tool Documentation",
        "",
    ]

    # Group by category
    by_category = {}
    for name, doc in TOOL_DOCS.items():
        if name in _EXCLUDED_TOOLS:
            continue
        cat = doc.get("category", "Other")
        by_category.setdefault(cat, []).append((name, doc))

    for category in sorted(by_category.keys()):
        lines.append(f"### {category}")
        lines.append("")

        for name, doc in sorted(by_category[category]):
            module = _TOOL_TO_MODULE.get(name, "other")
            lines.append(f"#### `{name}`")
            lines.append("")
            lines.append(f"**Python:** `novamind_api.{module}.{name}(...)`")
            lines.append("")
            lines.append(doc.get("description", ""))
            lines.append("")

            # Parameters
            params = doc.get("parameters", {})
            if params:
                lines.append("**Parameters:**")
                lines.append("")
                for pname, pdesc in params.items():
                    lines.append(f"- `{pname}`: {pdesc}")
                lines.append("")

            # Input schema
            schema = doc.get("inputSchema", {})
            if schema and schema.get("properties"):
                lines.append("**Input Schema:**")
                lines.append("```json")
                lines.append(json.dumps(schema, indent=2, default=str))
                lines.append("```")
                lines.append("")

            # Structured SDK result
            if schema := doc.get("output_schema"):
                lines.extend(["**Output Schema:**", "", "```json",
                              json.dumps(schema, indent=2, ensure_ascii=False), "```", ""])

            # Returns
            returns = doc.get("returns", {})
            if returns:
                lines.append("**Returns:**")
                if isinstance(returns, dict):
                    for rkey, rval in returns.items():
                        lines.append(f"- {rkey}: {rval}")
                else:
                    lines.append(f"- {returns}")
                lines.append("")

            # Impact
            impact = doc.get("impact", "")
            if impact:
                lines.append(f"**Impact:** {impact}")
                lines.append("")

            # Example
            example = doc.get("example_call", {})
            if example:
                lines.append("**Example:**")
                lines.append("```json")
                lines.append(json.dumps(example, indent=2, default=str))
                lines.append("```")
                lines.append("")

            # Sample I/O
            sample_io = doc.get("sample_io", [])
            if sample_io and isinstance(sample_io, list):
                lines.append("**Sample I/O:**")
                for sample in sample_io[:2]:  # Show at most 2 examples
                    label = sample.get("label", "Example")
                    lines.append(f"*{label}:*")
                    if "input" in sample:
                        lines.append("```json")
                        lines.append(json.dumps(sample["input"], indent=2, default=str))
                        lines.append("```")
                    if "output" in sample:
                        output = sample["output"]
                        if isinstance(output, str) and len(output) > 500:
                            output = output[:500] + "..."
                        lines.append("Output:")
                        lines.append("```")
                        lines.append(str(output))
                        lines.append("```")
                lines.append("")

            lines.append("---")
            lines.append("")

    output_path.write_text("\n".join(lines))


def render_tables_reference(output_path: Path):
    """Render TABLE_DOCS as a comprehensive markdown reference."""
    lines = [
        "# NovaMind Database Tables Reference",
        "",
        "Reference for all queryable database tables. Query via:",
        "- `novamind-operation query \"SELECT * FROM table_name LIMIT 10\"`",
        "- Python: `novamind_api.query(\"SELECT * FROM table_name LIMIT 10\")`",
        "",
        "`SELECT *` expands only the public columns below. Internal tables and columns are denied,",
        "including reads in filters, joins, ordering, aggregates and subqueries. Queries use a fresh",
        "read-only snapshot of the current world. Use single quotes for string literals.",
        "",
        "**Note:** Schema introspection queries (PRAGMA, sqlite_master) are blocked.",
        "Use this reference or `docs/tables/*.json` for schema information.",
        "",
        "---",
        "",
    ]

    for table_name, doc in sorted(TABLE_DOCS.items()):
        desc = doc.get("description", "")
        columns = doc.get("columns", {})

        lines.append(f"## `{table_name}`")
        lines.append("")
        lines.append(desc)
        lines.append("")

        if columns:
            lines.append("| Column | Description |")
            lines.append("|--------|-------------|")
            for col_name, col_desc in columns.items():
                # Escape pipe characters in descriptions
                col_desc_safe = col_desc.replace("|", "\\|")
                lines.append(f"| `{col_name}` | {col_desc_safe} |")
            lines.append("")

        lines.append("---")
        lines.append("")

    output_path.write_text("\n".join(lines))


def render_cli_reference(output_path: Path):
    """Render CLI reference documentation."""
    content = """# NovaMind CLI Reference

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
novamind-operation next-week \\
    "Holding prices and raising enterprise ad spend" \\
    1050000 1000000 1100000 \\
    1200000 1050000 1400000 \\
    1800000 1400000 2300000 \\
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
"""
    output_path.write_text(content)


def render_simulator_instructions(output_path: Path):
    """Copy and render simulator instructions with tool list filled in."""
    src = Path(__file__).parent.parent / "src" / "saas_bench" / "agents" / "simulator_instructions.md"
    content = src.read_text()

    # Fill in placeholders
    tool_list = get_tool_summary_table()
    # Calculate total_years from a default 365 days (agents will see their actual value at runtime)
    content = content.replace("{total_days}", "N")
    content = content.replace("{total_years}", "N/365")
    content = content.replace("{tool_list}", tool_list)

    output_path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description="Generate public documentation")
    parser.add_argument("--output", type=str, default="public/docs",
                        help="Output directory (default: public/docs)")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print("Generating documentation...")

    # API docs (JSON, grouped by module)
    api_dir = output / "api"
    render_api_docs(api_dir)
    print(f"  ✅ API docs → {api_dir}/ ({len(list(api_dir.glob('*.json')))} files)")

    # Table docs (JSON, one per table)
    tables_dir = output / "tables"
    render_table_docs(tables_dir)
    print(f"  ✅ Table docs → {tables_dir}/ ({len(list(tables_dir.glob('*.json')))} files)")

    # Tools reference (markdown)
    tools_ref = output / "tools-reference.md"
    render_tools_reference(tools_ref)
    print(f"  ✅ Tools reference → {tools_ref}")

    # Tables reference (markdown)
    tables_ref = output / "tables-reference.md"
    render_tables_reference(tables_ref)
    print(f"  ✅ Tables reference → {tables_ref}")

    # CLI reference (markdown)
    cli_ref = output / "cli-reference.md"
    render_cli_reference(cli_ref)
    print(f"  ✅ CLI reference → {cli_ref}")

    # Simulator instructions
    sim_instructions = output / "simulator-instructions.md"
    render_simulator_instructions(sim_instructions)
    print(f"  ✅ Simulator instructions → {sim_instructions}")

    print(f"\nDone! All docs generated in {output}/")


if __name__ == "__main__":
    main()
