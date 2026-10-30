# CEOBench CLI Reference

## novamind-operation

Entry point for novamind-operation CLI.

Commands:
    next-week   Advance the simulation by one week (7 days).
                REQUIRES 13 args: a rationale string + for each of 4 horizons
                (+7d, +28d, +84d, +182d), submit point + 95% CI low/high in USD.

All 12 cash forecast values are in USD. For 20.7 million USD, submit
20700000 or 20.7e6.

Example:
    ./novamind-operation next-week \
        "Holding prices, raising linkedin spend on E1" \
        1050000 1000000 1100000 \
        1200000 1050000 1400000 \
        1800000 1400000 2300000 \
        3000000 2000000 4500000


### Commands

#### `./novamind-operation next-week`

Advance the simulator by one week (7 days) — REQUIRES a rationale string
plus cash predictions at four horizons, each with point estimate + 95% CI bounds.

All 12 cash forecast values are in USD. For 20.7 million USD, submit
20700000 or 20.7e6.

Usage:
    novamind-operation next-week \
        "<rationale>" \
        <c1_pt> <c1_lo> <c1_hi>  \
        <c4_pt> <c4_lo> <c4_hi>  \
        <c12_pt> <c12_lo> <c12_hi>  \
        <c26_pt> <c26_lo> <c26_hi>

Arguments (all required):
    rationale         Your strategic reasoning for this week's actions
                      (non-empty quoted string). Replaces the old standalone
                      log_rationale tool. Recorded for analysis but does
                      not affect scoring.
    cash_1wk_point    Point estimate of cash in USD, +7 days.
    cash_1wk_lower    95% CI lower bound in USD, +7 days.
    cash_1wk_upper    95% CI upper bound in USD, +7 days.
    cash_4wk_point    Point estimate of cash in USD, +28 days.
    cash_4wk_lower    95% CI lower bound in USD, +28 days.
    cash_4wk_upper    95% CI upper bound in USD, +28 days.
    cash_12wk_point   Point estimate of cash in USD, +84 days.
    cash_12wk_lower   95% CI lower bound in USD, +84 days.
    cash_12wk_upper   95% CI upper bound in USD, +84 days.
    cash_26wk_point   Point estimate of cash in USD, +182 days (~6 months).
    cash_26wk_lower   95% CI lower bound in USD, +182 days.
    cash_26wk_upper   95% CI upper bound in USD, +182 days.

Constraints (per horizon): ``lower <= point <= upper``. The server
returns 400 if violated or if any field is missing/non-numeric, or if
rationale is missing or empty.

Predictions are recorded at submission time and scored on percent error
`(point - actual) / actual` once actual cash is known. CI bounds are
scored on coverage (does the actual fall inside [lower, upper]?) and
sharpness (interval width / actual). You are evaluated on prediction
accuracy + calibration at each horizon in addition to realized cash.

Calls the API server to step the simulation forward by one week.
Prints the dashboard to stdout, which includes key metrics,
this week's results, inbox notifications, and submitted cash forecasts in USD.

**NOTE:** The next_week call may take several minutes at large subscriber
counts. This is normal — just wait for the response.

Exit code 0 on success, 1 on failure (including missing rationale or predictions).

#### `./novamind-operation register-daily-script`

Register a Python source snapshot for the start of each week.

Usage: ./novamind-operation register-daily-script <script_path> [--session ID]

The legacy command name says daily; execution is weekly, in insertion order.
Registration uses the source filename. Re-register that filename to replace
its snapshot; editing the local file alone does not update it.
Scripts must explicitly import novamind_api, for example
`import novamind_api as nm`. Print the results you need in your weekly input.
In teams, scripts and outputs belong to their registering role. CEO scripts
can change business settings; analyst scripts can only read public data.
ask_analyst follow-ups do not rerun scripts.
The server receipt includes name, registered and sha256. Exit code 0 requires
server success; file, connection and server errors exit nonzero.

#### `./novamind-operation list-daily-scripts`

List the current role's server snapshots in insertion order.

Usage: ./novamind-operation list-daily-scripts [--session ID]

Returns the server's scripts list with name, size and sha256 for each entry.
These snapshots run at the start of each week. Local files are not a registry.
Exit code 0 requires server success; connection and server errors exit nonzero.

#### `./novamind-operation remove-daily-script`

Remove a registered server snapshot by filename, preserving local files.

Usage: ./novamind-operation remove-daily-script <script_name> [--session ID]

Returns the server's removed filename. In teams, only your role's registry
is affected. Exit code 0 requires server success; missing registrations,
connection failures and rejected removals exit nonzero.
