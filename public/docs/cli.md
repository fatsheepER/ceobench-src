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

## novamind

Entry point for novamind CLI.

Commands:
    register-daily-script   Register a script to run daily
    list-daily-scripts      List all registered daily scripts
    remove-daily-script     Remove a registered daily script

Examples:
    novamind register-daily-script strategy.py
    novamind list-daily-scripts
    novamind remove-daily-script strategy.py


### Commands

#### `novamind register-daily-script`

Register a Python script to run automatically at the start of each day.

The script content is snapshotted at registration time. Subsequent edits
to the source file will NOT affect the registered version. To update,
re-register the script.

If a script with the same filename already exists, it is overwritten.
Scripts are executed in alphabetical order at the start of each day,
with novamind_api pre-imported.

Args:
    script_path: Path to the Python script to register.

Example:
    novamind register-daily-script my_strategy.py

#### `novamind list-daily-scripts`

List all registered daily scripts.

Shows script names and sizes. Scripts run at the start of each day
in alphabetical order.

Example:
    novamind list-daily-scripts

#### `novamind remove-daily-script`

Remove a registered daily script.

Args:
    script_name: Filename of the script to remove.

Example:
    novamind remove-daily-script my_strategy.py
