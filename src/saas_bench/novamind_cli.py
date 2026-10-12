"""NovaMind CLI — Command-line interface for the NovaMind SaaS simulator.

The deployed novamind-operation client owns script management.
The package's legacy novamind entry point delegates to the same handlers.
"""

import argparse
from .novamind_api._capture import observed, print

import sys


# =========================================================================
# novamind-operation CLI
# =========================================================================

@observed
def _cmd_next_week(args):
    """Advance the simulator by one week (7 days) — REQUIRES a rationale string
    plus cash predictions at four horizons, each with point estimate + 95% CI bounds.

    All 12 cash forecast values are in USD. For 20.7 million USD, submit
    20700000 or 20.7e6.

    Usage:
        novamind-operation next-week \\
            "<rationale>" \\
            <c1_pt> <c1_lo> <c1_hi>  \\
            <c4_pt> <c4_lo> <c4_hi>  \\
            <c12_pt> <c12_lo> <c12_hi>  \\
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
    """
    from .novamind_api._client import next_week
    rationale = (args.rationale or "").strip()
    if not rationale:
        print("Error: rationale is required and must be a non-empty string.", file=sys.stderr)
        sys.exit(1)
    predictions = {
        "cash_1wk":  {"point": float(args.cash_1wk_point),  "lower": float(args.cash_1wk_lower),  "upper": float(args.cash_1wk_upper)},
        "cash_4wk":  {"point": float(args.cash_4wk_point),  "lower": float(args.cash_4wk_lower),  "upper": float(args.cash_4wk_upper)},
        "cash_12wk": {"point": float(args.cash_12wk_point), "lower": float(args.cash_12wk_lower), "upper": float(args.cash_12wk_upper)},
        "cash_26wk": {"point": float(args.cash_26wk_point), "lower": float(args.cash_26wk_lower), "upper": float(args.cash_26wk_upper)},
    }
    try:
        result = next_week(predictions=predictions, rationale=rationale)
        dashboard = result.get('dashboard', '')
        print(dashboard)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def operation_main():
    """Entry point for novamind-operation CLI.

    Commands:
        next-week   Advance the simulation by one week (7 days).
                    REQUIRES 13 args: a rationale string + for each of 4 horizons
                    (+7d, +28d, +84d, +182d), submit point + 95% CI low/high in USD.

    All 12 cash forecast values are in USD. For 20.7 million USD, submit
    20700000 or 20.7e6.

    Example:
        ./novamind-operation next-week \\
            "Holding prices, raising linkedin spend on E1" \\
            1050000 1000000 1100000 \\
            1200000 1050000 1400000 \\
            1800000 1400000 2300000 \\
            3000000 2000000 4500000
    """
    parser = argparse.ArgumentParser(
        prog='novamind-operation',
        description='NovaMind simulator operation commands',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    sub_next = subparsers.add_parser(
        'next-week',
        help='Advance by 7 days. Requires a rationale string + 12 cash forecasts in USD (point + 95%% CI low/high at +7d/+28d/+84d/+182d).',
        description='All 12 cash forecast values are in USD. For 20.7 million USD, submit 20700000 or 20.7e6.',
    )
    sub_next.add_argument('rationale', type=str, help='Your strategic reasoning for this week\'s actions (non-empty)')
    sub_next.add_argument('cash_1wk_point',  type=float, help='Point estimate of cash in USD, +7 days')
    sub_next.add_argument('cash_1wk_lower',  type=float, help='95%% CI lower bound in USD, +7 days')
    sub_next.add_argument('cash_1wk_upper',  type=float, help='95%% CI upper bound in USD, +7 days')
    sub_next.add_argument('cash_4wk_point',  type=float, help='Point estimate of cash in USD, +28 days')
    sub_next.add_argument('cash_4wk_lower',  type=float, help='95%% CI lower bound in USD, +28 days')
    sub_next.add_argument('cash_4wk_upper',  type=float, help='95%% CI upper bound in USD, +28 days')
    sub_next.add_argument('cash_12wk_point', type=float, help='Point estimate of cash in USD, +84 days')
    sub_next.add_argument('cash_12wk_lower', type=float, help='95%% CI lower bound in USD, +84 days')
    sub_next.add_argument('cash_12wk_upper', type=float, help='95%% CI upper bound in USD, +84 days')
    sub_next.add_argument('cash_26wk_point', type=float, help='Point estimate of cash in USD, +182 days (~6 months)')
    sub_next.add_argument('cash_26wk_lower', type=float, help='95%% CI lower bound in USD, +182 days')
    sub_next.add_argument('cash_26wk_upper', type=float, help='95%% CI upper bound in USD, +182 days')
    sub_next.set_defaults(func=_cmd_next_week)

    args = parser.parse_args()
    args.func(args)


# =========================================================================
# novamind CLI (daily script management)
# =========================================================================

try:
    from . import _public_cli
except ImportError:  # zipapp client lives at the archive root
    import _public_cli

_cmd_register_daily_script = _public_cli.cmd_register_daily_script
_cmd_list_daily_scripts = _public_cli.cmd_list_daily_scripts
_cmd_remove_daily_script = _public_cli.cmd_remove_daily_script


def novamind_main():
    """Entry point for novamind CLI.

    Commands:
        register-daily-script   Register a script to run weekly
        list-daily-scripts      List all registered daily scripts
        remove-daily-script     Remove a registered daily script

    Examples:
        novamind register-daily-script strategy.py
        novamind list-daily-scripts
        novamind remove-daily-script strategy.py
    """
    parser = argparse.ArgumentParser(
        prog='novamind',
        description='NovaMind daily script management',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # register-daily-script
    sub_reg = subparsers.add_parser('register-daily-script', help='Register a weekly script snapshot')
    sub_reg.add_argument('script_path', help='Path to the Python script')
    sub_reg.set_defaults(func=_cmd_register_daily_script)

    # list-daily-scripts
    sub_list = subparsers.add_parser('list-daily-scripts', help='List registered daily scripts')
    sub_list.set_defaults(func=_cmd_list_daily_scripts)

    # remove-daily-script
    sub_rm = subparsers.add_parser('remove-daily-script', help='Remove a daily script')
    sub_rm.add_argument('script_name', help='Filename of the script to remove')
    sub_rm.set_defaults(func=_cmd_remove_daily_script)

    args = parser.parse_args()
    args.func(args)


# =========================================================================
# CLI documentation generator
# =========================================================================

def get_cli_docs() -> str:
    """Generate CLI documentation from docstrings.

    Renders documentation for all CLI commands programmatically
    from their docstrings.

    Returns:
        Markdown-formatted CLI reference.
    """
    lines = [
        "# CEOBench CLI Reference",
        "",
        "## novamind-operation",
        "",
        f"{operation_main.__doc__}",
        "",
        "### Commands",
        "",
    ]

    # operation commands
    op_commands = [
        ("next-week", _cmd_next_week),
    ]
    for name, func in op_commands:
        doc = func.__doc__ or "No documentation."
        lines.append(f"#### `./novamind-operation {name}`")
        lines.append("")
        lines.append(doc.strip())
        lines.append("")

    # novamind commands
    nm_commands = [
        ("register-daily-script", _cmd_register_daily_script),
        ("list-daily-scripts", _cmd_list_daily_scripts),
        ("remove-daily-script", _cmd_remove_daily_script),
    ]
    for name, func in nm_commands:
        doc = func.__doc__ or "No documentation."
        lines.append(f"#### `./novamind-operation {name}`")
        lines.append("")
        lines.append(doc.strip())
        lines.append("")

    return "\n".join(lines)
