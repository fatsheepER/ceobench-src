#!/usr/bin/env bash
# Recover into a new attempt directory; leave the interrupted run untouched.
# Usage: scripts/resume_run.sh RUN_DIR [DESTINATION] [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/.."
args=()
for arg in "$@"; do
    # Recovery runs in the foreground; retain the old flag for existing callers.
    [[ "$arg" == --no-monitor ]] || args+=("$arg")
done
if [[ -x .venv/bin/python ]]; then
    exec .venv/bin/python scripts/recover_run.py "${args[@]}" --resume
else
    exec python3 scripts/recover_run.py "${args[@]}" --resume
fi
