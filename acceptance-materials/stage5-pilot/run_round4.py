"""Round four: seed 42 prefix to day 28, then Git/PF branches to day 35 (first-week gate) and on to 497."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from saas_bench.agents.bash_agent.run_test import BashAgentRunner
from saas_bench.agents.bash_agent.tools import BashAgentToolExecutor
from saas_bench.model_usage import load_pricing
from saas_bench.run_state import verify_build, write_json

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'round4'
FORK_DAY, GATE, END = 28, 35, 497


def install_week_pause(runner, request):
    """Request the existing forced checkpoint after the next complete week."""
    record = runner._weekly_record

    def weekly(status):
        result = record(status)
        day = status['day']
        if request.exists() and 0 < day < runner.total_days and day % 7 == 0:
            runner.stop_after_day = day
            write_json(runner.workspace_dir / 'pause-receipt.json',
                       dict(day=day, request=str(request)))
        return result
    runner._weekly_record = weekly


def environment():
    os.environ['CEOBENCH_SIMULATOR_LLM_PROVIDER'] = 'deepseek'
    os.environ['CEOBENCH_SIMULATOR_LLM_MODEL'] = 'deepseek-flash'
    for name in ('BOSSBENCH_LLM_REPLAY_DB', 'ORACLE_MODE', 'CEOBENCH_DASHBOARD_URL',
                 'CEOBENCH_PUBLIC_DIR', 'CEOBENCH_TEST_PUBLIC', 'NOVAMIND_PUBLIC_DIR'):
        os.environ.pop(name, None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prefix', 'git', 'pf'))
    parser.add_argument('--seed', type=int, choices=(42,), default=42)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--continue-from', type=Path)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--pricing-file', type=Path,
                        default=Path(__file__).resolve().parent / 'round4' / 'pricing-fixed-offpeak.json')
    parser.add_argument('--stop-after-day', type=int, choices=(GATE, 112, END),
                        help='Branches only: first-week gate (35), adoption gate (112), or end (497)')
    parser.add_argument('--pause-file', type=Path, help='Finish the current week and save when this host-side file exists')
    args = parser.parse_args(argv)
    args.smoke = False
    if args.attempt < 1:
        parser.error('attempt must be positive')
    if (args.mode in ('git', 'pf')) != bool(args.continue_from):
        parser.error('Git/PF must continue a fork; prefix must start fresh')
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise RuntimeError('Start with PYTHONHASHSEED=0')
    environment()
    kind = 'long'
    suffix = '' if args.attempt == 1 else f'-attempt{args.attempt}'
    stage = '' if args.mode == 'prefix' else f'-to{args.stop_after_day}'
    output = args.output_dir.resolve()
    pointer = output / f'{kind}-{args.mode}-{args.seed}{stage}{suffix}.json'
    if pointer.exists():
        raise RuntimeError('Run already registered; inspect it before starting another')
    rates = load_pricing(args.pricing_file)['rates']
    if not {'deepseek-v4.1-flash', 'deepseek-flash'} <= rates.keys():
        parser.error('Use pricing-opencode-go.json: rates for both Go and the official simulator are required')
    start_day = 0
    if args.continue_from:
        checkpoint = json.loads((args.continue_from / 'checkpoint.json').read_text())
        saved = json.loads((args.continue_from / 'manifest.json').read_text())
        if args.stop_after_day is None:
            parser.error('Branches need --stop-after-day 35, 112, or 497')
        expected = FORK_DAY if args.stop_after_day == GATE else checkpoint['day']
        if (checkpoint['day'] != expected or checkpoint['context_boundary'] != 'new_week' or
                args.stop_after_day == END and not GATE <= expected < END):
            raise ValueError('Continue only from the expected complete week boundary')
        origin = saved.get('fork_source') or saved.get('sql_evidence') or {}
        if expected == FORK_DAY and not origin.get('source_manifest_sha256'):
            raise ValueError('Continuation requires a fork made by scripts/fork_run.py')
        start_day = checkpoint['day']
    capture = args.mode in ('prefix', 'pf')
    runner = BashAgentRunner(
        provider='opencode', model='deepseek-v4.1-flash', base_url='https://opencode.ai/zen/go/v1',
        reasoning_effort='high', seed=args.seed, total_days=500, scenario='default',
        initial_cash=1_000_000.0, workspace_base=output / kind / args.mode,
        run_kind='pilot', pricing_file=args.pricing_file, continue_from=args.continue_from,
        execution_capture=capture, sql_capture=capture, text_registration=args.mode,
        pf_stale_checks=args.mode == 'pf',
        # The configured end needs no harness stop; the run completes on its own.
        stop_after_day=FORK_DAY if args.mode == 'prefix' else None if args.stop_after_day == END else args.stop_after_day,
    )
    # The runner loads .env; clear experiment overrides again after that load.
    environment()
    BashAgentToolExecutor(runner.agent_workspace, require_sandbox=True).verify_sandbox()
    build = verify_build(ROOT / 'public', root=ROOT)
    assert runner.total_days == 497
    record = dict(group=args.mode, seed=args.seed, attempt=args.attempt, purpose='round4-long-run',
                  included_in_results=False, start_day=start_day,
                  run_id=runner.run_id, path=str(runner.workspace_dir), status='running',
                  stop_after_day=runner.stop_after_day, effective_days=runner.total_days,
                  provider='opencode', model='deepseek-v4.1-flash',
                  agent_cost_basis='opencode_go_quota_usd', simulator_cost_basis='deepseek_usage_usd',
                  reasoning_effort='high', pricing_file=str(args.pricing_file.resolve()),
                  launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  continue_from=str(args.continue_from.resolve()) if args.continue_from else None,
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  build=build, started_at=datetime.now(timezone.utc).isoformat())
    write_json(pointer, record)
    print(record['path'], flush=True)
    if args.pause_file:
        install_week_pause(runner, args.pause_file.resolve())
    try:
        result = runner.run()
        write_json(runner.workspace_dir / 'result.json', result)
        record.update(status=result['outcome'], finished_at=datetime.now(timezone.utc).isoformat(), result=result,
                      stop_after_day=runner.stop_after_day,
                      paused_on_request=bool(args.pause_file and args.pause_file.exists()),
                      reached_stop=result['outcome'] == 'stopped' and result['days_run'] == runner.stop_after_day)
        write_json(pointer, record)
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      finished_at=datetime.now(timezone.utc).isoformat())
        write_json(pointer, record)
        raise


if __name__ == '__main__':
    main()
