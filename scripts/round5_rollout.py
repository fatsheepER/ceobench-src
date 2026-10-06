"""Independent three-week continuations with a fresh 500-day CEO conversation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import round5, round5_short
from saas_bench.agents.bash_agent.run_test import BashAgentRunner
from saas_bench.db_protection import load_session_db
from saas_bench.model_usage import summarize_usage_log
from saas_bench.payload_tokens import load_counter
from saas_bench.registration_prompt import integrate
from saas_bench.run_state import checkpoint_directory, file_hash, write_json


class RolloutRunner(BashAgentRunner):
    def setup(self):
        super().setup()
        self.agent.total_days = 500
        self.agent.system_prompt = integrate(self.agent._default_system_prompt(), self.text_registration == 'pf')
        self.payload_token_counter = load_counter(self.provider, self.model)
        if self.payload_token_counter is None:
            raise ValueError('Both groups require the verified input tokenizer')
        self.agent.usage_recorder.token_counter = self.payload_token_counter

    def _restore_from_checkpoint(self, checkpoint):
        # The server restores the world; prior conversation and charges stay outside this episode.
        self.agent.reset()
        self._suppress_force_step_day_once = True


def public_outcomes(run, start):
    """Score only public ledger cash and forecasts submitted during this episode."""
    import shutil
    checkpoint = json.loads((run / 'checkpoint.json').read_text())
    snapshot = checkpoint_directory(run, checkpoint)
    with TemporaryDirectory(prefix='round5-rollout-score-') as temporary:
        world = Path(temporary) / 'world.nmdb'
        shutil.copyfile(snapshot / 'world.nmdb', world)
        conn = load_session_db(world)
        try:
            end = checkpoint['day']
            def cash(day):
                return conn.execute('SELECT COALESCE(SUM(amount),0) FROM ledger WHERE day<=?', (day,)).fetchone()[0]
            initial, final = cash(start), cash(end)
            predictions = []
            rows = conn.execute('SELECT submit_day,horizon_days,predicted_value,predicted_lower,predicted_upper FROM predictions WHERE metric=? AND submit_day>=? AND submit_day<? ORDER BY submit_day,horizon_days', ('cash', start, end))
            for day, horizon, point, lower, upper in rows:
                target = day + horizon
                actual = cash(target) if target <= end else None
                predictions.append(dict(submit_day=day, horizon_days=horizon, unit='USD', point=point, lower=lower, upper=upper, matured=actual is not None, actual=actual,
                    absolute_error_usd=abs(point - actual) if actual is not None else None,
                    interval_covers_actual=(lower <= actual <= upper) if actual is not None and lower is not None and upper is not None else None))
            return dict(start_day=start, end_day=end, completed_weeks=(end-start)//7, initial_cash_usd=initial, final_cash_usd=final, net_cash_usd=final-initial, predictions=predictions)
        finally:
            conn.close()


def execute(source, destination, group, identity, output, *, api_key=None):
    if (output / 'rollout-hold.json').exists():
        raise ValueError('Resolve the episode engineering hold before launching')
    destination = round5_short.fork_task(source, destination, group, identity)
    checkpoint = json.loads((destination / 'checkpoint.json').read_text())
    start = checkpoint['day']
    cutoffs = checkpoint['request_log_cutoffs']
    end = start + 21
    round5.environment()
    runner = RolloutRunner(continue_from=destination, stop_after_day=end, api_key=api_key)
    round5.environment()
    round5.install_pause(runner, output / 'rollout-hold.json')
    record = dict(status='running', start_day=start, target_day=end, goal_days=500, expected_weeks=3, group=group, identity=identity, request_log_begin_bytes=cutoffs, started_at=datetime.now(timezone.utc).isoformat())
    write_json(destination / 'rollout-result.json', record)
    started = time.monotonic()
    try:
        result = runner.run(verbose=False)
        write_json(destination / 'result.json', result)
        record.update(status=result['outcome'], result=result, reached_stop=result['outcome']=='stopped' and result['days_run']==end)
        if result['outcome'] not in ('stopped', 'bankrupt'):
            raise RuntimeError('Episode did not finish or naturally terminate')
        record['public_outcomes'] = public_outcomes(destination, start)
        if record['reached_stop'] and record['public_outcomes']['completed_weeks'] != 3:
            raise RuntimeError('Three-week endpoint was not preserved')
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record.update(elapsed_seconds=time.monotonic()-started, finished_at=datetime.now(timezone.utc).isoformat())
        record['usage'] = {}
        for role in ('agent','simulator'):
            log = destination / 'logs' / (role + '_requests.jsonl')
            if log.exists():
                suffix = destination / (role + '-episode-requests.jsonl')
                with log.open('rb') as reading, suffix.open('wb') as stream:
                    reading.seek(cutoffs.get(log.name, 0))
                    while chunk := reading.read(1024*1024):
                        stream.write(chunk)
                record['usage'][role] = summarize_usage_log(suffix)
                record['request_log_sha256_' + role] = file_hash(suffix)
        agent_log = destination / 'agent-episode-requests.jsonl'
        if agent_log.exists() and getattr(runner, 'payload_token_counter', None):
            record['input_attribution'] = round5_short.attribution(agent_log, runner.payload_token_counter)
        write_json(destination / 'rollout-result.json', record)
        runner.client.close()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--group', choices=('git','pf'), required=True)
    parser.add_argument('--identity', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    round5.verify_frozen(args.output_dir)
    frozen = json.loads((args.output_dir / 'rollout-frozen.json').read_text())
    for path, checksum in frozen['host_scripts'].items():
        if file_hash(path) != checksum:
            raise ValueError('Rollout host changed after freeze: ' + path)
    execute(args.source, args.destination, args.group, args.identity, args.output_dir)


if __name__ == '__main__':
    main()
