"""Fixed-state preparation tasks using the real runner, sandbox and delivery chain."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import signal
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from saas_bench.agents.bash_agent.run_test import BashAgentRunner, _joined
from saas_bench.model_usage import ModelUsage
from saas_bench.payload_tokens import load_counter
from saas_bench.run_state import checkpoint_directory, clone_sql_run, tree_hash, write_json


QUESTION = """Prepare the company's decisions for the next week of its existing 497-day mission.
Do not execute business changes, buy surveys or advance time. You may inspect available
materials, analyze, write scripts and register texts. Finish with submit_decision.
Submit proposed actions with reasons; all four cash forecasts with point/lower/upper in
USD; key facts with dated, versioned evidence; and old judgments to keep, revise or mark
unknown. Include current_cash_usd and last_week_net_cash_usd among the facts, plus facts
relevant to your decisions. Review prior judgments actually present in your materials,
including evidence against them. Explain concrete missing information rather than inventing
it. New research may be proposed with decisions that depend on it explicitly marked.
For evidence use a file path and observed Git commit or 'working', or an exact PF handle;
quote the supporting passage. The initial dashboard can be cited as source
initial_dashboard, version D followed by its day number. Save ephemeral query results
if needed for a file citation.
Forecasts are decision estimates, not known future outcomes. Multiple strategies are valid.
All twelve forecast numbers are dollars: 20.7 million USD is 20700000 or 20.7e6.
"""
MAX_CALLS = 80
MAX_SECONDS = 3600
REVIEW_RULES = {
    'facts': 'Check numerical facts against public state, not hidden simulator parameters.',
    'evidence_support': 'Resolved citations alone earn no semantic credit. Compare the quoted source with the claim and inference.',
    'time_version': 'Distinguish measurement day, acquisition day and valid historical use; verify quoted versions.',
    'missing_material': 'Credit specific justified unknowns; penalize invented facts and unsupported certainty.',
    'judgments': 'Assess necessary revisions, retention of correct judgments, contrary evidence and scope. Do not reward change alone.',
    'strategy': 'Accept multiple reasoned strategies. Forecasts are checked for USD and coherent bounds; future accuracy is unobserved.',
    'scale': 'Each semantic dimension is 0 (incorrect/absent), 1 (partial), or 2 (correct). Review all assigned tasks, including failures, blind after the batch.',
}


def freeze_task(source, destination):
    """Public facts and scoring rules only, stored outside the model workspace."""
    from saas_bench.db_protection import load_session_db
    cp = json.loads((source / 'checkpoint.json').read_text())
    snapshot = checkpoint_directory(source, cp)
    with TemporaryDirectory(prefix='round5-public-answer-') as temporary:
        world = Path(temporary) / 'world.nmdb'
        shutil.copyfile(snapshot / 'world.nmdb', world)
        conn = load_session_db(world)
        try:
            cash = conn.execute('SELECT COALESCE(SUM(amount),0) FROM ledger').fetchone()[0]
            net = conn.execute('SELECT COALESCE(SUM(amount),0) FROM ledger WHERE day>? AND day<=?',
                               (cp['day'] - 7, cp['day'])).fetchone()[0]
        finally:
            conn.close()
    frozen = dict(question=QUESTION, snapshot_day=cp['day'], snapshot_sha256=tree_hash(snapshot),
                  public_facts={'current_cash_usd': cash, 'last_week_net_cash_usd': net},
                  sufficient_evidence=['Dated dashboard and actual public SQL results for this state, including saved outputs',
                      'Original versioned receipts, files and statements for historical claims; evaluate all available combinations'],
                  review_rules=REVIEW_RULES, max_calls=MAX_CALLS, max_seconds=MAX_SECONDS,
                  submission_schema=Submission.model_json_schema())
    if destination.exists():
        raise ValueError('Task specification is already frozen')
    write_json(destination, frozen)
    return frozen


def resolve_citation(runner, citation):
    """Resolve bytes, not truth. Semantic support is a separate review dimension."""
    source, version = citation['source'], citation['version']
    if source == 'initial_dashboard':
        if version != 'D' + str(runner.task_day):
            raise ValueError('Dashboard version differs from the task state')
        return runner.initial_dashboard
    if re.fullmatch(r'r[1-9][0-9]*(\.[1-9][0-9]*)?', source):
        record_id, _, revision = source.partition('.')
        records = runner.tool_executor.text_registry._load()['records'].get(record_id, [])
        number = revision or (version.split('.')[-1] if version != source else str(len(records)))
        record = next((r for r in records if str(r['revision']) == number), None)
        if record is None:
            raise ValueError('Unknown registered text version')
        return json.dumps(record, ensure_ascii=False)
    if runner.evidence_store and ('@v' in source or source.startswith('r')):
        from saas_bench.evidence_handles import index
        entries = index(runner.evidence_store).lookup(source)
        if not entries:
            raise ValueError('Unknown evidence handle')
        if version not in (source, source.rsplit('@', 1)[-1]):
            raise ValueError('Evidence handle and version disagree')
        contents = [runner.evidence_store.get_content(entry)[1] for entry in entries]
        return '\n'.join(c.decode(errors='replace') if isinstance(c, bytes) else str(c) for c in contents)
    path = Path(source)
    if path.is_absolute() or '..' in path.parts or path.parts[:1] in (('.git',), ('sessions',)):
        raise ValueError('Citation must name an available workspace file')
    if version == 'working':
        target = (runner.agent_workspace / path).resolve()
        if not target.is_relative_to(runner.agent_workspace.resolve()):
            raise ValueError('Citation escapes workspace')
        if not target.is_file():
            raise ValueError('Citation must be a regular file')
        return target.read_text()
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', version + '^{commit}'],
                                     cwd=runner.agent_workspace, text=True, stderr=subprocess.DEVNULL).strip()
    return subprocess.check_output(['git', 'show', commit + ':' + source], cwd=runner.agent_workspace,
                                   text=True, stderr=subprocess.DEVNULL)


def score_submission(runner, specification):
    answer = runner.submission
    if answer is None:
        return dict(submitted=False, facts={}, citations=[], semantic_status='pending_blind_review',
                    review_rules=specification['review_rules'])
    checks = {}
    for name, expected in specification['public_facts'].items():
        found = [f for f in answer['facts'] if f['name'] == name]
        checks[name] = (len(found) == 1 and type(found[0]['value']) in (int, float) and
                        abs(found[0]['value'] - expected) <= .01 and found[0]['unit'] == 'USD' and
                        found[0]['as_of_day'] == specification['snapshot_day'])
    citations = []
    for kind in ('facts', 'judgments'):
        for position, item in enumerate(answer[kind]):
            for citation in item['evidence']:
                try:
                    content = resolve_citation(runner, citation)
                    citations.append(dict(kind=kind, position=position, resolved=True,
                        quote_matches=citation['quote'] in content, content=content, citation=citation))
                except (ValueError, KeyError, OSError, UnicodeError, subprocess.CalledProcessError) as exc:
                    citations.append(dict(kind=kind, position=position, resolved=False,
                                          error_type=type(exc).__name__, citation=citation))
    return dict(submitted=True, facts=checks, citations=citations,
                semantic_status='pending_blind_review', review_rules=specification['review_rules'])


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Citation(StrictModel):
    source: str = Field(min_length=1)
    version: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class Forecast(StrictModel):
    point: FiniteFloat
    lower: FiniteFloat
    upper: FiniteFloat

    @model_validator(mode='after')
    def bounds(self):
        if not self.lower <= self.point <= self.upper:
            raise ValueError('Require lower <= point <= upper')
        return self


class Forecasts(StrictModel):
    cash_1wk: Forecast
    cash_4wk: Forecast
    cash_12wk: Forecast
    cash_26wk: Forecast


class Fact(StrictModel):
    name: str = Field(min_length=1)
    value: FiniteFloat | str | bool | None
    unit: str
    as_of_day: int = Field(ge=0)
    claim: str = Field(min_length=1)
    evidence: list[Citation]


class Judgment(StrictModel):
    prior: str = Field(min_length=1)
    decision: Literal['keep', 'revise', 'unknown']
    conclusion: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence: list[Citation]


class Submission(StrictModel):
    snapshot_day: int = Field(ge=0)
    proposed_actions: list[str] = Field(min_length=1)
    cash_forecasts: Forecasts
    facts: list[Fact] = Field(min_length=1)
    judgments: list[Judgment]
    missing_information: list[str]


class TaskLimit(BaseException):
    pass


def attribution(path, counter):
    """Text-only local tokens; API totals remain the primary billed measure."""
    seen, totals = set(), defaultdict(lambda: dict(first=0, repeated=0, appearances=0))
    for line in path.open():
        row = json.loads(line)
        if row['event'] != 'request':
            continue
        for message in row['request'].get('messages', []):
            for field in ('content', 'reasoning_content', 'tool_calls'):
                value = message.get(field)
                if not value:
                    continue
                text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                kind = message['role'] + ':' + field
                key = (kind, hashlib.sha256(text.encode()).hexdigest())
                totals[kind]['repeated' if key in seen else 'first'] += counter.count(text)
                totals[kind]['appearances'] += 1
                seen.add(key)
    return dict(method='Exact repeated text fields per request; excludes API template overhead. Not a causal decomposition.',
                tokenizer=counter.metadata, text_tokens=dict(totals))


class PreparationRunner(BashAgentRunner):
    def _server_environment(self):
        return dict(super()._server_environment(), CEOBENCH_READ_ONLY_TASK='1')

    def prepare(self):
        self.setup()
        status = self._get_game_status()
        self.task_day = status['day']
        self.submission = None
        counter = load_counter(self.provider, self.model)
        if counter is None:
            raise ValueError('Preparation requires a verified tokenizer for both groups')
        self.task_log = self.logs_dir / 'short_agent_requests.jsonl'
        if self.task_log.exists():
            raise ValueError('This task has already been attempted')
        self.agent.usage_recorder = ModelUsage(self.task_log, 'agent', self._pricing, self.evidence_store, counter)
        self.agent.total_turns = 0
        self.agent.system_prompt += '\n\n## Decision preparation overrides the weekly workflow\n' + QUESTION
        self.agent.no_tool_feedback = 'Use tools to investigate, or finish with submit_decision. Do not advance time.'
        self.agent.tool_descriptions.append(dict(name='submit_decision', description='Submit the final decision preparation and end this task.',
                                                parameters=Submission.model_json_schema()))
        self.tool_executor.submit_handler = self.submit
        original_request = self.agent._request_model
        self.started = time.monotonic()

        def limited_request(*args, **kwargs):
            if (self.agent.usage_recorder.summary['calls'] >= MAX_CALLS or
                    time.monotonic() - self.started >= MAX_SECONDS):
                raise TaskLimit('Model call or elapsed time limit reached')
            return original_request(*args, **kwargs)
        self.agent._request_model = limited_request
        dashboard = self._get_dashboard()
        self.initial_dashboard = str(dashboard)
        check = self.tool_executor.weekly_check(self.task_day)
        self.observation = _joined(dashboard, '\n\n', check) if check else dashboard
        self.observation = _joined(self.observation, '\n\n', QUESTION)
        return status

    def submit(self, arguments):
        submission = Submission.model_validate(arguments)
        if submission.snapshot_day != self.task_day:
            raise ValueError('snapshot_day differs from this fixed state')
        self.submission = submission.model_dump(mode='json')
        write_json(self.workspace_dir / 'submission.json', self.submission)
        return 'Decision preparation submitted. The task is complete; no business action was executed.'

    def run_preparation(self, specification=None):
        record = dict(status='failed', started_at=datetime.now(timezone.utc).isoformat(),
                      included_in_formal_tasks=self.run_kind == 'formal', group=self.text_registration,
                      source=str(self.continue_from), max_calls=MAX_CALLS, max_seconds=MAX_SECONDS)
        try:
            status = self.prepare()
            info = dict(day=self.task_day, cash=status['cash'])
            while self.submission is None:
                if time.monotonic() - self.started >= MAX_SECONDS:
                    raise TaskLimit('Elapsed time limit reached')
                action = self.agent.act(self.observation, 0, False, info)
                if action is None:
                    raise RuntimeError('Agent returned no action')
                while action is not None:
                    call_id = self.agent._pending_tool_calls[0]['id']
                    self.observation = (self._execute_tool(action.tool, action.arguments or {}) if self.submission is None
                                        else 'Task already submitted; this remaining call was not executed.')
                    self.agent.record_tool_result(self.observation, call_id)
                    self._log_tool_result(self.agent.total_turns, self.task_day, action.tool, action.arguments,
                                          self.observation, call_id=call_id)
                    self._check_capture_health()
                    action = self.agent.next_tool_action()
            if self._get_game_status()['day'] != self.task_day:
                raise RuntimeError('Fixed-state task advanced time')
            record['status'] = 'submitted'
            if specification:
                write_json(self.workspace_dir / 'automatic-score.json', score_submission(self, specification))
        except TaskLimit as exc:
            record.update(status='limit', error=str(exc))
        except BaseException as exc:
            record.update(error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            record['finished_at'] = datetime.now(timezone.utc).isoformat()
            if self.agent:
                record['usage'] = self.agent.usage_recorder.summary
            if getattr(self, 'started', None):
                record['elapsed_seconds'] = time.monotonic() - self.started
            if getattr(self, 'task_log', None) and self.task_log.exists():
                record['input_attribution'] = attribution(self.task_log, load_counter(self.provider, self.model))
            write_json(self.workspace_dir / 'task-result.json', record)
            if not getattr(self.tool_executor, 'preserved_process', None):
                self._stop_server()
        return record


def fork_task(source, destination, group, identity):
    cp = json.loads((source / 'checkpoint.json').read_text())
    seal = json.loads((source / 'seal.json').read_text())
    if cp['day'] not in (28, 84) or tree_hash(checkpoint_directory(source, cp)) != seal['snapshot_sha256']:
        raise ValueError('Preparation must start from an intact sealed D28 or D84 state')
    result = clone_sql_run(source, destination, identity, text_registration=group)
    for path in [result, *result.rglob('*')]:
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | 0o200)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--group', choices=('git', 'pf'), required=True)
    parser.add_argument('--identity', required=True)
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    from round5 import environment, verify_frozen
    environment()
    verify_frozen(args.output_dir.resolve())
    specification = json.loads(args.specification.read_text())
    cp = json.loads((args.source / 'checkpoint.json').read_text())
    if specification['snapshot_sha256'] != tree_hash(checkpoint_directory(args.source, cp)):
        raise ValueError('Task specification differs from sealed state')
    folder = fork_task(args.source.resolve(), args.destination.resolve(), args.group, args.identity)
    runner = PreparationRunner(continue_from=folder)
    environment()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TaskLimit('Task supervisor deadline reached')))
    try:
        print(json.dumps(runner.run_preparation(specification)), flush=True)
    finally:
        runner.client.close()


if __name__ == '__main__':
    main()
