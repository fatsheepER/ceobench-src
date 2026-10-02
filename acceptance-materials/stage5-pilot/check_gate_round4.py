"""Check D35 request contracts, checkpoints and accounting without behavioral analysis."""
import json
import os
from pathlib import Path
import sqlite3
import sys
import argparse
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analyze_pf_run import tool_batch_accounting

from saas_bench.agents.bash_agent.agent import BashAgent
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from saas_bench.registration_prompt import integrate
from saas_bench.run_state import checkpoint_directory, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'round4')
out = parser.parse_args().output_dir.resolve()
checks = []
os.environ.pop('ORACLE_MODE', None)
for mode in ('git', 'pf'):
    record = json.loads((out / f'long-{mode}-42-to35.json').read_text())
    run = Path(record['path'])
    assert record['status'] == 'stopped' and record['reached_stop']
    cp = json.loads((run / 'checkpoint.json').read_text())
    assert cp['day'] == 35 and cp['context_boundary'] == 'new_week'
    snapshot = checkpoint_directory(run, cp)
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['text_registration'] == mode
    assert not any((run / name).exists() for name in ('branch_stop.json', 'sql-evidence.fault.json'))
    rows = [json.loads(line) for line in (run / 'logs/agent_requests.jsonl').open()]
    ids = {r['call_id'] for r in rows if r['event'] == 'request' and r.get('day', -1) == 28}
    wire = [r['body'] for r in rows if r['event'] == 'http_request' and r['call_id'] in ids]
    assert wire
    expected_tools = get_bash_agent_tool_descriptions(True, mode == 'pf')
    expected = [{'type': 'function', 'function': {k: t[k] for k in ('name', 'description', 'parameters')}} for t in expected_tools]
    base = BashAgent._default_system_prompt(type('Prompt', (), {'total_days': 497})())
    prompt = integrate(base, pf=mode == 'pf')
    for body in wire:
        assert body['tools'] == expected
        system = body['messages'][0]['content']
        assert system.startswith(prompt) and '497 simulated days' in system
        assert ('[pf: MEMORY.md@' if mode == 'pf' else '[git: MEMORY.md last committed') in system
        assert body['model'] == 'deepseek-v4.1-flash'
    tools = [json.loads(line) for line in (run / 'logs' / f"tool_results_{record['run_id']}.jsonl").open()]
    tools = [t for t in tools if t['day'] >= 28 and not t['tool'].startswith('_')]
    raw = [json.loads(line) for name in (run / 'logs').glob('raw_responses_*.jsonl') for line in name.open()]
    timing = [json.loads(line) for name in (run / 'logs').glob('timing_*.jsonl') for line in name.open()]
    batches = tool_batch_accounting([r for r in raw if r['day'] >= 28], tools,
                                   [t for t in timing if t['day'] >= 28])
    assert batches['status'] == 'passed', batches
    receipts = [t['result'] for t in tools if t['tool'] in ('text_create', 'text_revise')]
    assert all(r.startswith(('Registered ', 'Revised ', 'Error:')) for r in receipts), receipts
    pf_calls = [call for t in tools for call in t.get('pf_calls', [t['pf_call']] if t.get('pf_call') else [])]
    capture = None
    if mode == 'pf':
        assert any('[pf:' in t['result'] for t in tools), 'PF output handles missing'
        with sqlite3.connect((snapshot / 'sql-evidence.sqlite').as_uri() + '?mode=ro&immutable=1', uri=True) as conn:
            assert conn.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
            assert conn.execute('SELECT count(*) FROM requests WHERE event_id NOT IN (SELECT event_id FROM results)').fetchone()[0] == 0
            assert conn.execute('SELECT count(*) FROM client_calls WHERE received IS NULL').fetchone()[0] == 0
            capture = conn.execute('SELECT count(*) FROM requests').fetchone()[0]
            for call in pf_calls:
                if call.get('event_id'):
                    assert conn.execute('SELECT 1 FROM results WHERE event_id=?', (call['event_id'],)).fetchone()
    else:
        assert not list(run.rglob('sql-evidence*'))
    usage = json.loads((run / 'usage_summary.json').read_text())
    assert usage['day'] == cp['day']
    assert usage['agent'] == cp['usage']
    for role in ('agent', 'simulator'):
        assert usage[role]['missing_cost'] == 0
        assert usage[role]['known_cost_usd'] is not None
        assert (snapshot / 'request_logs' / f'{role}_requests.jsonl').exists()
    checks.append(dict(mode=mode, status='passed', day=35, real_http_requests=len(wire),
                       prompt_and_tool_contracts=True, memory_history_line=True, tool_batch_accounting=batches,
                       registration_receipts=len(receipts), pf_calls=pf_calls, capture_events=capture,
                       usage={role: {k: usage[role][k] for k in ('calls', 'errors', 'missing_cost', 'known_cost_usd')} for role in ('agent', 'simulator')}))
write_json(out / 'gate-D35-verification.json', dict(status='passed', branches=checks))
print(json.dumps(checks, ensure_ascii=False, indent=2))
