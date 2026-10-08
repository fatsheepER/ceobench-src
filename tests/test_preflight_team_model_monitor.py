import importlib.util
import json
from pathlib import Path

import pytest


def load_monitor(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('team_model_monitor', scripts / 'team_model_monitor.py')
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    return monitor


def test_copied_prefix_retry_partial_line_and_restart_alert_once(tmp_path, monkeypatch):
    monitor = load_monitor(monkeypatch)

    def path(attempt):
        log = tmp_path / 'engineering/runs/pf' / attempt / 'runtime/private/growth/model-usage.jsonl'
        log.parent.mkdir(parents=True)
        return log

    old = dict(event='http_error', call_id='old-call', attempt_id='old-attempt', error='ReadTimeout')
    prefix = json.dumps(old) + '\n'
    first = path('attempt-001')
    first.write_text(prefix)
    state = {}
    errors = monitor.poll(tmp_path, state)
    assert len(errors) == 1 and errors[0]['attempt_id'] == 'old-attempt'
    assert monitor.poll(tmp_path, state) == []
    resumed = path('attempt-002')
    retry = dict(old, attempt_id='retry-attempt', error='ConnectError')
    partial = json.dumps(retry)
    cut = len(partial) // 2
    resumed.write_text(prefix + partial[:cut])
    assert monitor.poll(tmp_path, state) == []
    assert state['offsets'][str(resumed)] == len(prefix.encode())
    state = json.loads(json.dumps(state))
    with resumed.open('a') as stream:
        stream.write(partial[cut:] + '\n')
    errors = monitor.poll(tmp_path, state)
    assert len(errors) == 1 and errors[0]['attempt_id'] == 'retry-attempt'
    assert errors[0]['call_id'] == 'old-call' and errors[0]['issue'] == 'transport_error'
    assert state['offsets'][str(resumed)] == resumed.stat().st_size
    provider = dict(event='http_response', call_id='old-call', attempt_id='retry-attempt', status=503)
    fresh = dict(old, call_id='fresh-call', attempt_id='fresh-attempt')
    with resumed.open('a') as stream:
        stream.write(json.dumps(provider) + '\n' + json.dumps(fresh) + '\n')
    errors = monitor.poll(tmp_path, state)
    assert [(row['event'], row['attempt_id']) for row in errors] == [
        ('http_response', 'retry-attempt'), ('http_error', 'fresh-attempt')]
    state = json.loads(json.dumps(state))
    latest = path('attempt-003')
    latest.write_text(resumed.read_text())
    assert monitor.poll(tmp_path, state) == []
    assert state['offsets'][str(latest)] == latest.stat().st_size
    state['offsets'] = {}
    assert monitor.poll(tmp_path, state) == []
    assert not (tmp_path / 'engineering/HOLD').exists()
    invalid = path('attempt-004')
    invalid.write_text(json.dumps(dict(event='http_error', call_id='unknown-attempt')) + '\n')
    with pytest.raises(ValueError, match='Immutable model receipt ID'):
        monitor.poll(tmp_path, state)
    assert str(invalid) not in state['offsets']


def test_freeze_and_missing_cost_still_hold_with_copied_receipts(tmp_path, monkeypatch):
    monitor = load_monitor(monkeypatch)
    route = dict(event='http_request', call_id='route-call', attempt_id='route-attempt',
                 endpoint=monitor.round5.ENDPOINT + 'chat/completions', body={'model': 'changed-model'})
    simulator = dict(event='http_request', role='simulator', call_id='simulator-call', attempt_id='simulator-attempt',
                     endpoint=monitor.round5.ENDPOINT + 'chat/completions', body={'model': monitor.round5.MODEL})
    served = dict(event='response', call_id='served-call', response={'model': 'changed-model'},
                  usage=dict(input_tokens=1, output_tokens=1, cached_tokens=0), cost_usd=.01)
    cost = dict(served, call_id='cost-call', response={'model': monitor.round5.MODEL}, cost_usd=None)
    original = tmp_path / 'engineering/runs/pf/attempt-001/runtime/private/ceo/model-usage.jsonl'
    original.parent.mkdir(parents=True)
    original.write_text(''.join(json.dumps(row) + '\n' for row in (route, simulator, served, cost)))
    state = {}
    errors = monitor.poll(tmp_path, state)
    assert [row.get('issue', row['event']) for row in errors] == [
        'model_route_changed', 'simulator_thinking_changed', 'served_model_changed',
        'missing_cost', 'missing_uniform_cost']
    hold = tmp_path / 'engineering/HOLD'
    assert hold.exists()
    resumed = tmp_path / 'engineering/runs/pf/attempt-002/runtime/private/ceo/model-usage.jsonl'
    resumed.parent.mkdir(parents=True)
    resumed.write_text(original.read_text())
    hold.unlink()
    assert monitor.poll(tmp_path, json.loads(json.dumps(state))) == []
    assert hold.exists()
