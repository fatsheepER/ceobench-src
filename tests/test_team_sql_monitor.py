import importlib.util
import json
from pathlib import Path


def test_team_sql_monitor_reports_faults_without_hold_or_source_gate(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('team_sql_monitor', scripts / 'team_sql_monitor.py')
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    monkeypatch.setattr(monitor.round5, 'quota_health', lambda: {'rolling': {'status': 'ok', 'percent': 1}})
    (tmp_path / 'engineering').mkdir()
    (tmp_path / 'host-logs').mkdir()
    (tmp_path / 'engineering/manifest.json').write_text(json.dumps({'source_sha256': 'old'}))
    fault = tmp_path / 'engineering/runs/pf/attempt-001/runtime/private/evidence.fault.json'
    fault.parent.mkdir(parents=True)
    fault.write_text('{}')
    log = tmp_path / 'host-logs/worker.log'
    log.write_text('[public_sql] ' + json.dumps(dict(error_kind='service',
        error_type='OperationalError', executed_sql='PRIVATE QUERY')) + '\n')
    original = fault.read_bytes()
    state = {}
    first = monitor.poll(tmp_path, state)
    assert first['status'] == 'observations'
    assert first['new_fault_paths'] == [str(fault)]
    assert first['freeze_checks'] == []
    assert len(first['service_failures']) == 1
    assert 'PRIVATE QUERY' not in (tmp_path / 'sql-health.json').read_text()
    assert not (tmp_path / 'engineering/HOLD').exists()
    second = monitor.poll(tmp_path, json.loads(json.dumps(state)))
    assert second['new_fault_paths'] == [] and second['service_failures'] == []
    assert fault.read_bytes() == original
    hold = tmp_path / 'engineering/HOLD'
    hold.touch()
    fault.unlink()
    health = monitor.poll(tmp_path, state)
    assert health['status'] == 'no_new_service_faults' and health['historical_hold']
    assert hold.exists()
