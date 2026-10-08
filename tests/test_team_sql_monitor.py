import importlib.util
import json
from pathlib import Path


def test_team_sql_monitor_preserves_fault_hold_across_polls_and_restart(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('team_sql_monitor', scripts / 'team_sql_monitor.py')
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    monkeypatch.setattr(monitor, 'verify_build', lambda *args, **kwargs: None)
    monkeypatch.setattr(monitor, 'tree_hash', lambda path: 'frozen-source')
    monkeypatch.setattr(monitor.round5, 'quota_health', lambda: {'rolling': {'status': 'ok', 'percent': 1}})

    def output(name):
        root = tmp_path / name
        (root / 'engineering').mkdir(parents=True)
        (root / 'host-logs').mkdir()
        (root / 'engineering' / 'manifest.json').write_text(json.dumps({'source_sha256': 'frozen-source'}))
        return root

    root = output('evidence-fault')
    fault = root / 'engineering/runs/engineering-s42-r1-pf/attempt-002/runtime/private/evidence.fault.json'
    fault.parent.mkdir(parents=True)
    fault.write_text(json.dumps({'reason': 'SQL evidence capture failed; collection stopped',
                                'capture_status': 'missing'}))
    state = {}
    first = monitor.poll(root, state)
    assert first['status'] == 'hold'
    assert first['fault_paths'] == first['new_fault_paths'] == [str(fault)]
    assert (root / 'engineering/HOLD').exists()
    second = monitor.poll(root, state)
    assert second['status'] == 'hold'
    assert second['fault_paths'] == [str(fault)]
    assert second['new_fault_paths'] == []
    saved = json.loads((root / 'sql-health.json').read_text())
    restarted = monitor.poll(root, {key: saved[key] for key in ('offsets', 'known_fault_paths')})
    assert restarted['status'] == 'hold' and restarted['fault_paths'] == [str(fault)]
    assert restarted['new_fault_paths'] == []
    events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    assert len(events) == 1 and events[0]['fault_paths'] == [str(fault)]
    fault.unlink()
    assert monitor.poll(root, {})['status'] == 'hold'

    root = output('ordinary-errors')
    log = root / 'host-logs/worker.log'
    log.write_text(''.join('[public_sql] ' + json.dumps({'error_kind': kind}) + '\n'
                           for kind in ('syntax', 'permission')))
    health = monitor.poll(root, {})
    assert health['status'] == 'no_new_service_faults'
    assert health['sql_events_since_last_check'] == 2 and health['service_failures'] == []
    assert not (root / 'engineering/HOLD').exists() and not (root / 'events.jsonl').exists()
    (root / 'engineering/HOLD').touch()
    assert monitor.poll(root, {})['status'] == 'hold'

    root = output('service-error')
    log = root / 'host-logs/worker.log'
    log.write_text('[public_sql] ' + json.dumps({'error_kind': 'service', 'failure_stage': 'capture',
                                               'error_type': 'OperationalError',
                                               'executed_sql': 'PRIVATE QUERY'}) + '\n')
    state = {}
    health = monitor.poll(root, state)
    assert health['status'] == 'hold' and len(health['service_failures']) == 1
    assert 'PRIVATE QUERY' not in (root / 'sql-health.json').read_text()
    next_health = monitor.poll(root, state)
    assert next_health['status'] == 'hold' and next_health['service_failures'] == []

    for trigger in ('freeze', 'quota'):
        root = output(trigger)

        def fault_check():
            if trigger == 'freeze':
                monkeypatch.setattr(monitor, 'tree_hash', lambda path: 'changed-source')
            else:
                monkeypatch.setattr(monitor.round5, 'quota_health', lambda: {'rolling': {'status': 'ok', 'percent': 95}})

        fault_check()
        state = {}
        assert monitor.poll(root, state)['status'] == 'hold'
        assert monitor.poll(root, state)['status'] == 'hold'
        monitor.main(['--output-dir', str(root), '--once'])
        health = json.loads((root / 'sql-health.json').read_text())
        assert health['status'] == 'hold'
        state = {key: health[key] for key in ('offsets', 'known_fault_paths', 'active_check_alerts')}
        events = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
        assert len(events) == 1
        assert events[0]['checks'] == ['source_changed' if trigger == 'freeze' else 'go_quota_near_limit']
        monkeypatch.setattr(monitor, 'tree_hash', lambda path: 'frozen-source')
        monkeypatch.setattr(monitor.round5, 'quota_health', lambda: {'rolling': {'status': 'ok', 'percent': 1}})
        assert monitor.poll(root, state)['status'] == 'hold'
        assert state['active_check_alerts'] == []
        fault_check()
        assert monitor.poll(root, state)['status'] == 'hold'
        assert len((root / 'events.jsonl').read_text().splitlines()) == 2
        monkeypatch.setattr(monitor, 'tree_hash', lambda path: 'frozen-source')
        monkeypatch.setattr(monitor.round5, 'quota_health', lambda: {'rolling': {'status': 'ok', 'percent': 1}})
