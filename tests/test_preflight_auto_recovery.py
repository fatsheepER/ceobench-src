import importlib.util
import json
from pathlib import Path

from saas_bench.run_state import write_json
from saas_bench.systemd_guard import scope_command
from test_preflight_team_controller import controller
from test_preflight_team_observer import observer


def test_scope_and_historical_hold_observations_do_not_block_requests(observer, controller, monkeypatch):
    observer.hold.touch()
    write_json(observer.output / 'health.json', dict(status='hold', batch_id='old', time='old'))
    controller.gate.scope = dict(unit='own.scope')
    monkeypatch.setattr(controller.module, 'check_worker_cgroup',
                        lambda proof: ['Worker cgroup limit changed: memory.high'])
    controller.gate(None)
    rows = [json.loads(line) for line in
            (observer.host_logs / 'worker-scope-observations.jsonl').read_text().splitlines()]
    assert rows[0]['observations'] == ['Worker cgroup limit changed: memory.high']
    assert not (controller.policy.parent / 'HOLD').exists()


def test_single_group_can_use_an_old_paired_manifest_and_ignore_control_freeze(observer):
    observer.expected['runs'] = [{key: observer.manifest['runs'][1][key] for key in ('run_id', 'output_dir')}]
    path = Path(__file__).resolve().parents[1] / 'scripts/team_driver.py'
    spec = importlib.util.spec_from_file_location('auto_team_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = observer.output.parent / 'single-group.json'
    write_json(policy, dict(identity=observer.expected, control_freeze='/absent/old.json',
                            control_freeze_sha256='old'))
    config, manifest = module.verify_controls(policy)
    assert [row['run_id'] for row in manifest['runs']] == ['pf']
    assert config['identity']['source_commit'] == 'frozen-commit'


def test_worker_memory_defaults_and_driver_forwarding(observer, monkeypatch):
    command = scope_command([], 'one.scope')
    assert '--property=MemoryHigh=17179869184' in command
    assert '--property=MemoryMax=21474836480' in command
    path = Path(__file__).resolve().parents[1] / 'scripts/team_driver.py'
    spec = importlib.util.spec_from_file_location('memory_team_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = observer.output.parent / 'memory-policy.json'
    write_json(policy, dict(identity=observer.expected))
    calls = []

    def supervise(path, **kwargs):
        calls.append(kwargs)
        write_json(path.parent / 'controller-exit.json', dict(children_reaped=True, hold=False, failed=False))
        write_json(path.parent / 'controller-terminal.json',
            {kwargs['run_id']: {}} if kwargs.get('run_id') else dict(git={}, pf={}))

    monkeypatch.setattr(module.controller, 'supervise', supervise)
    assert module.main(['--policy', str(policy), '--live']) == 0
    assert calls == [dict(resume=False, memory_high=17179869184,
                          memory_max=21474836480, memory_swap_max=0)]
    assert json.loads((policy.parent / 'driver-exit.json').read_text())['auto_resume'] is False
    assert module.main(['--policy', str(policy), '--live', '--run-id', 'pf', '--week', '1']) == 0
    assert calls[-1] == dict(resume=False, memory_high=17179869184,
                            memory_max=21474836480, memory_swap_max=0, run_id='pf', week=1)
    assert module.main(['--policy', str(policy), '--live', '--run-id', 'pf', '--week', '0']) == 0
    assert calls[-1]['week'] == 0
    assert any('week' in row['message'].lower() for row in calls[-1]['startup_warnings'])


def test_driver_reports_clean_operator_stop_without_requiring_unstarted_runs(observer, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/team_driver.py'
    spec = importlib.util.spec_from_file_location('operator_stop_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = observer.output.parent / 'operator-policy.json'
    write_json(policy, dict(identity=observer.expected))

    def supervise(path, **kwargs):
        write_json(path.parent / 'controller-exit.json', dict(children_reaped=True,
            hold=False, failed=False, operator_stopped=True))
        write_json(path.parent / 'controller-terminal.json', dict(pf=dict(disposition='paused',
            status='paused', operator_stop=True, unit=dict(scope_empty=True))))

    monkeypatch.setattr(module.controller, 'supervise', supervise)
    assert module.main(['--policy', str(policy), '--live']) == 0
    assert json.loads((policy.parent / 'driver-exit.json').read_text())['returncode'] == 0


def test_driver_keeps_unclean_operator_stop_as_failure(observer, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/team_driver.py'
    spec = importlib.util.spec_from_file_location('unclean_stop_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = observer.output.parent / 'unclean-policy.json'
    write_json(policy, dict(identity=observer.expected))

    def supervise(path, **kwargs):
        write_json(path.parent / 'controller-exit.json', dict(children_reaped=True,
            hold=False, failed=True, operator_stopped=True))
        write_json(path.parent / 'controller-terminal.json', dict(pf=dict(disposition='failed')))

    monkeypatch.setattr(module.controller, 'supervise', supervise)
    assert module.main(['--policy', str(policy), '--live']) == 2
