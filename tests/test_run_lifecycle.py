import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx
from openai import OpenAI
import pytest

from scripts import round5, round5_rollout
from saas_bench.run_lifecycle import RunCancelled, WorkerLifecycle
from test_preflight_integration import offline_runner, packed_public


@pytest.mark.parametrize('phase', ['model', 'backoff', 'tool'])
def test_parent_sigkill_stops_worker_and_server(tmp_path, phase):
    worker = tmp_path / 'worker.py'
    worker.write_text('''
import os, subprocess, sys, time
from pathlib import Path
from saas_bench.run_lifecycle import WorkerLifecycle, RunCancelled
root, phase = Path(sys.argv[1]), sys.argv[2]
server = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
(root/'server').write_text(str(server.pid))
try:
    with WorkerLifecycle(parent_pid=int(sys.argv[3])) as control:
        (root/'ready').write_text(str(os.getpid()))
        if phase == 'model':
            with control.model_call():
                time.sleep(60)
        elif phase == 'backoff':
            control.sleep(60)
        else:
            time.sleep(.5)
            (root/'tool-result').write_text('known result recorded')
            control.check()
except RunCancelled as exc:
    (root/'receipt').write_text(str(exc))
finally:
    server.terminate()
    server.wait()
    (root/'done').touch()
''')
    parent_code = 'import os,subprocess,sys,time; subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2],sys.argv[3],str(os.getpid())]); time.sleep(60)'
    parent = subprocess.Popen([sys.executable, '-c', parent_code, str(worker), str(tmp_path), phase])
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path/'ready').exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert (tmp_path/'ready').exists()
        parent.kill()
        parent.wait()
        deadline = time.monotonic() + 5
        while not (tmp_path/'done').exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert (tmp_path/'done').exists()
        assert (tmp_path/'receipt').read_text() == 'supervisor_exited'
        assert (tmp_path/'tool-result').exists() == (phase == 'tool')
        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path/'server').read_text()), 0)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()


def test_hold_only_interrupts_retry(tmp_path):
    hold = tmp_path/'hold.json'
    with WorkerLifecycle(hold) as control:
        hold.touch()
        control.check()
        with control.model_call():
            pass
        with pytest.raises(RunCancelled, match='hold_during_retry'):
            control.sleep(60)


def test_retry_hold_checkpoint_restores_exact_conversation(offline_runner, tmp_path, monkeypatch):
    runner = offline_runner()
    monkeypatch.setattr(runner, 'setup', lambda: None)
    hold = tmp_path/'hold.json'
    round5.install_pause(runner, hold)
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        assert len(requests) == 1, 'hold must stop regeneration before another request'
        hold.touch()
        return httpx.Response(200, json=dict(id='offline', model='test-model', object='chat.completion', created=0,
            choices=[dict(index=0, finish_reason='stop', message=dict(role='assistant', content='No tool'))],
            usage=dict(prompt_tokens=13, completion_tokens=7)))
    client = OpenAI(api_key='offline', base_url='https://api.deepseek.com/', max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    runner.agent.client = runner.agent.usage_recorder.attach(client)
    runner.agent.current_day = runner._get_game_status()['day']
    runner.agent.turns_today = 6
    result = runner.run(verbose=False)
    assert result['outcome'] == 'paused' and result['context_boundary'] == 'same_week'
    checkpoint = runner._load_checkpoint()
    assert checkpoint['usage']['calls'] == 1
    assert checkpoint['usage']['known']['input_tokens'] == 13
    assert checkpoint['usage']['missing_cost'] == 1
    original = runner.agent._snapshot_path.read_bytes()
    restored = offline_runner(runner.workspace_dir)
    assert restored.agent._snapshot_path.read_bytes() == original
    assert restored.agent._observation_recorded
    assert restored.agent.turns_today == 6
    assert restored.agent.usage_recorder.summary == checkpoint['usage']
    before = [m.to_dict() if hasattr(m, 'to_dict') else vars(m) for m in restored.agent.conversation]
    monkeypatch.setattr(restored.agent, '_call_llm', lambda: None)
    restored.agent.act(restored.agent._last_observation, 0, False, {'day': checkpoint['day']})
    assert [vars(m) for m in restored.agent.conversation] == before
    restored.agent.conversation.clear()
    rollout = round5_rollout.RolloutRunner.__new__(round5_rollout.RolloutRunner)
    rollout.agent = restored.agent
    rollout._restore_from_checkpoint(checkpoint)
    assert [vars(m) for m in restored.agent.conversation] == before
    client.close()


def test_model_cancellation_preserves_unknown_fee_and_stops_server(offline_runner, tmp_path, monkeypatch):
    import threading
    runner = offline_runner()
    monkeypatch.setattr(runner, 'setup', lambda: None)
    round5.install_pause(runner, tmp_path/'hold.json')
    lifecycle = runner.lifecycle
    round5.install_pause(runner, tmp_path/'other-hold.json')
    assert runner.lifecycle is lifecycle and runner.agent.lifecycle is lifecycle
    server = runner._server_proc
    def respond(request):
        timer = threading.Timer(.1, lambda: os.kill(os.getpid(), signal.SIGUSR1))
        timer.start()
        try:
            time.sleep(20)
            raise AssertionError('Model cancellation did not interrupt the transport')
        finally:
            timer.join()
    client = OpenAI(api_key='offline', base_url='https://api.deepseek.com/', max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    runner.agent.client = runner.agent.usage_recorder.attach(client)
    result = runner.run(verbose=False)
    assert result['outcome'] == 'paused' and result['context_boundary'] == 'same_week'
    assert server.poll() is not None
    checkpoint = runner._load_checkpoint()
    assert checkpoint['usage']['calls'] == 1
    assert checkpoint['usage']['missing_cost'] == 1
    rows = [json.loads(line) for line in (runner.logs_dir/'agent_requests.jsonl').read_text().splitlines()]
    response = next(row for row in rows if row['event'] == 'response')
    assert response['cost_usd'] is None
    assert all(value is None for value in response['usage'].values())
    assert not runner.agent._pending_tool_calls
    client.close()


def test_healthy_hold_finishes_week(offline_runner, tmp_path, monkeypatch):
    runner = offline_runner()
    monkeypatch.setattr(runner, 'setup', lambda: None)
    hold = tmp_path/'hold.json'
    round5.install_pause(runner, tmp_path/'original-hold.json')
    round5.install_pause(runner, hold)
    calls = []
    def respond(request):
        calls.append(request)
        hold.touch()
        command = "./novamind-operation next-week 'offline hold check'" + ' 100000 -100000 1000000'*4
        return httpx.Response(200, json=dict(id='offline', model='test-model', object='chat.completion', created=0,
            choices=[dict(index=0, finish_reason='tool_calls', message=dict(role='assistant', content='',
                tool_calls=[dict(id='advance', type='function', function=dict(name='bash', arguments=json.dumps(dict(command=command))))]))],
            usage=dict(prompt_tokens=13, completion_tokens=7)))
    client = OpenAI(api_key='offline', base_url='https://api.deepseek.com/', max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    runner.agent.client = runner.agent.usage_recorder.attach(client)
    result = runner.run(verbose=False)
    assert result['outcome'] == 'paused' and result['context_boundary'] == 'new_week'
    assert result['days_run'] == 7 and len(calls) == 1
    client.close()
