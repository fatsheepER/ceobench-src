import copy
import json
import os
from pathlib import Path

import pytest

from test_checkpoint_driver import Driver


@pytest.mark.parametrize('api', ['chat', 'responses', 'empty-input', 'no-requests'])
def test_driver_close_writes_request_samples_without_changing_wire(tmp_path, monkeypatch, api):
    destination = Path(os.environ.get('CEOBENCH_R228_SAMPLES', tmp_path / 'samples'))
    monkeypatch.setenv('CEOBENCH_R228_SAMPLES', str(destination))
    instructions = 'SAVED-SCRIPT-ceo'
    arguments = json.dumps({'hint': 'SAVED-SCRIPT-ceo'})
    output = 'SAVED-SCRIPT-ceo SAVED-SCRIPT-ceo SAVED-SCRIPT-growth'
    answer = 'SAVED-OPENING-ANSWER SAVED-OPENING-ANSWER'
    if api == 'chat':
        body = dict(messages=[
            dict(role='system', content=instructions),
            dict(role='user', content='SAVED-OPENING-ANSWER'),
            dict(role='assistant', content=None, tool_calls=[dict(id='call-1', type='function',
                function=dict(name='bash', arguments=arguments))]),
            dict(role='tool', tool_call_id='call-1', content=output),
            dict(role='assistant', content=[dict(type='text', text=answer)]),
        ])
    else:
        body = dict(instructions=instructions, input=[
            dict(role='user', content='SAVED-OPENING-ANSWER'),
            dict(type='reasoning', summary=[],
                encrypted_content='opaque SAVED-SCRIPT-ceo SAVED-OPENING-ANSWER'),
            dict(type='function_call', call_id='call-1', name='bash', arguments=arguments),
            dict(type='function_call_output', call_id='call-1', output=output),
            dict(type='message', role='assistant', content=[dict(type='output_text', text=answer)]),
        ])
    if api == 'empty-input':
        body = dict(input=[])
    wire = [] if api == 'no-requests' else [dict(role='ceo', day=7, phase='ceo', body=body,
        endpoint='/chat/completions' if api == 'chat' else '/responses')]
    proof = dict(wire=wire, errors=[], ready=dict(day=7), model_server_closed=True,
                 clients_closed=[True], closed=dict(live_role_threads=[], sockets_remaining=[]))
    sample = dict(selection='week-2', week=2, resume=True, code=0,
                  controller=dict(children_reaped=True), workers=[proof])
    original = copy.deepcopy(sample)
    driver = Driver(None, tmp_path, monkeypatch)
    driver.samples.append(sample)
    driver.close()

    path = destination / (tmp_path.name + '.json')
    stored = json.loads(path.read_text())
    requests = [] if api == 'no-requests' else [dict(role='ceo', day=7, phase='ceo',
        message_roles=[] if api == 'empty-input' else ['system', 'user', 'assistant', 'tool', 'assistant'],
        script_marker_count=0 if api == 'empty-input' else 4,
        opening_answer_count=0 if api == 'empty-input' else 3)]
    expected_proof = {key: value for key, value in proof.items() if key != 'wire'} | dict(requests=requests)
    assert stored == dict(invocations=[dict(sample, workers=[expected_proof])],
                          children_reaped=True, watchdog_closed=True, budget_expired=False)
    assert sample == original
    first = path.read_bytes()
    driver.close()
    assert path.read_bytes() == first
