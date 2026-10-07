"""Real independent Git/PF worlds; only model responses use the offline transport."""
import json

import httpx
from openai import OpenAI
import pytest

from scripts import round5, round5_rollout
from saas_bench.agents.bash_agent.agent import Message
from saas_bench.payload_tokens import load_counter
from saas_bench.model_usage import ModelUsage
from saas_bench.run_state import checkpoint_directory, tree_hash
from test_preflight_integration import advance, offline_runner, packed_public
from test_preflight_usage import reply


@pytest.mark.parametrize('group', ['git','pf'])
@pytest.mark.parametrize('start', [28,84])
def test_three_week_rollouts_keep_500_day_goal_and_fresh_context(offline_runner, tmp_path, monkeypatch, group, start):
    pricing = tmp_path / 'pricing.json'
    pricing.write_text(json.dumps(dict(source='offline',basis='offline',rates={'test-model':dict(input=.001,output=.002,cache_read=.0001)})))
    prefix = offline_runner(execution_capture=True,text_registration='prefix',pricing_file=pricing,total_days=500)
    for day in range(7,start+1,7):
        assert advance(prefix)['success']
        prefix._commit_weeks_up_to(day)
    prefix.agent.conversation = [Message(role='assistant',content='DO_NOT_RESTORE_OLD_CONVERSATION')]
    prefix.agent.total_turns = 13
    # A prefix simulator call may be stamped with the exact departure day.
    prior = ModelUsage(prefix.logs_dir/'simulator_requests.jsonl','simulator',{'test-model':dict(input=.001,output=.002,cache_read=.0001)})
    prior.call('chat',{'model':'test-model'},lambda:reply('chat'),day=start)
    old_call = json.loads(prior.path.read_text().splitlines()[-1])['call_id']
    prefix._save_checkpoint(start)
    source = tmp_path / 'sealed'
    round5.seal_state(prefix.workspace_dir,source)
    checksum = tree_hash(checkpoint_directory(source,json.loads((source/'checkpoint.json').read_text())))
    counter = load_counter('opencode','deepseek-v4.1-flash')
    monkeypatch.setattr(round5_rollout,'load_counter',lambda *args:counter)
    monkeypatch.setattr(round5,'environment',lambda:None)
    requests = []
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert '500 simulated days' in body['messages'][0]['content']
        assert 'DO_NOT_RESTORE_OLD_CONVERSATION' not in json.dumps(body)
        assert 'Decision preparation' not in body['messages'][0]['content']
        assert 'three-week' not in body['messages'][0]['content']
        assert len(requests)<=3
        command = "./novamind-operation next-week 'fixed offline action'" + ' 100000 -100000 1000000'*4
        message = dict(role='assistant',content='',reasoning_content='offline reasoning',tool_calls=[dict(id=f'week{len(requests)}',type='function',function=dict(name='bash',arguments=json.dumps({'command':command})))])
        return httpx.Response(200,json=dict(id='offline',model='test-model',object='chat.completion',created=0,choices=[dict(index=0,finish_reason='tool_calls',message=message)],usage=dict(prompt_tokens=100,completion_tokens=10,prompt_cache_hit_tokens=20)))
    client = OpenAI(api_key='offline-only',base_url='https://api.deepseek.com/',max_retries=0,http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    setup = round5_rollout.RolloutRunner.setup
    def prepare(runner):
        setup(runner)
        runner.client.close()
        runner.client = client
        runner.agent.client = runner.agent.usage_recorder.attach(client)
    monkeypatch.setattr(round5_rollout.RolloutRunner,'setup',prepare)
    destination = tmp_path/'episode'
    result = round5_rollout.execute(source,destination,group,'offline-'+group,tmp_path,api_key='offline-only')
    assert result['reached_stop'] and result['public_outcomes']['end_day']==start+21
    assert result['public_outcomes']['completed_weeks']==3 and len(requests)==3
    assert result['usage']['agent']['calls']==3 and result['usage']['agent']['known']['input_tokens']==300
    assert result['usage']['agent']['missing_cost']==0
    assert old_call not in (destination/'simulator-episode-requests.jsonl').read_text()
    predictions = result['public_outcomes']['predictions']
    assert len(predictions)==12 and sum(p['matured'] for p in predictions)==3
    assert all(p['actual'] is not None for p in predictions if p['horizon_days']==7)
    assert tree_hash(checkpoint_directory(source,json.loads((source/'checkpoint.json').read_text())))==checksum
    if group=='git':
        assert not list(destination.rglob('sql-evidence*'))
