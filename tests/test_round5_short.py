"""Real restored worlds and sandbox tools; only the external model is replaced."""
from datetime import datetime, timezone
import json
import urllib.error

import httpx
from openai import OpenAI
import pytest

from scripts import round5, round5_short
from saas_bench.model_usage import cost_usd
from saas_bench.payload_tokens import load_counter
from test_preflight_integration import advance, offline_runner, packed_public


@pytest.mark.parametrize('group', ['git', 'pf'])
@pytest.mark.parametrize('limited', [False, True])
def test_preparation_restores_world_blocks_writes_and_counts_regeneration(offline_runner, tmp_path, monkeypatch, group, limited):
    pricing = tmp_path / 'pricing.json'
    pricing.write_text(json.dumps(dict(source='offline fixture', basis='offline fixture',
        rates={'test-model': dict(input=.001, output=.002, cache_read=.0001)})))
    prefix = offline_runner(execution_capture=True, text_registration='prefix', pricing_file=pricing)
    for day in (7, 14, 21, 28):
        assert advance(prefix)['success']
        prefix._commit_weeks_up_to(day)
    prefix._save_checkpoint(28)
    sealed = tmp_path / 'A28'
    round5.seal_state(prefix.workspace_dir, sealed)
    specification = round5_short.freeze_task(sealed, tmp_path / 'specification.json')
    fork = round5_short.fork_task(sealed, tmp_path / group, group, 'short-offline-' + group)
    runner = round5_short.PreparationRunner(continue_from=fork, api_key='offline-only')
    counter = load_counter('opencode', 'deepseek-v4.1-flash')
    monkeypatch.setattr(round5_short, 'load_counter', lambda *args: counter)
    if limited:
        monkeypatch.setattr(round5_short, 'MAX_CALLS', 2)
    runner.prepare()
    try:
        before = runner._get_game_status()
        for path, body in [('/next-week', {}), ('/daily-scripts', {'name': 'bad', 'content': 'print(1)'}),
                           ('/call', {'tool': 'set_prices', 'args': {'A': 1}}),
                           ('/call', {'tool': 'research_group', 'args': {'group_id': 'S1'}}),
                           ('/call', {'tool': 'python_exec', 'args': {'code': 'print(1)'}})]:
            with pytest.raises(urllib.error.HTTPError) as error:
                runner._http_post(path, body)
            assert error.value.code == 403
        assert runner._http_post('/query', {'sql': 'SELECT SUM(amount) AS cash FROM ledger'})['rows']
        assert runner._get_game_status() == before
        assert not runner.agent.conversation and runner.agent.total_turns == 0
        actual = specification['public_facts']
        answer = dict(snapshot_day=28, proposed_actions=['Keep existing prices while reviewing evidence.'],
            cash_forecasts={h: dict(point=actual['current_cash_usd'], lower=-100000, upper=1e9)
                           for h in ('cash_1wk', 'cash_4wk', 'cash_12wk', 'cash_26wk')},
            facts=[dict(name=name, value=value, unit='USD', as_of_day=28, claim=name,
                        evidence=[dict(source='cash.txt', version='working', quote='offline public-state receipt')])
                   for name, value in actual.items()], judgments=[], missing_information=['Future outcomes are not observed.'])
        responses = [None, ('write_file', {'path': 'cash.txt', 'content': 'offline public-state receipt'}),
                     ('submit_decision', answer)]
        requests = []
        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            action = responses.pop(0)
            message = dict(role='assistant', content='preparation', reasoning_content='offline reasoning')
            if action:
                message['tool_calls'] = [dict(id='call-' + str(len(requests)), type='function',
                    function=dict(name=action[0], arguments=json.dumps(action[1])))]
            return httpx.Response(200, json=dict(id='offline', model='test-model', object='chat.completion',
                created=0, choices=[dict(index=0, finish_reason='tool_calls' if action else 'stop', message=message)],
                usage=dict(prompt_tokens=100, completion_tokens=10, prompt_cache_hit_tokens=20)))
        client = OpenAI(api_key='offline-only', base_url='https://api.deepseek.com/', max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        runner.agent.client = client
        runner.agent.usage_recorder.attach(client)
        monkeypatch.setattr(runner, 'prepare', lambda: before)
        result = runner.run_preparation(specification)
        if limited:
            assert result['status'] == 'limit' and result['usage']['calls'] == 2
            assert not (fork / 'submission.json').exists()
            client.close()
            return
        assert result['status'] == 'submitted'
        assert result['usage']['calls'] == 3 and result['usage']['known']['input_tokens'] == 300
        assert result['usage']['missing_cost'] == 0
        assert any('submit_decision' in m.get('content', '') for m in requests[1]['messages'] if m['role'] == 'user')
        assert result['input_attribution']['text_tokens']['system:content']['repeated'] > 0
        score = json.loads((fork / 'automatic-score.json').read_text())
        assert all(score['facts'].values()) and score['semantic_status'] == 'pending_blind_review'
        assert all(c['quote_matches'] for c in score['citations'])
        runner.submission['facts'][0]['value'] /= 1000000
        runner.submission['facts'][0]['evidence'][0]['quote'] = 'invented support'
        invalid = round5_short.score_submission(runner, specification)
        assert not invalid['facts']['current_cash_usd'] and not invalid['citations'][0]['quote_matches']
        assert (fork / 'submission.json').exists()
        if group == 'pf':
            with runner.evidence_store.connect() as conn:
                assert conn.execute("SELECT COUNT(*) FROM requests WHERE request LIKE '%submit_decision%'").fetchone()[0]
        else:
            assert not list(fork.rglob('sql-evidence*'))
        client.close()
    finally:
        runner._stop_server()
        runner.client.close()


def test_pricing_switches_at_utc_boundaries_and_weekends():
    rates = dict(input=.00015, output=.0006, cache_read=.000003,
                 peak_schedule='weekday_01_04_06_10_utc', peak_multiplier=2)
    usage = dict(input_tokens=1000, output_tokens=1000, cached_tokens=0, cache_creation_tokens=None)
    def price(day, hour):
        return cost_usd(usage, 'chat', rates, datetime(2026, 10, day, hour, tzinfo=timezone.utc))
    assert price(5, 9) == pytest.approx(.0015)
    assert price(5, 10) == pytest.approx(.00075)
    assert price(5, 1) == pytest.approx(.0015)
    assert price(5, 4) == pytest.approx(.00075)
    assert price(10, 9) == pytest.approx(.00075)
