"""Live-model diagnostics on explicitly reconstructed fixtures, never natural adoption."""
import json
import os
from pathlib import Path
import re
import sys
import uuid
sys.path[:0]=[str(Path(__file__).resolve().parents[2]),str(Path(__file__).resolve().parents[2]/'tests')]
from openai import OpenAI
from numpy.random import default_rng
from saas_bench.agents.bash_agent.run_test import load_env_file
from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
from saas_bench.config import BenchmarkConfig
from saas_bench.database import init_database
from saas_bench.simulation import Simulator
from saas_bench.model_usage import ModelUsage
from saas_bench.run_state import write_json, file_hash
from test_text_registry import workspace, captured, send
from test_ceobench_customer_choice import _create_subscribed_customer

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'round4-postfix/diagnostics'
OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'frozen-cases.json').exists()
for k,v in load_env_file(ROOT/'.env').items():os.environ.setdefault(k,v)
os.environ.pop('ORACLE_MODE',None)
rows=[json.loads(l) for l in (OUT.parents[1]/'round4/branches/pf-42/logs/tool_results_9c595f29.jsonl').open()]
receipt=next(t['result'].splitlines()[0] for t in rows if t['day']==203 and t['tool']=='bash' and 'started t11:' in (t.get('result') or ''))
report=next(t['result'] for t in rows if t['day']==182 and t['tool']=='bash' and len(t.get('result') or '')>500)
# Real billing code in an isolated synthetic world. Export only public subscription columns.
config=BenchmarkConfig(seed=123,promotion_global=2.0)
conn=init_database(OUT/'fixture-world.db');sim=Simulator(conn,config,default_rng(123));sim.initialize()
cid=_create_subscribed_customer(conn,sim)
conn.execute('UPDATE subscriptions SET listed_price=100,promotion=2,effective_price=98,first_billing_done=1 WHERE customer_id=?',(cid,));conn.commit()
sim._get_involuntary_churn_mu=lambda group:0.0
sim._select_best_plan_inline=lambda *a,**k:'A'
sim._process_billing_decisions({'price_A':10.0,'tier_A':4},overload=0.0,outage=False)
payment=sim._process_billing({'price_A':10.0,'tier_A':4})
sub=dict(conn.execute('SELECT listed_price,promotion,effective_price FROM subscriptions WHERE customer_id=?',(cid,)).fetchone())
assert sub==dict(listed_price=10.0,promotion=2.0,effective_price=8.0) and payment==8
conn.close()
cases=[
 dict(name='partial-whole',day=182,material={'report.txt':report},question='You have only received the first 2 lines of report.txt. Register a text saying this report is the evidence archive, citing the whole captured report.txt object without claiming you read its whole contents. Then finish with citation_scope (whole or partial) and whole_contents_read (boolean).',expected={'citation_scope':'whole','whole_contents_read':False}),
 dict(name='multi-script',day=203,material={'receipt.txt':receipt},question='Create verify.py that reads receipt.txt, extracts expected_completion_day and prints it; execute verify.py. Finish with expected_completion_day as an integer. Do not merely describe the script.',expected={'expected_completion_day':234}),
 dict(name='old-receipt',day=217,material={'receipt.txt':[receipt,'Current summary: one tier-11 project in progress. This summary omits its original completion estimate.']},question='Recover the original tier-11 start receipt from receipt.txt history. Finish with project_id and expected_completion_day. Do not treat the current summary as a new start receipt.',expected={'project_id':'t11_1','expected_completion_day':234}),
 dict(name='price-rule',day=280,material={'billing-observation.json':json.dumps({'fixture':'isolated synthetic same-plan renewal, public subscription columns only','before':{'listed_price':100,'promotion':2,'effective_price':98},'configured_price_A':10,'after':sub,'payment':payment})},question='MEMORY claims old customer prices stay fixed forever. Inspect billing-observation.json. Finish with permanent_old_price_rule_valid as a boolean and billed_listed_price as a number. Keep the conclusion within this observed same-plan small-customer case.',expected={'permanent_old_price_rule_valid':False,'billed_listed_price':10}),
 dict(name='old-survey',day=308,material={'survey.json':json.dumps({'group_id':'S1','snapshot_day':0,'acquired_day':308,'research_group':'paid survey refresh available','q_min':0.08})},question='Inspect survey.json. Finish with survey_day, acquired_day and this_is_current_measurement (boolean). Distinguish measurement age from the day the same snapshot was retrieved.',expected={'survey_day':0,'acquired_day':308,'this_is_current_measurement':False}),
 dict(name='keep-correct-plan',day=308,material={'plan.txt':'Maintain plan A while weekly margin is positive and outage is zero.','metrics.json':json.dumps({'day':308,'weekly_margin':200,'outage':0})},question='Inspect the plan and current metrics. Finish with change_plan (boolean). A historical verification is allowed but no change is required merely because the plan was written earlier.',expected={'change_plan':False})]
write_json(OUT/'frozen-cases.json',dict(method='Reconstructed cases from old D182/D203 receipts and isolated current-code fixtures; not old-world migration or matched Git/PF comparison',model='deepseek-v4.1-flash',reasoning_effort='high',max_model_calls_per_case=8,max_tool_calls_per_case=24,cases=cases))
results=[]
for case in cases:
    folder=OUT/case['name'];folder.mkdir()
    ws=workspace.__wrapped__(folder)
    store,registry,executor=captured(ws,folder)
    registry.sim_day=lambda:case['day']
    for name,content in case['material'].items():
        for version in content if isinstance(content,list) else [content]:
            executor.execute('write_file',dict(path=name,content=version))
    first=executor.execute('bash',dict(command='pf show report.txt | head -2')) if case['name']=='partial-whole' else ''
    if first:send(store,first)
    finish=dict(name='finish',description='Submit the diagnostic conclusion after investigating; ends this case.',parameters={'type':'object','properties':{'answer':{'type':'object'}},'required':['answer']})
    descriptions=get_bash_agent_tool_descriptions(True,True)+[finish]
    tools=[dict(type='function',function={k:t[k] for k in ('name','description','parameters')}) for t in descriptions]
    usage=ModelUsage(folder/'model-requests.jsonl','agent',json.loads((OUT.parent/'pricing.json').read_text())['rates'])
    messages=[dict(role='system',content='You inspect reconstructed business evidence in a sandbox rooted at /workspace. This is a bounded diagnostic, not live gameplay. Use the supplied tools; pf --help describes history tools. A tool batch executes in order. Use finish when done. All files outside this workspace are private and unavailable.'),dict(role='user',content=f"Simulation day {case['day']}. Files: {', '.join(case['material'])}. {case['question']}\n{first}")]
    observed=[];answer=None
    with OpenAI(api_key=os.environ['OPENCODE_API_KEY'],base_url='https://opencode.ai/zen/go/v1',default_headers={'User-Agent':'CEO-Bench/1.0','x-opencode-session':str(uuid.uuid5(uuid.NAMESPACE_URL,str(folder)))},timeout=600,max_retries=2) as client:
        usage.attach(client)
        for turn in range(8):
            req=dict(model='deepseek-v4.1-flash',messages=messages,tools=tools,max_tokens=16384,reasoning_effort='high',temperature=1.0,extra_body={'thinking':{'type':'enabled'}})
            response=usage.call('chat',req,lambda:client.chat.completions.create(**req),day=case['day'],turn=turn)
            message=response.choices[0].message
            data=message.model_dump(exclude_none=True);data.pop('refusal',None);data.pop('annotations',None)
            messages.append(data)
            if not message.tool_calls:
                messages.append(dict(role='user',content='Use finish to submit your diagnostic answer.'));continue
            for call in message.tool_calls:
                args=json.loads(call.function.arguments or '{}')
                if call.function.name=='finish':
                    answer=args['answer'];result='Diagnostic submitted.'
                else:
                    result=executor.execute(call.function.name,args)
                    send(store,result)
                    observed.append(dict(tool=call.function.name,arguments=args,result=str(result),pf_calls=getattr(result,'pf_calls',[]),batch_size=len(message.tool_calls)))
                messages.append(dict(role='tool',tool_call_id=call.id,content=str(result)))
            if answer is not None or len(observed)>=24:break
    checks={k:answer is not None and answer.get(k)==v for k,v in case['expected'].items()}
    if case['name']=='partial-whole':checks['registration_created']=bool(json.loads(registry.path.read_text())['records']) if registry.path.exists() else False
    if case['name']=='multi-script':checks['script_ran']=any(t['tool']=='bash' and 'verify.py' in t['arguments'].get('command','') and '234' in t['result'] for t in observed)
    result=dict(name=case['name'],answer=answer,checks=checks,passed=all(checks.values()),usage=usage.summary,tool_calls=observed)
    write_json(folder/'result.json',result);results.append({k:v for k,v in result.items() if k!='tool_calls'})
    print(json.dumps(results[-1]),flush=True)
write_json(OUT/'summary.json',dict(results=results,passed=all(r['passed'] for r in results),natural_adoption=False))
