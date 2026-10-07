"""No model calls: the actual launcher preserves 500 days and stops exactly at 112."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import run_round4
os.environ['PYTHONHASHSEED']='0'
with TemporaryDirectory() as temp:
    root=Path(temp);source=root/'source';source.mkdir()
    (source/'checkpoint.json').write_text(json.dumps(dict(day=35,context_boundary='new_week')))
    (source/'manifest.json').write_text('{}')
    pricing=root/'pricing.json';pricing.write_text(json.dumps(dict(source='fixed reference',basis='USD/1k',rates={m:dict(input=.00015,output=.0006,cache_read=.000003) for m in ('deepseek-v4.1-flash','deepseek-flash')})))
    seen=[]
    def fake(**args):
        seen.append(args)
        return SimpleNamespace(total_days=497,run_id='offline',workspace_dir=root/'run',agent_workspace=root/'workspace',stop_after_day=args['stop_after_day'],_weekly_record=lambda s:None,run=lambda:dict(outcome='stopped',days_run=112))
    with patch.object(run_round4,'BashAgentRunner',fake),patch.object(run_round4,'BashAgentToolExecutor'),patch.object(run_round4,'verify_build',return_value={}):
        run_round4.main(['pf','--continue-from',str(source),'--stop-after-day','112','--pricing-file',str(pricing),'--output-dir',str(root/'out')])
    assert seen[0]['total_days']==500 and seen[0]['stop_after_day']==112
    record=json.loads((root/'out/long-pf-42-to112.json').read_text())
    assert record['reached_stop'] and record['start_day']==35
print('launcher D112 self-check passed')
