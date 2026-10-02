"""PF counters may describe a run; they cannot authorize D112 continuation."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import supervise_postfix as s
with TemporaryDirectory() as temp:
    out=Path(temp);pointer=out/'long-pf-42-to112.json';pointer.write_text('{}')
    (out/'frozen.json').write_text(json.dumps({'D112_rule':'Execution agent reviews actual evidence and actions'}))
    (out/'baseline-pf-first12.json').write_text(json.dumps({'weeks':[]}))
    good={'group':'pf','tool_batch_accounting':{'status':'passed','errors':[]},
          'weeks':[{'day':35,'pf_calls':[{'command':'show','outcome':'succeeded','capture_match':True}],'clock_in_results':0}]}
    with patch.object(s,'OUT',out),patch.object(s,'integrity'),patch.object(s,'branch_audit',return_value=good),patch.object(s,'notify'):
        result=s.audit112([{'path':str(out/'run'),'group':'pf'}])
    assert result['current']['successes']==1
    assert result['decision']=='awaiting_agent_review'
    assert not (out/'review-D112.json').exists()
print('D112 agent-review gate self-check passed')
