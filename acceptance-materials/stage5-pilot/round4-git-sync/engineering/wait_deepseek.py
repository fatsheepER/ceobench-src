"""Quiet recovery probes; no simulation operations or experiment decisions."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI
from saas_bench.model_usage import ModelUsage, load_pricing

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def usable(raw):
    return bool(raw.get('model') == 'deepseek-flash' and raw.get('usage') and
                raw.get('choices') and raw['choices'][0].get('message', {}).get('content'))


if '--self-check' in sys.argv:
    assert not usable({'error': {'message': '900-second timeout'}})
    assert not usable({'model': 'deepseek-flash', 'usage': {'x': 1}, 'choices': []})
    assert usable({'model': 'deepseek-flash', 'usage': {'x': 1},
                   'choices': [{'message': {'content': 'OK'}}]})
    print('probe self-check passed')
    raise SystemExit

load_dotenv(ROOT / '.env')
client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url='https://api.deepseek.com',
                timeout=45, max_retries=0)
recorder = ModelUsage(HERE / 'recovery-probe-requests.jsonl', 'recovery_probe',
                      load_pricing(HERE.parent / 'pricing.json')['rates'])
recorder.attach(client)
request = dict(model='deepseek-flash', messages=[{'role': 'user', 'content': 'Reply OK.'}],
               max_tokens=16, extra_body={'thinking': {'type': 'disabled'}})
previous, successes = None, 0
while True:
    started = time.monotonic()
    try:
        reply = recorder.call('chat', request, lambda: client.chat.completions.create(**request))
        status = 'ok' if usable(reply.model_dump(mode='json')) else 'invalid_response'
    except Exception as exc:
        status = type(exc).__name__
    successes = successes + 1 if status == 'ok' else 0
    row = dict(at=datetime.now(timezone.utc).isoformat(), status=status,
               elapsed_seconds=round(time.monotonic() - started, 3), consecutive_successes=successes)
    with (HERE / 'recovery-probes.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    if status != previous or successes == 2:
        print(json.dumps(row), flush=True)
    previous = status
    if successes == 2:
        break
    time.sleep(300 if status == 'ok' else 600)
