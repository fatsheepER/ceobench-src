"""Request-level PF reads: reconstruction, recovery, replay costs, and SDK wires."""
from contextlib import closing
import json
import os
from pathlib import Path

import pytest

from saas_bench.execution_capture import model_request, text_sources, restore_sources
from saas_bench.pf_read import accounting, apply_delta, make_delta, prepare_request, restore_request
from saas_bench.sql_evidence import encoded
from test_text_registry import workspace, captured, call, declaration
from test_pf_queries import read


class Counter:
    # Deterministic test costs, never used or reported as model token counts.
    metadata = {'tokenizer_id': 'test-utf8-bytes'}

    def count(self, text):
        return len(text.encode())


def deliver(store, texts, context='week', status='succeeded', counter=Counter()):
    request = dict(messages=[dict(role='tool', tool_call_id=f'm{i}', content=text)
                             for i, text in enumerate(texts)])
    restored = prepare_request(store, request, context, counter)
    sources = text_sources(request)
    event = model_request(store, encoded(request), sources, 'call', 'attempt', context)
    store.complete(event, status, send_state='response_received')
    ledger = json.loads(store.get_content(event + ':pf_reads')[1])
    restore_request(request, restored)
    assert [m['content'] for m in request['messages']] == texts
    return event, ledger


def captured_file(executor, content, path='facts.txt'):
    result = executor.execute('write_file', dict(path=path, content=content))
    assert not result.startswith('Error:'), result
    return read(executor, target={'path': path})[2]


def external_file(executor, content, path='facts.txt'):
    """A version the model never wrote: bytes appear via another program, then a Bash boundary."""
    (executor.workspace_path / path).write_bytes(content.encode())
    executor.execute('bash', {'command': 'true'})
    return read(executor, target={'path': path})[2]


def sample():
    return ''.join(f'记录 {i:03d}: a useful observation about this business 🙂\r\n' for i in range(100))


@pytest.mark.parametrize('command', ['pf show facts.txt', 'pf diff facts.txt'])
@pytest.mark.parametrize('view', [('head', 2), ('tail', 2)])
def test_pf_pipes_filter_delivery_and_preserve_read_accounting(workspace, tmp_path, command, view):
    from saas_bench.pf_cli import apply_view
    store, registry, executor = captured(workspace, tmp_path)
    for text in ('a\nb\nc\n', 'A\nB\nC\n'):
        (workspace / 'facts.txt').write_text(text)
        executor.execute('bash', {'command': 'true'})
    full = executor.execute('bash', {'command': command})
    result = executor.execute('bash', {'command': f'{command} | {view[0]} -{view[1]}'})
    assert result == apply_view(str(full), view)
    if command == 'pf show facts.txt' and view[0] == 'tail':
        assert result == 'B\nC\n'
    event, reads = deliver(store, [result])
    assert len(reads) == 1 and reads[0]['complete'] is False
    assert reads[0]['mode'] == ('DIFF' if 'diff' in command else 'FULL')
    assert reads[0]['actual_tokens'] == Counter().count(result)
    for item in result.origins:
        meta, raw = store.get_content(item['version_id'])
        a, b = item['source_range']
        c, d = item['request_range']
        assert result[c:d] == raw.decode()[a:b]
        if meta['layer'] == 'file_bytes':
            assert not item['full_source']
    with pytest.raises(ValueError, match='delivered|neither read nor wrote'):
        call(registry, 'create', **declaration({'path': 'facts.txt'}))


def test_official_model_tokenizer_is_pinned_and_validated(tmp_path, monkeypatch):
    from saas_bench.payload_tokens import load_counter, tokenizer_config
    config = tokenizer_config('deepseek', 'deepseek-flash')
    assert config['status'] == 'available' and config['tokenizer_id'].endswith('/v41')
    assert config['revision'] in config['url'] and len(config['sha256']) == 64
    assert tokenizer_config('opencode', 'deepseek/deepseek-v4.1-flash')['sha256'] == config['sha256']
    assert load_counter('openai', 'unknown-model') is None
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    cache = tmp_path / '.cache/ceobench/tokenizers' / (config['sha256'] + '.json')
    cache.parent.mkdir(parents=True)
    cache.write_text('corrupt')
    load_counter.cache_clear()
    with pytest.raises(ValueError, match='Cached tokenizer checksum mismatch'):
        load_counter('deepseek', 'deepseek-flash')


def test_trajectory_reconstructs_and_accounts_actual_replays(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    original = sample()
    full = captured_file(executor, original)
    e1, a = deliver(store, [full])
    assert a[0]['mode'] == 'FULL'
    unchanged = read(executor, target={'path': 'facts.txt'})[2]
    e2, b = deliver(store, [full, unchanged])
    assert b[-1]['mode'] == 'UNCHANGED'
    updated = original.replace('记录 012:', '修订 012:')
    delta1 = captured_file(executor, updated)
    e3, c = deliver(store, [full, unchanged, delta1])
    assert c[-1]['mode'] == 'DELTA'
    assert apply_delta(original, c[-1]['edits']) == updated
    last = updated.replace('记录 041:', '修订 041:').removesuffix('\r\n')
    delta2 = captured_file(executor, last)
    e4, d = deliver(store, [full, unchanged, delta1, delta2])
    assert d[-1]['mode'] == 'DELTA'
    assert apply_delta(updated, d[-1]['edits']) == last
    assert len(d[-1]['chain']) >= 2
    before = accounting(store)
    assert before['net_saved_tokens'] == sum(
        item['full_tokens'] - item['actual_tokens'] for ledger in (a, b, c, d) for item in ledger)
    # Reconstructed content can support a reference, but isn't a literal occurrence.
    call(registry, 'create', **declaration({'path': 'facts.txt'}))
    binding = store.load_state('declaration:r1.1')['references'][0]
    occurrence = binding['delivered_in'][0]['occurrence']
    assert occurrence['reconstructible'] and not occurrence['full_source']
    assert occurrence['representation'] == 'DELTA'
    full_again = read(executor, target={'path': 'facts.txt'}, full=True)[2]
    e5, f = deliver(store, [full, unchanged, delta1, delta2, full_again])
    assert f[-1]['mode'] == 'FULL' and f[-1]['recovery_of'] == delta2.pf_read['id']
    # Removing an intermediate base forces the historical compact message to full.
    e6, g = deliver(store, [full, delta2])
    assert g[-1]['mode'] == 'FULL' and g[-1]['materialization']
    new_week = read(executor, target={'path': 'facts.txt'})[2]
    e7, h = deliver(store, [new_week], context='next_week')
    assert h[0]['mode'] == 'FULL' and h[0]['reason'] == 'no_complete_base'
    ledgers = [a, b, c, d, f, g, h]
    paired = {delta2.pf_read['id'], full_again.pf_read['id']}
    expected = sum(item['full_tokens'] - item['actual_tokens']
                   for ledger in ledgers for item in ledger if item['read_id'] not in paired)
    # Every occurrence of the recovered full text is extra input; materialization is not.
    expected -= sum(item['actual_tokens'] for ledger in ledgers for item in ledger
                    if item['read_id'] == full_again.pf_read['id'])
    summary = accounting(store)
    assert summary['net_saved_tokens'] == expected
    assert summary['materialized_full_tokens'] == sum(
        item['actual_tokens'] for ledger in ledgers for item in ledger if item['materialization']) > 0
    assert summary['recovery_pairs'] == 1 and summary['occurrences'] == sum(map(len, ledgers))
    assert summary['read_modes'] == {'FULL': 3, 'UNCHANGED': 1, 'DELTA': 2}
    for event in (e1, e2, e3, e4, e5, e6, e7):
        rows = json.loads(store.get_content(event + ':pf_reads')[1])
        for item in rows:
            actual = store.get_content(item['actual_version'])[1].decode()
            full_text = store.get_content(item['full_version'])[1].decode()
            assert item['actual_tokens'] == Counter().count(actual)
            assert item['full_tokens'] == Counter().count(full_text)
            assert item['message_id'].startswith('m')
    store.assert_healthy()
    if destination := os.environ.get('CEOBENCH_PF_READ_ARTIFACTS'):
        folder = Path(destination)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'constructed-ledger.json').write_text(json.dumps(dict(
            token_basis='synthetic test costs, not model tokens', summary=summary,
            requests=ledgers), ensure_ascii=False, indent=2))
        store.snapshot(folder / 'constructed-evidence.sqlite')


def test_adjacent_recovery_once_and_explicit_full_always(workspace, tmp_path):
    store, _, executor = captured(workspace, tmp_path)
    full = captured_file(executor, sample())
    deliver(store, [full])
    compact = read(executor, target={'path': 'facts.txt'})[2]
    deliver(store, [full, compact])
    recovery = read(executor, target={'path': 'facts.txt'})[2]
    _, ledger = deliver(store, [full, compact, recovery])
    assert ledger[-1]['reason'] == 'adjacent_recovery'
    assert ledger[-1]['recovery_of'] == compact.pf_read['id']
    repeat = read(executor, target={'path': 'facts.txt'})[2]
    _, ledger = deliver(store, [full, compact, recovery, repeat])
    assert ledger[-1]['mode'] == 'UNCHANGED'
    again = read(executor, target={'path': 'facts.txt'}, full=True)[2]
    _, ledger = deliver(store, [full, compact, recovery, repeat, again])
    assert ledger[-1]['reason'] == 'requested_full' and ledger[-1]['recovery_of'] == repeat.pf_read['id']
    # Both compact reads are zeroed and both recovered full texts are extra input.
    summary = accounting(store)
    assert summary['recovery_pairs'] == 2 and summary['known_conservative_saved_tokens'] == 0
    assert summary['net_saved_tokens'] == -summary['known_recovery_replay_tokens'] < 0



def test_explicit_full_pairs_only_with_the_immediately_preceding_compact_read(workspace, tmp_path):
    store, _, executor = captured(workspace, tmp_path)
    full = captured_file(executor, sample())
    deliver(store, [full])
    compact = read(executor, target={'path': 'facts.txt'})[2]
    _, ledger = deliver(store, [full, compact])
    assert ledger[-1]['mode'] == 'UNCHANGED'
    # Another request separates the compact read from the explicit full request.
    deliver(store, [full, compact])
    late = read(executor, target={'path': 'facts.txt'}, full=True)[2]
    _, ledger = deliver(store, [full, compact, late])
    assert ledger[-1]['mode'] == 'FULL' and ledger[-1]['reason'] == 'requested_full'
    assert ledger[-1]['recovery_of'] is None
    summary = accounting(store)
    assert summary['recovery_pairs'] == 0 and summary['known_conservative_saved_tokens'] > 0

def test_missing_base_diff_failed_delivery_and_truncation(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    first = captured_file(executor, sample())
    old = json.loads(first.split('\n')[0])['target']['version']
    deliver(store, [first])
    second = external_file(executor, sample().replace('记录 007', '修改 007'))
    diff = read(executor, target={'path': 'facts.txt'}, baseline={'version': old}, mode='diff')[2]
    _, ledger = deliver(store, [diff])
    assert ledger[0]['mode'] == 'DIFF'
    with pytest.raises(ValueError, match='not been delivered'):
        call(registry, 'create', **declaration({'version': json.loads(second.split('\n')[0])['target']['version']}))
    # Neither a failed response nor an absent prior request can supply a baseline.
    deliver(store, [second], status='failed')
    fresh = read(executor, target={'path': 'facts.txt'})[2]
    _, ledger = deliver(store, [fresh])
    assert ledger[0]['mode'] == 'FULL'
    assert accounting(store)['excluded_occurrences'] == {'failed': 1}
    assert accounting(store)['active_diff_tokens'] > 0
    long = captured_file(executor, sample() * 8, path='long.txt')
    _, ledger = deliver(store, [long, read(executor, target={'path': 'long.txt'})[2]])
    assert all(v['mode'] == 'FULL' and v['reason'] == 'partial_or_truncated' for v in ledger)
    event = store.begin(encoded({'sql': 'SELECT n'}), 'test')
    store.finish(event, 200, '', encoded(dict(success=True, columns=['n'], rows=[{'n': 1}],
                                            row_count=1, truncated=True)), {})
    store.delivered(event, 'sent')
    query = read(executor, target={'sql': 'SELECT n'})[2]
    _, ledger = deliver(store, [query, read(executor, target={'sql': 'SELECT n'})[2]])
    assert all(v['reason'] == 'partial_or_truncated' for v in ledger)


def test_fallback_costs_verification_and_negative_net(workspace, tmp_path, monkeypatch):
    store, _, executor = captured(workspace, tmp_path)
    full = captured_file(executor, sample())
    compact = read(executor, target={'path': 'facts.txt'})[2]
    _, first = deliver(store, [full, compact])
    assert first[-1]['mode'] == 'UNCHANGED'
    saved = first[-1]['full_tokens'] - first[-1]['actual_tokens']
    # A materialized replay already costs the counterfactual full text: zero, not negative.
    for _ in range(3):
        _, ledger = deliver(store, [compact])
        assert ledger[0]['materialization'] and ledger[0]['actual_tokens'] == ledger[0]['full_tokens']
    assert accounting(store)['net_saved_tokens'] == saved
    # Replays of a recovered full text are deducted and can outweigh earlier savings.
    deliver(store, [full], context='recovery')
    zeroed = read(executor, target={'path': 'facts.txt'})[2]
    deliver(store, [full, zeroed], context='recovery')
    recovered = read(executor, target={'path': 'facts.txt'})[2]
    replays = [deliver(store, [full, zeroed, recovered], context='recovery')[1] for _ in range(4)]
    assert replays[0][-1]['reason'] == 'adjacent_recovery'
    summary = accounting(store)
    assert summary['known_recovery_replay_tokens'] == sum(r[-1]['actual_tokens'] for r in replays)
    assert summary['net_saved_tokens'] == saved - summary['known_recovery_replay_tokens'] < 0
    small = captured_file(executor, 'x', path='small.txt')
    smaller = read(executor, target={'path': 'small.txt'})[2]
    _, ledger = deliver(store, [small, smaller])
    assert ledger[-1]['mode'] == 'FULL' and ledger[-1]['reason'] == 'compact_not_smaller'
    no_count = read(executor, target={'path': 'facts.txt'})[2]
    _, ledger = deliver(store, [full, no_count], counter=None)
    assert ledger[-1]['reason'] == 'tokenizer_unavailable'
    assert accounting(store)['net_saved_tokens'] is None
    changed = captured_file(executor, sample().replace('记录 005', '修订 005'))
    monkeypatch.setattr('saas_bench.pf_read.make_delta', lambda *_: [[0, 1, 'wrong']])
    _, ledger = deliver(store, [full, changed])
    assert ledger[-1]['mode'] == 'FULL' and ledger[-1]['reason'] == 'delta_verification_failed'
    for before, after in [('', '🙂'), ('中\r\nx', '中\nx\n'), ('same', 'same'), ('x\n', ''),
                          ('no newline', 'changed'), ('a\nb\nc', 'start\na\nc\nend')]:
        # Call the imported original function after injecting the bad generator.
        assert apply_delta(before, make_delta(before, after)) == after


def test_final_wire_validation_and_private_snapshot_sources(workspace, tmp_path):
    store, _, executor = captured(workspace, tmp_path)
    full = captured_file(executor, sample())
    compact = read(executor, target={'path': 'facts.txt'})[2]
    snapshot = dict(messages=[dict(role='tool', content=full), dict(role='tool', content=compact)])
    sources = text_sources(snapshot)
    restored = json.loads(encoded(snapshot))
    restore_sources(restored, sources)
    assert restored['messages'][1]['content'].pf_read == compact.pf_read
    prepare_request(store, restored, 'week', Counter())
    wire_sources = text_sources(restored)
    assert json.loads(restored['messages'][1]['content'].split('\n')[0])['delivery'] == 'UNCHANGED'
    # Even a self-consistent source record must fail if its required base is gone.
    bad = dict(messages=[restored['messages'][1]])
    bad_sources = text_sources(bad)
    with pytest.raises(ValueError, match='cannot be reconstructed'):
        model_request(store, encoded(bad), bad_sources, 'call', 'bad', 'week')
    # Serialized mutation is rejected before it can be sent to a provider.
    restored['messages'][1]['content'] += 'corrupt'
    with pytest.raises(ValueError, match='serialization changed'):
        model_request(store, encoded(restored), wire_sources, 'call', 'bad2', 'week')


@pytest.mark.parametrize('api', ['chat', 'responses', 'messages'])
def test_real_sdk_wire_replays_and_failed_attempts(workspace, tmp_path, monkeypatch, api):
    import httpx
    from openai import OpenAI
    from anthropic import Anthropic
    from saas_bench.model_usage import ModelUsage
    from test_preflight_usage import reply
    store, _, executor = captured(workspace, tmp_path)
    full = captured_file(executor, sample())
    compact = read(executor, target={'path': 'facts.txt'})[2]
    received = []
    monkeypatch.setattr('time.sleep', lambda _: None)
    def handle(request):
        received.append(json.loads(request.content))
        if len(received) == 1:
            return httpx.Response(500, json={'error': {'type': 'server_error', 'message': 'offline retry'}})
        return httpx.Response(200, json=reply(api))
    client = (Anthropic if api == 'messages' else OpenAI)(api_key='offline', max_retries=1,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    usage = ModelUsage(tmp_path / 'requests.jsonl', 'agent', evidence_store=store, token_counter=Counter())
    usage.attach(client)
    kwargs = dict(model='test-model')
    if api == 'chat':
        kwargs['messages'] = [dict(role='tool', tool_call_id=f'call{i}', content=v) for i, v in enumerate((full, compact))]
        invoke = lambda: client.chat.completions.create(**kwargs)
    elif api == 'responses':
        kwargs['input'] = [dict(type='function_call_output', call_id=f'call{i}', output=v) for i, v in enumerate((full, compact))]
        invoke = lambda: client.responses.create(**kwargs)
    else:
        kwargs.update(max_tokens=10, messages=[dict(role='user', content=[
            dict(type='tool_result', tool_use_id=f'call{i}', content=v) for i, v in enumerate((full, compact))])])
        invoke = lambda: client.messages.create(**kwargs)
    for _ in range(2):
        usage.call(api, kwargs, invoke)
    assert received[0] == received[1] == received[2]
    assert 'UNCHANGED' in json.dumps(received[0])
    assert accounting(store)['occurrence_modes'] == {'FULL': 2, 'UNCHANGED': 2}
    assert accounting(store)['excluded_occurrences'] == {'failed': 2}
    with closing(store.connect()) as conn:
        events = conn.execute("SELECT event_id FROM requests WHERE json_extract(request,'$.kind')='model_request'").fetchall()
    for row in events:
        ledger = json.loads(store.get_content(row[0] + ':pf_reads')[1])
        assert [v['message_id'] for v in ledger] == ['call0', 'call1']
    client.close()
    store.assert_healthy()


def test_agent_snapshot_restore_pruning_and_new_week(workspace, tmp_path):
    import httpx
    from openai import OpenAI
    from saas_bench.agents.bash_agent.agent import BashAgent, Message
    from saas_bench.agents.bash_agent.tools import get_bash_agent_tool_descriptions
    from saas_bench.model_usage import ModelUsage
    from test_preflight_usage import reply
    store, _, executor = captured(workspace, tmp_path)
    full = captured_file(executor, sample())
    compact = read(executor, target={'path': 'facts.txt'})[2]
    wires = []
    def handle(request):
        wires.append(json.loads(request.content))
        body = reply('chat')
        body['choices'][0]['message']['tool_calls'] = [dict(id='call', type='function',
            function=dict(name='pf_read', arguments='{"target":{"path":"facts.txt"}}'))]
        return httpx.Response(200, json=body)
    client = OpenAI(api_key='offline', max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    def agent():
        value = BashAgent(get_bash_agent_tool_descriptions(True, True), client, workspace_path=workspace,
            usage_recorder=ModelUsage(None, 'agent', evidence_store=store, token_counter=Counter()))
        value._snapshot_path = workspace / 'conversation.json'
        value.current_day = 0
        return value
    first = agent()
    first.conversation = [Message('system', first.system_prompt), Message('tool', full, tool_call_id='a'),
                          Message('tool', compact, tool_call_id='b')]
    first._save_conversation_snapshot(strict=True)
    second = agent()
    assert second.load_conversation_snapshot(first._snapshot_path)
    assert second.conversation[-1].content.pf_read == compact.pf_read
    first._call_openai()
    second._call_openai()
    assert wires[0] == wires[1]
    assert json.loads(wires[1]['messages'][2]['content'].split('\n')[0])['delivery'] == 'UNCHANGED'
    second.conversation = [second.conversation[0], second.conversation[2]]
    second._call_openai()
    assert json.loads(wires[-1]['messages'][1]['content'].split('\n')[0])['delivery'] == 'FULL'
    second._refresh_context('new dashboard', 7)
    current = read(executor, target={'path': 'facts.txt'})[2]
    second.conversation.append(Message('tool', current, tool_call_id='new-week'))
    second._call_openai()
    assert json.loads(wires[-1]['messages'][-1]['content'].split('\n')[0])['delivery'] == 'FULL'
    assert accounting(store)['materialized_full_tokens'] > 0
    store.assert_healthy()
    client.close()
