"""Literal history search preserves visibility, pagination and excerpt provenance."""
import json

import pytest

from saas_bench.execution_capture import model_request, text_sources
from saas_bench.pf_cli import parse_argv
from test_text_registry import workspace, captured, call, declaration, send


def test_literal_search_pages_do_not_expand_to_future_or_private_evidence(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    for name in ('old.txt', 'recent.txt'):
        (workspace / name).write_text('a literal Q_MIN.* value')
        executor.execute('read_file', {'path': name})
    # A duplicate acquisition must not become another hit.
    executor.execute('read_file', {'path': 'recent.txt'})
    event = store.begin_event('model_request')
    store.version(event, 'private', 'Q_MIN.* secret', layer='model_request_wire')
    store.complete(event)
    refresh = store.begin_event('pf_weekly_check')
    child = store.begin_event('stale_file_read', parent=refresh)
    store.version(child, 'file', 'Q_MIN.* automatic refresh', layer='file_bytes', object_id='refresh.txt')
    store.complete(child)
    store.complete(refresh)
    args = dict(text='q_min.*', limit=1)
    first = executor.pf_queries.answer('pf_search', args)
    assert first['total'] == 4  # two files and their two distinct read-tool returns
    cursor = first['next_cursor']
    (workspace / 'future.txt').write_text('q_min.* future')
    executor.execute('read_file', {'path': 'future.txt'})
    items = first['items'][:]
    while cursor:
        rest = executor.pf_queries.answer('pf_search', {'cursor': cursor})
        assert rest['total'] == 4
        items.extend(rest['items'])
        cursor = rest['next_cursor']
    assert {'recent.txt@v1', 'old.txt@v1'} <= {item['version'] for item in items}
    assert not any(word in json.dumps(items) for word in ('future', 'secret', 'refresh.txt'))
    assert parse_argv(['search', '--text', 'q_min.*']) == ('pf_search', {'text': 'q_min.*'})
    assert executor.pf_queries.answer('pf_search', {'object': {'id': 'q_min.*'}})['total'] == 0


def test_empty_object_search_suggests_a_shell_quoted_literal_search(workspace, tmp_path):
    import shlex
    _, _, executor = captured(workspace, tmp_path)
    term = 'cash $(touch unintended) "plan"'
    result = executor.execute('pf_search', {'object': {'id': term}})
    command = result.split('Try literal history search: ', 1)[1]
    assert shlex.split(command) == ['pf', 'search', '--text', term]
    assert '0 saved items' not in result


def test_search_delivers_only_exact_excerpt_and_cannot_support_unseen_field(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    body = json.dumps(dict(padding='x' * 800, match='needle', tail='y' * 800, unseen=42))
    (workspace / 'data.json').write_text(body)
    executor.execute('read_file', {'path': 'data.json'})
    output = executor.execute('pf_search', {'text': 'needle'})
    assert not output.startswith('Error:'), output
    event = send(store, output)
    occurrences = json.loads(store.get_content(event + ':occurrences')[1])
    excerpt = next(item for item in occurrences if registry.resolver.identity(item['version_id'])[1] == 'data.json')
    start, end = excerpt['source_range']
    assert 'needle' in body[start:end] and 'unseen' not in body[start:end]
    assert not excerpt['full_source'] and start > 0 and end < len(body)
    with pytest.raises(ValueError, match='this request'):
        call(registry, 'create', **declaration(references=[dict(evidence={'path': 'data.json'}, select={'path': '/unseen'})]))
    call(registry, 'create', **declaration({'version': 'data.json@v1'}))
    assert store.load_state('declaration:r1.1')['references'][0]['reading_scope'] == 'partial'
