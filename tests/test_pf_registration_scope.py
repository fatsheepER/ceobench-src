"""Registration binds explicit history without treating old contexts as current reads."""
import json

from saas_bench.execution_capture import model_request, text_sources
from test_text_registry import workspace, captured, declaration


def request(store, text, context):
    body = dict(messages=[dict(role='tool', content=text)])
    event = model_request(store, json.dumps(body).encode(), text_sources(body), 'call', 'attempt', context)
    store.complete(event, send_state='response_received')
    return event


def test_explicit_citation_never_searches_old_requests_or_changes_requested_version(workspace, tmp_path, monkeypatch):
    store, registry, executor = captured(workspace, tmp_path)
    old = executor.execute('read_file', {'path': 'evidence.json'})
    old_request = request(store, old, 'old-week')
    (workspace / 'evidence.json').write_text('{"n":8}')
    new = executor.execute('read_file', {'path': 'evidence.json'})
    current_request = request(store, new, 'new-week')
    read_event = store.read_event
    def bounded(event, **kwargs):
        assert event != old_request, 'Registration searched a previous context'
        return read_event(event, **kwargs)
    monkeypatch.setattr(store, 'read_event', bounded)
    receipt = executor.execute('text_create', declaration({'version': 'evidence.json@v1'}),
                               model_request_event=current_request, model_context_id='new-week')
    assert receipt.startswith('Registered'), receipt
    binding = store.load_state('declaration:r1.1')['references'][0]
    assert registry.resolver.handle(binding['version_id']) == 'evidence.json@v1'
    assert binding['reading_scope'] == 'not_in_request' and not binding['delivered_in']
    assert 'latest captured evidence.json@v2' in receipt and 'as you last saw it' not in receipt
    assert 'not present in this request' in receipt


def test_path_and_selected_fields_require_current_request_not_previous_week(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    request(store, executor.execute('read_file', {'path': 'evidence.json'}), 'old-week')
    current = request(store, 'New weekly dashboard', 'new-week')
    for ref in [dict(evidence={'path': 'evidence.json'}),
                dict(evidence={'version': 'evidence.json@v1'}, select={'path': '/n'})]:
        text = executor.execute('text_create', declaration(references=[ref]),
                                model_request_event=current, model_context_id='new-week')
        assert text.startswith('Error:') and 'this request' in text, text
    assert not registry.path.exists()


def test_same_batch_write_is_bound_to_its_request_and_not_later_context(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    current = request(store, 'Write the file, then register it', 'week-one')
    scope = dict(model_request_event=current, model_context_id='week-one')
    executor.execute('write_file', {'path': 'plan.json', 'content': '{"n":9}'}, **scope)
    text = executor.execute('text_create', declaration({'path': 'plan.json'}), **scope)
    assert text.startswith('Registered'), text
    binding = store.load_state('declaration:r1.1')['references'][0]
    written = store.read_event(binding['authored_by'])['request']
    assert written['model_request_event'] == current and written['model_context_id'] == 'week-one'
    later = request(store, 'Next context', 'week-two')
    text = executor.execute('text_create', declaration({'path': 'plan.json'}),
                            model_request_event=later, model_context_id='week-two')
    assert text.startswith('Error:') and 'this request' in text, text


def test_new_computed_file_does_not_fall_back_to_old_authored_fields(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    current = request(store, 'Write a plan', 'week')
    scope = dict(model_request_event=current, model_context_id='week')
    executor.execute('write_file', {'path': 'plan.json', 'content': '{"n":7}'}, **scope)
    executor.execute('bash', {'command': "python -c 'import json; open(\"plan.json\",\"w\").write(json.dumps(dict(n=8)))'"}, **scope)
    result = executor.execute('text_create', declaration(references=[dict(
        evidence={'path': 'plan.json'}, select={'path': '/n'})]), **scope)
    assert result.startswith('Error:') and 'read the computed file' in result
