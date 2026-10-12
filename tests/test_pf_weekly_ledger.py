import json
import shlex
import subprocess
import sys

import pytest

from saas_bench.execution_capture import CapturedText, origin
from saas_bench.pf_queries import weekly_condition_visibility
from saas_bench.registration_schema import parse_applies
from test_text_registry import workspace, captured, call, declaration, send
from test_preflight_integration import packed_public
from test_public_sql import server


def weekly_event(store):
    with store.connect() as conn:
        row = conn.execute("SELECT event_id FROM requests WHERE json_extract(request, '$.kind')='pf_weekly_check' ORDER BY rowid DESC LIMIT 1").fetchone()
    return row[0], store.read_event(row[0])['result']


def file_ref(purpose='current', value=10):
    return dict(evidence={'path': 'evidence.json'}, purpose=purpose, select={'path': '/n'},
                predicate={'type': 'threshold', 'op': '<', 'value': value})


@pytest.mark.parametrize('applies, expected', [('7-', 'yes'), ('7-13', 'no'), ('14', 'yes'),
    ('21-', 'no'), ('0-6', 'no'), ('unknown: open scope', 'unknown')])
def test_effective_receipt_and_nonblocking_stored_advice(workspace, tmp_path, applies, expected):
    store, registry, executor = captured(workspace, tmp_path)
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    text = registry.execute('create', declaration(references=[file_ref()], applies_at=parse_applies(applies)))
    assert f'Next weekly check if the run continues: day 14; applies: {expected}.' in text
    assert 'Stored-file predicate:' in text and 'fixed query windows' in text
    revised = registry.execute('revise', dict(record='r1', reason='Keep inherited scope and evidence'))
    assert f'day 14; applies: {expected}.' in revised and 'Stored-file predicate:' in revised
    history = json.loads(registry.path.read_text())['records']['r1']
    assert history[-1]['references'] == history[0]['references']
    assert history[-1]['applies_at'] == history[0]['applies_at']
    executor.pf_queries.stale_checks = False
    assert 'Automatic PF weekly checks: disabled.' in registry.execute('revise', dict(record='r1', reason='Disabled'))
    executor.pf_queries.stale_checks = True
    registry.sim_day = lambda: None
    assert 'Next automatic PF weekly check: unknown.' in registry.execute('revise', dict(record='r1', reason='Unknown day'))
    assert 'Stored-file predicate:' not in registry.execute('create', declaration(references=[file_ref('historical_only')]))


@pytest.mark.parametrize('mode', ['git', 'prefix'])
def test_non_pf_receipt_never_claims_automatic_pf(workspace, tmp_path, mode):
    store, registry, executor = captured(workspace, tmp_path, mode=mode)
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    text = registry.execute('create', declaration(references=[file_ref()]))
    assert 'PF weekly check' not in text
    assert 'Stored-file predicate:' not in text


def test_all_skips_persist_and_old_visibility_is_readable(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    for purpose, applies in [('historical_only', '0-'), ('current', '0-6'), ('current', '21-'), ('current', '0-')]:
        call(registry, 'create', **declaration(references=[file_ref(purpose)], applies_at=parse_applies(applies)))
    call(registry, 'retire', record='r4', reason='Done')
    digest = executor.weekly_check(14)
    event, result = weekly_event(store)
    assert {c['record']: c['check']['reason'] for c in result['conditions']} == {
        'r1.1': 'historical_only', 'r2.1': 'applies_ended', 'r3.1': 'not_yet_applicable', 'r4.2': 'retired'}
    assert all(c['source_version'] and c['computation_day'] is None for c in result['conditions'])
    assert all(c['status'] == 'skipped' and c['checked_day'] is None and c['weekly_day'] == 14
               and not c['digest_ranges'] for c in result['conditions'])
    assert 'past the applies window' in digest
    assert executor.weekly_check(21) is not None  # r3 becomes eligible
    call(registry, 'retire', record='r3', reason='Done')
    assert executor.weekly_check(28) is None
    event, result = weekly_event(store)
    assert len(result['conditions']) == 4 and result['digest_version'] is None
    assert not any(v['rendered'] or v['delivered_requests'] for v in weekly_condition_visibility(store, event))
    old = store.begin_event('pf_weekly_check', {'day': 0})
    store.complete(old, checked_texts=0)
    assert weekly_condition_visibility(store, old) == []


def test_distinct_conditions_same_source_visibility_omissions_and_partial_delivery(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration(references=[file_ref(value=n) for n in (8, 9, 10, 11)]))
    digest = executor.weekly_check(14)
    event, result = weekly_event(store)
    conditions = result['conditions']
    assert [c['reference_index'] for c in conditions] == [0, 1, 2, 3]
    assert len({c['condition_id'] for c in conditions}) == 4
    assert [c['condition_id'] for c in conditions] == [
        'ceo:r1.1:0', 'ceo:r1.1:1', 'ceo:r1.1:2', 'ceo:r1.1:3']
    assert len({c['source_version'] for c in conditions}) == 1
    assert [c['status'] for c in conditions] == ['holds'] * 4
    assert [bool(c['digest_ranges']) for c in conditions] == [True, True, True, False]
    for c in conditions[:3]:
        a, b = c['digest_ranges'][0]
        assert f"r1.1 reference {c['reference_index'] + 1}: evidence.json" in digest[a:b]
    send(store, digest, state='failed')
    assert not any(c['delivered_requests'] for c in weekly_condition_visibility(store, event))
    summary = str(digest).splitlines()[2]
    start = str(digest).index(summary)
    send(store, CapturedText(summary, [origin(result['digest_version'], str(digest), start, start + len(summary))]))
    assert not any(c['delivered_requests'] for c in weekly_condition_visibility(store, event))
    a, b = conditions[0]['digest_ranges'][0]
    partial = CapturedText(str(digest)[a:b-1], [origin(result['digest_version'], str(digest), a, b-1)])
    send(store, partial)
    assert not weekly_condition_visibility(store, event)[0]['delivered_requests']
    delivered = send(store, digest)
    visibility = weekly_condition_visibility(store, event)
    assert [c['delivered_requests'] for c in visibility] == [[delivered], [delivered], [delivered], []]
    (workspace / 'evidence.json').write_text('{"n":9}')
    changed_digest = executor.weekly_check(21)
    changed_event, result = weekly_event(store)
    assert [c['status'] for c in result['conditions']] == ['fails', 'fails', 'holds', 'holds']
    first = result['conditions'][0]
    assert len(first['digest_ranges']) == 2
    a, b = first['digest_ranges'][1]
    assert 'reference 1 (fails)' in changed_digest[a:b]
    delivered_line = send(store, CapturedText(str(changed_digest)[a:b],
        [origin(result['digest_version'], str(changed_digest), a, b)]))
    visibility = weekly_condition_visibility(store, changed_event)[0]
    assert visibility['delivered_requests'] == [delivered_line]
    assert [d['delivered_requests'] for d in visibility['range_deliveries']] == [[], [delivered_line]]
    (workspace / 'evidence.json').unlink()
    executor.weekly_check(28)
    _, result = weekly_event(store)
    assert [c['status'] for c in result['conditions']] == ['cannot_check'] * 4
    assert all(c['check']['reason'] == 'file_missing' for c in result['conditions'])


@pytest.mark.parametrize('phase', ['_check', '_cached_trace'])
@pytest.mark.parametrize('failure', [TimeoutError('budget'), RuntimeError('check failed')])
def test_interrupted_week_has_complete_conditions(workspace, tmp_path, monkeypatch, failure, phase):
    store, registry, executor = captured(workspace, tmp_path)
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration(references=[file_ref(), file_ref(value=9)]))
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(executor.pf_queries, phase, fail)
    with pytest.raises(type(failure), match=str(failure)):
        executor.weekly_check(14)
    event, result = weekly_event(store)
    assert result['status'] == ('timed_out' if isinstance(failure, TimeoutError) else 'failed')
    assert len(result['conditions']) == 2 and not result['check_complete']
    assert all(c['check']['reason'] == 'check_interrupted' and not c['digest_ranges'] for c in result['conditions'])
    store.assert_healthy()


def test_pending_ordinary_finding_is_not_rechecked(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration({'path': 'evidence.json'}))
    (workspace / 'evidence.json').write_text('{"n":8}')
    executor.weekly_check(14)
    first_pending = store.load_state('pf_review')
    assert first_pending['pending']
    digest = executor.weekly_check(21)
    _, result = weekly_event(store)
    assert result['conditions'][0]['check']['reason'] == 'pending_not_rechecked'
    assert result['conditions'][0]['status'] == 'skipped'
    assert result['conditions'][0]['checked_day'] is None
    assert result['conditions'][0]['current_source_version'] is None
    assert result['conditions'][0]['last_verification']['checked_day'] == 14
    assert store.load_state('pf_review') == first_pending
    assert 'not rechecked this week' in digest


@pytest.mark.parametrize('client', ['novamind-client', 'novamind-operation'])
def test_packed_wrong_pf_route_is_nonzero_and_safely_copyable(packed_public, client):
    args = ['depend', "r1; printf 'unsafe'", '$(touch unwanted)']
    result = subprocess.run([sys.executable, str(packed_public / client), 'pf', *args], text=True, capture_output=True)
    assert result.returncode == 2
    correction = result.stderr.strip().split('Run: ', 1)[1]
    assert shlex.split(correction) == ['pf', *args]
    assert result.stdout == ''


def test_explicit_unknown_is_durably_skipped_without_changing_selection(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    call(registry, 'create', **declaration(references=[dict(evidence={'unknown': 'not available'},
        purpose='current', select={'path': '/n'}, predicate={'type': 'threshold', 'op': '>', 'value': 0})]))
    digest = executor.weekly_check(14)
    _, result = weekly_event(store)
    assert result['conditions'][0]['status'] == 'skipped'
    assert result['conditions'][0]['check']['reason'] == 'explicit_unknown'
    assert digest is None


def test_csv_handle_advice_and_direct_query_has_no_advice(workspace, tmp_path, server):
    from test_pf_stale import record_sql, cite
    from saas_bench.public_sql import execute_query
    store, registry, executor = captured(workspace, tmp_path)
    (workspace / 'cost.csv').write_text('key,n\nB,7\n')
    send(store, executor.execute('read_file', {'path': 'cost.csv'}))
    text = registry.execute('create', declaration(references=[dict(evidence={'version': 'cost.csv@v1'},
        purpose='current', select={'row': {'key': 'B'}, 'col': 'n'},
        predicate={'type': 'threshold', 'op': '<', 'value': 10})]))
    assert 'Stored-file predicate:' in text
    server.sql_evidence = store
    version = record_sql(store, 'SELECT amount FROM ledger', execute_query(server, 'SELECT amount FROM ledger'))
    result = cite(store, registry, executor, version, select={'col': 'amount'},
                  predicate={'type': 'threshold', 'op': '>', 'value': 0})
    assert 'warnings' not in result


def test_actual_team_pf_prompts_receipts_and_weekly_delivery(tmp_path, monkeypatch, packed_public):
    monkeypatch.setenv('NOVAMIND_PUBLIC_DIR', str(packed_public))
    from test_preflight_team_flow import make_team, calls, final, ADVANCE
    from saas_bench.role_policy import ROLES
    monkeypatch.setattr('time.sleep', lambda _: None)
    counters = {}
    def respond(team, role, body):
        day = team.server.tools.current_day
        key = (role, day)
        n = counters[key] = counters.get(key, 0) + 1
        assert 'PF is a separate command' in body['messages'][0]['content']
        bash = next(t for t in body['tools'] if t['function']['name'] == 'bash')
        assert 'never `novamind-operation pf`' in bash['function']['description']
        rendered = json.dumps(body['messages'])
        if day == 0:
            if n == 1:
                return calls(('bash', {'command': './novamind-operation pf help'}))
            if n == 2:
                assert 'Error: PF is a separate Bash command. Run: pf help' in rendered
                return calls(('bash', {'command': 'pf help'}))
            if n == 3:
                assert 'usage: pf <command>' in body['messages'][-1]['content']
                assert team.roles[role].executor.last_status == 'succeeded'
                return calls(('write_file', {'path': 'evidence.json', 'content': '{"n":7}'}))
            if n == 4:
                return calls(('read_file', {'path': 'evidence.json'}))
            if n == 5:
                return calls(('text_create', declaration(references=[file_ref()], applies_at={'start_day': 0})))
            assert 'Next weekly check if the run continues: day 7; applies: yes.' in rendered
            assert 'Stored-file predicate:' in rendered
        else:
            assert 'checked day 7; holds on stored bytes' in rendered
        return calls(('bash', {'command': ADVANCE})) if role == 'ceo' else final(role + ' offline advice')
    with make_team('pf', respond, 'pf-weekly-ledger') as team:
        outcome = team.run(stop_after_day=14)
        assert outcome.reason == 'observation_end', team.failure
        evidence = {}
        for role in ROLES:
            store = team.roles[role].store
            with store.connect() as conn:
                rows = conn.execute("SELECT event_id FROM requests WHERE json_extract(request, '$.kind')='pf_weekly_check' AND json_extract(request, '$.role')=? ORDER BY rowid", (role,)).fetchall()
            assert len(rows) == 2
            assert store.read_event(rows[0][0])['result']['conditions'] == []
            event = rows[-1][0]
            result = store.read_event(event)['result']
            assert result['conditions'][0]['role'] == role
            assert result['conditions'][0]['status'] == 'holds'
            visibility = weekly_condition_visibility(store, event)
            assert visibility[0]['rendered'] and visibility[0]['delivered_requests']
            evidence[role] = dict(event=event, result=result, visibility=visibility)
        (team.test_root / 'weekly-ledger-acceptance.json').write_text(json.dumps(evidence, indent=2))


def test_interruption_preserves_completed_comparisons(workspace, tmp_path, monkeypatch):
    store, registry, executor = captured(workspace, tmp_path)
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration(references=[file_ref(value=8), file_ref(value=6), file_ref(value=9)]))
    actual = executor.pf_queries._check
    def partial(rows, day):
        actual(rows[:2], day=day)
        raise TimeoutError('partial budget')
    monkeypatch.setattr(executor.pf_queries, '_check', partial)
    with pytest.raises(TimeoutError, match='partial budget'):
        executor.weekly_check(14)
    _, result = weekly_event(store)
    assert not result['check_complete'] and result['digest_version'] is None
    assert [c['status'] for c in result['conditions']] == ['holds', 'fails', 'skipped']
    assert [c['checked_day'] for c in result['conditions']] == [14, 14, None]
    assert result['conditions'][2]['check']['reason'] == 'check_interrupted'
    assert not any(c['digest_ranges'] for c in result['conditions'])


def test_private_cli_wrong_pf_route_uses_same_correction(monkeypatch, capsys):
    from saas_bench.novamind_cli import operation_main
    args = ['depend', '$(touch unwanted)']
    monkeypatch.setattr(sys, 'argv', ['novamind-operation', 'pf', *args])
    with pytest.raises(SystemExit) as exc:
        operation_main()
    assert exc.value.code == 2
    assert shlex.split(capsys.readouterr().err.split('Run: ', 1)[1]) == ['pf', *args]


def test_legacy_pending_reference_indexes_are_restored(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration(references=[dict(evidence={'path': 'evidence.json'}, purpose='current'),
                                                      file_ref(value=6), file_ref(value=6)]))
    (workspace / 'evidence.json').write_text('{"n":8}')
    executor.weekly_check(14)
    pending = store.load_state('pf_review')
    for entry in pending['pending'].values():
        conditions = []
        for row in entry['conditions']:
            row['edge'].pop('reference_index', None)
            if row not in conditions:
                conditions.append(row)
        entry['conditions'] = conditions
    store.save_state('pf_review', pending)
    from saas_bench.pf_queries import PFQueries
    executor.pf_queries = PFQueries(registry)
    digest = executor.weekly_check(21)
    event, result = weekly_event(store)
    assert [c['reference_index'] for c in result['conditions']] == [0, 1, 2]
    assert [c['status'] for c in result['conditions']] == ['skipped', 'fails', 'fails']
    assert all(c['checked_day'] == 21 for c in result['conditions'][1:])
    send(store, digest)
    assert weekly_condition_visibility(store, event)[1]['delivered_requests']


@pytest.mark.parametrize('next_day', [14, 21])
def test_interrupted_continuing_condition_preserves_prior_verification(workspace, tmp_path, monkeypatch, next_day):
    store, registry, executor = captured(workspace, tmp_path)
    send(store, executor.execute('read_file', {'path': 'evidence.json'}))
    call(registry, 'create', **declaration(references=[dict(evidence={'path': 'evidence.json'}, purpose='current'),
                                                      file_ref(value=6)]))
    (workspace / 'evidence.json').write_text('{"n":8}')
    executor.weekly_check(14)
    _, first = weekly_event(store)
    assert first['conditions'][1]['status'] == 'fails'
    pending = store.load_state('pf_review')
    def fail(*args, **kwargs):
        raise TimeoutError('before comparison')
    monkeypatch.setattr(executor.pf_queries, '_check', fail)
    with pytest.raises(TimeoutError, match='before comparison'):
        executor.weekly_check(next_day)
    _, result = weekly_event(store)
    ordinary = result['conditions'][0]
    assert ordinary['check']['reason'] == 'pending_not_rechecked'
    assert ordinary['last_verification']['checked_day'] == 14
    condition = result['conditions'][1]
    assert condition['status'] == 'skipped' and condition['checked_day'] is None
    assert condition['weekly_day'] == next_day and condition['current_source_version'] is None
    assert condition['check']['reason'] == 'check_interrupted'
    assert condition['last_verification']['predicate_result'] == 'fails'
    assert condition['last_verification']['checked_day'] == 14
    assert store.load_state('pf_review') == pending


@pytest.mark.parametrize('cancel', [False, True])
def test_index_interruption_is_durable_with_unavailable_inventory(workspace, tmp_path, monkeypatch, cancel):
    from saas_bench.run_lifecycle import RunCancelled
    store, registry, executor = captured(workspace, tmp_path)
    call(registry, 'create', **declaration())
    def fail(*args, **kwargs):
        raise RunCancelled('index held') if cancel else TimeoutError('index budget')
    monkeypatch.setattr(executor.pf_queries, '_index', fail)
    with pytest.raises(RunCancelled if cancel else TimeoutError):
        executor.weekly_check(14)
    event, result = weekly_event(store)
    assert result['status'] == ('cancelled' if cancel else 'timed_out')
    assert not result['check_complete'] and not result['inventory_complete']
    assert result['coverage'] == 'unavailable' and result['conditions'] == []
    assert result['digest_version'] is None
    store.assert_healthy()


def test_enabled_empty_inventory_has_complete_no_digest_attempt(workspace, tmp_path):
    store, registry, executor = captured(workspace, tmp_path)
    assert executor.weekly_check(14) is None
    _, result = weekly_event(store)
    assert result['check_complete'] and result['inventory_complete']
    assert result['conditions'] == [] and result['digest_version'] is None
