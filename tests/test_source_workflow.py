from contextlib import closing
import json
from pathlib import Path
import re
import shlex

import pytest

from saas_bench.config import BenchmarkConfig
from test_preflight_team_flow import ADVANCE, calls, final, make_team
from test_receipt_outputs import artifact
from test_relation_batch import create


SQL = {
    'Q0': 'SELECT COUNT(*) AS entries FROM ledger',
    'Q1': 'SELECT day,category,SUM(amount) AS total FROM ledger GROUP BY day,category ORDER BY day,category',
    'Q2': 'SELECT day,price_A FROM config_history ORDER BY day DESC LIMIT 1',
}


def bash(command):
    return calls(('bash', {'command': command}))


def python(code, public=False):
    return bash(('./novamind-operation python-c ' if public else 'python -c ') + shlex.quote(code))


def output(body):
    return next(message['content'] for message in reversed(body['messages']) if message['role'] == 'tool')


def marked(text, marker):
    match = re.search(re.escape(marker) + r' (\{[^\n]*\})', text)
    assert match, text
    return json.loads(match[1])


def source_cite(bound):
    return bound['ref']['cite']


def detail_pages(command):
    body = yield bash(command)
    pieces = []
    while True:
        text = output(body)
        match = re.search(r'Detail characters (\d+):(\d+)/(\d+)[^\n]*\n', text)
        assert match, text
        start, end, total = map(int, match.groups())
        assert start == sum(map(len, pieces))
        pieces.append(text[match.end():match.end() + end - start])
        if end == total:
            return json.loads(''.join(pieces))
        cursor = re.search(r'More from the same snapshot: pf more (c\d+)', text)
        assert cursor, text
        body = yield bash('pf more ' + cursor[1])


def query_action(label, sql_label=None):
    code = 'import json; from pathlib import Path; import novamind_api as nm; '
    code += 'result=nm.query(' + repr(SQL[sql_label or label]) + '); '
    code += 'Path(' + repr(label + '.json') + ').write_text(json.dumps(result)); '
    code += 'print(' + repr('MATERIAL ' + label + ' ') + ' + json.dumps(result))'
    return python(code, public=True)


def handoff(body):
    for message in reversed(body['messages']):
        if 'SOURCE-HANDOFF ' in (message.get('content') or ''):
            return marked(message['content'], 'SOURCE-HANDOFF')
    raise AssertionError('Exact source handoff did not enter this role input')


def reader_command(mode, cite, body):
    if mode == 'pf':
        return 'pf show ' + shlex.quote(cite) + ' --full'
    owner, rest = cite.split(':', 1)
    path, commit = rest.rsplit('@', 1)
    assert re.fullmatch('[0-9a-f]{7,40}', commit), cite
    line = re.search(r'Readable workspaces: ([^\n]+)', body['messages'][0]['content'])
    workspaces = line[1].split(', ')
    workspace, = [workspace for workspace in workspaces if Path(workspace).name == owner]
    return shlex.join(['git', '-C', workspace, 'show', commit + ':' + path])


def exact_json(text, mode):
    if mode == 'pf':
        header, text = text.split('\n', 1)
        extent = json.loads(header)
        assert extent['range'] == [0, extent['total_chars']] and extent['next_cursor'] is None
    return json.JSONDecoder().raw_decode(text)[0]


@pytest.mark.parametrize('mode', ['git', 'pf'])
def test_public_sources_survive_revision_overwrite_and_next_week_handoff(mode):
    observed = {}

    def growth():
        body = yield query_action('Q0')
        observed['Q0'] = marked(output(body), 'MATERIAL Q0')
        earlier = dict(day=0, path='Q0.json', sql=SQL['Q0'])
        body = yield calls(('write_file', {'path': 'MEMORY.md',
            'content': 'EARLIER-SOURCE ' + json.dumps(earlier)}))
        body = yield final('D0 ledger count is saved in Q0.json for later comparison.')
        earlier = marked(body['messages'][0]['content'], 'EARLIER-SOURCE')
        assert earlier['day'] == 0
        materials = {}
        for label, sql_label in [('Q1', 'Q1'), ('Q2', 'Q2'), ('Q1-repeat', 'Q1')]:
            body = yield query_action(label, sql_label)
            materials[label] = marked(output(body), 'MATERIAL ' + label)
            assert materials[label]['success'] and not materials[label].get('truncated')
        assert materials['Q1'] == materials['Q1-repeat']
        observed['materials'] = materials
        sources = {label: 'growth:' + label + '.json' for label in ('Q1', 'Q2')}
        if mode == 'pf':
            body = yield bash('pf candidates --history --sql-accesses --table ledger --column category')
            assert 'authorization observations' in output(body)
            command = next(line.strip() for line in output(body).splitlines()
                           if line.strip().startswith('pf candidates '))
            body = yield bash(command)
            assert 'not proof of use' in output(body)
            cursor = re.search(r'Exact events[^\n]+: pf more (c\d+)', output(body))[1]
            matches = yield from detail_pages('pf more ' + cursor)
            repeated = [item for item in matches if item['name'] == SQL['Q1']]
            assert len(repeated) == 2 and len({item['event_ref'] for item in repeated}) == 2
            assert len({item['version'] for item in repeated}) == 1
            items = yield from detail_pages('pf candidates --kind sql --detail')
            assert len(items) == 3 and all(item['day'] == 7 for item in items)
            for label in sources:
                matches = [item for item in items if item['name'] == SQL[label]]
                assert len(matches) == (2 if label == 'Q1' else 1)
                sources[label] = matches[0]['version']
            observed['sql_candidates'] = items
        code = "import json; from pathlib import Path; "
        code += "q1=json.loads(Path('Q1.json').read_text()); q2=json.loads(Path('Q2.json').read_text()); "
        code += "data={'F1':{'day':7,'categories':q1['rows']},'F2':{'day':7,'configuration':q2['rows']}}; "
        code += "[Path(name+'.json').write_text(json.dumps(value)) for name,value in data.items()]; print('PROJECTED '+json.dumps(data))"
        body = yield python(code)
        projected = marked(output(body), 'PROJECTED')
        assert projected == {'F1': {'day': 7, 'categories': materials['Q1']['rows']},
                             'F2': {'day': 7, 'configuration': materials['Q2']['rows']}}
        targets = {name: 'growth:' + name + '.json' for name in projected}
        if mode == 'pf':
            for name in targets:
                version = re.search(r'(?<!\w)(' + name + r'\.json@v\d+)', output(body))
                assert version, output(body)
                targets[name] = 'growth:' + version[1]
        missing = create('missing', targets['F2'], [])
        missing['sources'] = [{'ref': {'missing': {'description': 'Unsaved earlier forecast',
            'reason': 'No saved bytes or query receipt exists.', 'last_known_ref': 'unavailable-forecast'}}}]
        body = yield calls(('relation_batch', {'items': [
            create('F1', targets['F1'], [sources['Q1']]),
            create('F2', targets['F2'], [sources['Q2']]),
            create('invalid', targets['F2'], ['growth:never-saved.json']), missing]}))
        receipt = json.loads(output(body))
        assert (receipt['status'], receipt['saved'], receipt['rejected']) == ('partial', 3, 1)
        entries = {item['item_id']: item for item in receipt['items']}
        assert entries['invalid']['status'] == 'rejected'
        assert entries['missing']['sources'][0]['ref']['source_state'] == 'missing'
        assert entries['missing']['sources'][0]['ref']['readable'] is False
        first = entries['F1']
        refs = {'F1': source_cite(first['target']), 'F2': source_cite(entries['F2']['target']),
                'Q1': source_cite(first['sources'][0]['ref']),
                'Q2': source_cite(entries['F2']['sources'][0]['ref'])}
        summary = dict(day=7, earlier=earlier, refs=refs, relation=first['relation'],
                       first_revision=first['relation_revision'], revision=first['relation_revision'])
        body = yield python("from pathlib import Path; "
            "[Path(name+'.json').write_text('OVERWRITTEN') for name in ['Q1','Q2','F1','F2']]; "
            "print('OVERWRITE COMPLETE')")
        assert 'OVERWRITE COMPLETE' in output(body)
        observed.update(batch=receipt, projected=projected)
        body = yield final('D7 observed materials; forecast unavailable. SOURCE-HANDOFF ' + json.dumps(summary))
        assert 'ADD-EARLIER-Q0' in body['messages'][-1]['content']
        summary = handoff(body)
        earlier = marked(body['messages'][0]['content'], 'EARLIER-SOURCE')
        if mode == 'pf':
            items = yield from detail_pages('pf candidates --history --kind sql --text ' + shlex.quote(earlier['sql']) + ' --detail')
            item, = items
            assert item['day'] == earlier['day'] == 0
            q0 = item['version']
            body = yield bash('pf show ' + shlex.quote(q0) + ' --full')
        else:
            body = yield bash(shlex.join(['git', 'log', '-1', '--format=%H', '--', earlier['path']]))
            commit = output(body).strip()
            assert re.fullmatch('[0-9a-f]{40}', commit), output(body)
            q0 = 'growth:' + earlier['path'] + '@' + commit
            body = yield bash(reader_command(mode, q0, body))
        recovered_q0 = exact_json(output(body), mode)
        assert recovered_q0['success'] and recovered_q0['rows']
        body = yield calls(('relation_batch', {'items': [dict(item_id='add-Q0', op='revise',
            relation=summary['relation'], expected_revision=summary['revision'],
            sources=[{'ref': {'cite': summary['refs']['Q1']}}, {'ref': {'cite': q0}}],
            reason='CEO requested the earlier D0 count; retain D7 Q1.')]}))
        revision = json.loads(output(body))
        assert revision['saved'] == 1 and revision['items'][0]['revision'] == 2
        revised = revision['items'][0]
        summary['refs']['Q0'] = source_cite(revised['sources'][1]['ref'])
        summary['revision'] = revised['relation_revision']
        observed.update(revision=revision, recovered_q0=recovered_q0, handoff=summary)
        yield final('Added D0 Q0 without replacing D7 Q1. SOURCE-HANDOFF ' + json.dumps(summary))

    def ceo():
        body = yield bash(ADVANCE)
        initial = handoff(body)
        assert initial['day'] == 7 and 'Q0' not in initial['refs']
        body = yield calls(('ask_analyst', {'role': 'growth',
            'message': 'ADD-EARLIER-Q0: Recover the D0 count and add it to F1 using a full Q1+Q0 revision; return exact receipt locators.'}))
        assert 'Added D0 Q0 without replacing D7 Q1' in output(body)
        summary = handoff(body)
        assert summary['revision'] != initial['revision']
        assert summary['first_revision'] == initial['revision']
        refs = summary['refs']
        if mode == 'pf':
            for target, source, excluded in [('F1', 'Q1', 'Q2'), ('F2', 'Q2', 'Q1')]:
                body = yield bash('pf depend ' + shlex.quote(refs[target]))
                assert refs[source] in output(body) and refs[excluded] not in output(body)
                cursor = re.search(r'Complete reasons[^\n]+: pf more (c\d+)', output(body))[1]
                forward = yield from detail_pages('pf more ' + cursor)
                available = {source_cite(item['source']) for item in forward['items'] if item['source']['source_state'] != 'missing'}
                assert available == ({refs['Q1'], refs['Q0']} if target == 'F1' else {refs['Q2']})
                if target == 'F2':
                    assert 'unavailable-forecast' in json.dumps(forward)
                body = yield bash('pf rdepend ' + shlex.quote(refs[source]))
                assert refs[target] in output(body)
                assert refs['F2' if target == 'F1' else 'F1'] not in output(body)
            history = yield from detail_pages('pf depend ' + shlex.quote(refs['F1']) + ' --all-revisions --detail')
            old = [item for item in history['items'] if item['relation_revision'] == summary['first_revision']]
            current = [item for item in history['items'] if item['relation_revision'] == summary['revision']]
            assert [source_cite(item['source']) for item in old] == [refs['Q1']]
            assert {source_cite(item['source']) for item in current} == {refs['Q1'], refs['Q0']}
        else:
            line = re.search(r'Readable workspaces: ([^\n]+)', body['messages'][0]['content'])[1]
            peer, = [path for path in line.split(', ') if Path(path).name == 'growth']
            code = 'import json; from pathlib import Path; history=json.loads(Path(' + repr(peer + '/documents/relations.json') + ').read_text())["records"]; '
            code += 'current=[versions[-1] for versions in history.values()]; refs=' + repr(refs) + '; '
            code += 'print(json.dumps(dict(history=history,forward={name:[r for r in current if r["target"]["ref"]["cite"]==refs[name]] for name in ["F1","F2"]},'
            code += 'reverse={name:[r for r in current if any(s["ref"]["ref"].get("cite")==refs[name] for s in r["sources"]) ] for name in ["Q1","Q2"]})))'
            body = yield python(code)
            navigation = json.loads(output(body))
            for target, source, expected in [('F1', 'Q1', [refs['Q1'], refs['Q0']]), ('F2', 'Q2', [refs['Q2']])]:
                reverse, = navigation['reverse'][source]
                assert source_cite(reverse['target']) == refs[target]
                assert [source_cite(s['ref']) for s in reverse['sources']] == expected
                assert reverse in navigation['forward'][target]
            histories = list(navigation['history'].values())
            versions, = [versions for versions in histories if versions[-1]['relation'] == summary['relation']]
            assert [version['relation_revision'] for version in versions] == [summary['first_revision'], summary['revision']]
            assert [source_cite(s['ref']) for s in versions[0]['sources']] == [refs['Q1']]
            assert 'unavailable-forecast' in output(body)
        read = {}
        for name in ('Q0', 'Q1', 'Q2', 'F1', 'F2'):
            body = yield bash(reader_command(mode, refs[name], body))
            assert 'OVERWRITTEN' not in output(body)
            read[name] = exact_json(output(body), mode)
        assert read['F1']['categories'] == read['Q1']['rows']
        assert read['F2']['configuration'] == read['Q2']['rows']
        commands = [reader_command(mode, refs[name], body) + ' > reuse-' + name + '.json' for name in ('Q0', 'F1', 'F2')]
        code = 'import json; from pathlib import Path; raw={name:Path("reuse-"+name+".json").read_text() for name in ["Q0","F1","F2"]}; '
        code += 'data={name:json.loads(text.split("\\n",1)[1] if ' + repr(mode == 'pf') + ' else text) for name,text in raw.items()}; '
        code += 'total=sum(row["total"] for row in data["F1"]["categories"]); price=data["F2"]["configuration"][0]["price_A"]; '
        code += 'result=dict(source_day=7,earlier_source_day=0,reuse_day=7,total=total,price_A=price,entries=data["Q0"]["rows"][0]["entries"],reserve_after_one_unit=total-price,refs=' + repr(refs) + '); '
        code += 'Path("reused-budget.json").write_text(json.dumps(result)); print("DOWNSTREAM "+json.dumps(result))'
        commands.append('python -c ' + shlex.quote(code))
        body = yield bash('\n'.join(commands))
        downstream = marked(output(body), 'DOWNSTREAM')
        assert downstream['total'] == sum(row['total'] for row in read['F1']['categories'])
        assert downstream['price_A'] == read['F2']['configuration'][0]['price_A']
        assert downstream['entries'] == read['Q0']['rows'][0]['entries']
        target = 'ceo:reused-budget.json'
        if mode == 'pf':
            target = 'ceo:' + re.search(r'(reused-budget\.json@v\d+)', output(body))[1]
        body = yield calls(('relation_batch', {'items': [create('downstream', target, [refs[name] for name in ('F1', 'F2', 'Q0')])]}))
        reuse = json.loads(output(body))
        assert reuse['saved'] == 1 and all(s['ref']['readable'] for s in reuse['items'][0]['sources'])
        observed.update(read=read, downstream=downstream, reuse=reuse)
        yield bash(ADVANCE)

    scripts = {'growth': growth(), 'ceo': ceo()}
    started = set()

    def respond(_team, role, body):
        if role == 'ops_finance':
            return final('No additional action.')
        if role not in started:
            started.add(role)
            return next(scripts[role])
        return scripts[role].send(body)

    with make_team(mode, respond, 'source-workflow') as team:
        result = team.run(stop_after_day=14)
        usages = {role: [json.loads(line) for line in runtime.usage.path.read_text().splitlines()]
                  for role, runtime in team.roles.items()}
        audits = {role: [json.loads(line) for line in runtime.audit.path.read_text().splitlines()]
                  if runtime.audit.path.exists() else [] for role, runtime in team.roles.items()}
        pf_events = {}
        if mode == 'pf':
            for role, runtime in team.roles.items():
                with closing(runtime.store.connect()) as conn:
                    events = conn.execute("""SELECT requests.request, results.record FROM requests
                        LEFT JOIN results USING(event_id) WHERE json_extract(request,'$.role')=?
                        AND substr(json_extract(request,'$.kind'),1,3)='pf_' ORDER BY requests.rowid""", (role,)).fetchall()
                pf_events[role] = [dict(request=json.loads(row[0]), result=json.loads(row[1]) if row[1] else None)
                                   for row in events]
        artifact('source-workflow.json', dict(reason=result.reason, failure=team.failure,
            wire=team.test_wire, usage=usages, audit=audits, pf_events=pf_events, observed=observed))
        assert result.reason == 'observation_end', team.failure
        for role, usage in usages.items():
            assert [row['body'] for row in usage if row['event'] == 'http_request'] == [
                row['body'] for row in team.test_wire if row['role'] == role]
        config = BenchmarkConfig(seed=123)
        saved = json.loads((team.roles['ceo'].workspace / 'reused-budget.json').read_text())
        assert saved == observed['downstream']
        assert (saved['total'], saved['price_A']) == (config.initial_cash, config.default_price_A)
        assert saved['reserve_after_one_unit'] == config.initial_cash - config.default_price_A
        assert saved['entries'] == 1
        assert observed['read']['Q0'] == observed['recovered_q0'] == observed['Q0']
        assert observed['read']['Q1'] == observed['materials']['Q1']
        assert observed['read']['Q2'] == observed['materials']['Q2']
        assert len([row for row in team.test_wire if row['role'] == 'ceo' and row['day'] == 7]) > 5
