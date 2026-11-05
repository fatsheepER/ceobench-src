import json

from saas_bench.role_policy import ROLES
from test_preflight_team_flow import calls, make_team


def test_capture_boundary_reaches_initial_models_and_log_show_help():
    with make_team('pf', lambda *args: calls(('text_list', {})), 'capture-guidance') as team:
        for role in ROLES:
            runtime = team.roles[role]
            runtime.agent.act('Synthetic dashboard', 0, False, {'day': 0})
            body = team.test_wire[-1]['body']
            system = '\n'.join(m['content'] for m in body['messages'] if m['role'] == 'system')
            for text in (system, runtime.executor.execute('bash', {'command': 'pf log --help'}),
                         runtime.executor.execute('bash', {'command': 'pf show --help'})):
                assert 'before and after the outer Bash call' in text
                assert 'not refreshed in PF until it ends' in text
                assert 'Use handles already returned' in text
                assert 'reports/w4_calc.out' not in text
            requests = [json.loads(line)['body'] for line in runtime.usage.path.read_text().splitlines()
                        if json.loads(line)['event'] == 'http_request']
            assert requests[-1] == body


def test_capture_waits_for_outer_bash_and_next_call_reads_new_bytes():
    with make_team('pf', lambda *args: calls(('text_list', {})), 'capture-boundary') as team:
        executor = team.roles['growth'].executor
        created = executor.execute('bash', {'command':
            "mkdir -p reports; printf 'first calculation\\n' > reports/w4_calc.out; "
            "pf log reports/w4_calc.out"})
        assert 'No captured version matches' in created
        assert 'reports/w4_calc.out@v1' in created
        assert 'first calculation' in executor.execute('bash', {
            'command': 'pf show growth:reports/w4_calc.out@v1 --full'})
        overwritten = executor.execute('bash', {'command':
            "printf 'revised calculation\\n' > reports/w4_calc.out; "
            "pf show reports/w4_calc.out --full"})
        assert 'first calculation' in overwritten
        assert 'revised calculation' not in overwritten
        assert 'reports/w4_calc.out@v2' in overwritten
        assert 'revised calculation' in executor.execute('bash', {
            'command': 'pf show growth:reports/w4_calc.out@v2 --full'})
