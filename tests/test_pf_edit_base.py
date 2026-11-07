from test_shared_workspace import shared_team


def test_stale_full_write_does_not_replace_published_peer_version():
    with shared_team('pf', workspace_protocol='snapshot-v1') as team:
        team._set_phase('analysis')
        first = team.roles['growth'].executor
        second = team.roles['ops_finance'].executor
        first.execute('write_file', dict(path='plans/price.txt', content='price=10\n'))
        second.execute('read_file', dict(path='plans/price.txt'))
        first.execute('write_file', dict(path='plans/price.txt', content='price=12\n'))
        result = second.execute('write_file', dict(path='plans/price.txt', content='price=11\n'))
        assert (team.roles['ceo'].workspace / 'plans/price.txt').read_text() == 'price=12\n', result
        assert 'needs_confirmation' in result, result
