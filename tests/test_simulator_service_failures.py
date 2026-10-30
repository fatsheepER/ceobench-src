"""Service outages cannot silently change the simulated world."""
from types import SimpleNamespace

import pytest
from numpy.random import default_rng

from saas_bench.config import BenchmarkConfig
from saas_bench.database import add_agent_social_post
from saas_bench.simulation import Simulator
from test_public_sql import server


def unavailable(*args, **kwargs):
    raise RuntimeError('provider exhausted retries: 429')


@pytest.mark.parametrize('kind', ['regular', 'macro', 'judge'])
def test_live_service_failure_holds_week_for_manual_resume(server, kind, caplog):
    customer = SimpleNamespace(generate_social_post=unavailable, complete_text=unavailable)
    sim = Simulator(server.conn, BenchmarkConfig(), default_rng(123), customer_simulator=customer)
    if kind == 'judge':
        server.conn.execute("INSERT INTO group_info_levels(group_id,info_level) VALUES('S1',1)")
        post = add_agent_social_post(server.conn, 0, 'A product update')
        step = lambda: sim._process_agent_social_posts({})
    else:
        regular = [dict(customer_id=1, group_id='S1', satisfaction=.5, post_type='regular', event_context='test', is_churned=False)] if kind == 'regular' else []
        macro = [dict(prompt='macro', pmi=50, customer_id=1)] if kind == 'macro' else []
        def step():
            sim._collect_social_posts_async(sim._submit_social_posts_async(regular, {}, macro))
    server.conn.commit()
    sim.step_week = step
    server.simulator = sim
    result = server.advance_week()
    assert result['success'] is False and result['error'] == 'step_week_failed'
    assert result['day'] == 0 and 'provider exhausted retries' in result['message']
    assert server._operation_failed
    assert server.tools.current_day == 0
    assert server.conn.execute('SELECT count(*) FROM social_media_posts').fetchone()[0] == 0
    if kind == 'judge':
        row = server.conn.execute('SELECT effect_by_group,views FROM agent_social_media_posts WHERE agent_post_id=?', (post,)).fetchone()
        assert row['effect_by_group'] == '{}' and row['views'] == 0
    result = server.advance_week()
    assert result['success'] is False and result['manual_resume_required'] is True
    assert result['error'] == 'previous_operation_failed' and result['day'] == 0
    assert 'HOLD for manual checkpoint resume' in caplog.text
    assert server.tools.current_day == 0
    assert server.conn.execute('SELECT count(*) FROM social_media_posts').fetchone()[0] == 0


def test_explicit_offline_template_mode_still_generates_posts(server):
    sim = Simulator(server.conn, BenchmarkConfig(), default_rng(123))
    candidate = dict(customer_id=1, group_id='S1', satisfaction=.5, post_type='regular', event_context='test', is_churned=False)
    assert sim._submit_social_posts_async([candidate], {}, []) is None
    assert server.conn.execute('SELECT count(*) FROM social_media_posts').fetchone()[0] == 1
