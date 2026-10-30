from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

from saas_bench.api_server import NovaMindAPIServer
from saas_bench.business_ledger import BusinessOperation, ensure_schema
from saas_bench.config import BenchmarkConfig
from saas_bench.database import add_ledger_entry, init_database
from saas_bench.tools import AgentTools


@pytest.fixture
def world(tmp_path):
    conn = init_database(Path(':memory:'))
    tools = AgentTools(conn, 0, tmp_path, config=BenchmarkConfig())
    server = NovaMindAPIServer(tools, conn=conn)
    yield server
    conn.close()


def test_old_world_inception_and_recovery_reference_uniqueness(world):
    conn = world.conn
    for table in ('business_effects', 'business_operations', 'business_ledger_metadata'):
        conn.execute('DROP TABLE ' + table)
    conn.commit()
    ensure_schema(conn, 28)
    assert conn.execute('SELECT inception_day FROM business_ledger_metadata').fetchone()[0] == 28
    assert conn.execute('SELECT COUNT(*) FROM business_operations').fetchone()[0] == 0
    saved = sqlite3.connect(':memory:')
    conn.backup(saved)
    world.execute_tool('set_promotion', {'global_promotion': 1})
    abandoned = conn.execute('SELECT operation_ref FROM business_operations').fetchone()[0]
    saved.backup(conn)
    world.execute_tool('set_promotion', {'global_promotion': 1})
    retained = conn.execute('SELECT operation_ref,operation_seq FROM business_operations').fetchone()
    assert retained[0] != abandoned and retained[1] == 1
    saved.close()


def test_runtime_commit_failure_restores_memory_and_sql(world):
    conn = world.conn
    old = deepcopy(vars(world.tools.config))
    conn.execute("CREATE TEMP TRIGGER reject_effect BEFORE INSERT ON business_effects BEGIN SELECT RAISE(ABORT,'ledger write failed'); END")
    with pytest.raises(sqlite3.IntegrityError, match='ledger write failed'):
        world.execute_tool('set_promotion', {'global_promotion': 50})
    assert vars(world.tools.config) == old
    assert conn.execute('SELECT COUNT(*) FROM config_overrides').fetchone()[0] == 0
    assert conn.execute('SELECT COUNT(*) FROM business_effects').fetchone()[0] == 0
    assert not conn.in_transaction and world._operation_failed


def test_commit_adapter_covers_sql_and_context_commits(world):
    operation = BusinessOperation(world.conn, actor='ceo', initiator='host', tool='post_social_media', day=0, parameters={})
    with operation.bind(world.tools) as conn:
        conn.execute("INSERT INTO agent_social_media_posts(day,content) VALUES(0,'first')")
        conn.cursor().execute('COMMIT')
        with conn:
            conn.execute("INSERT INTO agent_social_media_posts(day,content) VALUES(0,'second')")
        with pytest.raises(RuntimeError, match='savepoints'):
            conn.execute('SAVEPOINT outermost')
    rows = world.conn.execute('SELECT effect_seq,item_index FROM business_effects ORDER BY effect_seq').fetchall()
    assert [tuple(row) for row in rows] == [(1, 0), (2, 0)]


def test_research_first_commit_survives_dirty_failure_without_hidden_values(world):
    conn = world.conn
    conn.execute("INSERT INTO group_info_levels(group_id,info_level,is_discoverable) VALUES('D_S1',0,1)")
    add_ledger_entry(conn, 0, 'initial_funding', 100000, 'test')
    conn.commit()
    world.tools.config.market_research_discover_prob = 1

    def dirty_failure(*args):
        world.tools.conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES(0,'compute',-9,'uncommitted')")
        raise RuntimeError('/private/provider/wire')

    world.tools.get_group_insights = dirty_failure
    with pytest.raises(RuntimeError, match='/private/provider'):
        world.execute_tool('research_market', {})
    assert conn.in_transaction and world._operation_failed
    row = conn.execute('SELECT outcome,result_json FROM business_operations').fetchone()
    assert tuple(row) == ('unfinished', None)
    effects = [dict(row) for row in conn.execute('SELECT * FROM business_effects')]
    assert len(effects) == 2
    assert {row['effect_seq'] for row in effects} == {1}
    assert '/private/provider' not in json.dumps(effects)
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM ledger WHERE note='uncommitted'").fetchone()[0] == 0
    assert conn.execute('SELECT COUNT(*) FROM business_effects').fetchone()[0] == 2


def test_advance_partial_day_records_real_commits(world):
    conn = world.conn
    sim = SimpleNamespace(conn=conn, current_day=0)
    world.simulator = sim

    def fail():
        sim.current_day = 1
        add_ledger_entry(sim.conn, 1, 'compute', -1, 'committed')
        sim.conn.commit()
        add_ledger_entry(sim.conn, 1, 'compute', -2, 'dirty')
        raise RuntimeError('/private/advance/details')

    sim.step_week = fail
    response = world.advance_week(rationale='test')
    assert response['success'] is False
    assert world._operation_failed and conn.in_transaction
    effects = conn.execute('SELECT kind,after_json FROM business_effects').fetchall()
    assert len(effects) == 1 and effects[0][0] == 'advance_progress'
    assert json.loads(effects[0][1]) == {'reached_day': 1, 'last_completed_day': None}
    assert conn.execute('SELECT outcome FROM business_operations').fetchone()[0] == 'unfinished'
    assert sim.conn is conn and world.tools.conn is conn and world.conn is conn


def test_all_failure_items_survive_public_result_projection(world):
    result = world.execute_tool('send_enterprise_deal', {'deals': [{'customer_id': 99999, 'price': 1}]})
    assert not result.success
    row = world.conn.execute('SELECT outcome,result_json FROM business_operations').fetchone()
    assert row[0] == 'rejected'
    assert json.loads(row[1])['data']['results'] == result.data['results']
