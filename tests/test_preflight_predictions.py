from types import SimpleNamespace

from saas_bench.api_server import NovaMindAPIServer
from saas_bench.database import save_predictions
from saas_bench.scoring import interval_score, score_predictions
import pytest


@pytest.mark.parametrize('actual,expected', [(8, 4), (10, 4), (12, 4), (7, 44), (13, 44), (0, 324), (-1, 364)])
def test_interval_score(actual, expected):
    assert interval_score(8, 12, actual) == expected


def test_maturity_uses_day_even_without_new_ledger_entries(make_initialized_sim):
    conn, _, config = make_initialized_sim()
    cash = config.initial_cash
    save_predictions(conn, 0, {h: {'cash': {'point': cash, 'lower': cash - 1, 'upper': cash + 1}}
                               for h in (7, 28, 84, 182)}, 0)
    result = score_predictions(conn, 28)
    assert result['four_week_count'] == 1
    assert result['four_week_mean_interval_score'] == 2
    assert result['status_counts'] == {'pending': 2, 'scored': 2}
    for outcome, status in [('bankrupt', 'unmatured_bankruptcy'), ('failed', 'unmatured_failure')]:
        assert score_predictions(conn, 28, outcome=outcome)['status_counts'][status] == 2
    assert score_predictions(conn, 28, planned_end_day=28)['status_counts']['excluded_after_planned_end'] == 2
    assert score_predictions(conn, 6)['four_week_count'] == 0
    conn.execute('UPDATE predictions SET predicted_lower=NULL WHERE horizon_days=28')
    assert score_predictions(conn, 28)['status_counts']['invalid_prediction'] == 1


def test_zero_and_negative_cash(make_initialized_sim):
    conn, _, _ = make_initialized_sim()
    conn.execute('DELETE FROM ledger')
    save_predictions(conn, 0, {28: {'cash': {'point': 0, 'lower': -1, 'upper': 1}}}, 0)
    row = score_predictions(conn, 28)['rows'][0]
    assert row['actual_cash'] == 0 and row['absolute_percentage_error'] is None
    conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES (28,'operations',-2,'test')")
    assert score_predictions(conn, 28)['rows'][0]['interval_score'] == 42


def test_actual_cash_is_the_target_advance_before_new_week_purchases(make_initialized_sim):
    conn, _, _ = make_initialized_sim()
    conn.execute('DELETE FROM ledger')
    conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES (0,'initial_funding',100,'test')")
    conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES (7,'market_research',-5,'week 0 purchase')")
    conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES (28,'operations',-10,'day 28 of the advance')")
    # Bought on day 28 after the advance: not part of the cash the forecast targets.
    conn.execute("INSERT INTO ledger(day,category,amount,note) VALUES (28,'research_project',-40,'new week')")
    save_predictions(conn, 0, {28: {'cash': {'point': 85, 'lower': 80, 'upper': 90}}}, 0)
    row = score_predictions(conn, 28)['rows'][0]
    assert row['actual_cash'] == 85 and row['ledger_cash_through_target_day'] == 45
    assert row['covered'] and row['interval_score'] == 10


def test_prediction_failure_rolls_back_before_any_world_change(make_initialized_sim, tmp_path, monkeypatch):
    conn, simulator, _ = make_initialized_sim()
    initial_rng = simulator.rng.bit_generator.state
    def must_not_run(*args):
        raise AssertionError('World advanced after failed prediction write')
    simulator.step_week = must_not_run
    shocks = SimpleNamespace(check_and_generate_shocks=must_not_run)
    server = NovaMindAPIServer(SimpleNamespace(workspace_path=tmp_path, current_day=0),
                               simulator=simulator, conn=conn, shock_manager=shocks)
    def fail(conn, day, predictions, now):
        conn.execute("INSERT INTO predictions (submit_day,horizon_days,metric,predicted_value,submitted_at) VALUES (0,7,'cash',1,0)")
        raise OSError('injected failure after first row')
    monkeypatch.setattr('saas_bench.database.save_predictions', fail)
    result = server.advance_week({7: {'cash': 1}})
    assert result == {'success': False, 'error': 'prediction_save_failed', 'day': 0}
    assert conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0] == 0
    assert simulator.current_day == 0 and simulator.rng.bit_generator.state == initial_rng
    assert not server._operation_failed
