import pytest

from saas_bench.config import BenchmarkConfig
from saas_bench.database import get_cash


@pytest.mark.parametrize('days', [1, 7])
def test_day_and_week_costs_include_leads_without_charging_twice(make_initialized_sim, days):
    config = BenchmarkConfig(seed=123, lead_acquisition_cost=3.5)
    config.targeted_ad_spend = {'content_marketing': {'S1': 1000.0}}
    conn, simulator, _ = make_initialized_sim(config=config)
    before = get_cash(conn)
    result = simulator.step_day() if days == 1 else simulator.step_week()
    costs = -conn.execute('SELECT COALESCE(SUM(amount), 0) FROM ledger '
                         'WHERE day BETWEEN 1 AND ? AND amount < 0', (days,)).fetchone()[0]
    leads = -conn.execute("SELECT SUM(amount) FROM ledger WHERE category='lead_acquisition_cost' "
                         'AND day BETWEEN 1 AND ?', (days,)).fetchone()[0]
    assert leads > 0
    assert leads == pytest.approx(result.new_leads * config.lead_acquisition_cost)
    assert result.total_costs == pytest.approx(costs)
    assert result.cash == pytest.approx(before + result.payments_received - costs)
