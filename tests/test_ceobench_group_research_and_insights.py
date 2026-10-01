"""Regression tests for CEOBench group research and insight visibility."""

import pytest

from saas_bench.config import CUSTOMER_GROUPS
from saas_bench.database import get_cash, get_group_info_level


def test_group_research_is_delayed_and_updates_visibility_on_completion(
    make_initialized_sim,
    make_agent_tools,
):
    conn, sim, config = make_initialized_sim()
    tools = make_agent_tools(conn, config)

    cash_before = get_cash(conn)
    result = tools.research_group("S1", target_level=2)

    assert result.success
    assert get_cash(conn) == pytest.approx(cash_before - config.research_cost_level_2)
    assert get_group_info_level(conn, "S1") == 1

    pending = conn.execute(
        """
        SELECT group_id, from_level, to_level, cost, started_day,
               expected_completion_day, status
        FROM pending_group_research
        WHERE group_id = 'S1'
        """
    ).fetchone()
    assert pending["from_level"] == 1
    assert pending["to_level"] == 2
    assert pending["cost"] == pytest.approx(config.research_cost_level_2)
    assert pending["started_day"] == 0
    assert pending["expected_completion_day"] == config.group_research_delay_level_2
    assert pending["status"] == "in_progress"

    sim.current_day = config.group_research_delay_level_2 - 1
    sim._process_group_research({})
    assert get_group_info_level(conn, "S1") == 1

    sim.current_day = config.group_research_delay_level_2
    sim._process_group_research({})

    assert get_group_info_level(conn, "S1") == 2
    completed = conn.execute(
        "SELECT status FROM pending_group_research WHERE group_id = 'S1'"
    ).fetchone()
    assert completed["status"] == "completed"

    group_cfg = CUSTOMER_GROUPS["S1"]
    snapshot = conn.execute(
        """
        SELECT snapshot_day, snapshot_c_max, snapshot_q_min, snapshot_market_cap
        FROM group_insight_snapshots
        WHERE group_id = 'S1'
        """
    ).fetchone()
    assert snapshot["snapshot_day"] == config.group_research_delay_level_2
    assert snapshot["snapshot_c_max"] == pytest.approx(group_cfg.c_max_mean)
    assert snapshot["snapshot_q_min"] == pytest.approx(group_cfg.q_min_mean)
    assert snapshot["snapshot_market_cap"] == pytest.approx(
        group_cfg.base_market_cap
        * (
            1
            + group_cfg.annual_cap_growth_rate
            * config.group_research_delay_level_2
            / 365.0
        )
    )


def test_group_insights_are_deterministic_and_hide_true_internal_parameters(
    make_initialized_sim,
    make_agent_tools,
):
    conn, _sim, config = make_initialized_sim()
    tools = make_agent_tools(conn, config)

    first = tools.get_group_insights("S1")
    second = tools.get_group_insights("S1")

    assert first.success
    assert first.data == second.data
    assert first.data["group_id"] == "S1"
    assert first.data["info_level"] == 1
    assert first.data["noise"] == "±65%"
    assert first.data['snapshot_day'] == 0
    tools.set_current_day(308)
    assert tools.get_group_insights('S1').data['snapshot_day'] == 0

    estimates = first.data["estimates"]
    assert {
        "willingness_to_pay",
        "usage_volume",
        "quality_floor_q_min",
        "contract_lockin_aversion",
        "market_cap",
        "annual_market_cap_growth_rate",
    }.issubset(estimates)
    assert "c_max_mean" not in estimates
    assert "q_min_mean" not in estimates
    assert "network_influence" in first.data
    assert "reputation_influence" in first.data
