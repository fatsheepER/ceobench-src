from copy import deepcopy

import pytest

from saas_bench.config import (
    BenchmarkConfig,
    GROUP_PREFERENCE_DRIFT,
    INDIVIDUAL_PREFERENCE_DRIFT,
)
from saas_bench.database import (
    get_global_drift, get_global_state, set_global_state,
    update_global_drift, update_group_drift,
)
from saas_bench.enterprise import (
    create_negotiation_thread,
    get_negotiation_state,
    get_negotiation_states_batch,
)


PLAN_CONFIG = {
    key: value
    for plan in ("A", "B", "C")
    for key, value in (
        (f"price_{plan}", 40.0),
        (f"tier_{plan}", 4),
        (f"quota_{plan}", 1_000),
    )
}


def _fixed_customer(sim, group_id):
    return {
        **sim._generate_customer_from_group(group_id),
        "steepness_left": 1.0,
        "steepness_right": 1.0,
        "q_min": 0.25,
        "q_max": 0.75,
        "c_max": 100.0,
        "willingness_to_pay": 100.0,
        "usage_demand": 1.0,
        "usage_scale": 1.0,
        "seat_count": 1,
    }


def _set_market_drift(conn, sim, group_id, *, quality=0.3, budget=20.0):
    update_global_drift(conn, 0.2)
    update_group_drift(conn, group_id, quality - 0.2, budget, sim.current_day)
    sim._cache_step_day_globals(PLAN_CONFIG)


@pytest.mark.parametrize("day, competitor", [(90, False), (61, True)])
def test_step_day_refreshes_drift_before_same_day_decisions(
    make_initialized_sim, monkeypatch, day, competitor,
):
    config = BenchmarkConfig(
        seed=123,
        competitor_events_disabled=not competitor,
        competitor_event_late_cutoff_days=0,
        competitor_event_mean_interval=1,
        competitor_event_min_interval=0,
        competitor_event_boost_min=0.3,
        competitor_event_boost_max=0.3,
        competitor_event_magnitude_scale_min=1.0,
        competitor_event_magnitude_scale_max=1.0,
    )
    conn, sim, _ = make_initialized_sim(config=config)
    sim.current_day = day - 1
    params = _fixed_customer(sim, "E1")
    customer_id = sim._create_customer(params)
    sim._create_enterprise_lead(customer_id, params)
    thread_id = conn.execute("SELECT thread_id FROM enterprise_turns LIMIT 1").fetchone()[0]
    baseline = (params["q_min"], params["q_max"], params["c_max"])

    class ReachedDecisions(Exception):
        pass

    def inspect_same_day(*_args):
        group = conn.execute(
            "SELECT drift_q_bias_total, drift_c_max_total FROM group_parameters WHERE group_id = 'E1'"
        ).fetchone()
        q_offset = get_global_drift(conn) + group[0]
        assert (q_offset if competitor else group[1]) != 0
        expected = (baseline[0] + q_offset, baseline[1] + q_offset,
                    max(15.0, baseline[2] + group[1]))
        assert sim._apply_drift_offsets("E1", *baseline) == pytest.approx(expected)
        state = get_negotiation_state(conn, thread_id)
        assert (state.q_min, state.q_max, state.c_max) == pytest.approx(expected)
        assert tuple(conn.execute(
            "SELECT q_min, q_max, c_max FROM customers WHERE customer_id = ?", (customer_id,)
        ).fetchone()) == pytest.approx(baseline)
        raise ReachedDecisions

    monkeypatch.setattr(sim, "_update_customer_satisfaction", inspect_same_day)
    with pytest.raises(ReachedDecisions):
        sim.step_day()


def test_replayed_competitor_drain_refreshes_drift_cache(make_initialized_sim):
    conn, sim, _ = make_initialized_sim()
    sim.current_day = 1
    sim._cache_step_day_globals(PLAN_CONFIG)
    set_global_state(conn, "unreleased_targeted_dev_E1", 2.0)

    sim._fire_replayed_competitor_event({
        "boost_amount": 0.3, "post_end_day": 4,
    })

    group = conn.execute(
        "SELECT drift_q_bias_total FROM group_parameters WHERE group_id = 'E1'"
    ).fetchone()[0]
    assert 0 < get_global_state(conn, "unreleased_targeted_dev_E1", 0) < 2.0
    expected = 0.3 + group
    q_min, q_max, _ = sim._apply_drift_offsets("E1", 0.25, 0.75, 100.0)
    assert (q_min, q_max) == pytest.approx((0.25 + expected, 0.75 + expected))


@pytest.mark.parametrize("group_id", ["S1", "E1"])
def test_generated_and_stored_personal_baseline_is_independent_of_market_drift(
    make_initialized_sim, group_id,
):
    conn, sim, _config = make_initialized_sim()
    rng_state = deepcopy(sim._group_rngs[group_id].bit_generator.state)
    baseline = sim._generate_customer_from_group(group_id)
    sim._group_rngs[group_id].bit_generator.state = rng_state
    _set_market_drift(conn, sim, group_id)
    generated = sim._generate_customer_from_group(group_id)

    assert [generated[key] for key in ("q_min", "q_max", "c_max")] == pytest.approx(
        [baseline[key] for key in ("q_min", "q_max", "c_max")]
    )
    customer_id = sim._create_customer(generated)
    stored = conn.execute(
        """SELECT c.q_min, c.q_max, c.c_max,
                  cs.current_q_min, cs.current_q_max, cs.current_c_max
           FROM customers c JOIN customer_state cs USING (customer_id)
           WHERE customer_id = ?""",
        (customer_id,),
    ).fetchone()
    assert tuple(stored) == pytest.approx(
        [baseline[key] for key in ("q_min", "q_max", "c_max")] * 2
    )
    if group_id == "S1":
        sim._create_subscription(customer_id, "A", 5.0)
        assert conn.execute(
            "SELECT effective_c_max FROM subscriptions WHERE customer_id = ?", (customer_id,),
        ).fetchone()[0] == pytest.approx(max(15.0, baseline["c_max"] + 20.0))


@pytest.mark.parametrize("group_id", ["S1", "E1"])
@pytest.mark.parametrize("quality_drift, accepts", [(0.3, True), (0.6, False)])
def test_batch_leads_use_effective_preferences_without_storing_market_drift(
    make_initialized_sim, monkeypatch, group_id, quality_drift, accepts,
):
    config = BenchmarkConfig(seed=123, base_product_quality=0.8, satisfaction_ema_alpha=1.0)
    config.targeted_ad_spend = {
        "content_marketing": {group_id: 100.0 if group_id == "S1" else 10_000.0}
    }
    conn, sim, _config = make_initialized_sim(config=config)
    params = _fixed_customer(sim, group_id)
    conn.execute("UPDATE group_info_levels SET info_level = 0")
    conn.execute("UPDATE group_info_levels SET info_level = 1 WHERE group_id = ?", (group_id,))
    _set_market_drift(conn, sim, group_id, quality=quality_drift)
    monkeypatch.setattr(sim, "_generate_customer_from_group", lambda _gid: dict(params))
    monkeypatch.setattr(sim, "_draw_anonymous_quality_noise", lambda: 1.0)
    monkeypatch.setattr(sim, "_get_involuntary_churn_mu", lambda _gid: 0.0)

    generated = sim._generate_new_customers(PLAN_CONFIG)

    assert generated["total_leads"] > 0
    assert generated["total_new"] == (generated["total_leads"] if accepts else 0)
    rows = conn.execute(
        """SELECT c.customer_id, c.q_min, c.q_max, c.c_max,
                  cs.current_q_min, cs.current_q_max, cs.current_c_max
           FROM customers c JOIN customer_state cs USING (customer_id)"""
    ).fetchall()
    assert len(rows) == generated["total_leads"]
    for row in rows:
        assert tuple(row)[1:] == pytest.approx((0.25, 0.75, 100.0) * 2)

    if not accepts:
        return
    if group_id == "E1":
        thread_id = conn.execute("SELECT thread_id FROM enterprise_turns LIMIT 1").fetchone()[0]
        scalar = get_negotiation_state(conn, thread_id)
        batch = get_negotiation_states_batch(conn, [thread_id])[thread_id]
        for state in (scalar, batch):
            assert (state.q_min, state.q_max, state.c_max) == pytest.approx((0.55, 1.05, 120.0))
    else:
        assert sim._process_billing_decisions(PLAN_CONFIG, 0.0, False)[0] == 0
        snapshots = conn.execute("SELECT effective_c_max FROM subscriptions").fetchall()
        assert [row[0] for row in snapshots] == pytest.approx([120.0] * len(rows))
        events = sim._update_customer_satisfaction(PLAN_CONFIG, 0.0, False)
        expected = sim._compute_satisfaction(1.0, 1.0, 120.0, 0.8, 40.0, 1.05, 0.55)
        assert [events[row[0]]["new_satisfaction"] for row in rows] == pytest.approx(
            [expected] * len(rows)
        )


@pytest.mark.parametrize("budget_drift, expected_plan", [(20.0, "A"), (-20.0, "B")])
def test_initial_plan_choice_uses_effective_budget(make_initialized_sim, budget_drift, expected_plan):
    config = BenchmarkConfig(seed=123, base_product_quality=0.8)
    conn, sim, _config = make_initialized_sim(config=config)
    params = {**_fixed_customer(sim, "S1"), "q_min": 0.05, "q_max": 0.15}
    plan_config = {**PLAN_CONFIG, "price_A": 110.0, "tier_A": 5,
                   "price_B": 20.0, "tier_B": 3, "price_C": 200.0}
    _set_market_drift(conn, sim, "S1", quality=0.0, budget=budget_drift)

    assert sim._choose_plan_for_customer_curve(params, plan_config) == expected_plan
    assert params["c_max"] == 100.0


@pytest.mark.parametrize("snapshot, personal_budget, budget_drift, expected_budget", [
    (120.0, 100.0, 40.0, 120.0),
    (None, 100.0, 20.0, 120.0),
    (None, 100.0, -200.0, 15.0),
    (None, 10.0, 0.0, 15.0),
])
def test_daily_satisfaction_uses_frozen_effective_budget_or_drifted_fallback(
    make_initialized_sim, snapshot, personal_budget, budget_drift, expected_budget,
):
    config = BenchmarkConfig(seed=123, base_product_quality=0.8, satisfaction_ema_alpha=1.0)
    conn, sim, _config = make_initialized_sim(config=config)
    customer_id = sim._create_customer({**_fixed_customer(sim, "S1"), "c_max": personal_budget})
    sim._create_subscription(customer_id, "A", 12.0)
    conn.execute(
        "UPDATE subscriptions SET effective_c_max = ?, daily_usage_rate = 1.0 WHERE customer_id = ?",
        (snapshot, customer_id),
    )
    _set_market_drift(conn, sim, "S1", budget=budget_drift)

    events = sim._update_customer_satisfaction(PLAN_CONFIG, 0.0, False)

    expected = sim._compute_satisfaction(1.0, 1.0, expected_budget, 0.8, 12.0, 1.05, 0.55)
    assert events[customer_id]["new_satisfaction"] == pytest.approx(expected)


@pytest.mark.parametrize("budget_drift, expected_budget", [(20.0, 120.0), (-200.0, 15.0)])
def test_billing_snapshots_effective_budget(make_initialized_sim, budget_drift, expected_budget):
    conn, sim, _config = make_initialized_sim()
    customer_id = sim._create_customer(_fixed_customer(sim, "S1"))
    sim._create_subscription(customer_id, "A", 5.0)
    _set_market_drift(conn, sim, "S1", budget=budget_drift)

    sim._process_billing(PLAN_CONFIG)

    assert conn.execute(
        "SELECT effective_c_max FROM subscriptions WHERE customer_id = ?", (customer_id,),
    ).fetchone()[0] == pytest.approx(expected_budget)
    assert conn.execute(
        "SELECT current_c_max FROM customer_state WHERE customer_id = ?", (customer_id,),
    ).fetchone()[0] == pytest.approx(100.0)


@pytest.mark.parametrize("thread_type", ["new_lead", "renewal", "plan_change"])
def test_enterprise_deal_snapshots_effective_budget(make_initialized_sim, thread_type):
    conn, sim, _config = make_initialized_sim()
    params = _fixed_customer(sim, "E1")
    customer_id = sim._create_customer(params)
    if thread_type == "new_lead":
        sim._create_enterprise_lead(customer_id, params)
        thread_id = conn.execute("SELECT thread_id FROM enterprise_turns LIMIT 1").fetchone()[0]
    else:
        sim._create_subscription(customer_id, "A", 40.0)
        conn.execute("UPDATE subscriptions SET effective_c_max = 80.0 WHERE customer_id = ?", (customer_id,))
        thread_id, _ = create_negotiation_thread(conn, customer_id, thread_type, sim.current_day)
    _set_market_drift(conn, sim, "E1")
    state = get_negotiation_state(conn, thread_id)

    sim._finalize_deal(thread_id, state, 40.0, {"plan": "A", "price_per_seat": 40.0})

    assert conn.execute(
        "SELECT effective_c_max FROM subscriptions WHERE customer_id = ?", (customer_id,),
    ).fetchone()[0] == pytest.approx(120.0)


def test_individual_drift_changes_personal_baseline_before_market_offset(make_initialized_sim):
    config = BenchmarkConfig(seed=123, drift_grace_period_days=0)
    conn, sim, _config = make_initialized_sim(config=config)
    customer_id = sim._create_customer(_fixed_customer(sim, "S1"))
    sim._create_subscription(customer_id, "A", 40.0)
    _set_market_drift(conn, sim, "S1")
    sim.current_day = 30

    sim._apply_preference_drift(days=30)

    personal = conn.execute(
        "SELECT current_q_min, current_q_max, current_c_max FROM customer_state WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()
    rates = INDIVIDUAL_PREFERENCE_DRIFT["S1"]
    expected_personal = (
        0.25 * (1 + rates["q_bias_drift"]) ** 30,
        0.75 * (1 + rates["q_bias_drift"]) ** 30,
        100.0 * (1 + rates["c_max_drift"]) ** 30,
    )
    assert tuple(personal) == pytest.approx(expected_personal)
    effective = sim._apply_drift_offsets("S1", *personal)
    group_rates = GROUP_PREFERENCE_DRIFT["S1"]
    market_quality = 0.3 + 30 * (config.global_q_bias_drift + group_rates.get("q_bias_drift", 0.0))
    market_budget = 20.0 + 30 * group_rates.get("c_max_drift", 0.0)
    assert effective == pytest.approx((
        expected_personal[0] + market_quality,
        expected_personal[1] + market_quality,
        expected_personal[2] + market_budget,
    ))
