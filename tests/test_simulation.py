"""Monte Carlo simulator: determinism, valid ranges, hand-checkable inventory mechanics."""

import numpy as np
import pandas as pd
import pytest

from dfs import simulation as sim
from dfs.simulation import SimParams, simulate_inventory, simulate_series


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    forecast = rng.uniform(3, 8, 28)
    pool = sim.ResidualPool(paths=rng.normal(0, 2.5, (60, 28)), levels=rng.uniform(3, 8, 60))
    return forecast, pool


# ------------------------------------------------------------------------- analytics
def test_z_scores():
    assert sim.z_score(0.95) == pytest.approx(1.6448536, rel=1e-6)
    assert sim.z_score(0.50) == pytest.approx(0.0, abs=1e-12)


def test_analytic_safety_stock_and_reorder_point_formulae():
    out = sim.analytic_policy(mean_daily=10.0, sigma_daily=4.0, lead_time=9, service_level=0.95)
    assert out["safety_stock"] == pytest.approx(1.6448536 * 4.0 * 3.0, rel=1e-6)  # sqrt(9) = 3
    assert out["expected_lead_time_demand"] == pytest.approx(90.0)
    assert out["reorder_point"] == pytest.approx(90.0 + out["safety_stock"])


# ------------------------------------------------------------------ inventory mechanics
def test_inventory_hand_checked_without_stockout():
    demand = np.full((1, 5), 10.0)
    r = simulate_inventory(demand, 25, reorder_point=15, order_up_to=40, lead_time=2, record_trajectory=True)
    # d0: 25-10=15, position 15<=15 -> order 25 (arrives d2); d1: 5; d2: +25-10=20;
    # d3: 10, position 10<=15 -> order 30; d4: 0.
    assert r["trajectory"][0].tolist() == [15, 5, 20, 10, 0]
    assert r["lost"][0] == 0 and not r["stockout"][0]
    assert r["n_orders"][0] == 2


def test_inventory_stockout_and_lost_sales_accounting():
    demand = np.full((1, 4), 10.0)
    r = simulate_inventory(demand, 15, reorder_point=5, order_up_to=30, lead_time=3, record_trajectory=True)
    # d0: 15->5, position 5<=5 -> order 25 arrives d3. d1: sells 5, loses 5 -> 0. d2: loses 10.
    # d3: +25 -> sells 10 -> 15.
    assert r["trajectory"][0].tolist() == [5, 0, 0, 15]
    assert r["lost"][0] == 15
    assert r["stockout"][0]
    assert r["demand"][0] == 40


def test_replenishment_respects_lead_time():
    demand = np.zeros((1, 6))
    demand[0, 0] = 10
    r = simulate_inventory(demand, 10, reorder_point=5, order_up_to=20, lead_time=4, record_trajectory=True)
    # d0: stock 0, position 0 -> order 20, arrives start of day 4.
    assert r["trajectory"][0].tolist() == [0, 0, 0, 0, 20, 20]


def test_inventory_position_prevents_duplicate_orders():
    demand = np.full((1, 6), 5.0)
    r = simulate_inventory(demand, 20, reorder_point=12, order_up_to=40, lead_time=3)
    # d1: on-hand 10 <= 12 triggers one order (arrives d4). While it is on order the inventory
    # position (on-hand + on-order) stays above the reorder point, so no second order is placed
    # even though on-hand alone is below it on d2 and d3.
    assert r["n_orders"][0] == 1


def test_no_stockout_with_huge_initial_stock():
    demand = np.random.default_rng(0).poisson(5, (200, 28)).astype(float)
    r = simulate_inventory(demand, 10_000, 0, 0, 7)
    assert not r["stockout"].any() and r["lost"].sum() == 0


# --------------------------------------------------------------------------- series sim
def test_simulation_is_deterministic_for_a_seed_and_differs_across_seeds():
    f, res = _inputs()
    a = simulate_series(f, res, SimParams(seed=1))
    b = simulate_series(f, res, SimParams(seed=1))
    c = simulate_series(f, res, SimParams(seed=2))
    assert a == b
    assert a["recommended"]["stockout_probability"] != c["recommended"]["stockout_probability"] or (
        a["recommended"]["expected_lost_units"] != c["recommended"]["expected_lost_units"]
    )


def test_probabilities_are_valid_and_uses_10000_paths_by_default():
    f, res = _inputs()
    out = simulate_series(f, res, SimParams())
    assert out["n_paths"] == 10_000
    for key in ("recommended", "current"):
        s = out[key]
        assert 0.0 <= s["stockout_probability"] <= 1.0
        assert 0.0 <= s["expected_fill_rate"] <= 1.0
        assert s["expected_lost_units"] >= 0


def test_zero_demand_series_has_no_stockouts_and_full_fill():
    out = simulate_series(np.zeros(28), sim.ResidualPool(np.zeros((3, 28)), np.ones(3)), SimParams())
    assert out["recommended"]["stockout_probability"] == 0.0
    assert out["recommended"]["expected_fill_rate"] == 1.0
    assert out["safety_stock"] == 0.0


def test_higher_service_level_raises_safety_stock_and_reorder_point():
    f, res = _inputs()
    lo = simulate_series(f, res, SimParams(service_level=0.85))
    hi = simulate_series(f, res, SimParams(service_level=0.99))
    assert hi["safety_stock"] > lo["safety_stock"]
    assert hi["reorder_point"] > lo["reorder_point"]


def test_longer_lead_time_raises_reorder_point_and_safety_stock():
    f, res = _inputs()
    a = simulate_series(f, res, SimParams(lead_time_days=3))
    b = simulate_series(f, res, SimParams(lead_time_days=21))
    assert b["reorder_point"] > a["reorder_point"]
    assert b["safety_stock"] / a["safety_stock"] == pytest.approx(np.sqrt(21 / 3))


def test_demand_increase_raises_stockout_risk_for_fixed_stock_baseline():
    f, res = _inputs()
    base = simulate_series(f, res, SimParams(demand_change=0.0, initial_cover_days=7, current_reorder_point=0))
    up = simulate_series(f, res, SimParams(demand_change=0.5, initial_cover_days=7, current_reorder_point=0))
    assert up["current"]["stockout_probability"] >= base["current"]["stockout_probability"]
    assert up["current"]["expected_fill_rate"] <= base["current"]["expected_fill_rate"]


def test_recommended_policy_is_at_least_as_safe_as_no_safety_stock():
    f, res = _inputs()
    out = simulate_series(f, res, SimParams(initial_cover_days=3, service_level=0.97))
    assert out["recommended"]["stockout_probability"] <= out["current"]["stockout_probability"]
    assert out["recommended"]["expected_fill_rate"] >= out["current"]["expected_fill_rate"]


def test_trajectory_percentiles_are_ordered():
    f, res = _inputs()
    out = simulate_series(f, res, SimParams(), return_trajectory=True)
    p = out["trajectory_percentiles"]
    assert len(p[50]) == 28
    assert all(a <= b <= c for a, b, c in zip(p[10], p[50], p[90], strict=True))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"demand_change": -0.31}, {"demand_change": 0.51}, {"lead_time_days": 0},
        {"lead_time_days": 29}, {"service_level": 0.79}, {"service_level": 0.996},
        {"initial_cover_days": -1},
    ],
)  # fmt: skip
def test_params_outside_documented_ranges_are_rejected(kwargs):
    f, res = _inputs()
    with pytest.raises(ValueError):
        simulate_series(f, res, SimParams(**kwargs))


# ---------------------------------------------------------------------- residual sampling
def test_pool_sampling_returns_whole_paths_scaled_by_level_ratio():
    paths = np.arange(56, dtype=float).reshape(2, 28) + 1
    pool = sim.ResidualPool(paths=paths, levels=np.array([4.0, 16.0]))
    out = sim.sample_residual_paths(pool, target_level=16.0, n_paths=400, rng=np.random.default_rng(0))
    assert out.shape == (400, 28)
    # path 0 is scaled by sqrt(16/4) = 2, path 1 by 1; every output row is exactly one source path
    candidates = {tuple(paths[0] * 2.0), tuple(paths[1])}
    assert {tuple(r) for r in out} <= candidates
    assert len({tuple(r) for r in out}) == 2  # both paths get drawn


def test_pool_scaling_respects_level_floor():
    pool = sim.ResidualPool(paths=np.ones((1, 28)), levels=np.array([0.0]))
    out = sim.sample_residual_paths(pool, target_level=0.0, n_paths=3, rng=np.random.default_rng(0))
    assert np.allclose(out, 1.0)  # both levels floored to the same value -> ratio 1


def test_invalid_pools_are_rejected():
    with pytest.raises(ValueError):
        sim.ResidualPool(paths=np.zeros((0, 28)), levels=np.zeros(0))
    with pytest.raises(ValueError):
        sim.ResidualPool(paths=np.zeros((2, 28)), levels=np.zeros(3))


def test_sigma_comes_from_level_scaled_pool():
    pool = sim.ResidualPool(paths=np.random.default_rng(0).normal(0, 2, (200, 28)), levels=np.full(200, 4.0))
    out = simulate_series(np.full(28, 4.0), pool, SimParams())
    assert out["inputs"]["sigma_daily_error"] == pytest.approx(np.std(pool.paths, ddof=1))
    doubled = simulate_series(np.full(28, 16.0), pool, SimParams())  # target level 4x -> sigma x2
    assert doubled["inputs"]["sigma_daily_error"] == pytest.approx(2 * np.std(pool.paths, ddof=1))


def test_simulated_demand_total_band_is_ordered():
    f, pool = _inputs()
    out = simulate_series(f, pool, SimParams())
    d = out["demand_total"]
    assert d["p10"] <= d["p50"] <= d["p90"]


# ------------------------------------------------------------------------------ backtest
def test_backtest_series_rows_and_realized_outcomes():
    f, res = _inputs()
    actual = np.random.default_rng(3).poisson(f)
    rows = sim.backtest_series(f, res, actual, n_paths=500)
    assert len(rows) == len(sim.BACKTEST_GRID) * 2
    assert {r["policy"] for r in rows} == {"recommended", "no_safety_stock"}
    for r in rows:
        assert 0 <= r["pred_stockout_prob"] <= 1
        assert r["realized_stockout"] in (0, 1)
        assert r["realized_fill_rate"] is None or 0 <= r["realized_fill_rate"] <= 1
        assert r["pred_total_p10"] <= r["pred_total_p90"]
        assert r["realized_total"] == actual.sum()


def test_realized_outcome_matches_manual_replay():
    actual = np.array([10.0, 10.0, 10.0, 10.0])
    out = sim.realized_outcome(actual, 15, 5, 30, 3)
    assert out["stockout"] == 1
    assert out["fill_rate"] == pytest.approx(1 - 15 / 40)


def test_calibration_summary_perfectly_calibrated_has_matching_means():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    df = pd.DataFrame(
        {
            "policy": "recommended", "pred_stockout_prob": p,
            "realized_stockout": (rng.uniform(size=p.size) < p).astype(int),
            "pred_fill_rate": 0.9, "realized_fill_rate": 0.9,
            "pred_total_p10": 10.0, "pred_total_p90": 20.0,
            "realized_total": np.where(np.arange(p.size) % 4 == 0, 30.0, 15.0),
        }
    )  # fmt: skip
    s = sim.calibration_summary(df)["recommended"]
    assert s["demand_total_coverage_80"] == pytest.approx(0.75)
    assert s["mean_predicted_stockout_prob"] == pytest.approx(s["realized_stockout_frequency"], abs=0.02)
    assert s["brier_score"] <= s["brier_score_constant_baseline"] + 1e-3
