"""Vectorized Monte Carlo inventory / stockout simulator.

Method (see README "Simulator methodology" for the full discussion)
------------------------------------------------------------------
Demand paths
    demand[p, t] = max(0, round((forecast[t] + e[p, t]) * (1 + demand_change)))
    where e[p, :] is one complete 28-day residual path (actual - forecast) drawn with replacement
    from a *pool* of validation residual paths: every item-store series in every validation fold
    that may be used (all three folds for the future forecast; strictly earlier folds in the
    backtest). Each path is rescaled to the target series' demand level:

        e = residual_path * sqrt(level_target / level_source)

    where level is the mean forecast over the 28 days (known at the forecast origin). The square
    root is the Poisson-style variance scaling (error s.d. grows with the square root of the
    level); it was fixed a priori, not tuned.

    Why whole 28-day paths from a pool: an earlier version bootstrapped 7-day blocks of the
    series' own residuals. A probability-integral-transform diagnostic on the held-out folds
    showed that 59-71% of realized 28-day demand totals fell outside the simulated 10-90% band
    (20% expected): the paths were far too narrow because forecast errors persist as a level
    shift across the whole horizon, which short blocks cannot reproduce. Whole-path resampling
    keeps that persistence; pooling across series supplies enough distinct paths.

    The simulation horizon (28 days) equals the validation horizon, so residuals are never
    extrapolated to longer horizons; residual paths are used as observed, horizon step by
    horizon step. The demand-change scenario scales forecast and noise proportionally.

Inventory policy (the only one implemented)
    Daily-review (s, S) policy with lost sales and a fixed lead time L:
      1. start of day t: orders placed on day t-L arrive and join on-hand stock;
      2. demand occurs; sales = min(on-hand, demand); the rest is LOST (no backorders);
      3. end of day: inventory position = on-hand + on-order;
         if position <= s (reorder point) order up to S = s + Q, arriving at start of day t+L.
    Q = mean daily demand * max(L, 7) (a review cycle of at least one week).
    Starting state: on-hand = initial_cover_days * baseline mean daily forecast, nothing on order.

Analytical reference (the textbook approximation)
    safety_stock  = z * sigma_daily * sqrt(L)
    reorder_point = mean_daily * L + safety_stock
    where sigma_daily is the standard deviation of the level-scaled pooled daily residuals.
    Assumes daily forecast errors are independent and identically distributed, roughly normal,
    with a constant lead time and a cycle service level. Real errors are autocorrelated and
    skewed, so the simulated service level generally differs from z.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

N_PATHS = 10_000
HORIZON = 28
LEVEL_FLOOR = 0.5
SCALE_EXPONENT = 0.5


@dataclass(frozen=True)
class SimParams:
    demand_change: float = 0.0  # -0.30 .. +0.50
    lead_time_days: int = 7  # 1 .. 28
    service_level: float = 0.95  # 0.80 .. 0.995
    initial_cover_days: float = 14.0
    n_paths: int = N_PATHS
    seed: int = 42
    current_reorder_point: float | None = None  # defaults to lead-time demand, no safety stock

    def validate(self) -> None:
        if not -0.30 <= self.demand_change <= 0.50:
            raise ValueError("demand_change must be within [-0.30, 0.50]")
        if not 1 <= self.lead_time_days <= 28:
            raise ValueError("lead_time_days must be within [1, 28]")
        if not 0.80 <= self.service_level <= 0.995:
            raise ValueError("service_level must be within [0.80, 0.995]")
        if self.initial_cover_days < 0:
            raise ValueError("initial_cover_days must be >= 0")
        if self.n_paths < 100:
            raise ValueError("n_paths must be >= 100")


def z_score(service_level: float) -> float:
    return NormalDist().inv_cdf(service_level)


def analytic_policy(mean_daily: float, sigma_daily: float, lead_time: int, service_level: float) -> dict:
    """Textbook safety stock / reorder point (assumptions in the module docstring)."""
    z = z_score(service_level)
    safety = z * sigma_daily * np.sqrt(lead_time)
    lt_demand = mean_daily * lead_time
    return {
        "z": float(z),
        "safety_stock": float(safety),
        "expected_lead_time_demand": float(lt_demand),
        "reorder_point": float(lt_demand + safety),
    }


# ----------------------------------------------------------------------------- residual pool
@dataclass(frozen=True)
class ResidualPool:
    """Validation residual paths (actual - forecast) with the demand level of their source."""

    paths: np.ndarray  # (N, H)
    levels: np.ndarray  # (N,) mean forecast over the path's days

    def __post_init__(self):
        if self.paths.ndim != 2 or len(self.paths) == 0:
            raise ValueError("residual pool must contain at least one (N, H) path")
        if len(self.levels) != len(self.paths):
            raise ValueError("levels must align with paths")

    def scale_factors(self, target_level: float, idx: np.ndarray | slice = slice(None)) -> np.ndarray:
        ratio = max(target_level, LEVEL_FLOOR) / np.maximum(self.levels[idx], LEVEL_FLOOR)
        return ratio**SCALE_EXPONENT

    def scaled(self, target_level: float) -> np.ndarray:
        """All paths rescaled to the target series' demand level."""
        return self.paths * self.scale_factors(target_level)[:, None]


def sample_residual_paths(
    pool: ResidualPool, target_level: float, n_paths: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw complete residual paths with replacement -> (n_paths, H)."""
    idx = rng.integers(0, len(pool.paths), size=n_paths)
    return pool.paths[idx] * pool.scale_factors(target_level, idx)[:, None]


# -------------------------------------------------------------------------------- inventory
def simulate_inventory(
    demand: np.ndarray,
    initial_on_hand: float,
    reorder_point: float,
    order_up_to: float,
    lead_time: int,
    record_trajectory: bool = False,
) -> dict[str, np.ndarray]:
    """Run the (s, S) lost-sales policy on every demand path at once.

    demand: (P, H) non-negative integers. Returns per-path arrays.
    """
    n_paths, horizon = demand.shape
    on_hand = np.full(n_paths, float(initial_on_hand))
    on_order = np.zeros(n_paths)
    arrivals = np.zeros((n_paths, horizon + lead_time + 1))
    lost_total = np.zeros(n_paths)
    stockout = np.zeros(n_paths, dtype=bool)
    n_orders = np.zeros(n_paths)
    traj = np.zeros((n_paths, horizon)) if record_trajectory else None

    for t in range(horizon):
        arrived = arrivals[:, t]
        on_hand += arrived
        on_order -= arrived
        sold = np.minimum(on_hand, demand[:, t])
        lost = demand[:, t] - sold
        on_hand -= sold
        lost_total += lost
        stockout |= lost > 0
        position = on_hand + on_order
        need = position <= reorder_point
        qty = np.where(need, np.maximum(order_up_to - position, 0.0), 0.0)
        arrivals[:, t + lead_time] += qty
        on_order += qty
        n_orders += qty > 0
        if traj is not None:
            traj[:, t] = on_hand
    total_demand = demand.sum(axis=1)
    return {
        "stockout": stockout,
        "lost": lost_total,
        "demand": total_demand,
        "ending_on_hand": on_hand,
        "n_orders": n_orders,
        "trajectory": traj,
    }


def _scenario_summary(res: dict[str, np.ndarray]) -> dict:
    total = res["demand"].sum()
    fill = 1.0 if total == 0 else float(1.0 - res["lost"].sum() / total)
    return {
        "stockout_probability": float(res["stockout"].mean()),
        "expected_fill_rate": fill,
        "expected_lost_units": float(res["lost"].mean()),
        "expected_ending_inventory": float(res["ending_on_hand"].mean()),
        "expected_orders": float(res["n_orders"].mean()),
    }


def simulate_series(
    forecast: np.ndarray,
    pool: ResidualPool,
    params: SimParams,
    return_trajectory: bool = False,
) -> dict:
    """Simulate one item-store: recommended policy vs. a baseline policy, same demand paths.

    forecast: (H,) point forecast; pool: validation residual paths available at the origin.
    """
    params.validate()
    rng = np.random.default_rng(params.seed)
    scale = 1.0 + params.demand_change
    baseline_mean = float(np.mean(forecast))
    noise = sample_residual_paths(pool, baseline_mean, params.n_paths, rng)
    demand = np.maximum(0.0, np.rint((forecast[None, :] + noise) * scale))

    mean_daily = baseline_mean * scale
    sigma_daily = float(np.std(pool.scaled(baseline_mean), ddof=1)) * scale
    L = params.lead_time_days
    policy = analytic_policy(mean_daily, sigma_daily, L, params.service_level)
    q = mean_daily * max(L, 7)
    initial = float(np.rint(params.initial_cover_days * baseline_mean))

    rop_rec = policy["reorder_point"]
    rop_cur = (
        params.current_reorder_point
        if params.current_reorder_point is not None
        else policy["expected_lead_time_demand"]
    )
    out = {}
    for name, rop in (("recommended", rop_rec), ("current", rop_cur)):
        res = simulate_inventory(
            demand, initial, rop, rop + q, L, record_trajectory=return_trajectory and name == "recommended"
        )
        out[name] = {"reorder_point": float(rop), "order_up_to": float(rop + q), **_scenario_summary(res)}
        if return_trajectory and name == "recommended":
            traj = res["trajectory"]
            out["trajectory_percentiles"] = {p: np.percentile(traj, p, axis=0).tolist() for p in (10, 50, 90)}
    totals = np.percentile(demand.sum(axis=1), (10, 50, 90))
    out.update(
        {
            "inputs": {
                "mean_daily_demand": mean_daily,
                "sigma_daily_error": sigma_daily,
                "initial_on_hand": initial,
                "order_quantity": float(q),
                "z": policy["z"],
            },
            "safety_stock": policy["safety_stock"],
            "reorder_point": policy["reorder_point"],
            "expected_lead_time_demand": policy["expected_lead_time_demand"],
            "simulated_lead_time_demand": float(demand[:, :L].sum(axis=1).mean()),
            "demand_total": {"p10": float(totals[0]), "p50": float(totals[1]), "p90": float(totals[2])},
            "n_paths": params.n_paths,
            "seed": params.seed,
        }
    )
    return out


# ------------------------------------------------------------------------------- backtest
BACKTEST_GRID = [(L, sl, cover) for L in (7, 14) for sl in (0.90, 0.95) for cover in (7.0, 14.0)]


def realized_outcome(
    actual: np.ndarray, initial_on_hand: float, reorder_point: float, order_up_to: float, lead_time: int
) -> dict:
    """Apply the same policy to the single realized demand path."""
    res = simulate_inventory(actual[None, :].astype(float), initial_on_hand, reorder_point, order_up_to, lead_time)
    demand = float(res["demand"][0])
    return {
        "stockout": int(res["stockout"][0]),
        "fill_rate": None if demand == 0 else float(1.0 - res["lost"][0] / demand),
    }


def backtest_series(
    forecast: np.ndarray,
    pool: ResidualPool,
    actual: np.ndarray,
    n_paths: int = N_PATHS,
    seed: int = 42,
) -> list[dict]:
    """For one series and one held-out fold: simulate with information available at the origin
    (the fold's forecast and the residual pool from EARLIER folds only), then replay the realized
    demand through the identical policy. Returns one row per (grid point, policy)."""
    rows = []
    for L, sl, cover in BACKTEST_GRID:
        p = SimParams(0.0, L, sl, cover, n_paths=n_paths, seed=seed)
        sim = simulate_series(forecast, pool, p)
        inp = sim["inputs"]
        for policy, key in (("recommended", "recommended"), ("no_safety_stock", "current")):
            s = sim[key]
            real = realized_outcome(actual, inp["initial_on_hand"], s["reorder_point"], s["order_up_to"], L)
            rows.append(
                {
                    "policy": policy,
                    "lead_time_days": L,
                    "service_level": sl,
                    "initial_cover_days": cover,
                    "pred_stockout_prob": s["stockout_probability"],
                    "realized_stockout": real["stockout"],
                    "pred_fill_rate": s["expected_fill_rate"],
                    "realized_fill_rate": real["fill_rate"],
                    "safety_stock": sim["safety_stock"] if policy == "recommended" else 0.0,
                    "reorder_point": s["reorder_point"],
                    "pred_total_p10": sim["demand_total"]["p10"],
                    "pred_total_p90": sim["demand_total"]["p90"],
                    "realized_total": float(np.sum(actual)),
                }
            )
    return rows


def calibration_summary(df, bins=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0001)) -> dict:
    """Predicted vs realized outcomes for a backtest frame (columns as in backtest_series).

    `demand_total_coverage_80`: share of realized 28-day demand totals inside the simulated
    10th-90th percentile band (80% expected if the demand paths are well calibrated).
    """
    import pandas as pd

    out = {}
    for policy, g in df.groupby("policy"):
        pred = g["pred_stockout_prob"].to_numpy()
        real = g["realized_stockout"].to_numpy()
        gb = g.assign(bin=pd.cut(g["pred_stockout_prob"], bins=list(bins), right=False))
        reliability = [
            {
                "bin": str(b),
                "n": int(len(x)),
                "mean_predicted": float(x["pred_stockout_prob"].mean()),
                "realized_frequency": float(x["realized_stockout"].mean()),
            }
            for b, x in gb.groupby("bin", observed=True)
        ]
        has_fill = g["realized_fill_rate"].notna()
        cov = None
        if {"pred_total_p10", "pred_total_p90", "realized_total"} <= set(g.columns):
            inside = (g["realized_total"] >= g["pred_total_p10"]) & (g["realized_total"] <= g["pred_total_p90"])
            cov = float(inside.mean())
        out[policy] = {
            "n": int(len(g)),
            "mean_predicted_stockout_prob": float(pred.mean()),
            "realized_stockout_frequency": float(real.mean()),
            "brier_score": float(np.mean((pred - real) ** 2)),
            "brier_score_constant_baseline": float(np.mean((real.mean() - real) ** 2)),
            "mean_predicted_fill_rate": float(g["pred_fill_rate"].mean()),
            "mean_realized_fill_rate": float(g.loc[has_fill, "realized_fill_rate"].mean()) if has_fill.any() else None,
            "demand_total_coverage_80": cov,
            "reliability": reliability,
        }
    return out
