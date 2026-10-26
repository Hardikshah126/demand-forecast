"""Rolling-origin evaluation, empirical prediction intervals and segment metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dfs import metrics
from dfs.config import HORIZON, N_FOLDS
from dfs.features import Panel
from dfs.models import (
    BASELINES,
    fit_xgboost,
    recursive_xgboost_forecast,
)

Q_LOW, Q_HIGH = 0.10, 0.90  # central 80% interval


@dataclass(frozen=True)
class Fold:
    fold: int
    origin: int  # last day index visible to the model (inclusive)
    eval_idx: np.ndarray  # day indices forecast: origin+1 .. origin+horizon


def make_folds(n_obs: int, horizon: int = HORIZON, n_folds: int = N_FOLDS) -> list[Fold]:
    """Chronological, non-overlapping, consecutive folds ending at the last observed day.

    Fold k trains on days <= origin_k and forecasts the next `horizon` days; fold k+1's origin is
    fold k's last evaluated day, so every evaluated day is also training data for later folds
    only (expanding window). With n_obs=1913, horizon=28: origins at day index 1828, 1856, 1884
    (i.e. d_1829, d_1857, d_1885 in M5 numbering).
    """
    folds = []
    for k in range(1, n_folds + 1):
        last = n_obs - 1 - (n_folds - k) * horizon
        origin = last - horizon
        folds.append(Fold(k, origin, np.arange(origin + 1, origin + horizon + 1)))
    return folds


def empirical_intervals(
    pred: np.ndarray, residuals: np.ndarray, q_low: float = Q_LOW, q_high: float = Q_HIGH
) -> tuple[np.ndarray, np.ndarray]:
    """Per-series empirical interval: pred + quantiles of that series' past residuals.

    pred: (S, H); residuals: (S, R) of (actual - pred) from strictly earlier folds.
    The lower bound is floored at zero because demand cannot be negative.
    """
    lo_q = np.quantile(residuals, q_low, axis=1)[:, None]
    hi_q = np.quantile(residuals, q_high, axis=1)[:, None]
    return np.clip(pred + lo_q, 0.0, None), np.clip(pred + hi_q, 0.0, None)


@dataclass
class ModelOutput:
    """Predictions for one model: fold -> (S, H) array. fold 0 is the future forecast."""

    preds: dict[int, np.ndarray] = field(default_factory=dict)
    lower: dict[int, np.ndarray | None] = field(default_factory=dict)
    upper: dict[int, np.ndarray | None] = field(default_factory=dict)


def run_models(
    panel: Panel,
    eligible: np.ndarray,
    train_frames: dict[str, pd.DataFrame],
    xgb_sets: dict[str, str],
    seed: int,
    horizon: int = HORIZON,
    n_folds: int = N_FOLDS,
    log=print,
) -> tuple[dict[str, ModelOutput], list[Fold]]:
    """Produce fold forecasts (1..n_folds) and the future forecast (fold 0) for every model.

    `xgb_sets` maps a model name to its feature set, e.g. {"xgboost": "v1", "xgboost_v2": "v2"}.
    Intervals for fold k use residuals of folds < k only; fold 1 therefore has none.
    """
    folds = make_folds(panel.n_obs, horizon, n_folds)
    all_folds = [*folds, Fold(0, panel.n_obs - 1, np.arange(panel.n_obs, panel.n_obs + horizon))]
    names = [*BASELINES, *xgb_sets]
    out = {n: ModelOutput() for n in names}

    for f in all_folds:
        y_hist = panel.y[:, : f.origin + 1]
        for name, fn in BASELINES.items():
            out[name].preds[f.fold] = fn(y_hist, horizon)[eligible]
        for name, fset in xgb_sets.items():
            train = train_frames[name]
            model = fit_xgboost(train[train["t"] <= f.origin], fset, seed)
            out[name].preds[f.fold] = recursive_xgboost_forecast(
                model, panel, f.origin, horizon, fset
            )[eligible]
            log(f"[models] {name}: fold {f.fold} done (origin idx {f.origin})")

    # Intervals from strictly earlier validation residuals.
    for name in names:
        res_by_fold = {
            f.fold: panel.y[eligible][:, f.eval_idx] - out[name].preds[f.fold] for f in folds
        }
        for f in all_folds:
            prior = [res_by_fold[k] for k in sorted(res_by_fold) if f.fold == 0 or k < f.fold]
            if not prior:
                out[name].lower[f.fold] = out[name].upper[f.fold] = None
                continue
            lo, hi = empirical_intervals(out[name].preds[f.fold], np.concatenate(prior, axis=1))
            out[name].lower[f.fold], out[name].upper[f.fold] = lo, hi
    return out, folds


# ------------------------------------------------------------------------ metric tables
def _horizon_bucket(h: int) -> str:
    lo = (h - 1) // 7 * 7 + 1
    return f"{lo}-{lo + 6}"


def metric_rows(
    panel: Panel, eligible: np.ndarray, outputs: dict[str, ModelOutput], folds: list[Fold]
) -> pd.DataFrame:
    """Long table of metrics. fold 0 = pooled over all validation folds."""
    series = panel.series[eligible].reset_index(drop=True)
    rows: list[dict] = []

    def add(model, fold, seg_type, seg_value, actual, pred, lo=None, hi=None):
        rows.append(
            {"model": model, "fold": fold, "segment_type": seg_type, "segment_value": seg_value,
             **metrics.summarize(actual, pred, lo, hi)}  # fmt: skip
        )

    for name, o in outputs.items():
        # Gather pooled arrays with their metadata.
        parts = []
        for f in folds:
            a = panel.y[eligible][:, f.eval_idx]
            p = o.preds[f.fold]
            lo, hi = o.lower[f.fold], o.upper[f.fold]
            h = np.broadcast_to(np.arange(1, a.shape[1] + 1), a.shape)
            dow = np.broadcast_to(panel.cal["dow"].to_numpy()[f.eval_idx], a.shape)
            parts.append((f, a, p, lo, hi, h, dow))
            add(name, f.fold, "overall", "all", a, p, lo, hi)

        A = np.concatenate([x[1] for x in parts], axis=1)
        P = np.concatenate([x[2] for x in parts], axis=1)
        H = np.concatenate([x[5] for x in parts], axis=1)
        D = np.concatenate([x[6] for x in parts], axis=1)
        # pooled interval coverage uses only folds that have intervals
        with_int = [x for x in parts if x[3] is not None]
        lo_all = np.concatenate([x[3] for x in with_int], axis=1) if with_int else None
        hi_all = np.concatenate([x[4] for x in with_int], axis=1) if with_int else None
        a_int = np.concatenate([x[1] for x in with_int], axis=1) if with_int else None
        add(name, 0, "overall", "all", A, P)
        if with_int:
            cov = metrics.summarize(a_int, a_int, lo_all, hi_all)["interval_coverage_80"]
            rows[-1]["interval_coverage_80"] = cov

        S = A.shape[0]
        for col, seg in (("store_id", "store"), ("cat_id", "category"), ("dept_id", "dept")):
            for val, idx in series.groupby(col).groups.items():
                add(name, 0, seg, str(val), A[idx], P[idx])
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for d in range(7):
            m = D == d
            add(name, 0, "weekday", weekdays[d], A[m], P[m])
        buckets = np.vectorize(_horizon_bucket)(H)
        for b in sorted(set(buckets.ravel()), key=lambda s: int(s.split("-")[0])):
            m = buckets == b
            add(name, 0, "horizon", b, A[m], P[m])
        for i in range(S):
            key = f"{series.at[i, 'item_id']}|{series.at[i, 'store_id']}"
            add(name, 0, "series", key, A[i], P[i])
    return pd.DataFrame(rows)


def residual_rows(
    panel: Panel, eligible: np.ndarray, outputs: dict[str, ModelOutput], folds: list[Fold]
) -> pd.DataFrame:
    series = panel.series[eligible].reset_index(drop=True)
    frames = []
    for name, o in outputs.items():
        for f in folds:
            res = panel.y[eligible][:, f.eval_idx] - o.preds[f.fold]
            n, h = res.shape
            level = o.preds[f.fold].mean(axis=1)  # known at the origin
            frames.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "fold": f.fold,
                        "item_id": np.repeat(series["item_id"].to_numpy(), h),
                        "store_id": np.repeat(series["store_id"].to_numpy(), h),
                        "horizon": np.tile(np.arange(1, h + 1), n),
                        "residual": res.ravel(),
                        "level": np.repeat(level, h),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def forecast_rows(
    panel: Panel, eligible: np.ndarray, outputs: dict[str, ModelOutput], folds: list[Fold],
    horizon: int = HORIZON,
) -> pd.DataFrame:  # fmt: skip
    series = panel.series[eligible].reset_index(drop=True)
    all_folds = [*folds, Fold(0, panel.n_obs - 1, np.arange(panel.n_obs, panel.n_obs + horizon))]
    frames = []
    for name, o in outputs.items():
        for f in all_folds:
            p = o.preds[f.fold]
            n, h = p.shape
            actual = (
                panel.y[eligible][:, f.eval_idx] if f.fold > 0 else np.full((n, h), np.nan)
            )
            lo, hi = o.lower[f.fold], o.upper[f.fold]
            frames.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "fold": f.fold,
                        "item_id": np.repeat(series["item_id"].to_numpy(), h),
                        "store_id": np.repeat(series["store_id"].to_numpy(), h),
                        "date": np.tile(panel.dates[f.eval_idx].date, n),
                        "horizon": np.tile(np.arange(1, h + 1), n),
                        "y_pred": p.ravel(),
                        "lower_80": np.full(n * h, np.nan) if lo is None else lo.ravel(),
                        "upper_80": np.full(n * h, np.nan) if hi is None else hi.ravel(),
                        "actual": actual.ravel(),
                    }
                )
            )
    out = pd.concat(frames, ignore_index=True)
    out["actual"] = out["actual"].astype("Int64")
    return out
