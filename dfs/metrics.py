"""Forecast accuracy metrics.

Sign convention: error = prediction - actual, so positive bias means over-forecasting.
WAPE = sum|error| / sum(actual). When sum(actual) == 0 the ratio is undefined and `None` is
returned (never 0, inf or NaN) so callers can show an explicit "undefined".
"""

from __future__ import annotations

import numpy as np


def _arrays(actual, pred) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(pred, dtype=float).ravel()
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: actual {a.shape} vs pred {p.shape}")
    if a.size == 0:
        raise ValueError("empty input")
    return a, p


def wape(actual, pred) -> float | None:
    a, p = _arrays(actual, pred)
    denom = a.sum()
    if denom == 0:
        return None
    return float(np.abs(p - a).sum() / denom)


def mae(actual, pred) -> float:
    a, p = _arrays(actual, pred)
    return float(np.abs(p - a).mean())


def rmse(actual, pred) -> float:
    a, p = _arrays(actual, pred)
    return float(np.sqrt(((p - a) ** 2).mean()))


def bias(actual, pred) -> float:
    """Mean error in units (prediction - actual)."""
    a, p = _arrays(actual, pred)
    return float((p - a).mean())


def bias_pct(actual, pred) -> float | None:
    """Total error relative to total actual demand; None if total actual is zero."""
    a, p = _arrays(actual, pred)
    denom = a.sum()
    if denom == 0:
        return None
    return float((p - a).sum() / denom)


def summarize(actual, pred, lower=None, upper=None) -> dict:
    a, p = _arrays(actual, pred)
    out = {
        "n_obs": int(a.size),
        "total_actual": float(a.sum()),
        "wape": wape(a, p),
        "mae": mae(a, p),
        "rmse": rmse(a, p),
        "bias": bias(a, p),
        "bias_pct": bias_pct(a, p),
        "abs_error": float(np.abs(p - a).sum()),
        "interval_coverage_80": None,
    }
    if lower is not None and upper is not None:
        lo = np.asarray(lower, dtype=float).ravel()
        hi = np.asarray(upper, dtype=float).ravel()
        ok = ~(np.isnan(lo) | np.isnan(hi))
        if ok.any():
            out["interval_coverage_80"] = float(((a[ok] >= lo[ok]) & (a[ok] <= hi[ok])).mean())
    return out
