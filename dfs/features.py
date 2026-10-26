"""Feature construction that is safe by construction against temporal leakage.

The single entry point for sales-derived features, `sales_features(y_past)`, receives ONLY the
sales history known at the forecast origin (columns up to and including day t-1) and returns the
features for day t. It cannot read day t or later because those columns are never passed in.
During recursive forecasting the unknown days inside the horizon are filled with the model's own
predictions, never with actuals.

Calendar, SNAP, event and price columns are known-in-advance covariates for day t and are looked
up by column t (see `Panel.covariates_at`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES_V1: list[str] = [
    "lag_7", "lag_14", "lag_28", "roll_mean_7", "roll_mean_28",
    "dow", "month", "day_of_month", "is_weekend", "event_flag", "event_type", "snap",
    "sell_price", "price_change_1w", "price_missing",
    "item_idx", "store_idx", "dept_idx", "cat_idx",
]  # fmt: skip

# Extra sales-derived features for the controlled experiment (xgboost_v2). Defined here so the
# training frame and the recursive loop use exactly the same code.
EXTRA_V2: list[str] = ["roll_mean_91", "roll_std_28", "same_dow_mean_4w", "roll_max_28"]
FEATURES_V2: list[str] = FEATURES_V1 + EXTRA_V2

FEATURE_SETS = {"v1": FEATURES_V1, "v2": FEATURES_V2}
MIN_HISTORY = 91  # days of history required before a row may be used by any feature set


def sales_features(y_past: np.ndarray, feature_set: str = "v1") -> dict[str, np.ndarray]:
    """Sales-derived features for the day that follows `y_past` (shape: series x days).

    NaN (pre-launch) values propagate: a rolling mean over a window that touches a missing day is
    missing, matching the SQL (`COUNT(*) = window` guard in sql/features.sql).
    """
    out = {
        "lag_7": y_past[:, -7],
        "lag_14": y_past[:, -14],
        "lag_28": y_past[:, -28],
        "roll_mean_7": y_past[:, -7:].mean(axis=1),
        "roll_mean_28": y_past[:, -28:].mean(axis=1),
    }
    if feature_set == "v2":
        if y_past.shape[1] >= 91:
            out["roll_mean_91"] = y_past[:, -91:].mean(axis=1)
        else:
            out["roll_mean_91"] = np.full(y_past.shape[0], np.nan)
        out["roll_std_28"] = y_past[:, -28:].std(axis=1)
        out["same_dow_mean_4w"] = y_past[:, [-7, -14, -21, -28]].mean(axis=1)
        out["roll_max_28"] = y_past[:, -28:].max(axis=1)
    return out


@dataclass
class Panel:
    """Dense (series x day) view of the data plus known-in-advance covariates.

    Column index j corresponds to `dates[j]`. `y` is NaN before an item-store's launch and after
    the last observed day (`n_obs` columns are observed).
    """

    series: pd.DataFrame  # one row per series: item_id, store_id, dept_id, cat_id, *_idx
    dates: pd.DatetimeIndex
    y: np.ndarray  # (S, T_all) float64
    n_obs: int
    sell_price: np.ndarray  # (S, T_all)
    price_change_1w: np.ndarray
    snap: np.ndarray
    cal: pd.DataFrame  # date-indexed: dow, month, day_of_month, is_weekend, event_flag, event_type

    @property
    def n_series(self) -> int:
        return len(self.series)

    def covariates_at(self, t: int) -> dict[str, np.ndarray]:
        c = self.cal.iloc[t]
        n = self.n_series
        price = self.sell_price[:, t]
        out = {k: np.full(n, c[k], dtype=float) for k in
               ("dow", "month", "day_of_month", "is_weekend", "event_flag", "event_type")}  # fmt: skip
        out["snap"] = self.snap[:, t]
        out["sell_price"] = price
        out["price_change_1w"] = self.price_change_1w[:, t]
        out["price_missing"] = np.isnan(price).astype(float)
        for k in ("item_idx", "store_idx", "dept_idx", "cat_idx"):
            out[k] = self.series[k].to_numpy(dtype=float)
        return out

    def feature_matrix(self, y_known: np.ndarray, t: int, feature_set: str = "v1") -> pd.DataFrame:
        """Features for day t given sales history `y_known` with columns < t (anything else is
        ignored: we slice, so later columns can never leak in)."""
        feats = sales_features(y_known[:, :t], feature_set)
        feats.update(self.covariates_at(t))
        return pd.DataFrame({k: feats[k] for k in FEATURE_SETS[feature_set]})


def eligible_mask(panel: Panel, first_origin: int, min_history: int = MIN_HISTORY) -> np.ndarray:
    """Series with a full, gap-free history of at least `min_history` days before the first
    forecast origin. All models are evaluated on exactly this set."""
    window = panel.y[:, first_origin - min_history + 1 : first_origin + 1]
    return ~np.isnan(window).any(axis=1)


def extra_training_features(panel: Panel, feature_set: str) -> pd.DataFrame:
    """Sales-derived columns beyond features_daily's base set, for every (series, day) row."""
    if feature_set == "v1":
        return pd.DataFrame()
    rows = []
    for t in range(MIN_HISTORY, panel.n_obs):
        f = sales_features(panel.y[:, :t], feature_set)
        part = pd.DataFrame({k: f[k] for k in EXTRA_V2})
        part["series_pos"] = np.arange(panel.n_series)
        part["t"] = t
        rows.append(part)
    return pd.concat(rows, ignore_index=True)
