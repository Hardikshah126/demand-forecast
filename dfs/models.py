"""Forecasting strategies. Every function sees only history up to the forecast origin."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from dfs.features import FEATURE_SETS, Panel

XGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    tree_method="hist",
)


# ------------------------------------------------------------------------------ baselines
def naive_forecast(y_hist: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last observed day."""
    return np.repeat(y_hist[:, -1:], horizon, axis=1)


def seasonal_naive_forecast(y_hist: np.ndarray, horizon: int) -> np.ndarray:
    """Same weekday in the most recent observed week: day h uses y[o + h - 7*ceil(h/7)]."""
    cols = [h - 7 * math.ceil(h / 7) - 1 for h in range(1, horizon + 1)]  # negative offsets
    return y_hist[:, cols]


def moving_average_forecast(y_hist: np.ndarray, horizon: int, window: int = 28) -> np.ndarray:
    return np.repeat(y_hist[:, -window:].mean(axis=1, keepdims=True), horizon, axis=1)


BASELINES = {
    "naive": naive_forecast,
    "seasonal_naive": seasonal_naive_forecast,
    "moving_average_28": moving_average_forecast,
}


# -------------------------------------------------------------------------------- xgboost
def fit_xgboost(train: pd.DataFrame, feature_set: str, seed: int) -> XGBRegressor:
    """Fit one global model on rows whose target day is <= the forecast origin."""
    feats = FEATURE_SETS[feature_set]
    train = train[train["lag_28"].notna()]
    model = XGBRegressor(random_state=seed, n_jobs=-1, **XGB_PARAMS)
    model.fit(train[feats], train["sales"].astype(float))
    return model


def recursive_xgboost_forecast(
    model: XGBRegressor, panel: Panel, origin: int, horizon: int, feature_set: str
) -> np.ndarray:
    """Multi-day forecast by recursion (one step at a time).

    Day origin+h is predicted from features whose sales history consists of actuals up to the
    origin plus the model's own earlier predictions. No actual from inside the horizon is read:
    the working array starts with NaN beyond the origin and is only ever filled with predictions.
    """
    work = np.full((panel.n_series, origin + 1 + horizon), np.nan)
    work[:, : origin + 1] = panel.y[:, : origin + 1]
    for h in range(1, horizon + 1):
        t = origin + h
        x = panel.feature_matrix(work, t, feature_set)
        pred = np.clip(model.predict(x), 0.0, None)
        work[:, t] = pred
    return work[:, origin + 1 :]
