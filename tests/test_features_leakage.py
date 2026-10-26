"""Feature calculations and temporal-leakage guarantees."""

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from dfs.features import FEATURE_SETS, eligible_mask, sales_features
from dfs.models import recursive_xgboost_forecast
from tests.conftest import make_panel


def test_lags_and_rolling_windows_end_at_t_minus_1():
    y = np.arange(1.0, 101.0)[None, :]  # day j has value j+1
    t = 60
    f = sales_features(y[:, :t])
    assert f["lag_7"][0] == y[0, t - 7]
    assert f["lag_14"][0] == y[0, t - 14]
    assert f["lag_28"][0] == y[0, t - 28]
    assert f["roll_mean_7"][0] == pytest.approx(y[0, t - 7 : t].mean())
    assert f["roll_mean_28"][0] == pytest.approx(y[0, t - 28 : t].mean())
    # The window must not include day t itself.
    assert y[0, t] > f["roll_mean_7"][0]


def test_v2_extra_features_values():
    y = np.random.default_rng(0).poisson(5, size=(3, 120)).astype(float)
    t = 100
    f = sales_features(y[:, :t], "v2")
    assert f["roll_mean_91"] == pytest.approx(y[:, t - 91 : t].mean(axis=1))
    assert f["roll_std_28"] == pytest.approx(y[:, t - 28 : t].std(axis=1))
    assert f["roll_max_28"] == pytest.approx(y[:, t - 28 : t].max(axis=1))
    expected = y[:, [t - 7, t - 14, t - 21, t - 28]].mean(axis=1)
    assert f["same_dow_mean_4w"] == pytest.approx(expected)


@pytest.mark.parametrize("fset", ["v1", "v2"])
def test_features_at_t_ignore_sales_on_or_after_t(panel, fset):
    t = 150
    a = panel.y.copy()
    b = panel.y.copy()
    b[:, t:] = np.random.default_rng(9).uniform(1000, 2000, size=b[:, t:].shape)  # future differs wildly
    fa = panel.feature_matrix(a, t, fset)
    fb = panel.feature_matrix(b, t, fset)
    pd.testing.assert_frame_equal(fa, fb)


def test_changing_a_past_day_does_change_features(panel):
    """Sanity check that the leakage test above is not vacuous."""
    t = 150
    b = panel.y.copy()
    b[:, t - 1] += 100
    assert not panel.feature_matrix(panel.y, t).equals(panel.feature_matrix(b, t))


def test_feature_columns_match_declared_sets(panel):
    for name, cols in FEATURE_SETS.items():
        assert list(panel.feature_matrix(panel.y, 150, name).columns) == cols


def test_nan_history_propagates_instead_of_zero(panel):
    y = panel.y.copy()
    y[0, :140] = np.nan  # series 0 "launches" at day 140
    f = panel.feature_matrix(y, 150)
    assert np.isnan(f.loc[0, "lag_28"])  # day 122 is before the launch at day 140
    assert not np.isnan(f.loc[0, "lag_7"])  # day 143 is after it
    assert np.isnan(f.loc[0, "roll_mean_28"])
    assert not np.isnan(f.loc[1, "roll_mean_28"])


def _tiny_model(panel, feature_set):
    rows = []
    for t in range(60, panel.n_obs):
        x = panel.feature_matrix(panel.y, t, feature_set)
        x["target"] = panel.y[:, t]
        rows.append(x)
    df = pd.concat(rows)
    m = XGBRegressor(n_estimators=20, max_depth=3, random_state=0, n_jobs=1)
    m.fit(df[FEATURE_SETS[feature_set]], df["target"])
    return m


def test_recursive_forecast_never_reads_actuals_inside_the_horizon():
    p1 = make_panel(seed=3)
    origin, horizon = 200, 28
    model = _tiny_model(p1, "v1")
    out1 = recursive_xgboost_forecast(model, p1, origin, horizon, "v1")

    p2 = make_panel(seed=3)
    p2.y[:, origin + 1 : p2.n_obs] = np.random.default_rng(5).uniform(500, 900, (p2.n_series, p2.n_obs - origin - 1))
    out2 = recursive_xgboost_forecast(model, p2, origin, horizon, "v1")
    assert out1.shape == (p1.n_series, horizon)
    np.testing.assert_array_equal(out1, out2)


def test_recursive_forecast_depends_on_history_before_origin():
    p1 = make_panel(seed=3)
    model = _tiny_model(p1, "v1")
    base = recursive_xgboost_forecast(model, p1, 200, 28, "v1")
    p2 = make_panel(seed=3)
    p2.y[:, 190:201] += 30
    assert not np.allclose(base, recursive_xgboost_forecast(model, p2, 200, 28, "v1"))


def test_recursive_predictions_are_non_negative_and_finite():
    p = make_panel(seed=4)
    out = recursive_xgboost_forecast(_tiny_model(p, "v1"), p, 220, 28, "v1")
    assert np.isfinite(out).all() and (out >= 0).all()


def test_eligible_mask_excludes_recently_launched_series(panel):
    y = panel.y.copy()
    y[2, :150] = np.nan
    panel.y = y
    mask = eligible_mask(panel, first_origin=160, min_history=91)
    assert mask.tolist() == [True, True, False, True, True, True]
