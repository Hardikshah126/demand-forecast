"""Chronological folds, baseline/XGBoost output shapes, intervals and metric tables."""

import numpy as np
import pandas as pd
import pytest

from dfs import evaluation
from dfs.evaluation import empirical_intervals, make_folds, metric_rows, run_models
from dfs.features import FEATURE_SETS, eligible_mask
from dfs.models import (
    moving_average_forecast,
    naive_forecast,
    seasonal_naive_forecast,
)
from tests.conftest import make_panel


# ------------------------------------------------------------------------------------ folds
def test_folds_are_chronological_consecutive_and_end_at_last_day():
    folds = make_folds(1913, 28, 3)
    assert [f.fold for f in folds] == [1, 2, 3]
    assert folds[-1].eval_idx[-1] == 1912
    for a, b in zip(folds, folds[1:], strict=False):
        assert a.eval_idx[-1] < b.eval_idx[0]  # no overlap, strictly ordered
        assert b.origin == a.eval_idx[-1]  # next origin = previous fold's last day
    for f in folds:
        assert len(f.eval_idx) == 28
        assert f.eval_idx[0] == f.origin + 1  # forecast starts right after the origin
        assert (np.diff(f.eval_idx) == 1).all()


def test_first_origin_matches_item_selection_cutoff():
    """The top-100 item ranking uses days <= d_1829 == the first fold's origin (index 1828)."""
    from dfs.ingest import FIRST_ORIGIN_DAY

    assert make_folds(1913)[0].origin + 1 == FIRST_ORIGIN_DAY


def test_every_model_sees_only_pre_origin_training_rows(monkeypatch):
    panel = make_panel(n_series=4, n_obs=260)
    seen = []

    import dfs.evaluation as ev

    real_fit = ev.fit_xgboost

    def spy(train, fset, seed):
        seen.append(int(train["t"].max()))
        return real_fit(train, fset, seed)

    monkeypatch.setattr(ev, "fit_xgboost", spy)
    frame = _training_frame(panel)
    folds = make_folds(panel.n_obs)
    el = eligible_mask(panel, folds[0].origin)
    run_models(panel, el, {"xgboost": frame}, {"xgboost": "v1"}, seed=1, log=lambda *_: None)
    # 3 folds + the future fit, in order, each strictly limited to t <= origin
    assert seen == [f.origin for f in folds] + [panel.n_obs - 1]


def _training_frame(panel):
    rows = []
    for t in range(30, panel.n_obs):
        x = panel.feature_matrix(panel.y, t, "v1")
        x["series_pos"] = np.arange(panel.n_series)
        x["t"] = t
        x["sales"] = panel.y[:, t]
        rows.append(x)
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------------------- baselines
def test_naive_repeats_last_value():
    y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 9.0]])
    out = naive_forecast(y, 5)
    assert out.shape == (2, 5)
    assert (out == np.array([[3.0], [9.0]])).all()


def test_seasonal_naive_uses_same_weekday_last_week():
    y = np.arange(1.0, 29.0)[None, :]  # 28 days, last value 28
    out = seasonal_naive_forecast(y, 28)
    assert out.shape == (1, 28)
    # day 1 after origin has the weekday of y[-7]; day 7 of y[-1]; day 8 repeats day 1's source
    assert out[0, 0] == y[0, -7]
    assert out[0, 6] == y[0, -1]
    assert out[0, 7] == y[0, -7]
    assert out[0, 27] == y[0, -1]


def test_moving_average_uses_last_28_days_only():
    y = np.concatenate([np.full(10, 1000.0), np.full(28, 2.0)])[None, :]
    assert (moving_average_forecast(y, 4) == 2.0).all()


def test_run_models_output_shapes_and_future_fold():
    panel = make_panel(n_series=5, n_obs=260)
    folds = make_folds(panel.n_obs)
    el = eligible_mask(panel, folds[0].origin)
    out, folds = run_models(
        panel, el, {"xgboost": _training_frame(panel)}, {"xgboost": "v1"}, seed=1, log=lambda *_: None
    )
    assert set(out) == {"naive", "seasonal_naive", "moving_average_28", "xgboost"}
    for o in out.values():
        for k in (0, 1, 2, 3):
            assert o.preds[k].shape == (int(el.sum()), 28)
            assert np.isfinite(o.preds[k]).all()
        assert o.lower[1] is None and o.upper[1] is None  # fold 1 has no earlier residuals
        for k in (2, 3, 0):
            assert (o.lower[k] <= o.upper[k]).all()
            assert (o.lower[k] >= 0).all()


def test_xgboost_is_deterministic_for_a_seed():
    panel = make_panel(n_series=4, n_obs=260)
    el = np.ones(4, dtype=bool)
    frame = _training_frame(panel)
    a, _ = run_models(panel, el, {"xgboost": frame}, {"xgboost": "v1"}, seed=7, log=lambda *_: None)
    b, _ = run_models(panel, el, {"xgboost": frame}, {"xgboost": "v1"}, seed=7, log=lambda *_: None)
    np.testing.assert_array_equal(a["xgboost"].preds[3], b["xgboost"].preds[3])


# ----------------------------------------------------------------------------- intervals
def test_interval_shape_order_and_floor():
    pred = np.full((3, 28), 1.0)
    res = np.random.default_rng(0).normal(0, 3, (3, 56))
    lo, hi = empirical_intervals(pred, res)
    assert lo.shape == hi.shape == (3, 28)
    assert (lo <= hi).all() and (lo >= 0).all()


def test_interval_matches_requested_quantiles():
    res = np.tile(np.arange(-50, 51, dtype=float), (1, 1))  # one series, uniform residuals
    lo, hi = empirical_intervals(np.array([[100.0]]), res)
    assert lo[0, 0] == pytest.approx(100 + np.quantile(res, 0.10))
    assert hi[0, 0] == pytest.approx(100 + np.quantile(res, 0.90))


def test_interval_empirical_coverage_is_close_to_80_on_iid_noise():
    rng = np.random.default_rng(1)
    train = rng.normal(0, 4, (200, 400))
    test = rng.normal(0, 4, (200, 400))
    lo, hi = empirical_intervals(np.full((200, 400), 100.0), train)  # far from the zero floor
    coverage = ((100.0 + test >= lo) & (100.0 + test <= hi)).mean()
    assert coverage == pytest.approx(0.80, abs=0.03)


def test_metric_rows_cover_segments_and_have_consistent_overall():
    panel = make_panel(n_series=6, n_obs=260)
    folds = make_folds(panel.n_obs)
    el = eligible_mask(panel, folds[0].origin)
    out, folds = run_models(
        panel, el, {"xgboost": _training_frame(panel)}, {"xgboost": "v1"}, seed=1, log=lambda *_: None
    )
    m = metric_rows(panel, el, out, folds)
    assert set(m.segment_type) == {"overall", "store", "category", "dept", "weekday", "horizon", "series"}
    assert set(m[m.segment_type == "weekday"].segment_value) == {
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    }  # fmt: skip
    assert set(m[m.segment_type == "horizon"].segment_value) == {"1-7", "8-14", "15-21", "22-28"}
    # pooled abs_error equals the sum over folds
    for name in out:
        sub = m[(m.model == name) & (m.segment_type == "overall")]
        per_fold = sub[sub.fold > 0].abs_error.sum()
        assert sub[sub.fold == 0].abs_error.iloc[0] == pytest.approx(per_fold)
    # all models evaluated on identical observation counts per fold
    counts = m[(m.segment_type == "overall") & (m.fold > 0)].groupby(["fold"]).n_obs.nunique()
    assert (counts == 1).all()


def test_feature_sets_have_unique_names():
    for cols in FEATURE_SETS.values():
        assert len(cols) == len(set(cols))
    assert evaluation.Q_LOW == 0.10 and evaluation.Q_HIGH == 0.90
