import numpy as np
import pytest

from dfs import metrics


def test_wape_known_value():
    actual = np.array([10, 20, 30])
    pred = np.array([12, 18, 33])
    assert metrics.wape(actual, pred) == pytest.approx((2 + 2 + 3) / 60)


def test_wape_perfect_forecast_is_zero():
    assert metrics.wape([1, 2, 3], [1, 2, 3]) == 0.0


def test_wape_is_undefined_when_total_actual_is_zero():
    assert metrics.wape([0, 0, 0], [1, 2, 3]) is None
    assert metrics.bias_pct([0, 0], [1, 1]) is None


def test_wape_handles_zero_sales_days_inside_a_nonzero_segment():
    actual = np.array([0, 0, 10])
    pred = np.array([1, 1, 8])
    assert metrics.wape(actual, pred) == pytest.approx(4 / 10)


def test_mae_rmse():
    actual = np.array([0.0, 0.0, 0.0, 0.0])
    pred = np.array([1.0, -1.0, 2.0, -2.0])
    assert metrics.mae(actual, pred) == pytest.approx(1.5)
    assert metrics.rmse(actual, pred) == pytest.approx(np.sqrt((1 + 1 + 4 + 4) / 4))


def test_bias_sign_positive_means_overforecast():
    assert metrics.bias([10, 10], [12, 14]) == pytest.approx(3.0)
    assert metrics.bias([10, 10], [8, 6]) == pytest.approx(-3.0)
    assert metrics.bias_pct([10, 10], [12, 14]) == pytest.approx(6 / 20)


def test_shape_mismatch_and_empty_raise():
    with pytest.raises(ValueError):
        metrics.mae([1, 2], [1])
    with pytest.raises(ValueError):
        metrics.wape([], [])


def test_summarize_includes_interval_coverage():
    actual = np.array([1.0, 5.0, 9.0, 20.0])
    pred = np.array([2.0, 5.0, 8.0, 10.0])
    s = metrics.summarize(actual, pred, lower=[0, 4, 7, 5], upper=[3, 6, 9, 15])
    assert s["interval_coverage_80"] == pytest.approx(0.75)
    assert s["n_obs"] == 4
    assert s["abs_error"] == pytest.approx(1 + 0 + 1 + 10)


def test_summarize_ignores_nan_intervals():
    s = metrics.summarize([1.0, 2.0], [1.0, 2.0], lower=[np.nan, np.nan], upper=[np.nan, np.nan])
    assert s["interval_coverage_80"] is None
