"""Shared fixtures: synthetic panel, fake repository, and an isolated-schema PostgreSQL."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from dfs.config import get_settings
from dfs.features import Panel


# ------------------------------------------------------------------------- synthetic panel
def make_calendar(n_days: int, start: str = "2020-01-04") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days)
    cal = pd.DataFrame({"date": dates})
    cal["d"] = [f"d_{i + 1}" for i in range(n_days)]
    cal["wm_yr_wk"] = 11000 + np.arange(n_days) // 7
    cal["weekday"] = cal["date"].dt.day_name()
    cal["dow"] = cal["date"].dt.dayofweek.astype("int16")
    cal["month"] = cal["date"].dt.month
    cal["year"] = cal["date"].dt.year
    cal["day_of_month"] = cal["date"].dt.day.astype("int16")
    cal["event_name_1"] = None
    cal["event_type_1"] = None
    cal["event_name_2"] = None
    cal["event_type_2"] = None
    ev = np.arange(n_days) % 40 == 5
    cal.loc[ev, "event_name_1"] = "TestDay"
    cal.loc[ev, "event_type_1"] = "National"
    cal["snap_ca"] = (np.arange(n_days) % 10 < 3).astype(int)
    cal["snap_tx"] = (np.arange(n_days) % 10 < 4).astype(int)
    cal["snap_wi"] = (np.arange(n_days) % 10 < 5).astype(int)
    return cal


def make_panel(n_series: int = 6, n_obs: int = 260, horizon: int = 28, seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    n_all = n_obs + horizon
    cal = make_calendar(n_all)
    level = rng.uniform(2, 12, size=(n_series, 1))
    weekly = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_all) / 7)
    y = rng.poisson(level * weekly[None, :]).astype(float)
    y[:, n_obs:] = np.nan
    price = np.tile(rng.uniform(1, 5, size=(n_series, 1)), (1, n_all)).round(2)
    price_change = np.full_like(price, np.nan)
    price_change[:, 7:] = price[:, 7:] / price[:, :-7] - 1
    series = pd.DataFrame(
        {
            "item_id": [f"ITEM_{i % 3}" for i in range(n_series)],
            "store_id": [f"ST_{i // 3}" for i in range(n_series)],
            "dept_id": "DEPT_1",
            "cat_id": ["CAT_A" if i % 2 == 0 else "CAT_B" for i in range(n_series)],
            "state_id": "CA",
        }
    )
    for col, name in (("item_id", "item_idx"), ("store_id", "store_idx"), ("dept_id", "dept_idx"), ("cat_id", "cat_idx")):
        series[name] = pd.factorize(series[col], sort=True)[0]
    cal_cov = pd.DataFrame(
        {
            "dow": cal["dow"].to_numpy(),
            "month": cal["month"].to_numpy(),
            "day_of_month": cal["day_of_month"].to_numpy(),
            "is_weekend": (cal["dow"] >= 5).astype(int).to_numpy(),
            "event_flag": cal["event_name_1"].notna().astype(int).to_numpy(),
            "event_type": cal["event_type_1"].notna().astype(int).to_numpy() * 2,
        }
    )
    return Panel(
        series=series,
        dates=pd.DatetimeIndex(cal["date"]),
        y=y,
        n_obs=n_obs,
        sell_price=price,
        price_change_1w=price_change,
        snap=(np.arange(n_all)[None, :] % 10 < 3).astype(float).repeat(n_series, axis=0),
        cal=cal_cov,
    )


@pytest.fixture
def panel() -> Panel:
    return make_panel()


# ------------------------------------------------------------------------------ fake repo
class FakeRepository:
    """In-memory repository with two series, used to test the HTTP contract without Postgres."""

    MODELS = ["naive", "seasonal_naive", "moving_average_28", "xgboost"]
    SERIES = [("FOODS_3_090", "CA_1"), ("HOBBIES_1_001", "TX_1")]

    def __init__(self, healthy: bool = True, with_run: bool = True):
        self.healthy = healthy
        self.with_run = with_run
        self.last = date(2016, 4, 24)

    def ping(self):
        return self.healthy

    def latest_run(self):
        if not self.with_run:
            return None
        return {
            "run_id": 7,
            "created_at": datetime(2026, 1, 1, tzinfo=UTC),
            "seed": 42,
            "horizon": 28,
            "n_series": 2,
            "last_date": self.last,
            "best_model": "xgboost",
            "notes": None,
            "config": {
                "models": self.MODELS,
                "folds": [
                    {"fold": k, "origin": "2016-01-31", "eval_start": "2016-02-01", "eval_end": "2016-02-28"}
                    for k in (1, 2, 3)
                ],
                "future_start": "2016-04-25",
                "future_end": "2016-05-22",
                "interval": "per-series empirical 10th/90th percentile of earlier-fold residuals",
                "backtest_model": "xgboost",
            },
        }

    def _meta(self, item, store):
        if (item, store) not in self.SERIES:
            return None
        return {
            "item_id": item, "store_id": store, "dept_id": "FOODS_3", "cat_id": "FOODS",
            "state_id": store[:2], "first_date": date(2011, 1, 29), "last_date": self.last,
        }  # fmt: skip

    def series_meta(self, item, store):
        return self._meta(item, store)

    def list_series(self, run_id, model, store, category, search, limit, offset):
        rows = []
        for item, st in self.SERIES:
            if store and st != store:
                continue
            rows.append({**self._meta(item, st), "wape": 0.4, "avg_daily_sales_28d": 5.0})
        return len(rows), rows[offset : offset + limit]

    def actuals(self, item, store, start, end):
        d, out = start, []
        while d <= end:
            out.append({"date": d, "actual": 5})
            d += timedelta(days=1)
        return out

    def forecast_points(self, run_id, item, store, fold, models):
        if (item, store) not in self.SERIES:
            return []
        out = []
        for m in models or self.MODELS:
            for h in range(1, 29):
                out.append({
                    "model": m, "date": date(2016, 4, 24) + timedelta(days=h), "horizon": h,
                    "y_pred": 5.0 + 0.1 * h, "lower_80": 2.0, "upper_80": 9.0,
                    "actual": None if fold == 0 else 5,
                })  # fmt: skip
        return out

    def series_metrics(self, run_id, item, store):
        return [
            {"model": m, "fold": 0, "wape": 0.5, "mae": 2.0, "rmse": 3.0, "bias": 0.1, "bias_pct": 0.02, "n_obs": 84}
            for m in self.MODELS
        ]

    def metrics(self, run_id, segment_type, model, fold):
        rows = []
        for m in self.MODELS:
            if model and m != model:
                continue
            for f in (0, 1, 2, 3):
                if fold is not None and f != fold:
                    continue
                rows.append({
                    "model": m, "fold": f, "segment_type": segment_type, "segment_value": "all",
                    "n_obs": 100, "total_actual": 500.0, "wape": 0.5 if m == "xgboost" else 0.6,
                    "mae": 2.0, "rmse": 3.0, "bias": 0.1, "bias_pct": 0.02, "abs_error": 250.0,
                    "interval_coverage_80": None,
                })  # fmt: skip
        return rows

    def top_errors(self, run_id, model, n, sort_by, min_volume):
        rows = [
            {"item_id": "FOODS_3_090", "store_id": "CA_1", "dept_id": "FOODS_3", "cat_id": "FOODS",
             "wape": 0.9, "mae": 4.0, "bias_pct": 0.3, "abs_error": 300.0, "total_actual": 330.0},
            {"item_id": "HOBBIES_1_001", "store_id": "TX_1", "dept_id": "HOBBIES_1", "cat_id": "HOBBIES",
             "wape": 0.7, "mae": 1.0, "bias_pct": -0.2, "abs_error": 90.0, "total_actual": 130.0},
        ]  # fmt: skip
        return rows[:n]

    def residual_pool(self, run_id, model):
        rng = np.random.default_rng(1)
        return [
            {"item_id": item, "store_id": st, "fold": f, "horizon": h,
             "residual": float(rng.normal(0, 2)), "level": 5.0 + 0.1 * f}
            for item, st in self.SERIES
            for f in (1, 2, 3)
            for h in range(1, 29)
        ]  # fmt: skip

    def backtest_rows(self, run_id, model):
        rng = np.random.default_rng(2)
        rows = []
        for policy in ("recommended", "no_safety_stock"):
            for i in range(40):
                p = float(rng.uniform(0, 1))
                rows.append({
                    "fold": 2 + i % 2, "policy": policy, "lead_time_days": 7, "service_level": 0.95,
                    "initial_cover_days": 14.0, "pred_stockout_prob": p,
                    "realized_stockout": int(rng.uniform() < p), "pred_fill_rate": 0.95,
                    "realized_fill_rate": 0.94, "pred_total_p10": 100.0, "pred_total_p90": 180.0,
                    "realized_total": 140.0 if i % 5 else 300.0,
                })  # fmt: skip
        return rows


@pytest.fixture
def fake_repo() -> FakeRepository:
    return FakeRepository()


@pytest.fixture
def client(fake_repo):
    from fastapi.testclient import TestClient

    from dfs.api.main import create_app

    with TestClient(create_app(fake_repo)) as c:
        yield c


# ------------------------------------------------------------------ isolated PostgreSQL schema
def _db_url() -> str | None:
    return get_settings().database_url or None


@pytest.fixture(scope="session")
def pg_schema_url():
    """Create a throw-away schema (own tables/views) and yield a DATABASE_URL that targets it.

    Skips when no PostgreSQL is reachable, so the suite runs without Docker; CI provides one.
    """
    import psycopg

    url = _db_url()
    if not url:
        pytest.skip("DATABASE_URL not set")
    schema = f"test_{uuid.uuid4().hex[:10]}"
    try:
        admin = psycopg.connect(url, autocommit=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PostgreSQL not reachable: {type(exc).__name__}")
    admin.execute(f'CREATE SCHEMA "{schema}"')
    sep = "&" if "?" in url else "?"
    scoped = f"{url}{sep}options=-csearch_path%3D{schema}"
    yield scoped
    admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
    admin.close()
