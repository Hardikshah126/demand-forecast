"""PostgreSQL-backed checks on a synthetic dataset in a throw-away schema.

Covers: idempotent schema, SQL data-quality checks (pass and fail), SQL-vs-Python feature
equivalence, end-to-end persistence of a training run and the API reading it back.
Skipped automatically when PostgreSQL is unreachable.
"""

import numpy as np
import pandas as pd
import pytest

from dfs.api.main import create_app
from dfs.api.repository import PostgresRepository
from dfs.data import load_panel, load_training_frame
from dfs.db import connect, run_sql_file
from dfs.features import FEATURE_SETS
from dfs.ingest import load_database
from dfs.quality import run_sql_checks
from tests.conftest import make_calendar

pytestmark = pytest.mark.db

N_DAYS, N_FUTURE = 260, 28
LAUNCH = {("I1", "S1"): 0, ("I1", "S2"): 0, ("I2", "S1"): 30, ("I2", "S2"): 150}  # last: too new to evaluate


def synthetic_tables(seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cal = make_calendar(N_DAYS + N_FUTURE)
    obs_dates = cal["date"].iloc[:N_DAYS]
    sales, prices = [], []
    for (item, store), start in LAUNCH.items():
        level = rng.uniform(3, 10)
        for i in range(start, N_DAYS):
            wk = 1 + 0.3 * np.sin(2 * np.pi * i / 7)
            sales.append((item, store, obs_dates.iloc[i], int(rng.poisson(level * wk))))
        for wk_id in sorted(cal["wm_yr_wk"].unique()):
            first_week = cal.loc[cal.wm_yr_wk == wk_id, "date"].min()
            if first_week < cal["date"].iloc[start] - pd.Timedelta(days=6):
                continue
            if (item, store) == ("I1", "S1") and wk_id == 11010:
                continue  # a week with no price: "not on sale", must stay NULL (never 0)
            prices.append((item, store, wk_id, 2.0 + (0.5 if wk_id > 11020 else 0.0)))
    return {
        "dim_item": pd.DataFrame({"item_id": ["I1", "I2"], "dept_id": "D1", "cat_id": ["C1", "C2"]}),
        "dim_store": pd.DataFrame({"store_id": ["S1", "S2"], "state_id": ["CA", "TX"]}),
        "dim_date": cal.rename(columns={}),
        "fact_sales": pd.DataFrame(sales, columns=["item_id", "store_id", "date", "sales"]),
        "fact_price": pd.DataFrame(prices, columns=["item_id", "store_id", "wm_yr_wk", "sell_price"]),
    }


@pytest.fixture(scope="module")
def loaded(pg_schema_url):
    tables = synthetic_tables()
    with connect(pg_schema_url) as conn:
        run_sql_file(conn, "schema.sql")
        counts = load_database(conn, tables)
        conn.commit()
    return pg_schema_url, tables, counts


def test_schema_is_idempotent(loaded):
    url, _, _ = loaded
    with connect(url) as conn:
        run_sql_file(conn, "schema.sql")
        run_sql_file(conn, "schema.sql")
        conn.commit()


def test_rows_loaded(loaded):
    _, tables, counts = loaded
    assert counts["fact_sales"] == len(tables["fact_sales"])
    assert counts["dim_date"] == N_DAYS + N_FUTURE


def _checks(url):
    with connect(url) as conn:
        last = conn.execute("SELECT MAX(date) FROM fact_sales").fetchone()[0]
        return {r.name: r for r in run_sql_checks(conn, n_series=4, expected_last_date=last)}


def test_sql_quality_checks_pass_on_clean_data(loaded):
    results = _checks(loaded[0])
    failed = {n: r.value for n, r in results.items() if not r.passed}
    assert not failed, failed


def test_missing_price_week_is_null_not_zero(loaded):
    with connect(loaded[0]) as conn:
        rows = conn.execute(
            """SELECT COUNT(*) FILTER (WHERE price_missing = 1), COUNT(*) FILTER (WHERE sell_price = 0)
               FROM features_daily WHERE item_id = 'I1' AND store_id = 'S1'"""
        ).fetchone()
    assert rows[0] == 7 and rows[1] == 0


def test_quality_checks_detect_injected_violations(loaded):
    url = loaded[0]
    with connect(url) as conn:
        conn.execute("DELETE FROM features_daily WHERE item_id='I1' AND store_id='S1' AND date = (SELECT MIN(date)+100 FROM fact_sales)")
        conn.execute("DELETE FROM features_daily WHERE item_id='I1' AND store_id='S1' AND date = (SELECT MIN(date)+50 FROM fact_sales)")
        conn.execute("DELETE FROM fact_sales WHERE item_id='I1' AND store_id='S1' AND date = (SELECT MIN(date)+50 FROM fact_sales)")
        conn.execute("UPDATE fact_sales SET sales = -3 WHERE item_id='I2' AND store_id='S1' AND date = (SELECT MAX(date) FROM fact_sales)")
        conn.execute("UPDATE fact_price SET sell_price = 0.01 WHERE item_id='I2' AND store_id='S2' AND wm_yr_wk = 11030")
        last = conn.execute("SELECT MAX(date) FROM fact_sales").fetchone()[0]
        res = {r.name: r for r in run_sql_checks(conn, n_series=5, expected_last_date=last)}
        conn.rollback()  # leave the shared fixture data untouched
    assert res["negative_sales"].value == 1
    assert res["missing_dates_within_series"].value == 1
    assert res["unexpected_series_count"].value == 1
    assert res["features_row_count_mismatch"].value >= 1


def test_duplicate_key_is_rejected_by_primary_key(loaded):
    import psycopg

    with connect(loaded[0]) as conn:
        with pytest.raises(psycopg.errors.UniqueViolation):
            conn.execute(
                "INSERT INTO fact_sales SELECT item_id, store_id, date, sales FROM fact_sales LIMIT 1"
            )
        conn.rollback()


def test_sql_features_equal_python_features(loaded):
    """features_daily (SQL window functions) must equal the Python recursive-path features."""
    url, _, _ = loaded
    with connect(url) as conn:
        panel = load_panel(conn)
        frame = load_training_frame(conn, panel, "v1")
    cols = FEATURE_SETS["v1"]
    checked = 0
    for t in (40, 90, 150, 200, 259):
        py = panel.feature_matrix(panel.y, t, "v1")
        sql_rows = frame[frame["t"] == t].set_index("series_pos").sort_index()
        for pos in sql_rows.index:
            a = sql_rows.loc[pos, cols].to_numpy(dtype=float)
            b = py.loc[pos, cols].to_numpy(dtype=float)
            np.testing.assert_allclose(a, b, equal_nan=True, err_msg=f"series {pos}, t={t}")
            checked += 1
    assert checked >= 15


def test_sql_features_have_no_lookahead(loaded):
    """Recompute lag_7 / roll_mean_7 from fact_sales independently and compare to the table."""
    with connect(loaded[0]) as conn:
        bad = conn.execute(
            """WITH s AS (
                 SELECT item_id, store_id, date, sales,
                        ROW_NUMBER() OVER (PARTITION BY item_id, store_id ORDER BY date) rn
                 FROM fact_sales)
               SELECT COUNT(*) FROM features_daily f
               JOIN s ON s.item_id=f.item_id AND s.store_id=f.store_id AND s.date=f.date
               JOIN s p ON p.item_id=s.item_id AND p.store_id=s.store_id AND p.rn = s.rn - 7
               WHERE f.lag_7 IS DISTINCT FROM p.sales::float8"""
        ).fetchone()[0]
    assert bad == 0


# ------------------------------------------------------------------ end-to-end persistence
@pytest.fixture(scope="module")
def trained(loaded, monkeypatch_module):
    from dfs.pipeline import run_training

    url = loaded[0]
    monkeypatch_module.setenv("DATABASE_URL", url)
    run_id = run_training(seed=3, notes="pytest", with_experiment=True, backtest_paths=300, log=lambda *_: None)
    return url, run_id


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_training_persists_all_artifacts(trained):
    url, run_id = trained
    with connect(url) as conn:
        run = conn.execute("SELECT n_series, best_model, horizon FROM model_runs WHERE run_id=%s", [run_id]).fetchone()
        assert run[0] == 3 and run[2] == 28  # the series launched on day 150 is excluded (too new)
        models = {r[0] for r in conn.execute("SELECT DISTINCT model FROM forecasts WHERE run_id=%s", [run_id])}
        assert models == {"naive", "seasonal_naive", "moving_average_28", "xgboost", "xgboost_v2"}
        n = conn.execute("SELECT COUNT(*) FROM forecasts WHERE run_id=%s AND model='xgboost'", [run_id]).fetchone()[0]
        assert n == 3 * 28 * 4  # series x horizon x (3 folds + future)
        fut = conn.execute(
            "SELECT COUNT(*) FILTER (WHERE actual IS NOT NULL), MIN(date) FROM forecasts WHERE run_id=%s AND fold=0",
            [run_id],
        ).fetchone()
        assert fut[0] == 0  # the future has no actuals
        f1 = conn.execute(
            "SELECT COUNT(*) FILTER (WHERE lower_80 IS NOT NULL) FROM forecasts WHERE run_id=%s AND fold=1", [run_id]
        ).fetchone()[0]
        assert f1 == 0  # no earlier residuals for fold 1
        assert conn.execute("SELECT COUNT(*) FROM validation_residuals WHERE run_id=%s", [run_id]).fetchone()[0] > 0
        assert conn.execute("SELECT COUNT(*) FROM simulator_backtest WHERE run_id=%s", [run_id]).fetchone()[0] > 0
        best = conn.execute(
            """SELECT model FROM metrics WHERE run_id=%s AND fold=0 AND segment_type='overall'
               ORDER BY wape LIMIT 1""",
            [run_id],
        ).fetchone()[0]
        assert best == run[1]  # best model is derived from the stored metrics


def test_forecast_intervals_and_actuals_are_consistent(trained):
    url, run_id = trained
    with connect(url) as conn:
        bad = conn.execute(
            "SELECT COUNT(*) FROM forecasts WHERE run_id=%s AND lower_80 > upper_80", [run_id]
        ).fetchone()[0]
        assert bad == 0
        # fold actuals in `forecasts` must be the true sales
        mismatch = conn.execute(
            """SELECT COUNT(*) FROM forecasts f JOIN fact_sales s
                 ON s.item_id=f.item_id AND s.store_id=f.store_id AND s.date=f.date
               WHERE f.run_id=%s AND f.fold>0 AND f.actual <> s.sales""",
            [run_id],
        ).fetchone()[0]
        assert mismatch == 0


def test_api_serves_persisted_results(trained):
    url, run_id = trained
    repo = PostgresRepository(url, max_size=2)
    repo.open()
    try:
        from fastapi.testclient import TestClient

        with TestClient(create_app(repo)) as client:
            assert client.get("/health").json() == {"status": "ok", "database": "ok", "latest_run_id": run_id}
            series = client.get("/series").json()
            assert series["total"] == 3
            item = series["items"][0]
            fc = client.get("/forecast", params={"item": item["item_id"], "store": item["store_id"], "fold": 3}).json()
            assert set(fc["models"]) >= {"naive", "xgboost"}
            assert fc["models"]["xgboost"][0]["actual"] is not None
            with connect(url) as conn:
                db_val = conn.execute(
                    """SELECT y_pred FROM forecasts WHERE run_id=%s AND model='xgboost' AND fold=3
                       AND item_id=%s AND store_id=%s ORDER BY date LIMIT 1""",
                    [run_id, item["item_id"], item["store_id"]],
                ).fetchone()[0]
            assert fc["models"]["xgboost"][0]["y_pred"] == pytest.approx(db_val)
            metrics = client.get("/metrics").json()
            with connect(url) as conn:
                db_best = conn.execute("SELECT best_model FROM model_runs WHERE run_id=%s", [run_id]).fetchone()[0]
            assert metrics["run"]["best_model"] == db_best and metrics["rows"]
            top = client.get("/errors/top", params={"n": 2, "min_volume": 0}).json()
            assert len(top["items"]) == 2
            sim = client.post("/simulate", json={"item": item["item_id"], "store": item["store_id"]})
            assert sim.status_code == 200, sim.text
            assert sim.json()["n_paths"] == 10_000
            bt = client.get("/simulation/backtest").json()
            assert bt["by_policy"]["recommended"]["n"] > 0
            assert client.get("/forecast", params={"item": "I2", "store": "S2"}).status_code == 404  # excluded series
    finally:
        repo.close()
