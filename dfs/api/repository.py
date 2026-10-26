"""Database access for the API. Read-only; every value is a bound parameter."""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


class Repository(Protocol):
    """The query surface the service layer depends on (also implemented by test fakes)."""

    def ping(self) -> bool: ...
    def latest_run(self) -> dict[str, Any] | None: ...
    def list_series(
        self, run_id: int, model: str, store: str | None, category: str | None,
        search: str | None, limit: int, offset: int,
    ) -> tuple[int, list[dict[str, Any]]]: ...  # fmt: skip
    def series_meta(self, item: str, store: str) -> dict[str, Any] | None: ...
    def actuals(self, item: str, store: str, start: date, end: date) -> list[dict[str, Any]]: ...
    def forecast_points(
        self, run_id: int, item: str, store: str, fold: int, models: list[str] | None
    ) -> list[dict[str, Any]]: ...
    def series_metrics(
        self, run_id: int, item: str, store: str
    ) -> list[dict[str, Any]]: ...
    def metrics(
        self, run_id: int, segment_type: str, model: str | None, fold: int | None
    ) -> list[dict[str, Any]]: ...
    def top_errors(
        self, run_id: int, model: str, n: int, sort_by: str, min_volume: float
    ) -> list[dict[str, Any]]: ...
    def residual_pool(self, run_id: int, model: str) -> list[dict[str, Any]]: ...
    def backtest_rows(self, run_id: int, model: str) -> list[dict[str, Any]]: ...


class PostgresRepository:
    def __init__(self, dsn: str, max_size: int = 5):
        self._pool = ConnectionPool(
            dsn, min_size=1, max_size=max_size, open=False,
            kwargs={"autocommit": True, "row_factory": dict_row},
        )  # fmt: skip

    def open(self) -> None:
        self._pool.open()

    def close(self) -> None:
        self._pool.close()

    def _all(self, query: str, params: list | tuple | None = None) -> list[dict[str, Any]]:
        with self._pool.connection() as conn:
            return conn.execute(query, params).fetchall()

    def _one(self, query: str, params: list | tuple | None = None) -> dict[str, Any] | None:
        with self._pool.connection() as conn:
            return conn.execute(query, params).fetchone()

    # ------------------------------------------------------------------ health / runs
    def ping(self) -> bool:
        try:
            return self._one("SELECT 1 AS ok") is not None
        except Exception:
            return False

    def latest_run(self) -> dict[str, Any] | None:
        return self._one(
            """SELECT run_id, created_at, seed, horizon, n_series, last_date, best_model, config, notes
               FROM model_runs ORDER BY run_id DESC LIMIT 1"""
        )

    # ----------------------------------------------------------------------- series
    def list_series(self, run_id, model, store, category, search, limit, offset):
        where = ["m.run_id = %(run)s", "m.model = %(model)s", "m.fold = 0", "m.segment_type = 'series'"]
        params: dict[str, Any] = {"run": run_id, "model": model, "limit": limit, "offset": offset}
        if store:
            where.append("sp.store_id = %(store)s")
            params["store"] = store
        if category:
            where.append("i.cat_id = %(cat)s")
            params["cat"] = category
        if search:
            where.append("sp.item_id ILIKE %(search)s")
            params["search"] = f"%{search}%"
        base = f"""
            FROM metrics m
            JOIN v_series_span sp ON m.segment_value = sp.item_id || '|' || sp.store_id
            JOIN dim_item i ON i.item_id = sp.item_id
            JOIN dim_store st ON st.store_id = sp.store_id
            WHERE {" AND ".join(where)}"""
        total = self._one(f"SELECT COUNT(*) AS n {base}", params)["n"]
        rows = self._all(
            f"""SELECT sp.item_id, sp.store_id, i.dept_id, i.cat_id, st.state_id,
                       sp.first_date, sp.last_date, m.wape,
                       (SELECT AVG(f.sales)::float8 FROM fact_sales f
                         WHERE f.item_id = sp.item_id AND f.store_id = sp.store_id
                           AND f.date > sp.last_date - 28) AS avg_daily_sales_28d
                {base}
                ORDER BY sp.store_id, sp.item_id
                LIMIT %(limit)s OFFSET %(offset)s""",
            params,
        )
        return total, rows

    def series_meta(self, item, store):
        return self._one(
            """SELECT sp.item_id, sp.store_id, i.dept_id, i.cat_id, st.state_id,
                      sp.first_date, sp.last_date
               FROM v_series_span sp
               JOIN dim_item i ON i.item_id = sp.item_id
               JOIN dim_store st ON st.store_id = sp.store_id
               WHERE sp.item_id = %s AND sp.store_id = %s""",
            [item, store],
        )

    def actuals(self, item, store, start, end):
        return self._all(
            """SELECT date, sales AS actual FROM fact_sales
               WHERE item_id = %s AND store_id = %s AND date >= %s AND date <= %s
               ORDER BY date""",
            [item, store, start, end],
        )

    # --------------------------------------------------------------------- forecasts
    def forecast_points(self, run_id, item, store, fold, models):
        query = """SELECT model, date, horizon, y_pred, lower_80, upper_80, actual
                   FROM forecasts
                   WHERE run_id = %(run)s AND item_id = %(item)s AND store_id = %(store)s
                     AND fold = %(fold)s"""
        params: dict[str, Any] = {"run": run_id, "item": item, "store": store, "fold": fold}
        if models:
            query += " AND model = ANY(%(models)s)"
            params["models"] = models
        return self._all(query + " ORDER BY model, date", params)

    def series_metrics(self, run_id, item, store):
        return self._all(
            """SELECT model, fold, wape, mae, rmse, bias, bias_pct, n_obs
               FROM metrics
               WHERE run_id = %s AND segment_type = 'series' AND segment_value = %s""",
            [run_id, f"{item}|{store}"],
        )

    # ------------------------------------------------------------------------ metrics
    def metrics(self, run_id, segment_type, model, fold):
        query = """SELECT model, fold, segment_type, segment_value, n_obs, total_actual, wape,
                          mae, rmse, bias, bias_pct, abs_error, interval_coverage_80
                   FROM metrics WHERE run_id = %(run)s AND segment_type = %(seg)s"""
        params: dict[str, Any] = {"run": run_id, "seg": segment_type}
        if model:
            query += " AND model = %(model)s"
            params["model"] = model
        if fold is not None:
            query += " AND fold = %(fold)s"
            params["fold"] = fold
        return self._all(query + " ORDER BY model, fold, segment_value", params)

    def top_errors(self, run_id, model, n, sort_by, min_volume):
        # sort_by is validated against this allow-list; it never comes from user text directly.
        order = {"wape": "m.wape DESC", "abs_error": "m.abs_error DESC"}[sort_by]
        return self._all(
            f"""SELECT sp.item_id, sp.store_id, i.dept_id, i.cat_id, m.wape, m.mae, m.bias_pct,
                       m.abs_error, m.total_actual
                FROM metrics m
                JOIN v_series_span sp ON m.segment_value = sp.item_id || '|' || sp.store_id
                JOIN dim_item i ON i.item_id = sp.item_id
                WHERE m.run_id = %s AND m.model = %s AND m.fold = 0 AND m.segment_type = 'series'
                  AND m.total_actual >= %s AND m.wape IS NOT NULL
                ORDER BY {order}, sp.item_id, sp.store_id
                LIMIT %s""",
            [run_id, model, min_volume, n],
        )

    # -------------------------------------------------------------------- simulation
    def residual_pool(self, run_id, model):
        return self._all(
            """SELECT item_id, store_id, fold, horizon, residual, level FROM validation_residuals
               WHERE run_id = %s AND model = %s
               ORDER BY item_id, store_id, fold, horizon""",
            [run_id, model],
        )

    def backtest_rows(self, run_id, model):
        return self._all(
            """SELECT fold, policy, lead_time_days, service_level, initial_cover_days,
                      pred_stockout_prob, realized_stockout, pred_fill_rate, realized_fill_rate,
                      pred_total_p10, pred_total_p90, realized_total
               FROM simulator_backtest WHERE run_id = %s AND model = %s""",
            [run_id, model],
        )
