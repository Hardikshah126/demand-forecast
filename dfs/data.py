"""Load the dense modelling panel and the training frame from PostgreSQL."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dfs.db import read_frame
from dfs.features import FEATURE_SETS, Panel, extra_training_features


def _pivot(df: pd.DataFrame, value: str, pos: dict, t_index: dict, shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, np.nan)
    rows = pd.Series(list(zip(df["item_id"], df["store_id"], strict=True))).map(pos).to_numpy()
    cols = df["date"].map(t_index).to_numpy()
    out[rows, cols] = df[value].to_numpy(dtype=float)
    return out


def load_panel(conn) -> Panel:
    series = read_frame(
        conn,
        """SELECT sp.item_id, sp.store_id, i.dept_id, i.cat_id, st.state_id
           FROM v_series_span sp
           JOIN dim_item i USING (item_id) JOIN dim_store st USING (store_id)
           ORDER BY sp.store_id, sp.item_id""",
    )
    for col, name in (("item_id", "item_idx"), ("store_id", "store_idx"),
                      ("dept_id", "dept_idx"), ("cat_id", "cat_idx")):  # fmt: skip
        series[name] = pd.factorize(series[col], sort=True)[0]
    cal = read_frame(conn, "SELECT * FROM v_calendar ORDER BY date")
    cal["date"] = pd.to_datetime(cal["date"])
    dates = pd.DatetimeIndex(cal["date"])
    shape = (len(series), len(dates))
    pos = {k: i for i, k in enumerate(zip(series["item_id"], series["store_id"], strict=True))}
    t_index = {d: i for i, d in enumerate(dates)}

    sales = read_frame(conn, "SELECT item_id, store_id, date, sales FROM fact_sales")
    sales["date"] = pd.to_datetime(sales["date"])
    y = _pivot(sales, "sales", pos, t_index, shape)
    n_obs = int(sales["date"].map(t_index).max()) + 1

    cov = read_frame(
        conn, "SELECT item_id, store_id, date, sell_price, price_change_1w, snap FROM v_covariates"
    )
    cov["date"] = pd.to_datetime(cov["date"])
    return Panel(
        series=series,
        dates=dates,
        y=y,
        n_obs=n_obs,
        sell_price=_pivot(cov, "sell_price", pos, t_index, shape),
        price_change_1w=_pivot(cov, "price_change_1w", pos, t_index, shape),
        snap=_pivot(cov, "snap", pos, t_index, shape),
        cal=cal.set_index("date")[
            ["dow", "month", "day_of_month", "is_weekend", "event_flag", "event_type"]
        ].reset_index(drop=True),
    )


def load_training_frame(conn, panel: Panel, feature_set: str = "v1") -> pd.DataFrame:
    """Rows of features_daily (SQL-built) with series_pos / t, target column `sales`.

    For feature sets with extra sales-derived columns, those are added with the same Python
    function used during recursive forecasting.
    """
    df = read_frame(conn, "SELECT * FROM features_daily")
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(
        panel.series[["item_id", "store_id", "item_idx", "store_idx", "dept_idx", "cat_idx"]]
        .assign(series_pos=np.arange(panel.n_series)),
        on=["item_id", "store_id"],
        how="left",
    )
    t_index = {d: i for i, d in enumerate(panel.dates)}
    df["t"] = df["date"].map(t_index)
    extra = extra_training_features(panel, feature_set)
    if len(extra):
        df = df.merge(extra, on=["series_pos", "t"], how="left")
    keep = ["series_pos", "t", "date", "sales", *FEATURE_SETS[feature_set]]
    return df[keep].sort_values(["t", "series_pos"]).reset_index(drop=True)
