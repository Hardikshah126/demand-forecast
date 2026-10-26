"""Reproducible M5 ingestion: download, subset, wide -> long, and load the star schema.

Subset rule (documented in the README): stores CA_1, TX_1, WI_1 and the 100 items with the
highest total unit sales across those stores *before the first validation origin* (so the
item selection itself cannot peek at validation or future data).
"""

from __future__ import annotations

import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from dfs.config import HORIZON, M5_LAST_DAY, N_FOLDS, N_ITEMS, STORES, get_settings
from dfs.db import copy_frame, run_sql_file

M5_FILES = ("calendar.csv", "sales_train_validation.csv", "sell_prices.csv")
# Mirror of the official Kaggle files (Kaggle itself requires a login and rules acceptance).
MIRROR_URL = "https://huggingface.co/datasets/kashif/M5/resolve/main/{name}"

# First day of the first validation fold's *training* data boundary: folds evaluate the last
# N_FOLDS * HORIZON observed days, so item selection uses days d_1 .. d_{FIRST_ORIGIN}.
FIRST_ORIGIN_DAY = M5_LAST_DAY - N_FOLDS * HORIZON  # 1829


def download(raw_dir: Path | None = None, force: bool = False) -> list[Path]:
    raw_dir = raw_dir or get_settings().raw_data_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for name in M5_FILES:
        dest = raw_dir / name
        if dest.exists() and dest.stat().st_size > 0 and not force:
            print(f"[download] {name}: already present ({dest.stat().st_size:,} bytes)")
        else:
            print(f"[download] {name}: fetching ...")
            urllib.request.urlretrieve(MIRROR_URL.format(name=name), dest)  # noqa: S310
        out.append(dest)
    return out


# ----------------------------------------------------------------------------- subsetting
def _iter_store_rows(path: Path, stores: tuple[str, ...], chunksize: int = 2000) -> Iterator[pd.DataFrame]:
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {c: np.int32 for c in header if c.startswith("d_")}
    for chunk in pd.read_csv(path, chunksize=chunksize, dtype=dtypes):
        chunk = chunk[chunk["store_id"].isin(stores)]
        if len(chunk):
            yield chunk


def load_subset_wide(
    sales_path: Path,
    stores: tuple[str, ...] = STORES,
    n_items: int = N_ITEMS,
    selection_end_day: int = FIRST_ORIGIN_DAY,
) -> tuple[pd.DataFrame, list[str]]:
    """Return the wide sales rows for the chosen item-store series and the chosen item ids."""
    frames = list(_iter_store_rows(sales_path, stores))
    wide = pd.concat(frames, ignore_index=True)
    day_cols = [f"d_{i}" for i in range(1, selection_end_day + 1)]
    row_total = wide[day_cols].sum(axis=1)
    totals = row_total.groupby(wide["item_id"]).sum()
    # Deterministic tie-break on item_id
    top = totals.sort_values(ascending=False, kind="stable").head(n_items)
    top_items = sorted(top.index)
    subset = wide[wide["item_id"].isin(top_items)].reset_index(drop=True)
    return subset, top_items


# --------------------------------------------------------------------------- transformations
def prepare_calendar(cal: pd.DataFrame) -> pd.DataFrame:
    cal = cal.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal["dow"] = cal["date"].dt.dayofweek.astype("int16")
    cal["day_of_month"] = cal["date"].dt.day.astype("int16")
    for c in ("event_name_1", "event_type_1", "event_name_2", "event_type_2"):
        cal[c] = cal[c].where(cal[c].notna(), None)
    cal = cal.rename(columns={"snap_CA": "snap_ca", "snap_TX": "snap_tx", "snap_WI": "snap_wi"})
    return cal


def wide_to_long(wide: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Melt d_1..d_N columns into one row per (item_id, store_id, date)."""
    day_cols = [c for c in wide.columns if c.startswith("d_")]
    long = wide.melt(
        id_vars=["item_id", "store_id"], value_vars=day_cols, var_name="d", value_name="sales"
    )
    date_by_d = calendar.set_index("d")["date"]
    long["date"] = long["d"].map(date_by_d)
    if long["date"].isna().any():
        raise ValueError("Some sales days are missing from the calendar.")
    long["sales"] = long["sales"].astype("int32")
    return (
        long[["item_id", "store_id", "date", "sales"]]
        .sort_values(["item_id", "store_id", "date"])
        .reset_index(drop=True)
    )


def load_prices(prices_path: Path, items: list[str], stores: tuple[str, ...]) -> pd.DataFrame:
    prices = pd.read_csv(
        prices_path,
        dtype={"store_id": "category", "item_id": "category", "wm_yr_wk": "int32", "sell_price": "float64"},
    )
    prices = prices[prices["item_id"].isin(items) & prices["store_id"].isin(stores)].copy()
    prices["item_id"] = prices["item_id"].astype(str)
    prices["store_id"] = prices["store_id"].astype(str)
    return prices.reset_index(drop=True)


def trim_prelaunch(long: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Drop days before an item-store's first priced week.

    In M5 a missing price means the item was not on sale in that store that week, so the zeros
    before the first price are "not yet launched", not zero demand. Keeping them would teach
    models (and baselines) that new items sell nothing.
    """
    week_start = calendar.groupby("wm_yr_wk")["date"].min()
    first_week = prices.groupby(["item_id", "store_id"])["wm_yr_wk"].min().rename("first_wk")
    launch = first_week.map(week_start).rename("launch_date").reset_index()
    merged = long.merge(launch, on=["item_id", "store_id"], how="left")
    if merged["launch_date"].isna().any():
        raise ValueError("Some item-store series have no price history at all.")
    return merged[merged["date"] >= merged["launch_date"]].drop(columns="launch_date").reset_index(drop=True)


# ------------------------------------------------------------------------------- orchestration
def build_tables(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    raw_dir = raw_dir or get_settings().raw_data_dir
    missing = [f for f in M5_FILES if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {raw_dir}. Run `dfs download` first.")
    calendar = prepare_calendar(pd.read_csv(raw_dir / "calendar.csv"))
    wide, items = load_subset_wide(raw_dir / "sales_train_validation.csv")
    prices = load_prices(raw_dir / "sell_prices.csv", items, STORES)
    long = trim_prelaunch(wide_to_long(wide, calendar), prices, calendar)
    dim_item = wide[["item_id", "dept_id", "cat_id"]].drop_duplicates().sort_values("item_id")
    dim_store = wide[["store_id", "state_id"]].drop_duplicates().sort_values("store_id")
    return {
        "dim_item": dim_item,
        "dim_store": dim_store,
        "dim_date": calendar,
        "fact_sales": long,
        "fact_price": prices,
    }


DIM_DATE_COLS = [
    "date", "d", "wm_yr_wk", "weekday", "dow", "month", "year", "day_of_month",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_ca", "snap_tx", "snap_wi",
]  # fmt: skip


def load_database(conn, tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Rebuild the star schema contents. Truncates dependent tables first (idempotent rebuild)."""
    conn.execute(
        "TRUNCATE features_daily, fact_sales, fact_price, dim_item, dim_store, dim_date "
        "RESTART IDENTITY CASCADE"
    )
    counts = {
        "dim_item": copy_frame(conn, "dim_item", tables["dim_item"]),
        "dim_store": copy_frame(conn, "dim_store", tables["dim_store"]),
        "dim_date": copy_frame(conn, "dim_date", tables["dim_date"], DIM_DATE_COLS),
        "fact_sales": copy_frame(conn, "fact_sales", tables["fact_sales"]),
        "fact_price": copy_frame(conn, "fact_price", tables["fact_price"]),
    }
    run_sql_file(conn, "features.sql")  # (re)builds v_covariates and features_daily
    return counts
