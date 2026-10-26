"""Data validation and duplicate detection (pandas-level) and the SQL check parser."""

import pandas as pd

from dfs.ingest import trim_prelaunch, wide_to_long
from dfs.quality import parse_checks, validate_long_sales, validate_prices
from tests.conftest import make_calendar


def _long(n=10, series=("A", "B")):
    dates = pd.date_range("2020-01-01", periods=n)
    return pd.DataFrame(
        [(s, "S1", d, 1) for s in series for d in dates], columns=["item_id", "store_id", "date", "sales"]
    )


def test_clean_frame_passes():
    v = validate_long_sales(_long(), expected_series=2, expected_last_date="2020-01-10")
    assert not any(v.values())


def test_duplicate_keys_detected():
    df = pd.concat([_long(), _long().iloc[:3]], ignore_index=True)
    assert validate_long_sales(df)["duplicate_keys"] == 3


def test_negative_sales_detected():
    df = _long()
    df.loc[0, "sales"] = -1
    assert validate_long_sales(df)["negative_sales"] == 1


def test_missing_dates_detected():
    df = _long().drop(index=[3, 4])
    assert validate_long_sales(df)["missing_dates"] == 2


def test_unexpected_series_count_and_end_date_detected():
    df = _long()
    v = validate_long_sales(df, expected_series=3, expected_last_date="2020-01-12")
    assert v["series_count_mismatch"] == 1
    assert v["series_not_ending_on_last_date"] == 2


def test_invalid_prices_detected():
    p = pd.DataFrame(
        {"item_id": ["A"] * 3, "store_id": "S", "wm_yr_wk": [1, 2, 2], "sell_price": [1.0, 0.0, -2.0]}
    )
    v = validate_prices(p)
    assert v["invalid_prices"] == 2
    assert v["duplicate_keys"] == 1


def test_sql_check_file_parses_and_every_check_is_a_single_query():
    from dfs.db import sql_path

    checks = parse_checks(sql_path("quality.sql").read_text(encoding="utf-8"))
    names = [c[0] for c in checks]
    assert len(names) == len(set(names)) >= 10
    for name, sev, body in checks:
        assert sev in {"error", "info"}, name
        assert body.upper().startswith("SELECT") and ";" not in body, name
    expected = {"duplicate_keys_fact_sales", "negative_sales", "missing_dates_within_series", "invalid_prices"}
    assert expected <= set(names)


def test_wide_to_long_and_prelaunch_trimming():
    cal = make_calendar(21)  # three weeks: wm_yr_wk 11000, 11001, 11002
    wide = pd.DataFrame(
        {"item_id": ["I1", "I2"], "store_id": ["S1", "S1"], **{f"d_{i}": [i, 100 + i] for i in range(1, 22)}}
    )
    long = wide_to_long(wide, cal)
    assert len(long) == 42 and long["sales"].min() >= 1
    # I1 is priced from week 11001; I2 from the first week. Missing price != price 0.
    prices = pd.DataFrame(
        {
            "item_id": ["I1", "I1", "I2", "I2", "I2"], "store_id": "S1",
            "wm_yr_wk": [11001, 11002, 11000, 11001, 11002], "sell_price": 2.0,
        }
    )  # fmt: skip
    trimmed = trim_prelaunch(long, prices, cal)
    first = trimmed.groupby("item_id")["date"].min()
    assert first["I1"] == cal.loc[cal.wm_yr_wk == 11001, "date"].min()
    assert first["I2"] == cal["date"].min()
    assert len(trimmed) == 42 - 7
