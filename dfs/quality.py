"""Data-quality checks: pure pandas validators (unit-testable) and the SQL check runner."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from dfs.db import sql_path


@dataclass
class CheckResult:
    name: str
    severity: str  # "error" | "info"
    value: int

    @property
    def passed(self) -> bool:
        return self.severity == "info" or self.value == 0


# --------------------------------------------------------------------- pandas-level validators
def validate_long_sales(
    long: pd.DataFrame, expected_series: int | None = None, expected_last_date=None
) -> dict[str, int]:
    """Return violation counts for a long sales frame (item_id, store_id, date, sales)."""
    key = ["item_id", "store_id", "date"]
    out = {
        "duplicate_keys": int(long.duplicated(key).sum()),
        "negative_sales": int((long["sales"] < 0).sum()),
        "null_sales": int(long["sales"].isna().sum()),
    }
    g = long.groupby(["item_id", "store_id"])["date"].agg(["min", "max", "count"])
    span = (g["max"] - g["min"]).dt.days + 1
    out["missing_dates"] = int((span - g["count"]).sum())
    if expected_series is not None:
        out["series_count_mismatch"] = abs(len(g) - expected_series)
    if expected_last_date is not None:
        out["series_not_ending_on_last_date"] = int((g["max"] != pd.Timestamp(expected_last_date)).sum())
    return out


def validate_prices(prices: pd.DataFrame) -> dict[str, int]:
    return {
        "duplicate_keys": int(prices.duplicated(["item_id", "store_id", "wm_yr_wk"]).sum()),
        "invalid_prices": int((prices["sell_price"].isna() | (prices["sell_price"] <= 0)).sum()),
    }


# -------------------------------------------------------------------------- SQL check runner
_BLOCK = re.compile(r"--\s*check:\s*(?P<name>\w+)\s*\|\s*(?P<sev>\w+)\s*\n(?P<body>.*?)(?=\n--\s*check:|\Z)", re.S)


def parse_checks(text: str) -> list[tuple[str, str, str]]:
    checks = []
    for m in _BLOCK.finditer(text):
        body = "\n".join(
            line for line in m.group("body").splitlines() if not line.strip().startswith("--")
        ).strip().rstrip(";")
        checks.append((m.group("name"), m.group("sev"), body))
    return checks


def run_sql_checks(conn, n_series: int, expected_last_date) -> list[CheckResult]:
    checks = parse_checks(sql_path("quality.sql").read_text(encoding="utf-8"))
    params = {"n_series": n_series, "expected_last_date": str(expected_last_date)}
    results = []
    for name, sev, body in checks:
        value = conn.execute(body, params).fetchone()[0]
        results.append(CheckResult(name, sev, int(value)))
    return results
