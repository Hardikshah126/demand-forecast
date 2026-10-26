"""Database helpers (psycopg 3). All values are passed as bound parameters."""

from __future__ import annotations

import io
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql

from dfs.config import ROOT, get_settings

SQL_DIR = ROOT / "sql"


def connect(url: str | None = None) -> psycopg.Connection:
    url = url or get_settings().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not set (copy .env.example to .env).")
    return psycopg.connect(url)


@contextmanager
def transaction(url: str | None = None):
    conn = connect(url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def run_sql_file(conn: psycopg.Connection, name: str) -> None:
    conn.execute((SQL_DIR / name).read_text(encoding="utf-8"))


def copy_frame(
    conn: psycopg.Connection, table: str, df: pd.DataFrame, columns: Sequence[str] | None = None
) -> int:
    """Bulk-load a DataFrame with COPY. NaN/NaT become NULL."""
    columns = list(columns or df.columns)
    buf = io.StringIO()
    df[columns].to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    stmt = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT csv, NULL '\\N')").format(
        sql.Identifier(table), sql.SQL(", ").join(sql.Identifier(c) for c in columns)
    )
    with conn.cursor() as cur, cur.copy(stmt) as cp:
        while chunk := buf.read(1 << 20):
            cp.write(chunk)
    return len(df)


def read_frame(conn: psycopg.Connection, query: str, params: Iterable | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(query, list(params) if params is not None else None)
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def sql_path(name: str) -> Path:
    return SQL_DIR / name
