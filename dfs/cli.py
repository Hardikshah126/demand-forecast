"""Command line entry point: `dfs <command>`. Each stage is a separate command."""

from __future__ import annotations

import argparse
import json
import sys

from psycopg import sql

from dfs.config import M5_LAST_DAY, N_ITEMS, STORES, get_settings


def cmd_download(args) -> int:
    from dfs.ingest import download

    download(force=args.force)
    return 0


def cmd_db_init(args) -> int:
    """Apply the idempotent schema and (re)create the read-only API role."""
    from dfs.db import connect, run_sql_file

    s = get_settings()
    with connect() as conn:
        run_sql_file(conn, "schema.sql")
        if s.readonly_db_password:
            role = sql.Identifier(s.readonly_db_user)
            exists = conn.execute(
                "SELECT 1 FROM pg_roles WHERE rolname = %s", [s.readonly_db_user]
            ).fetchone()
            verb = "ALTER" if exists else "CREATE"
            conn.execute(
                sql.SQL("{} ROLE {} LOGIN PASSWORD {}").format(
                    sql.SQL(verb), role, sql.Literal(s.readonly_db_password)
                )
            )
            dbname = conn.execute("SELECT current_database()").fetchone()[0]
            conn.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(dbname), role))
            conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(role))
            conn.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(role))
            conn.execute(
                sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {}").format(role)
            )
            print(f"[db-init] read-only role '{s.readonly_db_user}' ready")
        else:
            print("[db-init] READONLY_DB_PASSWORD not set: skipped read-only role")
        conn.commit()
    print("[db-init] schema applied")
    return 0


def cmd_ingest(args) -> int:
    from dfs.db import connect
    from dfs.ingest import build_tables, load_database
    from dfs.quality import validate_long_sales, validate_prices

    tables = build_tables()
    n_series = len(STORES) * N_ITEMS
    pre = validate_long_sales(tables["fact_sales"], n_series)
    pre.update({f"price_{k}": v for k, v in validate_prices(tables["fact_price"]).items()})
    print("[ingest] pre-load validation:", json.dumps(pre))
    bad = {k: v for k, v in pre.items() if v}
    if bad:
        print(f"[ingest] validation failed: {bad}", file=sys.stderr)
        return 1
    with connect() as conn:
        counts = load_database(conn, tables)
        conn.commit()
    print("[ingest] loaded rows:", json.dumps(counts))
    return 0


def cmd_features(args) -> int:
    """Rebuild v_calendar / v_covariates / features_daily from the loaded star schema."""
    from dfs.db import connect, run_sql_file

    with connect() as conn:
        run_sql_file(conn, "features.sql")
        n = conn.execute("SELECT COUNT(*) FROM features_daily").fetchone()[0]
        conn.commit()
    print(f"[features] features_daily rows: {n:,}")
    return 0


def cmd_quality(args) -> int:
    from dfs.db import connect
    from dfs.quality import run_sql_checks

    with connect() as conn:
        expected_last = conn.execute(
            "SELECT date FROM dim_date WHERE d = %s", [f"d_{M5_LAST_DAY}"]
        ).fetchone()[0]
        results = run_sql_checks(conn, len(STORES) * N_ITEMS, expected_last)
    width = max(len(r.name) for r in results)
    for r in results:
        flag = "PASS" if r.passed else "FAIL"
        print(f"  [{flag}] {r.name:<{width}}  {r.severity:<5} value={r.value}")
    failed = [r for r in results if not r.passed]
    print(f"[quality] {len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


def cmd_train(args) -> int:
    from dfs.pipeline import run_training

    run_id = run_training(seed=args.seed, notes=args.notes, with_experiment=not args.no_experiment)
    print(f"[train] persisted run_id={run_id}")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("dfs.api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dfs", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="download the M5 CSVs into data/raw")
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=cmd_download)

    sub.add_parser("db-init", help="apply schema.sql and create the read-only role").set_defaults(
        func=cmd_db_init
    )
    sub.add_parser("ingest", help="rebuild star schema + features_daily from raw CSVs").set_defaults(
        func=cmd_ingest
    )
    sub.add_parser("features", help="rebuild features_daily with SQL").set_defaults(func=cmd_features)
    sub.add_parser("quality", help="run SQL data-quality checks").set_defaults(func=cmd_quality)

    t = sub.add_parser("train", help="evaluate 4 strategies over 3 folds and persist results")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--notes", default=None)
    t.add_argument("--no-experiment", action="store_true", help="skip the xgboost_v2 experiment")
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("serve", help="start the FastAPI server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
