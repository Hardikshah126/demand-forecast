"""Training / evaluation pipeline: forecasts, metrics, residuals and simulator backtest."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from dfs import simulation
from dfs.config import HORIZON, MODELS, N_FOLDS
from dfs.data import load_panel, load_training_frame
from dfs.db import connect, copy_frame
from dfs.evaluation import (
    forecast_rows,
    make_folds,
    metric_rows,
    residual_rows,
    run_models,
)
from dfs.features import FEATURE_SETS, Panel, eligible_mask
from dfs.models import XGB_PARAMS

CORE_MODELS = MODELS  # naive, seasonal_naive, moving_average_28, xgboost


def pick_best_model(metrics_df: pd.DataFrame) -> str:
    """Best model by pooled-fold overall WAPE (computed from the results, never assumed)."""
    pooled = metrics_df[
        (metrics_df.segment_type == "overall") & (metrics_df.fold == 0) & metrics_df.wape.notna()
    ]
    return str(pooled.sort_values("wape").iloc[0]["model"])


def backtest_rows(
    panel: Panel, eligible: np.ndarray, outputs, folds, model: str, n_paths: int, log=print
) -> pd.DataFrame:
    """Simulator backtest on held-out folds 2..N (fold k uses only residual paths of folds < k)."""
    series = panel.series[eligible].reset_index(drop=True)
    y = panel.y[eligible]
    preds = {f.fold: outputs[model].preds[f.fold] for f in folds}
    res_by_fold = {f.fold: y[:, f.eval_idx] - preds[f.fold] for f in folds}
    rows = []
    t0 = time.time()
    for f in folds:
        if f.fold == 1:
            continue  # no earlier residuals exist
        prior_ids = [k for k in res_by_fold if k < f.fold]  # strictly earlier folds only
        pool = simulation.ResidualPool(
            paths=np.concatenate([res_by_fold[k] for k in prior_ids]),
            levels=np.concatenate([preds[k].mean(axis=1) for k in prior_ids]),
        )
        for i in range(len(series)):
            actual = y[i, f.eval_idx]
            for r in simulation.backtest_series(preds[f.fold][i], pool, actual, n_paths=n_paths):
                rows.append(
                    {"model": model, "fold": f.fold,
                     "item_id": series.at[i, "item_id"], "store_id": series.at[i, "store_id"], **r}  # fmt: skip
                )
        log(f"[backtest] fold {f.fold} done ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def run_training(
    seed: int = 42,
    notes: str | None = None,
    with_experiment: bool = True,
    backtest_paths: int = simulation.N_PATHS,
    log=print,
) -> int:
    t0 = time.time()
    with connect() as conn:
        panel = load_panel(conn)
        xgb_sets = {"xgboost": "v1"}
        if with_experiment:
            xgb_sets["xgboost_v2"] = "v2"
        frames = {
            name: load_training_frame(conn, panel, fset) for name, fset in xgb_sets.items()
        }
    log(f"[train] panel {panel.y.shape}, observed days {panel.n_obs}, loaded in {time.time() - t0:.0f}s")

    folds = make_folds(panel.n_obs, HORIZON, N_FOLDS)
    eligible = eligible_mask(panel, folds[0].origin)
    log(f"[train] eligible series: {int(eligible.sum())}/{len(eligible)}")
    outputs, folds = run_models(panel, eligible, frames, xgb_sets, seed, log=log)

    mdf = metric_rows(panel, eligible, outputs, folds)
    best = pick_best_model(mdf)
    log(f"[train] best model by pooled WAPE: {best}")
    rdf = residual_rows(panel, eligible, outputs, folds)
    fdf = forecast_rows(panel, eligible, outputs, folds)
    bdf = backtest_rows(panel, eligible, outputs, folds, best, backtest_paths, log=log)

    config = {
        "stores": sorted(panel.series["store_id"].unique().tolist()),
        "n_items": int(panel.series["item_id"].nunique()),
        "folds": [
            {"fold": f.fold, "origin": str(panel.dates[f.origin].date()),
             "eval_start": str(panel.dates[f.eval_idx[0]].date()),
             "eval_end": str(panel.dates[f.eval_idx[-1]].date())}  # fmt: skip
            for f in folds
        ],
        "future_start": str(panel.dates[panel.n_obs].date()),
        "future_end": str(panel.dates[panel.n_obs + HORIZON - 1].date()),
        "xgb_params": XGB_PARAMS,
        "feature_sets": {k: FEATURE_SETS[v] for k, v in xgb_sets.items()},
        "models": [*CORE_MODELS, *[m for m in xgb_sets if m not in CORE_MODELS]],
        "interval": "per-series empirical 10th/90th percentile of residuals from earlier folds",
        "backtest_paths": backtest_paths,
        "backtest_model": best,
    }
    with connect() as conn:
        run_id = conn.execute(
            """INSERT INTO model_runs (seed, horizon, n_series, last_date, best_model, config, notes)
               VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s) RETURNING run_id""",
            [seed, HORIZON, int(eligible.sum()), panel.dates[panel.n_obs - 1].date(), best,
             json.dumps(config), notes],
        ).fetchone()[0]  # fmt: skip
        for df in (fdf, mdf, rdf, bdf):
            df.insert(0, "run_id", run_id)
        copy_frame(conn, "forecasts", fdf)
        copy_frame(conn, "metrics", mdf)
        copy_frame(conn, "validation_residuals", rdf)
        copy_frame(conn, "simulator_backtest", bdf)
        conn.commit()
    log(f"[train] persisted run {run_id}: {len(fdf):,} forecasts, {len(mdf):,} metrics, "
        f"{len(rdf):,} residuals, {len(bdf):,} backtest rows in {time.time() - t0:.0f}s")  # fmt: skip
    return run_id
