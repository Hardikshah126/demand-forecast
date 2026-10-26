"""Service layer: business rules on top of the repository. No SQL and no HTTP concerns here."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from dfs import simulation
from dfs.api import schemas
from dfs.api.repository import Repository

SIM_ASSUMPTIONS = [
    "Demand paths = point forecast + a complete 28-day validation residual path drawn with "
    "replacement from the pool of all series' residual paths in the three validation folds "
    "(rescaled by the square root of the demand-level ratio), rounded, floored at zero, then "
    "scaled by (1 + demand change).",
    "Whole paths are resampled because forecast errors persist across the horizon; the "
    "simulation never runs beyond the 28-day validation horizon, so residuals are not "
    "extrapolated. Series-specific error behavior beyond its demand level is not modelled.",
    "Policy: daily-review (s, S) with a fixed lead time, lost sales (no backorders), order-up-to "
    "S = s + mean daily demand x max(lead time, 7 days). Nothing is on order at the start.",
    "Initial stock = initial cover (days) x the baseline mean daily forecast; the demand-change "
    "scenario then stresses that stock.",
    "Safety stock = z x sigma_daily x sqrt(lead time) and reorder point = mean daily demand x "
    "lead time + safety stock. This assumes independent, roughly normal daily errors; real "
    "errors are autocorrelated and skewed, so simulated service can differ from the target.",
    "Stockout probability is P(at least one stockout day in 28 days); the service-level target "
    "is a per-replenishment-cycle target, so the 28-day figure is higher.",
    "'Current' compares against a baseline reorder point (default: expected lead-time demand "
    "with no safety stock) from the same starting stock. Outputs are simulated estimates, not "
    "measured history.",
]


_POOL_CACHE: dict[tuple[int, int, str], simulation.ResidualPool] = {}
_POOL_CACHE_MAX = 4


class NotFoundError(Exception):
    pass


class BadRequestError(Exception):
    pass


class NoRunError(Exception):
    pass


def run_info(run: dict[str, Any]) -> schemas.RunInfo:
    cfg = run["config"]
    return schemas.RunInfo(
        run_id=run["run_id"],
        created_at=run["created_at"].isoformat(),
        seed=run["seed"],
        horizon=run["horizon"],
        n_series=run["n_series"],
        last_date=run["last_date"],
        best_model=run["best_model"],
        models=cfg["models"],
        folds=[schemas.FoldInfo(**f) for f in cfg["folds"]],
        future_start=cfg["future_start"],
        future_end=cfg["future_end"],
        interval_method=cfg["interval"],
        notes=run.get("notes"),
    )


class Service:
    def __init__(self, repo: Repository):
        self.repo = repo

    # ----------------------------------------------------------------------- helpers
    def _run(self) -> dict[str, Any]:
        run = self.repo.latest_run()
        if run is None:
            raise NoRunError("No model run found. Run `dfs train` to populate forecasts.")
        return run

    def _check_model(self, run: dict[str, Any], model: str | None) -> str:
        models = run["config"]["models"]
        if model is None:
            return run["best_model"]
        if model not in models:
            raise BadRequestError(f"Unknown model '{model}'. Available: {', '.join(models)}")
        return model

    # ------------------------------------------------------------------------ health
    def health(self) -> schemas.HealthResponse:
        if not self.repo.ping():
            return schemas.HealthResponse(status="degraded", database="unavailable")
        run = self.repo.latest_run()
        return schemas.HealthResponse(
            status="ok", database="ok", latest_run_id=run["run_id"] if run else None
        )

    # ------------------------------------------------------------------------ series
    def list_series(self, store, category, search, limit, offset) -> schemas.SeriesListResponse:
        run = self._run()
        total, rows = self.repo.list_series(
            run["run_id"], run["best_model"], store, category, search, limit, offset
        )
        return schemas.SeriesListResponse(
            run_id=run["run_id"], total=total, limit=limit, offset=offset,
            items=[schemas.SeriesItem(**r) for r in rows],
        )  # fmt: skip

    # ---------------------------------------------------------------------- forecast
    def forecast(self, item, store, models, fold, history_days) -> schemas.ForecastResponse:
        run = self._run()
        n_folds = len(run["config"]["folds"])
        if not 0 <= fold <= n_folds:
            raise BadRequestError(f"fold must be between 0 and {n_folds}")
        if models:
            for m in models:
                self._check_model(run, m)
        if self.repo.series_meta(item, store) is None:
            raise NotFoundError(f"Series {item} / {store} not found")
        pts = self.repo.forecast_points(run["run_id"], item, store, fold, models or None)
        if not pts:
            raise NotFoundError(f"No forecasts stored for {item} / {store} (fold {fold})")
        by_model: dict[str, list[schemas.ForecastPoint]] = defaultdict(list)
        for p in pts:
            by_model[p["model"]].append(schemas.ForecastPoint(**{k: v for k, v in p.items() if k != "model"}))
        start = min(p["date"] for p in pts)
        end = max(p["date"] for p in pts)
        hist = self.repo.actuals(item, store, start - timedelta(days=history_days), start - timedelta(days=1))
        sm = {
            r["model"]: schemas.SeriesMetric(**{k: r[k] for k in ("wape", "mae", "rmse", "bias", "bias_pct", "n_obs")})
            for r in self.repo.series_metrics(run["run_id"], item, store)
            if r["fold"] == 0 and r["model"] in by_model
        }  # fmt: skip
        return schemas.ForecastResponse(
            run_id=run["run_id"], item_id=item, store_id=store, fold=fold,
            horizon_start=start, horizon_end=end,
            history=[schemas.ActualPoint(**h) for h in hist],
            models=dict(by_model), series_metrics=sm, best_model=run["best_model"],
        )  # fmt: skip

    # ----------------------------------------------------------------------- metrics
    def metrics(self, segment_type, model, fold) -> schemas.MetricsResponse:
        run = self._run()
        if model:
            self._check_model(run, model)
        rows = self.repo.metrics(run["run_id"], segment_type, model, fold)
        return schemas.MetricsResponse(
            run=run_info(run), segment_type=segment_type,
            rows=[schemas.MetricRow(**r) for r in rows],
        )  # fmt: skip

    def top_errors(self, n, model, sort_by, min_volume) -> schemas.TopErrorsResponse:
        run = self._run()
        model = self._check_model(run, model)
        rows = self.repo.top_errors(run["run_id"], model, n, sort_by, min_volume)
        items = []
        for r in rows:
            bp = r["bias_pct"]
            direction = (
                "undefined" if bp is None
                else "overforecast" if bp > 0.05
                else "underforecast" if bp < -0.05
                else "balanced"
            )  # fmt: skip
            items.append(schemas.TopError(**r, direction=direction))
        return schemas.TopErrorsResponse(
            run_id=run["run_id"], model=model, sort_by=sort_by, items=items
        )

    # --------------------------------------------------------------------- simulation
    def simulate(self, req: schemas.SimulateRequest) -> schemas.SimulateResponse:
        run = self._run()
        model = self._check_model(run, req.model)
        if self.repo.series_meta(req.item, req.store) is None:
            raise NotFoundError(f"Series {req.item} / {req.store} not found")
        pts = self.repo.forecast_points(run["run_id"], req.item, req.store, 0, [model])
        if not pts:
            raise NotFoundError(f"No future forecast stored for {req.item} / {req.store}")
        pts.sort(key=lambda p: p["date"])
        forecast = np.array([p["y_pred"] for p in pts], dtype=float)
        pool = self._pool(run["run_id"], model, len(forecast))
        params = simulation.SimParams(
            demand_change=req.demand_change_pct / 100.0,
            lead_time_days=req.lead_time_days,
            service_level=req.service_level,
            initial_cover_days=req.initial_cover_days,
            n_paths=req.n_paths,
            seed=req.seed,
            current_reorder_point=req.current_reorder_point,
        )
        sim = simulation.simulate_series(forecast, pool, params, return_trajectory=True)
        inp = sim["inputs"]
        return schemas.SimulateResponse(
            run_id=run["run_id"], model=model, item_id=req.item, store_id=req.store,
            horizon_start=pts[0]["date"], horizon_end=pts[-1]["date"],
            n_paths=sim["n_paths"], seed=sim["seed"],
            mean_daily_demand=inp["mean_daily_demand"], sigma_daily_error=inp["sigma_daily_error"],
            initial_on_hand=inp["initial_on_hand"], order_quantity=inp["order_quantity"], z=inp["z"],
            safety_stock=sim["safety_stock"], reorder_point=sim["reorder_point"],
            expected_lead_time_demand=sim["expected_lead_time_demand"],
            simulated_lead_time_demand=sim["simulated_lead_time_demand"],
            recommended=schemas.ScenarioResult(**sim["recommended"]),
            current=schemas.ScenarioResult(**sim["current"]),
            trajectory_percentiles={f"p{k}": v for k, v in sim["trajectory_percentiles"].items()},
            dates=[p["date"] for p in pts],
            assumptions=SIM_ASSUMPTIONS,
        )  # fmt: skip

    def _pool(self, run_id: int, model: str, horizon: int) -> simulation.ResidualPool:
        """Residual paths of every series and validation fold (immutable per run, so cached)."""
        key = (id(self.repo), run_id, model)
        if key not in _POOL_CACHE:
            rows = self.repo.residual_pool(run_id, model)
            if len(_POOL_CACHE) >= _POOL_CACHE_MAX:
                _POOL_CACHE.pop(next(iter(_POOL_CACHE)))
            _POOL_CACHE[key] = _build_pool(rows, horizon)
        return _POOL_CACHE[key]

    def backtest(self) -> schemas.BacktestResponse:
        run = self._run()
        model = run["config"].get("backtest_model") or run["best_model"]
        rows = self.repo.backtest_rows(run["run_id"], model)
        if not rows:
            raise NotFoundError("No simulator backtest stored for the latest run")
        df = pd.DataFrame(rows)
        summary = simulation.calibration_summary(df)
        return schemas.BacktestResponse(
            run_id=run["run_id"], model=model,
            description=(
                "Held-out backtest: for each series and each of folds 2-3, the simulator used only "
                "the fold's forecast and residuals from EARLIER folds, then the identical policy "
                "was replayed on the realized demand. Compares simulated probabilities with "
                "measured historical outcomes."
            ),
            folds=sorted(int(x) for x in df["fold"].unique()),
            by_policy={k: schemas.PolicyCalibration(**v) for k, v in summary.items()},
        )  # fmt: skip


def _build_pool(rows: list[dict[str, Any]], horizon: int) -> simulation.ResidualPool:
    """Rows (item, store, fold, horizon, residual, level) -> (N, horizon) residual paths."""
    if not rows:
        raise NotFoundError("No validation residuals stored for this model")
    df = pd.DataFrame(rows).sort_values(["item_id", "store_id", "fold", "horizon"])
    if len(df) % horizon:
        raise NotFoundError("Incomplete validation residuals for this model")
    paths = df["residual"].to_numpy(dtype=float).reshape(-1, horizon)
    levels = df["level"].to_numpy(dtype=float).reshape(-1, horizon)[:, 0]
    return simulation.ResidualPool(paths=paths, levels=levels)
