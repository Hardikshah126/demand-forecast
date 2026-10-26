"""Pydantic request / response models (the public API contract)."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["ok", "unavailable"]
    latest_run_id: int | None = None


# ----------------------------------------------------------------------------- runs
class FoldInfo(BaseModel):
    fold: int
    origin: date
    eval_start: date
    eval_end: date


class RunInfo(BaseModel):
    run_id: int
    created_at: str
    seed: int
    horizon: int
    n_series: int
    last_date: date
    best_model: str | None
    models: list[str]
    folds: list[FoldInfo]
    future_start: date
    future_end: date
    interval_method: str
    notes: str | None = None


# --------------------------------------------------------------------------- series
class SeriesItem(BaseModel):
    item_id: str
    store_id: str
    dept_id: str
    cat_id: str
    state_id: str
    first_date: date
    last_date: date
    avg_daily_sales_28d: float | None
    wape: float | None = Field(None, description="Pooled validation WAPE of the best model")


class SeriesListResponse(BaseModel):
    run_id: int
    total: int
    limit: int
    offset: int
    items: list[SeriesItem]


# -------------------------------------------------------------------------- forecast
class ActualPoint(BaseModel):
    date: date
    actual: int


class ForecastPoint(BaseModel):
    date: date
    horizon: int
    y_pred: float
    lower_80: float | None
    upper_80: float | None
    actual: int | None


class SeriesMetric(BaseModel):
    wape: float | None
    mae: float
    rmse: float
    bias: float
    bias_pct: float | None
    n_obs: int


class ForecastResponse(BaseModel):
    run_id: int
    item_id: str
    store_id: str
    fold: int = Field(description="0 = future forecast, 1..3 = validation folds")
    horizon_start: date
    horizon_end: date
    history: list[ActualPoint]
    models: dict[str, list[ForecastPoint]]
    series_metrics: dict[str, SeriesMetric]
    best_model: str | None


# ---------------------------------------------------------------------------- metrics
class MetricRow(BaseModel):
    model: str
    fold: int = Field(description="0 = pooled over validation folds 1..3")
    segment_type: str
    segment_value: str
    n_obs: int
    total_actual: float
    wape: float | None
    mae: float
    rmse: float
    bias: float
    bias_pct: float | None
    abs_error: float
    interval_coverage_80: float | None


class MetricsResponse(BaseModel):
    run: RunInfo
    segment_type: str
    rows: list[MetricRow]


class TopError(BaseModel):
    item_id: str
    store_id: str
    dept_id: str
    cat_id: str
    wape: float | None
    mae: float
    bias_pct: float | None
    abs_error: float
    total_actual: float
    direction: Literal["overforecast", "underforecast", "balanced", "undefined"]


class TopErrorsResponse(BaseModel):
    run_id: int
    model: str
    sort_by: str
    items: list[TopError]


# -------------------------------------------------------------------------- simulate
class SimulateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: str = Field(min_length=1, max_length=64)
    store: str = Field(min_length=1, max_length=16)
    model: str | None = Field(None, description="Forecast model; defaults to the run's best model")
    demand_change_pct: float = Field(0.0, ge=-30.0, le=50.0)
    lead_time_days: int = Field(7, ge=1, le=28)
    service_level: float = Field(0.95, ge=0.80, le=0.995)
    initial_cover_days: float = Field(14.0, ge=0.0, le=90.0)
    current_reorder_point: float | None = Field(None, ge=0.0, le=1_000_000.0)
    n_paths: int = Field(10_000, ge=1_000, le=20_000)
    seed: int = Field(42, ge=0, le=2**31 - 1)


class ScenarioResult(BaseModel):
    reorder_point: float
    order_up_to: float
    stockout_probability: float
    expected_fill_rate: float
    expected_lost_units: float
    expected_ending_inventory: float
    expected_orders: float


class SimulateResponse(BaseModel):
    run_id: int
    model: str
    item_id: str
    store_id: str
    horizon_start: date
    horizon_end: date
    n_paths: int
    seed: int
    mean_daily_demand: float
    sigma_daily_error: float
    initial_on_hand: float
    order_quantity: float
    z: float
    safety_stock: float
    reorder_point: float
    expected_lead_time_demand: float
    simulated_lead_time_demand: float
    recommended: ScenarioResult
    current: ScenarioResult
    trajectory_percentiles: dict[str, list[float]]
    dates: list[date]
    assumptions: list[str]


# -------------------------------------------------------------------------- backtest
class ReliabilityBin(BaseModel):
    bin: str
    n: int
    mean_predicted: float
    realized_frequency: float


class PolicyCalibration(BaseModel):
    n: int
    mean_predicted_stockout_prob: float
    realized_stockout_frequency: float
    brier_score: float
    brier_score_constant_baseline: float
    mean_predicted_fill_rate: float
    mean_realized_fill_rate: float | None
    demand_total_coverage_80: float | None = Field(
        None, description="Share of realized 28-day demand totals inside the simulated 10-90% band"
    )
    reliability: list[ReliabilityBin]


class BacktestResponse(BaseModel):
    run_id: int
    model: str
    description: str
    folds: list[int]
    by_policy: dict[str, PolicyCalibration]
