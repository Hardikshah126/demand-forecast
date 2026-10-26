// Mirrors dfs/api/schemas.py. Dates are ISO strings (YYYY-MM-DD).

export interface FoldInfo {
  fold: number;
  origin: string;
  eval_start: string;
  eval_end: string;
}

export interface RunInfo {
  run_id: number;
  created_at: string;
  seed: number;
  horizon: number;
  n_series: number;
  last_date: string;
  best_model: string | null;
  models: string[];
  folds: FoldInfo[];
  future_start: string;
  future_end: string;
  interval_method: string;
  notes: string | null;
}

export interface SeriesItem {
  item_id: string;
  store_id: string;
  dept_id: string;
  cat_id: string;
  state_id: string;
  first_date: string;
  last_date: string;
  avg_daily_sales_28d: number | null;
  wape: number | null;
}

export interface SeriesListResponse {
  run_id: number;
  total: number;
  limit: number;
  offset: number;
  items: SeriesItem[];
}

export interface ActualPoint {
  date: string;
  actual: number;
}

export interface ForecastPoint {
  date: string;
  horizon: number;
  y_pred: number;
  lower_80: number | null;
  upper_80: number | null;
  actual: number | null;
}

export interface SeriesMetric {
  wape: number | null;
  mae: number;
  rmse: number;
  bias: number;
  bias_pct: number | null;
  n_obs: number;
}

export interface ForecastResponse {
  run_id: number;
  item_id: string;
  store_id: string;
  fold: number;
  horizon_start: string;
  horizon_end: string;
  history: ActualPoint[];
  models: Record<string, ForecastPoint[]>;
  series_metrics: Record<string, SeriesMetric>;
  best_model: string | null;
}

export interface MetricRow {
  model: string;
  fold: number;
  segment_type: string;
  segment_value: string;
  n_obs: number;
  total_actual: number;
  wape: number | null;
  mae: number;
  rmse: number;
  bias: number;
  bias_pct: number | null;
  abs_error: number;
  interval_coverage_80: number | null;
}

export interface MetricsResponse {
  run: RunInfo;
  segment_type: string;
  rows: MetricRow[];
}

export interface TopError {
  item_id: string;
  store_id: string;
  dept_id: string;
  cat_id: string;
  wape: number | null;
  mae: number;
  bias_pct: number | null;
  abs_error: number;
  total_actual: number;
  direction: "overforecast" | "underforecast" | "balanced" | "undefined";
}

export interface TopErrorsResponse {
  run_id: number;
  model: string;
  sort_by: string;
  items: TopError[];
}

export interface SimulateRequest {
  item: string;
  store: string;
  model?: string | null;
  demand_change_pct: number;
  lead_time_days: number;
  service_level: number;
  initial_cover_days: number;
  current_reorder_point?: number | null;
  n_paths?: number;
  seed?: number;
}

export interface ScenarioResult {
  reorder_point: number;
  order_up_to: number;
  stockout_probability: number;
  expected_fill_rate: number;
  expected_lost_units: number;
  expected_ending_inventory: number;
  expected_orders: number;
}

export interface SimulateResponse {
  run_id: number;
  model: string;
  item_id: string;
  store_id: string;
  horizon_start: string;
  horizon_end: string;
  n_paths: number;
  seed: number;
  mean_daily_demand: number;
  sigma_daily_error: number;
  initial_on_hand: number;
  order_quantity: number;
  z: number;
  safety_stock: number;
  reorder_point: number;
  expected_lead_time_demand: number;
  simulated_lead_time_demand: number;
  recommended: ScenarioResult;
  current: ScenarioResult;
  trajectory_percentiles: Record<"p10" | "p50" | "p90", number[]>;
  dates: string[];
  assumptions: string[];
}

export interface ReliabilityBin {
  bin: string;
  n: number;
  mean_predicted: number;
  realized_frequency: number;
}

export interface PolicyCalibration {
  n: number;
  mean_predicted_stockout_prob: number;
  realized_stockout_frequency: number;
  brier_score: number;
  brier_score_constant_baseline: number;
  mean_predicted_fill_rate: number;
  mean_realized_fill_rate: number | null;
  demand_total_coverage_80: number | null;
  reliability: ReliabilityBin[];
}

export interface BacktestResponse {
  run_id: number;
  model: string;
  description: string;
  folds: number[];
  by_policy: Record<string, PolicyCalibration>;
}
