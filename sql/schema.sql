-- Idempotent schema for the Demand Forecasting & Stockout Simulation project.
-- Safe to re-run: every object uses IF NOT EXISTS / CREATE OR REPLACE.

-- ---------------------------------------------------------------- dimensions
CREATE TABLE IF NOT EXISTS dim_item (
    item_id  TEXT PRIMARY KEY,
    dept_id  TEXT NOT NULL,
    cat_id   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_store (
    store_id  TEXT PRIMARY KEY,
    state_id  TEXT NOT NULL
);

-- One row per calendar day, including the 28+ days after the last observed sale
-- (needed for known-in-advance covariates when producing the future forecast).
CREATE TABLE IF NOT EXISTS dim_date (
    date          DATE PRIMARY KEY,
    d             TEXT NOT NULL UNIQUE,        -- M5 day key, e.g. 'd_1913'
    wm_yr_wk      INTEGER NOT NULL,
    weekday       TEXT NOT NULL,
    dow           SMALLINT NOT NULL,           -- 0 = Monday ... 6 = Sunday
    month         SMALLINT NOT NULL,
    year          SMALLINT NOT NULL,
    day_of_month  SMALLINT NOT NULL,
    event_name_1  TEXT,
    event_type_1  TEXT,
    event_name_2  TEXT,
    event_type_2  TEXT,
    snap_ca       SMALLINT NOT NULL,
    snap_tx       SMALLINT NOT NULL,
    snap_wi       SMALLINT NOT NULL
);

-- ---------------------------------------------------------------------- facts
-- Sales rows exist from each item-store's first priced week onward (launch date);
-- pre-launch zeros are NOT stored because they are not real demand.
CREATE TABLE IF NOT EXISTS fact_sales (
    item_id   TEXT NOT NULL REFERENCES dim_item (item_id),
    store_id  TEXT NOT NULL REFERENCES dim_store (store_id),
    date      DATE NOT NULL REFERENCES dim_date (date),
    sales     INTEGER NOT NULL,
    PRIMARY KEY (item_id, store_id, date)
);
CREATE INDEX IF NOT EXISTS ix_fact_sales_date ON fact_sales (date);

-- Weekly price. A missing (item, store, week) row means "not on sale", never price 0.
CREATE TABLE IF NOT EXISTS fact_price (
    item_id     TEXT NOT NULL REFERENCES dim_item (item_id),
    store_id    TEXT NOT NULL REFERENCES dim_store (store_id),
    wm_yr_wk    INTEGER NOT NULL,
    sell_price  NUMERIC(8, 2) NOT NULL,
    PRIMARY KEY (item_id, store_id, wm_yr_wk)
);

-- ------------------------------------------------------------------- features
-- Features for day t use only sales on days <= t-1 (see sql/features.sql).
-- Calendar / price / SNAP columns are known-in-advance covariates for day t.
CREATE TABLE IF NOT EXISTS features_daily (
    item_id         TEXT NOT NULL,
    store_id        TEXT NOT NULL,
    date            DATE NOT NULL,
    sales           INTEGER NOT NULL,          -- the target (day t)
    lag_7           DOUBLE PRECISION,
    lag_14          DOUBLE PRECISION,
    lag_28          DOUBLE PRECISION,
    roll_mean_7     DOUBLE PRECISION,          -- mean of days t-7 .. t-1
    roll_mean_28    DOUBLE PRECISION,          -- mean of days t-28 .. t-1
    dow             SMALLINT NOT NULL,
    month           SMALLINT NOT NULL,
    day_of_month    SMALLINT NOT NULL,
    is_weekend      SMALLINT NOT NULL,
    event_flag      SMALLINT NOT NULL,
    event_type      SMALLINT NOT NULL,         -- 0 none,1 Cultural,2 National,3 Religious,4 Sporting
    snap            SMALLINT NOT NULL,         -- SNAP day for the store's state
    sell_price      DOUBLE PRECISION,          -- NULL when not on sale (not 0)
    price_change_1w DOUBLE PRECISION,          -- price(t)/price(t-7) - 1, NULL if either missing
    price_missing   SMALLINT NOT NULL,
    PRIMARY KEY (item_id, store_id, date),
    FOREIGN KEY (item_id, store_id, date) REFERENCES fact_sales (item_id, store_id, date)
);

-- ----------------------------------------------------------- modelling output
CREATE TABLE IF NOT EXISTS model_runs (
    run_id       SERIAL PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    seed         INTEGER NOT NULL,
    horizon      SMALLINT NOT NULL,
    n_series     INTEGER NOT NULL,
    last_date    DATE NOT NULL,                -- last observed sales date
    best_model   TEXT,                         -- by pooled WAPE, computed not assumed
    config       JSONB NOT NULL,
    notes        TEXT
);

-- fold 1..3 = chronological validation folds; fold 0 = future forecast (no actuals).
CREATE TABLE IF NOT EXISTS forecasts (
    run_id    INTEGER NOT NULL REFERENCES model_runs (run_id) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    fold      SMALLINT NOT NULL,
    item_id   TEXT NOT NULL,
    store_id  TEXT NOT NULL,
    date      DATE NOT NULL,
    horizon   SMALLINT NOT NULL,               -- 1..28 days after the forecast origin
    y_pred    DOUBLE PRECISION NOT NULL,
    lower_80  DOUBLE PRECISION,                -- NULL when no prior residuals exist (fold 1)
    upper_80  DOUBLE PRECISION,
    actual    INTEGER,                         -- NULL for fold 0
    PRIMARY KEY (run_id, model, fold, item_id, store_id, date)
);
CREATE INDEX IF NOT EXISTS ix_forecasts_series ON forecasts (run_id, item_id, store_id);

-- Validation residuals (actual - prediction), reused by the simulator.
CREATE TABLE IF NOT EXISTS validation_residuals (
    run_id    INTEGER NOT NULL REFERENCES model_runs (run_id) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    fold      SMALLINT NOT NULL,
    item_id   TEXT NOT NULL,
    store_id  TEXT NOT NULL,
    horizon   SMALLINT NOT NULL,
    residual  DOUBLE PRECISION NOT NULL,
    level     DOUBLE PRECISION NOT NULL,   -- mean forecast of that series/fold (for level scaling)
    PRIMARY KEY (run_id, model, fold, item_id, store_id, horizon)
);

-- fold 0 here means "pooled over folds 1-3". segment_type/segment_value describe the slice.
-- wape is NULL when total actual demand in the slice is zero (undefined).
CREATE TABLE IF NOT EXISTS metrics (
    run_id         INTEGER NOT NULL REFERENCES model_runs (run_id) ON DELETE CASCADE,
    model          TEXT NOT NULL,
    fold           SMALLINT NOT NULL,
    segment_type   TEXT NOT NULL,   -- overall|store|category|dept|weekday|horizon|series
    segment_value  TEXT NOT NULL,   -- 'all' for overall; 'ITEM|STORE' for series
    n_obs          INTEGER NOT NULL,
    total_actual   DOUBLE PRECISION NOT NULL,
    wape           DOUBLE PRECISION,
    mae            DOUBLE PRECISION NOT NULL,
    rmse           DOUBLE PRECISION NOT NULL,
    bias           DOUBLE PRECISION NOT NULL,       -- mean(pred - actual); >0 = over-forecast
    bias_pct       DOUBLE PRECISION,                -- sum(pred - actual)/sum(actual)
    abs_error      DOUBLE PRECISION NOT NULL,       -- sum |pred - actual|
    interval_coverage_80 DOUBLE PRECISION,          -- share of actuals inside the 80% interval
    PRIMARY KEY (run_id, model, fold, segment_type, segment_value)
);

-- Simulator backtest: predicted vs realized outcomes on held-out folds.
CREATE TABLE IF NOT EXISTS simulator_backtest (
    run_id               INTEGER NOT NULL REFERENCES model_runs (run_id) ON DELETE CASCADE,
    model                TEXT NOT NULL,
    fold                 SMALLINT NOT NULL,
    item_id              TEXT NOT NULL,
    store_id             TEXT NOT NULL,
    policy               TEXT NOT NULL,              -- recommended | no_safety_stock
    lead_time_days       SMALLINT NOT NULL,
    service_level        DOUBLE PRECISION NOT NULL,
    initial_cover_days   DOUBLE PRECISION NOT NULL,
    pred_stockout_prob   DOUBLE PRECISION NOT NULL,
    realized_stockout    SMALLINT NOT NULL,
    pred_fill_rate       DOUBLE PRECISION NOT NULL,
    realized_fill_rate   DOUBLE PRECISION,           -- NULL if realized demand was zero
    safety_stock         DOUBLE PRECISION NOT NULL,
    reorder_point        DOUBLE PRECISION NOT NULL,
    pred_total_p10       DOUBLE PRECISION NOT NULL,  -- simulated 28-day demand total, 10th pct
    pred_total_p90       DOUBLE PRECISION NOT NULL,
    realized_total       DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, model, fold, item_id, store_id, policy,
                 lead_time_days, service_level, initial_cover_days)
);
