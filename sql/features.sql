-- Feature transformations. Re-runnable: rebuilds the view and the features_daily table.
--
-- Leakage contract: every sales-derived feature for day t is computed from days <= t-1.
--   lag_k        = sales on day t-k                         (k = 7, 14, 28)
--   roll_mean_n  = mean(sales on days t-n .. t-1)           (n = 7, 28)
-- Window frames therefore END at "1 PRECEDING" and never include the current row.
-- The ROWS-based windows are valid because fact_sales has no missing dates inside a series
-- (enforced by sql/quality.sql: missing_dates).
--
-- Calendar, SNAP and price columns are known-in-advance covariates for day t (calendar and
-- SNAP schedules are published; the M5 price table contains planned prices for future weeks).

-- Views are dropped first because CREATE OR REPLACE cannot change a column's type.
DROP VIEW IF EXISTS v_covariates;
DROP VIEW IF EXISTS v_calendar;
DROP VIEW IF EXISTS v_series_span;

CREATE VIEW v_series_span AS
SELECT item_id, store_id, MIN(date) AS first_date, MAX(date) AS last_date
FROM fact_sales
GROUP BY item_id, store_id;

-- Calendar covariates, one row per day (shared by SQL features and the Python panel loader).
CREATE VIEW v_calendar AS
SELECT
    d.date,
    d.wm_yr_wk,
    d.dow,
    d.month,
    d.day_of_month,
    (d.dow >= 5)::int AS is_weekend,
    (d.event_name_1 IS NOT NULL OR d.event_name_2 IS NOT NULL)::int AS event_flag,
    CASE d.event_type_1
        WHEN 'Cultural' THEN 1 WHEN 'National' THEN 2
        WHEN 'Religious' THEN 3 WHEN 'Sporting' THEN 4 ELSE 0
    END::int AS event_type,
    d.snap_ca, d.snap_tx, d.snap_wi
FROM dim_date d;

-- Known-in-advance covariates for every series, from its launch date through the last day in
-- dim_date (this extends past the last observed sale, which is what the future forecast uses).
CREATE VIEW v_covariates AS
SELECT
    sp.item_id,
    sp.store_id,
    c.date,
    c.dow, c.month, c.day_of_month, c.is_weekend, c.event_flag, c.event_type,
    CASE st.state_id WHEN 'CA' THEN c.snap_ca WHEN 'TX' THEN c.snap_tx WHEN 'WI' THEN c.snap_wi END::int AS snap,
    p.sell_price::double precision AS sell_price,
    CASE WHEN p.sell_price IS NOT NULL AND pp.sell_price IS NOT NULL
         THEN (p.sell_price / pp.sell_price - 1)::double precision END AS price_change_1w,
    (p.sell_price IS NULL)::int AS price_missing
FROM v_series_span sp
JOIN dim_store st ON st.store_id = sp.store_id
JOIN v_calendar c ON c.date >= sp.first_date
LEFT JOIN fact_price p
       ON p.item_id = sp.item_id AND p.store_id = sp.store_id AND p.wm_yr_wk = c.wm_yr_wk
LEFT JOIN v_calendar c7 ON c7.date = c.date - 7
LEFT JOIN fact_price pp
       ON pp.item_id = sp.item_id AND pp.store_id = sp.store_id AND pp.wm_yr_wk = c7.wm_yr_wk;

TRUNCATE features_daily;

INSERT INTO features_daily (
    item_id, store_id, date, sales,
    lag_7, lag_14, lag_28, roll_mean_7, roll_mean_28,
    dow, month, day_of_month, is_weekend, event_flag, event_type, snap,
    sell_price, price_change_1w, price_missing
)
SELECT
    s.item_id, s.store_id, s.date, s.sales,
    LAG(s.sales::double precision, 7)  OVER w AS lag_7,
    LAG(s.sales::double precision, 14) OVER w AS lag_14,
    LAG(s.sales::double precision, 28) OVER w AS lag_28,
    CASE WHEN COUNT(*) OVER w7  = 7  THEN AVG(s.sales::double precision) OVER w7  END AS roll_mean_7,
    CASE WHEN COUNT(*) OVER w28 = 28 THEN AVG(s.sales::double precision) OVER w28 END AS roll_mean_28,
    c.dow, c.month, c.day_of_month, c.is_weekend, c.event_flag, c.event_type, c.snap,
    c.sell_price, c.price_change_1w, c.price_missing
FROM fact_sales s
JOIN v_covariates c
  ON c.item_id = s.item_id AND c.store_id = s.store_id AND c.date = s.date
WINDOW
    w   AS (PARTITION BY s.item_id, s.store_id ORDER BY s.date),
    w7  AS (w ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING),
    w28 AS (w ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING);

ANALYZE features_daily;
