-- Data-quality checks. Each block is:   -- check: <name> | <error|info>
-- followed by a query returning ONE integer (number of violations; 0 = pass).
-- Named parameters (%(n_series)s, %(expected_last_date)s) are bound by dfs.quality.

-- check: duplicate_keys_fact_sales | error
SELECT COUNT(*)::int FROM (
    SELECT 1 FROM fact_sales GROUP BY item_id, store_id, date HAVING COUNT(*) > 1
) t;

-- check: duplicate_keys_fact_price | error
SELECT COUNT(*)::int FROM (
    SELECT 1 FROM fact_price GROUP BY item_id, store_id, wm_yr_wk HAVING COUNT(*) > 1
) t;

-- check: negative_sales | error
SELECT COUNT(*)::int FROM fact_sales WHERE sales < 0;

-- check: missing_dates_within_series | error
SELECT COALESCE(SUM((last_date - first_date + 1) - n_rows), 0)::int FROM (
    SELECT MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS n_rows
    FROM fact_sales GROUP BY item_id, store_id
) t;

-- check: series_not_ending_on_expected_last_date | error
SELECT COUNT(*)::int FROM v_series_span WHERE last_date <> %(expected_last_date)s::date;

-- check: invalid_prices | error
SELECT COUNT(*)::int FROM fact_price WHERE sell_price IS NULL OR sell_price <= 0;

-- check: unexpected_series_count | error
SELECT ABS(COUNT(*) - %(n_series)s)::int FROM v_series_span;

-- check: unexpected_item_store_combinations | error
SELECT COUNT(*)::int FROM (
    SELECT item_id, store_id FROM fact_sales
    EXCEPT
    SELECT i.item_id, s.store_id FROM dim_item i CROSS JOIN dim_store s
) t;

-- check: sales_dates_missing_from_dim_date | error
SELECT COUNT(*)::int FROM fact_sales f LEFT JOIN dim_date d ON d.date = f.date WHERE d.date IS NULL;

-- check: features_row_count_mismatch | error
SELECT ABS((SELECT COUNT(*) FROM fact_sales) - (SELECT COUNT(*) FROM features_daily))::int;

-- check: lag_features_null_after_warmup | error
-- After a series' first 28 days every lag/rolling feature must exist.
SELECT COUNT(*)::int FROM (
    SELECT f.*, ROW_NUMBER() OVER (PARTITION BY item_id, store_id ORDER BY date) AS rn
    FROM features_daily f
) t
WHERE rn > 28 AND (lag_7 IS NULL OR lag_14 IS NULL OR lag_28 IS NULL
                   OR roll_mean_7 IS NULL OR roll_mean_28 IS NULL);

-- check: zero_price_stored_as_price | error
SELECT COUNT(*)::int FROM features_daily WHERE sell_price = 0;

-- check: sales_days_without_price_share_x1000 | info
-- Share (per mille) of sales rows whose week has no price: "not on sale" days kept as-is.
SELECT COALESCE(ROUND(1000.0 * SUM(price_missing) / NULLIF(COUNT(*), 0)), 0)::int FROM features_daily;
