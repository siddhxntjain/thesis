# Canonical Schema

## `ticker_year_panel.csv`
One row per `ticker, signal_year`.

Columns:
- IDs: `ticker, permno, cusip, cik, sector, industry`
- Timing: `signal_year, signal_anchor_date`
- Signals: `tls_score, delta_tls, year_sentiment, sentiment_mean, sentiment_median, sentiment_pos_share, sentiment_neg_share, co2_unscaled, co2_log, co2_scaled, year_assets`
- Outcomes: `ret_3m, ret_3m_start, ret_3m_end, ret_6m, ret_6m_start, ret_6m_end, ret_1y, ret_1y_start, ret_1y_end, ret_2y, ret_2y_start, ret_2y_end, ret_5y, ret_5y_start, ret_5y_end`
- QC: `n_months_3m, n_months_6m, n_months_1y, n_months_2y, n_months_5y, has_complete_3m, has_complete_6m, has_complete_1y, has_complete_2y, has_complete_5y`

## `benchmark_factors_monthly.csv`
One row per month end with FF benchmark series.

Columns:
- `month_end, rf, mkt_rf, smb, hml, rmw, cma, rm`
- `rm = rf + mkt_rf`

## `custom_factor_returns_monthly.csv`
One row per `factor_id, month_end`.

Columns:
- `factor_id, signal_name, construction_rule, month_end`
- `long_ret, short_ret, ls_ret, ls_excess`
- `n_long, n_short, universe_n, signal_year`
