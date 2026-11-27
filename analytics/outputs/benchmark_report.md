# Forecast benchmark and temporal backtest

- Evaluation: last 28 sales dates, chronologically held out.
- Scoring: uncensored product-days only; stock-out targets are excluded.
- Forecast cadence: one day ahead, using past observations and known calendar features.
- Model fitting: per-product target caps are calculated from the training period only.
- Baseline: prior 7 calendar days' mean uncensored sales.
- Selected model for final forecasts: **XGBoost**.
- Observed stock-out product-days within the holdout: 56.
- Selected ML model WAPE vs. naive baseline: 33.3% lower.

## Metrics

| Model | Scored rows | WAPE | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|---:|
| XGBoost | 644 | 0.1798 | 6.060 | 12.338 | -1.127 |
| CatBoost | 644 | 0.1856 | 6.252 | 13.369 | -1.384 |
| RandomForest | 644 | 0.1918 | 6.464 | 13.272 | -0.296 |
| Naive7DayAverage | 644 | 0.2697 | 9.088 | 17.596 | -0.256 |

## Interpretation

WAPE is the sum of absolute errors divided by the sum of actual uncensored demand; lower is better. MAE and RMSE are measured in units per product-day. Positive bias means over-forecasting. Model choice is based on the lowest WAPE among the three requested ML models (the naive baseline remains a comparator).

Detailed per-product metrics and row-level predictions, including the censored rows excluded from scoring, are in `product_metrics.csv` and `backtest_predictions.csv`.

A policy-level replay of historical lost units, average on-hand inventory, and excess/wastage is not estimated: the supplied files contain only a current inventory snapshot, not daily historical stock balances or receipts, and stock-out demand is censored. Reconstructing those outcomes would require unsupported assumptions.
