# Retail analytics pipeline

This folder implements the data-analysis, preprocessing, forecasting, decision, and backtest
work described in the repository's `plan.md`. It compares Random Forest, XGBoost, and CatBoost
against a 7-day-average baseline using a chronological 28-day holdout.

## Run

From the repository root:

```powershell
python -m pip install -r analytics/requirements.txt
python -m analytics.run
streamlit run analytics/app.py
```

By default, the pipeline reads `sales.csv`, `inventory.csv`, and `calendar.csv` from the
repository root and writes deliverables to `analytics/outputs/`. Optional arguments:

```powershell
python -m analytics.run --data-dir . --output-dir analytics/outputs --horizon 7 --holdout-days 28
```

The horizon accepts 3, 7, or 14 days. The calendar must contain enough dates after the last
sales date.

Run the pipeline before starting Streamlit. The frontend reads the generated files in
`analytics/outputs/`; it does not retrain models when the page loads.

## Pipeline

1. Validate schemas, dates, keys, numeric values, flags, and product/calendar coverage.
2. Profile the inputs; create an inspection-ready sales file with censored targets, outlier
   flags, and per-product 99th-percentile winsorization.
3. Generate product-day calendar and lag features. Lags/rolling windows use prior days only.
   Stock-out quantities become missing observations, not zero-demand labels.
4. Fit all three ML models on dates before the holdout; score only uncensored holdout rows.
   Training-target caps are learned from the training split only. The winning ML model is
   selected by lowest WAPE, with a naive baseline reported for comparison.
5. Refit the winning model on all available history, recursively forecast the requested future
   horizon, and produce order quantities, stock-out/excess risk, and priority.

The holdout is a walk-forward one-day-ahead evaluation: each date's forecast features can use
previous dates' observed demand, but never that date's target. This is appropriate for daily
replenishment decisions and is not a fixed 28-day-ahead forecast.

## Architecture and dataflow

```text
sales.csv ─────────┐
inventory.csv ─────┼─> analytics.data: validate, profile, censor and preprocess
calendar.csv ──────┘                       │
                                           v
                            analytics.features: product-day panel,
                            calendar fields, lag and rolling features
                                           │
                                           v
                            analytics.forecast: temporal benchmark
                            Random Forest / XGBoost / CatBoost / naive
                                           │
                         ┌─────────────────┴────────────────────┐
                         v                                      v
             holdout metrics/predictions          best-model refit and
             product scores/reports               recursive horizon forecasts
                         │                                      │
                         └──────────────────┬───────────────────┘
                                            v
                         analytics.decision: order, risk, priority
                                            │
                                            v
                         analytics/outputs: CSV and Markdown artifacts
                                            │
                                            v
                         analytics.app: Streamlit dashboard reads
                         the outputs and explains them in four tabs
```

### Frontend

`analytics/app.py` is a read-only presentation layer over generated outputs:

- **Overview:** dataset/recommendation KPIs, high-risk alerts, priority table, model WAPE chart.
- **Product detail:** historical uncensored sales, censored stock-out observations, daily forecast,
  calendar markers, per-product holdout score, and the order calculation explanation.
- **Backtest & models:** model comparison, selected-model daily actual-vs-predicted plot, and
  explicit notes on excluded censored rows and backtest limitations.
- **Data, plan & methodology:** data-quality report, plan-aligned assumptions, dataflow, and a
  catalog of generated files.

The app caches output reads with Streamlit `st.cache_data`. It does not silently substitute
sample values: if outputs are missing, it asks the user to run `python -m analytics.run`.

### Pipeline modules

- `data.py`: schema/value validation, CSV loading, stock-out target masking, capping/outlier
  inspection fields, and input-quality reporting.
- `features.py`: calendar join and leakage-safe product-level lag/rolling features.
- `forecast.py`: shared feature preprocessing and fitting/evaluation for the three tree models,
  naive baseline, feature importance, and recursive future forecasts.
- `decision.py`: service-level safety stock, stock position, order quantity, risk, and priority.
- `backtest.py`: holdout outputs and benchmark report.
- `run.py`: command-line orchestration and artifact generation.

### Technology stack

- **Python**: orchestration and analytics modules.
- **Pandas / NumPy**: CSV handling, joins, panel features, and metrics.
- **scikit-learn**: preprocessing, Random Forest, and evaluation metrics.
- **XGBoost / CatBoost**: gradient-boosted tree candidate models.
- **Streamlit**: interactive browser-based dashboard.
- **Plotly**: interactive model, sales, forecast, and risk visualizations.

## Generated deliverables

- `outputs/data_quality_report.md`: input dimensions, completeness, demand/stock-out profile,
  category coverage, calendar events, and preprocessing decisions.
- `outputs/preprocessed_sales.csv`: original sales plus `demand_target`, capped demand, and
  outlier flags. Stock-out rows remain visible but have a null target.
- `outputs/benchmark_report.md`, `model_metrics.csv`: WAPE, MAE, RMSE, bias for the baseline
  and all requested models.
- `outputs/product_metrics.csv`: holdout score by product for the selected ML model.
- `outputs/backtest_predictions.csv`: row-level predictions; censored rows are retained and
  marked `included_in_metrics=False`.
- `outputs/daily_forecasts.csv`: selected-model daily demand forecasts with calendar/event flags.
- `outputs/feature_importance.csv`: model feature importance for tree models.
- `outputs/forecast_recommendations.csv`: selected-model horizon forecasts joined to stock
  position, suggested order, stock-out/excess risk, and urgency score.

## Decisions and limits

- Expected demand is the observed sales proxy; 186 flagged stock-out rows in the supplied
  synthetic data are censored and cannot be scored as true demand.
- The model uses product/category, price, calendar/event/promotion indicators, and lag/rolling
  demand features. It has only one temporal holdout, so its score is an initial benchmark, not
  evidence of long-term generalization.
- The holdout winner is selected by WAPE. If there are multiple data updates, rerun the
  benchmark rather than assuming the same model remains best.
- The supplied inventory file is a single current snapshot, not historical daily balances or
  receipt transactions. Therefore a policy-level replay of lost units, average inventory, and
  excess/wastage cannot be estimated from these files without unsupported assumptions.
- Reorder math follows the plan's 7-day demand + safety stock - stock position rule, capped
  for perishables. Lead-time cover and the normal-approximation stock-out chance drive risk;
  probabilities are approximate because the history is short and synthetic.
- Calendar promotion flags are used for known future promotion/event days. The pipeline does
  not infer unprovided future product-specific discounts.
