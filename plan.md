# Retail Smart Stock & Replenishment Advisor — Plan (Streamlit)

TCS Technology Day | Team of 5 | 90 minutes | Python + Streamlit frontend, no paid keys, no logins

> Note: the original brief says "runs in the browser, no backend". Streamlit serves a browser UI from a local Python process. Confirm with the organizers that this is acceptable. Fallback: deploy on Streamlit Community Cloud (free, no login for viewers) or keep a pure-JS version.

## 1. Problem summary

A store must keep enough stock to meet demand without overstocking (markdowns, wastage on short-shelf-life items) or understocking (lost sales). Build an app that learns from historical sales and inventory, forecasts demand, and tells staff **which products to reorder and how much**.

### Required capabilities
1. Predict expected demand for a selected product for the upcoming period
2. Recommend additional stock to order
3. Identify products at risk of stock-out
4. Identify products with inventory well above expected demand
5. Respond sensibly to weekends, holidays, promotions, unusual sales patterns
6. Prioritize products needing urgent replenishment
7. Show stock-out / excess risk tied to each recommendation
8. Alert staff on high stock-out chance
9. Explain the reasoning behind prediction and recommendation
10. Compare recommendation against historical outcomes (demonstrate benefit)
11. Be tested against all relevant products in the provided test data

## 2. Data (synthetic files already generated)

| File | Columns |
|---|---|
| sales.csv | date, product_id, product_name, category, quantity_sold, selling_price, promo_flag, stockout_flag |
| inventory.csv | product_id, product_name, category, current_stock, reorder_level, quantity_received, last_received_date, incoming_on_order, stock_availability, lead_time_days, shelf_life_days |
| calendar.csv | date, day_of_week, day_type, is_holiday, event_name, event_type, promo_event |

- Sales: 25 products, 1 Jun – 26 Nov 2026, one row per product per day
- Calendar: 1 Jun – 31 Dec 2026 (covers the upcoming forecast window)
- If the organizers supply real files with different headers, map them in one `COLUMN_MAP` dict in `data.py`.

First 5 minutes: study the data. Check date range, missing days, zero-sales / stock-out days, outliers, promo coverage, products with little history.

## 3. Assumptions (state these in the demo)
- Lead time and shelf life come from inventory.csv (fallback to category defaults)
- Planning horizon default 7 days, user-selectable (3 / 7 / 14)
- Days with `stockout_flag = 1` are censored demand: excluded from base-rate estimation
- Demand is independent per product (no substitution)
- Target service level 95% (z = 1.65); perishables (shelf life ≤ 7 days) use 90%
- Promo and holiday uplift are estimated from history and applied to future dates marked in the calendar
- `incoming_on_order` arrives within the lead time and counts toward stock position
- "Today" = last sales date + 1 (27 Nov 2026)

## 4. Architecture

```
sales.csv ─┐
inventory ─┼─> data.py (load, validate, join) ─> forecast.py ─> decision.py ─> app.py (Streamlit UI)
calendar ──┘                                                        │
                                                                    └─> backtest.py ─> Backtest page
```

Single repo, run with `streamlit run app.py`. Libraries: `streamlit`, `pandas`, `numpy`, `plotly`. No ML libraries needed, so the model stays explainable.

### Files
```
app.py            # Streamlit entrypoint, page layout, navigation
data.py           # loaders, schema validation, joins, COLUMN_MAP, upload handling
features.py       # weekday, weekend, holiday, promo flags; cleaning; outlier capping
forecast.py       # per-product demand model + factor breakdown
decision.py       # reorder qty, safety stock, risk scores, priority, alert text
backtest.py       # replay on held-out weeks, compare vs historical outcomes
explain.py        # builds the plain-language "why" text
sample_data/      # the 3 CSVs
requirements.txt
```

### Streamlit design notes
- `@st.cache_data` on loaders and on the forecast table so interactions stay fast
- Sidebar: data source (use bundled sample / upload 3 CSVs), horizon, service level, category filter
- `st.session_state` for selected product
- Pages via tabs: Overview, Product detail, Backtest, Assumptions

## 5. Forecast model (explainable, no ML library)

Per product:
1. Base rate = trimmed mean (or median) of daily sales over the last ~8 weeks on normal days (non-promo, non-holiday, uncensored)
2. Weekday factor = avg sales on that weekday ÷ overall avg (per product, category fallback)
3. Promo uplift = avg promo-day sales ÷ avg normal-day sales (category fallback if fewer than ~5 promo days)
4. Event uplift = same for holiday / festival days (category fallback)
5. Daily forecast(d) = base × weekday_factor(d) × promo_uplift(d) × event_uplift(d), where d comes from calendar.csv
6. Horizon demand = sum of daily forecast over the horizon
7. Uncertainty σ = std dev of residuals (actual − fitted) on history

Sparse products: category averages and a "low confidence" tag.

## 6. Decision engine

- Lead-time demand = sum of daily forecast over `lead_time_days`
- Safety stock = z × σ_daily × √lead_time
- Stock position = current_stock + incoming_on_order
- Recommended order = max(0, horizon demand + safety stock − position); cap perishables so the order does not exceed what can sell within shelf life
- Days of cover = position ÷ forecast daily demand
- Stock-out risk: HIGH if cover < lead time; MEDIUM if cover < lead time + 2 days; else LOW. Also show probability via normal approximation, P(lead-time demand > position)
- Excess risk: HIGH if position > 2× horizon demand (or beyond shelf-life demand for perishables); MEDIUM if 1.5×; else LOW
- Priority score = stock-out probability × expected lost margin (price × shortfall); sort descending
- Alerts: `st.error` banner listing HIGH stock-out products; colored badges in the table

## 7. Backtest (proof of benefit)

- Hold out the last 4 weeks. Train only on data before.
- Replay day by day: (a) the historical policy (what actually happened, from stockout_flag and sales), (b) our recommended policy
- Compare: stock-out days, lost units, average inventory, excess/wastage units
- Show "X% fewer stock-outs, Y% less excess inventory" with `st.metric` and a Plotly bar chart
- Report forecast accuracy (WAPE) vs a naive baseline (last-7-day average)

## 8. UI (Streamlit)

1. Header + sidebar controls (horizon, service level, category, upload)
2. KPI strip with `st.metric`: products at risk, overstocked, total recommended order units
3. Alert banner for HIGH stock-out risk products
4. Priority table (`st.dataframe` with column_config and color-coded risk): product, stock, days of cover, forecast, order qty, stock-out risk, excess risk
5. Product detail tab: `st.selectbox` for product, Plotly chart of history + forecast with promo/holiday markers
6. "Why this recommendation?" expander: factor breakdown (base × weekday × promo × event) and order math in plain language
7. Backtest tab: before/after metrics and charts
8. Assumptions tab: list from section 3

## 9. Timeline (90 min)

| Time | Milestone |
|---|---|
| 0–10 | Read the problem and data, agree on function signatures and DataFrame schemas between modules |
| 10–35 | Parallel build: data/features, forecast, UI skeleton with dummy data, backtest skeleton |
| 35–55 | Decision engine, wire the UI to real forecast and decision output |
| 55–70 | Backtest, alerts, explain panel |
| 70–80 | Test across all 25 products, fix edge cases |
| 80–90 | Demo script, assumptions tab, polish |

## 10. Team split

1. Data/features: `data.py`, `features.py`, validation, upload handling
2. Forecast: `forecast.py`, accuracy vs naive baseline
3. Decision: `decision.py`, risk scores, priority, alerts
4. UI: `app.py`, layout, charts, explain panel (starts with dummy data)
5. Backtest/QA: `backtest.py`, testing on all products, demo notes, README

Agree on these interfaces at minute 10:
- `forecast_all(sales, calendar, horizon) -> DataFrame[product_id, daily_forecast, horizon_demand, sigma, base, wk_factor, promo_uplift, event_uplift, confidence]`
- `decide(forecast_df, inventory, z) -> DataFrame[product_id, position, days_cover, order_qty, stockout_risk, stockout_prob, excess_risk, priority]`

## 11. Edge cases
- Under 14 days of history: category fallback and "low confidence" tag
- Stock-out days in history: exclude from base rate
- Order result ≤ 0: show "No order needed"
- Upcoming promo or holiday with no history for that event: use category-level uplift
- Perishables: cap the order by shelf life
- Missing columns in uploaded CSVs: show `st.error` with the missing names
- Dates in calendar beyond sales: allowed, that is the forecast window

## 12. Run instructions
```
pip install streamlit pandas numpy plotly
streamlit run app.py
```

## 13. Demo script (3–4 min)
1. Load sample data, show the KPI strip
2. Point to the alert banner and the top urgent product
3. Open that product: forecast chart and "Why?" breakdown
4. Show an overstocked product (e.g. Toilet Paper 12pk)
5. Show weekend / Black Friday / Christmas effect on a forecast
6. Backtest tab: benefit numbers
7. State assumptions