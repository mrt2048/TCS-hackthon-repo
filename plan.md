# Retail Smart Stock & Replenishment Advisor — Plan

TCS Technology Day | Team of 5 | 90 minutes | Browser-only (no backend, no paid keys, no logins)

## 1. Problem summary

A store must keep enough stock to meet demand without overstocking (markdowns, wastage on short-shelf-life items) or understocking (lost sales). Build a web app that learns from historical sales and inventory, forecasts demand, and tells staff **which products to reorder and how much**.

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

## 2. Data

| File | Columns (assumed) |
|---|---|
| sales.csv | date, product_id, product_name, category, qty_sold, price, promo_flag (or promo_type) |
| inventory.csv | product_id, current_stock, reorder_level, qty_received, stock_available, lead_time_days*, shelf_life_days* |
| calendar.csv | date, day_type (weekday/weekend), is_holiday, event_name, promo_event |

\* Not in the spec; we assume per-category defaults if absent.

Study the data first (5 min): date range, missing days, zero-sales days (stock-outs?), outliers, promo coverage, products with too little history.

Synthetic data is allowed. The generator must include: weekly seasonality, promo uplift, holiday spikes, noise, a few outliers, perishable vs non-perishable categories, and some products deliberately set up as low-stock and overstocked.

## 3. Assumptions (state these in the demo)
- Lead time fixed per category (e.g. dairy 2d, grocery 4d, general 5d)
- Review/planning horizon default 7 days (user-selectable)
- Zero-sales days with zero stock are treated as stock-outs (censored demand), not true zero demand
- Demand is independent per product (no substitution effects)
- Target service level 95% (z = 1.65); perishables use 90%
- Promo and holiday uplift are estimated from history and reused for future dates in the calendar
- Incoming stock (qty_received) arrives within the lead time and counts toward cover

## 4. Architecture

```
Sales CSV ┐
Inventory ├─> Clean & Join ─> Forecast Engine ─> Decision Engine ─> Dashboard | Alerts | Backtest
Calendar  ┘     (browser)       (JS modules)       (JS modules)        (HTML/Chart.js)
```

All in-browser: single `index.html` + JS modules, CSVs loaded via file input or bundled synthetic data. Libraries via CDN: PapaParse (CSV), Chart.js (charts).

### Modules
- `data.js` — synthetic generator + CSV loader (PapaParse) + validation
- `features.js` — join sales to calendar; weekday, weekend, holiday, promo flags; cleaning (outlier capping, stock-out flagging)
- `forecast.js` — per-product demand model
- `decision.js` — reorder quantity, risk scores, priority
- `backtest.js` — historical replay and comparison
- `ui.js` — product picker, tables, charts, alerts, explain panel

## 5. Forecast model (explainable, no ML library)

Per product:
1. Base rate = trimmed mean/median of daily sales over the last ~8 weeks on normal days (non-promo, non-holiday, uncensored)
2. Weekday factor = avg sales for that weekday ÷ overall avg (per product, fallback to category)
3. Promo uplift = avg promo-day sales ÷ avg normal-day sales (fallback to category if < N promo days)
4. Event/holiday uplift = same approach for holiday/festival days
5. Daily forecast(d) = base × weekday_factor(d) × promo_uplift(d) × event_uplift(d)
6. Horizon demand = sum of daily forecasts over the upcoming period (from the calendar)
7. Uncertainty = std dev of residuals (actual − fitted) over history, used for safety stock

Fallbacks: sparse products use category averages; flag "low confidence".

## 6. Decision engine

- Expected lead-time demand = sum of daily forecast over lead time
- Safety stock = z × σ_daily × √lead_time (lower z for perishables)
- Position = current_stock + incoming
- Recommended order = max(0, horizon demand + safety stock − position); round up to pack size if available; cap perishables by shelf life
- Days of cover = position ÷ forecast daily demand
- Stock-out risk: HIGH if cover < lead time; MEDIUM if cover < lead time + buffer; else LOW. Optionally probability via normal approximation: P(demand over lead time > position)
- Excess risk: HIGH if position > 2× horizon demand (or > shelf-life demand for perishables); else MEDIUM/LOW
- Priority score = stock-out probability × expected lost margin (price × shortfall), sorted descending; urgent items first
- Alerts: banner + red badge for HIGH stock-out risk

## 7. Backtest (proof of benefit)

- Hold out last 4 weeks. Train on everything before.
- Replay day by day with (a) historical actual ordering, (b) our recommended policy
- Compare: stock-out days, lost units, average inventory, excess/wastage units
- Display "X% fewer stock-outs, Y% less excess inventory"
- Also report forecast accuracy (MAPE/WAPE) vs a naive baseline (last-7-day average)

## 8. UI

1. Header: load CSV buttons + "Use sample data"
2. KPI strip: products at risk, overstocked, total recommended order units
3. Priority table (sortable): product, stock, days of cover, forecast, order qty, stock-out risk, excess risk, badge
4. Product detail panel: history + forecast chart (Chart.js), horizon selector
5. "Why?" panel: plain-language explanation with the factor breakdown
6. Alerts banner for HIGH-risk items
7. Backtest tab: before/after comparison chart and summary
8. Assumptions panel

## 9. Timeline (90 min)

| Time | Milestone |
|---|---|
| 0–10 | Read data, agree assumptions and interfaces (JSON shapes between modules) |
| 10–35 | Parallel build: data/synthetic, features, forecast, UI skeleton |
| 35–55 | Decision engine + wire UI to engine |
| 55–70 | Backtest + alerts + explain panel |
| 70–80 | Test across all products, fix edge cases (new products, zero history, negative orders) |
| 80–90 | Demo script, assumptions slide, polish |

## 10. Team split

1. Data lead — synthetic generator, CSV loading, validation
2. Features/forecast — feature builder and demand model
3. Decision — reorder logic, risk, priority, alerts
4. UI — dashboard, charts, explain panel
5. Backtest/QA — backtest, testing on all products, demo notes

## 11. Edge cases to handle
- Product with < 14 days of history → category fallback, "low confidence" tag
- Stock-out days within history → exclude from base-rate estimation
- Negative or zero recommendation → show "No order needed"
- Promo or holiday upcoming but no history for that event → use category-level uplift
- Perishables → cap order by shelf life
- Missing CSV columns → friendly validation message

## 12. Demo script (3–4 min)
1. Load data → show KPI strip
2. Open priority table → point out the top urgent item and its alert
3. Click item → forecast chart + "Why?" explanation
4. Show an overstocked item
5. Show weekend/holiday/promo effect on a forecast
6. Backtest tab → benefit numbers
7. State assumptions
