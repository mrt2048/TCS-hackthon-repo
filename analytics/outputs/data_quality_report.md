# Retail data analysis and quality report

## Dataset overview

- Sales: 4,475 product-days, 25 products, 2026-06-01 to 2026-11-26 (179 of 179 dates).
- Inventory: 25 product records.
- Calendar: 214 dates, 2026-06-01 to 2026-12-31.
- Sales date gaps: 0; missing product-days: 0.

## Quality and demand profile

- Duplicate sales product-days: 0.
- Missing sales/inventory values: 0 / 0.
- Zero-sales rows: 70.
- Stock-out rows: 186 (4.16%); these are treated as censored demand and excluded from model fitting and scoring.
- Promo-marked sales rows: 315.
- Product-level IQR outlier flags on uncensored rows: 154.
- History per product: 179 to 179 days.
- Uncensored quantity quantiles (units): 0.5th=27, 0.9th=59, 0.95th=67.3, 0.99th=96.26, 1th=288

## Category coverage

- Bakery: 2 products
- Beverages: 3 products
- Dairy: 4 products
- Grocery: 5 products
- Household: 3 products
- Meat: 1 products
- Personal Care: 1 products
- Produce: 4 products
- Snacks: 2 products

## Calendar events

- 2026-08-15: Independence Day (holiday)
- 2026-09-14: Ganesh Chaturthi (festival)
- 2026-10-02: Gandhi Jayanti (holiday)
- 2026-10-20: Dussehra (festival)
- 2026-11-08: Diwali (festival, promotion)
- 2026-11-27: Black Friday (promo_event, promotion)
- 2026-12-25: Christmas (holiday)
- 2026-12-31: New Year's Eve (festival)

## Preprocessing decisions

- Dates are parsed and checked; duplicate keys, invalid flags, negative quantities, missing required fields, and inventory/product mismatches stop the pipeline.
- Stock-out sales are preserved in the audit CSV but their demand target is null; they do not become false zero-demand labels.
- The inspection CSV includes per-product 99th-percentile winsorized demand and an outlier flag. Backtest fitting computes its caps from training data only.
- Calendar events and promotion indicators are joined by date; unknown future event categories are supported by one-hot encoders.
