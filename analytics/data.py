"""Load, validate, and preprocess the retail advisor CSV inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SALES_COLUMNS = {
    "date",
    "product_id",
    "product_name",
    "category",
    "quantity_sold",
    "selling_price",
    "promo_flag",
    "stockout_flag",
}
INVENTORY_COLUMNS = {
    "product_id",
    "product_name",
    "category",
    "current_stock",
    "reorder_level",
    "quantity_received",
    "last_received_date",
    "incoming_on_order",
    "stock_availability",
    "lead_time_days",
    "shelf_life_days",
}
CALENDAR_COLUMNS = {
    "date",
    "day_of_week",
    "day_type",
    "is_holiday",
    "event_name",
    "event_type",
    "promo_event",
}


def _read_csv(path: Path, required: set[str], date_columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file was not found: {path}")
    frame = pd.read_csv(path)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")
    for column in date_columns:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ValueError(f"{path.name} contains missing or invalid dates in '{column}'")
    return frame


def load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load CSVs and reject schema, key, and value errors rather than hiding them."""
    sales = _read_csv(data_dir / "sales.csv", SALES_COLUMNS, ["date"])
    inventory = _read_csv(data_dir / "inventory.csv", INVENTORY_COLUMNS, ["last_received_date"])
    calendar = _read_csv(data_dir / "calendar.csv", CALENDAR_COLUMNS, ["date"])
    for frame, name, columns in (
        (sales, "sales.csv", SALES_COLUMNS),
        (inventory, "inventory.csv", INVENTORY_COLUMNS),
        (calendar, "calendar.csv", CALENDAR_COLUMNS - {"event_name", "event_type"}),
    ):
        if frame[list(columns)].isna().any().any():
            raise ValueError(f"{name} contains missing values in required fields")

    for frame, name, key in (
        (sales, "sales.csv", ["date", "product_id"]),
        (inventory, "inventory.csv", ["product_id"]),
        (calendar, "calendar.csv", ["date"]),
    ):
        if frame.duplicated(key).any():
            raise ValueError(f"{name} has duplicate keys: {', '.join(key)}")

    for column in ("quantity_sold", "selling_price", "promo_flag", "stockout_flag"):
        sales[column] = pd.to_numeric(sales[column], errors="coerce")
    if sales[["quantity_sold", "selling_price", "promo_flag", "stockout_flag"]].isna().any().any():
        raise ValueError("sales.csv contains missing or non-numeric quantity, price, or flag values")
    if (sales["quantity_sold"] < 0).any() or (sales["selling_price"] < 0).any():
        raise ValueError("Sales quantities and prices must be non-negative")
    for column in ("promo_flag", "stockout_flag"):
        if not sales[column].isin([0, 1]).all():
            raise ValueError(f"sales.csv column '{column}' must contain only 0 or 1")
        sales[column] = sales[column].astype("int8")

    inventory_numeric = (
        "current_stock",
        "reorder_level",
        "quantity_received",
        "incoming_on_order",
        "lead_time_days",
        "shelf_life_days",
    )
    for column in inventory_numeric:
        inventory[column] = pd.to_numeric(inventory[column], errors="coerce")
    if inventory[list(inventory_numeric)].isna().any().any():
        raise ValueError("inventory.csv contains missing or non-numeric stock/planning values")
    if (inventory[["current_stock", "reorder_level", "quantity_received", "incoming_on_order"]] < 0).any().any():
        raise ValueError("Inventory stock and receipt quantities must be non-negative")
    if (inventory[["lead_time_days", "shelf_life_days"]] <= 0).any().any():
        raise ValueError("Inventory lead time and shelf life must be positive")

    for column in ("is_holiday", "promo_event"):
        calendar[column] = pd.to_numeric(calendar[column], errors="coerce")
        if calendar[column].isna().any() or not calendar[column].isin([0, 1]).all():
            raise ValueError(f"calendar.csv column '{column}' must contain only 0 or 1")
        calendar[column] = calendar[column].astype("int8")
    calendar["event_name"] = calendar["event_name"].fillna("").astype(str)
    calendar["event_type"] = calendar["event_type"].fillna("").astype(str)

    product_ids = set(sales["product_id"].unique())
    inventory_ids = set(inventory["product_id"].unique())
    if product_ids != inventory_ids:
        raise ValueError(
            "Sales and inventory product IDs differ "
            f"(missing from inventory: {sorted(product_ids - inventory_ids)}; "
            f"missing from sales: {sorted(inventory_ids - product_ids)})"
        )
    if not set(sales["date"]).issubset(set(calendar["date"])):
        raise ValueError("calendar.csv does not cover every date in sales.csv")

    sales = sales.sort_values(["product_id", "date"]).reset_index(drop=True)
    inventory = inventory.sort_values("product_id").reset_index(drop=True)
    calendar = calendar.sort_values("date").reset_index(drop=True)
    return sales, inventory, calendar


def preprocess_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Expose censored targets and per-product 99th-percentile caps for inspection."""
    processed = sales.copy()
    processed["demand_target"] = processed["quantity_sold"].where(processed["stockout_flag"].eq(0))
    caps = (
        processed.loc[processed["demand_target"].notna()]
        .groupby("product_id")["demand_target"]
        .quantile(0.99)
    )
    processed["quantity_sold_capped"] = processed["demand_target"].clip(
        upper=processed["product_id"].map(caps)
    )
    processed["outlier_flag"] = False
    valid = processed["demand_target"].notna()
    processed.loc[valid, "outlier_flag"] = (
        processed.loc[valid, "demand_target"] > processed.loc[valid, "product_id"].map(caps)
    )
    return processed


def profile_data(
    sales: pd.DataFrame, inventory: pd.DataFrame, calendar: pd.DataFrame
) -> dict[str, object]:
    """Return descriptive checks used by the generated data-quality report."""
    expected_dates = pd.date_range(sales["date"].min(), sales["date"].max(), freq="D")
    expected_pairs = len(expected_dates) * sales["product_id"].nunique()
    observed_dates = set(sales["date"].unique())
    per_product_days = sales.groupby("product_id")["date"].nunique()
    uncensored = sales.loc[sales["stockout_flag"].eq(0)]
    q1 = uncensored.groupby("product_id")["quantity_sold"].quantile(0.25)
    q3 = uncensored.groupby("product_id")["quantity_sold"].quantile(0.75)
    upper_fence = q3 + 1.5 * (q3 - q1)
    outliers = uncensored["quantity_sold"] > uncensored["product_id"].map(upper_fence)
    events = calendar.loc[
        calendar["event_name"].ne(""),
        ["date", "event_name", "event_type", "promo_event"],
    ].copy()
    events["date"] = events["date"].dt.strftime("%Y-%m-%d")
    return {
        "sales_rows": len(sales),
        "product_count": int(sales["product_id"].nunique()),
        "sales_start": sales["date"].min().date().isoformat(),
        "sales_end": sales["date"].max().date().isoformat(),
        "sales_days_expected": len(expected_dates),
        "sales_days_observed": len(observed_dates),
        "sales_date_gaps": len(set(expected_dates) - observed_dates),
        "expected_product_days": expected_pairs,
        "missing_product_days": expected_pairs - len(sales),
        "sales_duplicates": int(sales.duplicated(["date", "product_id"]).sum()),
        "sales_missing_values": int(sales.isna().sum().sum()),
        "zero_sales_rows": int(sales["quantity_sold"].eq(0).sum()),
        "stockout_rows": int(sales["stockout_flag"].sum()),
        "stockout_rate": float(sales["stockout_flag"].mean()),
        "promo_rows": int(sales["promo_flag"].sum()),
        "uncensored_outlier_rows_iqr": int(outliers.sum()),
        "per_product_history_min": int(per_product_days.min()),
        "per_product_history_max": int(per_product_days.max()),
        "inventory_rows": len(inventory),
        "inventory_missing_values": int(inventory.isna().sum().sum()),
        "calendar_rows": len(calendar),
        "calendar_start": calendar["date"].min().date().isoformat(),
        "calendar_end": calendar["date"].max().date().isoformat(),
        "calendar_event_count": len(events),
        "events": events.to_dict("records"),
        "category_product_counts": (
            inventory.groupby("category")["product_id"].nunique().sort_index().to_dict()
        ),
        "quantity_quantiles": (
            sales["quantity_sold"].quantile([0.5, 0.9, 0.95, 0.99, 1.0]).to_dict()
        ),
    }


def write_quality_report(profile: dict[str, object], output_path: Path) -> None:
    quantiles = profile["quantity_quantiles"]
    categories = profile["category_product_counts"]
    events = profile["events"]
    lines = [
        "# Retail data analysis and quality report",
        "",
        "## Dataset overview",
        "",
        f"- Sales: {profile['sales_rows']:,} product-days, {profile['product_count']} products, "
        f"{profile['sales_start']} to {profile['sales_end']} "
        f"({profile['sales_days_observed']} of {profile['sales_days_expected']} dates).",
        f"- Inventory: {profile['inventory_rows']} product records.",
        f"- Calendar: {profile['calendar_rows']} dates, {profile['calendar_start']} to "
        f"{profile['calendar_end']}.",
        f"- Sales date gaps: {profile['sales_date_gaps']}; missing product-days: "
        f"{profile['missing_product_days']}.",
        "",
        "## Quality and demand profile",
        "",
        f"- Duplicate sales product-days: {profile['sales_duplicates']}.",
        f"- Missing sales/inventory values: {profile['sales_missing_values']} / "
        f"{profile['inventory_missing_values']}.",
        f"- Zero-sales rows: {profile['zero_sales_rows']}.",
        f"- Stock-out rows: {profile['stockout_rows']} "
        f"({profile['stockout_rate']:.2%}); these are treated as censored demand and excluded "
        "from model fitting and scoring.",
        f"- Promo-marked sales rows: {profile['promo_rows']}.",
        f"- Product-level IQR outlier flags on uncensored rows: "
        f"{profile['uncensored_outlier_rows_iqr']}.",
        f"- History per product: {profile['per_product_history_min']} to "
        f"{profile['per_product_history_max']} days.",
        "- Uncensored quantity quantiles (units): "
        + ", ".join(f"{float(q):g}th={float(v):g}" for q, v in quantiles.items()),
        "",
        "## Category coverage",
        "",
    ]
    lines.extend(f"- {category}: {count} products" for category, count in categories.items())
    lines.extend(["", "## Calendar events", ""])
    lines.extend(
        f"- {event['date']}: {event['event_name']} ({event['event_type'] or 'unspecified'}"
        f"{', promotion' if event['promo_event'] else ''})"
        for event in events
    )
    lines.extend(
        [
            "",
            "## Preprocessing decisions",
            "",
            "- Dates are parsed and checked; duplicate keys, invalid flags, negative quantities, "
            "missing required fields, and inventory/product mismatches stop the pipeline.",
            "- Stock-out sales are preserved in the audit CSV but their demand target is null; "
            "they do not become false zero-demand labels.",
            "- The inspection CSV includes per-product 99th-percentile winsorized demand and an "
            "outlier flag. Backtest fitting computes its caps from training data only.",
            "- Calendar events and promotion indicators are joined by date; unknown future "
            "event categories are supported by one-hot encoders.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
