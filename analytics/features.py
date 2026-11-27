"""Leakage-safe daily features for panel demand forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = [
    "product_id",
    "category",
    "day_of_week",
    "day_type",
    "event_type",
    "event_name",
]
NUMERIC_FEATURES = [
    "selling_price",
    "promotion_active",
    "is_holiday",
    "promo_event",
    "day_of_month",
    "month",
    "day_of_year",
    "time_index",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "mean_7",
    "mean_28",
    "std_7",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def make_panel(sales: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    panel = sales.merge(calendar, on="date", how="left", validate="many_to_one")
    panel["demand_observed"] = panel["quantity_sold"].where(panel["stockout_flag"].eq(0))
    panel["promotion_active"] = np.maximum(panel["promo_flag"], panel["promo_event"])
    panel["day_of_month"] = panel["date"].dt.day
    panel["month"] = panel["date"].dt.month
    panel["day_of_year"] = panel["date"].dt.dayofyear
    panel["time_index"] = (panel["date"] - panel["date"].min()).dt.days
    return panel.sort_values(["product_id", "date"]).reset_index(drop=True)


def add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Create lag/rolling values using only earlier calendar days for each product."""
    result = panel.copy()
    grouped = result.groupby("product_id", sort=False)["demand_observed"]
    for lag in (1, 7, 14, 28):
        result[f"lag_{lag}"] = grouped.shift(lag)
    for window in (7, 28):
        result[f"mean_{window}"] = grouped.transform(
            lambda values: values.shift(1).rolling(window, min_periods=1).mean()
        )
    result["std_7"] = grouped.transform(
        lambda values: values.shift(1).rolling(7, min_periods=2).std()
    )
    return result


def supervised_frame(sales: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    return add_lag_features(make_panel(sales, calendar))


def make_training_target(training: pd.DataFrame) -> pd.Series:
    """Winsorize fitting labels at a per-product 99th percentile from training only."""
    target = training["demand_observed"].copy()
    caps = target.groupby(training["product_id"]).transform(lambda values: values.quantile(0.99))
    return target.clip(upper=caps)
