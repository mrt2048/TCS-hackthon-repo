"""Translate demand forecasts into replenishment risk and order recommendations."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd


def decide(
    forecast: pd.DataFrame,
    residual_sigma: dict[str, float],
    global_residual_sigma: float,
    horizon: int = 7,
    service_level: float = 0.95,
) -> pd.DataFrame:
    if not 0 < service_level < 1:
        raise ValueError("service_level must be strictly between 0 and 1")
    result = forecast.copy()
    result["position"] = result["current_stock"] + result["incoming_on_order"]
    if "lead_time_demand" not in result.columns:
        raise ValueError("forecast data must include daily-summed lead_time_demand")
    result["sigma_daily"] = result["product_id"].map(residual_sigma).fillna(global_residual_sigma)
    result["days_cover"] = np.divide(
        result["position"],
        result["daily_forecast"],
        out=np.full(len(result), np.inf, dtype=float),
        where=result["daily_forecast"].to_numpy() > 0,
    )
    result["stockout_prob"] = result.apply(
        lambda row: (
            1.0
            if row["lead_time_demand"] > row["position"] and row["sigma_daily"] <= 0
            else 0.0
            if row["sigma_daily"] <= 0
            else 1.0
            - _normal_cdf(
                (row["position"] - row["lead_time_demand"])
                / (row["sigma_daily"] * math.sqrt(max(int(row["lead_time_days"]), 1)))
            )
        ),
        axis=1,
    )
    result["stockout_risk"] = np.select(
        [
            result["days_cover"] < result["lead_time_days"],
            result["days_cover"] < result["lead_time_days"] + 2,
        ],
        ["HIGH", "MEDIUM"],
        default="LOW",
    )
    result["safety_stock"] = result.apply(
        lambda row: (
            NormalDist().inv_cdf(0.90)
            if int(row["shelf_life_days"]) <= 7
            else NormalDist().inv_cdf(service_level)
        )
        * row["sigma_daily"]
        * math.sqrt(max(int(row["lead_time_days"]), 1)),
        axis=1,
    )
    result["order_qty"] = np.maximum(
        0, np.ceil(result["horizon_demand"] + result["safety_stock"] - result["position"])
    ).astype(int)
    perishable = result["shelf_life_days"] <= 7
    perishable_cap = np.maximum(
        0,
        np.ceil(
            result["daily_forecast"]
            * np.minimum(result["shelf_life_days"], horizon)
            - result["position"]
        ),
    ).astype(int)
    result.loc[perishable, "order_qty"] = np.minimum(
        result.loc[perishable, "order_qty"], perishable_cap.loc[perishable]
    )
    expected_exposure = result["daily_forecast"] * np.minimum(
        result["shelf_life_days"], horizon
    )
    result["excess_risk"] = np.select(
        [
            result["position"] > 2.0 * expected_exposure,
            result["position"] > 1.5 * expected_exposure,
        ],
        ["HIGH", "MEDIUM"],
        default="LOW",
    )
    shortfall = np.maximum(result["lead_time_demand"] - result["position"], 0)
    result["priority"] = result["stockout_prob"] * result["selling_price"] * shortfall
    result["high_risk_alert"] = result["stockout_risk"].eq("HIGH")
    result["order_status"] = np.where(result["order_qty"].gt(0), "Order recommended", "No order needed")
    return result.sort_values(["priority", "stockout_prob"], ascending=False).reset_index(drop=True)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
