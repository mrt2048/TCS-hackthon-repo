"""Model training, temporal validation, metrics, and future forecast generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from .features import FEATURE_COLUMNS, CATEGORICAL_FEATURES, NUMERIC_FEATURES, make_training_target, supervised_frame


@dataclass
class BenchmarkResult:
    metrics: pd.DataFrame
    product_metrics: pd.DataFrame
    predictions: pd.DataFrame
    best_model_name: str
    best_model: Pipeline
    residual_sigma: dict[str, float]
    global_residual_sigma: float
    feature_importance: pd.DataFrame


def _make_model(name: str) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    if name == "RandomForest":
        regressor: Any = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=42,
            n_jobs=-1,
        )
    elif name == "XGBoost":
        regressor = XGBRegressor(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="reg:squarederror",
            eval_metric="mae",
            random_state=42,
            n_jobs=-1,
        )
    elif name == "CatBoost":
        regressor = CatBoostRegressor(
            iterations=350,
            depth=6,
            learning_rate=0.04,
            loss_function="RMSE",
            verbose=False,
            random_seed=42,
            thread_count=-1,
            allow_writing_files=False,
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("preprocess", preprocess), ("regressor", regressor)])


def _metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = predicted - actual
    denominator = float(np.abs(actual).sum())
    return {
        "wape": float(np.abs(errors).sum() / denominator) if denominator else float("nan"),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "bias": float(errors.mean()),
    }


def _feature_importance(model: Pipeline, name: str) -> pd.DataFrame:
    encoder = model.named_steps["preprocess"]
    regressor = model.named_steps["regressor"]
    importances = getattr(regressor, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["model", "feature", "importance"])
    feature_names = encoder.get_feature_names_out()
    return pd.DataFrame(
        {"model": name, "feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)


def benchmark_models(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    holdout_days: int = 28,
) -> BenchmarkResult:
    panel = supervised_frame(sales, calendar)
    split_date = panel["date"].max() - pd.Timedelta(days=holdout_days - 1)
    training = panel.loc[panel["date"] < split_date].copy()
    testing = panel.loc[panel["date"] >= split_date].copy()
    training = training.loc[training["demand_observed"].notna()]
    scored = testing.loc[testing["demand_observed"].notna()].copy()
    if training.empty or scored.empty:
        raise ValueError("Not enough uncensored history to fit or score the temporal holdout")

    x_train = training[FEATURE_COLUMNS]
    x_scored = scored[FEATURE_COLUMNS]
    y_train = make_training_target(training)
    scored["actual"] = scored["demand_observed"]
    scored["naive_7d"] = scored["mean_7"].fillna(y_train.median()).clip(lower=0)

    fitted_models: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {
        "Naive7DayAverage": scored["naive_7d"].to_numpy(dtype=float)
    }
    for name in ("RandomForest", "XGBoost", "CatBoost"):
        model = _make_model(name)
        model.fit(x_train, y_train)
        fitted_models[name] = model
        predictions[name] = np.maximum(model.predict(x_scored), 0.0)

    metric_rows = []
    for name, predicted in predictions.items():
        metric_rows.append({"model": name, "rows_scored": len(scored), **_metric_values(
            scored["actual"].to_numpy(dtype=float), predicted
        )})
    metrics = pd.DataFrame(metric_rows).sort_values(["wape", "mae"]).reset_index(drop=True)
    best_model_name = str(metrics.iloc[0]["model"])
    if best_model_name == "Naive7DayAverage":
        best_model_name = min(
            (name for name in fitted_models),
            key=lambda name: float(metrics.loc[metrics["model"].eq(name), "wape"].iloc[0]),
        )

    prediction_rows = testing[
        ["date", "product_id", "product_name", "category", "stockout_flag"]
    ].copy()
    prediction_rows["actual"] = testing["demand_observed"]
    for name, values in predictions.items():
        prediction_rows[name] = np.nan
        prediction_rows.loc[scored.index, name] = values
    prediction_rows["included_in_metrics"] = testing["demand_observed"].notna()
    prediction_rows = prediction_rows.sort_values(["date", "product_id"])

    product_rows: list[dict[str, object]] = []
    best_predictions = predictions[best_model_name]
    for product_id, group in scored.groupby("product_id"):
        positions = scored.index.get_indexer(group.index)
        actual = group["actual"].to_numpy(dtype=float)
        predicted = best_predictions[positions]
        product_rows.append(
            {
                "product_id": product_id,
                "product_name": group["product_name"].iloc[0],
                "category": group["category"].iloc[0],
                "rows_scored": len(group),
                **_metric_values(actual, predicted),
            }
        )
    product_metrics = pd.DataFrame(product_rows).sort_values("product_id").reset_index(drop=True)

    best_residuals = scored["actual"].to_numpy(dtype=float) - best_predictions
    residual_frame = pd.DataFrame(
        {"product_id": scored["product_id"].to_numpy(), "residual": best_residuals}
    )
    residual_sigma = residual_frame.groupby("product_id")["residual"].std().dropna().to_dict()
    global_residual_sigma = float(np.std(best_residuals, ddof=1))
    feature_importance = pd.concat(
        [_feature_importance(model, name) for name, model in fitted_models.items()],
        ignore_index=True,
    )
    return BenchmarkResult(
        metrics=metrics,
        product_metrics=product_metrics,
        predictions=prediction_rows,
        best_model_name=best_model_name,
        best_model=fitted_models[best_model_name],
        residual_sigma={str(key): float(value) for key, value in residual_sigma.items()},
        global_residual_sigma=global_residual_sigma,
        feature_importance=feature_importance,
    )


def fit_final_model(
    sales: pd.DataFrame, calendar: pd.DataFrame, model_name: str
) -> Pipeline:
    panel = supervised_frame(sales, calendar)
    training = panel.loc[panel["demand_observed"].notna()]
    model = _make_model(model_name)
    model.fit(training[FEATURE_COLUMNS], make_training_target(training))
    return model


def forecast_horizon(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    inventory: pd.DataFrame,
    model: Pipeline,
    model_name: str,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recursively forecast future dates, using only historical or prior predicted demand."""
    last_sales_date = sales["date"].max()
    future_calendar = calendar.loc[calendar["date"] > last_sales_date].head(horizon)
    if len(future_calendar) < horizon:
        raise ValueError(
            f"calendar.csv contains only {len(future_calendar)} dates after {last_sales_date.date()}, "
            f"but the requested horizon is {horizon}"
        )
    history = sales.copy()
    products = sales.sort_values("date").groupby("product_id", as_index=False).tail(1)
    forecasts: list[pd.DataFrame] = []
    for date in future_calendar["date"]:
        day_sales = products[
            ["product_id", "product_name", "category", "selling_price"]
        ].copy()
        day_sales["date"] = date
        day_sales["quantity_sold"] = np.nan
        day_sales["stockout_flag"] = 0
        day_sales["promo_flag"] = int(
            calendar.loc[calendar["date"].eq(date), "promo_event"].iloc[0]
        )
        feature_panel = supervised_frame(
            pd.concat([history, day_sales], ignore_index=True),
            calendar,
        )
        today = feature_panel.loc[feature_panel["date"].eq(date)].copy()
        today["daily_forecast"] = np.maximum(model.predict(today[FEATURE_COLUMNS]), 0.0)
        today["forecast_model"] = model_name
        forecasts.append(today)
        observed_predictions = day_sales.copy()
        observed_predictions["quantity_sold"] = today["daily_forecast"].to_numpy()
        history = pd.concat([history, observed_predictions], ignore_index=True)

    daily = pd.concat(forecasts, ignore_index=True)
    summary = (
        daily.groupby(["product_id", "product_name", "category"], as_index=False)
        .agg(
            horizon_demand=("daily_forecast", "sum"),
            daily_forecast=("daily_forecast", "mean"),
            forecast_days=("date", "nunique"),
            selling_price=("selling_price", "last"),
            forecast_model=("forecast_model", "first"),
        )
    )
    event_days = future_calendar.loc[future_calendar["event_name"].ne(""), "event_name"]
    summary["forecast_dates"] = ", ".join(future_calendar["date"].dt.strftime("%Y-%m-%d"))
    summary["event_days"] = ", ".join(event_days.astype(str))
    summary["promo_days"] = int(future_calendar["promo_event"].sum())
    summary["holiday_days"] = int(future_calendar["is_holiday"].sum())
    result = summary.merge(
        inventory,
        on=["product_id", "product_name", "category"],
        how="left",
        validate="one_to_one",
    )
    daily = daily.merge(
        inventory[["product_id", "lead_time_days"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    daily["horizon_day"] = daily.groupby("product_id").cumcount()
    lead_demand = (
        daily.loc[daily["horizon_day"] < daily["lead_time_days"]]
        .groupby("product_id")["daily_forecast"]
        .sum()
        .rename("lead_time_demand")
    )
    recommendations = result.merge(
        lead_demand, on="product_id", how="left", validate="one_to_one"
    )
    daily_output = daily[
        [
            "date",
            "product_id",
            "product_name",
            "category",
            "daily_forecast",
            "promotion_active",
            "is_holiday",
            "event_name",
            "forecast_model",
        ]
    ].sort_values(["date", "product_id"])
    return recommendations, daily_output
