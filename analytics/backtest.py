"""Persist temporal holdout results and a compact benchmark report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .forecast import BenchmarkResult


def write_backtest_outputs(result: BenchmarkResult, output_dir: Path) -> None:
    result.metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    result.product_metrics.to_csv(output_dir / "product_metrics.csv", index=False)
    result.predictions.to_csv(output_dir / "backtest_predictions.csv", index=False)
    result.feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)


def write_benchmark_report(
    result: BenchmarkResult,
    holdout_days: int,
    output_path: Path,
) -> None:
    rows = result.metrics.to_dict("records")
    best_metric = result.metrics.loc[
        result.metrics["model"].eq(result.best_model_name)
    ].iloc[0]
    baseline_metric = result.metrics.loc[
        result.metrics["model"].eq("Naive7DayAverage")
    ].iloc[0]
    relative_wape_gain = (
        (baseline_metric["wape"] - best_metric["wape"]) / baseline_metric["wape"] * 100
        if baseline_metric["wape"] > 0
        else float("nan")
    )
    observed_stockouts = int(result.predictions["stockout_flag"].sum())
    lines = [
        "# Forecast benchmark and temporal backtest",
        "",
        f"- Evaluation: last {holdout_days} sales dates, chronologically held out.",
        "- Scoring: uncensored product-days only; stock-out targets are excluded.",
        "- Forecast cadence: one day ahead, using past observations and known calendar features.",
        "- Model fitting: per-product target caps are calculated from the training period only.",
        "- Baseline: prior 7 calendar days' mean uncensored sales.",
        f"- Selected model for final forecasts: **{result.best_model_name}**.",
        f"- Observed stock-out product-days within the holdout: {observed_stockouts}.",
        f"- Selected ML model WAPE vs. naive baseline: {relative_wape_gain:.1f}% lower.",
        "",
        "## Metrics",
        "",
        "| Model | Scored rows | WAPE | MAE | RMSE | Bias |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['model']} | {int(row['rows_scored'])} | {row['wape']:.4f} | "
        f"{row['mae']:.3f} | {row['rmse']:.3f} | {row['bias']:.3f} |"
        for row in rows
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "WAPE is the sum of absolute errors divided by the sum of actual uncensored demand; "
            "lower is better. MAE and RMSE are measured in units per product-day. Positive bias "
            "means over-forecasting. Model choice is based on the lowest WAPE among the three "
            "requested ML models (the naive baseline remains a comparator).",
            "",
            "Detailed per-product metrics and row-level predictions, including the censored rows "
            "excluded from scoring, are in `product_metrics.csv` and `backtest_predictions.csv`.",
            "",
            "A policy-level replay of historical lost units, average on-hand inventory, and "
            "excess/wastage is not estimated: the supplied files contain only a current inventory "
            "snapshot, not daily historical stock balances or receipts, and stock-out demand is "
            "censored. Reconstructing those outcomes would require unsupported assumptions.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
