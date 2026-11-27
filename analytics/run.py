"""Run data profiling, preprocessing, model comparison, backtest, and replenishment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .backtest import write_backtest_outputs, write_benchmark_report
from .data import load_inputs, preprocess_sales, profile_data, write_quality_report
from .decision import decide
from .forecast import benchmark_models, fit_final_model, forecast_horizon


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    horizon: int = 7,
    holdout_days: int = 28,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sales, inventory, calendar = load_inputs(data_dir)
    preprocessed = preprocess_sales(sales)
    preprocessed.to_csv(output_dir / "preprocessed_sales.csv", index=False)
    write_quality_report(profile_data(sales, inventory, calendar), output_dir / "data_quality_report.md")

    benchmark = benchmark_models(sales, calendar, holdout_days=holdout_days)
    write_backtest_outputs(benchmark, output_dir)
    write_benchmark_report(
        benchmark,
        holdout_days,
        output_dir / "benchmark_report.md",
    )

    final_model = fit_final_model(sales, calendar, benchmark.best_model_name)
    future, daily_forecasts = forecast_horizon(
        sales,
        calendar,
        inventory,
        final_model,
        benchmark.best_model_name,
        horizon,
    )
    daily_forecasts.to_csv(output_dir / "daily_forecasts.csv", index=False)
    recommendations = decide(
        future,
        benchmark.residual_sigma,
        benchmark.global_residual_sigma,
        horizon=horizon,
    )
    recommendations.to_csv(output_dir / "forecast_recommendations.csv", index=False)

    print(f"Selected model: {benchmark.best_model_name}")
    print(f"Outputs written to: {output_dir.resolve()}")
    print(f"Holdout rows scored: {int(benchmark.metrics['rows_scored'].iloc[0])}")
    print(f"Products benchmarked: {len(benchmark.product_metrics)}")
    print(f"Forecast horizon: {horizon} days")


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repository_root)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--horizon", type=int, default=7, choices=(3, 7, 14))
    parser.add_argument("--holdout-days", type=int, default=28)
    args = parser.parse_args()
    run_pipeline(args.data_dir, args.output_dir, args.horizon, args.holdout_days)


if __name__ == "__main__":
    main()
