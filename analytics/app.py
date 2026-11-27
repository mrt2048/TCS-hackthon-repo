"""Streamlit dashboard for the retail replenishment analytics deliverables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ANALYTICS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ANALYTICS_DIR / "outputs"
REQUIRED_OUTPUTS = (
    "data_quality_report.md",
    "benchmark_report.md",
    "model_metrics.csv",
    "product_metrics.csv",
    "backtest_predictions.csv",
    "daily_forecasts.csv",
    "preprocessed_sales.csv",
    "feature_importance.csv",
    "forecast_recommendations.csv",
)


@st.cache_data
def load_outputs(
    output_dir: Path, mtime_signature: tuple[int, ...]
) -> dict[str, object]:
    """Load generated artifacts once per Streamlit session."""
    del mtime_signature  # Included in the cache key so pipeline reruns refresh the dashboard.
    return {
        "data_quality_report": (output_dir / "data_quality_report.md").read_text(encoding="utf-8"),
        "benchmark_report": (output_dir / "benchmark_report.md").read_text(encoding="utf-8"),
        "model_metrics": pd.read_csv(output_dir / "model_metrics.csv"),
        "product_metrics": pd.read_csv(output_dir / "product_metrics.csv"),
        "backtest_predictions": pd.read_csv(
            output_dir / "backtest_predictions.csv", parse_dates=["date"]
        ),
        "daily_forecasts": pd.read_csv(
            output_dir / "daily_forecasts.csv", parse_dates=["date"]
        ),
        "preprocessed_sales": pd.read_csv(
            output_dir / "preprocessed_sales.csv", parse_dates=["date"]
        ),
        "feature_importance": pd.read_csv(output_dir / "feature_importance.csv"),
        "recommendations": pd.read_csv(output_dir / "forecast_recommendations.csv"),
    }


def _show_missing_outputs() -> None:
    missing = [name for name in REQUIRED_OUTPUTS if not (OUTPUT_DIR / name).is_file()]
    if missing:
        st.error("Analytics outputs are missing: " + ", ".join(missing))
        st.markdown("Generate them from the repository root, then refresh this page:")
        st.code("python -m analytics.run", language="powershell")
        st.stop()


def _render_overview(data: dict[str, object]) -> None:
    recommendations = data["recommendations"]
    metrics = data["model_metrics"]
    sales = data["preprocessed_sales"]
    best_model = metrics.loc[metrics["model"].ne("Naive7DayAverage")].sort_values("wape").iloc[0]
    high_risk = recommendations["stockout_risk"].eq("HIGH")
    excess = recommendations["excess_risk"].eq("HIGH")

    st.subheader("Replenishment overview")
    columns = st.columns(5)
    columns[0].metric("Products", f"{recommendations['product_id'].nunique()}")
    columns[1].metric("High stock-out risk", f"{int(high_risk.sum())}")
    columns[2].metric("High excess risk", f"{int(excess.sum())}")
    columns[3].metric("Units to order", f"{int(recommendations['order_qty'].sum()):,}")
    columns[4].metric("Best model WAPE", f"{best_model['wape']:.1%}")

    if high_risk.any():
        names = recommendations.loc[high_risk, "product_name"].head(8).tolist()
        st.error("Urgent replenishment: " + ", ".join(names))
    else:
        st.success("No products are currently classified as high stock-out risk.")

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Highest-priority recommendations")
        table_columns = [
            "product_name",
            "current_stock",
            "incoming_on_order",
            "days_cover",
            "horizon_demand",
            "order_qty",
            "stockout_risk",
            "excess_risk",
            "priority",
        ]
        st.dataframe(
            recommendations[table_columns].head(12),
            hide_index=True,
            use_container_width=True,
            column_config={
                "days_cover": st.column_config.NumberColumn(format="%.1f days"),
                "horizon_demand": st.column_config.NumberColumn(format="%.1f"),
                "priority": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    with right:
        st.markdown("#### Forecast model comparison")
        comparison = metrics.sort_values("wape")
        chart = px.bar(
            comparison,
            x="model",
            y="wape",
            color="model",
            labels={"model": "Model", "wape": "WAPE"},
            title="Lower WAPE indicates better forecast accuracy",
        )
        chart.update_layout(showlegend=False, yaxis_tickformat=".0%")
        st.plotly_chart(chart, use_container_width=True)
        stockout_rows = int(sales["stockout_flag"].sum())
        st.caption(
            f"{stockout_rows} historical product-days are flagged as stock-outs. "
            "Their observed sales are censored and are not treated as true demand."
        )


def _render_product_detail(data: dict[str, object]) -> None:
    sales = data["preprocessed_sales"]
    forecasts = data["daily_forecasts"]
    recommendations = data["recommendations"]
    products = (
        recommendations[["product_id", "product_name"]]
        .drop_duplicates()
        .sort_values("product_name")
    )
    product_labels = dict(zip(products["product_id"], products["product_name"]))
    selected_id = st.selectbox(
        "Choose a product",
        products["product_id"].tolist(),
        format_func=lambda product_id: f"{product_labels[product_id]} ({product_id})",
    )
    product_sales = sales.loc[sales["product_id"].eq(selected_id)].sort_values("date")
    product_forecast = forecasts.loc[forecasts["product_id"].eq(selected_id)].sort_values("date")
    recommendation = recommendations.loc[
        recommendations["product_id"].eq(selected_id)
    ].iloc[0]

    cards = st.columns(4)
    cards[0].metric("Forecast demand", f"{recommendation['horizon_demand']:.1f} units")
    cards[1].metric("Recommended order", f"{int(recommendation['order_qty'])} units")
    cards[2].metric("Stock-out risk", str(recommendation["stockout_risk"]))
    cards[3].metric("Stock-out probability", f"{recommendation['stockout_prob']:.1%}")

    chart = go.Figure()
    observed = product_sales.loc[product_sales["stockout_flag"].eq(0)]
    censored = product_sales.loc[product_sales["stockout_flag"].eq(1)]
    chart.add_trace(
        go.Scatter(
            x=observed["date"],
            y=observed["quantity_sold"],
            mode="lines",
            name="Historical sales (uncensored days)",
            line={"color": "#2f6f9f"},
        )
    )
    if not censored.empty:
        chart.add_trace(
            go.Scatter(
                x=censored["date"],
                y=censored["quantity_sold"],
                mode="markers",
                name="Observed sales on stock-out days (censored)",
                marker={"color": "#e38b2c", "symbol": "x", "size": 8},
            )
        )
    chart.add_trace(
        go.Scatter(
            x=product_forecast["date"],
            y=product_forecast["daily_forecast"],
            mode="lines+markers",
            name="Selected-model forecast",
            line={"color": "#168c68", "dash": "dash"},
        )
    )
    event_forecast = product_forecast.loc[
        product_forecast["event_name"].fillna("").ne("")
        | product_forecast["promotion_active"].eq(1)
        | product_forecast["is_holiday"].eq(1)
    ]
    if not event_forecast.empty:
        chart.add_trace(
            go.Scatter(
                x=event_forecast["date"],
                y=event_forecast["daily_forecast"],
                mode="markers+text",
                text=event_forecast["event_name"].fillna("Promotion"),
                textposition="top center",
                name="Forecast event/promotion",
                marker={"color": "#9b59b6", "size": 10},
            )
        )
    chart.update_layout(
        title=f"Sales history and {int(recommendation['forecast_days'])}-day forecast",
        xaxis_title="Date",
        yaxis_title="Units",
        legend_title="Series",
        hovermode="x unified",
    )
    st.plotly_chart(chart, use_container_width=True)

    with st.expander("Why this recommendation?"):
        position = float(recommendation["position"])
        safety_stock = float(recommendation["safety_stock"])
        demand = float(recommendation["horizon_demand"])
        uncapped_order = max(0, demand + safety_stock - position)
        st.markdown(
            f"""
            - **Forecast:** {demand:.1f} units over {int(recommendation['forecast_days'])} days,
              from {recommendation['forecast_model']}.
            - **Stock position:** {recommendation['current_stock']:.0f} on hand +
              {recommendation['incoming_on_order']:.0f} incoming = {position:.0f} units.
            - **Safety stock:** {safety_stock:.1f} units, based on holdout forecast residuals,
              lead time, and the service-level assumption.
            - **Order calculation:** max(0, horizon demand + safety stock - stock position)
              = max(0, {demand:.1f} + {safety_stock:.1f} - {position:.1f})
              = {uncapped_order:.1f} units before rounding/caps.
            - **Lead time:** {int(recommendation['lead_time_days'])} days; forecast demand over
              that lead time is {recommendation['lead_time_demand']:.1f} units.
            - **Calendar:** {int(recommendation['promo_days'])} promotion day(s);
              events in horizon: {recommendation['event_days'] or 'none'}.
            - **Risk:** {recommendation['stockout_risk']} stock-out risk and
              {recommendation['excess_risk']} excess risk.
            """
        )
        if int(recommendation["shelf_life_days"]) <= 7:
            st.info(
                "This is perishable stock. The suggested order is capped to forecast demand "
                "within its shelf life."
            )
    st.markdown("#### Per-product holdout score")
    product_metrics = data["product_metrics"]
    score = product_metrics.loc[product_metrics["product_id"].eq(selected_id)]
    if not score.empty:
        st.dataframe(score, hide_index=True, use_container_width=True)


def _render_backtest(data: dict[str, object]) -> None:
    metrics = data["model_metrics"].sort_values("wape")
    predictions = data["backtest_predictions"]
    winner = metrics.loc[metrics["model"].ne("Naive7DayAverage")].iloc[0]
    baseline = metrics.loc[metrics["model"].eq("Naive7DayAverage")].iloc[0]
    improvement = (baseline["wape"] - winner["wape"]) / baseline["wape"]
    left, right = st.columns(2)
    left.metric("Best ML model", str(winner["model"]))
    right.metric("WAPE improvement vs 7-day baseline", f"{improvement:.1%}")

    st.markdown("#### Chronological holdout comparison")
    display_metrics = metrics.copy()
    display_metrics["wape"] = display_metrics["wape"].map(lambda value: f"{value:.2%}")
    st.dataframe(
        display_metrics,
        hide_index=True,
        use_container_width=True,
    )
    chart = px.bar(
        metrics,
        x="model",
        y="wape",
        color="model",
        labels={"model": "Model", "wape": "WAPE"},
        title="Forecast error by model (lower is better)",
    )
    chart.update_layout(showlegend=False, yaxis_tickformat=".0%")
    st.plotly_chart(chart, use_container_width=True)

    eligible = predictions.loc[predictions["included_in_metrics"].astype(bool)].copy()
    daily = (
        eligible.groupby("date", as_index=False)
        .agg(actual=("actual", "sum"), forecast=(str(winner["model"]), "sum"))
        .sort_values("date")
    )
    timeseries = go.Figure()
    timeseries.add_trace(go.Scatter(x=daily["date"], y=daily["actual"], mode="lines", name="Actual"))
    timeseries.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["forecast"],
            mode="lines",
            name=str(winner["model"]),
        )
    )
    timeseries.update_layout(
        title="Daily total sales vs. selected-model predictions",
        xaxis_title="Holdout date",
        yaxis_title="Units across scored products",
        hovermode="x unified",
    )
    st.plotly_chart(timeseries, use_container_width=True)
    excluded = int((~predictions["included_in_metrics"].astype(bool)).sum())
    st.caption(
        f"Last 28 sales dates; {int(winner['rows_scored'])} uncensored product-days scored. "
        f"{excluded} stock-out rows were excluded because their demand is censored."
    )
    st.warning(
        "This is a forecast-accuracy backtest, not a simulated inventory-policy replay. "
        "The supplied inventory data is a current snapshot, without historical daily stock "
        "balances or receipt transactions."
    )


def _render_methodology(data: dict[str, object]) -> None:
    st.subheader("Plan requirements and delivered analytics")
    st.markdown(
        """
        The plan calls for explainable demand forecasts, replenishment quantities, stock-out and
        excess-risk identification, promotion/calendar awareness, urgency ranking, and comparison
        against historical outcomes. The analytics pipeline supplies the forecast, order/risk
        outputs, model benchmark, and a product-level explanation of the order calculation.
        """
    )
    st.markdown("#### Dataflow")
    st.code(
        "sales.csv + inventory.csv + calendar.csv\n"
        "  -> schema/value validation and data profiling (analytics.data)\n"
        "  -> censor stock-out demand; winsorize training targets; create lag/calendar features\n"
        "  -> chronological 28-day benchmark: Random Forest / XGBoost / CatBoost / naive baseline\n"
        "  -> refit best ML model; recursively forecast future calendar dates\n"
        "  -> decision engine: order quantity, stock-out/excess risk, priority (analytics.decision)\n"
        "  -> CSV/Markdown deliverables in analytics/outputs/\n"
        "  -> Streamlit dashboard reads and explains those generated artifacts",
        language="text",
    )
    st.markdown("#### Key assumptions")
    st.markdown(
        """
        - Planning horizon defaults to 7 days; the pipeline supports 3, 7, or 14 days.
        - The forecast begins on the day after the last sales date and uses known calendar events.
        - Stock-out observations are censored: preserved for review, excluded from fitting/scoring,
          and never treated as zero demand.
        - The final model is selected by lowest WAPE among the requested ML models; the 7-day
          average is a baseline, not a candidate winner.
        - Service level defaults to 95%; perishables with shelf life up to 7 days use 90%.
        - Inventory position includes on-hand plus incoming orders. Perishable orders are capped
          by forecast demand within shelf life.
        - Stock-out probabilities use a normal approximation to residual uncertainty and should
          be treated as estimates on this short synthetic history.
        """
    )
    st.markdown("#### Data-quality report")
    st.markdown(data["data_quality_report"])
    with st.expander("Full benchmark report"):
        st.markdown(data["benchmark_report"])


def main() -> None:
    st.set_page_config(
        page_title="Retail Stock & Replenishment Advisor",
        page_icon="📦",
        layout="wide",
    )
    st.title("Retail Smart Stock & Replenishment Advisor")
    st.caption(
        "A Streamlit frontend for the analytics deliverables described in plan.md. "
        "This page explains model results and recommendations; the pipeline generates the data."
    )
    _show_missing_outputs()
    mtime_signature = tuple(
        (OUTPUT_DIR / name).stat().st_mtime_ns for name in REQUIRED_OUTPUTS
    )
    data = load_outputs(OUTPUT_DIR, mtime_signature)
    overview, product, backtest, methodology = st.tabs(
        ["Overview", "Product detail", "Backtest & models", "Data, plan & methodology"]
    )
    with overview:
        _render_overview(data)
    with product:
        _render_product_detail(data)
    with backtest:
        _render_backtest(data)
    with methodology:
        _render_methodology(data)


if __name__ == "__main__":
    main()
