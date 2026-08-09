
"""
app.py

Managerial Dash dashboard for the completed Bike Share Toronto first model.

Historical source:
    outputs/full/hourly_panel.csv, through dashboard cache aggregates

Prediction source:
    outputs/full/forecast_results_with_risk.csv

Run:
    python app.py

Open:
    http://127.0.0.1:8050
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html

from prepare_dashboard_data import build_dashboard_cache


PROJECT_DIR = Path(
    os.environ.get(
        "BIKESHARE_PROJECT_DIR",
        Path(__file__).resolve().parent,
    )
).resolve()
CACHE_DIR = PROJECT_DIR / "outputs" / "dashboard_cache"

COLORS = {
    "navy": "#17365D",
    "blue": "#2F75B5",
    "teal": "#1F7A8C",
    "green": "#548235",
    "gold": "#C89B3C",
    "orange": "#D97A2B",
    "red": "#C00000",
    "light_blue": "#D9EAF7",
    "light_green": "#E2F0D9",
    "light_gold": "#FFF2CC",
    "light_red": "#FCE4D6",
    "grey": "#667085",
    "light_grey": "#F4F6F8",
    "white": "#FFFFFF",
}

SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]
RISK_ORDER = ["Low", "Moderate", "High", "Critical"]
GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
    ],
}


def ensure_cache() -> None:
    manifest = CACHE_DIR / "dashboard_manifest.json"
    if not manifest.exists():
        print(
            "Dashboard cache was not found. "
            "Building it from the completed first-model outputs..."
        )
        build_dashboard_cache(PROJECT_DIR)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def compact_number(value: float | int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def pct(value: float | None, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{decimals}%}"


def number(value: float | None, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:,.{decimals}f}"


def card(title: str, value: str, subtitle: str = "") -> html.Div:
    return html.Div(
        className="kpi-card",
        children=[
            html.Div(title, className="kpi-title"),
            html.Div(value, className="kpi-value"),
            html.Div(subtitle, className="kpi-subtitle"),
        ],
    )


def chart_card(title: str, graph_id: str, note: str = "") -> html.Div:
    children = [
        html.Div(title, className="chart-title"),
        dcc.Graph(
            id=graph_id,
            config=GRAPH_CONFIG,
            className="dashboard-graph",
        ),
    ]
    if note:
        children.append(html.Div(note, className="chart-note"))
    return html.Div(className="chart-card", children=children)


def style_figure(
    figure: go.Figure,
    title: str | None = None,
    y_title: str | None = None,
) -> go.Figure:
    """Apply the shared visual theme to a figure.

    NOTE: `title` is accepted but deliberately NOT drawn inside the figure.
    Every chart is already wrapped in a card that renders the heading as HTML
    (`chart-title`). Drawing it again in Plotly produced a duplicated heading
    and reserved ~55px of top margin that collided with the legend, which is
    what made the charts look cramped. The argument is kept so the existing
    call sites don't need to change.
    """
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["white"],
        plot_bgcolor=COLORS["white"],
        font={"family": "Arial, sans-serif", "color": "#344054", "size": 12},
        # Top margin now only needs to clear the horizontal legend, not a title.
        margin={"l": 62, "r": 24, "t": 34, "b": 52},
        height=360,
        hovermode="x unified",
        hoverlabel={
            "bgcolor": COLORS["white"],
            "bordercolor": "#D0D5DD",
            "font": {"family": "Arial, sans-serif", "size": 12},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "left",
            "x": 0,
            "title": {"text": ""},
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 12},
        },
        title=None,
    )
    figure.update_xaxes(
        showgrid=False,
        ticks="outside",
        ticklen=4,
        tickcolor="#D0D5DD",
        automargin=True,
    )
    figure.update_yaxes(
        gridcolor="#EAECF0",
        zerolinecolor="#D0D5DD",
        title={"text": y_title, "standoff": 12} if y_title else None,
        ticks="outside",
        ticklen=4,
        tickcolor="#D0D5DD",
        automargin=True,
    )
    return figure


def empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": COLORS["grey"]},
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["white"],
        plot_bgcolor=COLORS["white"],
        height=360,  # keep card height stable when a filter returns no rows
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    return figure


def filtered_values(
    selected: list | None,
    all_values: list,
) -> list:
    return all_values if not selected else selected


ensure_cache()

RUN_SUMMARY = load_json(CACHE_DIR / "run_summary.json")
FORECAST_METRICS = load_json(CACHE_DIR / "forecast_metrics.json")

HIST_DAILY = pd.read_csv(
    CACHE_DIR / "historical_daily.csv",
    parse_dates=["date"],
)
HIST_MONTHLY = pd.read_csv(
    CACHE_DIR / "historical_monthly.csv",
    parse_dates=["month_start"],
)
HIST_HOURLY = pd.read_csv(
    CACHE_DIR / "historical_hourly_profile.csv"
)
HIST_STATION = pd.read_csv(
    CACHE_DIR / "historical_station_year_daytype.csv"
)
HIST_SEASONAL = pd.read_csv(
    CACHE_DIR / "historical_seasonal.csv"
)
CLUSTERS = pd.read_csv(CACHE_DIR / "station_clusters.csv")
FORECAST = pd.read_csv(
    CACHE_DIR / "forecast_sample_enriched.csv",
    parse_dates=["demand_hour", "date"],
    low_memory=False,
)

YEARS = sorted(int(year) for year in RUN_SUMMARY["years_found"])
CLUSTER_NAMES = sorted(
    set(HIST_DAILY["cluster_name"].dropna().astype(str))
)
FORECAST_CLUSTERS = sorted(
    set(FORECAST["cluster_name"].dropna().astype(str))
)

RAW_YEAR_COUNTS = {
    int(year): int(count)
    for year, count in RUN_SUMMARY["raw_rows_by_parsed_year"].items()
}
TOTAL_RAW = int(RUN_SUMMARY["raw_rows"])
TOTAL_CLEAN = int(RUN_SUMMARY["clean_rows"])
TOTAL_QUARANTINED = int(RUN_SUMMARY["quarantined_rows"])
TOTAL_MISSING_REQUIRED = (
    TOTAL_RAW - TOTAL_CLEAN - TOTAL_QUARANTINED
)
CLEAN_RETENTION = TOTAL_CLEAN / TOTAL_RAW
GROWTH_2022_2025 = (
    RAW_YEAR_COUNTS[max(YEARS)] / RAW_YEAR_COUNTS[min(YEARS)] - 1
)

# ---------- Static overview figures ----------

clean_annual = (
    HIST_DAILY.groupby("year", as_index=False)["trip_starts"]
    .sum()
    .rename(columns={"trip_starts": "Clean trip starts"})
)
recorded_annual = pd.DataFrame(
    {
        "year": list(RAW_YEAR_COUNTS.keys()),
        "Recorded trips": list(RAW_YEAR_COUNTS.values()),
    }
)
annual_comparison = recorded_annual.merge(clean_annual, on="year", how="left")
annual_long = annual_comparison.melt(
    id_vars="year",
    value_vars=["Recorded trips", "Clean trip starts"],
    var_name="Measure",
    value_name="Trips",
)
OVERVIEW_ANNUAL_FIG = px.bar(
    annual_long,
    x="year",
    y="Trips",
    color="Measure",
    barmode="group",
    color_discrete_sequence=[COLORS["blue"], COLORS["teal"]],
)
OVERVIEW_ANNUAL_FIG = style_figure(
    OVERVIEW_ANNUAL_FIG,
    "Annual Ridership: Recorded vs Clean Trip Starts",
    "Trips",
)
OVERVIEW_ANNUAL_FIG.update_yaxes(tickformat="~s")
OVERVIEW_ANNUAL_FIG.update_xaxes(dtick=1)

overview_monthly = (
    HIST_MONTHLY.groupby("month_start", as_index=False)["trip_starts"]
    .sum()
)
OVERVIEW_MONTHLY_FIG = px.line(
    overview_monthly,
    x="month_start",
    y="trip_starts",
    markers=True,
    color_discrete_sequence=[COLORS["blue"]],
)
OVERVIEW_MONTHLY_FIG = style_figure(
    OVERVIEW_MONTHLY_FIG,
    "Monthly Clean Trip Starts",
    "Trip starts",
)
OVERVIEW_MONTHLY_FIG.update_yaxes(tickformat="~s")

cluster_summary = (
    CLUSTERS.groupby("cluster_name", as_index=False)
    .agg(
        stations=("station_id", "nunique"),
        trip_starts=("trip_count", "sum"),
    )
)
cluster_summary["station_share"] = (
    cluster_summary["stations"] / cluster_summary["stations"].sum()
)
cluster_summary["trip_share"] = (
    cluster_summary["trip_starts"] / cluster_summary["trip_starts"].sum()
)
cluster_long = cluster_summary.melt(
    id_vars="cluster_name",
    value_vars=["station_share", "trip_share"],
    var_name="Measure",
    value_name="Share",
)
cluster_long["Measure"] = cluster_long["Measure"].map(
    {
        "station_share": "Share of stations",
        "trip_share": "Share of trip starts",
    }
)
OVERVIEW_CLUSTER_FIG = px.bar(
    cluster_long,
    y="cluster_name",
    x="Share",
    color="Measure",
    barmode="group",
    orientation="h",
    color_discrete_sequence=[COLORS["gold"], COLORS["teal"]],
)
OVERVIEW_CLUSTER_FIG = style_figure(
    OVERVIEW_CLUSTER_FIG,
    "Working Station Segments: Portfolio Concentration",
    "Share",
)
OVERVIEW_CLUSTER_FIG.update_xaxes(tickformat=".0%")
OVERVIEW_CLUSTER_FIG.update_yaxes(title=None)

overview_hourly = (
    HIST_HOURLY.groupby(
        ["day_type", "hour"],
        as_index=False,
    )[["trip_starts", "active_station_hours"]]
    .sum()
)
overview_hourly["avg_starts_per_active_station_hour"] = (
    overview_hourly["trip_starts"]
    / overview_hourly["active_station_hours"].replace(0, np.nan)
)
OVERVIEW_HOURLY_FIG = px.line(
    overview_hourly,
    x="hour",
    y="avg_starts_per_active_station_hour",
    color="day_type",
    markers=True,
    color_discrete_map={
        "Weekday": COLORS["blue"],
        "Weekend": COLORS["orange"],
    },
)
OVERVIEW_HOURLY_FIG = style_figure(
    OVERVIEW_HOURLY_FIG,
    "Average Hourly Demand Pattern",
    "Trip starts per active station-hour",
)
OVERVIEW_HOURLY_FIG.update_xaxes(dtick=2)

# ---------- Layout helpers ----------

historical_filters = html.Div(
    className="filter-panel",
    children=[
        html.Div(
            className="filter-item",
            children=[
                html.Label("Years"),
                dcc.Dropdown(
                    id="hist-years",
                    options=[
                        {"label": str(year), "value": year}
                        for year in YEARS
                    ],
                    value=YEARS,
                    multi=True,
                    clearable=False,
                ),
            ],
        ),
        html.Div(
            className="filter-item",
            children=[
                html.Label("Day type"),
                dcc.Dropdown(
                    id="hist-day-type",
                    options=[
                        {"label": "All days", "value": "All"},
                        {"label": "Weekdays", "value": "Weekday"},
                        {"label": "Weekends", "value": "Weekend"},
                    ],
                    value="All",
                    clearable=False,
                ),
            ],
        ),
        html.Div(
            className="filter-item filter-wide",
            children=[
                html.Label("Working station segment"),
                dcc.Dropdown(
                    id="hist-clusters",
                    options=[
                        {"label": name, "value": name}
                        for name in CLUSTER_NAMES
                    ],
                    value=CLUSTER_NAMES,
                    multi=True,
                    clearable=False,
                ),
            ],
        ),
    ],
)

forecast_filters = html.Div(
    className="filter-panel",
    children=[
        html.Div(
            className="filter-item",
            children=[
                html.Label("Prediction date range"),
                dcc.DatePickerRange(
                    id="forecast-dates",
                    min_date_allowed=FORECAST["date"].min().date(),
                    max_date_allowed=FORECAST["date"].max().date(),
                    start_date=FORECAST["date"].min().date(),
                    end_date=FORECAST["date"].max().date(),
                    display_format="YYYY-MM-DD",
                ),
            ],
        ),
        html.Div(
            className="filter-item",
            children=[
                html.Label("Pressure tiers"),
                dcc.Dropdown(
                    id="forecast-risks",
                    options=[
                        {"label": risk, "value": risk}
                        for risk in RISK_ORDER
                    ],
                    value=RISK_ORDER,
                    multi=True,
                    clearable=False,
                ),
            ],
        ),
        html.Div(
            className="filter-item filter-wide",
            children=[
                html.Label("Working station segment"),
                dcc.Dropdown(
                    id="forecast-clusters",
                    options=[
                        {"label": name, "value": name}
                        for name in FORECAST_CLUSTERS
                    ],
                    value=FORECAST_CLUSTERS,
                    multi=True,
                    clearable=False,
                ),
            ],
        ),
    ],
)

app = Dash(__name__)
server = app.server
app.title = "Bike Share Toronto | Management Dashboard"

app.layout = html.Div(
    className="app-shell",
    children=[
        html.Div(
            className="top-banner",
            children=[
                html.Div(
                    children=[
                        html.H1("Bike Share Toronto Management Dashboard"),
                        html.P(
                            "Historical ridership and first-model forecasts | "
                            "2022–2025"
                        ),
                    ]
                ),
                html.Div(
                    className="scope-badge",
                    children="Completed base-model scope",
                ),
            ],
        ),
        html.Div(
            className="scope-note",
            children=[
                html.Strong("Interpretation note: "),
                "The forecast estimates station-hour net bike flow for observed "
                "active station-hours. Pressure tiers indicate predicted negative "
                "flow, not confirmed empty-station events, because starting inventory "
                "and station capacity are unavailable.",
            ],
        ),
        dcc.Tabs(
            id="main-tabs",
            value="overview",
            className="main-tabs",
            children=[
                dcc.Tab(
                    label="Executive Overview",
                    value="overview",
                    children=[
                        html.Div(
                            className="tab-content",
                            children=[
                                html.Div(
                                    className="kpi-grid six",
                                    children=[
                                        card(
                                            "Recorded trips",
                                            compact_number(TOTAL_RAW),
                                            "Raw records, 2022–2025",
                                        ),
                                        card(
                                            "Clean trip retention",
                                            pct(CLEAN_RETENTION),
                                            f"{compact_number(TOTAL_CLEAN)} retained",
                                        ),
                                        card(
                                            "Stations analysed",
                                            compact_number(
                                                RUN_SUMMARY["stations"]
                                            ),
                                            "Identified by station ID",
                                        ),
                                        card(
                                            "Recorded growth",
                                            pct(GROWTH_2022_2025),
                                            f"{min(YEARS)} to {max(YEARS)}",
                                        ),
                                        card(
                                            "Base-model MAE",
                                            f"{FORECAST_METRICS['mae']:.2f}",
                                            "Bikes per sampled active station-hour",
                                        ),
                                        card(
                                            "Elevated pressure",
                                            pct(
                                                FORECAST_METRICS[
                                                    "elevated_pressure_rate"
                                                ],
                                                2,
                                            ),
                                            "Moderate, High or Critical",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="two-column-grid",
                                    children=[
                                        html.Div(
                                            className="chart-card",
                                            children=[
                                                html.Div(
                                                    "Annual Ridership",
                                                    className="chart-title",
                                                ),
                                                dcc.Graph(
                                                    figure=OVERVIEW_ANNUAL_FIG,
                                                    config=GRAPH_CONFIG,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="chart-card",
                                            children=[
                                                html.Div(
                                                    "Monthly Trend",
                                                    className="chart-title",
                                                ),
                                                dcc.Graph(
                                                    figure=OVERVIEW_MONTHLY_FIG,
                                                    config=GRAPH_CONFIG,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="chart-card",
                                            children=[
                                                html.Div(
                                                    "Station Portfolio",
                                                    className="chart-title",
                                                ),
                                                dcc.Graph(
                                                    figure=OVERVIEW_CLUSTER_FIG,
                                                    config=GRAPH_CONFIG,
                                                ),
                                                html.Div(
                                                    "Segment names are heuristic working "
                                                    "labels based on usage patterns, not "
                                                    "geographic validation.",
                                                    className="chart-note",
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="chart-card",
                                            children=[
                                                html.Div(
                                                    "Demand by Hour",
                                                    className="chart-title",
                                                ),
                                                dcc.Graph(
                                                    figure=OVERVIEW_HOURLY_FIG,
                                                    config=GRAPH_CONFIG,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="insight-panel",
                                    children=[
                                        html.H3("Management Takeaways"),
                                        html.Ul(
                                            children=[
                                                html.Li(
                                                    f"Recorded trips increased from "
                                                    f"{RAW_YEAR_COUNTS[min(YEARS)]:,} "
                                                    f"in {min(YEARS)} to "
                                                    f"{RAW_YEAR_COUNTS[max(YEARS)]:,} "
                                                    f"in {max(YEARS)}, a "
                                                    f"{GROWTH_2022_2025:.1%} increase."
                                                ),
                                                html.Li(
                                                    f"The cleaning process retained "
                                                    f"{CLEAN_RETENTION:.1%} of raw records; "
                                                    f"{TOTAL_QUARANTINED / TOTAL_RAW:.2%} "
                                                    f"were quarantined as duration outliers."
                                                ),
                                                html.Li(
                                                    "Weekdays show distinct commuting peaks, "
                                                    "while weekends have a broader afternoon "
                                                    "demand pattern."
                                                ),
                                                html.Li(
                                                    f"The base Random Forest achieved an MAE "
                                                    f"of {FORECAST_METRICS['mae']:.2f} bikes "
                                                    f"and improved on the zero-net-change "
                                                    f"benchmark by "
                                                    f"{FORECAST_METRICS['improvement_vs_zero']:.1%}."
                                                ),
                                                html.Li(
                                                    "Operational pressure charts use the "
                                                    "saved 150,000-row chronological test "
                                                    "sample, not every station-hour."
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="Historical Operations",
                    value="historical",
                    children=[
                        html.Div(
                            className="tab-content",
                            children=[
                                historical_filters,
                                html.Div(
                                    id="hist-kpis",
                                    className="kpi-grid four",
                                ),
                                html.Div(
                                    className="two-column-grid",
                                    children=[
                                        chart_card(
                                            "Monthly Trip Starts",
                                            "hist-monthly-fig",
                                        ),
                                        chart_card(
                                            "Average Demand by Hour",
                                            "hist-hourly-fig",
                                            "Average is calculated per observed active "
                                            "station-hour.",
                                        ),
                                        chart_card(
                                            "Seasonal Demand",
                                            "hist-seasonal-fig",
                                        ),
                                        chart_card(
                                            "Top Stations by Trip Starts",
                                            "hist-top-stations-fig",
                                            "Station names were not retained in the "
                                            "memory-optimized first-model output.",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="table-card",
                                    children=[
                                        html.Div(
                                            "Top Station Detail",
                                            className="chart-title",
                                        ),
                                        dash_table.DataTable(
                                            id="hist-station-table",
                                            columns=[
                                                {
                                                    "name": "Station ID",
                                                    "id": "station_id",
                                                },
                                                {
                                                    "name": "Working Segment",
                                                    "id": "cluster_name",
                                                },
                                                {
                                                    "name": "Trip Starts",
                                                    "id": "trip_starts",
                                                    "type": "numeric",
                                                    "format": {
                                                        "specifier": ",.0f"
                                                    },
                                                },
                                                {
                                                    "name": "Trip Ends",
                                                    "id": "trip_ends",
                                                    "type": "numeric",
                                                    "format": {
                                                        "specifier": ",.0f"
                                                    },
                                                },
                                                {
                                                    "name": "Active Station-Hours",
                                                    "id": "active_station_hours",
                                                    "type": "numeric",
                                                    "format": {
                                                        "specifier": ",.0f"
                                                    },
                                                },
                                                {
                                                    "name": "Starts / Active Hour",
                                                    "id": "starts_per_active_hour",
                                                    "type": "numeric",
                                                    "format": {
                                                        "specifier": ".2f"
                                                    },
                                                },
                                            ],
                                            data=[],
                                            page_size=12,
                                            sort_action="native",
                                            filter_action="native",
                                            style_table={
                                                "overflowX": "auto"
                                            },
                                            style_header={
                                                "backgroundColor": COLORS["navy"],
                                                "color": "white",
                                                "fontWeight": "bold",
                                            },
                                            style_cell={
                                                "padding": "10px",
                                                "fontFamily": "Arial",
                                                "fontSize": "13px",
                                                "textAlign": "left",
                                                "whiteSpace": "normal",
                                                "height": "auto",
                                            },
                                            style_data_conditional=[
                                                {
                                                    "if": {
                                                        "row_index": "odd"
                                                    },
                                                    "backgroundColor": "#F9FAFB",
                                                }
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="Forecast & Pressure",
                    value="forecast",
                    children=[
                        html.Div(
                            className="tab-content",
                            children=[
                                forecast_filters,
                                html.Div(
                                    id="forecast-kpis",
                                    className="kpi-grid five",
                                ),
                                html.Div(
                                    className="two-column-grid",
                                    children=[
                                        chart_card(
                                            "Average Actual vs Predicted Net Flow",
                                            "forecast-daily-fig",
                                            "Daily averages across the filtered sampled "
                                            "test rows.",
                                        ),
                                        chart_card(
                                            "Predicted Pressure Distribution",
                                            "forecast-risk-fig",
                                        ),
                                        chart_card(
                                            "Stations with Most Elevated Pressure Events",
                                            "forecast-top-stations-fig",
                                            "Elevated means Moderate, High or Critical "
                                            "predicted negative flow.",
                                        ),
                                        chart_card(
                                            "Forecast Error by Hour",
                                            "forecast-hour-error-fig",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="two-column-grid",
                                    children=[
                                        chart_card(
                                            "Actual vs Predicted Net Flow",
                                            "forecast-scatter-fig",
                                            "Scatter is limited to a reproducible sample "
                                            "of up to 5,000 filtered rows.",
                                        ),
                                        html.Div(
                                            className="table-card",
                                            children=[
                                                html.Div(
                                                    "Priority Station Review",
                                                    className="chart-title",
                                                ),
                                                dash_table.DataTable(
                                                    id="forecast-station-table",
                                                    columns=[
                                                        {
                                                            "name": "Station ID",
                                                            "id": "station_id",
                                                        },
                                                        {
                                                            "name": "Working Segment",
                                                            "id": "cluster_name",
                                                        },
                                                        {
                                                            "name": "Sampled Rows",
                                                            "id": "sample_rows",
                                                            "type": "numeric",
                                                        },
                                                        {
                                                            "name": "Elevated Events",
                                                            "id": "elevated_events",
                                                            "type": "numeric",
                                                        },
                                                        {
                                                            "name": "Elevated Rate",
                                                            "id": "elevated_rate",
                                                            "type": "numeric",
                                                            "format": {
                                                                "specifier": ".1%"
                                                            },
                                                        },
                                                        {
                                                            "name": "MAE",
                                                            "id": "mae",
                                                            "type": "numeric",
                                                            "format": {
                                                                "specifier": ".2f"
                                                            },
                                                        },
                                                        {
                                                            "name": "Avg Predicted Net",
                                                            "id": "avg_predicted_net",
                                                            "type": "numeric",
                                                            "format": {
                                                                "specifier": ".2f"
                                                            },
                                                        },
                                                    ],
                                                    data=[],
                                                    page_size=12,
                                                    sort_action="native",
                                                    filter_action="native",
                                                    style_table={
                                                        "overflowX": "auto"
                                                    },
                                                    style_header={
                                                        "backgroundColor": COLORS["navy"],
                                                        "color": "white",
                                                        "fontWeight": "bold",
                                                    },
                                                    style_cell={
                                                        "padding": "10px",
                                                        "fontFamily": "Arial",
                                                        "fontSize": "13px",
                                                        "textAlign": "left",
                                                        "whiteSpace": "normal",
                                                        "height": "auto",
                                                    },
                                                    style_data_conditional=[
                                                        {
                                                            "if": {
                                                                "row_index": "odd"
                                                            },
                                                            "backgroundColor": "#F9FAFB",
                                                        }
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="Method & Limitations",
                    value="method",
                    children=[
                        html.Div(
                            className="tab-content",
                            children=[
                                html.Div(
                                    className="method-grid",
                                    children=[
                                        html.Div(
                                            className="method-card",
                                            children=[
                                                html.H3("Historical processing"),
                                                html.P(
                                                    "The completed pipeline read 37 files "
                                                    "covering 2022–2025, standardized headers "
                                                    "and encodings, parsed dates, removed rows "
                                                    "missing required fields, and quarantined "
                                                    "duration outliers."
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="method-card",
                                            children=[
                                                html.H3("Forecast target"),
                                                html.P(
                                                    "The base Random Forest predicts station-hour "
                                                    "net bike movement using station ID, hour, "
                                                    "day of week, weekend indicator, season and "
                                                    "the two-hour net-flow lag."
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="method-card",
                                            children=[
                                                html.H3("Model evaluation"),
                                                html.P(
                                                    f"The saved chronological test sample contains "
                                                    f"{FORECAST_METRICS['sample_rows']:,} rows. "
                                                    f"The model MAE is "
                                                    f"{FORECAST_METRICS['mae']:.3f}, compared with "
                                                    f"{FORECAST_METRICS['zero_baseline_mae']:.3f} "
                                                    f"for a zero-net-change benchmark."
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="method-card warning",
                                            children=[
                                                html.H3("Not an inventory forecast"),
                                                html.P(
                                                    "The project has no starting bike inventory, "
                                                    "available-dock count or station capacity. "
                                                    "Predicted pressure therefore represents "
                                                    "negative flow imbalance, not a confirmed "
                                                    "shortage."
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="method-card warning",
                                            children=[
                                                html.H3("Active station-hours only"),
                                                html.P(
                                                    "The historical panel contains station-hours "
                                                    "with observed starts or ends. It is not a "
                                                    "complete station-by-hour grid containing "
                                                    "explicit zero-activity rows."
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="method-card warning",
                                            children=[
                                                html.H3("Working segment labels"),
                                                html.P(
                                                    "Station segment names were assigned from "
                                                    "duration and temporal usage patterns. They "
                                                    "were not validated with station names, "
                                                    "coordinates, neighbourhoods or land use."
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="method-card warning",
                                            children=[
                                                html.H3("Sampled predictions"),
                                                html.P(
                                                    "Forecast visuals use the saved 150,000-row "
                                                    "test sample. Event counts should be interpreted "
                                                    "as sample-based indicators rather than complete "
                                                    "operational totals."
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            className="method-card",
                                            children=[
                                                html.H3("Dashboard performance"),
                                                html.P(
                                                    "The dashboard reads compact aggregates created "
                                                    "from the 11.27-million-row hourly panel. It does "
                                                    "not reload the raw trip files or retrain the model."
                                                ),
                                            ],
                                        ),
                                    ],
                                )
                            ],
                        )
                    ],
                ),
            ],
        ),
        html.Footer(
            className="footer",
            children=(
                "Bike Share Toronto consulting capstone | "
                "Historical data and completed first-model outputs"
            ),
        ),
    ],
)


@app.callback(
    Output("hist-kpis", "children"),
    Output("hist-monthly-fig", "figure"),
    Output("hist-hourly-fig", "figure"),
    Output("hist-seasonal-fig", "figure"),
    Output("hist-top-stations-fig", "figure"),
    Output("hist-station-table", "data"),
    Input("hist-years", "value"),
    Input("hist-day-type", "value"),
    Input("hist-clusters", "value"),
)
def update_historical(
    selected_years: list[int] | None,
    selected_day_type: str,
    selected_clusters: list[str] | None,
):
    years = filtered_values(selected_years, YEARS)
    clusters = filtered_values(selected_clusters, CLUSTER_NAMES)

    daily = HIST_DAILY[
        HIST_DAILY["year"].isin(years)
        & HIST_DAILY["cluster_name"].isin(clusters)
    ].copy()
    monthly = HIST_MONTHLY[
        HIST_MONTHLY["year"].isin(years)
        & HIST_MONTHLY["cluster_name"].isin(clusters)
    ].copy()
    hourly = HIST_HOURLY[
        HIST_HOURLY["year"].isin(years)
        & HIST_HOURLY["cluster_name"].isin(clusters)
    ].copy()
    seasonal = HIST_SEASONAL[
        HIST_SEASONAL["year"].isin(years)
        & HIST_SEASONAL["cluster_name"].isin(clusters)
    ].copy()
    station = HIST_STATION[
        HIST_STATION["year"].isin(years)
        & HIST_STATION["cluster_name"].isin(clusters)
    ].copy()

    if selected_day_type != "All":
        daily = daily[daily["day_type"] == selected_day_type]
        monthly = monthly[monthly["day_type"] == selected_day_type]
        hourly = hourly[hourly["day_type"] == selected_day_type]
        seasonal = seasonal[
            seasonal["day_type"] == selected_day_type
        ]
        station = station[station["day_type"] == selected_day_type]

    if daily.empty:
        empty = empty_figure("No historical rows match these filters.")
        return (
            [
                card("Trip starts", "0"),
                card("Average daily starts", "0"),
                card("Stations represented", "0"),
                card("Starts / active hour", "—"),
            ],
            empty,
            empty,
            empty,
            empty,
            [],
        )

    total_starts = float(daily["trip_starts"].sum())
    active_station_hours = float(
        daily["active_station_hours"].sum()
    )
    daily_totals = daily.groupby("date")["trip_starts"].sum()
    station_count = int(station["station_id"].nunique())
    average_daily = float(daily_totals.mean())
    average_active_hour = (
        total_starts / active_station_hours
        if active_station_hours
        else float("nan")
    )

    kpis = [
        card(
            "Trip starts",
            compact_number(total_starts),
            "Clean historical records",
        ),
        card(
            "Average daily starts",
            compact_number(average_daily),
            "Across selected dates",
        ),
        card(
            "Stations represented",
            compact_number(station_count),
            "Station IDs in selection",
        ),
        card(
            "Starts / active hour",
            number(average_active_hour, 2),
            "Observed active station-hours",
        ),
    ]

    monthly_plot = (
        monthly.groupby(
            ["month_start", "year"],
            as_index=False,
        )["trip_starts"]
        .sum()
        .sort_values("month_start")
    )
    monthly_figure = px.line(
        monthly_plot,
        x="month_start",
        y="trip_starts",
        color="year",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    monthly_figure = style_figure(
        monthly_figure,
        "Monthly Trip Starts",
        "Trip starts",
    )
    monthly_figure.update_yaxes(tickformat="~s")

    hourly_plot = (
        hourly.groupby(
            ["day_type", "hour"],
            as_index=False,
        )[["trip_starts", "active_station_hours"]]
        .sum()
    )
    hourly_plot["avg_starts"] = (
        hourly_plot["trip_starts"]
        / hourly_plot["active_station_hours"].replace(0, np.nan)
    )
    hourly_figure = px.line(
        hourly_plot,
        x="hour",
        y="avg_starts",
        color="day_type",
        markers=True,
        color_discrete_map={
            "Weekday": COLORS["blue"],
            "Weekend": COLORS["orange"],
        },
    )
    hourly_figure = style_figure(
        hourly_figure,
        "Average Demand by Hour",
        "Trip starts per active station-hour",
    )
    hourly_figure.update_xaxes(dtick=2)

    seasonal_plot = (
        seasonal.groupby(
            ["year", "season"],
            as_index=False,
        )["trip_starts"]
        .sum()
    )
    seasonal_plot["season"] = pd.Categorical(
        seasonal_plot["season"],
        categories=SEASON_ORDER,
        ordered=True,
    )
    seasonal_plot = seasonal_plot.sort_values(["year", "season"])
    seasonal_figure = px.bar(
        seasonal_plot,
        x="year",
        y="trip_starts",
        color="season",
        barmode="stack",
        category_orders={"season": SEASON_ORDER},
        color_discrete_sequence=[
            "#6BAED6",
            "#74C476",
            "#FDD0A2",
            "#FD8D3C",
        ],
    )
    seasonal_figure = style_figure(
        seasonal_figure,
        "Seasonal Trip Starts",
        "Trip starts",
    )
    seasonal_figure.update_yaxes(tickformat="~s")
    seasonal_figure.update_xaxes(dtick=1)

    station_summary = (
        station.groupby(
            ["station_id", "cluster_name"],
            as_index=False,
        )[[
            "trip_starts",
            "trip_ends",
            "net_change",
            "active_station_hours",
        ]]
        .sum()
    )
    station_summary["starts_per_active_hour"] = (
        station_summary["trip_starts"]
        / station_summary["active_station_hours"].replace(0, np.nan)
    )
    top_station = (
        station_summary.sort_values(
            "trip_starts", ascending=False
        )
        .head(15)
        .sort_values("trip_starts")
    )
    top_station["station_label"] = (
        "Station " + top_station["station_id"].astype(str)
    )
    top_figure = px.bar(
        top_station,
        x="trip_starts",
        y="station_label",
        orientation="h",
        color="cluster_name",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    top_figure = style_figure(
        top_figure,
        "Top Stations by Trip Starts",
        "Trip starts",
    )
    top_figure.update_xaxes(tickformat="~s")
    top_figure.update_yaxes(title=None)

    table_data = (
        station_summary.sort_values(
            "trip_starts", ascending=False
        )
        .head(50)[
            [
                "station_id",
                "cluster_name",
                "trip_starts",
                "trip_ends",
                "active_station_hours",
                "starts_per_active_hour",
            ]
        ]
        .round({"starts_per_active_hour": 2})
        .to_dict("records")
    )

    return (
        kpis,
        monthly_figure,
        hourly_figure,
        seasonal_figure,
        top_figure,
        table_data,
    )


@app.callback(
    Output("forecast-kpis", "children"),
    Output("forecast-daily-fig", "figure"),
    Output("forecast-risk-fig", "figure"),
    Output("forecast-top-stations-fig", "figure"),
    Output("forecast-hour-error-fig", "figure"),
    Output("forecast-scatter-fig", "figure"),
    Output("forecast-station-table", "data"),
    Input("forecast-dates", "start_date"),
    Input("forecast-dates", "end_date"),
    Input("forecast-risks", "value"),
    Input("forecast-clusters", "value"),
)
def update_forecast(
    start_date: str,
    end_date: str,
    selected_risks: list[str] | None,
    selected_clusters: list[str] | None,
):
    risks = filtered_values(selected_risks, RISK_ORDER)
    clusters = filtered_values(
        selected_clusters,
        FORECAST_CLUSTERS,
    )

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    filtered = FORECAST[
        (FORECAST["demand_hour"] >= start)
        & (FORECAST["demand_hour"] < end)
        & FORECAST["risk_level"].isin(risks)
        & FORECAST["cluster_name"].isin(clusters)
    ].copy()

    if filtered.empty:
        empty = empty_figure("No prediction rows match these filters.")
        return (
            [
                card("Sampled rows", "0"),
                card("MAE", "—"),
                card("R²", "—"),
                card("Within 2 bikes", "—"),
                card("Elevated pressure", "—"),
            ],
            empty,
            empty,
            empty,
            empty,
            empty,
            [],
        )

    actual = filtered["net_change"].astype(float)
    predicted = filtered["predicted_net_change"].astype(float)
    errors = filtered["absolute_error"].astype(float)
    denominator = ((actual - actual.mean()) ** 2).sum()
    r2 = (
        1 - ((actual - predicted) ** 2).sum() / denominator
        if denominator
        else float("nan")
    )
    mae = float(errors.mean())
    within_two = float((errors <= 2).mean())
    elevated_rate = float(filtered["elevated_pressure"].mean())

    kpis = [
        card(
            "Sampled rows",
            compact_number(len(filtered)),
            "Chronological test sample",
        ),
        card(
            "MAE",
            number(mae, 2),
            "Bikes per sampled active station-hour",
        ),
        card(
            "R²",
            number(r2, 3),
            "Filtered sample fit",
        ),
        card(
            "Within 2 bikes",
            pct(within_two),
            "Absolute forecast error",
        ),
        card(
            "Elevated pressure",
            pct(elevated_rate, 2),
            "Moderate, High or Critical",
        ),
    ]

    daily_plot = (
        filtered.groupby("date", as_index=False)
        .agg(
            actual_net=("net_change", "mean"),
            predicted_net=("predicted_net_change", "mean"),
        )
        .sort_values("date")
    )
    daily_long = daily_plot.melt(
        id_vars="date",
        value_vars=["actual_net", "predicted_net"],
        var_name="Measure",
        value_name="Average net flow",
    )
    daily_long["Measure"] = daily_long["Measure"].map(
        {
            "actual_net": "Actual",
            "predicted_net": "Predicted",
        }
    )
    daily_figure = px.line(
        daily_long,
        x="date",
        y="Average net flow",
        color="Measure",
        color_discrete_map={
            "Actual": COLORS["navy"],
            "Predicted": COLORS["orange"],
        },
    )
    daily_figure = style_figure(
        daily_figure,
        "Average Actual vs Predicted Net Flow",
        "Average net bikes",
    )
    daily_figure.add_hline(
        y=0,
        line_dash="dot",
        line_color="#98A2B3",
    )

    risk_counts = (
        filtered.groupby("risk_level", observed=True)
        .size()
        .reindex(RISK_ORDER, fill_value=0)
        .rename("Rows")
        .reset_index()
    )
    risk_figure = px.bar(
        risk_counts,
        x="risk_level",
        y="Rows",
        category_orders={"risk_level": RISK_ORDER},
        color="risk_level",
        color_discrete_map={
            "Low": COLORS["green"],
            "Moderate": COLORS["gold"],
            "High": COLORS["orange"],
            "Critical": COLORS["red"],
        },
    )
    risk_figure = style_figure(
        risk_figure,
        "Predicted Pressure Distribution",
        "Sampled rows",
    )
    risk_figure.update_layout(showlegend=False)
    risk_figure.update_yaxes(tickformat="~s")

    station_pressure = (
        filtered.groupby(
            ["station_id", "cluster_name"],
            as_index=False,
        )
        .agg(
            sample_rows=("station_id", "size"),
            elevated_events=("elevated_pressure", "sum"),
            high_critical_events=("high_critical_pressure", "sum"),
            mae=("absolute_error", "mean"),
            avg_predicted_net=("predicted_net_change", "mean"),
        )
    )
    station_pressure["elevated_rate"] = (
        station_pressure["elevated_events"]
        / station_pressure["sample_rows"].replace(0, np.nan)
    )
    top_pressure = (
        station_pressure.sort_values(
            ["elevated_events", "elevated_rate"],
            ascending=False,
        )
        .head(15)
        .sort_values("elevated_events")
    )
    top_pressure["station_label"] = (
        "Station " + top_pressure["station_id"].astype(str)
    )
    top_pressure_figure = px.bar(
        top_pressure,
        x="elevated_events",
        y="station_label",
        orientation="h",
        color="cluster_name",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    top_pressure_figure = style_figure(
        top_pressure_figure,
        "Stations with Most Elevated Pressure Events",
        "Elevated sampled events",
    )
    top_pressure_figure.update_yaxes(title=None)

    hourly_error = (
        filtered.groupby("hour", as_index=False)
        .agg(
            mae=("absolute_error", "mean"),
            sampled_rows=("hour", "size"),
        )
    )
    hour_error_figure = px.line(
        hourly_error,
        x="hour",
        y="mae",
        markers=True,
        color_discrete_sequence=[COLORS["red"]],
    )
    hour_error_figure = style_figure(
        hour_error_figure,
        "Forecast Error by Hour",
        "MAE",
    )
    hour_error_figure.update_xaxes(dtick=2)

    scatter_source = (
        filtered.sample(
            n=min(5_000, len(filtered)),
            random_state=42,
        )
        if len(filtered) > 5_000
        else filtered
    )
    scatter_figure = px.scatter(
        scatter_source,
        x="net_change",
        y="predicted_net_change",
        color="risk_level",
        category_orders={"risk_level": RISK_ORDER},
        color_discrete_map={
            "Low": COLORS["green"],
            "Moderate": COLORS["gold"],
            "High": COLORS["orange"],
            "Critical": COLORS["red"],
        },
        opacity=0.45,
        hover_data=["station_id", "demand_hour", "cluster_name"],
    )
    lower = float(
        min(
            scatter_source["net_change"].min(),
            scatter_source["predicted_net_change"].min(),
        )
    )
    upper = float(
        max(
            scatter_source["net_change"].max(),
            scatter_source["predicted_net_change"].max(),
        )
    )
    scatter_figure.add_shape(
        type="line",
        x0=lower,
        y0=lower,
        x1=upper,
        y1=upper,
        line={"dash": "dash", "color": "#667085"},
    )
    scatter_figure = style_figure(
        scatter_figure,
        "Actual vs Predicted Net Flow",
        "Predicted net bikes",
    )
    scatter_figure.update_xaxes(title="Actual net bikes")

    table_data = (
        station_pressure.sort_values(
            ["elevated_events", "elevated_rate"],
            ascending=False,
        )
        .head(50)[
            [
                "station_id",
                "cluster_name",
                "sample_rows",
                "elevated_events",
                "elevated_rate",
                "mae",
                "avg_predicted_net",
            ]
        ]
        .round(
            {
                "elevated_rate": 4,
                "mae": 2,
                "avg_predicted_net": 2,
            }
        )
        .to_dict("records")
    )

    return (
        kpis,
        daily_figure,
        risk_figure,
        top_pressure_figure,
        hour_error_figure,
        scatter_figure,
        table_data,
    )


if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8050")),
    )
