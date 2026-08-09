
"""
prepare_dashboard_data.py

Create compact dashboard-ready aggregates from the completed Bike Share Toronto
first-model outputs.

Inputs expected under outputs/full:
- hourly_panel.csv
- forecast_results_with_risk.csv
- station_clusters.csv
- run_summary.json

The raw 25-million-trip files are NOT read and the model is NOT retrained.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HISTORICAL_COLUMNS = [
    "station_id",
    "demand_hour",
    "bikes_out",
    "bikes_in",
    "net_change",
    "hour",
    "dow",
    "is_weekend",
    "year",
    "season",
]

FORECAST_COLUMNS = [
    "station_id",
    "demand_hour",
    "bikes_out",
    "bikes_in",
    "net_change",
    "hour",
    "dow",
    "is_weekend",
    "year",
    "season",
    "lag_2h_net",
    "predicted_net_change",
    "risk_level",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare compact data files for the Bike Share Dash dashboard."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Bike Share project folder. Defaults to the script folder.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Rows read from hourly_panel.csv at a time.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the dashboard cache even if it already exists.",
    )
    return parser.parse_args()


def normalize_weekend(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
    )
    if mapped.isna().any():
        bad = normalized[mapped.isna()].dropna().unique().tolist()[:10]
        raise ValueError(f"Unrecognized is_weekend values: {bad}")
    return mapped.astype(bool)


def combine_grouped(
    frames: list[pd.DataFrame],
    keys: list[str],
    sum_columns: list[str],
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=[*keys, *sum_columns])

    combined = pd.concat(frames, ignore_index=True)
    result = (
        combined.groupby(keys, observed=True, as_index=False)[sum_columns]
        .sum()
        .sort_values(keys)
        .reset_index(drop=True)
    )
    return result


def safe_r2(actual: pd.Series, predicted: pd.Series) -> float:
    actual_array = actual.to_numpy(dtype=float)
    predicted_array = predicted.to_numpy(dtype=float)
    denominator = np.sum((actual_array - actual_array.mean()) ** 2)
    if denominator == 0:
        return float("nan")
    numerator = np.sum((actual_array - predicted_array) ** 2)
    return float(1 - numerator / denominator)


def build_dashboard_cache(
    project_dir: Path,
    chunk_size: int = 500_000,
    force: bool = False,
) -> Path:
    project_dir = project_dir.resolve()
    full_dir = project_dir / "outputs" / "full"
    cache_dir = project_dir / "outputs" / "dashboard_cache"
    manifest_path = cache_dir / "dashboard_manifest.json"

    required = {
        "hourly_panel": full_dir / "hourly_panel.csv",
        "forecast": full_dir / "forecast_results_with_risk.csv",
        "clusters": full_dir / "station_clusters.csv",
        "summary": full_dir / "run_summary.json",
    }

    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The dashboard requires the completed first-model outputs.\n"
            "Missing files:\n- " + "\n- ".join(missing)
        )

    if manifest_path.exists() and not force:
        print(f"Dashboard cache already exists:\n  {cache_dir}")
        print("Use --force to rebuild it.")
        return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)

    with required["summary"].open("r", encoding="utf-8") as file:
        run_summary = json.load(file)

    allowed_years = {int(year) for year in run_summary["years_found"]}

    clusters = pd.read_csv(required["clusters"], low_memory=False)
    clusters["station_id"] = pd.to_numeric(
        clusters["station_id"], errors="coerce"
    ).astype("Int64")
    clusters = clusters.dropna(subset=["station_id"]).copy()
    clusters["station_id"] = clusters["station_id"].astype("int32")
    clusters["cluster_name"] = (
        clusters["cluster_name"]
        .astype("string")
        .fillna("Unassigned")
    )

    cluster_map = clusters.set_index("station_id")["cluster_name"]

    daily_parts: list[pd.DataFrame] = []
    hourly_parts: list[pd.DataFrame] = []
    station_parts: list[pd.DataFrame] = []
    seasonal_parts: list[pd.DataFrame] = []

    print("Reading the historical station-hour panel in chunks...")
    total_rows_read = 0

    dtype_map = {
        "station_id": "int32",
        "bikes_out": "int32",
        "bikes_in": "int32",
        "net_change": "int32",
        "hour": "int8",
        "dow": "int8",
        "year": "int16",
        "season": "string",
    }

    reader = pd.read_csv(
        required["hourly_panel"],
        usecols=HISTORICAL_COLUMNS,
        dtype=dtype_map,
        chunksize=chunk_size,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        total_rows_read += len(chunk)
        chunk["demand_hour"] = pd.to_datetime(
            chunk["demand_hour"], errors="coerce"
        )
        chunk = chunk.dropna(subset=["demand_hour", "station_id"]).copy()
        chunk = chunk[chunk["year"].isin(allowed_years)].copy()

        chunk["is_weekend"] = normalize_weekend(chunk["is_weekend"])
        chunk["day_type"] = np.where(
            chunk["is_weekend"], "Weekend", "Weekday"
        )
        chunk["date"] = chunk["demand_hour"].dt.normalize()
        chunk["cluster_name"] = (
            chunk["station_id"].map(cluster_map).fillna("Unassigned")
        )

        daily = (
            chunk.groupby(
                ["date", "year", "season", "day_type", "cluster_name"],
                observed=True,
                as_index=False,
            )
            .agg(
                trip_starts=("bikes_out", "sum"),
                trip_ends=("bikes_in", "sum"),
                net_change=("net_change", "sum"),
                active_station_hours=("station_id", "size"),
            )
        )
        daily_parts.append(daily)

        hourly = (
            chunk.groupby(
                ["year", "day_type", "cluster_name", "hour"],
                observed=True,
                as_index=False,
            )
            .agg(
                trip_starts=("bikes_out", "sum"),
                trip_ends=("bikes_in", "sum"),
                net_change=("net_change", "sum"),
                active_station_hours=("station_id", "size"),
            )
        )
        hourly_parts.append(hourly)

        station = (
            chunk.groupby(
                ["station_id", "year", "day_type", "cluster_name"],
                observed=True,
                as_index=False,
            )
            .agg(
                trip_starts=("bikes_out", "sum"),
                trip_ends=("bikes_in", "sum"),
                net_change=("net_change", "sum"),
                active_station_hours=("station_id", "size"),
            )
        )
        station_parts.append(station)

        seasonal = (
            chunk.groupby(
                ["year", "season", "day_type", "cluster_name"],
                observed=True,
                as_index=False,
            )
            .agg(
                trip_starts=("bikes_out", "sum"),
                trip_ends=("bikes_in", "sum"),
                net_change=("net_change", "sum"),
                active_station_hours=("station_id", "size"),
            )
        )
        seasonal_parts.append(seasonal)

        print(
            f"  chunk {chunk_number}: {len(chunk):,} usable rows "
            f"({total_rows_read:,} total rows read)"
        )

    metric_columns = [
        "trip_starts",
        "trip_ends",
        "net_change",
        "active_station_hours",
    ]

    historical_daily = combine_grouped(
        daily_parts,
        ["date", "year", "season", "day_type", "cluster_name"],
        metric_columns,
    )
    historical_hourly = combine_grouped(
        hourly_parts,
        ["year", "day_type", "cluster_name", "hour"],
        metric_columns,
    )
    historical_station = combine_grouped(
        station_parts,
        ["station_id", "year", "day_type", "cluster_name"],
        metric_columns,
    )
    historical_seasonal = combine_grouped(
        seasonal_parts,
        ["year", "season", "day_type", "cluster_name"],
        metric_columns,
    )

    historical_daily["month_start"] = (
        pd.to_datetime(historical_daily["date"])
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    historical_monthly = (
        historical_daily.groupby(
            ["month_start", "year", "season", "day_type", "cluster_name"],
            observed=True,
            as_index=False,
        )[metric_columns]
        .sum()
        .sort_values(
            ["month_start", "day_type", "cluster_name"]
        )
        .reset_index(drop=True)
    )

    historical_daily.to_csv(
        cache_dir / "historical_daily.csv", index=False
    )
    historical_monthly.to_csv(
        cache_dir / "historical_monthly.csv", index=False
    )
    historical_hourly.to_csv(
        cache_dir / "historical_hourly_profile.csv", index=False
    )
    historical_station.to_csv(
        cache_dir / "historical_station_year_daytype.csv", index=False
    )
    historical_seasonal.to_csv(
        cache_dir / "historical_seasonal.csv", index=False
    )

    print("Preparing the first-model prediction sample...")
    forecast = pd.read_csv(
        required["forecast"],
        usecols=FORECAST_COLUMNS,
        low_memory=False,
    )
    forecast["station_id"] = pd.to_numeric(
        forecast["station_id"], errors="coerce"
    ).astype("Int64")
    forecast["demand_hour"] = pd.to_datetime(
        forecast["demand_hour"], errors="coerce"
    )
    forecast = forecast.dropna(
        subset=[
            "station_id",
            "demand_hour",
            "net_change",
            "predicted_net_change",
        ]
    ).copy()
    forecast["station_id"] = forecast["station_id"].astype("int32")
    forecast["date"] = forecast["demand_hour"].dt.normalize()
    forecast["cluster_name"] = (
        forecast["station_id"].map(cluster_map).fillna("Unassigned")
    )
    forecast["absolute_error"] = (
        forecast["net_change"] - forecast["predicted_net_change"]
    ).abs()
    forecast["squared_error"] = (
        forecast["net_change"] - forecast["predicted_net_change"]
    ) ** 2
    forecast["pressure_score"] = (
        forecast["risk_level"]
        .map({"Low": 0, "Moderate": 1, "High": 2, "Critical": 3})
        .fillna(0)
        .astype("int8")
    )
    forecast["elevated_pressure"] = (
        forecast["risk_level"].isin(["Moderate", "High", "Critical"])
    )
    forecast["high_critical_pressure"] = (
        forecast["risk_level"].isin(["High", "Critical"])
    )

    forecast.to_csv(
        cache_dir / "forecast_sample_enriched.csv", index=False
    )

    actual = forecast["net_change"].astype(float)
    predicted = forecast["predicted_net_change"].astype(float)
    lag = pd.to_numeric(forecast["lag_2h_net"], errors="coerce").fillna(0)

    forecast_metrics = {
        "sample_rows": int(len(forecast)),
        "mae": float(forecast["absolute_error"].mean()),
        "rmse": float(math.sqrt(forecast["squared_error"].mean())),
        "r2": safe_r2(actual, predicted),
        "median_absolute_error": float(
            forecast["absolute_error"].median()
        ),
        "within_2_bikes": float(
            (forecast["absolute_error"] <= 2).mean()
        ),
        "within_3_bikes": float(
            (forecast["absolute_error"] <= 3).mean()
        ),
        "prediction_bias": float((predicted - actual).mean()),
        "zero_baseline_mae": float(actual.abs().mean()),
        "lag_baseline_mae": float((actual - lag).abs().mean()),
        "elevated_pressure_rate": float(
            forecast["elevated_pressure"].mean()
        ),
        "high_critical_pressure_rate": float(
            forecast["high_critical_pressure"].mean()
        ),
        "prediction_start": str(forecast["demand_hour"].min()),
        "prediction_end": str(forecast["demand_hour"].max()),
    }
    forecast_metrics["improvement_vs_zero"] = float(
        1 - forecast_metrics["mae"]
        / forecast_metrics["zero_baseline_mae"]
    )
    forecast_metrics["improvement_vs_lag"] = float(
        1 - forecast_metrics["mae"]
        / forecast_metrics["lag_baseline_mae"]
    )

    with (cache_dir / "forecast_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(forecast_metrics, file, indent=2)

    clusters.to_csv(cache_dir / "station_clusters.csv", index=False)
    with (cache_dir / "run_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(run_summary, file, indent=2)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_dir": str(project_dir),
        "source_hourly_panel": str(required["hourly_panel"]),
        "historical_rows_read": int(total_rows_read),
        "allowed_years": sorted(allowed_years),
        "forecast_sample_rows": int(len(forecast)),
        "cache_files": sorted(
            path.name for path in cache_dir.glob("*")
            if path.name != "dashboard_manifest.json"
        ),
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("\nDashboard cache created successfully.")
    print(f"Saved to:\n  {cache_dir}")
    return cache_dir


def main() -> None:
    args = parse_args()
    build_dashboard_cache(
        project_dir=args.project_dir,
        chunk_size=args.chunk_size,
        force=args.force,
    )


if __name__ == "__main__":
    main()
