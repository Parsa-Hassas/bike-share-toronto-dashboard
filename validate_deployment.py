"""Validate that the packaged dashboard contains its required deployment files."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "outputs" / "dashboard_cache"

REQUIRED_ROOT = [
    "app.py",
    "prepare_dashboard_data.py",
    "requirements.txt",
    "Procfile",
    "render.yaml",
    "assets/dashboard.css",
]

REQUIRED_CACHE = [
    "dashboard_manifest.json",
    "forecast_metrics.json",
    "forecast_sample_enriched.csv",
    "historical_daily.csv",
    "historical_hourly_profile.csv",
    "historical_monthly.csv",
    "historical_seasonal.csv",
    "historical_station_year_daytype.csv",
    "run_summary.json",
    "station_clusters.csv",
]

REQUIRED_HEADERS = {
    "forecast_sample_enriched.csv": {"station_id", "demand_hour", "net_change", "predicted_net_change", "risk_level"},
    "historical_daily.csv": {"date", "year", "cluster_name", "trip_starts", "active_station_hours"},
    "station_clusters.csv": {"station_id", "cluster_name"},
}


def main() -> None:
    missing = [name for name in REQUIRED_ROOT if not (ROOT / name).is_file()]
    missing += [f"outputs/dashboard_cache/{name}" for name in REQUIRED_CACHE if not (CACHE / name).is_file()]
    if missing:
        raise SystemExit("Missing required files:\n- " + "\n- ".join(missing))

    for name in ("dashboard_manifest.json", "forecast_metrics.json", "run_summary.json"):
        with (CACHE / name).open("r", encoding="utf-8") as file:
            json.load(file)

    for name, expected in REQUIRED_HEADERS.items():
        with (CACHE / name).open("r", encoding="utf-8", newline="") as file:
            actual = set(next(csv.reader(file)))
        absent = sorted(expected - actual)
        if absent:
            raise SystemExit(f"{name} is missing columns: {absent}")

    app_text = (ROOT / "app.py").read_text(encoding="utf-8")
    if "server = app.server" not in app_text:
        raise SystemExit("app.py does not expose server = app.server for Gunicorn.")

    print("DEPLOYMENT PACKAGE STATUS: PASS")
    print(f"Required root files: {len(REQUIRED_ROOT)}")
    print(f"Dashboard cache files: {len(REQUIRED_CACHE)}")


if __name__ == "__main__":
    main()
