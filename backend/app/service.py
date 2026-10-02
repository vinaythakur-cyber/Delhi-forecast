"""Business logic behind the API: queries, AQI, aggregation. Routes only call into this module.

No ML lives here (it reads stored forecasts) and no logic lives in the frontend.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.engine import Engine

from ml.forecast import read_latest_forecast
from ml.registry import get_evaluation
from pipeline.aqi import (
    BANDS,
    BASES,
    CATEGORIES,
    POLLUTANT_LABELS,
    POLLUTANT_ORDER,
    averaged_concentrations,
    category_for,
    compute_aqi,
    latest_aqi_summary,
)
from pipeline.config import get_settings
from pipeline.db import (
    get_meta,
    ingest_log,
    latest_ts,
    load_artifact,
    read_observations,
    utcnow,
)
from pipeline.locations import ALL_LOCATIONS, BY_ID, CELLS, CITY, NEIGHBOURHOODS

IST = pd.Timedelta(hours=5, minutes=30)
STALE_AFTER_HOURS = 3
HISTORY_METRICS = ["aqi", *POLLUTANT_ORDER]
UNITS = {"aqi": "", "pm2_5": "µg/m³", "pm10": "µg/m³", "no2": "µg/m³", "so2": "µg/m³", "co": "µg/m³", "o3": "µg/m³"}


class NotFound(Exception):
    pass


class NoData(Exception):
    pass


def iso(ts: pd.Timestamp | dt.datetime | None) -> str | None:
    if ts is None or (not isinstance(ts, str) and pd.isna(ts)):
        return None
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def num(v: Any, digits: int = 1) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), digits)


def check_location(location: str) -> None:
    if location not in BY_ID:
        raise NotFound(f"unknown location '{location}'; choose one of {', '.join(BY_ID)}")


def basis_name() -> str:
    return get_settings().aqi_basis if get_settings().aqi_basis in BASES else "pm2_5"


# ----------------------------------------------------------------------------- caching of the heavy part

_CACHE: dict[tuple, pd.DataFrame] = {}


def _full_frame(engine: Engine, location: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(observations, hourly AQI table) for the whole history, cached until new data arrives."""
    check_location(location)
    through = latest_ts(engine, location, "pm2_5")
    if through is None:
        raise NoData("no data yet: the first download is still running or has not been started")
    key = (str(engine.url), location, through, basis_name())
    if key not in _CACHE:
        if len(_CACHE) > 12:
            _CACHE.clear()
        obs = read_observations(engine, location)
        aqi = compute_aqi(obs, BASES[basis_name()])
        _CACHE[key] = (obs, aqi)
    return _CACHE[key]


def clear_cache() -> None:
    _CACHE.clear()


# ----------------------------------------------------------------------------- freshness / health


def freshness(engine: Engine, now: dt.datetime | None = None) -> dict[str, Any]:
    now = now or utcnow()
    through = latest_ts(engine, CITY.id, "pm2_5")
    with engine.connect() as conn:
        row = conn.execute(
            select(ingest_log.c.finished_at).where(ingest_log.c.status == "ok").order_by(ingest_log.c.id.desc()).limit(1)
        ).first()
    ingested = row[0] if row else None
    age = None if through is None else round((now - through).total_seconds() / 60)
    return {
        "data_through": iso(through),
        "last_ingest": iso(ingested),
        "age_minutes": age,
        "stale": age is None or age > STALE_AFTER_HOURS * 60,
        "note": "Hourly values come from the CAMS atmospheric model via Open-Meteo. The newest hours are model forecasts that get revised.",
    }


def health(engine: Engine) -> dict[str, Any]:
    model = load_artifact(engine, "forecaster")
    fresh = freshness(engine)
    return {
        "status": "ok" if fresh["data_through"] and model else "initializing",
        "data_through": fresh["data_through"],
        "last_ingest": fresh["last_ingest"],
        "last_forecast": get_meta(engine, "last_forecast_at"),
        "model_trained_at": iso(model[2]) if model else None,
        "aqi_basis": basis_name(),
        "stale": fresh["stale"],
    }


# ----------------------------------------------------------------------------- current


def locations_list() -> list[dict[str, Any]]:
    return [{"id": loc.id, "name": loc.name, "kind": loc.kind, "lat": loc.lat, "lon": loc.lon, "area": loc.area} for loc in ALL_LOCATIONS]


def current(engine: Engine, location: str) -> dict[str, Any]:
    obs, aqi = _full_frame(engine, location)
    recent = obs.iloc[-24 * 10 :]
    summary = latest_aqi_summary(recent, BASES[basis_name()])
    if summary is None:
        raise NoData("not enough recent data to compute the AQI")
    ts = summary["ts"]
    avg = averaged_concentrations(recent).loc[ts]
    counted = set(BASES[basis_name()])
    latest = obs.loc[ts]
    weather_cols = ["temp", "rh", "wind_speed", "wind_dir", "blh", "pressure"]
    recent_weather = obs[weather_cols].iloc[-24 * 14 :]
    wx = recent_weather.ffill().iloc[-1]  # latest known value of each variable (the weather feed can lag)
    wx_valid = recent_weather["temp"].dropna()
    wx_as_of = wx_valid.index[-1] if len(wx_valid) else None
    series = aqi["aqi"].dropna()
    prior = series.loc[: ts - pd.Timedelta(hours=3)]
    change = None if prior.empty else int(summary["aqi"] - prior.iloc[-1])
    return {
        "location": {"id": location, "name": BY_ID[location].name, "kind": BY_ID[location].kind},
        "observed_at": iso(ts),
        "freshness": freshness(engine),
        "aqi": {
            "value": summary["aqi"],
            "category": summary["category"],
            "color": summary["color"],
            "advice": summary["advice"],
            "dominant": summary["dominant"],
            "dominant_label": summary["dominant_label"],
            "change_3h": change,
            "basis": basis_name(),
            "basis_note": {
                "pm2_5": "Computed from PM2.5 only. The free model's ozone and PM10 (dust) are not reliable enough to set the headline.",
                "pm": "Computed from PM2.5 and PM10.",
                "all": "Official six-pollutant NAQI computed from model data.",
            }[basis_name()],
        },
        "pollutants": [
            {
                "key": p,
                "label": POLLUTANT_LABELS[p],
                "value": num(latest[p]),
                "unit": UNITS[p],
                "average": num(avg[p], 2) if p in avg.index else None,
                "average_unit": "mg/m³" if p == "co" else "µg/m³",
                "window": "8 h max" if p in ("co", "o3") else "24 h",
                "sub_index": summary["sub_indices"].get(p),
                "counts_toward_aqi": p in counted,
            }
            for p in POLLUTANT_ORDER
        ],
        "weather": {
            "temperature_c": num(wx["temp"]),
            "humidity_pct": num(wx["rh"], 0),
            "wind_kmh": num(wx["wind_speed"]),
            "wind_direction_deg": num(wx["wind_dir"], 0),
            "boundary_layer_m": num(wx["blh"], 0),
            "pressure_hpa": num(wx["pressure"], 0),
            "as_of": iso(wx_as_of),
            "lag_hours": None if wx_as_of is None else round((ts - wx_as_of).total_seconds() / 3600),
        },
        "categories": [{"name": c.name, "low": c.low, "high": c.high, "color": c.color} for c in CATEGORIES],
    }


# ----------------------------------------------------------------------------- history


def history(engine: Engine, location: str, metric: str, start: dt.datetime | None, end: dt.datetime | None, resolution: str) -> dict[str, Any]:
    if metric not in HISTORY_METRICS:
        raise ValueError(f"metric must be one of {', '.join(HISTORY_METRICS)}")
    if resolution not in ("hourly", "daily", "monthly"):
        raise ValueError("resolution must be hourly, daily or monthly")
    obs, aqi = _full_frame(engine, location)
    series = aqi["aqi"] if metric == "aqi" else obs[metric]
    end = end or series.index.max()
    start = start or end - pd.Timedelta(days=90)
    window = series.loc[start:end]
    if resolution == "hourly" and len(window) > 24 * 62:
        raise ValueError("hourly resolution is limited to 62 days; use daily or monthly for longer ranges")
    if resolution == "hourly":
        points = [{"t": iso(t), "v": num(v)} for t, v in window.items() if not np.isnan(v)]
    else:
        local = window.copy()
        local.index = local.index + IST  # group by the Delhi calendar day / month
        rule = "D" if resolution == "daily" else "MS"
        agg = local.resample(rule).agg(["mean", "min", "max", "count"])
        agg = agg[agg["count"] >= (12 if resolution == "daily" else 24 * 10)]
        points = [
            {"t": pd.Timestamp(t).strftime("%Y-%m-%d"), "v": num(r["mean"]), "min": num(r["min"]), "max": num(r["max"])}
            for t, r in agg.iterrows()
        ]
    return {
        "location": location,
        "metric": metric,
        "unit": UNITS[metric],
        "resolution": resolution,
        "timezone": "UTC" if resolution == "hourly" else "IST (Asia/Kolkata)",
        "range": {"start": iso(window.index.min()) if len(window) else None, "end": iso(window.index.max()) if len(window) else None},
        "points": points,
    }


# ----------------------------------------------------------------------------- forecast


def pm25_category_bands() -> list[dict[str, Any]]:
    """PM2.5 concentration ranges (ug/m3) of each AQI category, for shading the PM2.5 forecast chart."""
    highs = BANDS["pm2_5"].c_hi
    return [
        {"name": c.name, "low": 0.0 if i == 0 else float(highs[i - 1]), "high": float(highs[i]), "color": c.color}
        for i, c in enumerate(CATEGORIES)
    ]


def forecast(engine: Engine, location: str, history_hours: int = 48) -> dict[str, Any]:
    check_location(location)
    fc = read_latest_forecast(engine, location)
    if fc.empty:
        raise NoData("no forecast yet: run `python -m pipeline.jobs bootstrap` or wait for the first refresh")
    obs, aqi = _full_frame(engine, location)
    issued = pd.Timestamp(fc["issued_at"].iloc[0])
    recent_obs = obs.loc[issued - pd.Timedelta(hours=history_hours - 1) : issued]
    recent_aqi = aqi["aqi"].loc[recent_obs.index]

    def q3(row: pd.Series, name: str, digits: int = 1) -> dict[str, float | None]:
        return {q: num(row[f"{name}_{q}"], digits) for q in ("q10", "q50", "q90")}

    points = []
    for _, row in fc.iterrows():
        cat = category_for(row["aqi_q50"])
        points.append(
            {
                "target_ts": iso(row["target_ts"]),
                "horizon": int(row["horizon"]),
                "pm2_5": q3(row, "pm2_5"),
                "pm10": q3(row, "pm10"),
                "aqi": q3(row, "aqi", 0),
                "category": cat.name if cat else None,
                "color": cat.color if cat else None,
            }
        )
    evaluation = get_evaluation(engine)
    reference = None
    if evaluation:
        try:
            t = evaluation["targets"]["pm2_5"]["by_horizon"]
            reference = {h: {"model_mae": round(t["lightgbm"][h]["mae"], 1), "persistence_mae": round(t["persistence"][h]["mae"], 1)} for h in ("6", "24", "72") if h in t["lightgbm"]}
        except (KeyError, TypeError):
            reference = None
    return {
        "location": {"id": location, "name": BY_ID[location].name},
        "issued_at": iso(issued),
        "freshness": freshness(engine),
        "basis": basis_name(),
        "interval": "80% prediction interval (10th to 90th percentile), calibrated on held-out data",
        "recent": [
            {"t": iso(t), "pm2_5": num(recent_obs.loc[t, "pm2_5"]), "aqi": num(recent_aqi.loc[t], 0)}
            for t in recent_obs.index
        ],
        "points": points,
        "categories": [{"name": c.name, "low": c.low, "high": c.high, "color": c.color} for c in CATEGORIES],
        "pm25_category_bands": pm25_category_bands(),
        "error_reference": reference,
        "caveat": "Beyond about 24 hours the forecast is much less certain; the band widens to show that.",
    }


# ----------------------------------------------------------------------------- seasonal heatmap


def seasonal(engine: Engine, location: str, metric: str) -> dict[str, Any]:
    if metric not in HISTORY_METRICS:
        raise ValueError(f"metric must be one of {', '.join(HISTORY_METRICS)}")
    obs, aqi = _full_frame(engine, location)
    series = (aqi["aqi"] if metric == "aqi" else obs[metric]).dropna()
    local = series.copy()
    local.index = local.index + IST
    grid = local.groupby([local.index.month, local.index.hour]).mean().unstack()
    matrix = [[num(grid.loc[m, h]) if (m in grid.index and h in grid.columns) else None for h in range(24)] for m in range(1, 13)]
    ym = local.groupby([local.index.year, local.index.month]).mean()
    return {
        "location": location,
        "metric": metric,
        "unit": UNITS[metric],
        "timezone": "IST (Asia/Kolkata)",
        "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "month_by_hour": matrix,
        "year_month": [{"year": int(y), "month": int(m), "value": num(v)} for (y, m), v in ym.items()],
    }


# ----------------------------------------------------------------------------- map


def stations(engine: Engine, ground: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    cells = []
    cell_aqi: dict[str, dict[str, Any]] = {}
    for loc in CELLS:
        try:
            cur = current(engine, loc.id)
        except (NoData, NotFound):
            continue
        cell_aqi[loc.id] = cur["aqi"]
        cells.append(
            {
                "id": loc.id,
                "name": loc.name,
                "lat": loc.lat,
                "lon": loc.lon,
                "aqi": cur["aqi"]["value"],
                "category": cur["aqi"]["category"],
                "color": cur["aqi"]["color"],
                "observed_at": cur["observed_at"],
            }
        )
    neighbourhoods = [
        {
            "name": n.name,
            "lat": n.lat,
            "lon": n.lon,
            "cell": n.cell,
            "aqi": cell_aqi[n.cell]["value"],
            "category": cell_aqi[n.cell]["category"],
            "color": cell_aqi[n.cell]["color"],
        }
        for n in NEIGHBOURHOODS
        if n.cell in cell_aqi
    ]
    return {
        "cells": cells,
        "neighbourhoods": neighbourhoods,
        "ground_stations": ground or [],
        "ground_stations_enabled": bool(get_settings().openaq_api_key),
        "note": (
            "The model grid is coarse: every neighbourhood inside a cell shares the same value, so Delhi has only "
            "two independent model series. Real sensors appear as separate markers when an OpenAQ key is configured."
        ),
    }


# ----------------------------------------------------------------------------- model metrics


def model_metrics(engine: Engine) -> dict[str, Any]:
    evaluation = get_evaluation(engine)
    if evaluation is None:
        raise NoData("no evaluation available yet: run `python -m ml.evaluate`")
    model = load_artifact(engine, "forecaster")
    return {
        "evaluation": evaluation,
        "production_model": None if model is None else {"trained_at": iso(model[2]), **model[1]},
        "aqi_basis": basis_name(),
        "limitations": [
            "The series is an atmospheric model (CAMS) on a coarse grid, not sensor measurements. Delhi has only two independent cells.",
            "History starts in August 2022, so evaluation covers at most four winters and four Diwali periods.",
            "The model data has regime changes (PM10 and dust behaviour change in late 2024 and March 2025).",
            "No satellite fire counts, so stubble burning is represented only by a season flag.",
            "Accuracy falls with horizon; beyond 24 hours treat the forecast as a trend with a wide band.",
            "The AQI band is derived from PM quantile paths and is an approximation.",
        ],
    }
