"""Training of the production model and generation of the live 72-hour forecast.

The forecast origin is the last hour with an observation. For that single hour we build the
origin features, replicate them for horizons 1..72, and ask the model for the 10th, 50th and 90th
percentile of PM2.5 and PM10. The AQI forecast is then computed from those paths:

* the trailing 24-hour averages mix the last observed hours with the forecast hours, exactly as
  the AQI would have been computed had those hours happened;
* the AQI band uses the q10 path for the lower edge and the q90 path for the upper edge. This is
  an approximation (a quantile of an average is not the average of quantiles) and is documented
  on the model page;
* if the AQI basis includes gases (AQI_BASIS=all) they are held at their last observed value,
  because the models forecast PM2.5 and PM10 only.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.engine import Engine

from ml.registry import load_model, save_model
from ml.train import N_ROUNDS, QuantileForecaster
from pipeline.aqi import POLLUTANT_ORDER, compute_aqi
from pipeline.config import get_settings
from pipeline.db import (
    FORECAST_COLUMNS,
    forecasts,
    read_observations,
    set_meta,
    upsert,
    utcnow,
)
from pipeline.features import (
    LOCATION_CODES,
    MAX_HORIZON,
    TARGETS,
    TRAIN_HORIZONS,
    build_rows,
    calendar_features,
    feature_columns,
    origin_features,
)

log = logging.getLogger("ml.forecast")

FORECAST_LOCATIONS = ["delhi", "delhi-north", "delhi-south"]
CONTEXT_HOURS = 24 * 10  # enough history for the 168 h lags and rolling windows
KEEP_FORECASTS_DAYS = 7


def train_final(engine: Engine, stride: int = 3, n_random: int = 3) -> dict[str, Any]:
    """Fit the production model on all available history and store it in the registry."""
    frames = {loc: read_observations(engine, loc) for loc in FORECAST_LOCATIONS}
    frames = {k: v for k, v in frames.items() if len(v)}
    if not frames:
        raise RuntimeError("no observations in the database; run `python -m pipeline.ingest backfill` first")
    pool = build_rows(frames, TRAIN_HORIZONS, stride=stride, n_random=n_random, seed=0)
    features = feature_columns(pool)
    model = QuantileForecaster().fit(pool, features)
    meta = {
        "trained_at": utcnow().isoformat(),
        "data_through": max(f.index.max() for f in frames.values()).isoformat(),
        "training_rows": int(len(pool)),
        "n_features": len(features),
        "boosting_rounds": N_ROUNDS,
        "locations": list(frames),
        "interval_widening_log": model.offsets,
        "top_features": model.feature_importance("pm2_5", 10),
    }
    save_model(engine, model, meta)
    log.info("trained on %d rows, data through %s", len(pool), meta["data_through"])
    return meta


def inference_rows(frame: pd.DataFrame, location: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(rows for horizons 1..72 at the latest origin, observations up to the origin) or None."""
    valid = frame[["pm2_5", "pm10"]].dropna().index
    if len(valid) == 0:
        return None
    origin = valid.max()
    upto = frame.loc[:origin]
    feats = origin_features(upto).iloc[[-1]]
    horizons = np.arange(1, MAX_HORIZON + 1)
    rows = pd.concat([feats] * len(horizons), ignore_index=True)
    rows.insert(0, "location", location)
    rows.insert(1, "origin_ts", origin)
    rows.insert(2, "target_ts", origin + pd.to_timedelta(horizons, unit="h"))
    for tgt in TARGETS:
        rows[f"cur_{tgt}"] = float(upto[tgt].loc[origin])
    rows["horizon"] = horizons
    rows["loc_code"] = LOCATION_CODES.get(location, -1)
    cal = calendar_features(pd.DatetimeIndex(rows["target_ts"]))
    cal.index = rows.index
    return pd.concat([rows, cal], axis=1), upto


def aqi_forecast(upto: pd.DataFrame, rows: pd.DataFrame, preds: dict[str, pd.DataFrame], basis: list[str]) -> dict[str, np.ndarray]:
    """AQI for each future hour, along the q10 / q50 / q90 particulate paths."""
    origin = rows["origin_ts"].iloc[0]
    past = upto.loc[origin - pd.Timedelta(hours=47) : origin, POLLUTANT_ORDER].asfreq("h")
    future_index = pd.DatetimeIndex(rows["target_ts"])
    held = past.ffill().iloc[-1]
    out: dict[str, np.ndarray] = {}
    for q in ("q10", "q50", "q90"):
        future = pd.DataFrame({p: held[p] for p in POLLUTANT_ORDER}, index=future_index)
        future["pm2_5"] = preds["pm2_5"][q].to_numpy()
        future["pm10"] = preds["pm10"][q].to_numpy()
        hybrid = pd.concat([past, future])
        aqi = compute_aqi(hybrid, basis)["aqi"].reindex(future_index)
        out[q] = aqi.to_numpy(dtype=float)
    return out


def make_forecast(engine: Engine, now: dt.datetime | None = None) -> int:
    """Forecast every location from its latest observation and upsert the result."""
    loaded = load_model(engine)
    if loaded is None:
        raise RuntimeError("no trained model in the registry; run `python -m pipeline.jobs train`")
    model, _, _ = loaded
    basis = get_settings().basis_pollutants
    now = now or utcnow()
    stored: list[dict[str, Any]] = []
    for loc in FORECAST_LOCATIONS:
        frame = read_observations(engine, loc)
        if frame.empty:
            continue
        frame = frame.loc[frame.index.max() - pd.Timedelta(hours=CONTEXT_HOURS) :]
        built = inference_rows(frame, loc)
        if built is None:
            continue
        rows, upto = built
        preds = {t: model.predict(rows, t) for t in TARGETS}
        aqi = aqi_forecast(upto, rows, preds, basis)
        for i in range(len(rows)):
            rec: dict[str, Any] = {
                "location_id": loc,
                "issued_at": rows["origin_ts"].iloc[i].to_pydatetime(),
                "target_ts": rows["target_ts"].iloc[i].to_pydatetime(),
                "horizon": int(rows["horizon"].iloc[i]),
            }
            for tgt in TARGETS:
                for q in ("q10", "q50", "q90"):
                    rec[f"{tgt}_{q}"] = float(preds[tgt][q].iloc[i])
            for q in ("q10", "q50", "q90"):
                v = aqi[q][i]
                rec[f"aqi_{q}"] = None if np.isnan(v) else float(v)
            stored.append({k: rec.get(k) for k in ["location_id", "issued_at", "target_ts", "horizon", *FORECAST_COLUMNS]})
    n = upsert(engine, forecasts, stored)
    if stored:  # prune relative to the newest forecast, not the wall clock, so a data outage never erases it
        newest = max(r["issued_at"] for r in stored)
        with engine.begin() as conn:
            conn.execute(delete(forecasts).where(forecasts.c.issued_at < newest - dt.timedelta(days=KEEP_FORECASTS_DAYS)))
    set_meta(engine, "last_forecast_at", now.isoformat())
    log.info("stored %d forecast rows", n)
    return n


def read_latest_forecast(engine: Engine, location: str) -> pd.DataFrame:
    """The most recently issued forecast for a location, ordered by horizon."""
    with engine.connect() as conn:
        issued = conn.execute(select(forecasts.c.issued_at).where(forecasts.c.location_id == location).order_by(forecasts.c.issued_at.desc()).limit(1)).first()
        if issued is None:
            return pd.DataFrame()
    df = pd.read_sql(
        select(forecasts).where(forecasts.c.location_id == location, forecasts.c.issued_at == issued[0]).order_by(forecasts.c.horizon),
        engine,
    )
    df["issued_at"] = pd.to_datetime(df["issued_at"])
    df["target_ts"] = pd.to_datetime(df["target_ts"])
    return df
