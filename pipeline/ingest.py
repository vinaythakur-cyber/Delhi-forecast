"""Ingestion: fetch from Open-Meteo, validate, store. Idempotent and safe to schedule.

Run it from the repo root:

    python -m pipeline.ingest backfill    # download the full history (first run)
    python -m pipeline.ingest update      # top up the latest hours (what the scheduler calls)
    python -m pipeline.ingest status      # show what is stored

Why the overlap in `update`: the most recent hours come from CAMS *forecast* runs and are revised
when the analysis arrives. Re-fetching the last 72 hours and upserting lets those revisions land.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time as _time
from typing import Any

import httpx
import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from pipeline.config import Settings, get_settings
from pipeline.db import (
    OBS_COLUMNS,
    count_rows,
    get_engine,
    init_db,
    latest_ts,
    log_ingest,
    observations,
    read_observations,
    register_locations,
    set_meta,
    upsert_observations,
    utcnow,
)
from pipeline.locations import ALL_LOCATIONS, CELLS, CITY, Location

log = logging.getLogger("pipeline.ingest")
_sleep = _time.sleep  # indirection so tests can disable waiting

AQ_VARS = {
    "pm2_5": "pm2_5",
    "pm10": "pm10",
    "no2": "nitrogen_dioxide",
    "so2": "sulphur_dioxide",
    "co": "carbon_monoxide",
    "o3": "ozone",
}
WX_VARS = {
    "temp": "temperature_2m",
    "rh": "relative_humidity_2m",
    "wind_speed": "wind_speed_10m",
    "wind_dir": "wind_direction_10m",
    "blh": "boundary_layer_height",
    "precip": "precipitation",
    "pressure": "surface_pressure",
}
# Physically plausible ranges. Anything outside is a data fault and becomes NULL (never clipped).
LIMITS: dict[str, tuple[float, float]] = {
    "pm2_5": (0, 1500),
    "pm10": (0, 2000),
    "no2": (0, 1000),
    "so2": (0, 1500),
    "co": (0, 60000),
    "o3": (0, 1000),
    "temp": (-30, 60),
    "rh": (0, 100),
    "wind_speed": (0, 150),
    "wind_dir": (0, 360),
    "blh": (0, 10000),
    "precip": (0, 300),
    "pressure": (850, 1100),
}
ARCHIVE_LAG_DAYS = 6  # ERA5 archive is complete up to roughly this many days ago


class SourceError(RuntimeError):
    """The upstream API refused a request in a way retrying cannot fix."""


# ----------------------------------------------------------------------------- HTTP


def make_client(settings: Settings | None = None) -> httpx.Client:
    settings = settings or get_settings()
    return httpx.Client(timeout=settings.http_timeout_s, headers={"User-Agent": "delhi-aqi/1.0"})


def get_json(client: httpx.Client, url: str, params: dict[str, Any], settings: Settings, tries: int = 5) -> dict:
    last: Exception | None = None
    for attempt in range(tries):
        try:
            resp = client.get(url, params=params)
        except httpx.TransportError as exc:  # network blip, DNS, timeout
            last = exc
            log.warning("network error (%s), retry %d/%d", exc.__class__.__name__, attempt + 1, tries)
            _sleep(min(2**attempt, 20))
            continue
        if resp.status_code == 429:
            log.warning("rate limited by %s, waiting %.0fs", url, settings.rate_limit_wait_s)
            _sleep(settings.rate_limit_wait_s)
            last = SourceError("rate limited")
            continue
        if resp.status_code >= 500:
            last = SourceError(f"{resp.status_code} from {url}")
            _sleep(min(2**attempt, 20))
            continue
        if resp.status_code >= 400:
            try:
                reason = resp.json().get("reason", resp.text[:200])
            except ValueError:
                reason = resp.text[:200]
            raise SourceError(f"{resp.status_code} from {url}: {reason}")
        return resp.json()
    raise SourceError(f"giving up on {url} after {tries} tries: {last}")


# ----------------------------------------------------------------------------- parse / validate


def parse_hourly(payload: dict, mapping: dict[str, str]) -> pd.DataFrame:
    """Open-Meteo `hourly` block -> DataFrame indexed by naive-UTC `ts` with our column names."""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time")
    if not times:
        return pd.DataFrame(columns=list(mapping), index=pd.DatetimeIndex([], name="ts").as_unit("ns"), dtype="float64")
    idx = pd.DatetimeIndex(pd.to_datetime(times)).as_unit("ns").rename("ts")
    data = {}
    for internal, api_name in mapping.items():
        values = hourly.get(api_name)
        if values is None:
            data[internal] = np.full(len(idx), np.nan)
        else:
            data[internal] = pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce").to_numpy(dtype="float64")
    return pd.DataFrame(data, index=idx)


def validate_observations(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop duplicate hours, null out impossible values, and report data quality."""
    report: dict[str, Any] = {"rows_in": int(len(df)), "duplicates": 0, "out_of_range": {}}
    if df.empty:
        report["null_fraction"] = {}
        return df, report
    dup = df.index.duplicated(keep="last")
    report["duplicates"] = int(dup.sum())
    out = df[~dup].sort_index().copy()
    for col, (lo, hi) in LIMITS.items():
        if col not in out:
            continue
        bad = (out[col] < lo) | (out[col] > hi)
        n = int(bad.sum())
        if n:
            report["out_of_range"][col] = n
            out.loc[bad, col] = np.nan
    report["null_fraction"] = {c: round(float(v), 4) for c, v in out.isna().mean().items()}
    return out, report


# ----------------------------------------------------------------------------- fetchers


def _chunks(start: dt.date, end: dt.date, days: int = 366):
    cur = start
    while cur <= end:
        stop = min(cur + dt.timedelta(days=days - 1), end)
        yield cur, stop
        cur = stop + dt.timedelta(days=1)


def fetch_air_quality(
    client: httpx.Client, loc: Location, start: dt.date, end: dt.date, settings: Settings | None = None
) -> pd.DataFrame:
    settings = settings or get_settings()
    frames = []
    for a, b in _chunks(start, end):
        params = {
            "latitude": loc.lat,
            "longitude": loc.lon,
            "hourly": ",".join(AQ_VARS.values()),
            "start_date": a.isoformat(),
            "end_date": b.isoformat(),
            "timezone": "UTC",
        }
        frames.append(parse_hourly(get_json(client, settings.aq_api_url, params, settings), AQ_VARS))
        _sleep(settings.request_pause_s)
    return pd.concat(frames) if frames else pd.DataFrame(columns=list(AQ_VARS))


def fetch_weather_archive(
    client: httpx.Client, loc: Location, start: dt.date, end: dt.date, settings: Settings | None = None
) -> pd.DataFrame:
    settings = settings or get_settings()
    cap = utcnow().date() - dt.timedelta(days=ARCHIVE_LAG_DAYS)
    end = min(end, cap)
    frames = []
    for a, b in _chunks(start, end) if start <= end else []:
        params = {
            "latitude": loc.lat,
            "longitude": loc.lon,
            "hourly": ",".join(WX_VARS.values()),
            "start_date": a.isoformat(),
            "end_date": b.isoformat(),
            "timezone": "UTC",
        }
        frames.append(parse_hourly(get_json(client, settings.weather_archive_url, params, settings), WX_VARS))
        _sleep(settings.request_pause_s)
    return pd.concat(frames) if frames else pd.DataFrame(columns=list(WX_VARS), dtype="float64")


def fetch_weather_recent(
    client: httpx.Client, loc: Location, past_days: int = 10, settings: Settings | None = None
) -> pd.DataFrame:
    """Recent weather from the forecast endpoint (the ERA5 archive lags several days)."""
    settings = settings or get_settings()
    params = {
        "latitude": loc.lat,
        "longitude": loc.lon,
        "hourly": ",".join(WX_VARS.values()),
        "past_days": min(past_days, 92),
        "forecast_days": 1,
        "timezone": "UTC",
    }
    df = parse_hourly(get_json(client, settings.weather_forecast_url, params, settings, tries=2), WX_VARS)
    _sleep(settings.request_pause_s)
    return df


# ----------------------------------------------------------------------------- orchestration


def ingest_location(
    engine: Engine,
    client: httpx.Client,
    loc: Location,
    start: dt.date,
    end: dt.date | None = None,
    settings: Settings | None = None,
    now: dt.datetime | None = None,
    state: dict[str, bool] | None = None,
) -> tuple[int, dict[str, Any]]:
    settings = settings or get_settings()
    now = now or utcnow()
    end = end or now.date()
    state = state if state is not None else {}
    aq = fetch_air_quality(client, loc, start, end, settings)
    wx_archive = fetch_weather_archive(client, loc, start, end, settings)
    wx_recent = pd.DataFrame(columns=list(WX_VARS), dtype="float64")
    if state.get("recent_weather_ok", True):
        try:
            wx_recent = fetch_weather_recent(client, loc, settings=settings)
        except (SourceError, httpx.HTTPError) as exc:
            state["recent_weather_ok"] = False  # do not retry for every location in this run
            log.warning("recent weather unavailable (%s); using the ERA5 archive only for this run", exc)
    wx_recent = wx_recent[wx_recent.index >= pd.Timestamp(start)] if len(wx_recent) else wx_recent
    wx = wx_archive.combine_first(wx_recent) if len(wx_recent) else wx_archive
    df = aq.join(wx, how="outer") if len(wx) else aq
    df = df[df.index <= pd.Timestamp(now).floor("h")]  # never store hours that have not happened
    df, report = validate_observations(df)
    written = upsert_observations(engine, loc.id, df)
    return written, report


def _circular_mean_deg(series: pd.Series) -> pd.Series:
    rad = np.deg2rad(series)
    s = np.sin(rad).groupby(level=0).mean()
    c = np.cos(rad).groupby(level=0).mean()
    deg = np.mod(np.round(np.rad2deg(np.arctan2(s, c)), 6), 360.0)  # round first: -1e-15 would give 360
    return deg.where(series.groupby(level=0).count() > 0)


def build_city(engine: Engine, since: dt.datetime | None = None) -> int:
    """City series = mean over the model cells (at least half of them must be reporting)."""
    frames = [read_observations(engine, z.id, start=since) for z in CELLS]
    frames = [f for f in frames if len(f)]
    if not frames:
        return 0
    stack = pd.concat(frames)
    grouped = stack.groupby(level=0)
    mean = grouped.mean()
    count = grouped.count()
    mean = mean.where(count >= max(1, len(frames) // 2))
    mean["wind_dir"] = _circular_mean_deg(stack["wind_dir"]).reindex(mean.index)
    return upsert_observations(engine, CITY.id, mean)


def data_through(engine: Engine) -> dt.datetime | None:
    return latest_ts(engine, CITY.id, "pm2_5")


def backfill(engine: Engine | None = None, client: httpx.Client | None = None, start: str | None = None) -> int:
    settings = get_settings()
    engine = init_db(engine or get_engine())
    register_locations(engine, ALL_LOCATIONS)
    own_client = client is None
    client = client or make_client(settings)
    started = utcnow()
    total = 0
    try:
        first = dt.date.fromisoformat(start or settings.history_start)
        state: dict[str, bool] = {}
        for i, loc in enumerate(CELLS, 1):
            log.info("[%d/%d] backfilling %s from %s ...", i, len(CELLS), loc.name, first)
            n, report = ingest_location(engine, client, loc, first, settings=settings, state=state)
            total += n
            log.info("    %d rows, nulls: %s", n, {k: v for k, v in report["null_fraction"].items() if k in ("pm2_5", "blh")})
        total += build_city(engine)
        through = data_through(engine)
        set_meta(engine, "data_through", through.isoformat() if through else None)
        log_ingest(engine, "open-meteo", started, "ok", total, through, "backfill")
        log.info("backfill complete: %d rows, data through %s UTC", total, through)
        return total
    except Exception as exc:
        log_ingest(engine, "open-meteo", started, "error", total, None, f"backfill: {exc}")
        raise
    finally:
        if own_client:
            client.close()


def update(
    engine: Engine | None = None, client: httpx.Client | None = None, overlap_hours: int = 72
) -> int:
    """Top up every zone from (latest stored hour - overlap) to now. Falls back to backfill."""
    settings = get_settings()
    engine = init_db(engine or get_engine())
    if count_rows(engine, observations) == 0:
        return backfill(engine, client)
    register_locations(engine, ALL_LOCATIONS)
    own_client = client is None
    client = client or make_client(settings)
    started = utcnow()
    total = 0
    try:
        state: dict[str, bool] = {}
        for loc in CELLS:
            last = latest_ts(engine, loc.id, "pm2_5")
            first = (last - dt.timedelta(hours=overlap_hours)).date() if last else dt.date.fromisoformat(settings.history_start)
            n, _ = ingest_location(engine, client, loc, first, settings=settings, state=state)
            total += n
        since = utcnow() - dt.timedelta(hours=overlap_hours + 48)
        total += build_city(engine, since=since)
        through = data_through(engine)
        set_meta(engine, "data_through", through.isoformat() if through else None)
        log_ingest(engine, "open-meteo", started, "ok", total, through, "update")
        log.info("update complete: %d rows, data through %s UTC", total, through)
        return total
    except Exception as exc:
        log_ingest(engine, "open-meteo", started, "error", total, None, f"update: {exc}")
        raise
    finally:
        if own_client:
            client.close()


def status(engine: Engine | None = None) -> dict[str, Any]:
    engine = init_db(engine or get_engine())
    through = data_through(engine)
    return {
        "observation_rows": count_rows(engine, observations),
        "data_through_utc": through.isoformat() if through else None,
        "columns": OBS_COLUMNS,
    }


def quiet_http_logs() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Delhi AQI ingestion")
    parser.add_argument("command", choices=["backfill", "update", "status"])
    parser.add_argument("--start", help="backfill start date (YYYY-MM-DD)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    quiet_http_logs()
    if args.command == "backfill":
        backfill(start=args.start)
    elif args.command == "update":
        update()
    else:
        print(status())


if __name__ == "__main__":
    main()
