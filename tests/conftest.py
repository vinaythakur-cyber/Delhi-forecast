"""Shared fixtures. Every test runs offline against a throw-away SQLite file."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.db import OBS_COLUMNS, get_engine, init_db, register_locations, upsert_observations
from pipeline.locations import ALL_LOCATIONS, CELLS


@pytest.fixture()
def engine(tmp_path):
    get_engine.cache_clear()
    eng = init_db(get_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}"))
    register_locations(eng, ALL_LOCATIONS)
    yield eng
    eng.dispose()
    get_engine.cache_clear()


def synthetic_frame(hours: int = 24 * 400, seed: int = 0, end: str = "2026-01-01 00:00") -> pd.DataFrame:
    """A realistic-looking hourly series: daily cycle + winter season + AR(1) noise + weather."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp(end), periods=hours, freq="h", name="ts").as_unit("ns")
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    season = 1.0 + 1.2 * np.cos((doy - 355) / 365 * 2 * np.pi).clip(0)  # winter peak
    daily = 1.0 + 0.5 * np.sin((hour - 16) / 24 * 2 * np.pi)
    noise = np.zeros(hours)
    for i in range(1, hours):
        noise[i] = 0.97 * noise[i - 1] + rng.normal(0, 0.12)
    pm25 = 55 * season * daily * np.exp(noise)
    df = pd.DataFrame(index=idx)
    df["pm2_5"] = pm25
    df["pm10"] = pm25 * 1.7 + rng.normal(0, 5, hours)
    df["no2"] = 30 + 0.2 * pm25 + rng.normal(0, 4, hours)
    df["so2"] = 15 + rng.normal(0, 2, hours)
    df["co"] = 700 + 4 * pm25 + rng.normal(0, 50, hours)
    df["o3"] = 40 + 20 * np.sin((hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 5, hours)
    df["temp"] = 25 - 10 * np.cos((doy - 20) / 365 * 2 * np.pi) + 5 * np.sin((hour - 9) / 24 * 2 * np.pi)
    df["rh"] = (60 + rng.normal(0, 10, hours)).clip(10, 100)
    df["wind_speed"] = (8 + rng.normal(0, 3, hours)).clip(0.5, None)
    df["wind_dir"] = rng.uniform(0, 360, hours)
    df["blh"] = (600 + 500 * np.sin((hour - 9) / 24 * 2 * np.pi).clip(0) + rng.normal(0, 30, hours)).clip(50, None)
    df["precip"] = (rng.random(hours) < 0.03) * rng.exponential(1.5, hours)
    df["pressure"] = 1000 + rng.normal(0, 3, hours)
    return df[OBS_COLUMNS].clip(lower=0)


@pytest.fixture()
def seeded_engine(engine):
    """Engine pre-loaded with two cells and a city average of synthetic data."""
    frames = {}
    for k, loc in enumerate(CELLS):
        frame = synthetic_frame(seed=k)
        frames[loc.id] = frame
        upsert_observations(engine, loc.id, frame)
    city = sum(frames.values()) / len(frames)
    upsert_observations(engine, "delhi", city)
    return engine
