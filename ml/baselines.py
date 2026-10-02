"""The bar every model must clear.

persistence      "nothing changes": the forecast for t+h is the value at t.
seasonal naive   "same hour yesterday": the forecast for t+h is the most recent value that was
                 observed at the same hour of day as t+h, i.e. y(t + h - 24 * ceil(h / 24)).
                 It only ever reads data at or before t.

Both need no training, so they can be scored on any row table from `pipeline.features.build_rows`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SEASON = 24


def persistence(rows: pd.DataFrame, target: str) -> np.ndarray:
    return rows[f"cur_{target}"].to_numpy(dtype=float)


def seasonal_lag_hours(horizon: int, season: int = SEASON) -> int:
    """How many hours before the origin the seasonal-naive forecast looks (0 or negative offset)."""
    return horizon - season * math.ceil(horizon / season)  # always in (-season, 0]


def seasonal_naive(rows: pd.DataFrame, frames: dict[str, pd.DataFrame], target: str) -> np.ndarray:
    """Vectorised seasonal naive; `frames` are the hourly observations per location."""
    out = np.full(len(rows), np.nan)
    offsets = rows["horizon"].map(seasonal_lag_hours).to_numpy()
    for loc, frame in frames.items():
        mask = (rows["location"] == loc).to_numpy()
        if not mask.any():
            continue
        series = frame[target].asfreq("h")
        src = pd.DatetimeIndex(rows.loc[mask, "origin_ts"]) + pd.to_timedelta(offsets[mask], unit="h")
        out[mask] = series.reindex(src).to_numpy(dtype=float)
    return out


def baseline_predictions(rows: pd.DataFrame, frames: dict[str, pd.DataFrame], target: str) -> dict[str, np.ndarray]:
    seasonal = seasonal_naive(rows, frames, target)
    # at horizons where the seasonal source hour is unavailable fall back to persistence
    seasonal = np.where(np.isnan(seasonal), persistence(rows, target), seasonal)
    return {"persistence": persistence(rows, target), "seasonal_naive": seasonal}
