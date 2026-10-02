"""Leakage-safe features for direct multi-horizon forecasting.

Vocabulary
----------
origin t     the last hour we have observed; "now" at forecast time
horizon h    how many hours ahead we predict, 1..72
target       the value at t + h

THE RULE: every origin feature at row t is a function of data at or before t. That is enforced by
construction (only `shift(k >= 0)` and trailing windows ending at t are used) and checked by
tests/test_features.py, which destroys everything after t and asserts the features do not change.

Calendar features describe the *target* time t + h. They are legitimate because the calendar is
known in advance; they involve no observed data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

TARGETS = ["pm2_5", "pm10"]
EVAL_HORIZONS = [1, 3, 6, 12, 24, 48, 72]
TRAIN_HORIZONS = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24, 30, 36, 42, 48, 54, 60, 66, 72]
MAX_HORIZON = 72
LOCATION_CODES = {"delhi": 0, "delhi-north": 1, "delhi-south": 2}

LAGS = [0, 1, 2, 3, 6, 12, 24, 48, 168]
ROLL_WINDOWS = [6, 24, 72, 168]

# Approximate dates of the main Diwali day (Lakshmi Puja). Good to within a day, which is all a
# +/- 3 day window needs. Extend this list when the project runs past 2030.
DIWALI = pd.to_datetime(
    ["2020-11-14", "2021-11-04", "2022-10-24", "2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08",
     "2027-10-29", "2028-10-17", "2029-11-05", "2030-10-26"]
).to_numpy(dtype="datetime64[D]")

FIXED_HOLIDAYS = {(1, 1), (1, 26), (8, 15), (10, 2), (12, 25)}  # New Year, Republic, Independence, Gandhi Jayanti, Christmas


def calendar_features(ts: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar descriptors of timestamps (naive UTC in, Delhi local time out). No observed data."""
    ist = pd.DatetimeIndex(ts) + pd.Timedelta(hours=5, minutes=30)
    hour = ist.hour.to_numpy()
    month = ist.month.to_numpy()
    doy = ist.dayofyear.to_numpy()
    day = ist.normalize().to_numpy(dtype="datetime64[D]")

    # signed days since the nearest Diwali (negative = before it)
    pos = np.searchsorted(DIWALI, day)
    prev = DIWALI[np.clip(pos - 1, 0, len(DIWALI) - 1)]
    nxt = DIWALI[np.clip(pos, 0, len(DIWALI) - 1)]
    d_prev = (day - prev).astype(int)
    d_next = (day - nxt).astype(int)
    since = np.where(np.abs(d_prev) <= np.abs(d_next), d_prev, d_next)

    holiday = np.array([(m, d) in FIXED_HOLIDAYS for m, d in zip(month, ist.day, strict=True)]) | (since == 0)
    out = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "month_sin": np.sin(2 * np.pi * (month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (month - 1) / 12),
            "doy_sin": np.sin(2 * np.pi * doy / 366),
            "doy_cos": np.cos(2 * np.pi * doy / 366),
            "dow": ist.dayofweek.to_numpy(),
            "is_weekend": (ist.dayofweek.to_numpy() >= 5).astype(int),
            "is_holiday": holiday.astype(int),
            "diwali_offset": np.where((since >= -7) & (since <= 10), since, np.nan),
            "diwali_window": ((since >= -2) & (since <= 3)).astype(int),
            "stubble_season": (((month == 10) & (ist.day.to_numpy() >= 10)) | ((month == 11) & (ist.day.to_numpy() <= 30))).astype(int),
            "winter": np.isin(month, [11, 12, 1, 2]).astype(int),
        },
        index=ts,
    )
    return out


def origin_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Features at every hour t using only data at or before t. `frame` is hourly observations."""
    df = frame.asfreq("h")  # regular grid; gaps become NaN so shifts mean "k hours ago"
    out: dict[str, pd.Series] = {}

    for tgt in TARGETS:
        s = df[tgt]
        for k in LAGS:
            out[f"{tgt}_lag{k}"] = s.shift(k)
        for w in ROLL_WINDOWS:
            out[f"{tgt}_rm{w}"] = s.rolling(w, min_periods=max(2, w // 2)).mean()
        out[f"{tgt}_rs24"] = s.rolling(24, min_periods=12).std()
        out[f"{tgt}_d1"] = s - s.shift(1)
        out[f"{tgt}_d6"] = s - s.shift(6)
    out["pm_ratio"] = df["pm10"] / df["pm2_5"].where(df["pm2_5"] > 0)

    for col in ["no2", "so2", "co", "o3"]:
        out[f"{col}_lag0"] = df[col]
    for col in ["no2", "co", "o3"]:
        out[f"{col}_rm24"] = df[col].rolling(24, min_periods=12).mean()

    out["temp"] = df["temp"]
    out["rh"] = df["rh"]
    out["wind_speed"] = df["wind_speed"]
    out["wind_speed_rm6"] = df["wind_speed"].rolling(6, min_periods=3).mean()
    rad = np.deg2rad(df["wind_dir"])
    out["wind_sin"] = np.sin(rad)
    out["wind_cos"] = np.cos(rad)
    out["blh"] = df["blh"]
    out["blh_rm6"] = df["blh"].rolling(6, min_periods=3).mean()
    out["precip_sum6"] = df["precip"].rolling(6, min_periods=3).sum()
    out["precip_sum24"] = df["precip"].rolling(24, min_periods=12).sum()
    out["pressure"] = df["pressure"]
    out["pressure_d6"] = df["pressure"] - df["pressure"].shift(6)
    return pd.DataFrame(out, index=df.index)


META_COLUMNS = ["location", "origin_ts", "target_ts"] + [f"y_{t}" for t in TARGETS] + [f"cur_{t}" for t in TARGETS]


def build_rows(
    frames: dict[str, pd.DataFrame],
    horizons: Sequence[int],
    *,
    stride: int = 1,
    n_random: int | None = None,
    seed: int = 0,
    origin_filter: Callable[[pd.DatetimeIndex], np.ndarray] | None = None,
) -> pd.DataFrame:
    """Stack (origin, horizon) rows for all locations into one training / evaluation table.

    stride      keep every `stride`-th origin hour (neighbouring hours are nearly duplicates)
    n_random    if set, give every origin that many randomly drawn horizons instead of all of them
    origin_filter  optional function(origin index) -> boolean mask, e.g. "origins inside this fold"
    Rows carry the future values (y_*) so no other table is needed for scoring.
    """
    rng = np.random.default_rng(seed)
    horizons_arr = np.asarray(list(horizons))
    parts: list[pd.DataFrame] = []
    for loc, frame in frames.items():
        feats = origin_features(frame)
        df = frame.asfreq("h")
        idx = feats.index
        origins = np.arange(0, len(idx), stride)
        if origin_filter is not None:
            origins = origins[origin_filter(idx[origins])]
        if len(origins) == 0:
            continue
        if n_random is None:
            o_all = np.repeat(origins, len(horizons_arr))
            h_all = np.tile(horizons_arr, len(origins))
        else:
            o_all = np.repeat(origins, n_random)
            h_all = rng.choice(horizons_arr, size=len(o_all))
        keep = o_all + h_all < len(idx)
        o_all, h_all = o_all[keep], h_all[keep]
        block = feats.iloc[o_all].reset_index(drop=True)
        block.insert(0, "location", loc)
        block.insert(1, "origin_ts", idx[o_all])
        block.insert(2, "target_ts", idx[o_all + h_all])
        for tgt in TARGETS:
            values = df[tgt].to_numpy()
            block[f"y_{tgt}"] = values[o_all + h_all]
            block[f"cur_{tgt}"] = values[o_all]
        block["horizon"] = h_all
        block["loc_code"] = LOCATION_CODES.get(loc, -1)
        cal = calendar_features(pd.DatetimeIndex(block["target_ts"]))
        cal.index = block.index
        parts.append(pd.concat([block, cal], axis=1))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def feature_columns(rows: pd.DataFrame) -> list[str]:
    """Model inputs = everything that is not metadata or a target."""
    return [c for c in rows.columns if c not in META_COLUMNS]
