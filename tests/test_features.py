"""Feature tests. The leakage tests are the most important tests in the repository."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.baselines import baseline_predictions, persistence, seasonal_lag_hours
from pipeline.db import OBS_COLUMNS
from pipeline.features import (
    EVAL_HORIZONS,
    TRAIN_HORIZONS,
    build_rows,
    calendar_features,
    feature_columns,
    origin_features,
)
from tests.conftest import synthetic_frame


@pytest.fixture(scope="module")
def frame():
    return synthetic_frame(hours=24 * 60, seed=3)


def _same(a: pd.Series, b: pd.Series) -> bool:
    return np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)


@pytest.mark.parametrize("t_pos", [200, 500, 900, 1300])
def test_features_at_t_ignore_everything_after_t(frame, t_pos):
    """Destroy all data after t; every feature at t (and before) must be unchanged."""
    base = origin_features(frame)
    corrupted = frame.copy()
    rng = np.random.default_rng(0)
    corrupted.iloc[t_pos + 1 :] = rng.uniform(1e3, 1e5, size=corrupted.iloc[t_pos + 1 :].shape)
    changed = origin_features(corrupted)
    for col in base.columns:
        assert _same(base[col].iloc[: t_pos + 1], changed[col].iloc[: t_pos + 1]), f"{col} leaks the future"


def test_leakage_test_would_catch_a_leak(frame):
    """Sanity check of the test itself: a centred rolling mean must fail the same check."""
    t_pos = 500
    leaky = frame["pm2_5"].rolling(5, center=True).mean()
    corrupted = frame.copy()
    corrupted.iloc[t_pos + 1 :] = 1e5
    leaky_c = corrupted["pm2_5"].rolling(5, center=True).mean()
    assert not _same(leaky.iloc[: t_pos + 1], leaky_c.iloc[: t_pos + 1])


def test_features_do_use_the_value_at_t(frame):
    t_pos = 400
    changed = frame.copy()
    changed.iloc[t_pos, changed.columns.get_loc("pm2_5")] += 100
    a, b = origin_features(frame), origin_features(changed)
    assert a["pm2_5_lag0"].iloc[t_pos] != b["pm2_5_lag0"].iloc[t_pos]
    assert a["pm2_5_lag1"].iloc[t_pos] == b["pm2_5_lag1"].iloc[t_pos]
    assert a["pm2_5_lag1"].iloc[t_pos + 1] != b["pm2_5_lag1"].iloc[t_pos + 1]


def test_lag_columns_mean_k_hours_ago(frame):
    f = origin_features(frame)
    y = frame["pm2_5"]
    for k in [1, 6, 24, 168]:
        assert f[f"pm2_5_lag{k}"].iloc[300] == y.iloc[300 - k]
    assert f["pm2_5_rm24"].iloc[300] == pytest.approx(y.iloc[277:301].mean())  # window ends AT t, 24 values


def test_gaps_become_nan_not_misaligned_lags(frame):
    gappy = frame.drop(frame.index[300:310])  # remove 10 hours from the index entirely
    f = origin_features(gappy)
    t = frame.index[320]
    assert f.loc[t, "pm2_5_lag24"] == frame["pm2_5"].loc[t - pd.Timedelta(hours=24)]
    assert np.isnan(f.loc[frame.index[305], "pm2_5_lag0"])


def test_calendar_features_known_values():
    ts = pd.DatetimeIndex(["2024-11-01 06:30", "2024-10-25 18:30", "2025-01-26 00:00", "2024-07-04 12:00"])
    cal = calendar_features(ts)
    assert cal["diwali_window"].tolist() == [1, 0, 0, 0]  # 06:30 UTC is 12:00 IST on Diwali day
    assert cal["diwali_offset"].iloc[0] == 0 and cal["diwali_offset"].iloc[1] == -6  # 18:30 UTC on 25 Oct is already 26 Oct in IST
    assert np.isnan(cal["diwali_offset"].iloc[3])
    assert cal["is_holiday"].tolist() == [1, 0, 1, 0]  # Diwali day, Republic Day
    assert cal["stubble_season"].tolist() == [1, 1, 0, 0]
    assert cal["winter"].tolist() == [1, 0, 1, 0]
    assert cal["is_weekend"].iloc[1] == 0 and calendar_features(pd.DatetimeIndex(["2024-11-02 06:00"]))["is_weekend"].iloc[0] == 1


def test_calendar_uses_delhi_local_hour():
    cal = calendar_features(pd.DatetimeIndex(["2024-01-01 18:30"]))  # 00:00 IST next day
    assert cal["hour_sin"].iloc[0] == pytest.approx(0, abs=1e-9) and cal["hour_cos"].iloc[0] == pytest.approx(1)
    assert cal["dow"].iloc[0] == 1  # Tuesday 2 Jan in IST


def test_build_rows_targets_and_timestamps_are_correct(frame):
    rows = build_rows({"delhi": frame}, EVAL_HORIZONS, stride=7)
    assert (rows["target_ts"] - rows["origin_ts"] == pd.to_timedelta(rows["horizon"], unit="h")).all()
    assert (rows["horizon"] > 0).all()
    series = frame["pm2_5"]
    sample = rows.sample(200, random_state=1)
    for _, r in sample.iterrows():
        assert r["y_pm2_5"] == series.loc[r["target_ts"]]
        assert r["cur_pm2_5"] == series.loc[r["origin_ts"]]
        assert r["pm2_5_lag0"] == r["cur_pm2_5"]


def test_build_rows_never_uses_a_target_beyond_the_data(frame):
    rows = build_rows({"delhi": frame}, [72], stride=1)
    assert rows["target_ts"].max() <= frame.index.max()
    assert len(rows) == len(frame) - 72


def test_build_rows_random_horizons_and_filter(frame):
    rows = build_rows({"delhi": frame}, TRAIN_HORIZONS, stride=6, n_random=2, seed=1,
                      origin_filter=lambda i: i < pd.Timestamp(frame.index[500]))
    assert set(rows["horizon"]) <= set(TRAIN_HORIZONS) and len(set(rows["horizon"])) > 5
    assert rows["origin_ts"].max() < frame.index[500]
    again = build_rows({"delhi": frame}, TRAIN_HORIZONS, stride=6, n_random=2, seed=1,
                       origin_filter=lambda i: i < pd.Timestamp(frame.index[500]))
    pd.testing.assert_frame_equal(rows, again)  # seeded, hence reproducible


def test_feature_columns_exclude_metadata_and_targets(frame):
    rows = build_rows({"delhi": frame}, [24], stride=24)
    cols = feature_columns(rows)
    assert not any(c.startswith(("y_", "cur_")) for c in cols)
    assert not {"location", "origin_ts", "target_ts"} & set(cols)
    assert "horizon" in cols and "pm2_5_lag0" in cols and "diwali_window" in cols


def test_persistence_is_the_value_at_the_origin(frame):
    rows = build_rows({"delhi": frame}, [1, 24], stride=10)
    assert (persistence(rows, "pm2_5") == rows["pm2_5_lag0"].to_numpy()).all()


@pytest.mark.parametrize(("h", "offset"), [(1, -23), (5, -19), (24, 0), (25, -23), (30, -18), (48, 0), (72, 0)])
def test_seasonal_naive_looks_one_whole_number_of_days_back(h, offset):
    assert seasonal_lag_hours(h) == offset


def test_seasonal_naive_never_reads_the_future_and_matches_same_hour_of_day(frame):
    for h in range(1, 73):
        assert seasonal_lag_hours(h) <= 0  # at or before the origin
        assert (seasonal_lag_hours(h) - h) % 24 == 0  # same hour of day as the target
    rows = build_rows({"delhi": frame}, [30], stride=50)
    preds = baseline_predictions(rows, {"delhi": frame}, "pm2_5")
    r = rows.iloc[3]
    assert preds["seasonal_naive"][3] == frame["pm2_5"].loc[r["origin_ts"] - pd.Timedelta(hours=18)]
    assert preds["persistence"][3] == r["cur_pm2_5"]


def test_all_observation_columns_are_present_in_synthetic_data(frame):
    assert list(frame.columns) == OBS_COLUMNS
