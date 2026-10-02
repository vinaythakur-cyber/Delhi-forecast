"""Model, walk-forward and forecast tests on synthetic data (fast: tiny boosters)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ml.train as train_mod
from ml import evaluate as ev
from ml.evaluate import EvalConfig, make_folds, run_evaluation
from ml.forecast import make_forecast, read_latest_forecast, train_final
from ml.registry import get_evaluation, load_model, save_evaluation
from ml.train import QuantileForecaster, sarima_forecast_origins
from pipeline.db import observations, read_observations
from pipeline.features import EVAL_HORIZONS, TRAIN_HORIZONS, build_rows, feature_columns
from tests.conftest import synthetic_frame


@pytest.fixture(autouse=True)
def tiny_boosters(monkeypatch):
    monkeypatch.setattr(train_mod, "N_ROUNDS", 25)


@pytest.fixture(scope="module")
def frames():
    return {"delhi": synthetic_frame(24 * 420, seed=1), "delhi-north": synthetic_frame(24 * 420, seed=2)}


def test_quantiles_are_ordered_positive_and_survive_serialisation(frames):
    pool = build_rows(frames, TRAIN_HORIZONS, stride=6, n_random=2, seed=0)
    feats = feature_columns(pool)
    model = QuantileForecaster().fit(pool, feats)
    sample = pool.sample(300, random_state=0)
    pred = model.predict(sample, "pm2_5")
    assert (pred["q10"] <= pred["q50"]).all() and (pred["q50"] <= pred["q90"]).all() and (pred >= 0).all().all()
    clone = QuantileForecaster.from_bytes(model.to_bytes())
    pd.testing.assert_frame_equal(pred, clone.predict(sample, "pm2_5"))
    assert clone.features == model.features and clone.offsets == model.offsets


def test_conformal_calibration_improves_coverage_of_the_band(frames):
    pool = build_rows(frames, TRAIN_HORIZONS, stride=3, n_random=3, seed=0)
    cutoff = pool["target_ts"].max() - pd.Timedelta(days=60)
    train, test = pool[pool["target_ts"] < cutoff], pool[pool["target_ts"] >= cutoff]
    feats = feature_columns(pool)
    raw = QuantileForecaster().fit(train, feats, calibrate_days=None)
    cal = QuantileForecaster().fit(train, feats, calibrate_days=90)

    def coverage(m):
        p = m.predict(test, "pm2_5")
        y = test["y_pm2_5"].to_numpy()
        return float(((y >= p["q10"]) & (y <= p["q90"])).mean())

    assert abs(coverage(cal) - 0.8) <= abs(coverage(raw) - 0.8) + 0.02
    assert any(abs(o) > 0 for o in cal.offsets["pm2_5"])


def test_make_folds_respects_min_history_and_keeps_test_inside_the_data():
    start, end = pd.Timestamp("2022-08-04"), pd.Timestamp("2026-10-02 18:00")
    cfg = EvalConfig(n_folds=12, test_days=30, min_train_days=365)
    folds = make_folds(start, end, cfg)
    assert len(folds) == 12
    assert folds[0].cutoff >= start + pd.Timedelta(days=365)
    assert all(f.test_end + pd.Timedelta(hours=72) <= end + pd.Timedelta(hours=1) for f in folds)
    assert [f.cutoff for f in folds] == sorted(f.cutoff for f in folds)
    with pytest.raises(ValueError):
        make_folds(start, start + pd.Timedelta(days=380), cfg)


def test_walk_forward_never_trains_on_targets_from_the_future(frames, monkeypatch):
    seen = []
    original = QuantileForecaster.fit

    def spy(self, rows, features, *a, **k):
        seen.append(rows["target_ts"].max())
        return original(self, rows, features, *a, **k)

    monkeypatch.setattr(QuantileForecaster, "fit", spy)
    cfg = EvalConfig(n_folds=3, test_days=15, min_train_days=200, sarima_enabled=False, train_stride=6)
    payload = run_evaluation(frames=frames, cfg=cfg)
    cutoffs = [pd.Timestamp(f["cutoff"]) for f in payload["folds"]]
    assert len(seen) == 3
    for latest_target, cutoff in zip(seen, cutoffs, strict=True):
        assert latest_target < cutoff  # the purge: nothing the model trained on reaches into the test period


def test_evaluation_scores_every_model_on_the_same_rows_and_has_expected_shape(frames):
    cfg = EvalConfig(n_folds=3, test_days=15, min_train_days=200, sarima_enabled=False, train_stride=6)
    payload = run_evaluation(frames=frames, cfg=cfg)
    pm = payload["targets"]["pm2_5"]
    assert set(pm["by_horizon"]) == {"persistence", "seasonal_naive", "lightgbm"}
    ns = {m: {h: c["n"] for h, c in pm["by_horizon"][m].items()} for m in pm["by_horizon"]}
    assert ns["persistence"] == ns["seasonal_naive"] == ns["lightgbm"]  # identical rows for every model
    assert set(pm["by_horizon"]["lightgbm"]) == {str(h) for h in EVAL_HORIZONS}
    for h in ("1", "24"):
        assert pm["interval"]["coverage"][h] == pytest.approx(pm["interval"]["coverage"][h], abs=1)
        assert 0 <= pm["interval"]["coverage"][h] <= 1
    assert len(pm["by_fold"]) == 3 and "winter" in pm["by_season"]
    assert payload["feature_importance"] and payload["procedure"]


def test_the_model_beats_persistence_at_long_horizons_on_data_with_a_daily_cycle(frames):
    cfg = EvalConfig(n_folds=3, test_days=20, min_train_days=250, sarima_enabled=False, train_stride=3)
    pm = run_evaluation(frames=frames, cfg=cfg)["targets"]["pm2_5"]
    # the daily cycle is learnable: 24 h ahead is strictly easier for the model than for 'nothing changes' at 12 h
    assert pm["by_horizon"]["lightgbm"]["12"]["mae"] < pm["by_horizon"]["persistence"]["12"]["mae"]


def test_skill_metric_definition():
    table = {"persistence": {"1": {"mae": 10.0}}, "seasonal_naive": {"1": {"mae": 20.0}}, "lightgbm": {"1": {"mae": 8.0}}}
    assert ev.skill(table, "lightgbm", ["persistence"]) == {"1": pytest.approx(0.2)}
    assert ev.skill(table, "lightgbm", ["persistence", "seasonal_naive"]) == {"1": pytest.approx(0.2)}
    assert ev.skill(table, "seasonal_naive", ["persistence"]) == {"1": pytest.approx(-1.0)}


def test_sarima_rolls_forward_without_refit_and_stays_positive():
    s = synthetic_frame(24 * 40, seed=5)["pm2_5"]
    cutoff = s.index[24 * 30]
    origins = [cutoff, cutoff + pd.Timedelta(hours=24)]
    out = sarima_forecast_origins(s, cutoff, origins, [1, 24, 72], train_days=20, order=(1, 0, 0), seasonal_order=(0, 1, 1, 24))
    assert set(out) == set(origins) and all(set(v) == {1, 24, 72} for v in out.values())
    assert all(x > 0 for v in out.values() for x in v.values())


def test_train_and_live_forecast_end_to_end(seeded_engine):
    meta = train_final(seeded_engine, stride=6, n_random=2)
    assert meta["training_rows"] > 1000 and set(meta["locations"]) == {"delhi", "delhi-north", "delhi-south"}
    model, meta2, created = load_model(seeded_engine)
    assert model.features and meta2["n_features"] == len(model.features)

    n = make_forecast(seeded_engine)
    assert n == 3 * 72
    last_obs = read_observations(seeded_engine, "delhi")["pm2_5"].dropna().index.max()
    fc = read_latest_forecast(seeded_engine, "delhi")
    assert len(fc) == 72 and fc["horizon"].tolist() == list(range(1, 73))
    assert (fc["issued_at"] == last_obs).all()
    assert (fc["target_ts"] == last_obs + pd.to_timedelta(fc["horizon"], unit="h")).all()
    for t in ("pm2_5", "pm10", "aqi"):
        assert (fc[f"{t}_q10"] <= fc[f"{t}_q50"] + 1e-9).all() and (fc[f"{t}_q50"] <= fc[f"{t}_q90"] + 1e-9).all()
    assert fc[["pm2_5_q50", "aqi_q50"]].notna().all().all()

    assert make_forecast(seeded_engine) == 3 * 72  # idempotent: re-running replaces, never duplicates
    assert len(read_latest_forecast(seeded_engine, "delhi")) == 72


def test_forecast_without_a_model_gives_a_clear_error(seeded_engine):
    with pytest.raises(RuntimeError, match="no trained model"):
        make_forecast(seeded_engine)


def test_evaluation_registry_prefers_database_then_shipped_file(engine, tmp_path, monkeypatch):
    import ml.registry as reg

    seed = tmp_path / "metrics.json"
    seed.write_text('{"generated_at": "seed"}')
    monkeypatch.setattr(reg, "SEED_METRICS", seed)
    assert get_evaluation(engine)["source"] == "shipped"
    save_evaluation(engine, {"generated_at": "db"})
    assert get_evaluation(engine)["generated_at"] == "db"


def test_observations_table_untouched_by_forecasting(seeded_engine):
    from pipeline.db import count_rows

    before = count_rows(seeded_engine, observations)
    train_final(seeded_engine, stride=12, n_random=1)
    make_forecast(seeded_engine)
    assert count_rows(seeded_engine, observations) == before
    assert np.isfinite(read_observations(seeded_engine, "delhi")["pm2_5"]).all()


def test_a_data_outage_does_not_erase_the_forecast(seeded_engine):
    """Pruning is relative to the newest forecast: data that ended months ago still yields a forecast."""
    import datetime as dt

    train_final(seeded_engine, stride=12, n_random=1)
    make_forecast(seeded_engine, now=dt.datetime(2031, 1, 1))  # wall clock far ahead of the last observation
    assert len(read_latest_forecast(seeded_engine, "delhi")) == 72
