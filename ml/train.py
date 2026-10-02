"""Models.

LightGBM quantile forecaster
----------------------------
One *global* model per (target, quantile) serves every location and every horizon:
`horizon` and `loc_code` are ordinary inputs, so there are 2 targets x 3 quantiles = 6 boosters
instead of hundreds. Each booster predicts the *change from now* in log space,

    z = log1p(y[t+h]) - log1p(y[t])

so a forecast is anchored to persistence and the trees only learn the correction. Quantiles are
preserved by that monotone transform, so exp-ing the predicted quantiles gives valid quantiles of
the concentration itself. Predicted quantiles are sorted so they never cross.

Calibrated intervals
--------------------
Raw quantile regression under-covers: a nominal 80 % band typically contains the truth only ~70 %
of the time. We fix that with split-conformal calibration (CQR): the models are first fitted on
everything except the most recent `CALIBRATION_DAYS` of targets, the band is widened (or narrowed)
by the amount that would have made it cover 80 % on that held-out period, per horizon bucket, then
the models are refitted on all the data and the learned widening is applied. The calibration
period is strictly older than any forecast the model makes, so no future information is used.

SARIMA (statsmodels)
--------------------
A classical single-series baseline, used only inside the evaluation (see evaluate.py).
"""

from __future__ import annotations

import gzip
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from pipeline.features import TARGETS

QUANTILE_LEVELS = {"q10": 0.1, "q50": 0.5, "q90": 0.9}
N_ROUNDS = 250
CALIBRATION_DAYS = 120
COVERAGE = 0.8
HORIZON_BUCKETS = [3, 12, 36, 72]  # upper edges; each bucket gets its own widening

BASE_PARAMS: dict[str, Any] = {
    "objective": "quantile",
    "learning_rate": 0.06,
    "num_leaves": 31,
    "min_data_in_leaf": 60,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "verbosity": -1,
    "seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": max(1, (os.cpu_count() or 2)),
}


def _design(rows: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    return rows[list(features)].to_numpy(dtype="float32")


@dataclass
class QuantileForecaster:
    features: list[str] = field(default_factory=list)
    boosters: dict[str, dict[str, lgb.Booster]] = field(default_factory=dict)
    offsets: dict[str, list[float]] = field(default_factory=dict)  # conformal widening, log space, per horizon bucket
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ training
    def _fit_boosters(self, rows: pd.DataFrame, targets: Sequence[str]) -> dict[str, dict[str, lgb.Booster]]:
        cat = ["loc_code"] if "loc_code" in self.features else []
        boosters: dict[str, dict[str, lgb.Booster]] = {}
        for tgt in targets:
            sub = rows.dropna(subset=[f"y_{tgt}", f"cur_{tgt}"])
            z = np.log1p(sub[f"y_{tgt}"].to_numpy()) - np.log1p(sub[f"cur_{tgt}"].to_numpy())
            x = _design(sub, self.features)
            boosters[tgt] = {}
            for name, alpha in QUANTILE_LEVELS.items():
                params = {**BASE_PARAMS, "alpha": alpha}
                data = lgb.Dataset(x, label=z, feature_name=self.features, categorical_feature=cat, free_raw_data=False)
                boosters[tgt][name] = lgb.train(params, data, num_boost_round=N_ROUNDS)
        return boosters

    def fit(
        self,
        rows: pd.DataFrame,
        features: Sequence[str],
        targets: Sequence[str] = TARGETS,
        calibrate_days: int | None = CALIBRATION_DAYS,
    ) -> QuantileForecaster:
        self.features = list(features)
        self.offsets = {t: [0.0] * len(HORIZON_BUCKETS) for t in targets}
        if calibrate_days:
            split = rows["target_ts"].max() - pd.Timedelta(days=calibrate_days)
            proper, calib = rows[rows["target_ts"] < split], rows[rows["target_ts"] >= split]
            if len(proper) > 1000 and len(calib) > 200:
                self.boosters = self._fit_boosters(proper, targets)
                for tgt in targets:
                    self.offsets[tgt] = self._conformal_offsets(calib, tgt)
        self.boosters = self._fit_boosters(rows, targets)
        return self

    def _raw_log_quantiles(self, rows: pd.DataFrame, target: str) -> np.ndarray:
        x = _design(rows, self.features)
        return np.column_stack([self.boosters[target][q].predict(x) for q in QUANTILE_LEVELS])

    def _conformal_offsets(self, calib: pd.DataFrame, target: str) -> list[float]:
        sub = calib.dropna(subset=[f"y_{target}", f"cur_{target}"])
        z = np.log1p(sub[f"y_{target}"].to_numpy()) - np.log1p(sub[f"cur_{target}"].to_numpy())
        raw = self._raw_log_quantiles(sub, target)
        scores = np.maximum(raw[:, 0] - z, z - raw[:, 2])  # how far outside the band the truth fell
        bucket = np.searchsorted(HORIZON_BUCKETS, sub["horizon"].to_numpy(), side="left")
        offsets = []
        for b in range(len(HORIZON_BUCKETS)):
            s_b = scores[bucket == b]
            if len(s_b) < 50:
                offsets.append(0.0)
                continue
            level = min(1.0, COVERAGE * (1 + 1 / len(s_b)))
            offsets.append(float(np.quantile(s_b, level)))
        return offsets

    # ------------------------------------------------------------------ inference
    def predict(self, rows: pd.DataFrame, target: str) -> pd.DataFrame:
        """Quantile forecasts of `target` (ug/m3) for every row; columns q10, q50, q90, never crossing."""
        base = np.log1p(rows[f"cur_{target}"].to_numpy(dtype=float))
        raw = self._raw_log_quantiles(rows, target).copy()
        bucket = np.minimum(np.searchsorted(HORIZON_BUCKETS, rows["horizon"].to_numpy(), side="left"), len(HORIZON_BUCKETS) - 1)
        widen = np.asarray(self.offsets.get(target, [0.0] * len(HORIZON_BUCKETS)))[bucket]
        raw[:, 0] -= widen
        raw[:, 2] += widen
        values = np.expm1(base[:, None] + raw)
        values = np.sort(np.clip(values, 0, None), axis=1)
        return pd.DataFrame(values, columns=list(QUANTILE_LEVELS), index=rows.index)

    # ------------------------------------------------------------------ persistence
    def to_bytes(self) -> bytes:
        payload = {
            "features": self.features,
            "offsets": self.offsets,
            "meta": self.meta,
            "models": {t: {q: b.model_to_string() for q, b in qs.items()} for t, qs in self.boosters.items()},
        }
        return gzip.compress(json.dumps(payload).encode("utf-8"))

    @classmethod
    def from_bytes(cls, blob: bytes) -> QuantileForecaster:
        payload = json.loads(gzip.decompress(blob).decode("utf-8"))
        boosters = {t: {q: lgb.Booster(model_str=s) for q, s in qs.items()} for t, qs in payload["models"].items()}
        return cls(features=payload["features"], boosters=boosters, offsets=payload.get("offsets", {}), meta=payload.get("meta", {}))

    def feature_importance(self, target: str = "pm2_5", top: int = 15) -> list[tuple[str, float]]:
        booster = self.boosters[target]["q50"]
        gain = booster.feature_importance(importance_type="gain")
        pairs = sorted(zip(self.features, gain, strict=True), key=lambda p: -p[1])[:top]
        total = float(sum(gain)) or 1.0
        return [(n, round(float(g) / total, 4)) for n, g in pairs]


# ----------------------------------------------------------------------------- SARIMA


def sarima_forecast_origins(
    series: pd.Series,
    cutoff: pd.Timestamp,
    origins: Sequence[pd.Timestamp],
    horizons: Sequence[int],
    train_days: int = 60,
    order: tuple[int, int, int] = (2, 0, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24),
) -> dict[pd.Timestamp, dict[int, float]]:
    """Fit SARIMA on the `train_days` before `cutoff`, then roll forward through `origins`.

    The parameters are fitted once, on data strictly before `cutoff`. For each origin the model's
    state is extended with the data observed up to and including that origin (no refit) and asked
    for 72 steps. Log1p keeps forecasts positive. Returns {origin: {horizon: ug/m3}}.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    s = np.log1p(series.asfreq("h"))
    train = s.loc[cutoff - pd.Timedelta(days=train_days) : cutoff - pd.Timedelta(hours=1)]
    fit = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit(
        disp=False, maxiter=60
    )
    out: dict[pd.Timestamp, dict[int, float]] = {}
    res = fit
    last = train.index[-1]
    for origin in sorted(origins):
        chunk = s.loc[last + pd.Timedelta(hours=1) : origin]
        if len(chunk):
            res = res.append(chunk, refit=False)
            last = chunk.index[-1]
        fc = np.expm1(np.asarray(res.forecast(steps=max(horizons))))
        out[origin] = {h: float(fc[h - 1]) for h in horizons}
    return out
