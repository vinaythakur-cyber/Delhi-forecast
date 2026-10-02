"""Walk-forward evaluation. This is the only evaluation in the project; there is no random split.

Procedure
---------
1. Pick `n_folds` cut-off times spread over the history, each with at least `min_train_days` of
   history before it.
2. For fold k, train on rows whose TARGET time is before the cut-off. (Rows that start before the
   cut-off but end after it are purged, so no training example contains information from the test
   period.)
3. Test on forecast origins in [cut-off, cut-off + test_days), every `origin_step_h` hours, at the
   horizons in `horizons`. Features at each origin only use data up to that origin.
4. Every model is scored on exactly the same rows. Persistence and seasonal naive need no training.
5. Report MAE and RMSE per horizon, skill against persistence and against the better baseline,
   interval coverage of the 10-90 % band, a winter/other split, and a per-fold breakdown.

Run:  python -m ml.evaluate            (about 5-8 minutes on a laptop)
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from ml.baselines import baseline_predictions
from ml.registry import save_evaluation
from ml.train import QuantileForecaster, sarima_forecast_origins
from pipeline.db import get_engine, init_db, read_observations, utcnow
from pipeline.features import EVAL_HORIZONS, TARGETS, TRAIN_HORIZONS, build_rows, feature_columns

log = logging.getLogger("ml.evaluate")

LOCATIONS = ["delhi", "delhi-north", "delhi-south"]
PRIMARY = "delhi"
WINTER_MONTHS = (11, 12, 1, 2)


@dataclass
class EvalConfig:
    n_folds: int = 12
    test_days: int = 30
    min_train_days: int = 365
    origin_step_h: int = 6
    horizons: tuple[int, ...] = tuple(EVAL_HORIZONS)
    train_stride: int = 3
    train_n_random: int = 3
    sarima_every: int = 2  # run SARIMA on every 2nd fold (it is slow)
    sarima_enabled: bool = True
    seed: int = 0


@dataclass
class Fold:
    index: int
    cutoff: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(start: pd.Timestamp, end: pd.Timestamp, cfg: EvalConfig) -> list[Fold]:
    first = start + pd.Timedelta(days=cfg.min_train_days)
    last = end - pd.Timedelta(days=cfg.test_days) - pd.Timedelta(hours=max(cfg.horizons))
    if last <= first:
        raise ValueError("not enough history for the requested folds")
    cutoffs = pd.date_range(first, last, periods=cfg.n_folds).floor("h")
    return [Fold(i + 1, c, c + pd.Timedelta(days=cfg.test_days)) for i, c in enumerate(cutoffs)]


# ----------------------------------------------------------------------------- metrics


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def model_columns(target: str) -> dict[str, str]:
    return {"persistence": f"persistence_{target}", "seasonal_naive": f"seasonal_naive_{target}", "lightgbm": f"lgb_{target}_q50"}


def valid_rows(ev: pd.DataFrame, target: str, extra: list[str] | None = None) -> pd.DataFrame:
    cols = [f"y_{target}", *model_columns(target).values(), f"lgb_{target}_q10", f"lgb_{target}_q90", *(extra or [])]
    return ev.dropna(subset=cols)


def per_horizon(rows: pd.DataFrame, target: str, columns: dict[str, str], horizons) -> dict[str, dict[str, dict[str, float]]]:
    y_col = f"y_{target}"
    out: dict[str, dict[str, dict[str, float]]] = {m: {} for m in columns}
    for h in horizons:
        sub = rows[rows["horizon"] == h]
        if sub.empty:
            continue
        for m, col in columns.items():
            out[m][str(h)] = {"mae": _mae(sub[y_col].to_numpy(), sub[col].to_numpy()),
                              "rmse": _rmse(sub[y_col].to_numpy(), sub[col].to_numpy()), "n": int(len(sub))}
    return out


def skill(table: dict[str, dict[str, dict[str, float]]], model: str, versus: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for h, cell in table[model].items():
        ref = min(table[v][h]["mae"] for v in versus)
        out[h] = 1.0 - cell["mae"] / ref if ref > 0 else float("nan")
    return out


def intervals(rows: pd.DataFrame, target: str, horizons) -> dict[str, dict[str, float]]:
    y = f"y_{target}"
    cov, width = {}, {}
    for h in horizons:
        sub = rows[rows["horizon"] == h]
        if sub.empty:
            continue
        inside = (sub[y] >= sub[f"lgb_{target}_q10"]) & (sub[y] <= sub[f"lgb_{target}_q90"])
        cov[str(h)] = float(inside.mean())
        width[str(h)] = float((sub[f"lgb_{target}_q90"] - sub[f"lgb_{target}_q10"]).mean())
    return {"coverage": cov, "mean_width": width, "nominal": 0.8}


def summarise_target(ev: pd.DataFrame, target: str, cfg: EvalConfig, folds: list[Fold]) -> dict[str, Any]:
    cols = model_columns(target)
    rows = valid_rows(ev, target)
    city = rows[rows["location"] == PRIMARY]
    horizons = list(cfg.horizons)
    table = per_horizon(city, target, cols, horizons)
    winter = city[pd.DatetimeIndex(city["target_ts"] + pd.Timedelta(hours=5, minutes=30)).month.isin(WINTER_MONTHS)]
    other = city.drop(winter.index)
    by_season = {
        "winter": {"n": int(len(winter)), "by_horizon": per_horizon(winter, target, cols, horizons)},
        "other": {"n": int(len(other)), "by_horizon": per_horizon(other, target, cols, horizons)},
    }
    by_fold = []
    for f in folds:
        sub = city[city["fold"] == f.index]
        if sub.empty:
            continue
        entry: dict[str, Any] = {"fold": f.index, "cutoff": f.cutoff.isoformat(), "n": int(len(sub))}
        for m, col in cols.items():
            entry[m] = _mae(sub[f"y_{target}"].to_numpy(), sub[col].to_numpy())
        by_fold.append(entry)
    by_location = {
        loc: {m: _mae(g[f"y_{target}"].to_numpy(), g[c].to_numpy()) for m, c in cols.items()}
        for loc, g in rows.groupby("location")
    }
    return {
        "by_horizon": table,
        "skill_vs_persistence": {m: skill(table, m, ["persistence"]) for m in ("seasonal_naive", "lightgbm")},
        "skill_vs_best_baseline": {"lightgbm": skill(table, "lightgbm", ["persistence", "seasonal_naive"])},
        "interval": intervals(city, target, horizons),
        "interval_winter": intervals(winter, target, horizons),
        "by_season": by_season,
        "by_fold": by_fold,
        "by_location": by_location,
        "n_rows": int(len(city)),
    }


# ----------------------------------------------------------------------------- the run


def run_evaluation(
    frames: dict[str, pd.DataFrame] | None = None,
    engine: Engine | None = None,
    cfg: EvalConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or EvalConfig()
    if frames is None:
        engine = engine or get_engine()
        frames = {loc: read_observations(engine, loc) for loc in LOCATIONS}
    start = min(f.index.min() for f in frames.values())
    end = max(f.index.max() for f in frames.values())
    folds = make_folds(start, end, cfg)
    log.info("%d folds, first cut-off %s, last %s", len(folds), folds[0].cutoff, folds[-1].cutoff)

    pool = build_rows(frames, TRAIN_HORIZONS, stride=cfg.train_stride, n_random=cfg.train_n_random, seed=cfg.seed)
    features = feature_columns(pool)

    def in_test_window(idx: pd.DatetimeIndex) -> np.ndarray:
        mask = np.zeros(len(idx), dtype=bool)
        for f in folds:
            mask |= (idx >= f.cutoff) & (idx < f.test_end)
        return mask

    ev = build_rows(frames, cfg.horizons, stride=cfg.origin_step_h, origin_filter=in_test_window)
    ev["fold"] = 0
    for f in folds:
        ev.loc[(ev["origin_ts"] >= f.cutoff) & (ev["origin_ts"] < f.test_end), "fold"] = f.index
    ev = ev[ev["fold"] > 0].copy()

    for tgt in TARGETS:
        base = baseline_predictions(ev, frames, tgt)
        ev[f"persistence_{tgt}"] = base["persistence"]
        ev[f"seasonal_naive_{tgt}"] = base["seasonal_naive"]
        for q in ("q10", "q50", "q90"):
            ev[f"lgb_{tgt}_{q}"] = np.nan
    ev["sarima_pm2_5"] = np.nan

    fold_info: list[dict[str, Any]] = []
    last_model: QuantileForecaster | None = None
    for f in folds:
        t0 = time.time()
        train = pool[pool["target_ts"] < f.cutoff]  # PURGE: targets must lie before the cut-off
        assert train["target_ts"].max() < f.cutoff
        model = QuantileForecaster().fit(train, features)
        last_model = model
        mask = ev["fold"] == f.index
        for tgt in TARGETS:
            pred = model.predict(ev.loc[mask], tgt)
            for q in pred.columns:
                ev.loc[mask, f"lgb_{tgt}_{q}"] = pred[q].to_numpy()
        n_sar = 0
        if cfg.sarima_enabled and (f.index - 1) % cfg.sarima_every == 0:
            n_sar = _run_sarima(ev, frames[PRIMARY]["pm2_5"], f, cfg)
        fold_info.append(
            {"fold": f.index, "cutoff": f.cutoff.isoformat(), "test_end": f.test_end.isoformat(),
             "train_rows": int(len(train)), "test_rows": int(mask.sum()), "sarima_origins": n_sar}
        )
        log.info("fold %2d/%d cut-off %s  train %6d rows  %.0fs", f.index, len(folds), f.cutoff.date(), len(train), time.time() - t0)

    payload: dict[str, Any] = {
        "generated_at": utcnow().isoformat(),
        "data_range": {"start": start.isoformat(), "end": end.isoformat()},
        "config": {**asdict(cfg), "horizons": list(cfg.horizons)},
        "scope": "Delhi city average (mean of the two model cells); origins every "
                 f"{cfg.origin_step_h} h inside each {cfg.test_days}-day test window.",
        "folds": fold_info,
        "targets": {tgt: summarise_target(ev, tgt, cfg, folds) for tgt in TARGETS},
        "sarima": _summarise_sarima(ev, cfg) if cfg.sarima_enabled else None,
        "feature_importance": last_model.feature_importance("pm2_5") if last_model else [],
        "procedure": [
            "Walk-forward only: each fold trains on rows whose target time is before the cut-off.",
            "Every model is scored on exactly the same rows; persistence and seasonal naive are the baselines.",
            "Skill = 1 - MAE(model) / MAE(reference). Positive means better than the reference.",
            "Intervals are 10th to 90th percentile quantile forecasts; coverage should be near 80 %.",
        ],
    }
    return payload


# ----------------------------------------------------------------------------- SARIMA


def _run_sarima(ev: pd.DataFrame, city_pm25: pd.Series, fold: Fold, cfg: EvalConfig) -> int:
    window = ev[(ev["fold"] == fold.index) & (ev["location"] == PRIMARY)]
    origins = sorted(window.loc[window["origin_ts"].dt.hour == 0, "origin_ts"].unique())
    if not origins:
        return 0
    origins = [pd.Timestamp(o) for o in origins]
    forecasts = sarima_forecast_origins(city_pm25, fold.cutoff, origins, list(cfg.horizons))
    mask = (ev["fold"] == fold.index) & (ev["location"] == PRIMARY) & ev["origin_ts"].isin(origins)
    ev.loc[mask, "sarima_pm2_5"] = [
        forecasts[pd.Timestamp(o)][int(h)] for o, h in zip(ev.loc[mask, "origin_ts"], ev.loc[mask, "horizon"], strict=True)
    ]
    return len(origins)


def _summarise_sarima(ev: pd.DataFrame, cfg: EvalConfig) -> dict[str, Any] | None:
    sub = ev[(ev["location"] == PRIMARY)].dropna(subset=["sarima_pm2_5", *model_columns("pm2_5").values(), "y_pm2_5"])
    if sub.empty:
        return None
    cols = {**model_columns("pm2_5"), "sarima": "sarima_pm2_5"}
    table = per_horizon(sub, "pm2_5", cols, list(cfg.horizons))
    return {
        "scope": "Delhi city average PM2.5, daily origins in every 2nd fold; all four models scored on the same rows.",
        "n_rows": int(len(sub)),
        "by_horizon": table,
        "skill_vs_persistence": {m: skill(table, m, ["persistence"]) for m in ("seasonal_naive", "lightgbm", "sarima")},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Walk-forward evaluation")
    parser.add_argument("--folds", type=int, default=12)
    parser.add_argument("--no-sarima", action="store_true")
    parser.add_argument("--write-seed", action="store_true", help="also write ml/registry/metrics.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    engine = init_db(get_engine())
    cfg = EvalConfig(n_folds=args.folds, sarima_enabled=not args.no_sarima)
    started = dt.datetime.now()
    payload = run_evaluation(engine=engine, cfg=cfg)
    save_evaluation(engine, payload, write_seed=args.write_seed)
    pm = payload["targets"]["pm2_5"]["by_horizon"]
    for h in ("1", "6", "24", "72"):
        print(f"h={h:>2}  " + "  ".join(f"{m}: {pm[m][h]['mae']:.1f}" for m in pm))
    print(f"done in {(dt.datetime.now() - started).seconds}s")


if __name__ == "__main__":
    main()
