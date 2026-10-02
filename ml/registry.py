"""Model registry: where trained models live.

Models are stored in the database (table `model_artifacts`) so the API, the scheduler and the
Docker worker all see the same model regardless of which container trained it. A small
`ml/registry/forecaster.meta.json` is also written for humans; model binaries are never written to
the repository folder. `ml/registry/metrics.json` is the one committed file: the shipped
evaluation, used to seed the model-performance page on first start.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

from sqlalchemy.engine import Engine

from ml.train import QuantileForecaster
from pipeline.config import REGISTRY_DIR
from pipeline.db import latest_eval, load_artifact, save_artifact, save_eval

MODEL_NAME = "forecaster"
SEED_METRICS = REGISTRY_DIR / "metrics.json"


def save_model(engine: Engine, model: QuantileForecaster, meta: dict[str, Any]) -> None:
    model.meta = meta
    save_artifact(engine, MODEL_NAME, model.to_bytes(), meta)
    try:
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        (REGISTRY_DIR / "forecaster.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    except OSError:  # read-only filesystem (some hosts): the database copy is what matters
        pass


def load_model(engine: Engine) -> tuple[QuantileForecaster, dict[str, Any], dt.datetime] | None:
    found = load_artifact(engine, MODEL_NAME)
    if found is None:
        return None
    blob, meta, created = found
    model = QuantileForecaster.from_bytes(blob)
    return model, meta, created


def model_age_days(engine: Engine, now: dt.datetime) -> float | None:
    found = load_artifact(engine, MODEL_NAME)
    return None if found is None else (now - found[2]).total_seconds() / 86400


def save_evaluation(engine: Engine, payload: dict[str, Any], write_seed: bool = False) -> None:
    save_eval(engine, payload)
    if write_seed:
        SEED_METRICS.write_text(json.dumps(payload, indent=1, default=str))


def get_evaluation(engine: Engine) -> dict[str, Any] | None:
    """Latest evaluation from the database, falling back to the shipped seed file."""
    found = latest_eval(engine)
    if found is not None:
        return found
    if SEED_METRICS.exists():
        payload = json.loads(SEED_METRICS.read_text())
        payload["source"] = "shipped"
        return payload
    return None
