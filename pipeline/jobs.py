"""Operational jobs: the commands the launcher, Docker worker and CI call.

    python -m pipeline.jobs bootstrap [--if-empty]   backfill + train + forecast (first run)
    python -m pipeline.jobs refresh [--train]        ingest the latest hours + forecast (hourly)
    python -m pipeline.jobs train                    retrain the production model
    python -m pipeline.jobs forecast                 regenerate the forecast from the stored model
    python -m pipeline.jobs worker                   run `refresh` forever on a timer
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from typing import Any

from sqlalchemy.engine import Engine

from pipeline import ingest
from pipeline.config import get_settings
from pipeline.db import count_rows, get_engine, init_db, observations, utcnow

log = logging.getLogger("pipeline.jobs")
_lock = threading.Lock()  # one refresh at a time inside a process


def bootstrap(engine: Engine | None = None, if_empty: bool = False) -> dict[str, Any]:
    from ml.forecast import make_forecast, train_final
    from ml.registry import load_model

    engine = init_db(engine or get_engine())
    result: dict[str, Any] = {}
    with _lock:
        if count_rows(engine, observations) == 0:
            log.info("step 1/3: downloading history (about 4 years, a few minutes) ...")
            result["rows"] = ingest.backfill(engine)
        elif not if_empty:
            result["rows"] = ingest.update(engine)
        if load_model(engine) is None:
            log.info("step 2/3: training the forecasting models ...")
            result["model"] = train_final(engine)
        log.info("step 3/3: generating the 72-hour forecast ...")
        result["forecast_rows"] = make_forecast(engine)
    return result


def refresh(engine: Engine | None = None, force_train: bool = False) -> dict[str, Any]:
    from ml.forecast import train_final
    from ml.registry import model_age_days

    settings = get_settings()
    engine = init_db(engine or get_engine())
    with _lock:
        result: dict[str, Any] = {"rows": ingest.update(engine)}
        age = model_age_days(engine, utcnow())
        if force_train or age is None or age >= settings.retrain_days:
            log.info("retraining (model age: %s days)", "none" if age is None else f"{age:.1f}")
            result["model"] = train_final(engine)
        result["forecast_rows"] = _forecast(engine)
    return result


def _forecast(engine: Engine) -> int:
    from ml.forecast import make_forecast

    return make_forecast(engine)


def worker(interval_minutes: int | None = None) -> None:
    settings = get_settings()
    interval = (interval_minutes or settings.refresh_minutes) * 60
    log.info("worker started, refreshing every %d minutes", interval // 60)
    while True:
        started = time.time()
        try:
            res = refresh()
            log.info("refresh ok: %s", {k: v for k, v in res.items() if k != "model"})
        except Exception:  # keep the worker alive; the next tick retries
            log.exception("refresh failed")
        time.sleep(max(60, interval - (time.time() - started)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Delhi AQI jobs")
    parser.add_argument("command", choices=["bootstrap", "refresh", "train", "forecast", "worker"])
    parser.add_argument("--if-empty", action="store_true", help="bootstrap: skip the update when data already exists")
    parser.add_argument("--train", action="store_true", help="refresh: force retraining")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ingest.quiet_http_logs()
    if args.command == "bootstrap":
        print(bootstrap(if_empty=args.if_empty))
    elif args.command == "refresh":
        print(refresh(force_train=args.train))
    elif args.command == "train":
        from ml.forecast import train_final

        print(train_final(init_db(get_engine())))
    elif args.command == "forecast":
        from ml.forecast import make_forecast

        print(make_forecast(init_db(get_engine())))
    else:
        worker()


if __name__ == "__main__":
    main()
