"""In-process scheduler for single-container runs (the launcher and plain `uvicorn`).

On start it bootstraps if the database or model is missing, then refreshes on a timer. Docker
Compose turns this off (ENABLE_SCHEDULER=false) and runs `python -m pipeline.jobs worker` as its own
service instead, so the web process never does heavy work.
"""

from __future__ import annotations

import logging
import threading

from pipeline import jobs
from pipeline.config import get_settings

log = logging.getLogger("backend.scheduler")
RETRY_SECONDS = 120  # wait between attempts at first-time setup


class Scheduler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="refresh-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        interval = get_settings().refresh_minutes * 60
        while True:  # first-time setup: retry every 2 minutes until it succeeds (network down, rate limit, ...)
            try:
                jobs.bootstrap(if_empty=True)
                break
            except Exception:
                log.exception("first-time setup failed; retrying in %d s", RETRY_SECONDS)
                if self._stop.wait(RETRY_SECONDS):
                    return
        while not self._stop.wait(interval):
            try:
                jobs.refresh()
            except Exception:
                log.exception("scheduled refresh failed; will retry on the next tick")


scheduler = Scheduler()
