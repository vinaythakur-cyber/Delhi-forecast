#!/usr/bin/env python3
"""One-command launcher:   python run.py

What it does, in order (and skips whatever is already done):
  1. creates a virtual environment in .venv and installs requirements.txt
  2. first run only: downloads ~4 years of hourly data, trains the models, makes the first forecast
  3. starts the server and opens http://127.0.0.1:8000 in your browser

Needs only Python 3.11+ and an internet connection. Nothing else (no Node, no Docker, no database).
Re-running later starts instantly and refreshes the data every hour while it runs.
Options:  --no-browser   --port 8080   --reinstall   --skip-bootstrap
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
MARKER = VENV / ".deps-installed"
MIN_PY = (3, 11)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def in_project_venv() -> bool:
    return Path(sys.prefix).resolve() == VENV.resolve()


def step(msg: str) -> None:
    print(f"\n\033[1m==> {msg}\033[0m", flush=True)


def ensure_environment(reinstall: bool) -> None:
    if sys.version_info < MIN_PY:
        sys.exit(f"Python {MIN_PY[0]}.{MIN_PY[1]} or newer is required (you have {sys.version.split()[0]}). Install it from https://www.python.org/downloads/")
    if not venv_python().exists():
        step("Creating a virtual environment in .venv (one time)")
        venv.create(VENV, with_pip=True)
    needs_install = reinstall or not MARKER.exists() or MARKER.stat().st_mtime < (ROOT / "requirements.txt").stat().st_mtime
    if needs_install:
        step("Installing dependencies (one time, 1 to 3 minutes)")
        subprocess.check_call([str(venv_python()), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(ROOT / "requirements.txt")])
        MARKER.write_text("ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Delhi AQI launcher")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reinstall", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()

    if not in_project_venv():
        ensure_environment(args.reinstall)
        os.execv(str(venv_python()), [str(venv_python()), str(Path(__file__).resolve()), *sys.argv[1:]])  # re-run inside the venv

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from pipeline.config import get_settings

    settings = get_settings()
    port = args.port or settings.port

    if not args.skip_bootstrap:
        step("Preparing data and models (the first run downloads history and trains; later runs are instant)")
        from pipeline import ingest, jobs

        ingest.quiet_http_logs()
        import logging

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
        try:
            jobs.bootstrap(if_empty=True)
        except Exception as exc:  # network down, API changed, ...
            print(f"\nSetup could not finish: {exc}\nThe server will still start; it retries in the background.", flush=True)

    url = f"http://{settings.host}:{port}"
    step(f"Starting the server at {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(2.0), webbrowser.open(url)), daemon=True).start()
    import uvicorn

    uvicorn.run("backend.app.main:app", host=settings.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
