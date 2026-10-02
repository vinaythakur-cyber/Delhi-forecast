"""FastAPI application: JSON API under /api, and the built frontend at /."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.app.routes import api_router
from backend.app.scheduler import scheduler
from backend.app.service import NoData, NotFound
from pipeline.config import get_settings
from pipeline.db import get_engine, init_db

STATIC_DIR = Path(__file__).resolve().parent / "static"
VERSION = "1.0.0"
DESCRIPTION = (
    "Delhi air quality: current AQI, history, a 72-hour probabilistic forecast and an honest "
    "walk-forward evaluation. Data: CAMS model via Open-Meteo (CC BY 4.0)."
)


def create_app(start_scheduler: bool | None = None) -> FastAPI:
    settings = get_settings()
    run_scheduler = settings.enable_scheduler if start_scheduler is None else start_scheduler

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        init_db(get_engine())
        if run_scheduler:
            scheduler.start()
        yield
        scheduler.stop()

    app = FastAPI(title="Delhi AQI", version=VERSION, description=DESCRIPTION, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_list, allow_methods=["GET"], allow_headers=["*"]
    )

    @app.exception_handler(NotFound)
    async def _not_found(_: Request, exc: NotFound):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(NoData)
    async def _no_data(_: Request, exc: NoData):
        return JSONResponse({"detail": str(exc), "status": "initializing"}, status_code=503)

    @app.exception_handler(ValueError)
    async def _bad_value(_: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    app.include_router(api_router)

    if (STATIC_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
    else:

        @app.get("/", include_in_schema=False)
        def root():
            return {"message": "Delhi AQI API is running. The frontend is not built; see /docs for the API."}

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
app = create_app()
