"""FastAPI dependencies. Tests override `get_db` to point at a temporary database."""

from __future__ import annotations

from sqlalchemy.engine import Engine

from pipeline.db import get_engine, init_db


def get_db() -> Engine:
    return init_db(get_engine())
