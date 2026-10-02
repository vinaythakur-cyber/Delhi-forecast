"""Storage layer: schema, engine, idempotent upserts and small read helpers.

One schema serves both SQLite (default, zero setup) and PostgreSQL (Docker Compose / production).
All timestamps are stored as naive UTC. Conversion to IST happens only at the edges (API/UI).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
    func,
    select,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Engine

from pipeline.config import get_settings

POLLUTANTS = ["pm2_5", "pm10", "no2", "so2", "co", "o3"]  # all in ug/m3 (Open-Meteo units)
WEATHER = ["temp", "rh", "wind_speed", "wind_dir", "blh", "precip", "pressure"]
OBS_COLUMNS = POLLUTANTS + WEATHER

FORECAST_TARGETS = ["pm2_5", "pm10", "aqi"]
QUANTILES = ["q10", "q50", "q90"]
FORECAST_COLUMNS = [f"{t}_{q}" for t in FORECAST_TARGETS for q in QUANTILES]

metadata = MetaData()

locations = Table(
    "locations",
    metadata,
    Column("id", String(40), primary_key=True),
    Column("name", String(120), nullable=False),
    Column("lat", Float, nullable=False),
    Column("lon", Float, nullable=False),
    Column("kind", String(16), nullable=False),  # zone | city | station
    Column("area", String(120)),
    Column("source", String(40)),
)

observations = Table(
    "observations",
    metadata,
    Column("location_id", String(40), primary_key=True),
    Column("ts", DateTime, primary_key=True),  # naive UTC, hour start
    *[Column(c, Float) for c in OBS_COLUMNS],
)

forecasts = Table(
    "forecasts",
    metadata,
    Column("location_id", String(40), primary_key=True),
    Column("issued_at", DateTime, primary_key=True),  # naive UTC: last observed hour
    Column("target_ts", DateTime, primary_key=True),
    Column("horizon", Integer, nullable=False),
    *[Column(c, Float) for c in FORECAST_COLUMNS],
)

ingest_log = Table(
    "ingest_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source", String(40), nullable=False),
    Column("started_at", DateTime, nullable=False),
    Column("finished_at", DateTime),
    Column("status", String(16), nullable=False),  # ok | error
    Column("rows_written", Integer, default=0),
    Column("data_through", DateTime),
    Column("message", Text),
)

meta = Table(
    "meta",
    metadata,
    Column("key", String(64), primary_key=True),
    Column("value", Text),
    Column("updated_at", DateTime),
)

model_artifacts = Table(
    "model_artifacts",
    metadata,
    Column("name", String(64), primary_key=True),
    Column("created_at", DateTime, nullable=False),
    Column("blob", LargeBinary, nullable=False),
    Column("meta_json", Text),
)

eval_results = Table(
    "eval_results",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime, nullable=False),
    Column("payload", Text, nullable=False),
)


def utcnow() -> dt.datetime:
    """Naive UTC 'now' (the convention used for every stored timestamp)."""
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    url = url or get_settings().database_url
    kwargs: dict[str, Any] = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        path = url.replace("sqlite:///", "", 1)
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        kwargs["pool_pre_ping"] = True
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _pragmas(dbapi_conn, _record):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()

    return engine


def init_db(engine: Engine | None = None) -> Engine:
    engine = engine or get_engine()
    metadata.create_all(engine)
    return engine


# ----------------------------------------------------------------------------- writes


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _clean(value.item())
        except (ValueError, AttributeError):
            return value
    return value


def df_to_rows(df: pd.DataFrame, **const: Any) -> list[dict[str, Any]]:
    """Turn a DataFrame (index = ts) into insertable dicts, NaN -> NULL."""
    rows: list[dict[str, Any]] = []
    cols = list(df.columns)
    for ts, values in zip(df.index, df.to_numpy(dtype=object), strict=True):
        row: dict[str, Any] = {"ts": _clean(pd.Timestamp(ts))}
        row.update({c: _clean(v) for c, v in zip(cols, values, strict=True)})
        row.update(const)
        rows.append(row)
    return rows


def upsert(engine: Engine, table: Table, rows: list[dict[str, Any]], chunk_size: int = 800) -> int:
    """INSERT ... ON CONFLICT DO UPDATE on the primary key. Safe to re-run: that is idempotency."""
    if not rows:
        return 0
    insert_fn = sqlite.insert if engine.dialect.name == "sqlite" else postgresql.insert
    pk = [c.name for c in table.primary_key.columns]
    written = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), chunk_size):
            batch = rows[i : i + chunk_size]
            stmt = insert_fn(table).values(batch)
            update = {c.name: stmt.excluded[c.name] for c in table.columns if c.name not in pk}
            if update:
                stmt = stmt.on_conflict_do_update(index_elements=pk, set_=update)
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=pk)
            conn.execute(stmt)
            written += len(batch)
    return written


def upsert_observations(engine: Engine, location_id: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    frame = df.reindex(columns=OBS_COLUMNS)
    frame = frame.dropna(how="all")
    rows = df_to_rows(frame, location_id=location_id)
    return upsert(engine, observations, rows)


def register_locations(engine: Engine, locs, source: str = "open-meteo") -> None:
    rows = [
        {
            "id": loc.id,
            "name": loc.name,
            "lat": loc.lat,
            "lon": loc.lon,
            "kind": loc.kind,
            "area": loc.area,
            "source": source if loc.kind != "station" else "openaq",
        }
        for loc in locs
    ]
    upsert(engine, locations, rows)


def set_meta(engine: Engine, key: str, value: Any) -> None:
    upsert(engine, meta, [{"key": key, "value": json.dumps(value), "updated_at": utcnow()}])


def get_meta(engine: Engine, key: str, default: Any = None) -> Any:
    with engine.connect() as conn:
        row = conn.execute(select(meta.c.value).where(meta.c.key == key)).first()
    return json.loads(row[0]) if row and row[0] is not None else default


def log_ingest(
    engine: Engine,
    source: str,
    started_at: dt.datetime,
    status: str,
    rows_written: int = 0,
    data_through: dt.datetime | None = None,
    message: str = "",
) -> None:
    with engine.begin() as conn:
        conn.execute(
            ingest_log.insert().values(
                source=source,
                started_at=started_at,
                finished_at=utcnow(),
                status=status,
                rows_written=rows_written,
                data_through=data_through,
                message=message[:2000],
            )
        )


def save_artifact(engine: Engine, name: str, blob: bytes, meta_dict: dict[str, Any]) -> None:
    upsert(
        engine,
        model_artifacts,
        [{"name": name, "created_at": utcnow(), "blob": blob, "meta_json": json.dumps(meta_dict)}],
        chunk_size=1,
    )


def load_artifact(engine: Engine, name: str) -> tuple[bytes, dict[str, Any], dt.datetime] | None:
    with engine.connect() as conn:
        row = conn.execute(
            select(model_artifacts.c.blob, model_artifacts.c.meta_json, model_artifacts.c.created_at).where(
                model_artifacts.c.name == name
            )
        ).first()
    if row is None:
        return None
    return bytes(row[0]), json.loads(row[1] or "{}"), row[2]


def save_eval(engine: Engine, payload: dict[str, Any]) -> None:
    with engine.begin() as conn:
        conn.execute(eval_results.insert().values(created_at=utcnow(), payload=json.dumps(payload)))


def latest_eval(engine: Engine) -> dict[str, Any] | None:
    with engine.connect() as conn:
        row = conn.execute(select(eval_results.c.payload).order_by(eval_results.c.id.desc()).limit(1)).first()
    return json.loads(row[0]) if row else None


# ----------------------------------------------------------------------------- reads


def _normalise_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(pd.to_datetime(df.index)).as_unit("ns")
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx.rename("ts")
    return df


def read_observations(
    engine: Engine,
    location_id: str,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Observations for one location, indexed by naive-UTC `ts`, sorted, with float columns."""
    cols = columns or OBS_COLUMNS
    q = select(observations.c.ts, *[observations.c[c] for c in cols]).where(
        observations.c.location_id == location_id
    )
    if start is not None:
        q = q.where(observations.c.ts >= start)
    if end is not None:
        q = q.where(observations.c.ts <= end)
    df = pd.read_sql(q.order_by(observations.c.ts), engine)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    df = _normalise_index(df)
    return df.astype("float64")


def latest_ts(engine: Engine, location_id: str, column: str = "pm2_5") -> dt.datetime | None:
    with engine.connect() as conn:
        row = conn.execute(
            select(func.max(observations.c.ts)).where(
                observations.c.location_id == location_id, observations.c[column].is_not(None)
            )
        ).first()
    value = row[0] if row else None
    if value is None:
        return None
    return pd.Timestamp(value).to_pydatetime()


def count_rows(engine: Engine, table: Table) -> int:
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(table)).scalar() or 0)
