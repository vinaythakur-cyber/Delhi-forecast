"""Central configuration. Every setting can be overridden by an environment variable or `.env`.

Nothing else in the code base reads os.environ directly, so this file is the one place that
documents what is configurable.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REGISTRY_DIR = ROOT / "ml" / "registry"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # Storage
    database_url: str = f"sqlite:///{(DATA_DIR / 'delhi_aqi.db').as_posix()}"

    # Optional OpenAQ ground stations
    openaq_api_key: str = ""
    openaq_base_url: str = "https://api.openaq.org/v3"

    # Open-Meteo endpoints (no key needed)
    aq_api_url: str = "https://air-quality-api.open-meteo.com/v1/air-quality"
    weather_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    weather_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    history_start: str = "2022-08-04"
    http_timeout_s: float = 90.0
    request_pause_s: float = 0.3
    rate_limit_wait_s: float = 65.0

    # Scheduler
    enable_scheduler: bool = True
    refresh_minutes: int = 60
    retrain_days: int = 7

    # Web server
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = "http://localhost:3000"

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
