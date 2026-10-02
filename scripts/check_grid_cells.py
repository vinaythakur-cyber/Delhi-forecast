"""Reproduce the finding that Delhi has only a couple of independent Open-Meteo model cells.

    python scripts/check_grid_cells.py

For each neighbourhood it downloads 2 days of PM2.5 and groups places whose series are
identical. Places in the same group receive the same number: there is no extra spatial detail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import get_settings  # noqa: E402
from pipeline.locations import NEIGHBOURHOODS  # noqa: E402


def main() -> None:
    settings = get_settings()
    groups: dict[tuple, list[str]] = {}
    with httpx.Client(timeout=60) as client:
        for place in NEIGHBOURHOODS:
            params = {
                "latitude": place.lat,
                "longitude": place.lon,
                "hourly": "pm2_5",
                "start_date": "2024-11-01",
                "end_date": "2024-11-02",
                "timezone": "UTC",
            }
            data = client.get(settings.aq_api_url, params=params).json()
            groups.setdefault(tuple(data["hourly"]["pm2_5"]), []).append(f"{place.name} [{place.cell}]")
    print(f"{len(NEIGHBOURHOODS)} neighbourhoods -> {len(groups)} distinct series\n")
    for i, members in enumerate(groups.values(), 1):
        print(f"series {i}: " + ", ".join(members))


if __name__ == "__main__":
    main()
