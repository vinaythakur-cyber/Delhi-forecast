"""The places the app tracks.

IMPORTANT FINDING (see scripts/check_grid_cells.py and the EDA notebook): the Open-Meteo
air-quality product for India is a ~0.4 degree CAMS model grid. Querying 24 different Delhi
neighbourhoods returns exactly identical series for all places inside the same grid cell. Delhi
therefore has only TWO independent model series: a north/central cell and a south cell.

So the forecast is made for those two cells and their average, and neighbourhoods are shown on the
map as members of a cell. Real per-sensor data needs OpenAQ (optional, `OPENAQ_API_KEY`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    id: str
    name: str
    lat: float
    lon: float
    kind: str  # "cell" (model grid cell) | "city" (mean of cells) | "station" (ground sensor)
    area: str = ""


@dataclass(frozen=True)
class Neighbourhood:
    name: str
    lat: float
    lon: float
    cell: str  # id of the model cell whose values this place receives


CITY = Location("delhi", "Delhi (average of both cells)", 28.6139, 77.2090, "city", "All of Delhi")

CELLS: list[Location] = [
    Location("delhi-north", "North & Central Delhi", 28.6289, 77.2406, "cell", "Model cell containing ITO"),
    Location("delhi-south", "South Delhi", 28.5633, 77.1869, "cell", "Model cell containing R.K. Puram"),
]

ALL_LOCATIONS: list[Location] = [CITY, *CELLS]
BY_ID: dict[str, Location] = {loc.id: loc for loc in ALL_LOCATIONS}

# Membership verified on 2026-10-02 by checking that each place returns an identical series
# (python scripts/check_grid_cells.py reproduces this).
NEIGHBOURHOODS: list[Neighbourhood] = [
    Neighbourhood("Anand Vihar", 28.6469, 77.3158, "delhi-north"),
    Neighbourhood("ITO / Central Delhi", 28.6289, 77.2406, "delhi-north"),
    Neighbourhood("Mandir Marg", 28.6360, 77.2010, "delhi-north"),
    Neighbourhood("Punjabi Bagh", 28.6683, 77.1167, "delhi-north"),
    Neighbourhood("Vivek Vihar", 28.6720, 77.3150, "delhi-north"),
    Neighbourhood("Mundka", 28.6820, 77.0310, "delhi-north"),
    Neighbourhood("Rohini", 28.7324, 77.1190, "delhi-north"),
    Neighbourhood("Jahangirpuri", 28.7330, 77.1706, "delhi-north"),
    Neighbourhood("Burari", 28.7600, 77.2000, "delhi-north"),
    Neighbourhood("Bawana", 28.7762, 77.0511, "delhi-north"),
    Neighbourhood("Alipur", 28.8000, 77.1300, "delhi-north"),
    Neighbourhood("Narela", 28.8527, 77.0929, "delhi-north"),
    Neighbourhood("Lodhi Road", 28.5918, 77.2273, "delhi-south"),
    Neighbourhood("Dwarka", 28.5823, 77.0500, "delhi-south"),
    Neighbourhood("Mathura Road", 28.5700, 77.2700, "delhi-south"),
    Neighbourhood("R.K. Puram", 28.5633, 77.1869, "delhi-south"),
    Neighbourhood("IGI Airport", 28.5562, 77.1180, "delhi-south"),
    Neighbourhood("Vasant Vihar", 28.5590, 77.1590, "delhi-south"),
    Neighbourhood("Sirifort", 28.5500, 77.2200, "delhi-south"),
    Neighbourhood("Okhla", 28.5308, 77.2710, "delhi-south"),
    Neighbourhood("Aya Nagar", 28.4700, 77.1300, "delhi-south"),
]
