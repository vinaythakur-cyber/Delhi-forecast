"""India National Air Quality Index (CPCB, 2014).

How the index is built
----------------------
1. Each pollutant's concentration is averaged over its own window:
     PM2.5, PM10, NO2, SO2 -> 24-hour average
     CO, O3                -> maximum 8-hour running average inside the last 24 hours
2. Each average is turned into a *sub-index* by linear interpolation inside a breakpoint band:
       I = (I_hi - I_lo) / (B_hi - B_lo) * (C - B_lo) + I_lo
3. The AQI is the MAXIMUM sub-index; that pollutant is the "dominant" one.
4. CPCB data-sufficiency rules: at least 16 valid hours in the 24-hour window, at least three
   pollutants with a sub-index, and one of them must be PM2.5 or PM10.

Which pollutants count (the `basis`)
-------------------------------------
`compute_aqi` takes a `basis`. The default, ALL, is the official six-pollutant NAQI. The app
overrides it with BASES["pm2_5"] by default, for a reason found in the real data (see
notebooks/01_eda.ipynb): the free CAMS model series has (a) an ozone level that makes O3 the
"dominant" pollutant in over half of all hours, and (b) a step change in PM10 from March 2025
(monthly means of 500-700 ug/m3 in spring, versus 80-260 before). Both look like model artefacts,
so counting them would show "Severe" air for weeks every spring. Every pollutant still gets a
sub-index and is reported; `basis` only decides which ones can set the headline AQI.

Units: Open-Meteo reports every gas in ug/m3. CPCB defines CO in mg/m3, so CO is divided by 1000.

Assumption (documented, not from the CPCB table): CPCB gives the top band ("Severe") as open
ended, "251+". To keep the formula continuous we close that band using the slope of the band
below it, which reproduces the commonly quoted 380 for PM2.5 and 510 for PM10. The index is
capped at 500.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

AQI_MAX = 500
MIN_HOURS_24H = 16  # CPCB: at least 16 valid hours in a 24-hour average
MIN_HOURS_8H = 6  # at least 6 of 8 hours in an 8-hour average
MIN_SUBINDICES = 3

# (conc_low, conc_high, index_low, index_high); the last band is open ended.
_RAW_BANDS: dict[str, list[tuple[float, float | None, int, int]]] = {
    "pm2_5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (251, None, 401, 500)],
    "pm10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (431, None, 401, 500)],
    "no2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400), (401, None, 401, 500)],
    "so2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400), (1601, None, 401, 500)],
    "o3": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400), (749, None, 401, 500)],
    "co": [(0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10, 101, 200), (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, None, 401, 500)],
}

BASES: dict[str, list[str]] = {
    "pm2_5": ["pm2_5"],
    "pm": ["pm2_5", "pm10"],
    "all": ["pm2_5", "pm10", "no2", "so2", "co", "o3"],
}

AVERAGING = {"pm2_5": "24h", "pm10": "24h", "no2": "24h", "so2": "24h", "o3": "8h", "co": "8h"}
POLLUTANT_ORDER = ["pm2_5", "pm10", "no2", "so2", "co", "o3"]
POLLUTANT_LABELS = {"pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO2", "so2": "SO2", "co": "CO", "o3": "O3"}


@dataclass(frozen=True)
class Band:
    c_lo: np.ndarray
    c_hi: np.ndarray
    i_lo: np.ndarray
    i_hi: np.ndarray


def _build_bands() -> dict[str, Band]:
    out: dict[str, Band] = {}
    for pol, rows in _RAW_BANDS.items():
        c_lo = np.array([r[0] for r in rows], dtype=float)
        i_lo = np.array([r[2] for r in rows], dtype=float)
        i_hi = np.array([r[3] for r in rows], dtype=float)
        c_hi = np.array([r[1] if r[1] is not None else np.nan for r in rows], dtype=float)
        # close the open top band with the slope of the band below it
        prev_slope = (i_hi[-2] - i_lo[-2]) / (c_hi[-2] - c_lo[-2])
        c_hi[-1] = c_lo[-1] + (i_hi[-1] - i_lo[-1]) / prev_slope
        out[pol] = Band(c_lo, c_hi, i_lo, i_hi)
    return out


BANDS = _build_bands()


def to_cpcb_units(pollutant: str, value):
    """Open-Meteo ug/m3 -> CPCB units (CO in mg/m3, everything else unchanged)."""
    return value / 1000.0 if pollutant == "co" else value


def _knots(band: Band) -> tuple[np.ndarray, np.ndarray]:
    """(concentration, index) corner points: inside a band the CPCB formula is the straight line
    between its two corners; across the small gaps CPCB leaves between bands (30 -> 31 for PM2.5)
    we join the corners with a straight line so the index never jumps or dips."""
    xs: list[float] = []
    ys: list[float] = []
    for c_lo, c_hi, i_lo, i_hi in zip(band.c_lo, band.c_hi, band.i_lo, band.i_hi, strict=True):
        xs += [c_lo, c_hi]
        ys += [i_lo, i_hi]
    return np.array(xs), np.array(ys)


KNOTS = {pol: _knots(b) for pol, b in BANDS.items()}


def sub_index(pollutant: str, conc) -> np.ndarray | float:
    """Sub-index for an already-averaged concentration in CPCB units. NaN stays NaN; capped at 500."""
    xs, ys = KNOTS[pollutant]
    c = np.asarray(conc, dtype=float)
    scalar = c.ndim == 0
    c = np.atleast_1d(c)
    result = np.interp(c, xs, ys)  # clamps to 0 below and 500 above
    result = np.where(np.isnan(c) | (c < 0), np.nan, result)
    return float(result[0]) if scalar else result


# ----------------------------------------------------------------------------- categories


@dataclass(frozen=True)
class Category:
    name: str
    low: int
    high: int
    color: str
    advice: str


CATEGORIES = [
    Category("Good", 0, 50, "#2e9e4f", "Minimal impact."),
    Category("Satisfactory", 51, 100, "#8cc63f", "Minor breathing discomfort to sensitive people."),
    Category("Moderate", 101, 200, "#f2c500", "Breathing discomfort for people with lung disease, heart disease, children and older adults."),
    Category("Poor", 201, 300, "#f28c28", "Breathing discomfort for most people on prolonged exposure."),
    Category("Very Poor", 301, 400, "#e03c31", "Respiratory illness on prolonged exposure. Limit time outdoors."),
    Category("Severe", 401, 500, "#8b1a1a", "Affects healthy people and seriously affects those with existing disease. Avoid outdoor activity."),
]


def category_for(aqi: float | None) -> Category | None:
    if aqi is None or (isinstance(aqi, float) and np.isnan(aqi)):
        return None
    value = int(round(float(aqi)))
    for cat in CATEGORIES:
        if value <= cat.high:
            return cat
    return CATEGORIES[-1]


# ----------------------------------------------------------------------------- time series


def averaged_concentrations(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pollutant averaging windows in CPCB units, evaluated at every hour (trailing window)."""
    hourly = df.asfreq("h")
    out = pd.DataFrame(index=hourly.index)
    for pol in POLLUTANT_ORDER:
        if pol not in hourly:
            continue
        series = to_cpcb_units(pol, hourly[pol])
        if AVERAGING[pol] == "24h":
            out[pol] = series.rolling(24, min_periods=MIN_HOURS_24H).mean()
        else:
            avg8 = series.rolling(8, min_periods=MIN_HOURS_8H).mean()
            out[pol] = avg8.rolling(24, min_periods=MIN_HOURS_24H).max()
    return out


def compute_aqi(df: pd.DataFrame, basis: list[str] | None = None) -> pd.DataFrame:
    """Hourly AQI from hourly concentrations (ug/m3).

    Returns a sub-index column for every pollutant present, plus `aqi`, `dominant` and `category`
    computed from the pollutants in `basis` (default: all six, i.e. the official NAQI).
    The CPCB sufficiency rule (>= 3 sub-indices, one of them particulate) applies when the basis
    has three or more pollutants; a smaller basis needs all of its pollutants.
    """
    basis = list(basis or BASES["all"])
    avg = averaged_concentrations(df)
    result = pd.DataFrame(index=avg.index)
    for pol in avg.columns:
        result[f"si_{pol}"] = sub_index(pol, avg[pol].to_numpy())
    counted = [f"si_{p}" for p in basis if f"si_{p}" in result.columns]
    si = result[counted] if counted else pd.DataFrame(index=avg.index)
    n_valid = si.notna().sum(axis=1)
    if len(basis) >= MIN_SUBINDICES:
        has_pm = si.reindex(columns=["si_pm2_5", "si_pm10"]).notna().any(axis=1)
        ok = (n_valid >= MIN_SUBINDICES) & has_pm
    else:
        ok = n_valid >= len(basis)
    filled = si.fillna(-1.0)
    best = filled.max(axis=1) if counted else pd.Series(np.nan, index=avg.index)
    dominant = filled.idxmax(axis=1).str.removeprefix("si_") if counted else pd.Series(None, index=avg.index)
    result["aqi"] = np.where(ok, np.round(best), np.nan)
    result["dominant"] = dominant.where(ok)
    result["category"] = [c.name if (c := category_for(a)) else None for a in result["aqi"]]
    return result


def latest_aqi_summary(df: pd.DataFrame, basis: list[str] | None = None) -> dict | None:
    """The most recent hour with a valid AQI, as a plain dict for the API."""
    basis = list(basis or BASES["all"])
    res = compute_aqi(df, basis)
    valid = res.dropna(subset=["aqi"])
    if valid.empty:
        return None
    row = valid.iloc[-1]
    cat = category_for(row["aqi"])
    avg = averaged_concentrations(df).loc[valid.index[-1]]
    return {
        "ts": valid.index[-1],
        "aqi": int(row["aqi"]),
        "category": cat.name,
        "color": cat.color,
        "advice": cat.advice,
        "dominant": row["dominant"],
        "dominant_label": POLLUTANT_LABELS.get(row["dominant"], row["dominant"]),
        "basis": basis,
        "sub_indices": {
            p: (None if np.isnan(row[f"si_{p}"]) else int(round(row[f"si_{p}"])))
            for p in POLLUTANT_ORDER
            if f"si_{p}" in row
        },
        "averages": {p: (None if np.isnan(avg[p]) else round(float(avg[p]), 2)) for p in avg.index},
    }
