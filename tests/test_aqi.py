"""NAQI tests against hand-computed values from the CPCB 2014 breakpoint table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.aqi import (
    BANDS,
    CATEGORIES,
    category_for,
    compute_aqi,
    latest_aqi_summary,
    sub_index,
    to_cpcb_units,
)

# (pollutant, concentration in CPCB units, expected sub-index): band edges are exact by definition
EDGES = [
    ("pm2_5", 0, 0), ("pm2_5", 30, 50), ("pm2_5", 60, 100), ("pm2_5", 90, 200), ("pm2_5", 120, 300), ("pm2_5", 250, 400),
    ("pm10", 50, 50), ("pm10", 100, 100), ("pm10", 250, 200), ("pm10", 350, 300), ("pm10", 430, 400),
    ("no2", 40, 50), ("no2", 80, 100), ("no2", 180, 200), ("no2", 280, 300), ("no2", 400, 400),
    ("so2", 40, 50), ("so2", 80, 100), ("so2", 380, 200), ("so2", 800, 300), ("so2", 1600, 400),
    ("o3", 50, 50), ("o3", 100, 100), ("o3", 168, 200), ("o3", 208, 300), ("o3", 748, 400),
    ("co", 1.0, 50), ("co", 2.0, 100), ("co", 10, 200), ("co", 17, 300), ("co", 34, 400),
]


@pytest.mark.parametrize(("pollutant", "conc", "expected"), EDGES)
def test_band_edges_match_the_cpcb_table(pollutant, conc, expected):
    assert sub_index(pollutant, conc) == pytest.approx(expected)


def test_interpolation_inside_a_band_matches_hand_calculation():
    # PM2.5 = 45 is in the 31-60 -> 51-100 band: (100-51)/(60-31) * (45-31) + 51 = 74.655
    assert sub_index("pm2_5", 45) == pytest.approx(74.6552, abs=1e-3)
    # PM10 = 75 in 51-100 -> 51-100: (49/49) * (75-51) + 51 = 75
    assert sub_index("pm10", 75) == pytest.approx(75.0)
    # CO = 1.5 mg/m3 in 1.1-2.0 -> 51-100: (49/0.9) * 0.4 + 51 = 72.78
    assert sub_index("co", 1.5) == pytest.approx(72.7778, abs=1e-3)
    # NO2 = 200 in 181-280 -> 201-300: (99/99) * (200-181) + 201 = 220
    assert sub_index("no2", 200) == pytest.approx(220.0)


def test_open_top_band_is_closed_with_the_documented_slope_and_capped():
    assert BANDS["pm2_5"].c_hi[-1] == pytest.approx(380, abs=0.5)
    assert BANDS["pm10"].c_hi[-1] == pytest.approx(510, abs=0.5)
    assert sub_index("pm2_5", 380) == pytest.approx(500, abs=0.5)
    assert sub_index("pm2_5", 5000) == 500  # never above 500
    assert sub_index("pm10", 1e6) == 500


def test_gaps_between_bands_do_not_break_or_jump():
    # CPCB lists PM2.5 bands 0-30 and 31-60; 30.5 falls in the gap and must land between 50 and 51
    assert 49.9 < sub_index("pm2_5", 30.5) < 51.1


def test_nan_and_negative_are_nan():
    out = sub_index("pm2_5", np.array([np.nan, -1.0, 10.0]))
    assert np.isnan(out[0]) and np.isnan(out[1]) and out[2] == pytest.approx(16.6667, abs=1e-3)


@pytest.mark.parametrize("pollutant", list(BANDS))
def test_sub_index_is_monotonic_non_decreasing(pollutant):
    conc = np.linspace(0, BANDS[pollutant].c_hi[-1] * 1.2, 5000)
    values = sub_index(pollutant, conc)
    assert (np.diff(values) >= -1e-9).all()


@pytest.mark.parametrize(
    ("aqi", "name"),
    [(0, "Good"), (50, "Good"), (51, "Satisfactory"), (100, "Satisfactory"), (101, "Moderate"), (200, "Moderate"),
     (201, "Poor"), (300, "Poor"), (301, "Very Poor"), (400, "Very Poor"), (401, "Severe"), (500, "Severe")],
)
def test_category_boundaries_match_the_spec(aqi, name):
    assert category_for(aqi).name == name


def test_category_table_is_contiguous():
    assert [c.low for c in CATEGORIES] == [0, 51, 101, 201, 301, 401]
    assert [c.high for c in CATEGORIES] == [50, 100, 200, 300, 400, 500]
    assert category_for(float("nan")) is None


def test_co_is_converted_from_ug_to_mg():
    assert to_cpcb_units("co", 1500.0) == 1.5
    assert to_cpcb_units("pm2_5", 45.0) == 45.0


def _flat(hours=48, **values):
    idx = pd.date_range("2026-01-01", periods=hours, freq="h").as_unit("ns")
    return pd.DataFrame({k: np.full(hours, float(v)) for k, v in values.items()}, index=idx)


def test_aqi_is_the_maximum_sub_index_and_names_the_dominant_pollutant():
    df = _flat(pm2_5=45, pm10=100, no2=20, so2=10, co=500, o3=30)
    res = compute_aqi(df)
    last = res.iloc[-1]
    assert last["si_pm2_5"] == pytest.approx(74.655, abs=1e-2)
    assert last["si_pm10"] == pytest.approx(100.0)
    assert last["aqi"] == 100 and last["dominant"] == "pm10" and last["category"] == "Satisfactory"


def test_needs_16_valid_hours_in_the_24_hour_window():
    df = _flat(hours=30, pm2_5=100, pm10=150, no2=30)
    assert res_first_valid(compute_aqi(df)) == 15  # 16th hour is the first with 16 valid hours
    sparse = df.copy()
    sparse.iloc[5:15] = np.nan  # 10 missing hours inside the window ending at hour 29
    assert np.isnan(compute_aqi(sparse).iloc[-1]["aqi"])


def res_first_valid(res) -> int:
    return int(np.argmax(res["aqi"].notna().to_numpy()))


def test_needs_three_pollutants_and_one_must_be_particulate():
    only_two = compute_aqi(_flat(pm2_5=100, pm10=150))
    assert only_two["aqi"].isna().all()
    no_pm = compute_aqi(_flat(no2=100, so2=10, o3=60))
    assert no_pm["aqi"].isna().all()
    ok = compute_aqi(_flat(pm2_5=100, no2=30, so2=10))
    assert ok["aqi"].notna().iloc[-1]


def test_8_hour_pollutants_use_the_running_max_not_the_latest_value():
    df = _flat(hours=60, pm2_5=20, pm10=40, no2=20, o3=10)
    df.iloc[10:18, df.columns.get_loc("o3")] = 250.0  # an 8-hour smog episode
    res = compute_aqi(df)
    assert res["si_o3"].iloc[20] > 200  # the episode dominates the next 24h even after it ended
    assert res["si_o3"].iloc[50] < 20  # then it leaves the 24-hour window


def test_latest_summary_is_json_ready_and_consistent():
    df = _flat(hours=72, pm2_5=150, pm10=200, no2=60, so2=20, co=1200, o3=40)
    s = latest_aqi_summary(df)
    assert s["category"] == category_for(s["aqi"]).name
    assert s["dominant"] == "pm2_5" and s["aqi"] == int(round(sub_index("pm2_5", 150)))
    assert set(s["sub_indices"]) == {"pm2_5", "pm10", "no2", "so2", "co", "o3"}
    assert latest_aqi_summary(df.iloc[:5]) is None


def test_basis_decides_which_pollutants_can_set_the_aqi():
    from pipeline.aqi import BASES

    df = _flat(pm2_5=45, pm10=100, no2=20, so2=10, co=500, o3=250)  # ozone sub-index is huge
    full = compute_aqi(df, BASES["all"]).iloc[-1]
    pm25 = compute_aqi(df, BASES["pm2_5"]).iloc[-1]
    pm = compute_aqi(df, BASES["pm"]).iloc[-1]
    assert full["dominant"] == "o3" and full["aqi"] > 200
    assert pm25["aqi"] == 75 and pm25["dominant"] == "pm2_5"
    assert pm["aqi"] == 100 and pm["dominant"] == "pm10"
    assert pm25["si_o3"] == full["si_o3"]  # excluded pollutants are still reported


def test_single_pollutant_basis_needs_no_other_data():
    only_pm25 = compute_aqi(_flat(pm2_5=100), ["pm2_5"])
    assert only_pm25["aqi"].notna().iloc[-1]
    assert compute_aqi(_flat(pm10=100), ["pm2_5"])["aqi"].isna().all()
