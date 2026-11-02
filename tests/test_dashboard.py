"""The seasonal dashboard, on made-up data."""
import numpy as np
import pandas as pd

import dashboard


def setting(stations=45, days=60):
    """A network of stations close enough to be neighbours, with a smoke event over all of them on July 1 and 2."""
    names = [f"s{i}" for i in range(stations)]
    st = pd.DataFrame({"LAT": 49 + np.arange(stations) * .01, "LONG": -120.0, "REGION": ["A" if i % 2 else "B" for i in range(stations)]}, index=names)
    doy = range(1, 367)
    base = pd.DataFrame({n: 5.0 for n in names}, index=doy)
    offsets = pd.DataFrame({"season": ["warm", "cold"], "instrument": ["T640", "T640"], "offset": [0.0, 0.0]})
    rows = []
    for d in pd.date_range("2026-05-01", periods=days):
        for n in names:
            rows.append({"STATION_NAME": n, "EMS_ID": "E", "INSTRUMENT": "PM25_T640", "DATE": d.date().isoformat(), "hi": 9.0, "n": 24,
                         "mean": 40.0 if d in (pd.Timestamp("2026-06-05"), pd.Timestamp("2026-06-06")) else 5.0})
    raw = pd.DataFrame(rows)
    history = pd.DataFrame([{"year": y, "day": d, "region": r, "smoke_share": .1 if y == 2023 else 0.0} for y in (2022, 2023) for d in range(184) for r in "AB"])
    return raw, st, base, offsets, history


def test_panel_flags_the_event():
    raw, st, base, offsets, _ = setting()
    p, year = dashboard.season_panel(raw, st, base, offsets, pd.Series(dtype=str))
    assert year == 2026
    assert p[p.DATE == "2026-06-05"].smoke.all()
    assert p.smoke.sum() == 2 * 45


def test_sensor_faults_are_dropped_and_counted():
    raw, st, base, offsets, _ = setting()
    raw.loc[0, "mean"] = -582.0
    p, _ = dashboard.season_panel(raw, st, base, offsets, pd.Series(dtype=str))
    assert p.pm.min() > dashboard.MIN_DAILY
    assert 0 < dashboard.faulty_share(raw) < dashboard.MAX_FAULTY


def test_summary_compares_with_past_years_to_the_same_date():
    raw, st, base, offsets, history = setting()
    p, year = dashboard.season_panel(raw, st, base, offsets, pd.Series(dtype=str))
    s = dashboard.summary(p, dashboard.daily_share(p), history, year, "2026-06-29")
    assert s["smoke_days"] == 2
    assert s["data_through"] == "2026-06-29"
    assert s["years_compared"] == 2
    assert s["rank"] == 2  # 2023 is ahead with 0.1 x 60 = 6 days
    assert s["regions"]["A"]["highest_past_year"] == 2023
    assert not s["season_over"]


def test_page_has_no_missing_values():
    raw, st, base, offsets, history = setting()
    p, year = dashboard.season_panel(raw, st, base, offsets, pd.Series(dtype=str))
    html = dashboard.page(dashboard.summary(p, dashboard.daily_share(p), history, year, "2026-06-29"), "2026-06-30")
    assert "nan" not in html.lower().replace("finance", "")
    assert "unverified" in html


def test_problems_stop_a_stale_or_thin_file():
    raw, st, base, offsets, _ = setting()
    p, year = dashboard.season_panel(raw, st, base, offsets, pd.Series(dtype=str))
    assert dashboard.problems(p, year, "2026-06-30") == []
    assert "days old" in dashboard.problems(p, year, "2026-08-01")[0]
    assert "older than the published" in dashboard.problems(p, year, "2026-06-30", "2026-09-01")[0]
    assert "implausible" in dashboard.problems(p, year, "2026-06-30", faulty=.05)[0]
    thin, _ = dashboard.season_panel(raw[raw.STATION_NAME.isin(["s1", "s2"])], st, base, offsets, pd.Series(dtype=str))
    assert "stations reporting" in dashboard.problems(thin, year, "2026-06-30")[0]
