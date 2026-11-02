"""The smoke rule and the panel, on made-up data."""
import warnings

import numpy as np
import pandas as pd

import smoke


def small_panel():
    idx = pd.date_range("2020-01-01", "2021-12-31")
    w = pd.DataFrame(5.0, index=idx, columns=list("abcd"))
    near = smoke.neighbours(np.array([49, 49.1, 49.2, 60.]), np.array([-120, -120, -120, -130.]))
    return w, near


def test_regional_summer_rise_is_smoke():
    w, near = small_panel()
    w.loc["2020-08-10", list("abc")] = 40.0
    flag, excess = smoke.detect(w, smoke.baseline(w), near)
    assert flag.loc["2020-08-10", list("abc")].all()
    assert excess.loc["2020-08-10", "a"] == 35
    assert flag.values.sum() == 3


def test_one_station_alone_is_a_local_source():
    w, near = small_panel()
    w.loc["2020-08-20", "a"] = 40.0
    flag, _ = smoke.detect(w, smoke.baseline(w), near)
    assert not flag.values.any()


def test_winter_is_left_out():
    w, near = small_panel()
    w.loc["2020-01-15", list("abc")] = 40.0
    flag, _ = smoke.detect(w, smoke.baseline(w), near)
    assert not flag.values.any()


def test_isolated_station_needs_twice_the_threshold():
    w, near = small_panel()
    w.loc["2020-08-10", "d"] = 5 + smoke.THRESHOLD + 1
    w.loc["2020-08-11", "d"] = 5 + 2 * smoke.THRESHOLD + 1
    flag, _ = smoke.detect(w, smoke.baseline(w), near)
    assert not flag.loc["2020-08-10", "d"]
    assert flag.loc["2020-08-11", "d"]


def test_higher_threshold_flags_fewer():
    w, near = small_panel()
    w.loc["2020-08-10", list("abc")] = 5 + 10.0
    base = smoke.baseline(w)
    assert smoke.detect(w, base, near, 8)[0].values.sum() == 3
    assert smoke.detect(w, base, near, 12)[0].values.sum() == 0


def test_baseline_is_a_median_and_wraps_the_new_year():
    idx = pd.date_range("2020-01-01", "2022-12-31")
    w = pd.DataFrame({"a": np.where(idx.month == 1, 2.0, 7.0)}, index=idx, dtype=float)
    w.loc["2021-07-01":"2021-07-05", "a"] = 500.0
    base = smoke.baseline(w)
    assert base.loc["2021-07-03", "a"] == 7
    assert base.loc["2021-01-02", "a"] == 2   # the window reaches back into December, and January is the majority
    assert base.loc["2020-12-25", "a"] == 7


def test_baseline_needs_enough_days():
    idx = pd.date_range("2020-01-01", "2020-12-31")
    w = pd.DataFrame({"a": np.nan}, index=idx)
    w.iloc[:5, 0] = 3.0
    assert smoke.baseline(w).a.isna().all()


def test_moved_station_joins_and_keeps_the_fuller_row():
    f = pd.DataFrame({"STATION_NAME": ["Old\xa0Site", "New Site", "New Site"], "DATE": ["2020-01-01", "2020-01-01", "2020-01-02"], "n": [20, 24, 24], "mean": [1.0, 2.0, 3.0]})
    out = smoke.join_moved(f, pd.Series({"Old Site": "Town", "New Site": "Town"}))
    assert sorted(out["mean"]) == [2.0, 3.0]
    assert set(out.st) == {"Town"}
    assert set(out.raw) == {"New Site"}


def test_join_moved_is_one_step_and_quiet_with_no_merges():
    f = pd.DataFrame({"STATION_NAME": ["A", "B", "C"], "DATE": ["2020-01-01"] * 3, "n": [24] * 3, "mean": [1.0, 2.0, 3.0]})
    out = smoke.join_moved(f, pd.Series({"A": "B", "B": "D"})).sort_values("mean")
    assert list(out.st) == ["B", "D", "C"]  # A becomes B, and doesn't go on to D
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert list(smoke.join_moved(f, pd.Series(dtype=str)).st) == ["A", "B", "C"]


def made_up_instruments():
    """Stations with a known gap between instruments, each switching in a different year, with a common year pattern."""
    rng = np.random.default_rng(0)
    rows = []
    for s in range(30):
        switch = 2007 + s % 8
        for y in range(2005, 2016):
            kind = "TEOM" if y < switch else "SHARP"
            for d in pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="2D"):
                rows.append({"st": f"s{s}", "year": y, "DATE": d, "inst": kind,
                             "mean": 6 + s * .1 + .1 * (y - 2005) - (2.0 if kind == "TEOM" else 0) + rng.normal(0, .3)})
    return pd.DataFrame(rows)


def test_offsets_recover_a_known_gap_and_harmonise_removes_it():
    f = made_up_instruments()
    o = smoke.instrument_offsets(f, reps=10)
    assert np.allclose(o.offset, -2, atol=.1)
    assert (o.low <= o.offset).all() and (o.offset <= o.high).all()
    h = smoke.harmonise(f, o)
    gap = h[h.inst == "SHARP"].pm.mean() - h[h.inst == "TEOM"].pm.mean()
    raw_gap = f[f.inst == "SHARP"]["mean"].mean() - f[f.inst == "TEOM"]["mean"].mean()
    assert abs(raw_gap) > abs(gap)


def test_year_effects_ignore_stations_opening():
    """A dirty station opening in 2021 must not lift the province's level, as a plain average would."""
    rows = []
    for y in range(2018, 2024):
        rows += [{"st": "clean1", "region": "A", "year": y, "v": 5.0}, {"st": "clean2", "region": "B", "year": y, "v": 6.0}]
        if y >= 2021:
            rows.append({"st": "dirty", "region": "B", "year": y, "v": 20.0})
    f = pd.DataFrame(rows)
    e = smoke.year_effects(f, "v")
    assert np.allclose(e, e.iloc[0])
    assert f.groupby("year").v.mean().iloc[-1] > f.groupby("year").v.mean().iloc[0] + 1


def test_year_effects_weight_regions_equally():
    rows = [{"st": s, "region": "A" if s != "b" else "B", "year": y, "v": 10.0 if s != "b" else 20.0} for s in [f"a{i}" for i in range(9)] + ["b"] for y in (2020, 2021)]
    assert np.allclose(smoke.year_effects(pd.DataFrame(rows), "v"), 15)


def test_station_years_sets_smoke_days_back():
    days = pd.date_range("2020-01-01", "2020-12-31")
    p = pd.DataFrame({"st": "a", "region": "A", "year": 2020, "DATE": days, "pm": 5.0, "base": 5.0, "excess": 0.0, "smoke": False})
    p.loc[p.DATE == "2020-08-01", ["pm", "excess", "smoke"]] = [371.0, 366.0, True]
    y = smoke.station_years(p).iloc[0]
    assert y.measured == 6
    assert y.without == 5
    assert abs(y.share - 1 / 6) < 1e-9
    assert y.smoke_days == 1


def test_station_year_needs_enough_days():
    days = pd.date_range("2020-01-01", periods=smoke.MIN_DAYS - 1)
    p = pd.DataFrame({"st": "a", "region": "A", "year": 2020, "DATE": days, "pm": 5.0, "base": 5.0, "excess": 0.0, "smoke": False})
    assert smoke.station_years(p).empty
