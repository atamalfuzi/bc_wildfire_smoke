"""Trends, sheds and episodes, on made-up data."""
import numpy as np
import pandas as pd

import questions


def test_trend_finds_a_slope():
    s = pd.Series(np.arange(20) * .5 + 3.0, index=range(2000, 2020))
    t = questions.trend(s)
    assert abs(t["slope"] - .5) < 1e-9
    assert t["kendall_p"] < .001


def test_compare_periods_counts_every_arrangement():
    s = pd.Series([1.0, 1.0, 1.0, 5.0, 5.0], index=range(2015, 2020))
    c = questions.compare_periods(s, 2018)
    assert c["difference"] == 4
    assert abs(c["p"] - 1 / 10) < 1e-9  # one of the ten ways to choose two later years is this extreme


def test_episodes_join_short_gaps_and_split_long_ones():
    days = pd.date_range("2020-07-01", "2020-07-20")
    share = pd.Series(0.0, index=days)
    share.loc[["2020-07-02", "2020-07-03", "2020-07-06", "2020-07-15"]] = .5  # gap of 2 clear days joins; gap of 8 splits
    runs = questions.episodes(share)
    assert runs == [(pd.Timestamp("2020-07-02"), pd.Timestamp("2020-07-06")), (pd.Timestamp("2020-07-15"), pd.Timestamp("2020-07-15"))]


def two_groups(n=8, days=400, seed=0):
    """Two groups of stations that share smoke within, not between, plus one station on its own."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-07-01", periods=days)
    a, b = rng.normal(size=days), rng.normal(size=days)
    data = {f"a{i}": a + rng.normal(0, .2, days) for i in range(n)}
    data.update({f"b{i}": b + rng.normal(0, .2, days) for i in range(n)})
    data["loner"] = rng.normal(size=days) + .3 * a
    return pd.DataFrame(data, index=idx)


def test_clusters_find_groups_and_fold_a_loner_in():
    m = two_groups()
    labels = questions.cluster(m, 3)
    s = pd.Series(labels, index=m.columns)
    assert s.filter(like="a").nunique() == 1
    assert s.filter(like="b").nunique() == 1
    assert s["a0"] != s["b0"]
    assert s.nunique() == 2
    assert s["loner"] == s["a0"]  # it correlates with group a, and a lone station can't be a shed


def test_stability_is_high_for_clear_groups():
    m = two_groups(days=800)
    assert questions.stability(m, 2, reps=5) > .9


def test_nearest_picks_the_closest_point():
    i, km = questions.nearest(49.0, -123.0, np.array([60.0, 49.1]), np.array([-120.0, -123.0]))
    assert i == 1
    assert 10 < km < 12
