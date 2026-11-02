"""Chart years follow data.LAST_YEAR, on made-up data."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import figures


@pytest.fixture
def drawn(monkeypatch):
    """Keep each chart's axes and text instead of writing a file."""
    seen = {}
    monkeypatch.setattr(figures, "save", lambda fig, name, title, subtitle, caption: seen.update(axes=fig.axes, subtitle=subtitle, caption=caption))
    yield seen
    plt.close("all")


def years(last):
    index = range(2005, last + 1)
    v = np.linspace(.1, .3, len(index))
    return pd.DataFrame({"measured": 7 + v, "without": 6 + v, "share_8": v, "smoke_days": 10 * v}, index=index)


def test_trap_on_todays_years_reads_as_published(drawn):
    y = years(2025)
    fixed = pd.DataFrame({"share": y.share_8.loc[2014:]})
    figures.trap(y, fixed, pd.Series({"Lower Mainland": 19}))
    assert "all twelve years" in drawn["caption"]
    assert list(drawn["axes"][0].get_xticks()) == list(range(2014, 2026))


def test_trap_takes_a_new_year(drawn, monkeypatch):
    monkeypatch.setattr(figures, "LAST_YEAR", 2026)
    y = years(2026)
    fixed = pd.DataFrame({"share": y.share_8.loc[2014:]})
    figures.trap(y, fixed, pd.Series({"Lower Mainland": 19}))
    assert "all thirteen years" in drawn["caption"]
    assert list(drawn["axes"][0].get_xticks()) == list(range(2014, 2027))
    assert drawn["axes"][0].texts[-1].xy[0] == 2026


def test_trap_uses_digits_past_the_words_it_knows(drawn, monkeypatch):
    monkeypatch.setattr(figures, "LAST_YEAR", 2027)
    y = years(2027)
    figures.trap(y, pd.DataFrame({"share": y.share_8.loc[2014:]}), pd.Series({"Lower Mainland": 19}))
    assert "all 14 years" in drawn["caption"]


def test_smoke_days_labels_follow_last_year(drawn, monkeypatch):
    monkeypatch.setattr(figures, "LAST_YEAR", 2026)
    figures.smoke_days(years(2026))
    labels = [t.get_text() for t in drawn["axes"][0].texts]
    assert labels[0].startswith("2005 to 2016")
    assert labels[1].startswith("2017 to 2026")
    assert drawn["axes"][0].texts[1].xy[0] == 2021.5


def test_every_year_means_every_year_through_last_year(monkeypatch):
    rows = [{"st": "always", "region": "A", "year": y} for y in range(2014, 2027)] + [{"st": "stopped", "region": "B", "year": y} for y in range(2014, 2026)]
    sy = pd.DataFrame(rows)
    assert set(figures.reported_every_year(sy[sy.year <= 2025]).index) == {"A", "B"}  # through 2025 both report every year
    monkeypatch.setattr(figures, "LAST_YEAR", 2026)
    assert set(figures.reported_every_year(sy).index) == {"A"}
