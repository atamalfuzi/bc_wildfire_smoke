"""Question 1: how much of BC's PM2.5 is wildfire smoke, by station and year?

Builds the daily panel (moved stations joined, instruments put on one footing), flags smoke days with a rule any reader
can check, and tests the rule against BC's own list, satellite smoke polygons, PM10 and carbon monoxide.

Run from the repository folder, after data.py: python smoke.py
"""
import numpy as np
import pandas as pd
from scipy import stats

import data
from data import FIRST_YEAR, LAST_YEAR, OUTPUT, STUDY_START

THRESHOLD = 8.0      # ug/m3 above the station's usual level for the time of year
THRESHOLDS = (5.0, 8.0, 12.0)
SHARE = 0.5          # of reporting stations within REACH km that must be up by half the threshold
REACH = 200
SEASON = (5, 10)     # May to October: keeps out most wood smoke and winter inversions
MIN_HOURS = 18       # a day counts with at least this many hours
MIN_DAYS = 274       # a station-year counts with at least this many days (75%)
HIGH = 25.0          # ug/m3, the daily level BC's annual statistics files test against

KIND = {"PM25 SHARP5030": "SHARP", "PM25_SHARP5030": "SHARP", "PM25_SHARP5030i": "SHARP", "PM25_SHARP5030iQ": "SHARP",
        "PM25_R&P_TEOM": "TEOM", "BAM1020": "BAM", "PM25_T640": "T640", "API_T640X": "T640", "API T640": "T640"}


def join_moved(f, merges):
    """Carry moved stations under one name, using BC's table. Where a day has two rows, keep the one with more hours."""
    f = f.copy()
    f["raw"] = data.clean_name(f.STATION_NAME)
    f["st"] = f.raw.replace(merges.to_dict())
    return f.sort_values("n").drop_duplicates(["st", "DATE"], keep="last")


def load(years=range(FIRST_YEAR, LAST_YEAR + 1)):
    f = pd.concat((data.daily("PM25", y) for y in years), ignore_index=True)
    f = f[f.n >= MIN_HOURS]
    f = join_moved(f, data.station_merges()).reset_index(drop=True)
    f["DATE"] = pd.to_datetime(f.DATE)
    f["year"] = f.DATE.dt.year
    f["inst"] = f.INSTRUMENT.map(KIND).fillna("other")
    return f[["st", "raw", "DATE", "year", "mean", "hi", "n", "inst"]]


def warm(dates):
    return (dates.dt.month >= SEASON[0]) & (dates.dt.month <= SEASON[1])


def instrument_offsets(f, reps=200, seed=1):
    """How far each instrument reads from the SHARP on a station's annual median, from stations that changed instrument,
    with station and year effects. Separately for May to October and the cold months, as the TEOM's gap is seasonal.
    A median is little moved by smoke. Ranges are 90%, resampling stations."""
    rows = []
    for season, g in f[f.inst != "other"].assign(season=warm(f.DATE).map({True: "warm", False: "cold"})).groupby("season"):
        a = g.groupby(["st", "year", "inst"])["mean"].agg(["median", "size"]).reset_index()
        a = a[a["size"] >= 75]
        stations = pd.get_dummies(a.st, dtype=float)
        years = pd.get_dummies(a.year, dtype=float).iloc[:, 1:]
        kinds = pd.get_dummies(a.inst, dtype=float).drop(columns="SHARP")
        x = np.column_stack([stations.values, years.values, kinds.values])
        y = a["median"].values
        best = np.linalg.lstsq(x, y, rcond=None)[0][-kinds.shape[1]:]
        names, rng, boot = a.st.unique(), np.random.default_rng(seed), []
        where = {n: np.where(a.st.values == n)[0] for n in names}
        for _ in range(reps):
            idx = np.concatenate([where[n] for n in rng.choice(names, len(names))])
            boot.append(np.linalg.lstsq(x[idx], y[idx], rcond=None)[0][-kinds.shape[1]:])
        low, high = np.percentile(boot, [5, 95], axis=0)
        rows += [{"season": season, "instrument": k, "offset": best[i], "low": low[i], "high": high[i], "station_years": int(kinds[k].sum())}
                 for i, k in enumerate(kinds.columns)]
    return pd.DataFrame(rows)


def harmonise(f, offsets):
    """Put every reading on the SHARP's footing by taking off the instrument's offset for that part of the year."""
    season = warm(f.DATE).map({True: "warm", False: "cold"})
    off = offsets.set_index(["season", "instrument"]).offset
    adj = pd.Series(list(zip(season, f.inst)), index=f.index).map(off.to_dict()).fillna(0.0)
    return f.assign(pm=f["mean"] - adj)


def baseline(wide, half=15, min_days=30):
    """Each station's median by day of the year, +/- half days. A median so that smoke days move it little."""
    doy = wide.index.day_of_year.values
    near = np.abs((doy[None, :] - np.arange(1, 367)[:, None] + 183) % 366 - 183) <= half
    out = {}
    for c in wide:
        v, t = wide[c].values, np.full(366, np.nan)
        for k in range(366):
            s = v[near[k]]
            if np.isfinite(s).sum() >= min_days:
                t[k] = np.nanmedian(s)
        out[c] = t[doy - 1]
    return pd.DataFrame(out, index=wide.index)


def neighbours(lat, lon, km=REACH):
    r = np.radians
    a = np.sin((r(lat[:, None]) - r(lat[None])) / 2) ** 2 + np.cos(r(lat[:, None])) * np.cos(r(lat[None])) * np.sin((r(lon[:, None]) - r(lon[None])) / 2) ** 2
    d = 2 * 6371 * np.arcsin(np.sqrt(a))
    return (d <= km) & (d > 0)


def detect(wide, base, near, thresh=THRESHOLD, share=SHARE, min_neighbours=2, months=SEASON):
    """A smoke day: from May to October, a station is at least `thresh` above its usual level and at least `share` of
    the reporting stations within 200 km are up by half that. Local sources (a road, a mill, a stove) rarely lift a whole
    region at once; regional smoke does. A station with too few neighbours to check needs twice the threshold."""
    excess = wide - base
    up = (excess >= thresh / 2).values.astype(float)
    seen = excess.notna().values.astype(float)
    n_up, n_seen = up @ near.astype(float), seen @ near.astype(float)
    regional = (n_up >= share * n_seen) & (n_seen >= min_neighbours)
    alone = n_seen < min_neighbours
    x = excess.values
    flag = ((x >= thresh) & regional) | ((x >= 2 * thresh) & alone)
    m = wide.index.month
    flag = flag & ((m >= months[0]) & (m <= months[1]))[:, None]
    return pd.DataFrame(flag, index=wide.index, columns=wide.columns), excess


def panel(f, st):
    """Long table: one row per station-day with its excess and smoke flags at each threshold."""
    wide = f.pivot(index="DATE", columns="st", values="pm").asfreq("D")
    wide = wide[[c for c in wide if c in st.index]]
    base = baseline(wide)
    near = neighbours(st.LAT[wide.columns].values, st.LONG[wide.columns].values)
    out = f[f.st.isin(wide.columns)]

    def long(frame, name):
        return frame.stack().rename_axis(["DATE", "st"]).rename(name).reset_index()
    excess = None
    for t in THRESHOLDS:
        flag, excess = detect(wide, base, near, t)
        out = out.merge(long(flag, f"smoke{t:.0f}"), on=["DATE", "st"])
    out = out.merge(long(excess, "excess"), on=["DATE", "st"])
    out = out.rename(columns={f"smoke{THRESHOLD:.0f}": "smoke"})
    out["base"] = out.pm - out.excess
    out["region"] = out.st.map(st.REGION)
    return out


def year_effects(frame, column):
    """Year levels from station and year effects, with stations weighted so each region counts equally. A plain average
    moves as monitors open and close: over 2014 to 2025 the stations that report every year are mostly in the Lower Mainland."""
    f = frame.dropna(subset=[column])
    stations = pd.get_dummies(f.st, dtype=float).values
    years = sorted(f.year.unique())
    dummies = np.column_stack([(f.year == y).values.astype(float) for y in years[1:]])
    weight = np.sqrt(1 / f.groupby("region").st.transform("nunique").values)
    co = np.linalg.lstsq(np.column_stack([stations, dummies]) * weight[:, None], f[column].values * weight, rcond=None)[0]
    effect = np.r_[0, co[stations.shape[1]:]]
    net = f[column].values - effect[np.searchsorted(years, f.year.values)]
    level = pd.Series(net).groupby(f.region.values).mean().mean()
    return pd.Series(effect + level, index=years)


def station_years(p, column="smoke"):
    """Annual mean as measured and with smoke days set back to the station's usual level."""
    p = p.assign(without=p.pm.where(~p[column], p.base.where(p.excess > 0, p.pm)), high=p.pm > HIGH)
    y = p.groupby(["st", "region", "year"]).agg(measured=("pm", "mean"), without=("without", "mean"), smoke_days=(column, "sum"), high_days=("high", "sum"), days=("pm", "size")).reset_index()
    y = y[y.days >= MIN_DAYS]
    y["share"] = 1 - y.without / y.measured
    return y


def by_year(p):
    """Province-wide year levels and smoke share at each threshold."""
    out = pd.DataFrame()
    for t in THRESHOLDS:
        column = "smoke" if t == THRESHOLD else f"smoke{t:.0f}"
        y = station_years(p, column)
        m, w = year_effects(y, "measured"), year_effects(y, "without")
        if t == THRESHOLD:
            out["measured"], out["without"] = m, w
            out["smoke_days"] = year_effects(y, "smoke_days")
            out["high_days"] = year_effects(y, "high_days")
            out["stations"] = y.groupby("year").st.nunique()
        out[f"share_{t:.0f}"] = 1 - w / m
    return out


def validate(p, flagged, polygons, official):
    """Each check on the rule, as one row with the two numbers it compares."""
    rows = []
    def add(check, a, b="", note=""):
        rows.append({"check": check, "value": a, "other": b, "note": note})
    s = p[p.year >= STUDY_START]
    m = s.merge(flagged[["station", "date", "fire"]], left_on=["raw", "DATE"], right_on=["station", "date"], how="left")
    m["listed"] = m.station.notna()
    add("BC-listed station-days the rule also flags", m[m.listed].smoke.mean(), "", f"{int(m.listed.sum())} days")
    add("BC-listed days above 25 ug/m3", (m[m.listed]["mean"] > HIGH).mean())
    season = m[warm(m.DATE)]
    bands = pd.cut(season["mean"], [10, 15, 25, 50, 1e4], right=False)
    for band, g in season.groupby(bands):
        add(f"May-Oct days at {band.left:.0f} to {band.right:.0f} ug/m3: flagged by the rule, on BC's list", g.smoke.mean(), g.listed.mean(), f"{len(g)} days")
    cold = m[~warm(m.DATE) & (m["mean"] > HIGH)]
    add("Nov-Apr station-days above 25 ug/m3, left out by design", len(cold), "", ", ".join(cold.st.value_counts().head(3).index))

    h = s.merge(polygons, left_on=["raw", "DATE"], right_on=["STATION_NAME", "DATE"])
    add("Under any satellite smoke polygon: rule's days, other days", (h[h.smoke].hms >= 1).mean(), (h[~h.smoke].hms >= 1).mean(), f"{h.DATE.nunique()} days, Jul-Sep")
    add("Under a medium or heavy polygon: rule's days, other days", (h[h.smoke].hms >= 2).mean(), (h[~h.smoke].hms >= 2).mean())
    add("Polygon overhead but under 2 ug/m3 of excess at the ground", (h[h.hms >= 1].excess < 2).mean())
    for parameter, label in (("PM10", "PM2.5/PM10 ratio"), ("CO", "Carbon monoxide, ppm")):
        g = pd.concat(data.daily(parameter, y) for y in (2018, 2021, 2023))
        g = g[g.n >= MIN_HOURS].assign(raw=lambda d: data.clean_name(d.STATION_NAME), DATE=lambda d: pd.to_datetime(d.DATE)).rename(columns={"mean": "other"})
        j = s.merge(g[["raw", "DATE", "other"]], on=["raw", "DATE"])
        j = j[j.DATE.dt.month.between(7, 9)]
        if parameter == "PM10":
            j = j[(j.other > 1) & (j["mean"] > 0)]
            j["other"] = j["mean"] / j.other
            ranks = stats.rankdata(j.other)
            n1, n0 = j.smoke.sum(), (~j.smoke).sum()
            auc = (ranks[j.smoke.values].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
            add("AUC of the PM2.5/PM10 ratio alone for the rule's days", auc)
        add(f"{label}, Jul-Sep: rule's days, other days", j[j.smoke].other.median(), j[~j.smoke].other.median(), f"{j.raw.nunique()} stations, 2018, 2021, 2023")

    # Same stations as BC's with and without figures
    y = station_years(s)
    o = official[official.station_name.isin(y.st.unique())].pivot_table(index=["station_name", "year"], columns="tfee", values="value").dropna()
    o.columns = ["with_listed", "without_listed"]
    j = y.merge(o.reset_index(), left_on=["st", "year"], right_on=["station_name", "year"])
    removed = j.assign(rule=j.measured - j.without, bc=j.with_listed - j.without_listed).groupby("year")[["rule", "bc"]].mean()
    add("Yearly amount removed, rule against BC's figures (correlation)", np.corrcoef(removed.rule, removed.bc)[0, 1], "", f"{j.st.nunique()} stations")
    return pd.DataFrame(rows), removed, h


def season_files(p):
    """What the dashboard needs without downloading history: each station's usual level by day of the year, and how the
    study's seasons built up, as the share of each region's stations flagged on each day from May 1."""
    p = p.assign(doy=p.DATE.dt.day_of_year)
    usual = p.groupby(["doy", "st"]).base.first().unstack().round(2)
    usual.to_csv(OUTPUT / "usual_level.csv")
    s = p[(p.year >= STUDY_START) & warm(p.DATE)]
    s = s.assign(day=(s.DATE - pd.to_datetime(s.year.astype(str) + "-05-01")).dt.days)
    h = s.groupby(["year", "day", "region"]).smoke.mean().rename("smoke_share").reset_index()
    h.round(4).to_csv(OUTPUT / "season_history.csv", index=False)


def fixed_panel(p):
    """The trap: stations reporting in every study year, averaged plainly."""
    y = station_years(p[p.year >= STUDY_START])
    counts = y.groupby("st").year.nunique()
    keep = counts[counts == LAST_YEAR - STUDY_START + 1].index
    y = y[y.st.isin(keep)]
    t = y.groupby("year")[["measured", "without"]].mean()
    t["share"] = 1 - t.without / t.measured
    return t, y.drop_duplicates("st").region.value_counts()


def main():
    st_all = data.stations()
    st = st_all.copy()
    merges = data.station_merges()
    f = load()
    # A joined station takes the coordinates and region of its newest location
    last = f.sort_values("DATE").drop_duplicates("st", keep="last").set_index("st").raw
    st = st_all.reindex(last.values).set_index(last.index).dropna(subset=["LAT"])

    offsets = instrument_offsets(f)
    offsets.round(3).to_csv(OUTPUT / "instrument_offsets.csv", index=False)
    print(offsets.round(2))
    f = harmonise(f, offsets)
    p = panel(f, st)
    p.to_pickle(data.DATA / "panel.pkl")
    season_files(p)
    st[["LAT", "LONG", "REGION", "CITY"]].to_csv(OUTPUT / "stations.csv")

    print("station-days", len(p), "stations", p.st.nunique(), "moved stations joined", (f.raw != f.st).groupby(f.st).any().sum(), "merge rows", len(merges))
    years = by_year(p)
    years["area_burned_ha"] = data.area_burned()
    years.round(3).to_csv(OUTPUT / "by_year.csv")
    print(years.round(2))
    fixed, regions = fixed_panel(p)
    fixed.round(3).to_csv(OUTPUT / "fixed_panel.csv")
    print(fixed.round(2), regions.to_dict())

    polygons = pd.read_csv(data.DATA / "hms.csv", parse_dates=["DATE"])
    checks, removed, _ = validate(p, data.flagged_days(), polygons, data.official_annual())
    checks.to_csv(OUTPUT / "validation.csv", index=False)
    removed.round(3).to_csv(OUTPUT / "removed_vs_bc.csv")
    print(checks.to_string())

    sy = station_years(p)
    sy.round(3).to_csv(OUTPUT / "station_years.csv", index=False)
    regions = sy[sy.year >= STUDY_START].groupby(["region", "year"])[["smoke_days", "high_days", "measured", "without"]].mean().reset_index()
    regions.round(2).to_csv(OUTPUT / "by_region.csv", index=False)
    print(regions.pivot(index="region", columns="year", values="smoke_days").round(0).to_string())


if __name__ == "__main__":
    main()
