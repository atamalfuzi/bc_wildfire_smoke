"""Questions 2 to 4, from the panel smoke.py wrote.

2. Is the rest of the air getting cleaner, and is smoke getting worse? Trends with honest ranges.
3. Which places get smoke together, and how many people are there? Clusters of stations, then population.
4. How much arrives from outside BC? The largest episodes, each with its fire activity and the fires BC names.

Run from the repository folder, after smoke.py: python questions.py
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score

import data
from data import DATA, OUTPUT, STUDY_START
from smoke import HIGH, MIN_DAYS, station_years, warm

KS = range(3, 11)      # numbers of smoke sheds to try
SHED_NAMES = {"Vancouver International Airport #2": "South coast", "Courtenay Elementary School": "Mid-island and Sunshine Coast",
              "Kamloops Federal Building": "Southern Interior and Kootenays", "Prince George Plaza 400": "Central Interior",
              "Fort St John 85th Avenue": "Peace and Nechako", "Smithers": "Bulkley Valley", "Terrace Skeena Middle School": "North coast"}
MIN_SHED = 3           # a shed needs at least this many stations
BIG_FIRE_REACH = 1500  # km around an episode's stations in which fire power is counted
SOURCE = {"BC": "BC", "AB": "Alberta, NWT, Yukon", "NT": "Alberta, NWT, Yukon", "YT": "Alberta, NWT, Yukon", "SK": "Alberta, NWT, Yukon",
          "WA": "Washington, Idaho, Montana", "ID": "Washington, Idaho, Montana", "MT": "Washington, Idaho, Montana",
          "OR": "Oregon, California", "CA": "Oregon, California"}


# 2. Trends ----
def trend(series):
    """Theil-Sen slope per year with its 95% range, and Kendall's test."""
    slope, _, low, high = stats.theilslopes(series.values, series.index.values, alpha=0.95)
    return {"slope": slope, "low": low, "high": high, "kendall_p": stats.kendalltau(series.index.values, series.values).pvalue}


def compare_periods(series, split):
    """Difference in means from `split` on against before it, with an exact permutation test: every way of choosing which
    years count as the later ones."""
    v, later = series.values, series.index.values >= split
    observed = v[later].mean() - v[~later].mean()
    k, total, extreme, count = later.sum(), v.sum(), 0, 0
    for c in combinations(range(len(v)), int(k)):
        a = v[list(c)].sum()
        diff = a / k - (total - a) / (len(v) - k)
        extreme += abs(diff) >= abs(observed) - 1e-12
        count += 1
    return {"before": v[~later].mean(), "after": v[later].mean(), "difference": observed, "p": extreme / count}


def p_text(p):
    """An exact permutation p is never 0, as the observed split always counts, so a small one reads as a bound."""
    return "< 0.001" if p < .001 else f"{p:.3f}"


def rounded_periods(periods):
    """Three decimals for the file, except p, which rounding would turn into 0."""
    return periods.round(3).assign(p=periods.p)


def trends(years):
    rows = []
    for first in (data.FIRST_YEAR, STUDY_START):
        for column, label in (("measured", "PM2.5 as measured"), ("without", "PM2.5 with smoke removed"), ("smoke_days", "Smoke days per station"), ("share_8", "Smoke share")):
            rows.append({"from": first, "series": label, **trend(years.loc[first:, column])})
    top3 = years.share_8.nlargest(3).index
    rows.append({"from": data.FIRST_YEAR, "series": "Smoke days per station, leaving out the 3 worst years", **trend(years.smoke_days.drop(top3))})
    t = pd.DataFrame(rows)
    p = pd.DataFrame([{"series": label, **compare_periods(years[column], 2017)} for column, label in (("smoke_days", "Smoke days per station"), ("share_8", "Smoke share"), ("without", "PM2.5 with smoke removed"))])
    return t, p


# 3. Smoke sheds ----
def shed_matrix(p, min_days=500):
    """Stations by summer day, the log of their excess. Stations with too few summer days are placed later by distance."""
    w = p[(p.year >= STUDY_START) & p.DATE.dt.month.between(7, 9)].pivot(index="DATE", columns="st", values="excess")
    return np.log1p(w.loc[:, w.notna().sum() >= min_days].clip(lower=0))


def cluster(matrix, k):
    """Average-linkage clusters on rank correlation. A cluster under MIN_SHED stations joins the one it correlates with best."""
    c = matrix.corr(method="spearman", min_periods=200).fillna(0).values
    distance = 1 - c
    upper = distance[np.triu_indices_from(distance, 1)]
    labels = fcluster(linkage(upper, "average"), k, "maxclust")
    sizes = pd.Series(labels).value_counts()
    big = sizes[sizes >= MIN_SHED].index.values
    for small in sizes[sizes < MIN_SHED].index:
        members = labels == small
        labels[members] = big[np.argmax([c[np.ix_(members, labels == b)].mean() for b in big])]
    return pd.factorize(labels)[0] + 1


def stability(matrix, k, reps=60, seed=1):
    """How much the clusters change when the seasons are resampled: mean adjusted Rand index against the full-data clusters."""
    rng, full = np.random.default_rng(seed), cluster(matrix, k)
    years = matrix.index.year.unique().values
    scores = []
    for _ in range(reps):
        picks = rng.choice(years, len(years))
        sample = pd.concat([matrix[matrix.index.year == y] for y in picks], ignore_index=True)
        scores.append(adjusted_rand_score(full, cluster(sample, k)))
    return float(np.mean(scores))


def choose_k(matrix):
    """The cut that changes least when the seasons are resampled."""
    t = pd.DataFrame([{"k": k, "stability": stability(matrix, k), "sizes": np.bincount(cluster(matrix, k))[1:].tolist()} for k in KS])
    return int(t.sort_values(["stability", "k"], ascending=[False, False]).iloc[0].k), t


def nearest(lat, lon, lats, lons):
    r = np.radians
    a = np.sin((r(lats) - r(lat)) / 2) ** 2 + np.cos(r(lat)) * np.cos(r(lats)) * np.sin((r(lons) - r(lon)) / 2) ** 2
    d = 2 * 6371 * np.arcsin(np.sqrt(a))
    return int(np.nanargmin(d)), float(np.nanmin(d))


def assign_sheds(matrix, labels, st):
    """Clustered stations keep their shed; every other station joins the shed of its nearest clustered station."""
    shed = pd.Series(labels, index=matrix.columns)
    lats, lons = st.LAT[shed.index].values, st.LONG[shed.index].values
    out = {}
    for name, row in st.iterrows():
        out[name] = shed[name] if name in shed.index else shed.iloc[nearest(row.LAT, row.LONG, lats, lons)[0]]
    return pd.Series(out)


def areas():
    """Where people live: each municipality's polygon centre, and for unincorporated land the area-weighted centre of
    its regional district's other census subdivisions."""
    rows = []
    for f in data.census_subdivisions():
        g, q = f["geometry"], f["properties"]
        rings = g["coordinates"] if g["type"] == "Polygon" else max(g["coordinates"], key=lambda poly: len(poly[0]))
        xy = np.array(rings[0])
        rows.append({"csd": int(q["CENSUS_SUBDIVISION_ID"]) % 100000, "cd": int(q["CENSUS_SUBDIVISION_ID"]) // 1000 % 100,
                     "lon": xy[:, 0].mean(), "lat": xy[:, 1].mean(), "area": q["FEATURE_AREA_SQM"]})
    c = pd.DataFrame(rows)
    pop = data.population()
    known = set(pop.Region)
    fill = c[~c.csd.isin(known)]
    rest = fill.groupby("cd").apply(lambda g: pd.Series({"lon": np.average(g.lon, weights=g.area), "lat": np.average(g.lat, weights=g.area)}), include_groups=False)
    rest.index = rest.index * 1000 + 999
    places = pd.concat([c[c.csd.isin(known)].set_index("csd")[["lon", "lat"]], rest])
    return places, pop


def exposure(p, st, shed):
    """People by the number of days above HIGH at their nearest monitor reporting that year. This is air at a monitor, not
    what anyone breathed: most time is spent indoors and a monitor can be far away."""
    places, pop = areas()
    season = p[warm(p.DATE)]
    counts = season.groupby(["st", "year"]).agg(days=("pm", "size"), high=("pm", lambda s: (s > HIGH).sum()),
                                                smoke_high=("smoke", "sum")).reset_index()
    high = season[(season.pm > HIGH) & season.smoke].groupby(["st", "year"]).size().rename("smoke_high").reset_index()
    counts = counts.drop(columns="smoke_high").merge(high, how="left").fillna({"smoke_high": 0})
    rows, totals = [], []
    for year, c in counts[counts.days >= 138].groupby("year"):
        if year < STUDY_START:
            continue
        c = c[c.st.isin(st.index)]
        lats, lons = st.LAT[c.st].values, st.LONG[c.st].values
        people = pop[pop.Year == year].set_index("Region").Total
        seen = 0
        for place, row in places.iterrows():
            if place not in people.index:
                continue
            i, km = nearest(row.lat, row.lon, lats, lons)
            n = people[place]
            seen += n if km <= 100 else 0
            rows.append({"year": year, "shed": shed[c.st.iloc[i]], "people": n, "person_days": n * c.smoke_high.iloc[i], "all_high": n * c.high.iloc[i]})
        totals.append({"year": year, "covered_within_100km": seen / people.sum(), "people": people.sum()})
    r = pd.DataFrame(rows)
    by_shed = r.groupby(["year", "shed"])[["people", "person_days"]].sum().reset_index()
    by_year = r.groupby("year")[["people", "person_days", "all_high"]].sum()
    by_year["days_per_person"] = by_year.person_days / by_year.people
    by_year = by_year.join(pd.DataFrame(totals).set_index("year")[["covered_within_100km"]])
    return by_year, by_shed


# 4. Episodes ----
def episodes(daily_share, minimum=0.15, gap=2):
    """Runs of days when at least `minimum` of reporting stations are flagged, joining runs up to `gap` days apart."""
    days = daily_share.index[daily_share >= minimum]
    runs = []
    for d in days:
        if runs and (d - runs[-1][1]).days <= gap + 1:
            runs[-1][1] = d
        else:
            runs.append([d, d])
    return [(a, b) for a, b in runs]


def fire_power(start, end, lat, lon):
    """Share of satellite fire power by area, within reach of the affected stations, from three days before an episode to its end."""
    f = data.hotspots(start.year)
    f["date"] = pd.to_datetime(f.date)
    f = f[f.date.between(start - pd.Timedelta(days=3), end)]
    r = np.radians
    a = np.sin((r(f.la) - r(lat)) / 2) ** 2 + np.cos(r(lat)) * np.cos(r(f.la)) * np.sin((r(f.lo) - r(lon)) / 2) ** 2
    f = f[2 * 6371 * np.arcsin(np.sqrt(a)) <= BIG_FIRE_REACH]
    s = f.groupby(f.agency.map(SOURCE).fillna("Elsewhere")).frp.sum()
    return (s / s.sum()).round(3).to_dict() if s.sum() else {}


def describe_episodes(p, st, flagged, polygons, by_shed_people=None, top=15):
    season = p[(p.year >= STUDY_START) & warm(p.DATE)]
    share = season.groupby("DATE").smoke.mean()
    t = flagged.copy()
    rows = []
    for a, b in episodes(share):
        g = season[season.DATE.between(a, b)]
        hit = g[g.smoke]
        if not len(hit):
            continue
        lat, lon = st.LAT[hit.st].mean(), st.LONG[hit.st].mean()
        named = t[t.date.between(a, b)].fire.dropna().str.replace(r"\s*\(.*?\)", "", regex=True).str.strip().value_counts()
        listed = t[t.date.between(a, b)]
        poly = hit.merge(polygons, left_on=["raw", "DATE"], right_on=["STATION_NAME", "DATE"])
        peak = hit.loc[hit.pm.idxmax()]
        rows.append({"start": a.date(), "end": b.date(), "days": (b - a).days + 1, "stations_hit": hit.st.nunique(),
                     "station_days": len(hit), "excess_station_days": round(hit.excess.sum()),
                     "high_station_days": int((hit.pm > HIGH).sum()), "peak": round(peak.pm), "peak_station": peak.st,
                     "regions": ", ".join(hit.region.value_counts().head(3).index),
                     "under_polygon": round((poly.hms >= 1).mean(), 2) if len(poly) else None,
                     "bc_listed": len(listed), "named_fires": "; ".join(named.head(3).index), **fire_power(a, b, lat, lon)})
    e = pd.DataFrame(rows).sort_values("excess_station_days", ascending=False)
    return e, e.head(top)


def main():
    p = pd.read_pickle(DATA / "panel.pkl")
    st = pd.read_csv(OUTPUT / "stations.csv", index_col=0)
    years = pd.read_csv(OUTPUT / "by_year.csv", index_col=0)

    t, periods = trends(years)
    t.round(3).to_csv(OUTPUT / "trends.csv", index=False)
    rounded_periods(periods).to_csv(OUTPUT / "periods.csv", index=False)
    print(t.round(3).to_string())
    pd.DataFrame([{"first_year": f, "rank_correlation": stats.spearmanr(years.loc[f:].area_burned_ha, years.loc[f:].smoke_days)[0]} for f in (data.FIRST_YEAR, STUDY_START)]).round(3).to_csv(OUTPUT / "area_vs_smoke.csv", index=False)
    print(periods.round(3).assign(p=periods.p.map(p_text)).to_string())
    area = years.loc[STUDY_START:]
    print("area burned vs smoke days, rank correlation", stats.spearmanr(area.area_burned_ha, area.smoke_days), "all years", stats.spearmanr(years.area_burned_ha, years.smoke_days))

    matrix = shed_matrix(p)
    k, tried = choose_k(matrix)
    print(tried.to_string(), "chosen k", k)
    labels = cluster(matrix, k)
    tried.to_csv(OUTPUT / "shed_stability.csv", index=False)
    shed = assign_sheds(matrix, labels, st)
    region_score = adjusted_rand_score(st.REGION[matrix.columns], labels)
    print("adjusted Rand against admin regions", round(region_score, 2), "stations clustered", matrix.shape[1], "of", len(st))
    pd.DataFrame([{"k": k, "sheds": len(set(labels)), "stability": tried.set_index("k").stability[k], "adjusted_rand_vs_regions": region_score, "clustered": matrix.shape[1]}]).round(3).to_csv(OUTPUT / "shed_summary.csv", index=False)
    names = {shed[station]: name for station, name in SHED_NAMES.items()}
    if len(names) != shed.nunique():  # clusters moved since these names were chosen: look at them before publishing
        raise SystemExit(f"{shed.nunique()} sheds but {len(names)} names")
    shed = shed.map(names)
    shed.rename("shed").rename_axis("st").to_csv(OUTPUT / "sheds.csv")
    for s, g in shed.groupby(shed):
        print(s, (g.index.isin(matrix.columns)).sum(), list(g.index[g.index.isin(matrix.columns)])[:14])

    people, people_shed = exposure(p, st, shed)
    people.round(3).to_csv(OUTPUT / "people_by_year.csv")
    people_shed.round(0).to_csv(OUTPUT / "people_by_shed.csv", index=False)
    print(people.round(2).to_string())
    print(people_shed.assign(per=lambda d: d.person_days / d.people).pivot(index="shed", columns="year", values="per").round(1).to_string())

    polygons = pd.read_csv(DATA / "hms.csv", parse_dates=["DATE"])
    all_e, top = describe_episodes(p, st, data.flagged_days(), polygons)
    all_e.to_csv(OUTPUT / "episodes.csv", index=False)
    print(len(all_e), "episodes")
    print(top.drop(columns=["regions", "named_fires", "peak_station"]).to_string())
    print(top[["start", "regions", "named_fires", "peak_station"]].to_string())


if __name__ == "__main__":
    main()
