"""Two ideas tested and dropped, kept so the numbers can be checked.

Next-day forecast: does the monitor network, with or without satellite fire power by distance, beat "tomorrow equals today"?
Each season is held out in turn. Split conformal 90% ranges take their width from the season before the one held out.

Source split: does a regression of each region's smoke on fire power by province hold up on a season it wasn't fitted to?

Run from the repository folder, after smoke.py: python dropped.py
"""
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor as Boosting

import data
from data import DATA, HOTSPOT_YEARS, OUTPUT, STUDY_START
from questions import SOURCE
from smoke import HIGH, neighbours, warm

BANDS = [0, 150, 400, 800, 1500, 3000]  # km
BAND_COLUMNS = [f"within_{b}" for b in BANDS[1:]]
NETWORK = ["own", "own1", "own2", "near", "near1", "province", "lat", "lon", "doy"]


def fire_bands(stations, lat, lon):
    """Log fire power by day in distance bands around each station."""
    rows = []
    la, lo = np.radians(lat), np.radians(lon)
    for year in HOTSPOT_YEARS:
        g = data.hotspots(year).groupby(["date", "la", "lo"]).frp.sum().reset_index()
        for day, c in g.groupby("date"):
            a = np.sin((np.radians(c.la.values)[None] - la[:, None]) / 2) ** 2 + np.cos(la)[:, None] * np.cos(np.radians(c.la.values))[None] * np.sin((np.radians(c.lo.values)[None] - lo[:, None]) / 2) ** 2
            km = 2 * 6371 * np.arcsin(np.sqrt(a))
            f = c.frp.values
            for i, s in enumerate(stations):
                rows.append((day, s) + tuple(np.log1p(f[(km[i] >= u) & (km[i] < v)].sum()) for u, v in zip(BANDS, BANDS[1:])))
    out = pd.DataFrame(rows, columns=["DATE", "st"] + BAND_COLUMNS)
    out["DATE"] = pd.to_datetime(out.DATE)
    return out


def features(p, st):
    """One row per station-day from May to October: today's readings, the network's, and tomorrow's reading to predict."""
    wide = p[p.year >= STUDY_START].pivot(index="DATE", columns="st", values="pm").asfreq("D")
    near = neighbours(st.LAT[wide.columns].values, st.LONG[wide.columns].values, 300).astype(float)
    seen = wide.notna().values.astype(float)
    mean_near = pd.DataFrame(np.divide(np.nan_to_num(wide.values) @ near, seen @ near, out=np.full(wide.shape, np.nan), where=(seen @ near) > 0), index=wide.index, columns=wide.columns)
    province = wide.mean(axis=1)
    rows = []
    for c in wide:
        t = pd.DataFrame({"y": wide[c].shift(-1), "own": wide[c], "own1": wide[c].shift(1), "own2": wide[c].shift(2),
                          "near": mean_near[c], "near1": mean_near[c].shift(1), "province": province})
        t["st"], t["lat"], t["lon"] = c, st.LAT[c], st.LONG[c]
        rows.append(t)
    t = pd.concat(rows).rename_axis("DATE").reset_index()
    t["doy"], t["year"] = t.DATE.dt.day_of_year, t.DATE.dt.year
    return t[warm(t.DATE)].dropna(subset=["y", "own"]).reset_index(drop=True)


def score(y, today, pred, low=None, high=None):
    """Average error, days above HIGH caught and falsely called, and the first days of high readings caught."""
    first = (y > HIGH) & (today <= HIGH)
    out = {"mean_error": np.abs(pred - y).mean(), "caught_high": ((pred > HIGH) & (y > HIGH)).sum() / (y > HIGH).sum(),
           "false_alarms": ((pred > HIGH) & (y <= HIGH)).sum() / max((pred > HIGH).sum(), 1), "caught_first_days": (pred[first] > HIGH).mean(), "first_days": int(first.sum())}
    if low is not None:
        inside = (y >= low) & (y <= high)
        out["range_covers"], out["range_covers_high_days"] = inside.mean(), inside[y > HIGH].mean()
    return out


def held_out(t, columns):
    """Fit on every other season, calibrate on the season before, predict the one held out."""
    target = np.log1p(t.y.clip(lower=0))
    pred, low, high = [np.full(len(t), np.nan) for _ in range(3)]
    for year in sorted(t.year.unique()):
        test = (t.year == year).values
        cal = (t.year == (year - 1 if year > t.year.min() else year + 1)).values
        fit = ~test & ~cal
        model = Boosting(max_iter=200, learning_rate=.08, random_state=0).fit(t.loc[fit, columns], target[fit])
        width = np.quantile(np.abs(target[cal] - model.predict(t.loc[cal, columns])), .9)
        guess = model.predict(t.loc[test, columns])
        pred[test], low[test], high[test] = np.expm1(guess), np.expm1(guess - width), np.expm1(guess + width)
    return pred, low, high


def forecast(p, st):
    t = features(p, st)
    t = t.merge(fire_bands(list(st.index), st.LAT.values, st.LONG.values), on=["DATE", "st"], how="left").fillna({c: 0.0 for c in BAND_COLUMNS})
    rows = [{"model": "Tomorrow equals today", **score(t.y, t.own, t.own)}]
    for name, columns in (("Monitor network", NETWORK), ("Fire power by distance alone", BAND_COLUMNS + ["lat", "lon", "doy"]), ("Network and fire power", NETWORK + BAND_COLUMNS)):
        pred, low, high = held_out(t, columns)
        rows.append({"model": name, **score(t.y, t.own, pd.Series(pred, index=t.index), pd.Series(low, index=t.index), pd.Series(high, index=t.index))})
    return pd.DataFrame(rows)


def source_split(p):
    """Each region's daily excess on fire power by province over three days, no negative effects, fitted on all seasons but one."""
    groups = []
    for year in HOTSPOT_YEARS:
        h = data.hotspots(year)
        groups.append(h.assign(source=h.agency.map(SOURCE)).dropna(subset=["source"]).groupby(["date", "source"]).frp.sum())
    frp = pd.concat(groups).unstack(fill_value=0)
    frp.index = pd.to_datetime(frp.index)
    frp = frp.rolling(3, min_periods=1).mean() / 1e4
    rows = []
    summer = p[(p.year >= STUDY_START) & p.DATE.dt.month.between(7, 9)]
    for region, g in summer.groupby("region"):
        y = g.groupby("DATE").excess.mean().clip(lower=0)
        x = frp.reindex(y.index).fillna(0)
        for year in sorted(set(y.index.year)):
            train = y.index.year != year
            co, _ = nnls(x.values[train], y.values[train])
            fit = x.values[~train] @ co
            held = y.values[~train]
            rows.append({"region": region, "year": year, "r2": 1 - ((held - fit) ** 2).sum() / ((held - held.mean()) ** 2).sum()})
    return pd.DataFrame(rows)


def main():
    p = pd.read_pickle(DATA / "panel.pkl")
    st = pd.read_csv(OUTPUT / "stations.csv", index_col=0)
    st = st[st.index.isin(p[p.year >= STUDY_START].st.unique())]
    f = forecast(p, st)
    f.round(3).to_csv(OUTPUT / "forecast_check.csv", index=False)
    print(f.round(3).to_string())
    s = source_split(p)
    s.round(3).to_csv(OUTPUT / "source_split_check.csv", index=False)
    print(s.groupby("region").r2.describe().round(2).to_string())
    print("median held-out R2 overall", round(s.r2.median(), 3), "share of seasons above 0.25", round((s.r2 > .25).mean(), 3))


if __name__ == "__main__":
    main()
