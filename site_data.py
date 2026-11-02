"""Write output/smoke-data.js for the interactive charts on the project page."""
import json

import numpy as np
import pandas as pd

import data
from data import DATA, LAST_YEAR, OUTPUT, STUDY_START
from smoke import THRESHOLDS

TOLERANCE = 0.04   # degrees: outline detail kept for a small canvas
MIN_RING = 0.05    # square degrees: smaller islands are dropped


def simplify(ring, tolerance=TOLERANCE):
    """Douglas-Peucker: keep a point only if it sits farther than `tolerance` from the line between its neighbours kept so far."""
    a = np.asarray(ring)[:, :2]
    keep = np.zeros(len(a), bool)
    keep[[0, -1]] = True
    stack = [(0, len(a) // 2), (len(a) // 2, len(a) - 1)]  # closed rings start and end on the same point
    keep[len(a) // 2] = True
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        line = a[j] - a[i]
        length = np.hypot(*line) or 1
        distance = np.abs(line[0] * (a[i + 1:j, 1] - a[i, 1]) - line[1] * (a[i + 1:j, 0] - a[i, 0])) / length
        k = int(distance.argmax())
        if distance[k] > tolerance:
            k += i + 1
            keep[k] = True
            stack += [(i, k), (k, j)]
    return np.round(a[keep], 2).tolist()


def ring_area(ring):
    a = np.asarray(ring)
    return abs(np.dot(a[:-1, 0], a[1:, 1]) - np.dot(a[1:, 0], a[:-1, 1])) / 2


def outline_rings():
    rings = []
    for f in data.outline():
        g = f["geometry"]
        polygons = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        rings += [simplify(poly[0]) for poly in polygons if ring_area(poly[0]) >= MIN_RING]
    return rings


def station_days(p, years):
    """For each station a season of PM2.5 (whole numbers, -1 for no reading) and flags from May 1 to Oct 31, one digit a day:
    0 not smoke at any threshold, 1 smoke only at 5, 2 at 8 as well, 3 at 12 too."""
    p = p[p.year.isin(years) & p.DATE.dt.month.between(5, 10)].copy()
    p["i"] = p.DATE.dt.day_of_year - pd.to_datetime(p.year.astype(str) + "-05-01").dt.day_of_year
    p["code"] = 3 * p.smoke12.astype(int) + (p.smoke & ~p.smoke12).astype(int) * 2 + (p.smoke5 & ~p.smoke).astype(int)
    out = {}
    for st, g in p.groupby("st"):
        pm, flags = {}, {}
        for year, h in g.groupby("year"):
            a, b = np.full(184, -1), np.full(184, 9)
            a[h.i.values] = np.round(h["mean"].values).astype(int).clip(min=0)
            b[h.i.values] = h.code.values
            pm[int(year)] = a.tolist()
            flags[int(year)] = "".join(str(v) for v in b).replace("9", "-")
        out[st] = {"pm": pm, "flags": flags}
    return out


def main():
    p = pd.read_pickle(DATA / "panel.pkl")
    st = pd.read_csv(OUTPUT / "stations.csv", index_col=0)
    sheds = pd.read_csv(OUTPUT / "sheds.csv", index_col=0).shed
    y = pd.read_csv(OUTPUT / "by_year.csv", index_col=0)
    years = list(range(STUDY_START, LAST_YEAR + 1))
    days = station_days(p, years)
    fires = data.large_fires()
    keep = [s for s in st.index if s in days]
    data_ = {"years": years, "thresholds": [int(t) for t in THRESHOLDS],
             "stations": [{"name": s, "lat": round(float(st.LAT[s]), 3), "lon": round(float(st.LONG[s]), 3), "region": st.REGION[s], "shed": sheds[s],
                           "pm": days[s]["pm"], "flags": days[s]["flags"]} for s in keep],
             "byYear": {"years": [int(v) for v in y.index], "measured": y.measured.round(2).tolist(),
                        **{f"share{int(t)}": y[f"share_{int(t)}"].round(3).tolist() for t in THRESHOLDS}},
             "fires": {str(k): [[round(r.LATITUDE, 2), round(r.LONGITUDE, 2), int(r.CURRENT_SIZE), r.INCIDENT_NAME or r.FIRE_NUMBER] for r in g.itertuples()] for k, g in fires.groupby("FIRE_YEAR")},
             "outline": outline_rings()}
    text = "const SMOKE = " + json.dumps(data_, separators=(",", ":")) + ";\n"
    (OUTPUT / "smoke-data.js").write_text(text, encoding="utf-8")
    print(len(text) // 1024, "KB", len(keep), "stations", sum(len(r) for r in data_["outline"]), "outline points")


if __name__ == "__main__":
    main()
