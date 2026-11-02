"""Seasonal dashboard for wildfire smoke in BC: this year's smoke days against 2014 to 2025, by region, from the unverified
hourly file BC updates daily. Writes a static site to site/.

Needs no history download: the study's usual levels and season history are in output/. Run from the repository folder:
python dashboard.py
"""
from pathlib import Path
import datetime as dt
import json
import urllib.error
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from data import OUTPUT
from smoke import HIGH, KIND, MIN_HOURS, SEASON, THRESHOLD, detect, harmonise, join_moved, neighbours

SITE = Path("site")
URL = "https://atamalfuzi.github.io/bc_wildfire_smoke/"
WRITE_UP = "https://atamalfuzi.github.io/bc-wildfire-smoke.html"
MIN_STATIONS = 40        # reporting in the last week, or it isn't the usual network
MAX_DAILY = 2000         # ug/m3: the highest verified daily mean since 2005 is under 900
MIN_DAILY = -10          # unverified files hold sensor faults, such as a daily mean of -582; such days are dropped
MAX_FAULTY = 0.01        # but not more than this share of them
BLUE, GREY, MUTED, INK = "#2a78d6", "#c9c8c3", "#52514e", "#0b0b0b"


def season_panel(raw, st, base, offsets, merges):
    """This season's station-days with excess and smoke flags, on the study's footing and with its usual levels."""
    f = raw[(raw.n >= MIN_HOURS) & raw["mean"].between(MIN_DAILY, MAX_DAILY)]
    f = join_moved(f, merges).reset_index(drop=True)
    f["DATE"] = pd.to_datetime(f.DATE)
    f["year"] = f.DATE.dt.year
    f["inst"] = f.INSTRUMENT.map(KIND).fillna("other")
    f = harmonise(f, offsets)
    year = f[f.DATE.dt.month.between(*SEASON)].year.max()
    f = f[(f.year == year) & f.st.isin(st.index) & f.st.isin(base.columns)]
    wide = f.pivot(index="DATE", columns="st", values="pm")
    wide = wide.reindex(pd.date_range(f"{year}-05-01", min(pd.Timestamp(f"{year}-10-31"), wide.index.max())))
    usual = pd.DataFrame({c: base[c].reindex(wide.index.day_of_year).values for c in wide}, index=wide.index)
    near = neighbours(st.LAT[wide.columns].values, st.LONG[wide.columns].values)
    flag, excess = detect(wide, usual, near, THRESHOLD)
    long = wide.stack().rename_axis(["DATE", "st"]).rename("pm").reset_index()
    long["smoke"] = flag.stack().rename_axis(["DATE", "st"]).reindex(pd.MultiIndex.from_frame(long[["DATE", "st"]])).values
    long["excess"] = excess.stack().rename_axis(["DATE", "st"]).reindex(pd.MultiIndex.from_frame(long[["DATE", "st"]])).values
    long["region"] = long.st.map(st.REGION)
    return long.dropna(subset=["pm"]), int(year)


def daily_share(p):
    """Share of a region's reporting stations flagged each day; the mean across regions counts each region equally."""
    return p.groupby(["DATE", "region"]).smoke.mean().unstack()


def to_date(history, year, last_day):
    """Smoke days per station by region for each past year, up to the same day of the season."""
    h = history[history.day <= last_day]
    return h.groupby(["year", "region"]).smoke_share.sum().unstack()


def faulty_share(raw):
    full = raw[raw.n >= MIN_HOURS]
    return float((~full["mean"].between(MIN_DAILY, MAX_DAILY)).mean())


def problems(p, year, today, published=None, faulty=0.0):
    """Reasons not to publish. Keeps the live page if the file is stale, thin or implausible."""
    found = []
    last = p.DATE.max()
    recent = p[p.DATE > last - pd.Timedelta(days=7)]
    if recent.st.nunique() < MIN_STATIONS:
        found.append(f"only {recent.st.nunique()} stations reporting in the last week")
    if (pd.Timestamp(f"{year}-05-08") <= pd.Timestamp(today) <= pd.Timestamp(f"{year}-11-01")) and (pd.Timestamp(today) - last).days > 5:
        found.append(f"latest reading is {last.date()}, more than five days old")
    if faulty > MAX_FAULTY:
        found.append(f"{faulty:.1%} of station-days have implausible daily means")
    if p.region.isna().any():
        found.append("a station without a region")
    if published and last.date().isoformat() < published:
        found.append(f"data end {last.date()} is older than the published {published}")
    return found


def published_through():
    try:
        with urllib.request.urlopen(URL + "status.json", timeout=30) as r:
            return json.load(r)["data_through"]
    except (urllib.error.URLError, KeyError, ValueError):
        return None


def line_figure(history, share, year):
    """Cumulative smoke days per station, this year over each study year."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.96, bottom=0.12)
    ticks = [pd.Timestamp(f"2001-{m:02d}-01").day_of_year - 121 for m in range(5, 11)]
    past = history.groupby(["year", "day"]).smoke_share.mean().unstack(0).cumsum()
    for y in past:
        ax.plot(past.index, past[y], color=GREY, lw=1)
        if y in (2018, 2023):
            ax.annotate(str(y), (past.index[-1], past[y].iloc[-1]), xytext=(-3, 4 if y == 2023 else -11), textcoords="offset points", ha="right", color=MUTED, fontsize=9)
    now = share.mean(axis=1).fillna(0).cumsum()
    x = (now.index - pd.Timestamp(f"{year}-05-01")).days
    ax.plot(x, now.values, color=BLUE, lw=2.4)
    ax.annotate(str(year), (x[-1], now.iloc[-1]), xytext=(4, 4), textcoords="offset points", color=BLUE, fontweight="bold")
    ax.set_xticks(ticks, ["May", "Jun", "Jul", "Aug", "Sep", "Oct"])
    ax.set_ylabel("Smoke days per station, so far", color=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    ax.set_ylim(0, None)
    fig.savefig(SITE / "season.png", dpi=200, facecolor="white")
    plt.close(fig)


def strip_figure(p, st, year):
    """Every station by day this season, grouped by region: dark on smoke days."""
    w = p.pivot(index="st", columns="DATE", values="pm")
    order = st.REGION[w.index].sort_values(kind="stable").index
    w = w.loc[order]
    smoke = p.pivot(index="st", columns="DATE", values="smoke").reindex_like(w).astype(float)
    colours = np.full(w.shape + (3,), 0.96)
    levels = np.clip(np.nan_to_num(w.values), 0, 100) / 100
    t = np.sqrt(levels)[..., None]
    grey = np.array([.94, .94, .93]) * (1 - t * .35)
    hot = np.array([.96, .78, .68]) * (1 - t) + np.array([.62, .17, .05]) * t
    colours = np.where(smoke.values[..., None] > 0, hot, grey)
    colours[np.isnan(w.values)] = 1
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.subplots_adjust(left=0.16, right=0.99, top=0.98, bottom=0.07)
    ax.imshow(colours, aspect="auto", interpolation="nearest")
    regions = st.REGION[w.index].values
    edges = [i for i in range(1, len(regions)) if regions[i] != regions[i - 1]]
    for e in edges:
        ax.axhline(e - .5, color="white", lw=2)
    starts = [0] + edges
    ends = edges + [len(regions)]
    ax.set_yticks([(a + b - 1) / 2 for a, b in zip(starts, ends)], [regions[a] for a in starts], fontsize=8, color=MUTED)
    months = [i for i, d in enumerate(w.columns) if d.day == 1]
    ax.set_xticks(months, [w.columns[i].strftime("%b") for i in months], color=MUTED)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(SITE / "stations.png", dpi=200, facecolor="white")
    plt.close(fig)


def summary(p, share, history, year, today):
    last = p.DATE.max()
    day = (last - pd.Timestamp(f"{year}-05-01")).days
    past = to_date(history, year, day)
    per_region = share.sum().round(1)
    table = {}
    for region in sorted(per_region.index):
        g = p[p.region == region]
        smoky = g[g.smoke]  # on the rule's days only: one high reading alone in an unverified file may be a fault
        worst = smoky.loc[smoky.pm.idxmax()] if len(smoky) else None
        table[region] = {"smoke_days": float(per_region[region]), "median_past": float(past[region].median().round(1)), "highest_past": float(past[region].max().round(1)),
                         "highest_past_year": int(past[region].idxmax()), "stations": int(g.st.nunique()),
                         **({"peak": round(float(worst.pm)), "peak_station": worst.st, "peak_date": worst.DATE.date().isoformat()} if worst is not None else {"peak": None, "peak_station": "", "peak_date": ""})}
    total = float(share.mean(axis=1).sum())
    past_all = past.mean(axis=1)
    latest = p[p.DATE == last]
    return {"year": year, "data_through": last.date().isoformat(), "smoke_days": round(total, 1), "median_past": round(float(past_all.median()), 1),
            "rank": int((past_all > total).sum()) + 1, "years_compared": int(len(past_all)), "stations_reporting": int(p[p.DATE > last - pd.Timedelta(days=7)].st.nunique()),
            "latest_flagged": int(latest.smoke.sum()), "latest_reporting": int(len(latest)), "latest_high": int((latest.pm > HIGH).sum()),
            "season_over": last.month == 10 and last.day >= 31 or pd.Timestamp(today).month >= 11, "regions": table}


def day_text(text):
    d = pd.Timestamp(text)
    return f"{d:%B} {d.day}, {d.year}"


def short_day(text):
    d = pd.Timestamp(text)
    return f"{d:%b} {d.day}"


def peak(v):
    return f"{v['peak']} <span>{v['peak_station']}, {short_day(v['peak_date'])}</span>" if v["peak"] is not None else "None yet"


def page(s, updated):
    rows = "".join(f"<tr><th scope=\"row\">{r}</th><td>{v['smoke_days']:.0f}</td><td>{v['median_past']:.0f}</td><td>{v['highest_past']:.0f} <span>({v['highest_past_year']})</span></td><td>{peak(v)}</td></tr>" for r, v in s["regions"].items())
    state = (f"The {s['year']} season is over; this page next changes in May." if s["season_over"]
             else f"On {day_text(s['data_through'])}, {s['latest_flagged']} of {s['latest_reporting']} reporting stations were flagged as smoke and {s['latest_high']} were above 25 µg/m³.")
    rank = "the most of" if s["rank"] == 1 else f"number {s['rank']} of"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'none'; style-src 'self'; img-src 'self'; connect-src 'none'; object-src 'none'; base-uri 'self'; form-action 'none'" />
  <title>Wildfire smoke in BC: {s['year']} season | Ata Malfuzi</title>
  <meta name="description" content="Smoke days at BC's air monitors this season against 2014 to 2025, by region, from BC's unverified hourly data." />
  <link rel="canonical" href="{URL}" />
  <link rel="stylesheet" href="style.css" />
</head>
<body>
<main>
  <p class="eyebrow"><a href="https://atamalfuzi.github.io/">Ata Malfuzi</a> · Seasonal dashboard</p>
  <h1>Wildfire smoke in BC: {s['year']} season</h1>
  <p class="meta">Data through {day_text(s['data_through'])} · updated {day_text(updated)} · refreshes daily from May to October</p>
  <p class="links"><a href="{WRITE_UP}">Full write-up</a><a href="https://github.com/atamalfuzi/bc_wildfire_smoke">Code on GitHub</a><a href="status.json">Latest numbers (JSON)</a></p>
  <p class="note"><strong>These readings are unverified.</strong> BC checks and corrects its monitor data once a year, so numbers here can change. A smoke day is a rule's call from PM2.5 alone, not a measurement of smoke, and it says nothing about anyone's health. For current air quality advice, see <a href="https://www.env.gov.bc.ca/epd/bcairquality/readings/find-stations-map.html">BC's air quality site</a>.</p>
  <div class="cards">
    <section class="card"><h2>Smoke days per station so far</h2><p class="big">{s['smoke_days']:.0f}</p><p>Each region counted equally</p></section>
    <section class="card"><h2>Usual by this date</h2><p class="big">{s['median_past']:.0f}</p><p>Median of the {s['years_compared']} years 2014 to 2025</p></section>
    <section class="card"><h2>Rank among those years</h2><p class="big">{s['rank']} of {s['years_compared'] + 1}</p><p>{rank.capitalize()} {s['years_compared'] + 1} seasons to this date</p></section>
    <section class="card"><h2>Stations reporting</h2><p class="big">{s['stations_reporting']}</p><p>In the last seven days</p></section>
  </div>
  <p>{state}</p>

  <h2>The season so far</h2>
  <figure><img src="season.png" alt="Lines of cumulative smoke days per station from May to October for each year from 2014 to 2025 in grey, with {s['year']} in blue reaching {s['smoke_days']:.0f} days." width="1600" height="760" />
    <figcaption>Smoke days per station, added up from May 1. {s['year']} in blue, 2014 to 2025 in grey.</figcaption></figure>

  <h2>By region</h2>
  <p>Smoke days per station in each of BC's air monitoring regions, against the same dates in 2014 to 2025.</p>
  <div class="scroll"><table>
    <thead><tr><th scope="col">Region</th><th scope="col">{s['year']} so far</th><th scope="col">Median by this date</th><th scope="col">Highest by this date</th><th scope="col">Highest smoke day this season, µg/m³</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>

  <h2>Every station, every day</h2>
  <figure><img src="stations.png" alt="Grid of stations by day this season, grouped by region. Days flagged as smoke are orange to dark red by PM2.5 level, and others are grey." width="1600" height="1000" loading="lazy" />
    <figcaption>One row per station, grouped by region. Orange to dark red: flagged as smoke, darker for more PM2.5, up to 100 µg/m³. Grey: not flagged. White: no reading.</figcaption></figure>

  <h2>How a day counts</h2>
  <p>From May to October, a station-day counts as smoke if it is at least 8 µg/m³ above that station's usual level for the time of year, and at least half of the reporting stations within 200 km are up by 4 µg/m³ or more. Usual levels come from the verified record for 2005 to 2025. The <a href="{WRITE_UP}">write-up</a> checks this rule against BC's own list and satellite smoke maps and sets out its limits.</p>
  <footer>
    <p>Personal project: my own analysis of publicly available data. It is not associated with my work for any current or past employer and does not represent their views.</p>
    <p>Contains information licensed under the Open Government Licence - British Columbia. Unverified hourly data from the BC Ministry of Environment.</p>
  </footer>
</main>
</body>
</html>
"""


CSS = """:root { --text: #17201d; --muted: #4b5651; --rule: #e1e7e4; --accent: #0b6e5f; --bg-2: #f4f6f5; }
* { box-sizing: border-box; margin: 0; }
body { font-family: system-ui, sans-serif; color: var(--text); background: #fff; line-height: 1.55; }
main { max-width: 1000px; margin: 0 auto; padding: 2rem 1.25rem 3rem; }
h1 { font-family: Georgia, serif; font-size: clamp(1.7rem, 4vw, 2.3rem); line-height: 1.2; margin: .3rem 0 .5rem; }
h2 { font-family: Georgia, serif; font-size: 1.3rem; margin: 2rem 0 .5rem; }
a { color: var(--accent); }
.eyebrow, .meta, footer { color: var(--muted); font-size: .9rem; }
.links { display: flex; flex-wrap: wrap; gap: .4rem 1.2rem; margin: .8rem 0 1.5rem; font-weight: 600; }
.note { border-left: 3px solid var(--accent); background: var(--bg-2); padding: .7rem 1rem; margin: 0 0 1rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr)); gap: 1rem; margin-bottom: 1rem; }
.card { border: 1px solid var(--rule); border-radius: 8px; padding: 1rem 1.2rem; background: var(--bg-2); }
.card h2 { margin: 0 0 .3rem; font-family: inherit; font-size: .85rem; font-weight: 400; color: var(--muted); }
.card .big { font-size: 1.45rem; font-weight: 700; font-variant-numeric: tabular-nums; color: var(--text); }
.card p { font-size: .8rem; color: var(--muted); }
p + p { margin-top: .5rem; }
figure { margin: .5rem 0 0; }
figcaption { color: var(--muted); font-size: .8rem; margin-top: .3rem; }
img { width: 100%; height: auto; display: block; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; margin: .5rem 0 1rem; }
th, td { text-align: right; padding: .45rem .6rem; border-bottom: 1px solid var(--rule); }
thead th { vertical-align: bottom; font-size: .85rem; }
th:first-child { text-align: left; font-weight: 400; padding-left: 0; }
td:last-child, th:last-child { padding-right: 0; }
td span, th span { color: var(--muted); font-size: .85rem; font-weight: 400; }
footer { border-top: 1px solid var(--rule); margin-top: 2rem; padding-top: 1rem; display: grid; gap: .5rem; }
@media (max-width: 480px) { th, td { padding: .45rem .3rem; font-size: .9rem; } thead th { font-size: .8rem; } }
"""


def main():
    st = pd.read_csv(OUTPUT / "stations.csv", index_col=0)
    base = pd.read_csv(OUTPUT / "usual_level.csv", index_col=0)
    history = pd.read_csv(OUTPUT / "season_history.csv")
    offsets = pd.read_csv(OUTPUT / "instrument_offsets.csv")
    today = dt.date.today()
    raw = data.year_to_date()
    p, year = season_panel(raw, st, base, offsets, data.station_merges())
    issues = problems(p, year, today, published_through(), faulty_share(raw))
    if issues:
        raise SystemExit("Not publishing: " + "; ".join(issues))

    SITE.mkdir(exist_ok=True)
    share = daily_share(p)
    s = summary(p, share, history, year, today)
    line_figure(history, share, year)
    strip_figure(p, st, year)
    (SITE / "index.html").write_text(page(s, today.isoformat()), encoding="utf-8")
    (SITE / "style.css").write_text(CSS, encoding="utf-8")
    status = {"updated": today.isoformat(), **s}
    (SITE / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in status.items() if k != "regions"}, indent=2))


if __name__ == "__main__":
    main()
