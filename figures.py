"""Draw the static charts from the tables in output/."""
import json
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import data
from data import DATA, FIGURES, OUTPUT, STUDY_START

INK, MUTED, GREY, GRID = "#0b0b0b", "#52514e", "#a8a7a2", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SHED_COLOURS = {"South coast": BLUE, "Mid-island and Sunshine Coast": "#6250d6", "Southern Interior and Kootenays": ORANGE, "Central Interior": YELLOW,
                "Peace and Nechako": AQUA, "Bulkley Valley": "#e87ba4", "North coast": "#008300"}
SOURCE_COLOURS = {"BC": BLUE, "Alberta, NWT, Yukon": ORANGE, "Washington, Idaho, Montana": AQUA, "Oregon, California": YELLOW, "Elsewhere": GREY}
RAMP = LinearSegmentedColormap.from_list("blues", ["#f3f7fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
                     "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "legend.frameon": False,
                     "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.titlesize": 12})
SOURCE = "Source: BC Ministry of Environment verified hourly air quality data."


def save(fig, name, title, subtitle, caption):
    fig.text(0.015, 0.985, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.015, 0.925, subtitle, color=MUTED, va="top")
    fig.text(0.015, 0.012, "\n".join(textwrap.wrap(caption, 138)), color=MUTED, fontsize=9, va="bottom")
    fig.savefig(FIGURES / name, dpi=200, facecolor="white")
    plt.close(fig)


def with_without(y):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.15)
    ax.fill_between(y.index, y.without, y.measured, color=ORANGE, alpha=0.14, lw=0)
    ax.plot(y.index, y.measured, color=ORANGE, lw=2, marker="o", ms=4, label="As measured")
    ax.plot(y.index, y.without, color=BLUE, lw=2, marker="o", ms=4, label="Smoke days set back to the usual level")
    ax.set_ylim(0, 12.5)
    ax.set_xticks(range(2005, 2026, 2))
    ax.set_ylabel("Annual mean PM2.5, µg/m³")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)
    for yr in y.share_8.nlargest(3).index:
        ax.annotate(f"{y.share_8[yr]:.0%} smoke", (yr, y.measured[yr]), xytext=(-34 if yr == 2017 else 0, 8), textcoords="offset points", ha="center", color=MUTED)
    save(fig, "with_without.png", "Smoke sets the bad years, and hides a slow fall in the rest",
         "Annual PM2.5 across BC, as measured and with smoke days set back to each station's usual level",
         "Station and year effects, regions weighted equally, 2005 to 2025. Readings put on one instrument's footing. " + SOURCE)


def region_table(table, name, title, subtitle, caption, fmt, vmax):
    fig, ax = plt.subplots(figsize=(10, 5.2 if len(table) > 5 else 4))
    fig.subplots_adjust(left=0.26, right=0.985, top=0.83, bottom=0.14)
    ax.imshow(table.values, aspect="auto", cmap=RAMP, vmin=0, vmax=vmax)
    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns])
    ax.set_yticks(range(len(table)), table.index)
    ax.grid(False)
    ax.tick_params(length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks(np.arange(-.5, table.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, len(table)), minor=True)
    ax.grid(which="minor", color="white", lw=2)
    ax.tick_params(which="minor", length=0)
    for i in range(len(table)):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=10, color="white" if v >= vmax * .42 else MUTED)
    save(fig, name, title, subtitle, caption)


def trap(y, fixed, regions):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(left=0.07, right=0.8, top=0.83, bottom=0.15)
    a, b = y.share_8.loc[STUDY_START:] * 100, fixed.share * 100
    ax.plot(a.index, a, color=ORANGE, lw=2, marker="o", ms=4, label="All stations, regions weighted equally")
    ax.plot(b.index, b, color=BLUE, lw=2, marker="o", ms=4, label="Only stations reporting every year")
    ax.set_ylabel("Smoke share of annual PM2.5, %")
    ax.set_ylim(0, 40)
    ax.set_xticks(range(2014, 2026))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)
    for s, c, off, label in ((a, ORANGE, 2, "All stations"), (b, BLUE, -2, "Stations reporting\nevery year")):
        ax.annotate(label, (2025, s[2025]), xytext=(8, off), textcoords="offset points", color=c, va="center")
    save(fig, "station_trap.png", "Which stations you keep changes the answer",
         f"In 2023 smoke was {a[2023]:.0f}% of annual PM2.5 across BC, or {b[2023]:.0f}% on the stations that never stopped",
         f"The {regions.sum()} stations reporting in all twelve years: {regions.get('Lower Mainland', 0)} in the Lower Mainland, {regions.get('Cariboo', 0)} in the Cariboo, {regions.get('Omineca-Peace', 0)} in Omineca-Peace. " + SOURCE)


def validation(p, polygons, removed):
    h = p[p.year >= STUDY_START].merge(polygons, left_on=["raw", "DATE"], right_on=["STATION_NAME", "DATE"])
    edges = [-99, 5, 10, 15, 25, 50, 1e5]
    labels = ["Up to 5", "5 to 10", "10 to 15", "15 to 25", "25 to 50", "Over 50"]
    share = h.groupby(pd.cut(h["mean"], edges, labels=labels), observed=False).hms.apply(lambda s: (s >= 1).mean()) * 100
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.17, wspace=0.22)
    ax.bar(labels, share, color=BLUE, width=.8)
    for x, v in zip(labels, share):
        ax.annotate(f"{v:.0f}%", (x, v), xytext=(0, 3), textcoords="offset points", ha="center", color=MUTED)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlabel("Daily PM2.5 at the station, µg/m³")
    ax.set_ylabel("Station-days under a smoke polygon, %")
    ax.grid(axis="x", visible=False)
    ax.set_title("Satellite smoke maps, July to September")
    bx.plot(removed.index, removed.rule, color=ORANGE, lw=2, marker="o", ms=4, label="This rule")
    bx.plot(removed.index, removed.bc, color=BLUE, lw=2, marker="o", ms=4, label="BC's list of flagged days")
    bx.set_xticks(range(2014, 2026, 2))
    bx.set_ylabel("Taken off annual PM2.5, µg/m³")
    bx.set_ylim(0, removed.rule.max() * 1.3)
    bx.grid(axis="x", visible=False)
    bx.legend(loc="upper right")
    bx.set_title("BC's own list, same stations")
    save(fig, "validation.png", "Two independent checks on the smoke rule", "The rule never sees either of these", "Polygons from NOAA's Hazard Mapping System, drawn by analysts from satellite images. They show smoke anywhere overhead, so a clean day under one is common. "
         + SOURCE)


def instruments(o):
    o = o[o.instrument.isin(["TEOM", "T640", "BAM"])]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    fig.subplots_adjust(left=0.23, right=0.97, top=0.78, bottom=0.2)
    rows = [("TEOM", "warm"), ("TEOM", "cold"), ("BAM", "warm"), ("BAM", "cold"), ("T640", "warm"), ("T640", "cold")]
    t = o.set_index(["instrument", "season"])
    labels = {"warm": "May to Oct", "cold": "Nov to Apr"}
    for i, key in enumerate(rows):
        r = t.loc[key]
        colour = ORANGE if key[1] == "warm" else BLUE
        ax.plot([r.low, r.high], [i, i], color=colour, lw=2, zorder=2)
        ax.plot(r.offset, i, "o", color=colour, ms=8, mec="white", mew=1.5, zorder=3)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(rows)), [f"{k} {labels[s]}" for k, s in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Reads above or below the SHARP on a station's seasonal median, µg/m³")
    ax.grid(axis="y", visible=False)
    save(fig, "instruments.png", "Monitors changed, and they don't read alike", "Older TEOMs read about 2 µg/m³ lower, about a third of a typical year's level",
         "From stations that changed instrument, with station and year effects. Lines are 90% ranges, resampling stations. " + SOURCE)


def smoke_days(y):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.14)
    ax.bar(y.index, y.smoke_days, width=.8, color=[BLUE if v else GREY for v in y.index >= 2017])
    ax.set_xticks(range(2005, 2026, 2))
    ax.set_ylabel("Smoke days per station")
    ax.grid(axis="x", visible=False)
    before, after = y.smoke_days[y.index < 2017].mean(), y.smoke_days[y.index >= 2017].mean()
    ax.annotate(f"2005 to 2016: {before:.0f} a year", (2010.5, y.smoke_days[y.index < 2017].max()), xytext=(0, 6), textcoords="offset points", ha="center", color=MUTED)
    ax.annotate(f"2017 to 2025: {after:.0f} a year", (2021, y.smoke_days.max()), xytext=(0, 4), textcoords="offset points", ha="center", color=MUTED)
    ax.set_ylim(0, y.smoke_days.max() * 1.15)
    save(fig, "smoke_days.png", "More smoke days since 2017, but only a few very bad years", "Smoke days per station each year",
         "Station and year effects, regions weighted equally. The count depends on the threshold more than the share does; this uses 8 µg/m³. The 2017 break was picked after seeing the data. " + SOURCE)


def burned(y):
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.18)
    ax.scatter(y.area_burned_ha / 1e3, y.smoke_days, s=46, color=BLUE, edgecolor="white", lw=1.5, zorder=3)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_xlabel("Area burned inside BC, thousand hectares (log scale)")
    ax.set_ylabel("Smoke days per station")
    for yr in y.index:
        ax.annotate(str(yr), (y.area_burned_ha[yr] / 1e3, y.smoke_days[yr]), xytext=(7, 3), textcoords="offset points", color=MUTED, fontsize=9)
    save(fig, "burned.png", "BC's fires track its smoke, with clear exceptions", f"2020 and 2022 had little burning in BC, yet smoke was {y.share_8[2020]:.0%} and {y.share_8[2022]:.0%} of annual PM2.5",
         "Source: BC Wildfire Service fire perimeters; BC Ministry of Environment. Perimeters are final footprints, so they say nothing about when a fire burned.")


def shed_map(st, sheds, outline_rings):
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.08)
    for ring in outline_rings:
        a = np.array(ring)
        ax.fill(a[:, 0], a[:, 1], color="#eef0ee", lw=.5, ec="#eef0ee")
    s = st.join(sheds)
    for name, colour in SHED_COLOURS.items():
        g = s[s.shed == name]
        ax.scatter(g.LONG, g.LAT, s=26, color=colour, edgecolor="white", lw=.7, label=name, zorder=3)
    ax.set_aspect(1 / np.cos(np.radians(54)))
    ax.axis("off")
    ax.set_xlim(-139.5, -114)
    ax.legend(loc="lower left", fontsize=10, title="Smoke shed", title_fontsize=10)
    save(fig, "sheds.png", "Smoke doesn't follow administrative lines", "Stations that get smoke on the same days, from July to September, 2014 to 2025",
         "Average-linkage clusters of station correlation; stations with too few summers join their nearest neighbour's shed. " + SOURCE)


def dates(start, end):
    """May 16 to 19, 2023, or Aug 27 to Sep 11, 2025."""
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    last = f"{b.day}" if a.month == b.month else f"{b:%b} {b.day}"
    return f"{a:%b} {a.day} to {last}, {a.year}"


def episode_chart(e):
    e = e.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7.4))
    fig.subplots_adjust(left=0.2, right=0.98, top=0.83, bottom=0.14)
    for i, (_, r) in enumerate(e.iterrows()):
        left = 0
        for name, colour in SOURCE_COLOURS.items():
            v = r.get(name)
            if pd.notna(v) and v > 0:
                ax.barh(i, v * 100, left=left, color=colour, height=.76, edgecolor="white", lw=1.5)
                left += v * 100
    labels = [dates(r.start, r.end) for _, r in e.iterrows()]
    ax.set_yticks(range(len(e)), labels)
    ax.set_xlim(0, 100)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of satellite fire power within 1,500 km of the stations hit, %")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SOURCE_COLOURS.values()]
    ax.legend(handles, SOURCE_COLOURS.keys(), loc="upper center", bbox_to_anchor=(.42, 1.08), ncol=5, fontsize=9, handlelength=1, columnspacing=1)
    save(fig, "episodes.png", "The 15 largest smoke episodes: where the fire was", "Largest at the top. This shows where fires were burning, not where a day's smoke came from",
         "Fire power from three days before each episode to its end. Sensors see large fires more than small ones, and cloud or thick smoke hides some. Source: Natural Resources Canada hotspots; BC Ministry of Environment.")


def main():
    y = pd.read_csv(OUTPUT / "by_year.csv", index_col=0)
    p = pd.read_pickle(DATA / "panel.pkl")
    st = pd.read_csv(OUTPUT / "stations.csv", index_col=0)
    sheds = pd.read_csv(OUTPUT / "sheds.csv", index_col=0)
    sy = pd.read_csv(OUTPUT / "station_years.csv")
    with_without(y)
    regions = sy[sy.year >= STUDY_START].groupby(["region", "year"])[["measured", "without"]].mean()
    share = (1 - regions.without / regions.measured).unstack() * 100
    order = ["Omineca-Peace", "Skeena", "Cariboo", "Southern Interior", "Kootenay", "Lower Mainland", "Vancouver Island"]
    region_table(share.loc[order], "regions.png", "The worst place changes each year", "Smoke's share of annual PM2.5 by region, %",
                 "Average of stations with at least 274 valid days. Regions are BC's air monitoring regions. " + SOURCE, lambda v: f"{v:.0f}", 60)
    fixed = pd.read_csv(OUTPUT / "fixed_panel.csv", index_col=0)
    counts = sy[sy.year >= STUDY_START]
    full = counts.groupby("st").year.nunique()
    trap(y, fixed, counts[counts.st.isin(full[full == 12].index)].drop_duplicates("st").region.value_counts())
    validation(p, pd.read_csv(DATA / "hms.csv", parse_dates=["DATE"]), pd.read_csv(OUTPUT / "removed_vs_bc.csv", index_col=0))
    instruments(pd.read_csv(OUTPUT / "instrument_offsets.csv"))
    smoke_days(y)
    burned(y.loc[STUDY_START:])
    rings = [poly for f in data.outline() for poly in (f["geometry"]["coordinates"] if f["geometry"]["type"] == "Polygon" else sum(f["geometry"]["coordinates"], []))]
    shed_map(st, sheds, rings)
    people = pd.read_csv(OUTPUT / "people_by_shed.csv")
    people["names"] = people.shed
    t = people.assign(per=people.person_days / people.people).pivot(index="shed", columns="year", values="per").loc[list(SHED_COLOURS)]
    region_table(t, "people.png", "Days above 25 µg/m³ for the average person, by smoke shed", "Air at the nearest monitor, on days the rule calls smoke, May to October",
                 "Each community's population takes the count at its nearest monitor reporting that year. This is air at a monitor, not what anyone breathed. Source: BC Stats population; BC Ministry of Environment.", lambda v: f"{v:.0f}", 45)
    episode_chart(pd.read_csv(OUTPUT / "episodes.csv"))


if __name__ == "__main__":
    main()
