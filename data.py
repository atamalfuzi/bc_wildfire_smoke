"""Download and cache the open data this project uses. Everything lands in data/, which git ignores.

Run from the repository folder: python data.py
About 3 GB passes through (21 years of hourly PM2.5, 12 years of satellite hotspots) but only daily summaries are kept,
about 100 MB. The NOAA smoke polygons come last and take about an hour, because the server drops quick requests.
"""
from pathlib import Path
import datetime as dt
import io
import json
import re
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

DATA, OUTPUT, FIGURES = Path("data"), Path("output"), Path("figures")
for folder in (DATA, DATA / "daily", DATA / "hotspots", OUTPUT, FIGURES):
    folder.mkdir(exist_ok=True)

AIR = "ftp://ftp.env.gov.bc.ca/pub/outgoing/AIR"
WFS = "https://openmaps.gov.bc.ca/geo/pub/ows?service=WFS&version=2.0.0&request=GetFeature&outputFormat=json&srsName=EPSG:4326&"
HOTSPOTS = "https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/{}_hotspots.zip"
HMS = "https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile/{y}/{m:02d}/hms_smoke{d:%Y%m%d}.zip"
POPULATION = "https://catalogue.data.gov.bc.ca/dataset/86839277-986a-4a29-9f70-fa9b1166f6cb/resource/0e15d04d-127c-457a-b999-20800c929927/download/municipality-population.csv"
FIRST_YEAR, LAST_YEAR = 2005, 2025   # verified PM2.5 files read; verified data reach 2025 as of October 2026
STUDY_START = 2014                   # BC's list of flagged days starts here and most monitors have modern instruments
HOTSPOT_YEARS = range(STUDY_START, LAST_YEAR + 1)
HMS_YEARS = range(STUDY_START, LAST_YEAR + 1)


def fetch(url, path, tries=4):
    """Download once; later runs reuse the file."""
    path = Path(path)
    if path.exists() and path.stat().st_size:
        return path
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=300) as r:
                body = r.read()
            path.write_bytes(body)
            return path
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == tries - 1:
                raise
            print("retrying", url, repr(e)[:80])
            time.sleep(10 * (attempt + 1))


# Hourly air quality, kept as daily summaries ----
def daily_from_hourly(stream, date=None):
    """One row per station, instrument and day from an hourly CSV: mean, hours with a value, highest hour."""
    cols = ["DATE_PST", "STATION_NAME", "EMS_ID", "RAW_VALUE", "INSTRUMENT"]
    parts = []
    for c in pd.read_csv(stream, usecols=cols, chunksize=500_000, low_memory=False, encoding="latin-1"):
        c["DATE"] = c.DATE_PST.str[:10]
        c["RAW_VALUE"] = pd.to_numeric(c.RAW_VALUE, errors="coerce")
        c["ok"] = c.RAW_VALUE.notna()
        parts.append(c.groupby(["STATION_NAME", "EMS_ID", "INSTRUMENT", "DATE"], dropna=False).agg(
            total=("RAW_VALUE", "sum"), n=("ok", "sum"), hi=("RAW_VALUE", "max")).reset_index())
    g = pd.concat(parts).groupby(["STATION_NAME", "EMS_ID", "INSTRUMENT", "DATE"], dropna=False).agg(
        total=("total", "sum"), n=("n", "sum"), hi=("hi", "max")).reset_index()
    g["mean"] = g.total / g.n.where(g.n > 0)
    return g.drop(columns="total")


def daily(parameter, year):
    """Daily summary of one verified year. The raw file is about 100 MB and isn't kept."""
    path = DATA / "daily" / f"{parameter.lower()}_{year}.csv"
    if not path.exists():
        with urllib.request.urlopen(f"{AIR}/AnnualSummary/{year}/{parameter}.csv", timeout=600) as r:
            daily_from_hourly(r).to_csv(path, index=False)
        print("read", parameter, year)
    return pd.read_csv(path, low_memory=False)


def year_to_date():
    """Unverified hourly PM2.5 from the last validation date to now, for the dashboard. Fresh on every run."""
    with urllib.request.urlopen(f"{AIR}/Hourly_Raw_Air_Data/Year_to_Date/PM25.csv", timeout=600) as r:
        return daily_from_hourly(r)


# Reference tables ----
def clean_name(s):
    return s.astype(str).str.replace("\xa0", " ").str.replace("�", " ").str.strip()


def stations():
    path = fetch(f"{AIR}/Air_Monitoring_Stations/bc_air_monitoring_stations.csv", DATA / "stations.csv")
    s = pd.read_csv(path, encoding="latin-1").dropna(subset=["LAT", "LONG"]).drop_duplicates("STATION_NAME")
    s["REGION"] = s.REGION.astype(str).str[5:]
    return s.set_index("STATION_NAME")


NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def workbook(path):
    """Read every sheet of an .xlsx with the standard library: these are plain tables, so no openpyxl."""
    z = zipfile.ZipFile(path)
    shared = ["".join(t.text or "" for t in si.iter(NS + "t")) for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
    names = [s.get("name") for s in ET.fromstring(z.read("xl/workbook.xml")).find(NS + "sheets")]
    out = {}
    for i, name in enumerate(names, 1):
        rows = []
        for r in ET.fromstring(z.read(f"xl/worksheets/sheet{i}.xml")).iter(NS + "row"):
            row = {}
            for c in r:
                column = re.match(r"[A-Z]+", c.get("r")).group()
                v = c.find(NS + "v")
                if v is not None:
                    row[column] = shared[int(v.text)] if c.get("t") == "s" else v.text
            rows.append(row)
        f = pd.DataFrame(rows)
        out[name] = f[sorted(f.columns, key=lambda s: (len(s), s))]
    return out


def excel_date(s):
    return pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit="D", origin="1899-12-30")


def flagged_days():
    """BC's list of station-days with PM2.5 put down to transboundary flows or exceptional events, mostly wildfire smoke."""
    sheets = workbook(fetch(f"{AIR}/CAAQS/TFEE_list.xlsx", DATA / "TFEE_list.xlsx"))
    t = sheets["PM25 TFEE"].iloc[2:].copy()
    t.columns = ["station", "date", "fire"]
    t["station"] = clean_name(t.station)
    t["date"] = excel_date(t.date)
    return t.dropna(subset=["date"]).drop_duplicates(["station", "date"])


def station_merges():
    """BC's table of stations that moved, with the name to carry them under and the dates that apply."""
    sheets = workbook(fetch(f"{AIR}/CAAQS/BC_CAAQS_station_history.xlsx", DATA / "station_history.xlsx"))
    t = sheets["Station Merge"]
    t = t[t.iloc[:, 0].notna() & t.iloc[:, 1].notna()].iloc[1:].copy()
    t = t.iloc[:, :2]
    t.columns = ["station", "site"]
    t["station"] = clean_name(t.station)
    t["site"] = clean_name(t.site)
    return t.drop_duplicates("station").set_index("station").site


def official_annual():
    path = fetch(f"{AIR}/CAAQS/annual_results.csv", DATA / "annual_results.csv")
    c = pd.read_csv(path, low_memory=False)
    return c[(c.parameter == "PM25") & (c.metric == "annual")]


def wfs(type_name, path, **params):
    query = urllib.parse.urlencode({"typeNames": "pub:" + type_name, **params})
    return json.loads(fetch(WFS + query, path).read_text(encoding="utf8"))["features"]


def area_burned():
    """Hectares inside the final perimeters of fires in BC, by year."""
    rows = wfs("WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_FIRE_POLYS_SP", DATA / "perimeters.json", count=20000,
               propertyName="FIRE_YEAR,FIRE_SIZE_HECTARES", CQL_FILTER=f"FIRE_YEAR>={FIRST_YEAR}")
    f = pd.DataFrame([r["properties"] for r in rows])
    return f.groupby("FIRE_YEAR").FIRE_SIZE_HECTARES.sum()


def large_fires(min_ha=1000):
    """BC fires of at least min_ha, with start point, for the map."""
    rows = wfs("WHSE_LAND_AND_NATURAL_RESOURCE.PROT_HISTORICAL_INCIDENTS_SP", DATA / "incidents.json", count=20000,
               propertyName="FIRE_YEAR,FIRE_NUMBER,INCIDENT_NAME,IGNITION_DATE,CURRENT_SIZE,LATITUDE,LONGITUDE",
               CQL_FILTER=f"FIRE_YEAR>={STUDY_START} AND CURRENT_SIZE>={min_ha}")
    return pd.DataFrame([r["properties"] for r in rows])


def outline():
    """BC's natural resource regions, generalized: filled alike they draw the province."""
    return wfs("WHSE_ADMIN_BOUNDARIES.ADM_NR_REGIONS_SPG", DATA / "regions.json")


def census_subdivisions():
    """Census subdivision polygons, for where people live."""
    return wfs("WHSE_HUMAN_CULTURAL_ECONOMIC.CEN_CENSUS_SUBDIVSIONS_SVW", DATA / "csd.json")


def population():
    p = pd.read_csv(fetch(POPULATION, DATA / "population.csv"), usecols=["Region", "Region.Name", "Year", "Gender", "Total"])
    return p[(p.Gender == "T") & (p.Region > 0)].rename(columns={"Region.Name": "name"})


# Satellite fire detections, reduced ----
def hotspots(year):
    """Fire radiative power by day, half-degree cell and agency (province or US state) for one fire season."""
    path = DATA / "hotspots" / f"{year}.csv.gz"
    if not path.exists():
        z = zipfile.ZipFile(io.BytesIO(fetch(HOTSPOTS.format(year), DATA / f"hotspots_{year}.zip").read_bytes()))
        member = next(n for n in z.namelist() if n.lower().endswith(".csv") and "hotspots" in n.lower())
        parts = []
        for c in pd.read_csv(z.open(member), usecols=["rep_date", "lat", "lon", "frp", "agency"], chunksize=1_000_000, low_memory=False):
            c["date"] = c.rep_date.str[:10]
            c = c[c.date.str[5:7].isin(["05", "06", "07", "08", "09", "10"]) & c.lat.between(40, 70) & c.lon.between(-150, -100)]
            c["la"], c["lo"] = (c.lat * 2).round() / 2, (c.lon * 2).round() / 2
            parts.append(c.groupby(["date", "la", "lo", "agency"]).frp.agg(["sum", "size"]).reset_index())
        g = pd.concat(parts).groupby(["date", "la", "lo", "agency"]).sum().reset_index().rename(columns={"sum": "frp", "size": "detections"})
        g.round({"frp": 1}).to_csv(path, index=False)
        (DATA / f"hotspots_{year}.zip").unlink()
        print("read hotspots", year, len(g))
    return pd.read_csv(path)


# NOAA smoke polygons ----
def dbf_density(b):
    """The density field of a .dbf file, one value per polygon. Hand-parsed, as there is no geopandas here."""
    n, header, record = struct.unpack("<I", b[4:8])[0], struct.unpack("<H", b[8:10])[0], struct.unpack("<H", b[10:12])[0]
    fields, p, offset = [], 32, 1
    while b[p] != 13:
        fields.append((b[p:p + 11].split(b"\0")[0].decode(), offset, b[p + 16]))
        offset += b[p + 16]
        p += 32
    found = [f for f in fields if f[0].lower() == "density"]
    if not found:
        return [None] * n
    _, o, length = found[0]
    return [b[header + i * record + o:header + i * record + o + length].decode("latin-1").strip() for i in range(n)]


def shp_polygons(b):
    """Polygons of a .shp file as lists of rings, each an array of (x, y)."""
    p, out = 100, []
    while p < len(b):
        length = struct.unpack(">i", b[p + 4:p + 8])[0] * 2
        r = b[p + 8:p + 8 + length]
        p += 8 + length
        if struct.unpack("<i", r[:4])[0] != 5:
            out.append([])
            continue
        parts, points = struct.unpack("<ii", r[36:44])
        starts = list(struct.unpack(f"<{parts}i", r[44:44 + 4 * parts]))
        xy = np.frombuffer(r, "<f8", points * 2, 44 + 4 * parts).reshape(-1, 2)
        out.append([xy[a:e] for a, e in zip(starts, starts[1:] + [points])])
    return out


def inside(xs, ys, ring):
    """Which points fall inside a closed ring, by ray casting."""
    x0, y0, x1, y1 = ring[:-1, 0], ring[:-1, 1], ring[1:, 0], ring[1:, 1]
    hit = np.zeros(len(xs), bool)
    for i, (x, y) in enumerate(zip(xs, ys)):
        crosses = (y0 > y) != (y1 > y)
        if crosses.any():
            hit[i] = ((x1[crosses] - x0[crosses]) * (y - y0[crosses]) / (y1[crosses] - y0[crosses]) + x0[crosses] > x).sum() % 2 == 1
    return hit


DENSITY = {"Light": 1, "Medium": 2, "Heavy": 3}


def polygons_over_stations(raw, lon, lat):
    """Thickest smoke polygon over each station on one day: 0 none, 1 light, 2 medium, 3 heavy."""
    z = zipfile.ZipFile(io.BytesIO(raw))
    shp = next(n for n in z.namelist() if n.endswith(".shp"))
    level = np.zeros(len(lon), int)
    for density, rings in zip(dbf_density(z.read(shp[:-3] + "dbf")), shp_polygons(z.read(shp))):
        for ring in rings:
            if ring[:, 1].max() < 48 or ring[:, 1].min() > 61 or ring[:, 0].max() < -140 or ring[:, 0].min() > -114:
                continue
            hit = inside(lon, lat, ring)
            level[hit] = np.maximum(level[hit], DENSITY.get(density, 1))
    return level


def smoke_polygons():
    """July to September polygons over every station, cached a day at a time so a dropped connection loses nothing."""
    path = DATA / "hms.csv"
    s = stations()
    names, lon, lat = s.index.values, s.LONG.values, s.LAT.values
    done = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["DATE", "STATION_NAME", "hms"])
    have = set(done.DATE)
    days = [d.date() for y in HMS_YEARS for d in pd.date_range(f"{y}-07-01", f"{y}-09-30")]
    for round_ in range(4):  # the server resets connections when hit quickly, so go around again for what it dropped
        missing, new = [d for d in days if d.isoformat() not in have], []
        if not missing:
            break
        for d in missing:
            time.sleep(1.5)
            try:
                raw = urllib.request.urlopen(HMS.format(y=d.year, m=d.month, d=d), timeout=60).read()
                new.append(pd.DataFrame({"DATE": d.isoformat(), "STATION_NAME": names, "hms": polygons_over_stations(raw, lon, lat)}))
                have.add(d.isoformat())
            except (urllib.error.URLError, zipfile.BadZipFile, TimeoutError, ConnectionError, StopIteration, struct.error):
                pass
            if len(new) == 50:
                done = pd.concat([done, *new])
                done.to_csv(path, index=False)
                new = []
        done = pd.concat([done, *new])
        done.to_csv(path, index=False)
        print("smoke polygons round", round_, "days", len(have), "of", len(days))
    return done


def main():
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        daily("PM25", year)
    for year in (2018, 2021, 2023):  # co-pollutant check
        daily("PM10", year)
        daily("CO", year)
    stations(), flagged_days(), station_merges(), official_annual(), area_burned(), large_fires(), outline(), census_subdivisions(), population()
    for year in HOTSPOT_YEARS:
        hotspots(year)
    print("ready, except polygons")
    smoke_polygons()


if __name__ == "__main__":
    main()
