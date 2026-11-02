"""The hand-written file readers, on made-up files."""
import io
import struct
import zipfile

import numpy as np

import data


def test_daily_from_hourly_averages_and_counts():
    text = "\n".join(['"DATE_PST","STATION_NAME","EMS_ID","RAW_VALUE","INSTRUMENT"',
                      '"2023-08-01 01:00","A","E1",10,"BAM1020"', '"2023-08-01 02:00","A","E1",20,"BAM1020"', '"2023-08-01 03:00","A","E1",,"BAM1020"',
                      '"2023-08-02 01:00","A","E1",5,"BAM1020"'])
    d = data.daily_from_hourly(io.BytesIO(text.encode())).set_index("DATE")
    assert d.loc["2023-08-01", "mean"] == 15
    assert d.loc["2023-08-01", "n"] == 2
    assert d.loc["2023-08-01", "hi"] == 20
    assert d.loc["2023-08-02", "mean"] == 5


def test_inside_a_square():
    ring = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]], float)
    hit = data.inside(np.array([5.0, 15.0, -1.0]), np.array([5.0, 5.0, 5.0]), ring)
    assert hit.tolist() == [True, False, False]


def polygon_shp(ring):
    points = len(ring)
    content = struct.pack("<i", 5) + struct.pack("<4d", 0, 0, 0, 0) + struct.pack("<ii", 1, points) + struct.pack("<i", 0) + np.asarray(ring, "<f8").tobytes()
    return b"\0" * 100 + struct.pack(">ii", 1, len(content) // 2) + content


def density_dbf(values):
    width = 8
    header = bytearray(32)
    header[4:8] = struct.pack("<I", len(values))
    header[8:10] = struct.pack("<H", 32 + 32 + 1)
    header[10:12] = struct.pack("<H", 1 + width)
    field = bytearray(32)
    field[:7] = b"Density"
    field[16] = width
    return bytes(header) + bytes(field) + b"\r" + b"".join(b" " + v.ljust(width).encode() for v in values)


def test_reads_density_and_rings_from_a_shapefile():
    ring = [[-125.0, 50.0], [-120.0, 50.0], [-120.0, 55.0], [-125.0, 55.0], [-125.0, 50.0]]
    assert data.dbf_density(density_dbf(["Heavy", "Light"])) == ["Heavy", "Light"]
    rings = data.shp_polygons(polygon_shp(ring))
    assert np.allclose(rings[0][0], ring)


def test_polygons_over_stations_takes_the_thickest():
    ring = [[-125.0, 50.0], [-120.0, 50.0], [-120.0, 55.0], [-125.0, 55.0], [-125.0, 50.0]]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("hms.shp", polygon_shp(ring))
        z.writestr("hms.dbf", density_dbf(["Medium"]))
    level = data.polygons_over_stations(buffer.getvalue(), np.array([-122.0, -110.0]), np.array([52.0, 52.0]))
    assert level.tolist() == [2, 0]


def test_simplify_keeps_corners_and_drops_points_along_a_line():
    from site_data import ring_area, simplify
    edge = [[x / 10, 0.0] for x in range(0, 101)]
    ring = edge + [[10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]
    kept = simplify(ring)
    assert len(kept) < 8
    assert [10.0, 10.0] in kept and [0.0, 10.0] in kept
    assert abs(ring_area(kept) - 100) < 1e-6
