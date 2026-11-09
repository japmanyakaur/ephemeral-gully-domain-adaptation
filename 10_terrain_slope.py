"""Terrain slope of the Hisar-Fatehabad-Sirsa study area.

Measures how flat the study area is, to explain why ephemeral gullies are rare there.

Data: AWS Terrain Tiles (Terrarium PNG encoding, mostly SRTM ~30 m in this region),
      https://registry.opendata.aws/terrain-tiles/  (free, no account needed)
Area: the dataset's search box (about lat 29.06-29.69, lon 75.00-75.87) intersected with
      Hisar, Fatehabad and Sirsa districts (same district file as the rebuild notebook).
Method: zoom-12 tiles (~33 m pixels), two 5x5 box blurs to suppress SRTM noise, slope from
      elevation differences 5 pixels either side (~330 m baseline), sampled every 3 pixels.

Outputs (in --out):
  terrain_slope_summary.csv   median / p90 / p99 / max slope (%) and % of area above 1, 2, 3 %
  terrain_slope_map.png       slope map with the district outline and the 300 dataset locations
  steepest_spots.csv          centres of the steepest 1 km cells, to check in satellite imagery

Usage:  python 10_terrain_slope.py --out results/terrain
Needs internet. Runtime about 1-2 minutes.
"""
import argparse
import io
import math
import os

import numpy as np
import pandas as pd
import requests
from PIL import Image

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
DISTRICTS_URL = "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson"
KEEP = {"Hisar", "Fatehabad", "Sirsa"}
Z = 12
FETCH_BOX = dict(lon_min=74.98, lon_max=75.90, lat_min=29.03, lat_max=29.72)   # tiles to download
AREA_BOX = dict(lon_min=75.00, lon_max=75.87, lat_min=29.06, lat_max=29.69)    # dataset search box (approx.)


def lon2x(lon, z=Z):
    return (lon + 180) / 360 * 2 ** z


def lat2y(lat, z=Z):
    r = math.radians(lat)
    return (1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2 * 2 ** z


def fetch_tile(x, y, z=Z):
    r = requests.get(TILE_URL.format(z=z, x=x, y=y), timeout=60)
    r.raise_for_status()
    a = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=np.float64)
    return a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768      # Terrarium decoding, metres


def mosaic(fetch=fetch_tile):
    x0, x1 = int(lon2x(FETCH_BOX["lon_min"])), int(lon2x(FETCH_BOX["lon_max"]))
    y0, y1 = int(lat2y(FETCH_BOX["lat_max"])), int(lat2y(FETCH_BOX["lat_min"]))
    E = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256))
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            E[(ty - y0) * 256:(ty - y0 + 1) * 256, (tx - x0) * 256:(tx - x0 + 1) * 256] = fetch(tx, ty)
    return E, x0, y0


def box_blur(A, r=2):
    """Mean over a (2r+1)x(2r+1) window; edges keep their original values."""
    k = 2 * r + 1
    c = np.pad(A, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    B = A.copy()
    B[r:-r, r:-r] = s / (k * k)
    return B


def pixel_centres(shape, x0, y0, z=Z):
    h, w = shape
    n = 2 ** z
    lon = (x0 * 256 + np.arange(w)) / (256 * n) * 360 - 180
    m = math.pi - 2 * math.pi * (y0 * 256 + np.arange(h)) / (256 * n)
    lat = np.degrees(np.arctan(np.sinh(m)))
    return lat, lon


def slope_percent(S, lat, d=5):
    """Slope (%) from differences d pixels either side; NaN near the edges."""
    ps = 40075016.686 / (256 * 2 ** Z) * np.cos(np.radians(lat))[:, None]   # metres per pixel per row
    gx = np.full(S.shape, np.nan)
    gy = np.full(S.shape, np.nan)
    gx[:, d:-d] = (S[:, 2 * d:] - S[:, :-2 * d]) / (2 * d * ps)
    gy[d:-d, :] = (S[2 * d:, :] - S[:-2 * d, :]) / (2 * d * ps[d:-d, :])
    return np.hypot(gx, gy) * 100


def district_mask(lat, lon, step, geojson=None):
    from shapely.geometry import Point, shape
    from shapely.ops import unary_union
    from shapely.prepared import prep
    dj = geojson or requests.get(DISTRICTS_URL, timeout=300).json()
    geoms = [shape(f["geometry"]) for f in dj["features"]
             if f["properties"].get("NAME_1") == "Haryana" and f["properties"].get("NAME_2") in KEEP]
    union = unary_union(geoms)
    pu = prep(union)
    rows, cols = np.arange(0, len(lat), step), np.arange(0, len(lon), step)
    M = np.zeros((len(lat), len(lon)), bool)
    box = AREA_BOX
    for j in rows:
        if not box["lat_min"] <= lat[j] <= box["lat_max"]:
            continue
        for i in cols:
            if box["lon_min"] <= lon[i] <= box["lon_max"] and pu.contains(Point(lon[i], lat[j])):
                M[j, i] = True
    return M, union


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/terrain")
    ap.add_argument("--metadata", default="hisar_dataset_final/metadata.csv")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Downloading elevation tiles ...")
    E, x0, y0 = mosaic()
    S = box_blur(box_blur(E))
    lat, lon = pixel_centres(E.shape, x0, y0)
    slope = slope_percent(S, lat)

    print("Keeping points inside the three districts and the dataset box ...")
    M, union = district_mask(lat, lon, step=3)
    v = slope[M]
    v = v[np.isfinite(v)]
    summary = {
        "samples": len(v),
        "median_slope_pct": round(float(np.median(v)), 2),
        "p90_slope_pct": round(float(np.percentile(v, 90)), 2),
        "p99_slope_pct": round(float(np.percentile(v, 99)), 2),
        "max_slope_pct": round(float(v.max()), 2),
        "pct_area_above_1pct": round(float((v > 1).mean() * 100), 2),
        "pct_area_above_2pct": round(float((v > 2).mean() * 100), 2),
        "pct_area_above_3pct": round(float((v > 3).mean() * 100), 2),
    }
    pd.DataFrame([summary]).to_csv(f"{args.out}/terrain_slope_summary.csv", index=False)
    print(summary)

    # steepest ~1 km cells (30 x 30 px), to check in satellite imagery
    cells = []
    C = 30
    for j in range(0, slope.shape[0] - C, C):
        for i in range(0, slope.shape[1] - C, C):
            m = M[j:j + C, i:i + C]
            if m.sum() < 20:
                continue
            s = slope[j:j + C, i:i + C][m]
            cells.append(dict(lat=round(float(lat[j + C // 2]), 4), lon=round(float(lon[i + C // 2]), 4),
                              mean_slope_pct=round(float(np.nanmean(s)), 2)))
    pd.DataFrame(cells).sort_values("mean_slope_pct", ascending=False).head(40) \
        .to_csv(f"{args.out}/steepest_spots.csv", index=False)

    # map
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    j0, j1 = np.searchsorted(-lat, -AREA_BOX["lat_max"]), np.searchsorted(-lat, -AREA_BOX["lat_min"])
    i0, i1 = np.searchsorted(lon, AREA_BOX["lon_min"]), np.searchsorted(lon, AREA_BOX["lon_max"])
    full = np.where(np.isfinite(slope), slope, np.nan)[j0:j1, i0:i1]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.imshow(full, extent=[lon[i0], lon[i1 - 1], lat[j1 - 1], lat[j0]], cmap="magma_r", vmin=0, vmax=3)
    for g in getattr(union, "geoms", [union]):
        xs, ys = g.exterior.xy
        ax.plot(xs, ys, color="#1971c2", lw=1)
    if os.path.exists(args.metadata):
        md = pd.read_csv(args.metadata).drop_duplicates("tile_id")
        ax.scatter(md["lon"], md["lat"], s=6, c="#2b8a3e", label="dataset locations (300)")
        ax.legend(loc="lower right")
    ax.set_xlim(AREA_BOX["lon_min"], AREA_BOX["lon_max"])
    ax.set_ylim(AREA_BOX["lat_min"], AREA_BOX["lat_max"])
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Terrain slope, Hisar–Fatehabad–Sirsa (median {summary['median_slope_pct']}%)")
    fig.colorbar(im, ax=ax, label="Slope (%), SRTM-based, smoothed")
    fig.tight_layout()
    fig.savefig(f"{args.out}/terrain_slope_map.png", dpi=200)
    print(f"Saved to {args.out}/")


if __name__ == "__main__":
    main()
