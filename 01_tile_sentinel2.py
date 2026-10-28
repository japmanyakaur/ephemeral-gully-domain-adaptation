"""
Step 1: Cut 128x128 tiles from a Sentinel-2 TCI GeoTIFF.

Downloads the TCI (True Color Image) product at native 10m resolution
from allthingsspatial.io → "Download whole scene (true color)" (~364 MB).

This script:
  - Opens the full-scene GeoTIFF
  - Crops to an agricultural region (UTM coordinates)
  - Filters out urban/non-vegetated tiles
  - Saves 128x128 PNG tiles + metadata CSV

Usage:
    python 01_tile_sentinel2.py --input <TCI.tif> --output <output_dir>

Requirements:
    pip install rasterio numpy pillow
"""

import argparse
import csv
import os
import sys

import numpy as np
import rasterio
from PIL import Image


# ── DEFAULTS (Yamunanagar agricultural zone, UTM 43N) ────────────────
DEFAULT_CROP = {
    "xmin": 655000,
    "xmax": 685000,
    "ymin": 3340000,
    "ymax": 3370000,
}
TILE_SIZE = 128          # pixels per side
NUM_TILES = 50           # how many tiles to extract
MIN_VEG_FRAC = 0.30      # minimum vegetation fraction to keep tile


def parse_args():
    p = argparse.ArgumentParser(description="Tile a Sentinel-2 TCI GeoTIFF")
    p.add_argument("--input", required=True, help="Path to TCI GeoTIFF")
    p.add_argument("--output", default="haryana_tiles", help="Output directory")
    p.add_argument("--num-tiles", type=int, default=NUM_TILES)
    p.add_argument("--tile-size", type=int, default=TILE_SIZE)
    p.add_argument("--xmin", type=float, default=DEFAULT_CROP["xmin"])
    p.add_argument("--xmax", type=float, default=DEFAULT_CROP["xmax"])
    p.add_argument("--ymin", type=float, default=DEFAULT_CROP["ymin"])
    p.add_argument("--ymax", type=float, default=DEFAULT_CROP["ymax"])
    return p.parse_args()


def get_date_from_filename(filename):
    """Extract date from Sentinel-2 filename like T43RFP_20210305T..."""
    import re
    m = re.search(r"(\d{4})(\d{2})(\d{2})T", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return "unknown"


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    date_str = get_date_from_filename(os.path.basename(args.input))
    img_dir = os.path.join(args.output, "images", date_str)
    os.makedirs(img_dir, exist_ok=True)

    # ── Open TIFF and print metadata ──
    with rasterio.open(args.input) as src:
        print("=== INPUT TIFF ===")
        print(f"  Dimensions : {src.width} x {src.height}")
        print(f"  Bands      : {src.count}")
        print(f"  CRS        : {src.crs}")
        print(f"  Pixel size : {src.res[0]:.2f} x {src.res[1]:.2f} m")
        print(f"  Bounds     : {src.bounds}")
        print(f"  Dtype      : {src.dtypes}")
        print()

        # ── Crop to agricultural region ──
        inv = ~src.transform
        c1, r1 = inv * (args.xmin, args.ymax)   # top-left
        c2, r2 = inv * (args.xmax, args.ymin)    # bottom-right
        c1, r1 = max(0, int(c1)), max(0, int(r1))
        c2, r2 = min(src.width, int(c2)), min(src.height, int(r2))

        print(f"  Crop window: rows [{r1}:{r2}], cols [{c1}:{c2}]")
        print(f"  Crop size  : {c2-c1} x {r2-r1} px = "
              f"{(c2-c1)*src.res[0]/1000:.1f} x {(r2-r1)*src.res[1]/1000:.1f} km")

        window = rasterio.windows.Window(c1, r1, c2 - c1, r2 - r1)
        data = src.read(window=window)       # (bands, H, W)
        pixel_size = src.res[0]
        transform = src.transform

    n_bands, H, W = data.shape
    print(f"  Array shape: {data.shape}")

    # ── Find valid agricultural tile positions ──
    step = args.tile_size
    positions = []

    for r in range(0, H - args.tile_size + 1, step):
        for c in range(0, W - args.tile_size + 1, step):
            chip = data[:, r:r + args.tile_size, c:c + args.tile_size]

            # Skip tiles with too much nodata
            if np.count_nonzero(chip[0]) / (args.tile_size ** 2) < 0.9:
                continue

            # Vegetation filter: green > red → likely crops/vegetation
            r_ch = chip[0].astype(float)
            g_ch = chip[1].astype(float)
            green_frac = (g_ch > r_ch).sum() / (args.tile_size ** 2)

            if green_frac > MIN_VEG_FRAC:
                positions.append((r, c, green_frac))

    # Sort by vegetation fraction (most agricultural first)
    positions.sort(key=lambda x: -x[2])
    print(f"\n  Valid agricultural positions: {len(positions)}")

    # Spread tiles evenly across the region
    stride = max(1, len(positions) // args.num_tiles)
    selected = positions[::stride][:args.num_tiles]
    print(f"  Selected tiles: {len(selected)}")

    # ── Save tiles + metadata ──
    meta_rows = []

    for idx, (row, col, veg_frac) in enumerate(selected):
        chip = np.transpose(
            data[:3, row:row + args.tile_size, col:col + args.tile_size],
            (1, 2, 0),
        )
        fname = f"rgb_0_tile_{idx}.png"
        Image.fromarray(chip).save(os.path.join(img_dir, fname))

        # UTM coordinates of tile top-left corner
        utm_x = args.xmin + col * pixel_size
        utm_y = args.ymax - row * pixel_size

        meta_rows.append({
            "tile_id": idx,
            "filename": fname,
            "date": date_str,
            "utm_x": utm_x,
            "utm_y": utm_y,
            "pixel_size_m": pixel_size,
            "tile_size_px": args.tile_size,
            "ground_size_m": args.tile_size * pixel_size,
            "vegetation_frac": round(veg_frac, 3),
            "source": f"Sentinel-2 L2A TCI {os.path.basename(args.input)}",
            "label": "",
        })

    csv_path = os.path.join(args.output, "metadata.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=meta_rows[0].keys())
        w.writeheader()
        w.writerows(meta_rows)

    print(f"\n=== DONE ===")
    print(f"  {len(selected)} tiles → {img_dir}")
    print(f"  Metadata   → {csv_path}")
    print(f"  Tile size  : {args.tile_size}x{args.tile_size} px at {pixel_size:.0f}m")
    print(f"  Ground size: {args.tile_size * pixel_size:.0f}m x {args.tile_size * pixel_size:.0f}m per tile")
    print(f"\n  Next: run 02_super_resolve.py to enhance to 2.5m")


if __name__ == "__main__":
    main()
