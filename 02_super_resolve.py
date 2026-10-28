"""
Step 2: Apply 4x super-resolution to Sentinel-2 tiles.

Enhances 128x128 tiles (10m) → 512x512 tiles (effective 2.5m)
using the EDSR deep learning model via OpenCV's DNN super-resolution.

The EDSR (Enhanced Deep Residual SR) model is downloaded automatically
on first run (~38 MB).

Usage:
    python 02_super_resolve.py --input <tiles_dir> --output <sr_dir>

Requirements:
    pip install opencv-contrib-python numpy
"""

import argparse
import os
import sys
import time
import urllib.request

import cv2
import numpy as np


EDSR_X4_URL = (
    "https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x4.pb"
)
EDSR_X4_SIZE = 38_573_255  # expected file size in bytes
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def download_model(model_path):
    """Download the EDSR x4 model if not already present."""
    if os.path.exists(model_path) and os.path.getsize(model_path) == EDSR_X4_SIZE:
        return

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Downloading EDSR x4 model (~38 MB) ...")
    urllib.request.urlretrieve(EDSR_X4_URL, model_path)
    print(f"  Saved to {model_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Super-resolve Sentinel-2 tiles using EDSR x4"
    )
    p.add_argument(
        "--input",
        required=True,
        help="Input tiles directory (output of 01_tile_sentinel2.py)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <input>_sr)",
    )
    p.add_argument(
        "--model",
        default=os.path.join(MODEL_DIR, "EDSR_x4.pb"),
        help="Path to EDSR x4 .pb model file",
    )
    p.add_argument(
        "--scale",
        type=int,
        default=4,
        choices=[2, 3, 4],
        help="Upscaling factor (default: 4)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.output is None:
        args.output = args.input.rstrip("/\\") + "_sr"

    # ── Download model if needed ──
    download_model(args.model)

    # ── Load EDSR model ──
    print(f"Loading EDSR x{args.scale} model ...")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(args.model)
    sr.setModel("edsr", args.scale)
    print("  Model ready\n")

    # ── Find all image subdirectories ──
    # Structure: <input>/images/<date>/rgb_0_tile_*.png
    images_root = os.path.join(args.input, "images")
    if not os.path.isdir(images_root):
        # Maybe the user pointed directly to the date folder
        images_root = args.input

    # Collect all PNG tiles
    tiles = []
    for root, dirs, files in os.walk(images_root):
        for f in sorted(files):
            if f.endswith(".png") and f.startswith("rgb_0_tile_"):
                tiles.append(os.path.join(root, f))

    if not tiles:
        print(f"ERROR: No rgb_0_tile_*.png files found under {args.input}")
        sys.exit(1)

    print(f"Found {len(tiles)} tiles to process")

    # ── Copy metadata.csv if it exists ──
    src_csv = os.path.join(args.input, "metadata.csv")
    if os.path.exists(src_csv):
        os.makedirs(args.output, exist_ok=True)
        import shutil
        dst_csv = os.path.join(args.output, "metadata.csv")
        shutil.copy(src_csv, dst_csv)

        # Update pixel_size in metadata
        try:
            import csv
            rows = []
            with open(dst_csv, "r") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    if "pixel_size_m" in row:
                        orig = float(row["pixel_size_m"])
                        row["pixel_size_m"] = str(round(orig / args.scale, 2))
                    if "tile_size_px" in row:
                        orig = int(row["tile_size_px"])
                        row["tile_size_px"] = str(orig * args.scale)
                    rows.append(row)
            with open(dst_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Updated metadata: pixel_size → {10/args.scale:.1f}m, "
                  f"tile_size → {128*args.scale}px")
        except Exception as e:
            print(f"  Warning: could not update metadata CSV: {e}")

    # ── Process each tile ──
    start = time.time()

    for i, tile_path in enumerate(tiles):
        # Preserve directory structure
        rel_path = os.path.relpath(tile_path, args.input)
        out_path = os.path.join(args.output, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        img = cv2.imread(tile_path)
        if img is None:
            print(f"  WARNING: could not read {tile_path}, skipping")
            continue

        result = sr.upsample(img)
        cv2.imwrite(out_path, result)

        if (i + 1) % 10 == 0 or (i + 1) == len(tiles):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(tiles) - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{len(tiles)} tiles done "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    total = time.time() - start
    h, w = result.shape[:2]
    print(f"\n=== DONE ===")
    print(f"  {len(tiles)} tiles super-resolved in {total:.0f}s")
    print(f"  Output size : {w}x{h} px per tile")
    print(f"  Effective resolution: {10/args.scale:.1f}m")
    print(f"  Output dir  : {args.output}")


if __name__ == "__main__":
    main()
