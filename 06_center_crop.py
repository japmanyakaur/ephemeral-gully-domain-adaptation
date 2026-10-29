"""
Step 6: Center-crop SR tiles to reduce ground coverage.

After SEN2SR 4× super-resolution, each 512×512 tile covers 1280m × 1280m
(10× the Missouri dataset's 128m). Center-cropping to 128×128 brings
ground coverage down to 320m × 320m (only 2.5× Missouri).

This makes cross-region transfer learning more practical because the
model sees a similar spatial extent in both datasets.

Usage:
    python 06_center_crop.py --input <sr_dir> --output <cropped_dir>
    python 06_center_crop.py --input <sr_dir> --output <cropped_dir> --crop-size 256

Requirements:
    pip install Pillow
"""

import argparse
import csv
import os
import shutil
import sys

from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(
        description="Center-crop SR tiles to reduce ground coverage"
    )
    p.add_argument(
        "--input", required=True,
        help="Input directory with 512×512 SR tiles",
    )
    p.add_argument(
        "--output", default=None,
        help="Output directory (default: <input>_crop)",
    )
    p.add_argument(
        "--crop-size", type=int, default=128,
        help="Output tile size in pixels (default: 128). "
             "128 → 320m ground (2.5× Missouri), "
             "256 → 640m ground (5× Missouri)",
    )
    return p.parse_args()


def find_tiles(input_dir):
    """Find all rgb_0_tile_*.png files."""
    tiles = []
    for root, _dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if f.endswith(".png") and f.startswith("rgb_0_tile_"):
                tiles.append(os.path.join(root, f))
    return sorted(tiles)


def main():
    args = parse_args()

    if args.output is None:
        args.output = args.input.rstrip("/\\") + "_crop"

    tiles = find_tiles(args.input)
    if not tiles:
        print(f"ERROR: No rgb_0_tile_*.png files found under {args.input}")
        sys.exit(1)

    # Check input tile size
    sample = Image.open(tiles[0])
    in_size = sample.width
    if args.crop_size >= in_size:
        print(f"ERROR: crop-size ({args.crop_size}) must be smaller than "
              f"input tile size ({in_size})")
        sys.exit(1)

    # Effective resolution: 10m / 4x SR = 2.5m per pixel
    px_res = 2.5
    ground_m = args.crop_size * px_res

    print("=" * 60)
    print("  CENTER CROP — Reduce Ground Coverage")
    print("=" * 60)
    print(f"  Input tiles  : {len(tiles)} × {in_size}×{in_size} px")
    print(f"  Input ground : {in_size * px_res:.0f}m × {in_size * px_res:.0f}m")
    print(f"  Crop to      : {args.crop_size}×{args.crop_size} px")
    print(f"  Output ground: {ground_m:.0f}m × {ground_m:.0f}m")
    print(f"  Missouri ref : 128m × 128m  →  gap = {ground_m/128:.1f}×")
    print()

    offset = (in_size - args.crop_size) // 2

    count = 0
    for tile_path in tiles:
        rel_path = os.path.relpath(tile_path, args.input)
        out_path = os.path.join(args.output, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        img = Image.open(tile_path)
        cropped = img.crop((
            offset, offset,
            offset + args.crop_size,
            offset + args.crop_size,
        ))
        cropped.save(out_path)
        count += 1

    # Update metadata.csv if it exists
    src_csv = os.path.join(args.input, "metadata.csv")
    if os.path.exists(src_csv):
        dst_csv = os.path.join(args.output, "metadata.csv")
        rows = []
        with open(src_csv, "r") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                # Update pixel size stays 2.5m, tile size changes
                row["tile_size_px"] = str(args.crop_size)
                row["ground_size_m"] = str(ground_m)
                # Shift UTM coordinates to center of crop
                if "utm_x" in row and "utm_y" in row:
                    shift_m = offset * px_res
                    row["utm_x"] = str(float(row["utm_x"]) + shift_m)
                    row["utm_y"] = str(float(row["utm_y"]) - shift_m)
                rows.append(row)
        with open(dst_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"  Updated metadata → {dst_csv}")

    print(f"\n=== DONE ===")
    print(f"  {count} tiles → {args.output}")
    print(f"  Size: {args.crop_size}×{args.crop_size} px at {px_res}m = "
          f"{ground_m:.0f}m × {ground_m:.0f}m ground")


if __name__ == "__main__":
    main()
