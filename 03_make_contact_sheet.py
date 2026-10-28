"""
Step 3: Generate a contact sheet (grid preview) of all tiles.

Creates a single image showing all tiles side-by-side for quick
visual quality check — helps spot urban tiles, nodata, or bad crops.

Usage:
    python 03_make_contact_sheet.py --input <tiles_dir> --output overview.png

Requirements:
    pip install Pillow
"""

import argparse
import math
import os
import sys

from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(
        description="Create a contact sheet grid from dataset tiles"
    )
    p.add_argument(
        "--input",
        required=True,
        help="Tiles directory (output of step 01 or 02)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: <input>/contact_sheet.png)",
    )
    p.add_argument(
        "--cols",
        type=int,
        default=10,
        help="Number of columns in the grid (default: 10)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.output is None:
        args.output = os.path.join(args.input, "contact_sheet.png")

    # ── Find all tiles ──
    tiles = []
    for root, dirs, files in os.walk(args.input):
        for f in sorted(files):
            if f.endswith(".png") and f.startswith("rgb_0_tile_"):
                tiles.append(os.path.join(root, f))

    if not tiles:
        print(f"ERROR: No rgb_0_tile_*.png found under {args.input}")
        sys.exit(1)

    # Sort by tile index
    def tile_index(path):
        name = os.path.basename(path)
        try:
            return int(name.replace("rgb_0_tile_", "").replace(".png", ""))
        except ValueError:
            return 0

    tiles.sort(key=tile_index)

    # ── Read first tile to get dimensions ──
    sample = Image.open(tiles[0])
    tw, th = sample.size

    cols = min(args.cols, len(tiles))
    rows = math.ceil(len(tiles) / cols)

    print(f"  Tiles found : {len(tiles)}")
    print(f"  Tile size   : {tw}x{th} px")
    print(f"  Grid        : {cols} cols x {rows} rows")

    # ── Build contact sheet ──
    sheet = Image.new("RGB", (tw * cols, th * rows), (0, 0, 0))

    for i, path in enumerate(tiles):
        img = Image.open(path)
        r, c = divmod(i, cols)
        sheet.paste(img, (c * tw, r * th))

    sheet.save(args.output)
    print(f"\n=== DONE ===")
    print(f"  Contact sheet: {args.output}")
    print(f"  Size         : {tw * cols} x {th * rows} px")


if __name__ == "__main__":
    main()
