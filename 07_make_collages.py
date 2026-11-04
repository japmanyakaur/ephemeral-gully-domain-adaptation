"""
Step 7 — Make one temporal collage per location.

Takes the 5 dates of each location and stitches them into a single
2x3 grid image (6th cell is a legend), with the date written on each tile.
The VLM looks at ONE image per question, so this is how it "sees" all
years at once.

Usage (Windows):
    python 07_make_collages.py --dataset haryana_dataset_final --out collages_haryana
"""

import argparse
import csv
import os
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

CELL = 384          # each 128px tile is upscaled to 384px (3x) so the VLM doesn't lose detail
COLS, ROWS = 3, 2
PAD = 8
DATES = ["2020-01-30", "2021-01-27", "2022-01-27", "2023-01-27", "2024-01-27"]


def get_font(size):
    for name in ["arialbd.ttf", "DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_collage(tile_paths, font, small_font):
    W = COLS * CELL + (COLS + 1) * PAD
    H = ROWS * CELL + (ROWS + 1) * PAD
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for i, date in enumerate(DATES):
        r, c = divmod(i, COLS)
        x = PAD + c * (CELL + PAD)
        y = PAD + r * (CELL + PAD)
        path = tile_paths.get(date)
        if path and os.path.exists(path):
            tile = Image.open(path).convert("RGB").resize((CELL, CELL), Image.LANCZOS)
            canvas.paste(tile, (x, y))
        else:
            draw.rectangle([x, y, x + CELL, y + CELL], fill=(60, 60, 60))
            draw.text((x + CELL // 2, y + CELL // 2), "no image", fill="white",
                      font=small_font, anchor="mm")
        # date label (black box, white text) in top-left corner
        label = date[:7]  # e.g. 2020-01
        bbox = draw.textbbox((x + 6, y + 6), label, font=font)
        draw.rectangle([bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3], fill=(0, 0, 0))
        draw.text((x + 6, y + 6), label, fill="white", font=font)

    # 6th cell: plain legend so the grid shape matches Missouri's 2x3
    x = PAD + 2 * (CELL + PAD)
    y = PAD + 1 * (CELL + PAD)
    draw.rectangle([x, y, x + CELL, y + CELL], fill=(235, 235, 235))
    draw.multiline_text((x + CELL // 2, y + CELL // 2),
                        "Same field,\n5 winters\n(Jan 2020-2024)\n320 m x 320 m",
                        fill=(40, 40, 40), font=small_font, anchor="mm", align="center", spacing=8)
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="haryana_dataset_final")
    p.add_argument("--out", default="collages_haryana")
    p.add_argument("--only", type=int, default=None, help="make just one tile_id (for testing)")
    args = p.parse_args()

    with open(os.path.join(args.dataset, "metadata.csv")) as f:
        rows = list(csv.DictReader(f))

    by_loc = defaultdict(dict)
    for r in rows:
        by_loc[int(r["tile_id"])][r["date"]] = os.path.join(args.dataset, "images", r["date"], r["filename"])

    os.makedirs(args.out, exist_ok=True)
    font, small_font = get_font(30), get_font(28)

    ids = [args.only] if args.only is not None else sorted(by_loc)
    for tid in ids:
        img = make_collage(by_loc[tid], font, small_font)
        img.save(os.path.join(args.out, f"collage_{tid:03d}.png"))

    print(f"Made {len(ids)} collages in {args.out}/")


if __name__ == "__main__":
    main()