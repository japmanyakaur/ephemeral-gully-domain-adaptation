"""
Step 7 — Make one temporal collage per location (Haryana or Missouri).

Stitches all dates of a location into a single 2x3 grid image with a label
on each tile. The VLM looks at ONE image per question, so this is how it
"sees" all years at once.

  Hisar   : 5 dates  -> 5 tiles + 1 legend cell (dates read from metadata.csv)
  Missouri: 6 images -> full 2x3 grid (same layout as Tousi et al. WACV 2025)

Usage (Windows):
    python 07_make_collages.py --dataset hisar
    python 07_make_collages.py --dataset missouri --root "D:\\UserData\\Downloads\\GD_Dataset_ViGIR_WACV_2025"

    # test on one location first
    python 07_make_collages.py --dataset hisar --only 0
"""

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CELL = 384          # each 128px tile is upscaled to 384px (3x) so the VLM doesn't lose detail
COLS, ROWS = 3, 2
PAD = 8


# ───────────────────────── loaders (the only dataset-specific part) ─────────────────────────

def load_sentinel(root):
    """Sentinel-2 dataset (metadata.csv + images/<date>/).
    Returns ({tile_id: [(label_text, path_or_None), ...]} in time order, sorted list of dates)."""
    with open(os.path.join(root, "metadata.csv")) as f:
        rows = list(csv.DictReader(f))
    all_dates = sorted({r["date"] for r in rows})
    by_loc = defaultdict(dict)
    for r in rows:
        by_loc[int(r["tile_id"])][r["date"]] = os.path.join(root, "images", r["date"], r["filename"])
    locs = {tid: [(d[:7], dates.get(d)) for d in all_dates] for tid, dates in by_loc.items()}
    return locs, all_dates


def load_missouri(root):
    """GD_Images/rgb_{k}_tile_{id}.tif, k = 0..5 (6 temporal images per location)."""
    img_dir = Path(root) / "GD_Images"
    pattern = re.compile(r"rgb_(\d+)_tile_(\d+)\.tif$", re.IGNORECASE)
    by_loc = defaultdict(dict)
    for p in img_dir.iterdir():
        m = pattern.fullmatch(p.name)
        if m:
            by_loc[int(m.group(2))][int(m.group(1))] = str(p)

    # Ground-truth labels, saved next to the collages so later scripts can use them
    labels = {}
    gt = Path(root) / "GT_L6_Tol.json"
    if gt.is_file():
        with gt.open(encoding="utf-8") as f:
            labels = {int(k): str(v["label"]) for k, v in json.load(f).items()}

    out = {
        tid: [(f"Image {k + 1}", imgs.get(k)) for k in range(6)]
        for tid, imgs in by_loc.items()
    }
    return out, labels


# ───────────────────────── drawing (shared by both datasets) ─────────────────────────

def get_font(size):
    for name in ["arialbd.ttf", "DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def open_rgb(path):
    """Open PNG or TIFF as 8-bit RGB (handles 4-band / 16-bit TIFFs)."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.float32)
        if arr.max() > 255:
            lo, hi = np.percentile(arr, (1, 99))
            arr = (arr - lo) / max(hi - lo, 1e-6) * 255
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_collage(frames, legend, font, small_font):
    W = COLS * CELL + (COLS + 1) * PAD
    H = ROWS * CELL + (ROWS + 1) * PAD
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for i in range(COLS * ROWS):
        r, c = divmod(i, COLS)
        x = PAD + c * (CELL + PAD)
        y = PAD + r * (CELL + PAD)

        if i >= len(frames):                       # spare cell -> legend
            draw.rectangle([x, y, x + CELL, y + CELL], fill=(235, 235, 235))
            draw.multiline_text((x + CELL // 2, y + CELL // 2), legend, fill=(40, 40, 40),
                                font=small_font, anchor="mm", align="center", spacing=8)
            continue

        label, path = frames[i]
        if path and os.path.exists(path):
            canvas.paste(open_rgb(path).resize((CELL, CELL), Image.LANCZOS), (x, y))
        else:
            draw.rectangle([x, y, x + CELL, y + CELL], fill=(60, 60, 60))
            draw.text((x + CELL // 2, y + CELL // 2), "no image", fill="white",
                      font=small_font, anchor="mm")

        bbox = draw.textbbox((x + 6, y + 6), label, font=font)
        draw.rectangle([bbox[0] - 4, bbox[1] - 3, bbox[2] + 4, bbox[3] + 3], fill=(0, 0, 0))
        draw.text((x + 6, y + 6), label, fill="white", font=font)
    return canvas


# ───────────────────────── main ─────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["hisar", "missouri"], default="hisar")
    p.add_argument("--root", default=None,
                   help="dataset folder (default: hisar_dataset_final, or the Missouri path on D:)")
    p.add_argument("--out", default=None, help="output folder (default: collages_<dataset>)")
    p.add_argument("--only", type=int, default=None, help="make just one tile_id (for testing)")
    args = p.parse_args()

    out = args.out or f"collages_{args.dataset}"
    labels = {}
    if args.dataset == "hisar":
        root = args.root or "hisar_dataset_final"
        locs, dates = load_sentinel(root)
        legend = f"Same field,\n{len(dates)} winters\n({dates[0][:4]}-{dates[-1][:4]})\n320 m x 320 m"
    else:
        root = args.root or r"D:\UserData\Downloads\GD_Dataset_ViGIR_WACV_2025"
        locs, labels = load_missouri(root)
        legend = ""  # 6 images fill the grid

    os.makedirs(out, exist_ok=True)
    font, small_font = get_font(30), get_font(28)

    ids = [args.only] if args.only is not None else sorted(locs)
    for tid in ids:
        make_collage(locs[tid], legend, font, small_font).save(os.path.join(out, f"collage_{tid:03d}.png"))

    # index file that Pipeline A/B will read: one row per collage (+ ground truth if known)
    with open(os.path.join(out, "collages.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile_id", "collage", "n_images", "label"])
        for tid in ids:
            n = sum(1 for _, pth in locs[tid] if pth and os.path.exists(pth))
            w.writerow([tid, f"collage_{tid:03d}.png", n, labels.get(tid, "")])

    print(f"Made {len(ids)} {args.dataset} collages in {out}/  (index: {out}/collages.csv)")


if __name__ == "__main__":
    main()
