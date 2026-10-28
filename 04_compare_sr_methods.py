"""
Step 4: Visual comparison of EDSR vs SEN2SR super-resolution.

Creates side-by-side comparison images showing:
  - Original tile (128x128, 10m)
  - EDSR output (512x512, generic SR)
  - SEN2SR output (512x512, satellite-trained SR)

Also computes basic sharpness metrics (Laplacian variance) to
quantify which method produces sharper, more detailed output.

Usage:
    python 04_compare_sr_methods.py \\
        --original <tiles_dir> \\
        --edsr <edsr_sr_dir> \\
        --sen2sr <sen2sr_sr_dir> \\
        --output comparison_results

Requirements:
    pip install Pillow numpy opencv-contrib-python
"""

import argparse
import csv
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare EDSR and SEN2SR super-resolution results"
    )
    p.add_argument(
        "--original", required=True,
        help="Original tiles directory (128x128 px, from step 01)",
    )
    p.add_argument(
        "--edsr", required=True,
        help="EDSR super-resolved tiles (from step 02)",
    )
    p.add_argument(
        "--sen2sr", required=True,
        help="SEN2SR super-resolved tiles (from step 02b)",
    )
    p.add_argument(
        "--output", default="comparison_results",
        help="Output directory for comparison images and metrics",
    )
    p.add_argument(
        "--num-samples", type=int, default=10,
        help="Number of tiles to compare (default: 10, evenly sampled)",
    )
    return p.parse_args()


def find_tiles(input_dir):
    """Find all rgb_0_tile_*.png files."""
    tiles = []
    for root, dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if f.endswith(".png") and f.startswith("rgb_0_tile_"):
                tiles.append(os.path.join(root, f))

    def tile_index(path):
        name = os.path.basename(path)
        try:
            return int(name.replace("rgb_0_tile_", "").replace(".png", ""))
        except ValueError:
            return 0

    tiles.sort(key=tile_index)
    return tiles


def laplacian_variance(image_path):
    """
    Compute Laplacian variance — a measure of image sharpness.
    Higher = sharper / more detail.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return float(lap.var())


def edge_density(image_path):
    """
    Compute edge density using Canny edge detector.
    Higher = more visible edges and features.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    edges = cv2.Canny(img, 50, 150)
    return float(edges.sum()) / (img.shape[0] * img.shape[1])


def add_label(img, text, position="top"):
    """Add a text label to an image."""
    draw = ImageDraw.Draw(img)
    # Use a basic font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Calculate text position
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]

    x = (img.width - text_w) // 2
    y = 5 if position == "top" else img.height - bbox[3] - 10

    # Draw background rectangle for readability
    draw.rectangle([x - 5, y - 2, x + text_w + 5, y + bbox[3] + 2], fill="black")
    draw.text((x, y), text, fill="white", font=font)
    return img


def make_comparison_image(orig_path, edsr_path, sen2sr_path, output_path,
                          edsr_sharp, sen2sr_sharp):
    """
    Create a side-by-side comparison image.

    Layout:
    [ Original (upscaled) | EDSR x4 | SEN2SR x4 ]
    """
    target_size = 512

    # Load images
    orig = Image.open(orig_path).convert("RGB")
    orig_up = orig.resize((target_size, target_size), Image.NEAREST)  # nearest neighbor to show pixels

    edsr = Image.open(edsr_path).convert("RGB")
    sen2sr = Image.open(sen2sr_path).convert("RGB")

    # Ensure all are same size
    edsr = edsr.resize((target_size, target_size), Image.LANCZOS)
    sen2sr = sen2sr.resize((target_size, target_size), Image.LANCZOS)

    # Add labels
    orig_up = add_label(orig_up, f"Original 10m (128px)")
    edsr = add_label(edsr, f"EDSR x4 (sharp={edsr_sharp:.0f})")
    sen2sr = add_label(sen2sr, f"SEN2SR x4 (sharp={sen2sr_sharp:.0f})")

    # Combine side by side with 4px gap
    gap = 4
    combined = Image.new("RGB",
                         (target_size * 3 + gap * 2, target_size),
                         (40, 40, 40))
    combined.paste(orig_up, (0, 0))
    combined.paste(edsr, (target_size + gap, 0))
    combined.paste(sen2sr, (target_size * 2 + gap * 2, 0))

    combined.save(output_path, quality=95)
    return combined


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # ── Find matching tiles ──
    orig_tiles = find_tiles(args.original)
    edsr_tiles = find_tiles(args.edsr)
    sen2sr_tiles = find_tiles(args.sen2sr)

    if not orig_tiles:
        print(f"ERROR: No tiles found in {args.original}")
        sys.exit(1)
    if not edsr_tiles:
        print(f"ERROR: No tiles found in {args.edsr}")
        sys.exit(1)
    if not sen2sr_tiles:
        print(f"ERROR: No tiles found in {args.sen2sr}")
        sys.exit(1)

    # Match by filename
    orig_by_name = {os.path.basename(p): p for p in orig_tiles}
    edsr_by_name = {os.path.basename(p): p for p in edsr_tiles}
    sen2sr_by_name = {os.path.basename(p): p for p in sen2sr_tiles}

    common = sorted(set(orig_by_name) & set(edsr_by_name) & set(sen2sr_by_name))

    if not common:
        print("ERROR: No matching tiles found across all three directories")
        print(f"  Original tiles: {len(orig_tiles)}")
        print(f"  EDSR tiles    : {len(edsr_tiles)}")
        print(f"  SEN2SR tiles  : {len(sen2sr_tiles)}")
        sys.exit(1)

    # Sample evenly
    n = min(args.num_samples, len(common))
    step = max(1, len(common) // n)
    sampled = [common[i] for i in range(0, len(common), step)][:n]

    print(f"Comparing {n} tiles (from {len(common)} matched)\n")

    # ── Compute metrics and build comparisons ──
    metrics = []

    for i, name in enumerate(sampled):
        orig_path = orig_by_name[name]
        edsr_path = edsr_by_name[name]
        sen2sr_path = sen2sr_by_name[name]

        # Compute sharpness
        edsr_sharp = laplacian_variance(edsr_path)
        sen2sr_sharp = laplacian_variance(sen2sr_path)
        orig_sharp = laplacian_variance(orig_path)

        # Compute edge density
        edsr_edges = edge_density(edsr_path)
        sen2sr_edges = edge_density(sen2sr_path)
        orig_edges = edge_density(orig_path)

        metrics.append({
            "tile": name,
            "orig_sharpness": round(orig_sharp, 1),
            "edsr_sharpness": round(edsr_sharp, 1),
            "sen2sr_sharpness": round(sen2sr_sharp, 1),
            "edsr_vs_sen2sr": "SEN2SR" if sen2sr_sharp > edsr_sharp else "EDSR",
            "orig_edge_density": round(orig_edges, 4),
            "edsr_edge_density": round(edsr_edges, 4),
            "sen2sr_edge_density": round(sen2sr_edges, 4),
        })

        # Build comparison image
        out_name = f"compare_{i:02d}_{name}"
        out_path = os.path.join(args.output, out_name)
        make_comparison_image(orig_path, edsr_path, sen2sr_path, out_path,
                              edsr_sharp, sen2sr_sharp)

        print(f"  [{i+1}/{n}] {name}")
        print(f"        Sharpness:  EDSR={edsr_sharp:.0f}  SEN2SR={sen2sr_sharp:.0f}  "
              f"→ {'SEN2SR' if sen2sr_sharp > edsr_sharp else 'EDSR'} wins")

    # ── Save metrics CSV ──
    csv_path = os.path.join(args.output, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    # ── Summary ──
    sen2sr_wins = sum(1 for m in metrics if m["edsr_vs_sen2sr"] == "SEN2SR")
    edsr_wins = n - sen2sr_wins

    avg_edsr = np.mean([m["edsr_sharpness"] for m in metrics])
    avg_sen2sr = np.mean([m["sen2sr_sharpness"] for m in metrics])

    print(f"\n{'=' * 60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Tiles compared     : {n}")
    print(f"  EDSR wins          : {edsr_wins}")
    print(f"  SEN2SR wins        : {sen2sr_wins}")
    print(f"  Avg sharpness EDSR : {avg_edsr:.0f}")
    print(f"  Avg sharpness SEN2SR: {avg_sen2sr:.0f}")
    print(f"  Recommendation     : {'SEN2SR' if sen2sr_wins > edsr_wins else 'EDSR'}")
    print(f"\n  Comparison images  : {args.output}/")
    print(f"  Metrics CSV        : {csv_path}")

    # ── Build overview grid ──
    comp_images = sorted([
        os.path.join(args.output, f) for f in os.listdir(args.output)
        if f.startswith("compare_") and f.endswith(".png")
    ])

    if comp_images:
        sample = Image.open(comp_images[0])
        cw, ch = sample.size

        rows = len(comp_images)
        grid = Image.new("RGB", (cw, ch * rows), (40, 40, 40))

        for idx, path in enumerate(comp_images):
            img = Image.open(path)
            grid.paste(img, (0, ch * idx))

        grid_path = os.path.join(args.output, "overview_comparison.png")
        grid.save(grid_path, quality=95)
        print(f"  Overview grid      : {grid_path}")


if __name__ == "__main__":
    main()
