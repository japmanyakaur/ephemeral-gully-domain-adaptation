"""
Step 2b: Apply 4x super-resolution using SEN2SR (satellite-trained).

Enhances 128x128 tiles (10m) → 512x512 tiles (effective 2.5m) using the
SEN2SR framework — a Sentinel-2–specific SR model trained on 61,282 real
Sentinel-2 ↔ NAIP image pairs with a Fourier-based hard constraint that
prevents hallucination.

Reference:
    Aybar et al. (2026), "SEN2SR: A foundation model for Sentinel-2
    super-resolution", Remote Sensing of Environment.

Requires:
    pip install sen2sr mlstac torch numpy Pillow

Model is downloaded automatically from HuggingFace on first run (~20 MB
for CNN-Light / SEN2SR-Lite).

Usage:
    python 02b_super_resolve_sen2sr.py --input <tiles_dir> --output <sr_dir>

    # With a real NIR band (Band 8) for best results:
    python 02b_super_resolve_sen2sr.py --input <tiles_dir> --output <sr_dir> \\
        --nir-dir <path_to_nir_tiles>
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch
from PIL import Image


# ── Model catalogue ──────────────────────────────────────────────────
# HuggingFace mlm.json URLs for each SEN2SR variant
MODEL_CATALOGUE = {
    "cnn-light": {
        "url": "https://huggingface.co/tacofoundation/sen2sr/resolve/main/"
               "SEN2SRLite/NonReference_RGBN_x4/mlm.json",
        "local": "model/SEN2SRLite_RGBN",
        "description": "CNN Light (SEN2SR-Lite) — fastest, good quality",
    },
    "cnn-full": {
        "url": "https://huggingface.co/tacofoundation/sen2sr/resolve/main/"
               "SEN2SRLite/main/mlm.json",
        "local": "model/SEN2SRLite",
        "description": "CNN (SEN2SR-Lite full) — 10m+20m bands to 2.5m",
    },
}

DEFAULT_MODEL = "cnn-light"


def parse_args():
    p = argparse.ArgumentParser(
        description="Super-resolve Sentinel-2 tiles using SEN2SR (satellite-trained)"
    )
    p.add_argument(
        "--input", required=True,
        help="Input tiles directory (output of 01_tile_sentinel2.py)",
    )
    p.add_argument(
        "--output", default=None,
        help="Output directory (default: <input>_sen2sr)",
    )
    p.add_argument(
        "--model-name", default=DEFAULT_MODEL, choices=list(MODEL_CATALOGUE.keys()),
        help=f"SEN2SR model variant (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--model-dir", default=None,
        help="Directory where model weights are cached (default: ./model/)",
    )
    p.add_argument(
        "--nir-dir", default=None,
        help="Directory with matching NIR band tiles (same filenames). "
             "If not provided, a synthetic NIR is computed from RGB.",
    )
    p.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"],
        help="Compute device (default: cpu)",
    )
    return p.parse_args()


def download_model(model_name, model_dir=None):
    """Download the SEN2SR model from HuggingFace via mlstac."""
    import mlstac

    info = MODEL_CATALOGUE[model_name]
    local_dir = model_dir or info["local"]

    # Check if already downloaded
    mlm_local = os.path.join(local_dir, "mlm.json")
    if os.path.exists(mlm_local):
        print(f"  Model already cached in {local_dir}")
        loader = mlstac.load(local_dir)
    else:
        print(f"  Downloading {info['description']} ...")
        print(f"  URL: {info['url']}")
        loader = mlstac.download(info["url"], local_dir)
        print(f"  Saved to {local_dir}")

    return loader


def make_synthetic_nir(rgb: np.ndarray) -> np.ndarray:
    """
    Compute a synthetic NIR band from RGB.

    Vegetation reflects strongly in NIR and absorbs Red, so a simple
    approximation is:  NIR ≈ clamp(2*Green - Red, 0, 1)

    This preserves spatial structure (edges, textures) while giving the
    model a rough NIR signal. For best results, use real Band 8 data.

    Parameters
    ----------
    rgb : np.ndarray
        Float32 array of shape (3, H, W) in [0, 1], band order R, G, B.

    Returns
    -------
    np.ndarray
        Float32 array of shape (1, H, W) — synthetic NIR in [0, 1].
    """
    red = rgb[0]
    green = rgb[1]
    nir = np.clip(2.0 * green - red, 0.0, 1.0)
    return nir[np.newaxis, :, :]


def load_tile_as_rgbn(tile_path, nir_path=None):
    """
    Load a tile PNG and return a float32 RGBN tensor ready for SEN2SR.

    Parameters
    ----------
    tile_path : str
        Path to the RGB tile PNG (uint8, 0-255).
    nir_path : str or None
        Path to a matching NIR tile. If None, synthetic NIR is used.

    Returns
    -------
    torch.Tensor
        Shape (4, H, W), float32 in [0, 1], band order R, G, B, NIR.
    """
    img = Image.open(tile_path).convert("RGB")
    rgb = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    rgb = rgb.transpose(2, 0, 1)                    # (3, H, W)

    # Band order for SEN2SR RGBN: Red=B04, Green=B03, Blue=B02, NIR=B08
    # Our TCI is already RGB order, so: R=ch0, G=ch1, B=ch2
    # SEN2SR expects: B04(Red), B03(Green), B02(Blue), B08(NIR)
    # which is exactly R, G, B, NIR — same order.

    if nir_path and os.path.exists(nir_path):
        nir_img = Image.open(nir_path).convert("L")
        nir = np.array(nir_img, dtype=np.float32) / 255.0
        nir = nir[np.newaxis, :, :]
    else:
        nir = make_synthetic_nir(rgb)

    rgbn = np.concatenate([rgb, nir], axis=0)  # (4, H, W)

    tensor = torch.from_numpy(rgbn).float()
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    return tensor


def sr_output_to_rgb(output_tensor):
    """
    Convert the SEN2SR output tensor back to a uint8 RGB image.

    Parameters
    ----------
    output_tensor : torch.Tensor
        Shape (4, H, W) or (3, H, W), float32.

    Returns
    -------
    np.ndarray
        uint8 array of shape (H, W, 3) — RGB image ready to save.
    """
    # Take first 3 channels (R, G, B), ignore NIR
    rgb = output_tensor[:3].detach().cpu().numpy()
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    rgb = rgb.transpose(1, 2, 0)  # (H, W, 3)
    return rgb


def find_tiles(input_dir):
    """Find all rgb_0_tile_*.png files under the input directory."""
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


def main():
    args = parse_args()

    if args.output is None:
        args.output = args.input.rstrip("/\\") + "_sen2sr"

    print("=" * 60)
    print("  SEN2SR Super-Resolution (Satellite-Trained)")
    print("=" * 60)
    print(f"  Model     : {args.model_name}")
    print(f"  Device    : {args.device}")
    print(f"  NIR source: {'real (from --nir-dir)' if args.nir_dir else 'synthetic (computed from RGB)'}")
    print()

    # ── Step 1: Download / load model ──
    print("Loading SEN2SR model ...")
    try:
        loader = download_model(args.model_name, args.model_dir)
        model = loader.compiled_model(device=args.device)
        model.eval()
        print("  Model ready\n")
    except Exception as e:
        print(f"\nERROR loading model: {e}")
        print("\nTroubleshooting:")
        print("  1. Install dependencies:  pip install sen2sr mlstac torch")
        print("  2. Check internet connection (model downloads from HuggingFace)")
        print("  3. Try:  pip install --upgrade mlstac sen2sr")
        sys.exit(1)

    # ── Step 2: Find tiles ──
    tiles = find_tiles(args.input)
    if not tiles:
        print(f"ERROR: No rgb_0_tile_*.png files found under {args.input}")
        sys.exit(1)

    print(f"Found {len(tiles)} tiles to process\n")

    # ── Step 3: Copy metadata.csv ──
    src_csv = os.path.join(args.input, "metadata.csv")
    if os.path.exists(src_csv):
        os.makedirs(args.output, exist_ok=True)
        dst_csv = os.path.join(args.output, "metadata.csv")
        shutil.copy(src_csv, dst_csv)

        try:
            import csv
            rows = []
            with open(dst_csv, "r") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    if "pixel_size_m" in row:
                        orig = float(row["pixel_size_m"])
                        row["pixel_size_m"] = str(round(orig / 4, 2))
                    if "tile_size_px" in row:
                        orig = int(row["tile_size_px"])
                        row["tile_size_px"] = str(orig * 4)
                    rows.append(row)
            with open(dst_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Updated metadata: pixel_size → 2.5m, tile_size → 512px")
        except Exception as e:
            print(f"  Warning: could not update metadata CSV: {e}")

    # ── Step 4: Process each tile ──
    start = time.time()
    device = args.device

    for i, tile_path in enumerate(tiles):
        # Build matching NIR path if provided
        nir_path = None
        if args.nir_dir:
            rel = os.path.relpath(tile_path, args.input)
            nir_path = os.path.join(args.nir_dir, rel)

        # Preserve directory structure
        rel_path = os.path.relpath(tile_path, args.input)
        out_path = os.path.join(args.output, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Load and prepare input
        X = load_tile_as_rgbn(tile_path, nir_path)
        X = X.to(device)

        # Run super-resolution
        with torch.no_grad():
            sr_output = model(X[None]).squeeze(0)

        # Convert back to RGB image and save
        rgb_out = sr_output_to_rgb(sr_output)
        Image.fromarray(rgb_out).save(out_path)

        if (i + 1) % 10 == 0 or (i + 1) == len(tiles):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(tiles) - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{len(tiles)} tiles done "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    total = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  DONE — SEN2SR Super-Resolution")
    print(f"{'=' * 60}")
    print(f"  {len(tiles)} tiles processed in {total:.0f}s")
    print(f"  Output size : 512x512 px per tile")
    print(f"  Effective resolution: 2.5m")
    print(f"  NIR source  : {'real' if args.nir_dir else 'synthetic'}")
    print(f"  Output dir  : {args.output}")


if __name__ == "__main__":
    main()
