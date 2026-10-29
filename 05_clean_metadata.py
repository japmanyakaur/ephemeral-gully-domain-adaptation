"""
Step 5: Sync metadata.csv after manual tile deletion.

After the first tiling run, you visually inspect the tiles and delete
any that contain urban areas, clouds, or other unwanted content.
This script removes rows from metadata.csv for tiles that no longer
exist on disk, so that --reuse-locations only extracts verified
agricultural spots.

Usage:
    # 1. Delete unwanted tiles manually (urban, cloudy, etc.)
    # 2. Run this to sync metadata:
    python 05_clean_metadata.py --dir haryana_dataset

Requirements:
    (none beyond Python stdlib)
"""

import argparse
import csv
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="Remove metadata rows for deleted tiles"
    )
    p.add_argument(
        "--dir", required=True,
        help="Dataset directory containing metadata.csv and images/",
    )
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = os.path.join(args.dir, "metadata.csv")

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    # Read all rows
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)

    # Check which tiles still exist on disk
    kept = []
    removed = []

    for row in all_rows:
        date = row["date"]
        fname = row["filename"]
        tile_path = os.path.join(args.dir, "images", date, fname)

        if os.path.exists(tile_path):
            kept.append(row)
        else:
            removed.append(row)

    if not removed:
        print(f"All {len(all_rows)} tiles still exist. Nothing to clean.")
        return

    # Write cleaned CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept)

    # Summary
    unique_ids_kept = len(set(r["tile_id"] for r in kept))
    unique_ids_removed = len(set(r["tile_id"] for r in removed))

    print(f"=== METADATA CLEANED ===")
    print(f"  Before : {len(all_rows)} rows")
    print(f"  Removed: {len(removed)} rows ({unique_ids_removed} spot IDs)")
    print(f"  Kept   : {len(kept)} rows ({unique_ids_kept} spot IDs)")
    print(f"  Saved  : {csv_path}")
    print()
    if removed:
        print(f"  Removed tiles:")
        for r in removed[:20]:
            print(f"    {r['date']}/{r['filename']} (spot {r['tile_id']})")
        if len(removed) > 20:
            print(f"    ... and {len(removed) - 20} more")


if __name__ == "__main__":
    main()
