
import json
import re
from pathlib import Path
from collections import Counter

# 1. Set the path to your extracted dataset folder.
# CHANGE THIS if your actual folder is somewhere else.
ROOT = Path(r"D:\UserData\Downloads\GD_Dataset_ViGIR_WACV_2025")

IMAGE_DIR = ROOT / "GD_Images"
JSON_FILE = ROOT / "GT_L6_Tol.json"

# 2. Check that the paths exist.
if not ROOT.is_dir():
    raise FileNotFoundError(f"Dataset folder not found: {ROOT}")

if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

if not JSON_FILE.is_file():
    raise FileNotFoundError(
        f"JSON file not found: {JSON_FILE}\n"
        "Check the exact filename in File Explorer."
    )

# 3. Match filenames like rgb_0_tile_883.tif
pattern = re.compile(
    r"rgb_(\d+)_tile_(\d+)\.tif$",
    re.IGNORECASE
)

images_by_tile = {}
unmatched_files = []

# 4. Read image filenames and group by tile ID.
for path in IMAGE_DIR.iterdir():
    if not path.is_file():
        continue

    match = pattern.fullmatch(path.name)

    if match:
        image_number = int(match.group(1))
        tile_number = int(match.group(2))

        images_by_tile.setdefault(tile_number, []).append(image_number)

    elif path.suffix.lower() in {".tif", ".tiff"}:
        unmatched_files.append(path.name)

# 5. Read ground-truth labels.
with JSON_FILE.open("r", encoding="utf-8") as f:
    labels = json.load(f)

labels_by_tile = {
    int(tile_id): str(info["label"])
    for tile_id, info in labels.items()
}

# 6. Summarise image counts per tile.
images_per_tile = Counter(
    len(image_numbers)
    for image_numbers in images_by_tile.values()
)

positive = sum(label == "1" for label in labels_by_tile.values())
negative = sum(label == "0" for label in labels_by_tile.values())

image_tiles = set(images_by_tile)
label_tiles = set(labels_by_tile)

# 7. Print audit results.
print("\n=== MISSOURI DATASET AUDIT ===")
print(f"Dataset folder: {ROOT}")
print(f"Image folder: {IMAGE_DIR}")
print(f"JSON file: {JSON_FILE}")

print(f"\nTotal TIFF images matched: "
      f"{sum(len(v) for v in images_by_tile.values())}")

print(f"Unique image tile IDs: {len(image_tiles)}")

print(f"Tiles with exactly 6 temporal images: "
      f"images_per_tile.get(6, 0)")

print(f"Tiles with a different image count: "
      f"{sum(count for n, count in images_per_tile.items() if n != 6)}")

print(f"Labelled tiles in JSON: {len(labels_by_tile)}")
print(f"Positive labels (1): {positive}")
print(f"Negative labels (0): {negative}")

print(f"Image tiles without a label: "
      f"{len(image_tiles - label_tiles)}")

print(f"Labelled tiles without images: "
      f"{len(label_tiles - image_tiles)}")

print(f"TIFF files not matching expected naming: "
      f"{len(unmatched_files)}")

print("\nImage counts per tile:")
for count, number_of_tiles in sorted(images_per_tile.items()):
    print(f"  {count} image(s): {number_of_tiles} tile(s)")

print("\nExample image tile IDs:", sorted(image_tiles)[:10])
print("Example JSON tile IDs:", sorted(label_tiles)[:10])

if image_tiles - label_tiles:
    print("\nExample image tiles without labels:",
          sorted(image_tiles - label_tiles)[:10])

if label_tiles - image_tiles:
    print("\nExample labelled tiles without images:",
          sorted(label_tiles - image_tiles)[:10])

if unmatched_files:
    print("\nExample unmatched TIFF filenames:",
          unmatched_files[:10])

print("\n=== AUDIT COMPLETE ===")
