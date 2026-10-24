
from pathlib import Path
import re
import matplotlib.pyplot as plt
from PIL import Image

# Path to the extracted Missouri dataset
ROOT = Path(r"D:\UserData\Downloads\GD_Dataset_ViGIR_WACV_2025")
IMAGE_DIR = ROOT / "GD_Images"

# Select a tile to inspect
TILE_ID = 883

pattern = re.compile(
    rf"rgb_(\d+)_tile_{TILE_ID}\.tif$",
    re.IGNORECASE
)

images = []

for path in IMAGE_DIR.iterdir():
    match = pattern.fullmatch(path.name)
    if match:
        time_index = int(match.group(1))
        images.append((time_index, path))

# Sort by temporal index
images.sort(key=lambda item: item[0])

if not images:
    raise FileNotFoundError(
        f"No images found for tile {TILE_ID} in {IMAGE_DIR}"
    )

print(f"Tile ID: {TILE_ID}")
print(f"Number of images: {len(images)}")

fig, axes = plt.subplots(
    1, len(images),
    figsize=(4 * len(images), 5),
    squeeze=False
)

for ax, (time_index, path) in zip(axes[0], images):
    with Image.open(path) as image:
        ax.imshow(image.convert("RGB"))
        ax.set_title(f"Time index: {time_index}")
        ax.axis("off")

    print(f"Image: {path.name}")

plt.suptitle(f"Missouri temporal sequence — tile {TILE_ID}")
plt.tight_layout()
plt.show()
