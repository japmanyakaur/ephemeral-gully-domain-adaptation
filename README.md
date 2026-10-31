# Ephemeral Gully Domain Adaptation

A multi-temporal Sentinel-2 dataset and processing pipeline for ephemeral gully detection in Indian agricultural landscapes. Designed for cross-region transfer learning from the [Missouri Ephemeral Gully Dataset](https://data.mendeley.com/) (NAIP, ~1 m, USA) to Haryana, India.

---


## Dataset

| | Haryana (this work) | Missouri (source) |
|---|---|---|
| **Region** | Hisar–Sirsa–Fatehabad, Haryana, India | Missouri, USA |
| **Satellite** | Sentinel-2 L2A TCI | NAIP |
| **Tile** | T43REM (UTM Zone 43N) | — |
| **Dates** | 5 rabi-season scenes (Jan 2020–2024) | 6 NAIP dates per spot |
| **Spots** | 300 agricultural locations | 621 |
| **Images** | 1 481 | ~3 700 |
| **Tile size** | 128 × 128 px | 128 × 128 px |
| **Resolution** | 2.5 m (super-resolved from 10 m) | ~1 m |
| **Ground coverage** | 320 m × 320 m | ~128 m × 128 m |

### Download

➡️ [**haryana_dataset_final.zip (v1.0)**](../../releases/tag/v1.0)

---

## Pipeline

```
Sentinel-2 TCI scene (10 980 × 10 980 px, 10 m/px)
  │
  ├─ 01_tile_sentinel2.py        Crop to agricultural zone, extract 128×128 tiles
  │   ├─ Discovery mode           Vegetation filter → 300 spots
  │   └─ Reuse mode               Same UTM coords across dates → multi-temporal stack
  │
  ├─ 02b_super_resolve_sen2sr.py  4× super-resolution (10 m → 2.5 m)
  │                                SEN2SR-Lite with Fourier hard constraint
  │
  ├─ 06_center_crop.py            512×512 → 128×128 center crop
  │                                Reduces ground from 1280 m to 320 m
  │
  ├─ 03_make_contact_sheet.py     Visual QC grid
  ├─ 05_clean_metadata.py         Sync metadata after deleting bad tiles
  │
  ▼
  1 481 tiles (128×128 px, 2.5 m, 5 dates)
```

### How it works

**Step 1 — Tiling.** The script opens a full Sentinel-2 TCI GeoTIFF, crops to the Hisar agricultural zone (70 × 80 km), and slides a 128×128 px window across it. Each candidate tile is filtered: ≥90% valid pixels and ≥30% vegetation (green > red). The top 300 are selected, spread geographically. In reuse mode, it reads UTM coordinates from a previous run's `metadata.csv` and extracts tiles at the same locations from a different date, building the multi-temporal stack.

**Step 2 — Super-resolution.** SEN2SR-Lite upscales each 128×128 tile to 512×512 (effective 2.5 m). The model is trained on 61 282 real Sentinel-2 ↔ NAIP image pairs and uses a Fourier hard constraint that preserves original low-frequency content, preventing hallucinated detail. A synthetic NIR band is computed from RGB when real Band 8 is unavailable.

**Step 3 — Center crop.** The 512×512 SR tiles are center-cropped to 128×128 px, reducing ground coverage from 1 280 m to 320 m. This brings the scale closer to the Missouri dataset (128 m at ~1 m) while keeping the 2.5 m resolution.

**Step 4 — QC.** A contact sheet is generated for visual inspection. Cloudy, dark, or non-agricultural tiles are manually deleted, and the metadata CSV is synced to match.

### Supporting scripts

| Script | Purpose |
|--------|---------|
| `02_super_resolve.py` | EDSR ×4 (generic SR, for comparison) |
| `04_compare_sr_methods.py` | Side-by-side EDSR vs SEN2SR with sharpness metrics |
| `audit_missouri.py` | Inspect the Missouri source dataset |
| `inspect_temporal_images.py` | Verify multi-temporal alignment across dates |

---

## Usage

```bash
# Install
pip install -r requirements.txt
pip install torch sen2sr mlstac    # for super-resolution

# 1. Download Sentinel-2 scenes from sentinel-2.allthingsspatial.io
#    Search "Hisar, Haryana" → select T43REM → "Download whole scene (true color)"
#    Place .tiff files in raw_scenes/

# 2. Discovery tiling (first date)
python3 01_tile_sentinel2.py \
    --input raw_scenes/T43REM_20240127T054101_TCI_10m.tiff \
    --output haryana_dataset --num-tiles 300

# 3. Reuse tiling (remaining dates)
python3 01_tile_sentinel2.py \
    --input raw_scenes/T43REM_20230127T054109_TCI_10m.tiff \
    --output haryana_dataset \
    --reuse-locations haryana_dataset/metadata.csv
# Repeat for 2022, 2021, 2020...

# 4. Super-resolve
python3 02b_super_resolve_sen2sr.py \
    --input haryana_dataset --output haryana_dataset_sr --device cuda

# 5. Center-crop
python3 06_center_crop.py \
    --input haryana_dataset_sr --output haryana_dataset_final --crop-size 128

# 6. QC
python3 03_make_contact_sheet.py --input haryana_dataset_final
# Delete bad tiles manually, then:
python3 05_clean_metadata.py --dir haryana_dataset_final
```

---

## Output Structure

```
haryana_dataset_final/
├── metadata.csv
├── contact_sheet.png
└── images/
    ├── 2020-01-30/
    │   ├── rgb_0_tile_0.png
    │   ├── rgb_0_tile_1.png
    │   └── ...
    ├── 2021-01-27/
    ├── 2022-01-27/
    ├── 2023-01-27/
    └── 2024-01-27/
```

---

Sentinel-2 is the only freely available, legally redistributable satellite imagery covering Indian agriculture at this scale. Commercial sources (Google Earth, Planet) prohibit redistribution, and NAIP is US-only. SEN2SR super-resolution bridges the resolution gap from 10 m to 2.5 m while keeping the dataset fully open.


## References

- Aybar et al. (2026), [SEN2SR: A foundation model for Sentinel-2 super-resolution](https://doi.org/10.1016/j.rse.2025.115222), *Remote Sensing of Environment*
- [Missouri Ephemeral Gully Detection Dataset](https://data.mendeley.com/)
