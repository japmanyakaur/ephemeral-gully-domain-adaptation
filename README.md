# Ephemeral Gully Domain Adaptation

A multi-temporal Sentinel-2 dataset and processing pipeline for ephemeral gully detection in Indian agricultural landscapes, and an evaluation of zero-shot vision-language models (VLMs) transferred from the [Missouri Ephemeral Gully Dataset](https://data.mendeley.com/) (NAIP, ~1 m, USA) to the Hisar–Fatehabad–Sirsa agricultural belt in Haryana, India.

Method reference: Tousi et al., *A Zero-Shot Learning Approach for Ephemeral Gully Detection from Remote Sensing using Vision Language Models*, WACV 2025 Workshops ([arXiv 2503.01169](https://arxiv.org/abs/2503.01169)).

---

## Dataset (v2.0)

| | Hisar–Fatehabad–Sirsa (this work) | Missouri (source) |
|---|---|---|
| **Region** | Hisar, Fatehabad and Sirsa districts, Haryana, India | Missouri, USA |
| **Imagery** | Sentinel-2 L2A true colour (TCI), 10 m | NAIP aerial, ~1 m |
| **Sentinel-2 tile** | T43REN (UTM 43N) | — |
| **Area** | UTM 500 000–585 000 E, 3 215 000–3 285 000 N (85 × 70 km), only tiles inside the three districts | — |
| **Dates** | 5 rabi-season scenes: 2020-01-23, 2021-01-27, 2022-01-27, 2023-01-27, 2024-01-27 | 6 dates per location |
| **Locations** | 300 (Fatehabad 130, Hisar 94, Sirsa 76) | 621 |
| **Images** | 1 497 (297 locations have all 5 dates) | 3 726 |
| **Tile size** | 128 × 128 px | 128 × 128 px |
| **Resolution** | 2.5 m (SEN2SR 4× super-resolution of 10 m) | ~1 m |
| **Ground footprint** | 320 m × 320 m | ~128 m × 128 m |

`hisar_dataset_final/metadata.csv` lists every image with its date, UTM corner, **latitude/longitude of the tile centre and district**, so any location can be checked on a map.

### Download

➡️ **v2.0** (Releases): `hisar_dataset_final.zip` (dataset) and `collages_hisar.zip` (one 2×3 temporal collage per location)

> **Note on v1.0.** The v1.0 release (`haryana_dataset_final.zip`) was cut from Sentinel-2 tile **T43REM**, which turned out to cover the Haryana–Rajasthan border (Bhiwani, Jhunjhunu and Churu districts), not Hisar–Sirsa. It is kept for reference only and is superseded by v2.0.

---

## Pipeline

The whole dataset is rebuilt end to end by **`notebooks/00_rebuild_hisar_dataset.ipynb`** (runs on Kaggle, ~1 h, no manual downloads). It calls the scripts in this repo:

```
Sentinel-2 L2A scenes, tile T43REN (Earth Search STAC / AWS open data)
  │
  ├─ scene selection            per year: clearest scene in 1 Jan – 28 Feb, closest to 27 Jan
  │                             (cloud measured inside the study area with the SCL mask)
  ├─ 01_tile_sentinel2.py       discovery: 128×128 px agricultural tiles (≥30 % green > red)
  │                             → keep only tiles whose centre is in Hisar / Fatehabad / Sirsa → 300
  │                             reuse: same UTM locations cut from the other 4 dates
  ├─ cloud QC (automatic)       drop a tile-date if >5 % of its central 320 m is cloud/shadow (SCL)
  ├─ 05_clean_metadata.py       sync metadata with the remaining images
  ├─ 02b_super_resolve_sen2sr.py  SEN2SR-Lite 4× (10 m → 2.5 m), 128 → 512 px
  ├─ 06_center_crop.py          512 → 128 px centre crop (1 280 m → 320 m)
  ├─ dark/blank QC + 05         drop near-empty tiles, sync metadata
  ├─ location check             every tile tested against district boundaries; map saved
  ▼
  hisar_dataset_final/ (1 497 images) + collages_hisar/ (300 collages)
```

### Scripts

| Script | Purpose |
|---|---|
| `01_tile_sentinel2.py` | Crop a TCI scene to the study area and cut 128×128 tiles (discovery / reuse modes) |
| `02b_super_resolve_sen2sr.py` | SEN2SR-Lite 4× super-resolution |
| `06_center_crop.py` | Centre-crop super-resolved tiles to 320 m |
| `05_clean_metadata.py` | Remove metadata rows for deleted images |
| `07_make_collages.py` | One 2×3 temporal collage per location (`--dataset hisar` or `missouri`); dates are read from `metadata.csv` |
| `03_make_contact_sheet.py` | Visual QC grid |
| `02_super_resolve.py`, `04_compare_sr_methods.py` | EDSR baseline and EDSR-vs-SEN2SR comparison (method selection) |
| `audit_missouri.py`, `inspect_temporal_images.py` | Inspect the Missouri data / temporal alignment |

### Notebooks (Kaggle)

| Notebook | Purpose |
|---|---|
| `notebooks/00_rebuild_hisar_dataset.ipynb` | Build the v2.0 dataset and collages |
| `notebooks/08_pipeline_a.ipynb` | Pipeline A v1: direct yes/no VLM prompt (Qwen2.5-VL-7B) |
| `notebooks/08b_pipeline_a_rating.ipynb` | Pipeline A v2: 1–5 rating prompts ("short" vs "evidence") |
| `notebooks/08c_compare_models.ipynb` | Pipeline A "evidence" prompt across 6 open VLMs, with checkpoint/resume |

Results CSVs are in `results/`.

---

## Reproduce

1. Kaggle → New Notebook → File → Import Notebook → `notebooks/00_rebuild_hisar_dataset.ipynb`.
2. Settings: Internet **On**, Accelerator **None** (or GPU).
3. Save Version → **Save & Run All (Commit)**. Outputs: `hisar_dataset_final.zip`, `collages_hisar.zip`, `location_check.png`, `scene_previews.png`, `contact_sheet.png`, `scenes_chosen.csv`.

Local use of the scripts: `pip install -r requirements.txt` and `pip install torch sen2sr mlstac` for super-resolution.

---

Sentinel-2 imagery is open (Copernicus licence) and redistributable; commercial imagery (Google Earth, Planet) is not, and NAIP is US-only. SEN2SR bridges part of the resolution gap while keeping the dataset fully open.

## References

- Tousi et al. (2025), [A Zero-Shot Learning Approach for Ephemeral Gully Detection from Remote Sensing using Vision Language Models](https://arxiv.org/abs/2503.01169), WACV 2025 Workshops
- Aybar et al. (2026), [SEN2SR: A foundation model for Sentinel-2 super-resolution](https://doi.org/10.1016/j.rse.2025.115222), *Remote Sensing of Environment*
- [Missouri Ephemeral Gully Detection Dataset](https://data.mendeley.com/)
