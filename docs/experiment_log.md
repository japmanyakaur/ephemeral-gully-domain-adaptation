# Ephemeral Gully Detection in Hisar–Fatehabad–Sirsa: Project Status, Log and Results

*The one document for this project: where things stand, every step and result so far, and the plan, written so each part can go straight into the paper draft. Status on 9 November: dataset, labels, terrain analysis and all three pipelines (A, B, C) are done and on GitHub; the cross-resolution test, figures and the paper draft are next, on track for 30 November. AgriWEF 2027, IIT Kharagpur (conference 2–4 February 2027). Acceptance notification 15 November, registration by 25 November, full paper due 30 November 2026; proceedings or edited book for selected papers. Last updated 9 November 2026.*

## 1. Key results at a glance

These are the numbers to quote in the abstract, results and conclusion. Each one is explained in the sections below.

| Result | Value | Section |
|---|---|---|
| New dataset | 300 locations × 5 January dates (2020–2024), 1,497 Sentinel-2 images super-resolved to 2.5 m | 4 |
| How flat the study area is | Median terrain slope 0.45%; 99th percentile 2.0%; only 1.1% of the area steeper than 2% | 5 |
| Gullies in the 300 locations | 10 gully, 287 no gully, 3 skipped (3.4% of labelled locations are gullies) | 6 |
| Prompt design (Qwen2.5-VL-7B, Missouri) | Yes/no prompt: model says "gully" for 96% of fields, balanced accuracy 0.52. Evidence-first 1–5 rating on the 150 test fields: 27% "gully", balanced accuracy 0.60 | 8.1, 8.2 |
| Best model (Missouri, 150 test fields) | Gemma 4 26B: AUC 0.823, balanced accuracy 0.807 (LLaVA-Llama3 8B: 0.558; Qwen2.5-VL 7B: 0.601) | 8.3 |
| Gemma 4 26B on 471 unseen Missouri fields | AUC 0.862, balanced accuracy 0.817 (95% CI 0.78–0.85) | 8.4 |
| Gemma 4 26B on 297 Hisar fields | Flags 21 fields (7.1%); false-alarm rate 6.3% (18 of 287 no-gully fields; 95% CI 4.0–9.7%); finds 3 of 10 gullies (95% CI 11–60%); AUC 0.70; balanced accuracy 0.62 | 8.5 |
| Missouri vs Hisar | Gemma calls 59% of Missouri fields gully against 7% of Hisar fields; precision drops from 0.85 to 0.14 because gullies are rare in Hisar | 8.6 |
| Pipeline B (19 questions + reasoning), Missouri held-out | AUC 0.632, balanced accuracy 0.603; calls 90% of fields gully | 8.7 |
| Pipeline B on Hisar | Finds 7 of 10 gullies; false-alarm rate 28.6% (95% CI 23.7–34.1%); AUC 0.74 | 8.7 |
| Pipeline C (classifier on the 19 answers, trained on Missouri) on Hisar | Finds the same 7 of 10 gullies with false-alarm rate 20.9% (16.6–26.0%); removes 22 of Pipeline B's 82 false alarms and adds none (exact McNemar p < 0.001); AUC 0.79 (0.62–0.94) | 8.8 |
| Limited local labels | Retraining with 10–200 Hisar labels did not help (about 1 gully per 30 labelled fields). Using the local labels only to set the cut-off did: with 100 labels, about 54% of gullies found at a 7% false-alarm rate, against 30% at 6% for Pipeline A | 8.8 |
| Cost | About 5 seconds per location for Pipeline A and 11 seconds for Pipeline B on a free Kaggle GPU (2 × T4); Pipeline C trains in under a second on a laptop | 8.4, 8.7 |

**Main message for the paper.** A zero-shot vision-language model that works well on hilly US farmland (AUC 0.86) transfers poorly to flat, canal-irrigated farmland in Haryana. There, gullies are rare because the land is almost level, and most of the model's "gully" calls are false alarms. Gemma's own explanations show it reading irregular linear features such as tracks and field boundaries as erosion channels. The approaches trade off differently: direct prompting (A) gives few false alarms but finds 3 of 10 gullies; structured questions (B) find 7 of 10 but raise a false alarm on almost 3 in 10 fields; a small classifier trained on B's answers in Missouri (C) keeps the 7 and cuts B's false alarms by a quarter. With gullies this rare, a few hundred local labels contain too few gullies to retrain on, but they are enough to set the cut-off for the new region. All Hisar differences rest on 10 gullies, so their intervals are wide.

## 2. Step-by-step log

| Date | Step | Output |
|---|---|---|
| 28 Oct | Repository set up. Scripts for Sentinel-2 tiling, EDSR ×4 and SEN2SR ×4 super-resolution, a contact sheet for visual checks, and an EDSR vs SEN2SR comparison | 01–04 scripts |
| 29 Oct | Metadata cleaning and centre-crop scripts; tiling script can reuse the same locations on other dates (multi-temporal); first dataset uploaded (1,481 tiles) | 05, 06 scripts |
| 30 Oct – 1 Nov | README written; full-scene preview | README |
| 4 Nov | Collage script: one 2 × 3 image per location with all dates labelled | 07_make_collages.py |
| early Nov | Pipeline A tried on the laptop with Ollama. The laptop has no GPU (about 7 minutes per image), so all model runs moved to Kaggle's free GPUs | – |
| 6 Nov | First dataset found to be in the wrong Sentinel-2 tile (T43REM, Bhiwani–Jhunjhunu–Churu, south of the study area). Removed and rebuilt in tile T43REN with an automatic district check; released as v2.0 | 00_rebuild_hisar_dataset.ipynb, release v2.0 |
| 6 Nov | Collages made for Missouri (621) and Hisar (300) | collages_missouri, collages_hisar |
| 6 Nov | Experiment A1: yes/no prompt on all 621 Missouri locations | results/missouri/pipeline_a_missouri.csv |
| 6 Nov | Experiment A2: two 1–5 rating prompts on 150 Missouri test locations. The output files were later lost (overwritten in the Kaggle session), so the "short" prompt is not reported; the evidence prompt is reported from the A3 run of the same model | notebooks/08b_pipeline_a_rating.ipynb |
| 6 Nov | Experiment A3: open models compared on the same 150 locations. The rerun files for Gemma 4 31B and Qwen3-VL were later lost and Llama 3.2 Vision did not download, so only LLaVA-Llama3 8B, Qwen2.5-VL 7B and Gemma 4 26B are reported | results/missouri/, results/tables/model_comparison_test150.csv |
| 6 Nov | Experiment A4: Gemma 4 26B on all 621 Missouri and all 300 Hisar locations | results/pipeline_a_full/ |
| 7 Nov | Pipeline B notebook built (19 questions + reasoning step); robustness fixes (failed rows retried, fallback when the JSON format fails, stop after 10 errors in a row) | 09_pipeline_b.ipynb |
| 7–8 Nov | Labelling: the author labelled all 300 locations on a labelling web page that shows each location's five dates with links to Google Maps and Google Earth | labels/hisar_labels_final.csv |
| 8 Nov | Labels frozen: 10 gully, 287 no gully, 3 skip | labels/hisar_labels_final.csv |
| 8 Nov | Terrain analysis of the study area (slope from SRTM elevation) and a search of the steepest places in satellite imagery | 10_terrain_slope.py |
| 8 Nov | Pipeline A evaluated on Hisar against the frozen labels | Section 8.5 |
| 8 Nov | Pipeline B run on Kaggle: all 621 Missouri and 300 Hisar locations, no failed rows (run as an interactive draft session; the output files were downloaded before it closed) | results/pipeline_b/ |
| 8 Nov | Pipeline C built and run on a laptop CPU (about 2 minutes): Missouri cross-validation, transfer to Hisar, and four ways of using 10–200 local labels, 200 random draws each | 11_pipeline_c.py, results/pipeline_c/ |
| 8 Nov | Main results table for A, B and C on the same fields, with confidence intervals and a paired test | 12_results_tables.py, results/tables/ |
| 9 Nov | Terrain script run by the author and committed; the saved numbers match Section 5. Pipeline B, Pipeline C, results table, labels and this log pushed to GitHub. Pre-submission review: conference pages read (no template or page limit published), risks and strengths in Section 15, to-do list updated. GitHub clean-up: failed and partial result files removed; Tables 2 and 3 rebuilt only from saved files by 12_results_tables.py | results/terrain/ |

Claude also reviewed the locations in Google satellite imagery as a separate check. According to the author, this did not influence the labels, and it is not reported in the paper.

## 3. Problems met and how they were solved

These are useful for the methods and reproducibility parts of the paper.

| Problem | Fix |
|---|---|
| Laptop has no GPU: about 7 minutes per image | All model runs on Kaggle, 2 × NVIDIA T4, with Ollama |
| Draft (interactive) Kaggle runs lost their output files twice | Long runs only with Save Version → Save & Run All (Commit); every run saves a checkpoint and resumes; failed rows are retried, not counted as done |
| Two notebooks in one Kaggle session broke a run | One job per notebook |
| Small models copied the prompt's wording and answered "gully" for almost every field | Evidence-first 1–5 rating prompt (Experiments A1–A2) |
| Some model runs failed or were lost: Llama 3.2 Vision 11B did not download, Qwen3-VL-8B gave few valid answers, and the rerun files for Gemma 4 31B and Qwen3-VL were overwritten in the Kaggle session | Only models with saved per-location answers are reported (Table 3); failed files removed from GitHub; rule: download every output before the next run |
| First dataset was in the wrong tile (T43REM) | Rebuilt in T43REN; the notebook now stops if any tile falls outside Hisar, Fatehabad or Sirsa |
| Collage script had the old dates written into the code | It now reads the dates from the metadata |
| Which "gully" cut-off to use | Chosen once on the 150 Missouri test fields (score 3 or more = gully), then used unchanged on the 471 other Missouri fields and on Hisar |

## 4. Data

### 4.1 Missouri reference dataset

The dataset used by Tousi et al. (2025). We use it to check our setup and as the source region for transfer.

| Item | Value |
|---|---|
| Imagery | USDA NAIP aerial photographs, about 1 m, RGB |
| Locations | 621, each with 6 images over about 10 years (3,726 images) |
| Labels | 354 gully, 267 no gully (file GT_L6_Tol.json) |
| Test subset for prompt and model choice | 150 locations, 75 gully and 75 no gully, fixed random seed 42 |
| Held-out set | The other 471 locations (279 gully, 192 no gully), never used for any choice |

### 4.2 Hisar–Fatehabad–Sirsa dataset (our contribution, release v2.0)

| Item | Value |
|---|---|
| Imagery | Sentinel-2 Level-2A true colour (TCI), 10 m |
| Tile | T43REN (UTM zone 43N) |
| Search area | UTM 500,000–585,000 E, 3,215,000–3,285,000 N (85 × 70 km), then only tiles whose centre lies in Hisar, Fatehabad or Sirsa district |
| Locations | 300: Fatehabad 130, Hisar 94, Sirsa 76 |
| Images | 1,497 (297 locations have all 5 dates) |
| Resolution | 2.5 m after SEN2SR-Lite ×4 super-resolution |
| Tile size | 128 × 128 px = 320 m × 320 m |
| Metadata | Date, UTM corner, latitude and longitude of tile centre, district, vegetation fraction |

**Scenes.** For each year, the clearest scene between 1 January and 28 February, closest to 27 January, with cloud measured inside our area from the Sentinel-2 scene classification (SCL) layer.

| Date | Scene | Cloud, whole tile | Cloud, study area |
|---|---|---|---|
| 2020-01-23 | S2B_43REN_20200123 | 0.0% | 0.0% |
| 2021-01-27 | S2B_43REN_20210127 | 0.06% | 0.1% |
| 2022-01-27 | S2A_43REN_20220127 | 0.0% | 0.0% |
| 2023-01-27 | S2B_43REN_20230127 | 5.5% | 1.7% |
| 2024-01-27 | S2A_43REN_20240127 | 0.0% | 0.0% |

**Processing** (notebooks/00_rebuild_hisar_dataset.ipynb, which calls the repository scripts):

1. Scene search and download of only the study area from the Earth Search catalogue (Sentinel-2 open data on AWS).
2. Tiling (01_tile_sentinel2.py): 128 × 128 px windows with at least 90% valid pixels and at least 30% vegetation (green band brighter than red).
3. District filter: keep tiles whose centre is inside Hisar, Fatehabad or Sirsa (district boundaries from the geohacker/india GeoJSON); 300 kept, spread across the candidates.
4. The same 300 locations cut from the other four dates.
5. Cloud check: a tile-date is dropped if more than 5% of its central 320 m is cloud, cloud shadow or cirrus in the SCL layer.
6. Super-resolution (02b_super_resolve_sen2sr.py): SEN2SR-Lite, RGB plus synthetic NIR, 128 → 512 px.
7. Centre crop (06_center_crop.py): 512 → 128 px, 1,280 m → 320 m, to bring the footprint closer to Missouri's.
8. Dark or blank tiles removed; latitude, longitude and district added.
9. Location check: every tile tested against the district boundaries (100% inside) and plotted (results/hisar_build/location_check.png).
10. Collages (07_make_collages.py): one 2 × 3 image per location, the 5 dates labelled, plus a legend cell.

**Super-resolution choice.** On a 40-tile pilot set from the earlier test area, SEN2SR was sharper than EDSR in 10 of 10 sampled tiles (about 2.5 times higher sharpness score; entropy 5.68 bits, dynamic range 235 of 255). These figures come from the old area and are not used in the paper. The model was given the 8-bit true-colour product and a synthetic NIR band, not the surface-reflectance bands it was trained on (Section 15).

## 5. Terrain analysis: why gullies are rare here

**Method** (10_terrain_slope.py). Elevation from AWS Terrain Tiles (Terrarium encoding; in this region mostly SRTM, about 30 m). Zoom-12 tiles (about 33 m pixels), smoothed twice with a 5 × 5 average to suppress SRTM noise, slope from elevation differences 5 pixels either side (about 330 m), sampled every 3 pixels inside the three districts and the dataset's search area (about lat 29.06–29.69 °N, lon 75.00–75.87 °E). Computed on 8 November, then re-run with the script and committed (results/terrain/); the saved run gives median 0.45%, 99th percentile 2.03% and 1.07% of the area steeper than 2%.

| Measure | Value |
|---|---|
| Points sampled | 494,627 |
| Median slope | 0.45% (about 45 cm drop per 100 m) |
| 90th percentile | 1.0% |
| 99th percentile | 2.0% |
| Maximum | 6.0% |
| Share of the area steeper than 1% | 9.9% |
| Share of the area steeper than 2% | 1.1% |
| Share of the area steeper than 3% | 0.1% |

**Steepest places, checked in satellite imagery.** None is cropland with a gully:
- Agroha Mound (archaeological site, tree-covered), Hisar district
- canal and Ghaggar river embankments
- towns (Hisar, Sirsa)
- scrub and grazing land (for example the Central Sheep Breeding Farm north of Hisar)
- sand-dune farmland in south Sirsa, which was also searched: no gullies

A search of the Ghaggar banks and the dune area found no ephemeral gullies to add.

**For the paper.** Ephemeral gullies form where runoff concentrates on sloping cropland. The study area is almost level and irrigated by canals, so gullies are rare. This is consistent with Haryana's water-erosion problems being concentrated in the Shivalik hills in the north and the Aravalli hills in the south (Tribune India).

**Caveat.** SRTM is a surface model that includes trees and buildings, and its noise is about 1–2 m, so the smallest slopes are near the noise level. The conclusion "almost level" does not depend on this.

**Outputs.** results/terrain/terrain_slope_summary.csv, steepest_spots.csv and terrain_slope_map.png (the slope panel for Figure 1).

## 6. Labels

### 6.1 Protocol

- **Who:** the author (Japmanya), by visual interpretation.
- **Tools:** a labelling web page that shows each location's five January images side by side, with buttons that open the location in Google Maps and Google Earth.
- **Gully (1):** a thin, wavy or branching line that crosses the crop rows and follows a natural flow path towards a low corner or field edge.
- **No gully (0):** no such line. Straight or man-made lines are not gullies: canals, roads, farm tracks, field boundaries, tree rows, the Ghaggar river, harvest rows, irrigation wet patches, animal paths.
- **Skip:** the location cannot be judged (mostly houses or streets).
- **Dates:** 7–8 November 2026. Labels were frozen on 8 November and are not changed after seeing model results.

### 6.2 Final labels (labels/hisar_labels_final.csv)

| District | Gully | No gully | Skip | Total |
|---|---|---|---|---|
| Fatehabad | 3 | 126 | 1 | 130 |
| Hisar | 1 | 91 | 2 | 94 |
| Sirsa | 6 | 70 | 0 | 76 |
| **All** | **10** | **287** | **3** | **300** |

**Gully locations**

| Field | District | Latitude | Longitude |
|---|---|---|---|
| 3 | Sirsa | 29.677662 | 75.165361 |
| 15 | Fatehabad | 29.629809 | 75.680958 |
| 28 | Sirsa | 29.573676 | 75.178407 |
| 48 | Sirsa | 29.654641 | 75.072742 |
| 51 | Fatehabad | 29.583670 | 75.667432 |
| 63 | Fatehabad | 29.480845 | 75.376292 |
| 65 | Hisar | 29.352364 | 75.718663 |
| 71 | Sirsa | 29.492792 | 75.191470 |
| 117 | Sirsa | 29.619957 | 75.112381 |
| 143 | Sirsa | 29.296489 | 75.112026 |

Skipped: 288 and 290 (Hisar), 289 (Fatehabad), all built-up areas.

### 6.3 Limitations of the labels (for the paper)

- Visual interpretation, not field survey.
- One labeller; no second-labeller agreement measured. If time allows, a second person (a supervisor or anyone with an agriculture background) labels 30–50 random locations and the agreement is reported.
- A gully ploughed out before the available imagery can be missed.
- At 2.5 m a gully can be narrower than one pixel, so in the Sentinel-2 collages only indirect signs are visible.

## 7. Methods

### 7.1 Common setup

- Models run with Ollama on Kaggle (2 × NVIDIA T4, free tier).
- Temperature 0, seed 42, fixed JSON answer format (Ollama `format` with a JSON schema), so answers are repeatable and machine-readable. Context 8,192 tokens; answer length capped (700 tokens in A3–A4, 900 in Pipeline B).
- Input: one collage per location (Missouri: 6 dates; Hisar: 5 dates plus a legend cell).
- Metrics: AUC of the 1–5 score (0.5 = guessing, 1 = perfect); balanced accuracy (mean of the accuracy on gully and no-gully fields; 0.5 = guessing); accuracy; F1; precision; sensitivity (share of gullies found); specificity (share of no-gully fields correctly left alone); false-alarm rate = 1 − specificity; share of fields called gully. 95% intervals: bootstrap for Missouri, Wilson intervals for Hisar proportions.
- Cut-off: chosen on the 150 Missouri test fields (best balanced accuracy), then fixed. For Gemma 4 26B this is score ≥ 3.

### 7.2 Pipeline A: direct zero-shot prompting

The model sees the collage and answers in one step. Three prompt versions were tested; the full texts are in Appendix A.
- **A1, yes/no with a detailed gully description** (expert role, description of what a gully looks like, list of things not to count).
- **A2 "short":** short definition, "many fields have no gully", 1–5 score only. Tried, but its output file was lost, so it is not reported.
- **A2 "evidence":** same, but the model first lists every linear feature and says what it most likely is (crop rows, field boundary, road, tractor track, canal or erosion channel), then gives the 1–5 score. Used for all later runs.

### 7.3 Pipeline B: structured questions with reasoning

Following Tousi et al. (2025): the vision model (Gemma 4 26B) answers the paper's 19 yes/no questions about the collage in one call, plus a one-sentence description. A second, text-only call to the same model reads the description and the 19 answers, reasons briefly and gives a 1–5 score. Question 13 is cut off in the published paper and was completed as "…across the images?". Full questions in Appendix B. Cut-off chosen on the 150 Missouri test fields as for Pipeline A: score ≥ 3 (balanced accuracy 0.627; score ≥ 4 gave the same value, and the lower one was kept).

### 7.4 Pipeline C: lightweight classifier on VLM answers, with limited local labels

- **Features:** the 19 yes/no answers from Pipeline B. A second version adds the Pipeline A score (20 features).
- **Models:** logistic regression with class weights (main model), and a small neural network (one hidden layer of 16 units, as in Tousi et al.'s transfer-learning pipeline, trained on class-balanced resampled data) as a check. Cut-off: probability ≥ 0.5.
- **Experiment C1, Missouri only:** 5-fold stratified cross-validation on the 621 Missouri locations, repeated 5 times.
- **Experiment C2, transfer with no local labels:** train on Missouri (all 621, and test150 only for the main table), test on the 297 Hisar locations.
- **Experiment C3, limited local labels:** draw k labelled Hisar locations at random (k = 10, 25, 50, 100, 150, 200; 200 draws each; draws whose remaining part has no gully are skipped, which happened 5 times at k = 200) and test on the Hisar locations not drawn. The k labels are used in four ways: (a) added to the Missouri training data; (b) added with extra weight, so Hisar and Missouri carry equal total weight; (c) a classifier trained on the k Hisar labels alone; (d) the Missouri-trained classifier kept unchanged, with the k labels used only to set its cut-off (placed just above the scores of 90% of the no-gully fields among the k labels, aiming at about 10% false alarms). Pipelines A and B are scored on exactly the same test locations in every draw.
- **Intervals and tests:** bootstrap for AUC and balanced accuracy; Wilson intervals for rates on Hisar; exact McNemar test for paired false alarms of B and C on the same Hisar fields.

## 8. Results

### 8.1 Experiment A1: yes/no prompt (Qwen2.5-VL-7B, all 621 Missouri locations)

| Metric | Value |
|---|---|
| Accuracy | 0.586 |
| Majority baseline (always "gully") | 0.570 |
| Balanced accuracy | 0.521 |
| Precision / recall / F1 | 0.581 / 0.983 / 0.730 |
| Said "gully" | 599 of 621 (96%) |
| Confusion matrix | No-gully: 16 right, 251 wrong. Gully: 348 right, 6 wrong |

The model echoed the prompt: "natural drainage path" appeared in 89% of its explanations and "slightly curving" in 83%, both phrases from the prompt. Tousi et al.'s smallest model (LLaVA-Llama3-8B) also labelled every field as gully in their Pipelines B and C.

### 8.2 Experiment A2: prompt comparison (Qwen2.5-VL-7B)

| Prompt | Fields | AUC | Balanced accuracy | Said "gully" |
|---|---|---|---|---|
| yes/no (A1) | all 621 | – | 0.521 | 96.5% |
| yes/no (A1) | 150 test fields | – | 0.533 | 96.7% |
| evidence first, 1–5 score | 150 test fields | 0.601 | 0.600 | 27% |

On the same 150 fields, asking for evidence first and a 1–5 score removed the yes-bias (96.7% → 27% "gully") and raised balanced accuracy from 0.53 to 0.60. 90 of 150 correct is unlikely to be chance (one-sided binomial test, p = 0.009). Source: results/tables/prompt_comparison_qwen2.5vl.csv (12_results_tables.py).

### 8.3 Experiment A3: open models (evidence prompt, 150 Missouri test locations)

| Model | Size | AUC | Balanced accuracy | Said "gully" | s / location | Note |
|---|---|---|---|---|---|---|
| LLaVA-Llama3 | 8B | 0.558 | 0.560 | 11% | 1.9 | Tousi et al.'s small model |
| Qwen2.5-VL | 7B | 0.601 | 0.600 | 27% | 4.7 | |
| **Gemma 4** | **26B** | **0.823** | **0.807** | 52% | 5.1 | Best; from the full run (Section 8.4) |

Three more models were tried (Gemma 4 31B, Qwen3-VL 8B, Llama 3.2 Vision 11B). Llama 3.2 Vision did not download, Qwen3-VL gave few valid answers, and the rerun files of Gemma 4 31B and Qwen3-VL were lost, so none of them is reported. The balanced accuracies use a cut-off chosen on these same 150 fields, so they are slightly optimistic; AUC does not depend on a cut-off. Source: results/tables/model_comparison_test150.csv (12_results_tables.py).

### 8.4 Experiment A4: Gemma 4 26B on all of Missouri (evidence prompt)

Cut-off score ≥ 3, chosen on the 150 test fields.

| Set | Fields | AUC | Balanced accuracy | Accuracy | F1 | Said "gully" |
|---|---|---|---|---|---|---|
| Test (used to choose the cut-off) | 150 | 0.823 | 0.807 | 0.807 | 0.810 | 52% |
| **Held-out (never used for any choice)** | **471** | **0.862** | **0.817** (95% CI 0.78–0.85) | 0.824 | 0.852 | 59% |
| All | 621 | 0.853 | 0.815 | 0.820 | 0.843 | 58% |

- **Held-out confusion matrix:** no-gully fields 150 right, 42 wrong; gully fields 238 right, 41 wrong.
- **Scores, all 621:** 1 → 194, 2 → 69, 3 → 25, 4 → 265, 5 → 68.
- **Runtime:** about 5 seconds per location.
- **Comparison with Tousi et al.:** they report 0.714 accuracy for Qwen2-VL-72B on their own test split (Section 10), so our 0.82 is a rough comparison only.

Note: the notebook's own summary file (model_comparison_full.csv, removed from GitHub on 9 Nov) used a cut-off chosen on all 621 fields; the paper uses the protocol numbers above.

### 8.5 Gemma 4 26B on Hisar (297 labelled locations, frozen labels)

Same prompt and cut-off as Missouri (score ≥ 3).

| Metric | Value |
|---|---|
| Scores, all 300 | 1 → 262, 2 → 17, 3 → 2, 4 → 19 |
| Called "gully" | 21 of 297 (7.1%) |
| Confusion matrix | Gully: 3 found, 7 missed. No gully: 269 right, 18 false alarms |
| Specificity | 0.937 (95% CI 0.903–0.960) |
| **False-alarm rate** | **6.3% (95% CI 4.0–9.7%)** |
| Sensitivity (gullies found) | 0.30 (95% CI 0.11–0.60) |
| Precision | 0.14 (6 false alarms for every correct "gully") |
| Balanced accuracy | 0.62 |
| AUC | 0.70 |
| Accuracy | 0.916 (always answering "no gully" would score 0.966) |
| Runtime | 4.7 s per location |

**With cut-off 4 instead of 3 (sensitivity check):** 19 called gully; 3 found, 16 false alarms; false-alarm rate 5.6%; same AUC. The conclusion does not depend on the cut-off.

**By district (share called gully):** Sirsa 10.5% (8 of 76), Fatehabad 6.2% (8 of 129), Hisar 5.4% (5 of 92).

**The 3 gullies found:** 3, 51, 117. **The 7 missed** (score): 15 (2), 28 (1), 48 (1), 63 (1), 65 (1), 71 (1), 143 (2).

**The 18 false alarms:** 26, 31, 44, 94, 101, 104, 115, 138, 161, 168, 173, 185, 194, 211, 214, 225, 248, 282. In each case Gemma's explanation describes an irregular or winding line and calls it a possible erosion channel. In 14 of the 18, the explanation itself also mentions a road, track or path among the features. Examples:
- 26: "a dark, diagonal linear feature… which appears to be a field boundary or a small road/path"
- 101: "light-coloured, winding, irregular paths… which most likely represent erosion channels (gullies) or unpaved tracks/paths"
- 282: "a prominent, light-coloured, winding feature… which most likely represents a road or a large, established erosion channel"

**Error-analysis figure (to make):** 4–6 false-alarm collages with Gemma's explanation next to the Google satellite view of the same place.

**Do not report sensitivity as a firm number.** With 10 gullies, the 95% interval runs from 11% to 60%. The robust Hisar result is the false-alarm rate.

### 8.6 Missouri vs Hisar (Gemma 4 26B, same prompt and cut-off)

| | Missouri held-out | Hisar |
|---|---|---|
| Imagery | NAIP aerial, about 1 m, 6 dates over about 10 years | Sentinel-2 10 m → 2.5 m, 5 January dates |
| Terrain | Rolling farmland | Almost level (median slope 0.45%) |
| Locations | 471 | 297 |
| Gully share (labels) | 59% (279) | 3.4% (10) |
| Called "gully" by Gemma | 59% | 7.1% |
| AUC | 0.862 | 0.70 |
| Balanced accuracy | 0.817 | 0.62 |
| Sensitivity | 0.85 | 0.30 |
| Specificity | 0.78 | 0.94 |
| Precision | 0.85 | 0.14 |

The model is not simply over-calling in Hisar: it calls far fewer fields gully there (7% vs 59%). Gullies are so rare that even a 6% false-alarm rate gives six false alarms for every real gully found. For monitoring in a region like this, the false-alarm rate matters more than accuracy.

### 8.7 Pipeline B: 19 questions + reasoning (Gemma 4 26B)

All 621 Missouri and 300 Hisar locations answered; no failed rows. Cut-off score ≥ 3.

| | Missouri held-out (471) | Hisar (297) |
|---|---|---|
| AUC | 0.632 (95% CI 0.588–0.675) | 0.740 (0.587–0.866) |
| Balanced accuracy | 0.603 (0.574–0.635) | 0.707 (0.554–0.847) |
| Called "gully" | 90% | 30% |
| Gullies found | 274 of 279 (0.98) | 7 of 10 (0.70; 0.40–0.89) |
| False-alarm rate | 77.6% (149 of 192) | 28.6% (82 of 287; 23.7–34.1%) |
| Precision | 0.65 | 0.08 |

- **Scores, Missouri (621):** 1 → 49, 2 → 18, 3 → 1, 4 → 150, 5 → 403. **Hisar (300):** 1 → 120, 2 → 89, 3 → 2, 4 → 28, 5 → 61.
- **Test150 (used for the cut-off):** AUC 0.634, balanced accuracy 0.627.
- **Runtime:** about 7.5 s for the 19 answers plus 3.6 s for the reasoning step, so about 11 s per location (twice Pipeline A).
- **Gullies found on Hisar:** 3, 15, 48, 51, 65, 117, 143. **Missed:** 28, 63, 71 (also missed by Pipelines A and C).
- **Why B over-calls in Missouri:** the yes/no answers are saturated there. No-gully Missouri fields still get "yes" on most questions (for example, question 2: 61% of no-gully fields against 95% of gully fields). In Hisar, no-gully fields get "yes" far less often (question 2: 23%; gully fields 70%), so the answers separate the classes better there. On average a field gets 16.8 "yes" answers (Missouri gully), 13.2 (Missouri no gully), 13.4 (Hisar gully) and 7.9 (Hisar no gully).
- **Questions that carry no information:** questions 6, 8, 11, 15 and 19 are answered "yes" for about 90–100% of all fields in both regions. The 19 answers form only 34 distinct patterns in Missouri and 28 in Hisar.
- **Comparison with Tousi et al.:** their Pipeline B (Llama3.2-90B + Llama3.2) reached 0.643 accuracy; ours reaches 0.603 balanced accuracy on held-out Missouri with a 26B model. Different splits, so this is a rough comparison only.

### 8.8 Pipeline C: classifier on the 19 answers, with limited local labels

**C1, Missouri cross-validation (621 locations, 5 × 5 folds).**

| Features | Model | AUC | Balanced accuracy |
|---|---|---|---|
| 19 answers | Logistic regression | 0.700 ± 0.049 | 0.677 ± 0.038 |
| 19 answers | Neural network | 0.696 ± 0.049 | 0.669 ± 0.039 |
| 19 answers + Pipeline A score | Logistic regression | 0.861 ± 0.029 | 0.822 ± 0.033 |
| 19 answers + Pipeline A score | Neural network | 0.852 ± 0.033 | 0.818 ± 0.033 |

In Missouri a classifier on the answers beats B's own reasoning step (AUC 0.70 against 0.63), but stays well below Pipeline A (0.86). Adding A's score gives about the same as A alone. The neural network gives the same results as logistic regression, so logistic regression is used from here on.

**C2, transfer to Hisar with no local labels (297 locations, 10 gullies).**

| Classifier | Trained on | AUC | Gullies found | False-alarm rate |
|---|---|---|---|---|
| 19 answers, logistic regression | All 621 Missouri | 0.790 (0.622–0.942) | 7 of 10 | 20.9% (60 of 287; 16.6–26.0%) |
| 19 answers, logistic regression | Missouri test150 only | 0.720 (0.553–0.871) | 7 of 10 | 22.0% (63 of 287; 17.6–27.1%) |
| 19 answers, neural network | All 621 Missouri | 0.775 | 7 of 10 | 21.3% |
| 19 answers + A score, logistic regression | All 621 Missouri | 0.763 | 3 of 10 | 5.9% |

- **C against B on the same fields:** C flags the same 7 gullies. Its 60 false alarms are all among B's 82; it removes 22 and adds none (exact McNemar test, p = 4.8 × 10⁻⁷). Trained on test150 only, it removes 19 and adds none (p = 3.8 × 10⁻⁶).
- **AUC differences on Hisar are not significant:** C − A = +0.095 (95% bootstrap interval −0.026 to +0.253); C − B = +0.051 (−0.035 to +0.128).
- **With the A score added,** the classifier behaves like Pipeline A on Hisar (few false alarms, 3 of 10 found).
- **What the Missouri classifier relies on** (largest weights): questions 6, 17, 8, 2, 3, 7, 1 and 13 (results/pipeline_c/pipeline_c_question_weights.csv).

**C3, limited local labels (mean over 200 random draws; 19 answers; tested on the Hisar locations not drawn).**

| k Hisar labels | (a) Missouri + k: AUC / false alarms / found | (c) k only: AUC / false alarms / found | (d) Missouri model, cut-off set on k: false alarms / found | Pipeline A, same fields: false alarms / found | Pipeline B, same fields: false alarms / found |
|---|---|---|---|---|---|
| 10 | 0.78 / 21% / 70% | 0.56 / 10% / 13% | 7.1% / 39% | 6.3% / 30% | 28.6% / 70% |
| 25 | 0.78 / 21% / 70% | 0.62 / 15% / 24% | 8.7% / 51% | 6.2% / 30% | 28.6% / 70% |
| 50 | 0.78 / 21% / 70% | 0.62 / 20% / 35% | 6.3% / 47% | 6.2% / 31% | 28.5% / 70% |
| 100 | 0.79 / 21% / 70% | 0.63 / 19% / 42% | 7.1% / 54% | 6.3% / 30% | 28.4% / 70% |
| 150 | 0.78 / 22% / 70% | 0.69 / 18% / 50% | 7.2% / 55% | 6.1% / 29% | 28.4% / 70% |
| 200 | 0.77 / 23% / 68% | 0.70 / 19% / 54% | 7.8% / 57% | 6.3% / 26% | 28.9% / 68% |

- **(a) and (b):** adding local labels to the Missouri training data changed nothing, even when the Hisar labels were given equal total weight with Missouri (b: AUC 0.76–0.77, false alarms 22–23%, found 68–70%). A random set of k Hisar fields contains only k/30 gullies on average (about 3 in 100).
- **(c):** a classifier trained on Hisar labels alone is worse than the Missouri-trained one at every k up to 200, and very unstable (AUC standard deviation 0.15–0.22 across draws).
- **(d):** using the local labels only to set the cut-off works. From k = 25 on, it finds about half of the remaining gullies at a 6–9% false-alarm rate. Against Pipeline A on the same fields, it finds on average 16–31 percentage points more gullies (more in 65–82% of draws, fewer in 0–25%), with false alarms within 2.5 points of A's. This uses mostly no-gully labels, which are easy to collect.
- **Caveat:** all draws share the same 10 gullies, so the spread across draws does not remove the uncertainty of having 10 gullies. At k = 10 the cut-off is set from about 10 no-gully fields and varies a lot (false-alarm standard deviation 9 points), and there is no gain over Pipeline A (more gullies found in 46% of draws, fewer in 52%).
- **Figure 6:** results/pipeline_c/label_efficiency.png (ranking quality, false alarms and gullies found against k).

### 8.9 All three pipelines on the same fields (main results table)

From results/tables/main_results.csv (12_results_tables.py). Cut-offs fixed on Missouri test150; nothing tuned on the held-out Missouri fields or on Hisar.

| Region | Pipeline | AUC (95% CI) | Balanced accuracy (95% CI) | Gullies found | False-alarm rate | Called "gully" | Precision |
|---|---|---|---|---|---|---|---|
| Missouri held-out (471) | A | **0.862** (0.827–0.894) | **0.817** (0.780–0.852) | 238/279 = 0.85 | **21.9%** | 59% | 0.85 |
| Missouri held-out (471) | B | 0.632 (0.588–0.675) | 0.603 (0.574–0.635) | 274/279 = 0.98 | 77.6% | 90% | 0.65 |
| Missouri held-out (471) | C (trained on test150) | 0.693 (0.644–0.743) | 0.667 (0.632–0.706) | 259/279 = 0.93 | 59.4% | 79% | 0.69 |
| Hisar (297) | A | 0.695 (0.537–0.856) | 0.619 (0.474–0.772) | 3/10 | **6.3%** (4.0–9.7) | 7% | 0.14 |
| Hisar (297) | B | 0.740 (0.587–0.866) | 0.707 (0.554–0.847) | 7/10 | 28.6% (23.7–34.1) | 30% | 0.08 |
| Hisar (297) | C (trained on test150) | 0.720 (0.553–0.871) | 0.740 (0.585–0.882) | 7/10 | 22.0% (17.6–27.1) | 24% | 0.10 |
| Hisar (297) | C (trained on all 621 Missouri) | **0.790** (0.622–0.942) | **0.745** (0.592–0.885) | 7/10 | 20.9% (16.6–26.0) | 23% | 0.10 |

**How to read it.** In Missouri, direct prompting (A) is clearly best. In Hisar the order of AUC reverses (C > B > A), but the intervals overlap because there are only 10 gullies. The firm Hisar findings are the paired ones: C removes about a quarter of B's false alarms without losing a gully, and A keeps false alarms lowest but misses 7 of 10 gullies. Gullies 28, 63 and 71 are missed by all three pipelines.

**Other measures** (12_results_tables.py). Average precision on Hisar, where chance is 0.034: A 0.091, B 0.072, C trained on all 621 Missouri 0.210, C trained on test150 0.082. Held-out Missouri in Tousi et al.'s measures (accuracy, F1 for gully, macro F1): A 0.824, 0.852, 0.817; B 0.673, 0.781, 0.569; C 0.715, 0.794, 0.666.

## 9. Figures and tables for the paper

| # | Content | Source | Status |
|---|---|---|---|
| Figure 1 | Study area map: districts, 300 locations, terrain slope | results/hisar_build/location_check.png; 10_terrain_slope.py map | Done: location map (results/hisar_build/location_check.png) and slope map (results/terrain/terrain_slope_map.png) |
| Figure 2 | Example collages, Missouri and Hisar | collages | To pick |
| Figure 3 | Pipelines A, B, C diagram | – | To make |
| Figure 4 | Score distributions, Missouri gully / no gully vs Hisar | A4 results | To make |
| Figure 5 | False-alarm examples with Gemma's explanation | Section 8.5 | To make |
| Figure 6 | Label efficiency (Pipeline C) | results/pipeline_c/label_efficiency.png | Draft done; restyle for the paper |
| Table 1 | Datasets (Missouri vs Hisar) | Section 4 | Ready |
| Table 2 | Prompt comparison (Qwen2.5-VL-7B) | Section 8.2, results/tables/prompt_comparison_qwen2.5vl.csv | Ready |
| Table 3 | Model comparison (A3) | Section 8.3, results/tables/model_comparison_test150.csv | Ready |
| Table 4 | Pipelines A, B, C on Missouri held-out and Hisar | Section 8.9, results/tables/main_results.csv | Ready |
| Table 5 | Runtime per model on a free GPU | Sections 8.3, 8.7 | Ready |
| Table 6 | Limited local labels | Section 8.8 (C3) | Ready |

## 10. Reference numbers from Tousi et al. (2025), Missouri test split

| Pipeline | Model | Accuracy | Macro F1 |
|---|---|---|---|
| A (direct) | Qwen2-VL-72B | 0.714 | 0.671 |
| A (direct) | Llama3.2-90B | 0.569 | 0.501 |
| A (direct) | LLaVA-Llama3-8B | 0.437 | 0.383 |
| A (direct) | CLIP | 0.572 | 0.533 |
| B (questions + reasoning) | Llama3.2-90B + Llama3.2 | 0.643 | 0.642 |
| Transfer learning (MLP) | Llama3.2-90B + MLP | 0.646 | 0.642 |

Their test split differs from our 150-location subset, so these are rough reference points. The 72B and 90B models do not fit on free Kaggle GPUs.

## 11. Limitations to state in the paper

- Hisar labels: visual interpretation, one labeller, no field survey; gullies ploughed out before the imagery may be missed.
- Only 10 gullies in Hisar, so sensitivity has a very wide interval; the false-alarm rate is the reliable Hisar measure.
- At 2.5 m (super-resolved from 10 m) a gully can be narrower than one pixel.
- Prompt and cut-off were chosen on Missouri, then applied to Hisar unchanged (intended: this is the transfer test).
- Repeated runs vary by about 0.02 AUC even at temperature 0.
- Our Missouri test subset differs from Tousi et al.'s split.
- The largest models (72B, 90B) could not be run on free GPUs.
- The slope figures come from SRTM, a surface model with about 1–2 m noise.
- All Hisar comparisons between pipelines rest on 10 gullies. Only the paired false-alarm comparison of B and C is statistically clear; AUC differences on Hisar are not.
- In the local-label experiment every random draw reuses the same 10 gullies, so the averages are more stable than the evidence behind them.
- Local labels: variants (a) and (c) were planned first; (b) and (d) were added after the first run showed that (a) and (c) did not help. The 10% false-alarm aim for (d) was set before (d) was run and was not changed. All four variants are reported.
- The January images come after the post-monsoon tillage that fills ephemeral gullies, so gullies formed in the monsoon may already be erased.
- SEN2SR was given the 8-bit true-colour product and a synthetic NIR band, not the surface-reflectance bands it was trained on.
- Missouri and Hisar differ in more than location: sensor, resolution, footprint (128 m vs 320 m), season, labellers and gully share.

## 12. Still to do

| Step | Who | When |
|---|---|---|
| Cross-resolution test (abstract promise): Pipeline A (Gemma 4 26B) on Missouri at 2.5 m and 10 m detail, and on Hisar at plain 10 m without super-resolution; the notebook also records the Ollama and model versions | Claude prepares; Japmanya runs on Kaggle (about 2.5 h) | 10–13 Nov |
| Commit the GitHub clean-up: remove the failed and partial result files, add results/missouri/test150_ids.csv, the updated 12_results_tables.py, the rebuilt Tables 2 and 3 and this log | Japmanya | 10 Nov |
| Check whether the Missouri download marks Tousi et al.'s split (310 development, 311 test), for example a field in GT_L6_Tol.json; if it does, re-score A, B and C on their 311 test fields | Japmanya checks (5 min); Claude re-scores | 10 Nov |
| Email pjteam@agriwefiitkgp.in: template, page limit, theme, and the early-bird date (the website says 15 Nov, the brochure 25 Nov) | Japmanya | 10 Nov |
| Register after the acceptance email (15 Nov), by 25 Nov; student fee Rs 2,500 early or Rs 3,000 late, non-refundable | Japmanya | 15–25 Nov |
| Repository: licence (code MIT; labels and derived images CC BY 4.0 with "Contains modified Copernicus Sentinel data 2020–2024"), README update (labels, Pipelines B and C, results, correct Missouri dataset link, remove the old v1.0 note), Zenodo archive for a citable DOI | Claude drafts; Japmanya commits | 12–20 Nov |
| Extra analyses without new model runs: average precision and Tousi et al.'s measures (added to 12_results_tables.py); terrain slope at each location, gully vs no gully | Claude; Japmanya re-runs the terrain script (1 min) | 11–13 Nov |
| Figures 2–6 and the final tables. Figure 2 uses only examples with clear evidence; no Google Earth screenshots unless Google's attribution rules are met | Claude | 12–16 Nov |
| Optional, only if time: redo super-resolution from the real reflectance bands (only if the cross-resolution test shows that super-resolution changes the results); one-date vs five-date test on Hisar | – | If time |
| Paper draft, section by section | Claude drafts, Japmanya edits | 16–26 Nov |
| AI-use statement: AI tools were used for coding, analysis support and writing; check the AgriWEF and publisher policy and add the required statement | Japmanya | Before submission |
| Formatting and submission | Japmanya | 27–30 Nov |

## 13. What the abstract promises and where each part stands

The abstract promises to evaluate, assess and quantify. It does not promise an accuracy figure, so a weak result on Hisar is still a valid outcome.

| Abstract promise | Status | Where it goes in the paper |
|---|---|---|
| Evaluate the Tousi et al. (2025) zero-shot VLM approach | Done on Missouri (Pipeline A) | Methods, Results |
| Hisar–Sirsa agricultural belt, 10 m Sentinel-2 | Done: tile T43REN, 300 locations in Hisar, Fatehabad and Sirsa | Study area and data |
| Over 1,400 tiles, 300 locations, rabi windows 2020–2024 | Done: 1,497 images, 5 January scenes | Study area and data |
| SEN2SR super-resolution to 2.5 m | Done | Data processing |
| Each location labelled with visual interpretation and Google Earth | Done (8 Nov): 10 gully, 287 no gully, 3 skip | Data, Evaluation |
| Direct zero-shot VLM prompting on multi-temporal imagery | Done on Missouri and Hisar; on Hisar the false-alarm rate is 6.3% and 3 of 10 gullies are found | Results |
| Structured visual question answering with reasoning-based aggregation | Done (Pipeline B): Missouri held-out AUC 0.63; Hisar 7 of 10 gullies found, false-alarm rate 28.6% | Methods, Results |
| Lightweight classifier on VLM features with limited local labels | Done (Pipeline C): Hisar 7 of 10 found, false-alarm rate 20.9%; local labels help when used to set the cut-off, not when used to retrain | Methods, Results |
| Cross-region and cross-resolution generalisation; how much local labels recover | Cross-region done for A, B and C (Sections 8.6, 8.9); local-label curve in Section 8.8. Cross-resolution not yet separated: Missouri (about 1 m aerial) and Hisar (10 m Sentinel-2, super-resolved to 2.5 m) differ in both region and resolution, so a Missouri run at reduced resolution is planned | Results, Discussion |

## 14. Paper outline

Planned sections, with the figures and tables listed in Section 9.

| Section | Content | Figures and tables |
|---|---|---|
| Introduction | Ephemeral gullies, why detection matters in India; gap: methods tested only on US aerial imagery | – |
| Related work | Tousi et al. (2025) and its follow-up; satellite erosion mapping; SEN2SR | – |
| Study area and data | Hisar–Fatehabad–Sirsa, terrain, Sentinel-2 T43REN, dates, processing, labelling protocol; Missouri reference data | Figures 1–2, Table 1 |
| Methods | Pipelines A, B and C; prompts; models; metrics; cut-off protocol | Figure 3; prompts in an appendix |
| Results | Prompt comparison, model comparison, Missouri vs Hisar, false alarms, label efficiency, runtime | Tables 2–5, Figures 4–6 |
| Discussion | Why transfer is hard (terrain, rarity, resolution, man-made lines), what works on a free GPU, limitations | – |
| Conclusion | Main numbers and what they mean for low-cost monitoring | – |

## 15. Risks and strengths (pre-submission review, 9 November)

- Few gullies in flat terrain: confirmed (Section 5). Hisar is used as a low-prevalence test (false-alarm rate and specificity), with Missouri supplying the gully examples for Pipeline C.
- Sub-pixel gullies: at 2.5 m a gully can be narrower than one pixel, so only indirect signs show in the Sentinel-2 collages. The paper says so.
- Kaggle GPU quota: about 30 GPU hours a week. Pipeline B asks all 19 questions in one call to keep runs to a few hours.
- **Can the labels be checked?** At 10 m (2.5 m after super-resolution) an ephemeral gully is usually thinner than a pixel, so in our images roads, drains and tree lines are often clearer than the gully itself, and the labels relied on Google Earth. One labeller, no field visit. Decision (9 Nov): no second labeller; the paper says plainly that the labels come from one annotator using our images and Google Earth, without independent checking or a field visit, and Figure 2 uses only examples whose evidence is clear.
- **Can 10 m imagery show ephemeral gullies at all?** Response: the cross-resolution test (Missouri at 1, 2.5 and 10 m; Hisar with and without super-resolution).
- **Season.** The January (rabi) images come after the post-monsoon tillage that fills ephemeral gullies, so gullies formed in the monsoon may already be erased. Response: explain the choice (cloud-free, the same crop stage every year, as in the abstract) and name post-monsoon imagery as future work.
- **Super-resolution input.** SEN2SR expects surface-reflectance bands (B04, B03, B02 and B08, divided by 10,000). We gave it the 8-bit true-colour product and a synthetic NIR band computed from green and red. Response: state this exactly in the methods; the Hisar test with and without super-resolution shows how much it matters; redo it with the real bands only if it does.
- **Many differences at once.** Missouri and Hisar differ in sensor, resolution, footprint (128 m vs 320 m), dates and season, labellers, and gully share (57% vs 3.4%), so the drop cannot be put on "region" alone. Response: list these in the paper; the cross-resolution test separates resolution.
- **Indirect comparison with Tousi et al.** Different split, measures and models; our Pipeline B asks all 19 questions in one call and uses our own decision prompt. Response: re-score on their split if it is available; otherwise report their measures on our held-out set and call the comparison indirect. On the 471 held-out Missouri fields Pipeline A reaches accuracy 0.824 and macro F1 0.817 (their best: 0.714 and 0.671).
- **Few gullies.** Only 10; AUC differences on Hisar are not significant; local-label variants (b) and (d) were added after the first run. Response: intervals and paired tests; call the local-label result exploratory; add average precision (Hisar, chance level 0.034: A 0.091, B 0.072, C 0.210).
- **Reproducibility gaps.** Model versions and quantisation are not recorded, and the repository has no licence. Response: Section 12. (Fixed on 9 Nov: failed result files removed; Tables 2 and 3 now come only from saved files.)
- **Names.** Tousi et al.'s "Pipeline C" is a reasoning pipeline; ours is a classifier (their "transfer learning"). Response: define the names once, clearly.
- **Audience.** AgriWEF reviewers are likely to be agricultural and soil-and-water engineers. Response: frame the paper around soil conservation, give the standard gully definition, explain the terrain, and end with practical advice.

**Strong points to put forward**

- **New open dataset:** 300 locations × 5 January dates (1,497 Sentinel-2 images at 2.5 m) with coordinates, districts and labels, rebuilt end to end by a public notebook.
- **First test outside the US, to our knowledge:** vision-language gully detection on free satellite imagery in an Indian farming landscape.
- **Strong result on free hardware:** an open 26B model on a free Kaggle GPU reaches AUC 0.86 and accuracy 0.82 on 471 held-out Missouri fields, above the numbers reported for 72B and 90B models (different split, so the comparison is indirect).
- **Careful protocol:** cut-offs fixed on a separate subset; held-out evaluation; labels frozen before seeing results; every model and prompt reported, including failures; intervals and paired tests.
- **Useful lessons:** asking for evidence first removes the yes-bias (96.7% → 27% "gully" on the same 150 fields); a small classifier on the 19 answers removes a quarter of Pipeline B's false alarms without losing a gully (p < 0.001); where gullies are rare, local labels are better spent on setting the cut-off than on retraining.
- **Physical explanation:** the terrain analysis (median slope 0.45%) explains why gullies are rare and ties the AI result to the landscape.
- **Low cost:** about 5 seconds per location, so a district can be screened in hours.
- **Fits the conference:** Theme 5 (ML for soil and crop health; resource conservation) and Theme 1 (water-driven natural hazards).

## 16. Working rules

- One job per Kaggle notebook. Never import a second notebook into a session that is still running.
- Always start long runs with Save Version → Save & Run All (Commit), never Run All in the editor.
- After every run, download the Output files and push them to GitHub, one commit per change. Download them before starting another run in the same Kaggle session, or they are overwritten (this is how the A2 and A3 rerun files were lost).
- Keep the Missouri test subset fixed (the same 150 fields, seed 42) so all comparisons are fair.
- Report every model and prompt that was tried, not only the best one.
- Labels are frozen. Do not look at Pipeline A or B answers while labelling, and do not change labels after seeing model results.
- Before submission, add the AI-use statement the venue requires (AI tools were used for coding, analysis support and writing).

## Appendix A. Pipeline A prompts (verbatim)

**A1 (yes/no):**

> You are an expert in soil erosion and remote sensing.
>
> The image is a collage of {n} overhead images of the SAME agricultural field, taken at different times. Each tile has its date or image number written in the top-left corner. Ignore any dark grey "no image" tile and any light grey text tile - they are not imagery.
>
> An ephemeral gully is a shallow channel carved by concentrated rainwater runoff in cropland. Farmers fill it during tillage, but it tends to reappear in the same place. In overhead imagery it looks like a narrow, often slightly curving or branching line - bare or discoloured soil, a darker or lighter strip, or a gap in the crop - that follows the natural drainage path and usually shows up on several of the dates. Do NOT count field boundaries, roads, tractor tracks, irrigation channels or straight crop rows.
>
> Question: Is there an ephemeral gully in this field? First explain what you see in one or two sentences, then answer yes or no.

Answer format: `{"reason": string, "answer": "yes" | "no"}`

**Evidence prompt (A2 "evidence", A3, A4):**

> The image is a collage of {n} overhead images of the SAME agricultural field, taken on different dates. Each tile has its date or image number in the top-left corner. Ignore any grey "no image" tile or text tile.
>
> An ephemeral gully is a small erosion channel cut by rainwater runoff, which farmers fill in during tillage. Many fields have no gully, so only give a high score if you see clear evidence.
>
> First, list the linear features you can see (if any) and say what each most likely is, e.g. crop rows, field boundary, road, tractor track, canal, or erosion channel. Then give your score.
>
> Score from 1 to 5 how likely it is that this field contains an ephemeral gully: 1 = almost certainly no, 2 = probably no, 3 = unsure, 4 = probably yes, 5 = almost certainly yes.

Answer format: `{"features": string, "score": 1–5}`. The A2 "short" prompt (tried; results lost, not reported) is the same without the "First, list…" paragraph and returns only the score.

## Appendix B. Pipeline B questions and decision prompt

The 19 questions from Tousi et al. (2025):

1. Do you see a low point in the terrain?
2. Do narrow, winding paths or channels appear?
3. Are there winding paths that become intermittent recurrent?
4. Are there any linear depressions or ruts which appear more pronounced along natural drainage lines or slopes?
5. Are there narrow and shallow channels which appear intermittently deeper or more indented into the soil?
6. Are there areas where soil appears disturbed or vegetation is removed?
7. Does a specific path lack vegetation, suggesting an evolving or emerging channel?
8. Are there varying types and levels of coarseness in the texture of the soil?
9. Are there clear starting and ending points of potential channels?
10. Are there small rills or grooves indicating water flow?
11. Is there a varying exposure of lighter or darker colored soil?
12. Are there sediment accumulations forming?
13. Are there signs of water activity, like soil clumps or crusting, that appear or intensify in specific areas across the images? (ending completed by us; cut off in the paper)
14. Are there any branching patterns that resemble temporary streams?
15. Are there indications of nearby human activity, such as tillage or machinery tracks?
16. Do you see any sign of water flow patterns across the field in multiple images?
17. Do you see any edges in the images indicating removal of soil along the water pathway?
18. Do you see any cuts in the soil associated with water flow across the field?
19. Do you see any indication of human activity such as tillage that is not naturally happened in the field?

**Decision prompt (text-only second call):**

> A vision model examined a collage of overhead images of one agricultural field, taken in different years, and answered yes/no questions about it. Its description: {description}. Its answers: {qa}. An ephemeral gully is a small erosion channel cut by rainwater runoff, which farmers fill in during tillage and which tends to reappear in the same place. Many fields have no gully. Based only on these answers, reason briefly about whether this field contains an ephemeral gully, then score how likely it is: 1 = almost certainly no, 2 = probably no, 3 = unsure, 4 = probably yes, 5 = almost certainly yes.

## Links

- Code and data: [github.com/japmanyakaur/ephemeral-gully-domain-adaptation](https://github.com/japmanyakaur/ephemeral-gully-domain-adaptation)
- Dataset release v2.0: [Releases → v2.0](https://github.com/japmanyakaur/ephemeral-gully-domain-adaptation/releases/tag/v2.0)
- Tousi et al. (2025), A Zero-Shot Learning Approach for Ephemeral Gully Detection from Remote Sensing using Vision Language Models: [arXiv 2503.01169](https://arxiv.org/abs/2503.01169)
- Follow-up, Weakly Supervised Ephemeral Gully Detection: [arXiv 2511.13891](https://arxiv.org/abs/2511.13891)
- Aybar et al. (2026), SEN2SR: [Remote Sensing of Environment](https://doi.org/10.1016/j.rse.2025.115222)
- AWS Terrain Tiles: [registry.opendata.aws/terrain-tiles](https://registry.opendata.aws/terrain-tiles/)
- Tribune India, Rs 20 crore allocated to check soil erosion in Shivaliks, Aravallis: [tribuneindia.com](https://www.tribuneindia.com/news/haryana/rs-20-crore-allocated-to-check-soil-erosion-in-shivaliks-aravallis-510947)
- This document (Google Doc): [Ephemeral Gully Detection: Project Status, Log and Results](https://docs.google.com/document/d/1ShEZ169PsSUGEAet1WTp3hefvXNTAd52gyHE8ZqOmDM/edit)
