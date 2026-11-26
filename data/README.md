# Data

Raw data is **not** stored in git. Run `python data/download.py` to fetch it into `data/raw/` (PlantVillage colour + segmented, PlantDoc images and PlantDoc leaf boxes; `--variants color` for colour only). The fixed split is committed in `data/splits/`.

| Dataset | Role | Source | Licence | Size (stated by source) | Verified count |
|---|---|---|---|---|---|
| PlantVillage (color, segmented) | Train / val / test | Hugging Face `mohanty/PlantVillage` (revision `a12d9d3`) | CC BY-SA 3.0 (Hugging Face card) | Card: 43,596 train + 10,709 test = 54,305 (color config); 38 classes | color: 43,596 train + 10,709 test = 54,305, 38 classes; columns `image, image_path, label, crop, disease, leaf_id` (verified 2026-11-26) |
| PlantDoc | External field-condition test set (never trained on) | github.com/pratikkayal/PlantDoc-Dataset (commit `5467f60`) | CC BY 4.0 | 2,342 train + 236 test images, 28 classes, one folder per class | 2,342 train + 236 test = 2,578 images, 28 class folders (verified 2026-11-26) |
| PlantDoc leaf boxes | Leaf crops for the PlantDoc test | github.com/pratikkayal/PlantDoc-Object-Detection-Dataset (commit `4730a23`), `train_labels.csv` + `test_labels.csv` | CC BY 4.0 | 2,598 images | 8,921 boxes on 2,565 images |

All 28 `plantvillage_class` names in `plantdoc_class_map.csv` match a label name of the dataset exactly (28 distinct targets out of 38 classes).

### Download notes
- The Hugging Face repo is a loading script (`plant_village.py`), which `datasets` >= 4 (we pin 5.2.0) no longer runs: `load_dataset("mohanty/PlantVillage", "color")` fails with "BuilderConfig 'color' not found". `download.py` therefore fetches the same files (`data.zip`, `splits/<variant>_{train,test}.txt`, `leaf_grouping/leaf-map.json`) at a pinned revision, repeats the script's logic (including how `leaf_id` is derived) and saves the same features with `save_to_disk`, image bytes embedded.
- PlantDoc is downloaded as a zip of a pinned commit, not `git clone`: 86 file names contain `?` (not allowed on Windows) and 6 pairs differ only by letter case (`CAR1.jpg` / `car1.jpg`). On Windows `?` becomes `_` (the loader maps the exclusion list and box files the same way) and the second file of each case pair is skipped. On Colab/Linux nothing changes.

## Split policy
- Test = the official Hugging Face test split, **never modified** and used once, at the end.
- Validation (about 15% of all images) is carved out of the official train split with scikit-learn `StratifiedGroupKFold` (groups = `leaf_id`, stratified by class, `shuffle=True, random_state=42`). The fold count is chosen so one fold is closest to 15% of all images (5 folds), and the first fold is validation.
- Before that, the train pool is cleaned so nothing can leak into test: train images whose `leaf_id` also appears in test are dropped, train images byte-identical (same MD5) to a test image are dropped, and exact duplicates inside train are reduced to one copy.
- `data/splits/{train,val,test}.csv` holds the result (`image_id` = Hugging Face `image_path`, `leaf_id`, `label` = class name, `file_hash` = MD5 of the image file). Every member loads the same split through `src.data.load_plantvillage()`. Regenerate it with `python -m src.data make-splits` (byte-identical on re-run) and check it with `python -m src.data check` (no download needed).
- The segmented variant reuses the colour split's validation **leaves**, so all variants validate on the same physical leaves.
- Class weights (`compute_class_weights`) come from the train split only; augmentation (`flip`, `rotate`, optionally `brightness`, `contrast`, `zoom`) runs on the train split only.
- PlantDoc: evaluation only (we never train on it, so its train and test folders are both used). `data/plantdoc_class_map.csv` maps all 28 PlantDoc classes onto PlantVillage names.

## Leakage findings (PlantVillage colour)
- Several photos per leaf: the 43,596 official-train images come from 16,124 `leaf_id`s. 9,595 leaves have 1 photo, 344 have 2, 109 have 3, 4,416 have 4, and 1,660 have 5 to 33. Splitting by image would put photos of one leaf in both train and validation.
- **The official split is not fully leaf-disjoint: 227 `leaf_id`s are in both official train and official test** (290 train images). These train images are dropped.
- No official-train image is byte-identical to a test image. 14 exact duplicate copies inside train were removed (none carried two labels). Test holds 2 duplicate copies; it is left untouched.
- 10,518 train and 2,676 test rows have no entry in the dataset's leaf map and get a `fallback_<file id>` leaf ID, so each is treated as its own leaf. This is the best the source data allows; the hash check still guarantees no identical file crosses splits.
- `check_no_leakage` passes on the committed split: no shared `leaf_id` and no shared file hash between any two splits. `tests/test_data.py` shows the check catches a planted shared leaf and a planted identical file.

## Final split sizes
| Split | Images | Share of 54,305 | Leaves |
|---|---:|---:|---:|
| Train | 34,633 | 63.8% | 12,717 |
| Validation | 8,659 | 15.9% | 3,180 |
| Test (official) | 10,709 | 19.7% | 4,118 |
| Dropped from train (leaf in test 290, duplicate copy 14) | 304 | 0.6% | |

The official test is 20% of the data, so keeping it untouched gives about 64/16/20 rather than 70/15/15. Class shares in validation match train within 0.05 percentage points. Imbalance on train: largest class `Orange___Haunglongbing_(Citrus_greening)` (3,621) / smallest `Potato___healthy` (96) = **37.7x**; balanced class weights range from 0.25 to 9.49.

### Per class
| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Apple___Apple_scab | 404 | 99 | 127 | 630 |
| Apple___Black_rot | 381 | 94 | 146 | 621 |
| Apple___Cedar_apple_rust | 179 | 44 | 52 | 275 |
| Apple___healthy | 1,082 | 272 | 286 | 1,640 |
| Blueberry___healthy | 944 | 234 | 324 | 1,502 |
| Cherry_(including_sour)___Powdery_mildew | 667 | 166 | 219 | 1,052 |
| Cherry_(including_sour)___healthy | 584 | 147 | 123 | 854 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 307 | 77 | 107 | 491 |
| Corn_(maize)___Common_rust_ | 780 | 195 | 217 | 1,192 |
| Corn_(maize)___Northern_Leaf_Blight | 582 | 146 | 220 | 948 |
| Corn_(maize)___healthy | 554 | 138 | 250 | 942 |
| Grape___Black_rot | 812 | 200 | 168 | 1,180 |
| Grape___Esca_(Black_Measles) | 879 | 220 | 284 | 1,383 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 696 | 176 | 204 | 1,076 |
| Grape___healthy | 258 | 65 | 100 | 423 |
| Orange___Haunglongbing_(Citrus_greening) | 3,621 | 906 | 980 | 5,507 |
| Peach___Bacterial_spot | 1,403 | 352 | 542 | 2,297 |
| Peach___healthy | 231 | 58 | 71 | 360 |
| Pepper,_bell___Bacterial_spot | 666 | 165 | 166 | 997 |
| Pepper,_bell___healthy | 955 | 240 | 283 | 1,478 |
| Potato___Early_blight | 668 | 168 | 164 | 1,000 |
| Potato___Late_blight | 628 | 160 | 212 | 1,000 |
| Potato___healthy | 96 | 24 | 32 | 152 |
| Raspberry___healthy | 171 | 38 | 162 | 371 |
| Soybean___healthy | 3,293 | 824 | 973 | 5,090 |
| Squash___Powdery_mildew | 1,161 | 290 | 378 | 1,829 |
| Strawberry___Leaf_scorch | 728 | 181 | 200 | 1,109 |
| Strawberry___healthy | 302 | 74 | 80 | 456 |
| Tomato___Bacterial_spot | 1,351 | 340 | 436 | 2,127 |
| Tomato___Early_blight | 625 | 160 | 215 | 1,000 |
| Tomato___Late_blight | 1,221 | 305 | 375 | 1,901 |
| Tomato___Leaf_Mold | 624 | 156 | 172 | 952 |
| Tomato___Septoria_leaf_spot | 1,147 | 287 | 337 | 1,771 |
| Tomato___Spider_mites Two-spotted_spider_mite | 1,084 | 272 | 320 | 1,676 |
| Tomato___Target_Spot | 893 | 223 | 288 | 1,404 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 3,406 | 851 | 1,100 | 5,357 |
| Tomato___Tomato_mosaic_virus | 246 | 61 | 66 | 373 |
| Tomato___healthy | 1,004 | 251 | 330 | 1,585 |
| **Total** | **34,633** | **8,659** | **10,709** | **54,001** |

## PlantDoc: leaf crops (decision)
**Decision: crops.** The PlantDoc object-detection release has a box for every annotated leaf (`train_labels.csv`, `test_labels.csv`: file name, image size, class, box). It is CC BY 4.0, like the classification set, and its file names match 2,562 of the 2,578 classification images. So the survey's choice works now: `load_plantdoc_shared_classes(class_names)` defaults to `mode="crops"` and scores one example per box. `mode="full"` scores whole photos for comparison.

How crops are made: boxes are matched by file name and class, stored as fractions of the image they were drawn on, cut out of the classification image, then passed through the same `preprocess` as PlantVillage. Skipped: boxes smaller than 5% of the image on a side, 3 images with no box, and 5 images whose box file was drawn on a rotated (EXIF) copy, where the box would land in the wrong place.

| PlantDoc class | PlantVillage class | Full images | Leaf crops |
|---|---|---:|---:|
| Apple Scab Leaf | Apple___Apple_scab | 80 | 142 |
| Apple leaf | Apple___healthy | 44 | 160 |
| Apple rust leaf | Apple___Cedar_apple_rust | 80 | 167 |
| Bell_pepper leaf | Pepper,_bell___healthy | 60 | 320 |
| Bell_pepper leaf spot | Pepper,_bell___Bacterial_spot | 65 | 249 |
| Blueberry leaf | Blueberry___healthy | 95 | 759 |
| Cherry leaf | Cherry_(including_sour)___healthy | 56 | 236 |
| Corn Gray leaf spot | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 58 | 66 |
| Corn leaf blight | Corn_(maize)___Northern_Leaf_Blight | 165 | 295 |
| Corn rust leaf | Corn_(maize)___Common_rust_ | 107 | 114 |
| Peach leaf | Peach___healthy | 49 | 300 |
| Potato leaf early blight | Potato___Early_blight | 84 | 256 |
| Potato leaf late blight | Potato___Late_blight | 84 | 187 |
| Raspberry leaf | Raspberry___healthy | 99 | 450 |
| Soyabean leaf | Soybean___healthy | 51 | 206 |
| Squash Powdery mildew leaf | Squash___Powdery_mildew | 124 | 247 |
| Strawberry leaf | Strawberry___healthy | 86 | 429 |
| Tomato Early blight leaf | Tomato___Early_blight | 78 | 188 |
| Tomato Septoria leaf spot | Tomato___Septoria_leaf_spot | 135 | 400 |
| Tomato leaf | Tomato___healthy | 58 | 350 |
| Tomato leaf bacterial spot | Tomato___Bacterial_spot | 96 | 214 |
| Tomato leaf late blight | Tomato___Late_blight | 103 | 203 |
| Tomato leaf mosaic virus | Tomato___Tomato_mosaic_virus | 50 | 242 |
| Tomato leaf yellow virus | Tomato___Tomato_Yellow_Leaf_Curl_Virus | 66 | 593 |
| Tomato mold leaf | Tomato___Leaf_Mold | 85 | 259 |
| grape leaf | Grape___healthy | 66 | 210 |
| grape leaf black rot | Grape___Black_rot | 55 | 119 |
| **Total (27 classes)** | | **2,179** | **7,349 (from 2,171 images)** |

Counts are after the exclusion list and after dropping 2 files that are not JPEG/PNG/BMP/GIF. They were measured on Windows, where the 6 case-clash files are missing, so a Colab run can differ by a few images. `Tomato two spotted spider mites leaf` has no images left.

Limitations: crops from one photo are not independent (Blueberry gets 759 crops from 95 photos), so we also report full-image results; the boxes come from the PlantDoc authors and were not re-checked by us.

## PlantDoc exclusion list (`plantdoc_exclude.csv`)
All 2,578 images of the 28 mapped PlantDoc classes (train + test folders) were reviewed by eye. 392 are listed for removal, one row per file (`path` relative to the PlantDoc repo root, `plantdoc_class`, one-word `reason`). The loader drops every listed file.

| Reason | Rule | Count |
|---|---|---|
| watermark | large stock-photo watermark across the leaf (small corner credits are kept) | 211 |
| composite | multi-panel figure (two or more photos side by side) | 54 |
| product | studio/stock product shot (fruit on white, dried or craft leaves) | 45 |
| text | title, caption or banner text on the image | 37 |
| slide | screenshot of a slide, document or web page | 23 |
| conflict | the same file (identical bytes) appears under two different classes; both copies dropped | 16 |
| drawing | illustration or diagram, not a photo | 5 |
| duplicate | identical file in both train and test of the same class; the train copy is dropped | 1 |

Images with several leaves, whole plants, hands or plain backgrounds are kept: that is the field shift we want to measure. After exclusion, `Tomato two spotted spider mites leaf` has 0 images (it had only 2), and Apple leaf (44 left of 91) and Peach leaf (50 of 112) lose about half, mostly stock photos. `src/evaluate.py` reports classes with no images as absent and leaves them out of macro-F1 and AUC.
