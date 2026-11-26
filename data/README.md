# Data

Raw data is **not** stored in git. Run `python data/download.py` to fetch it into `data/raw/`.

| Dataset | Role | Source | Licence | Size (stated by source) | Verified count (fill in EDA) |
|---|---|---|---|---|---|
| PlantVillage (color, segmented) | Train / val / test | Hugging Face `mohanty/PlantVillage` | CC BY-SA 3.0 (Hugging Face card) | Card: 43,596 train + 10,709 test = 54,305 (color config); 38 classes | |
| PlantDoc | External field-condition test set (never trained on) | github.com/pratikkayal/PlantDoc-Dataset | CC BY 4.0 | 2,342 train + 236 test images, 28 classes, one folder per class | |

## Split policy
- PlantVillage: the official leaf-grouped 80/20 split; validation (about 15% of all images) is carved from the 80% part, also grouped by leaf, giving roughly 70/15/15.
- Split on the `leaf_id` column of the Hugging Face copy: several photos show the same physical leaf. Do NOT use the TFDS copy for training; it has no leaf IDs.
- Load the named configs (`color`, `segmented`). The Hugging Face viewer shows a `default` subset of about 163k rows, which mixes all three variants.
- PlantDoc: evaluation only (we never train on it, so its train and test folders can both be used). `data/plantdoc_class_map.csv` maps all 28 PlantDoc classes onto PlantVillage names.
- PlantDoc photos often show several leaves or a whole plant. Decide in Week 10 between full images and leaf crops, and state it in the report.

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
