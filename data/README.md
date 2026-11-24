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
