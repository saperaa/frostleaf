| # | Finding | Modelling decision |
|---|---|---|
| 1 | Classes range from 152 to 5,507 images (36.2x imbalance) | Primary metric macro-F1 (not accuracy); balanced class weights; report per-class recall |
| 2 | 78.2% of images share their leaf with another photo (up to 33 per leaf); 227 leaves cross the official splits | Split by leaf_id (StratifiedGroupKFold) and assert no leaf_id is in two splits |
| 3 | 21 groups of byte-identical images (5 across splits, 0 with conflicting labels) | Keep one copy per group, in the split of its leaf; drop conflicting-label groups; check_no_leakage also compares hashes |
| 4 | 1 images are not 256x256 RGB | Resize to the config image_size (224, or 160 if Colab timing requires) and scale pixels to [0, 1] in one shared preprocess() |
| 5 | Diseases of the same crop look alike (e.g. tomato early vs late blight) | Error analysis groups confusions by crop; at least two augmentations (flip, rotation) that keep lesion shape |
| 6 | Background brightness differs by up to 0.51 (0-1 scale) between classes | Possible shortcut: compare colour vs segmented PlantVillage in an ablation, and test on PlantDoc |
| 7 | PlantDoc gives 2,186 usable field photos over 27 classes; 1 classes have under 20 | Report PlantDoc macro-F1 over present classes only, with per-class support; treat small classes as indicative |
