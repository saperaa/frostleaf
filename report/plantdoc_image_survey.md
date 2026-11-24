# PlantDoc image survey (Phase 1)

Goal: decide whether we evaluate on full PlantDoc photos or on cropped single
leaves.

Looked at: 30 images, at least one from each of 10 different classes, from
https://github.com/pratikkayal/PlantDoc-Dataset (train/ and test/ folders).

Sample used: 15 classes (Apple Scab Leaf, Apple rust leaf, Bell_pepper leaf spot,
Blueberry leaf, Cherry leaf, Corn leaf blight, Corn rust leaf, Peach leaf,
Potato leaf early blight, Raspberry leaf, Squash Powdery mildew leaf,
Tomato Septoria leaf spot, Tomato leaf late blight, Tomato leaf mosaic virus,
grape leaf black rot); for each class, the first file (alphabetically) in
`test/` and the first in `train/`. Each photo was counted once, under the
category that best describes it.

| What the photo shows | Count (out of 30) |
|---|---|
| One leaf, centred, fills most of the frame | 13 |
| Several leaves | 9 |
| Whole plant or field | 6 |
| Diagram, drawing or text overlay (not a real photo) | 2 |

## Decision
Leaf crops, because: only 13 of 30 photos (43%) look like PlantVillage (one
leaf filling the frame); the other 17 show several leaves, a whole plant, or
are not real photos. Our models are trained on single, centred PlantVillage
leaves, so scoring full PlantDoc images would mix two effects: the field
conditions we want to measure (lighting, background, real lesions) and a
framing mismatch the model was never meant to handle. Crops isolate the
field-condition shift. Diagrams and text-overlay images are dropped, and we
report how many images were removed per class.

## Notes
- Classes that look hardest: Tomato leaf mosaic virus (both samples are
  whole plants with subtle mottling, no single clear leaf); Corn leaf blight
  vs Corn Gray leaf spot (similar elongated lesions); Potato leaf early blight
  (small lesions in dense foliage); the healthy classes shot as plants or
  branches (Peach leaf, Raspberry leaf).
- Anything odd (watermarks, wrong labels, duplicates):
  - Text overlay / not a photo: `test/Peach leaf/00pe.jpg` has a banner
    "Peach Leaf Benefits, Reviews And Dosage"; `test/Tomato Septoria leaf spot/0.jpg`
    is a screenshot of a slide with a title and citation text.
  - Watermarks / credits: `test/Corn rust leaf/0796.39maizerust.jpg`
    (© Gary Munkvold), `train/Tomato leaf late blight/118A_Tom_Late.jpg`
    (© T.A. Zitter), `test/Potato leaf early blight/3023.jpg` (EPPO credit line).
  - Possible wrong labels: `train/Apple Scab Leaf/11NovMW_W4-A_4b.jpg` looks
    like rose black spot (rose leaves and thorny stems), not apple;
    `test/Apple rust leaf/02.-Rust-2017-207u24s.jpg` shows no visible rust.
  - Unusual images: a circular microscope-style view
    (`train/Corn rust leaf/36687400253_a227c2ca28_b.jpg`), a two-photo
    composite (`test/grape leaf black rot/03gb.jpg`), and a product photo of
    dried leaves (`train/Blueberry leaf/10-Pcs-...Leaf-shaped.jpg`).
  - Repo issues: `Tomato two spotted spider mites leaf` has only 2 train
    images and no test folder. One file name in `train/Apple Scab Leaf/`
    contains `?`, so the folder cannot be checked out on Windows (Colab/Linux
    is fine).
  - No duplicates seen in this sample of 30.
