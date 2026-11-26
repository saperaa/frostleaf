# FrostLeaf
**When fine-tuning silently stops: weight-update underflow in pure-bf16 transfer learning for plant disease classification**
AIE231 Neural Networks course project, Alamein International University.

## Problem
A farmer photographs a sick leaf with a phone and needs the disease name to pick a treatment. The model classifies the leaf into one of 38 crop/disease classes (PlantVillage). A wrong prediction means the wrong treatment, or none, and a lost crop. Rare diseases matter as much as common ones, so the **primary metric is macro-F1**.

| Item | Choice |
|---|---|
| Input | RGB leaf photo, resized to 160 × 160 × 3, pixels scaled to [0, 1] (`src.data.preprocess`) |
| Output | 38-way softmax over crop/disease classes |
| Loss | Sparse categorical cross-entropy (integer labels; equivalent to one-hot + categorical cross-entropy), with balanced class weights |
| Primary metric | Macro-F1 on validation for model selection; on test once, at the end |
| Secondary | Accuracy, per-class recall, one-vs-rest ROC-AUC, macro-F1 on PlantDoc field photos |

## Research question
Storing weights in bfloat16 saves memory, but bf16 keeps only about 3 significant digits. Any update smaller than half a bf16 step (about 0.4% of the weight) rounds away. Fine-tuning uses very small learning rates, so:

- **H1:** In pure-bf16 weight storage, fine-tuning (Model C, lr ≈ 1e-5) silently freezes most updates and ends up close to the frozen-base model (Model B). Training from scratch (Model A, lr ≈ 1e-3) is far less affected.
- **H2:** Stochastic rounding restores fp32-level fine-tuning at the same weight storage cost.
- **H3:** Optimizers resist freezing differently: plain SGD the least, SGD+momentum more, Adam the most.
- **H4:** Weight decay applied as its own rounded step is swallowed completely, but decay fused with the optimizer step partly survives through dithering.
- **Secondary:** the effect on field images (PlantDoc), where the gap between Model B and Model C is larger.

Setting studied: **bf16 weight storage + fp32 optimizer state** (controlled isolation of weight-update underflow), simulated in a custom `tf.GradientTape` train step so it runs on any GPU. See `src/precision.py`.

## Team
| Member | GitHub | Role (spec roles; 5 members, so 1 and 3 merged) | Owns |
|---|---|---|---|
| Mohamed Alaa | [@saperaa](https://github.com/saperaa) | Tech Lead + Baseline & Analysis | repo skeleton, `src/config.py`, `src/train.py`, `src/baselines.py`, `notebooks/01_eda.ipynb` |
| Mohamed Mostafa | [@mufargella](https://github.com/mufargella) | Data Engineer | `src/data.py`, `data/` (download, splits, PlantDoc map) |
| [@Homd11](https://github.com/Homd11) | [@Homd11](https://github.com/Homd11) | Model Architect | `src/precision.py`, `src/models.py` (Models A–C), `report/model_a_design.md` |
| Muhammed Adel | [@MuhammedAdel68](https://github.com/MuhammedAdel68) | Experimentation Engineer | `src/experiment_log.py`, `experiments/`, `notebooks/00_colab_timing.ipynb` |
| Marawan Khaled | [@marawankhaled77](https://github.com/marawankhaled77) | Evaluation & Deployment | `src/evaluate.py`, `src/predict.py`, `data/plantdoc_exclude.csv`, `app/` |

## Status
- **Phase 1 (topic settled): done.** Problem, metric, hypotheses, roles, repository.
- **Phase 2 (baselines ready): done.** Real-data EDA (`notebooks/01_eda.ipynb`), leaf-grouped leak-free splits (`data/splits/`), both baselines measured, Colab timing (160 px), experiment logger.
- **Phase 3 (models training): next.** Models A, B and C end to end with `src/train.py`.

## Setup and commands
```bash
pip install -r requirements.txt               # Python 3.12 (Colab)
pytest -q                                     # unit tests
python data/download.py                       # PlantVillage + PlantDoc into data/raw/ (~3 min on Colab)
python -m src.data check                      # leakage check on the committed split
python -m src.baselines --author <name>       # both baselines on validation (~8 min on Colab CPU)
python -m experiments.sanity_bf16_freeze      # bf16 freeze sanity check (~1 min on CPU)
```
Training (`src/train.py`), evaluation and the demo are added in Phases 3–4.

## Datasets
- PlantVillage: https://huggingface.co/datasets/mohanty/PlantVillage (Hughes & Salathé 2015; Mohanty et al. 2016)
- PlantDoc: https://github.com/pratikkayal/PlantDoc-Dataset (Singh et al. 2019), external test set only

## Repository layout
```
configs/      one YAML per experiment (each overrides one factor of base.yaml)
data/         download script + data README (no raw data in git)
notebooks/    01_eda.ipynb, 02_results.ipynb
src/          config, data, models, precision, train, evaluate, predict
experiments/  log.csv (every run), curves/, sanity check
models/       final saved model + preprocessing objects
app/          Gradio demo
report/       report PDF + slides
tests/        pytest unit tests
```

## Baselines (validation split)
Every model is reported as an improvement over these numbers. Produced by `python -m src.baselines --author <name>` on 34,633 train / 8,659 validation images (plain images, no augmentation). The logistic regression overfits (train macro-F1 0.984 vs 0.617 validation) and stopped at its 300-iteration limit before converging; its validation one-vs-rest ROC-AUC is 0.967. Details: `experiments/baselines/`.

| Baseline | Input | Val macro-F1 | Val accuracy |
|---|---|---|---|
| Majority class | none | 0.005 | 10.5% |
| Logistic regression (balanced) | 32 x 32 x 3 pixels, standardised on train | **0.617** | 69.0% |

## Results
Filled in at the end of Phase 4.
