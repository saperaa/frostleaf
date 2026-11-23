# FrostLeaf
**When fine-tuning silently stops: weight-update underflow in pure-bf16 transfer learning for plant disease classification**
AIE231 Neural Networks course project, Alamein International University.

## Problem
A farmer photographs a sick leaf with a phone and needs the disease name to pick a treatment. The model classifies the leaf into one of 38 crop/disease classes (PlantVillage). A wrong prediction means the wrong treatment, or none, and a lost crop. Rare diseases matter as much as common ones, so the **primary metric is macro-F1**.

## Research question
Storing weights in bfloat16 saves memory, but bf16 keeps only about 3 significant digits. Any update smaller than half a bf16 step (about 0.4% of the weight) rounds away. Fine-tuning uses very small learning rates, so:

- **H1:** In pure-bf16 weight storage, fine-tuning (Model C, lr ≈ 1e-5) silently freezes most updates and ends up close to the frozen-base model (Model B). Training from scratch (Model A, lr ≈ 1e-3) is far less affected.
- **H2:** Stochastic rounding restores fp32-level fine-tuning at the same weight storage cost.
- **H3:** Optimizers resist freezing differently: plain SGD the least, SGD+momentum more, Adam the most.
- **H4:** Weight decay applied as its own rounded step is swallowed completely, but decay fused with the optimizer step partly survives through dithering.
- **Secondary:** the effect on field images (PlantDoc), where the gap between Model B and Model C is larger.

Setting studied: **bf16 weight storage + fp32 optimizer state** (controlled isolation of weight-update underflow), simulated in a custom `tf.GradientTape` train step so it runs on any GPU. See `src/precision.py`.

## Status: Phase 1 (topic settled)
| Done | Owner | Work |
|---|---|---|
| [x] | Tech Lead | Problem, metric, hypotheses; repo structure; config system; training entry point |
| [ ] | Data Engineer | Dataset download script, data README, PlantDoc class map, `src/data.py` outline |
| [ ] | Model Architect | `src/precision.py` (bf16 train step) + unit tests, `src/models.py` outline |
| [ ] | Experimentation Engineer | Experiment log, precision ablation configs, sanity experiment + results |
| [ ] | Evaluation & Deployment | `src/evaluate.py`, `src/predict.py` outlines, PlantDoc image survey |

## Setup
```bash
pip install -r requirements.txt
pytest -q                                   # unit tests (after the Model Architect's PR)
python -m experiments.sanity_bf16_freeze    # Phase 1 sanity check (after the Experimentation PR)
python data/download.py                     # datasets into data/raw/ (after the Data Engineer's PR)
```

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

## Results
Filled in at the end of Phase 4.
