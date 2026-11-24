"""Baselines and neural architectures.

Owners: Tech Lead (baselines) and Model Architect (Models A, B, C).

Every network is built as a `src.precision.PrecisionModel`, so the same
architecture can be trained in fp32, bf16_rtn or bf16_sr by changing the config.

Models (spec 3.4, image track):
* Model A: custom CNN trained from scratch (larger LR, e.g. 1e-3).
* Model B: ResNet-50 (ImageNet weights) frozen + new classifier head.
* Model C: Model B with the last block unfrozen, small LR (e.g. 1e-5).
  Start from Model B's trained weights (head included).
  Keep BatchNorm layers in inference mode (training=False) while fine-tuning.

For each model the report needs: architecture diagram, output shape per layer,
hand-computed parameter counts checked against model.summary(), and a one- or
two-sentence justification for every design choice.
"""

from __future__ import annotations


def majority_class_baseline(train_labels):
    """Trivial baseline: always predict the most frequent training class."""
    raise NotImplementedError("Tech Lead: implement in Week 10")


def logistic_regression_baseline(train_features, train_labels):
    """Simple learned baseline on downsampled pixels or frozen features."""
    raise NotImplementedError("Tech Lead: implement in Week 10")


def build_model_a_scratch(cfg: dict, num_classes: int):
    """Model A: custom CNN from scratch, returned as a PrecisionModel."""
    raise NotImplementedError("Model Architect: implement in Week 11")


def build_model_b_frozen(cfg: dict, num_classes: int):
    """Model B: frozen ResNet-50 base + new head, returned as a PrecisionModel."""
    raise NotImplementedError("Model Architect: implement in Week 11")


def build_model_c_finetune(cfg: dict, num_classes: int):
    """Model C: ResNet-50 with last block unfrozen, returned as a PrecisionModel."""
    raise NotImplementedError("Model Architect: implement in Week 11")
