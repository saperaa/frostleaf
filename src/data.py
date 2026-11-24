"""Data loading, cleaning, splitting and preprocessing.

Owner: Data Engineer.

Leakage rules this module must enforce (spec 3.2):
* Split by LEAF, not by image: several PlantVillage photos show the same leaf.
  Use the Hugging Face copy (mohanty/PlantVillage), whose rows carry a
  `leaf_id` column; its 80/20 split already respects leaf grouping. Carve
  validation out of its train part, grouped by `leaf_id`, to reach ~70/15/15.
  Never use the TFDS copy for splitting: it has no leaf IDs.
* The test split is used ONCE, at the end.
* Anything fitted (normalisation stats, class weights) uses the train split only.
* Augmentation applies to the train split only.
* PlantDoc is an external field-condition test set: never trained on, and only
  the classes that also exist in PlantVillage are scored.
"""

from __future__ import annotations


def load_plantvillage(variant: str = "color"):
    """Load PlantVillage (color | segmented | grayscale) with leaf-grouped splits.

    Returns:
        Train, validation and test datasets plus the class-name list.
    """
    raise NotImplementedError("Data Engineer: implement in Week 10")


def load_plantdoc_shared_classes(class_names: list[str]):
    """Load PlantDoc images whose class maps onto a PlantVillage class.

    Returns:
        A test dataset and the PlantDoc -> PlantVillage class mapping used.
    """
    raise NotImplementedError("Data Engineer: implement in Week 10")


def compute_class_weights(train_labels) -> dict[int, float]:
    """Inverse-frequency class weights computed on the TRAIN split only."""
    raise NotImplementedError("Data Engineer: implement in Week 10")


def build_augmentation(names: list[str]):
    """Return a Keras preprocessing pipeline (e.g. flip, rotate) for training only."""
    raise NotImplementedError("Data Engineer: implement in Week 10")


def check_no_leakage(train_ids, val_ids, test_ids) -> None:
    """Assert that no leaf ID appears in more than one split."""
    raise NotImplementedError("Data Engineer: implement in Week 10")
