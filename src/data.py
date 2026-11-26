"""Data loading, cleaning, splitting and preprocessing.

Owner: Data Engineer.

Leakage rules this module enforces (spec 3.2):
* Split by LEAF, not by image: several PlantVillage photos show the same leaf.
  We use the Hugging Face copy (mohanty/PlantVillage), whose rows carry a
  `leaf_id` column; its 80/20 split already respects leaf grouping. Validation
  is carved out of its train part, grouped by `leaf_id` and stratified by
  class, to reach ~70/15/15.
* Identical files (same MD5) never sit in two splits: train images that share
  a leaf or a file hash with the official test split are dropped from train,
  and exact duplicates inside the train pool are reduced to one copy.
* The test split is never modified and is used ONCE, at the end.
* Anything fitted (class weights) uses the train split only.
* Augmentation applies to the train split only.
* PlantDoc is an external field-condition test set: never trained on, and only
  the classes that also exist in PlantVillage are scored.

The fixed split lives in data/splits/{train,val,test}.csv (columns image_id,
leaf_id, label, file_hash). Regenerate it with:

    python -m src.data make-splits
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import warnings
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
CACHE_DIR = os.path.join(DATA_DIR, "cache")  # decoded-image caches (git-ignored)
PLANTDOC_DIR = os.path.join(RAW_DIR, "plantdoc")
PLANTDOC_BOXES_DIR = os.path.join(RAW_DIR, "plantdoc_boxes")
PLANTDOC_CLASS_MAP = os.path.join(DATA_DIR, "plantdoc_class_map.csv")
PLANTDOC_EXCLUDE = os.path.join(DATA_DIR, "plantdoc_exclude.csv")

SPLIT_COLUMNS = ["image_id", "leaf_id", "label", "file_hash"]
SPLIT_NAMES = ("train", "val", "test")
SPLITS_VARIANT = "color"  # the variant the committed split CSVs were built from
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
_UNSAFE_CHARS = re.compile(r'[<>:"|?*]')

# name in configs -> builder of one Keras augmentation layer
_AUGMENTATIONS = {
    "flip": lambda layers: layers.RandomFlip("horizontal_and_vertical"),
    "rotate": lambda layers: layers.RandomRotation(0.1, fill_mode="reflect"),
    "brightness": lambda layers: layers.RandomBrightness(0.2, value_range=(0.0, 1.0)),
    "contrast": lambda layers: layers.RandomContrast(0.2),
    "zoom": lambda layers: layers.RandomZoom(0.1, fill_mode="reflect"),
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _default_config() -> dict[str, Any]:
    """Return configs/base.yaml, the source of every default in this module."""
    return load_config("base.yaml")


def _image_size(image_size: int | None) -> int:
    """Return `image_size`, or `data.image_size` from base.yaml when it is None."""
    if image_size is not None:
        return int(image_size)
    return int(_default_config()["data"]["image_size"])


# ---------------------------------------------------------------------------
# Leakage check
# ---------------------------------------------------------------------------
def _as_frame(ids: Any) -> pd.DataFrame:
    """Turn a split description into a DataFrame with a `leaf_id` column.

    Accepts a DataFrame (must have `leaf_id`; `file_hash` is used if present),
    a dict of columns, or a plain iterable of leaf IDs.
    """
    if isinstance(ids, pd.DataFrame):
        frame = ids
    elif isinstance(ids, dict):
        frame = pd.DataFrame(ids)
    else:
        frame = pd.DataFrame({"leaf_id": list(ids)})
    if "leaf_id" not in frame.columns:
        raise ValueError("split description needs a 'leaf_id' column")
    return frame


def check_no_leakage(train_ids, val_ids, test_ids) -> None:
    """Assert that no leaf and no identical image file is in two splits.

    Args:
        train_ids, val_ids, test_ids: one description per split. Either a
            DataFrame / dict with a `leaf_id` column (and optionally a
            `file_hash` column, the MD5 of the image bytes), or a plain
            iterable of leaf IDs. The CSVs in data/splits/ can be passed as
            loaded by `load_split_tables()`.

    Raises:
        AssertionError: naming every pair of splits that shares a leaf ID or
            a file hash, with the count and a few examples.
    """
    frames = {name: _as_frame(ids) for name, ids in zip(SPLIT_NAMES, (train_ids, val_ids, test_ids))}
    problems = []
    columns = ["leaf_id"]
    if all("file_hash" in f.columns for f in frames.values()):
        columns.append("file_hash")  # hash check only when every split has hashes
    for column in columns:
        values = {name: set(f[column].dropna()) for name, f in frames.items()}
        for i, a in enumerate(SPLIT_NAMES):
            for b in SPLIT_NAMES[i + 1:]:
                shared = values[a] & values[b]
                if shared:
                    examples = ", ".join(sorted(map(str, shared))[:3])
                    problems.append(f"{len(shared)} {column}(s) in both {a} and {b} (e.g. {examples})")
    if problems:
        raise AssertionError("data leakage: " + "; ".join(problems))


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def file_md5(data: bytes) -> str:
    """Return the MD5 hex digest of raw file bytes (used to find identical images)."""
    return hashlib.md5(data).hexdigest()


def make_splits(
    official_train: pd.DataFrame,
    official_test: pd.DataFrame,
    val_fraction: float = 0.15,
    seed: int = 42,
    val_leaves: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Carve a leaf-grouped, class-stratified validation split out of official train.

    Cleaning, in order (the test split is never changed):
    1. drop train images whose `leaf_id` also appears in test;
    2. drop train images whose `file_hash` also appears in test;
    3. among identical files inside train, keep one copy (or none, if the
       copies carry different labels).
    Then validation is chosen with scikit-learn StratifiedGroupKFold
    (groups = leaf_id, y = label, shuffled with `seed`): the number of folds is
    picked so one fold is closest to `val_fraction` of ALL images, and the
    first fold becomes validation.

    Args:
        official_train, official_test: DataFrames with columns
            image_id, leaf_id, label, file_hash.
        val_fraction: target validation size as a fraction of all images.
        seed: random seed for the fold shuffle.
        val_leaves: if given, skip the fold search and put exactly the train
            images of these leaves in validation (used to reuse the committed
            colour split for another PlantVillage variant).

    Returns:
        (train, val, test, dropped) where `dropped` counts the rows removed by
        each cleaning step.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    test = official_test.reset_index(drop=True)
    pool = official_train.reset_index(drop=True)
    dropped = {}

    mask = pool["leaf_id"].isin(set(test["leaf_id"]))
    dropped["train_leaf_in_test"] = int(mask.sum())
    pool = pool[~mask]

    mask = pool["file_hash"].isin(set(test["file_hash"]))
    dropped["train_file_in_test"] = int(mask.sum())
    pool = pool[~mask]

    labels_per_hash = pool.groupby("file_hash")["label"].nunique()
    conflicting = set(labels_per_hash[labels_per_hash > 1].index)
    mask = pool["file_hash"].isin(conflicting)
    dropped["train_duplicate_label_conflict"] = int(mask.sum())
    pool = pool[~mask]
    mask = pool.duplicated("file_hash", keep="first")
    dropped["train_duplicate_copy"] = int(mask.sum())
    pool = pool[~mask].reset_index(drop=True)

    if val_leaves is not None:
        in_val = pool["leaf_id"].isin(set(val_leaves)).to_numpy()
    else:
        n_total = len(official_train) + len(official_test)
        n_splits = max(2, int(round(len(pool) / (val_fraction * n_total))))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        _, val_idx = next(sgkf.split(np.zeros(len(pool)), pool["label"], groups=pool["leaf_id"]))
        in_val = np.zeros(len(pool), dtype=bool)
        in_val[val_idx] = True

    train = pool[~in_val].reset_index(drop=True)
    val = pool[in_val].reset_index(drop=True)
    return train, val, test, dropped


def save_split_tables(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                      splits_dir: str = SPLITS_DIR) -> None:
    """Write the three split tables to `splits_dir`/{train,val,test}.csv."""
    os.makedirs(splits_dir, exist_ok=True)
    for name, frame in zip(SPLIT_NAMES, (train, val, test)):
        frame[SPLIT_COLUMNS].sort_values("image_id").to_csv(
            os.path.join(splits_dir, f"{name}.csv"), index=False)


def load_split_tables(splits_dir: str = SPLITS_DIR) -> dict[str, pd.DataFrame]:
    """Read data/splits/{train,val,test}.csv into a dict of DataFrames.

    These need no downloaded images, so every member (and the leakage check)
    can use them on a fresh clone.
    """
    tables = {}
    for name in SPLIT_NAMES:
        path = os.path.join(splits_dir, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing; run `python -m src.data make-splits`")
        tables[name] = pd.read_csv(path, dtype=str)
    return tables


# ---------------------------------------------------------------------------
# PlantVillage (Hugging Face save_to_disk copy)
# ---------------------------------------------------------------------------
def _plantvillage_path(variant: str) -> str:
    """Return data/raw/plantvillage_<variant>, failing clearly if it is missing."""
    path = os.path.join(RAW_DIR, f"plantvillage_{variant}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; run `python data/download.py --variants {variant}`")
    return path


def _load_hf(variant: str):
    """Load the saved DatasetDict with images left as undecoded bytes."""
    from datasets import Image, load_from_disk

    ds = load_from_disk(_plantvillage_path(variant))
    return ds.cast_column("image", Image(decode=False))


def _image_bytes(image: dict) -> bytes:
    """Return the raw file bytes of an undecoded Hugging Face image cell."""
    if image.get("bytes") is not None:
        return image["bytes"]
    with open(image["path"], "rb") as f:
        return f.read()


def build_metadata(variant: str = "color") -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Read every row's id, leaf, label and file hash from the saved dataset.

    Returns:
        ({"train": DataFrame, "test": DataFrame}, class_names), where each
        DataFrame has the columns of SPLIT_COLUMNS plus `row`, the row index
        inside the Hugging Face split, and `label` holds the class name.
    """
    ds = _load_hf(variant)
    class_names = list(ds["train"].features["label"].names)
    frames = {}
    for split in ("train", "test"):
        part = ds[split]
        hashes = [file_md5(_image_bytes(img)) for img in part["image"]]
        frames[split] = pd.DataFrame({
            "image_id": part["image_path"],
            "leaf_id": part["leaf_id"],
            "label": [class_names[i] for i in part["label"]],
            "file_hash": hashes,
            "row": np.arange(len(part)),
        })
    return frames, class_names


def make_plantvillage_splits(variant: str = SPLITS_VARIANT, splits_dir: str = SPLITS_DIR,
                             cfg: dict | None = None) -> dict[str, Any]:
    """Build the fixed split from the downloaded dataset, check it and save it.

    Reproducible: same data + same seed (config `seed`, 42) give the same CSVs.

    Returns:
        A summary dict: split sizes, rows dropped by each cleaning step, and the
        per-class count table.
    """
    cfg = cfg or _default_config()
    frames, class_names = build_metadata(variant)
    train, val, test, dropped = make_splits(
        frames["train"], frames["test"], cfg["data"]["val_fraction"], cfg["seed"])
    check_no_leakage(train, val, test)
    save_split_tables(train, val, test, splits_dir)
    counts = pd.DataFrame({name: f["label"].value_counts() for name, f in
                           zip(SPLIT_NAMES, (train, val, test))}).reindex(class_names).fillna(0).astype(int)
    return {"sizes": {n: len(f) for n, f in zip(SPLIT_NAMES, (train, val, test))},
            "dropped": dropped, "per_class": counts}


def _rows_for_splits(variant: str, splits_dir: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Return per-split metadata (with HF row indices) following the committed split.

    For the colour variant the CSV rows are matched by `image_id`. For other
    variants (different file names) the same validation LEAVES are reused, so
    every variant validates on the same physical leaves.
    """
    tables = load_split_tables(splits_dir)
    ds = _load_hf(variant)
    class_names = list(ds["train"].features["label"].names)
    meta = {}
    for split in ("train", "test"):
        part = ds[split]
        meta[split] = pd.DataFrame({
            "image_id": part["image_path"],
            "leaf_id": part["leaf_id"],
            "label": [class_names[i] for i in part["label"]],
            "row": np.arange(len(part)),
        })
    if variant == SPLITS_VARIANT:
        out = {}
        for name, source in (("train", "train"), ("val", "train"), ("test", "test")):
            wanted = tables[name][["image_id", "file_hash"]]
            rows = meta[source].merge(wanted, on="image_id", how="inner")
            if len(rows) != len(wanted):
                raise ValueError(f"{name}.csv lists {len(wanted) - len(rows)} images missing from "
                                 f"the downloaded {variant} data; re-run download.py")
            out[name] = rows
        return out, class_names

    warnings.warn(f"variant '{variant}': reusing the validation leaves of the {SPLITS_VARIANT} split")
    for split in ("train", "test"):  # hashes are needed for the cleaning steps
        part = ds[split]
        meta[split]["file_hash"] = [file_md5(_image_bytes(img)) for img in part["image"]]
    train, val, test, _ = make_splits(meta["train"], meta["test"],
                                      val_leaves=set(tables["val"]["leaf_id"]))
    return {"train": train, "val": val, "test": test}, class_names


def preprocess(image, image_size: int | None = None):
    """Resize an image to the config's `image_size` and scale pixels to [0, 1].

    This is the ONLY preprocessing shared by every model, the baselines and
    src/predict.py. Model-specific scaling (e.g. ResNet's) lives inside the
    model.

    Args:
        image: uint8 tensor/array in [0, 255], shape (H, W, 3) or (N, H, W, 3).
        image_size: output side length; defaults to `data.image_size` in
            configs/base.yaml.

    Returns:
        float32 tensor of shape (size, size, 3) or (N, size, size, 3) in [0, 1].
    """
    import tensorflow as tf

    size = _image_size(image_size)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (size, size), method="bilinear")
    return tf.clip_by_value(image / 255.0, 0.0, 1.0)


def decode_image(data):
    """Decode JPEG/PNG/BMP/GIF bytes into a uint8 (H, W, 3) tensor."""
    import tensorflow as tf

    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return image


def build_augmentation(names: list[str]):
    """Return a Keras Sequential of random augmentations, for the TRAIN split only.

    Keras random layers are identity unless called with `training=True`;
    `make_dataset` calls this pipeline only on the training dataset.

    Args:
        names: augmentation names from the config, any of
            flip (horizontal + vertical), rotate (up to +/-36 degrees),
            brightness, contrast, zoom. Inputs are expected in [0, 1].

    Returns:
        A `keras.Sequential` named "augmentation".

    Raises:
        ValueError: on an unknown name.
    """
    import keras

    unknown = [n for n in names if n not in _AUGMENTATIONS]
    if unknown:
        raise ValueError(f"unknown augmentation(s) {unknown}; choose from {sorted(_AUGMENTATIONS)}")
    return keras.Sequential([_AUGMENTATIONS[n](keras.layers) for n in names], name="augmentation")


def make_dataset(images, labels, *, training: bool, image_size: int | None = None,
                 batch_size: int = 32, augmentation: list[str] | None = None,
                 seed: int = 42, shuffle: bool | None = None, cache: str | None = None,
                 shuffle_buffer: int = 2048):
    """Build a batched tf.data pipeline: decode/preprocess, and augment only if training.

    Args:
        images: either a uint8 array (N, H, W, 3), or a callable
            `fetch(i) -> bytes` returning the encoded file of example i.
        labels: int class indices, length N.
        training: True only for the train split. Augmentation and per-epoch
            shuffling happen only when this is True.
        image_size, batch_size, seed: pipeline settings.
        augmentation: augmentation names (used only when `training`).
        shuffle: override shuffling (defaults to `training`).
        cache: optional file prefix (e.g. ``data/cache/color_train``). The
            decoded uint8 images are written there during the first full pass
            and read back on later epochs, so JPEG decoding and the Python
            fetch happen once. The cache holds images BEFORE `preprocess`, so
            resizing and scaling stay identical to the uncached path and to
            `predict.py`. Only used with a callable `images`.
        shuffle_buffer: with `cache`, the cache is written in one random order
            (seeded) and each epoch is reshuffled with a buffer of this size.

    Returns:
        A `tf.data.Dataset` of (float32 images in [0, 1], int32 labels).
    """
    import tensorflow as tf

    labels = np.asarray(labels, dtype=np.int32)
    shuffle = training if shuffle is None else shuffle
    size = _image_size(image_size)

    if callable(images) and cache:
        fetch = images
        order = np.random.default_rng(seed).permutation(len(labels)) if shuffle else np.arange(len(labels))

        def generate_fixed():
            """Yield (encoded bytes, label) in one fixed order, so the cache is reusable."""
            for i in order:
                yield fetch(int(i)), labels[i]

        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        ds = tf.data.Dataset.from_generator(generate_fixed, output_signature=(
            tf.TensorSpec((), tf.string), tf.TensorSpec((), tf.int32)))
        ds = ds.map(lambda b, y: (decode_image(b), y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache(cache)
        if shuffle:
            ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda x, y: (preprocess(x, size), y), num_parallel_calls=tf.data.AUTOTUNE)
    elif callable(images):
        fetch = images
        rng = np.random.default_rng(seed)

        def generate():
            """Yield (encoded bytes, label), in a new random order each epoch if shuffling."""
            order = rng.permutation(len(labels)) if shuffle else np.arange(len(labels))
            for i in order:
                yield fetch(int(i)), labels[i]

        ds = tf.data.Dataset.from_generator(generate, output_signature=(
            tf.TensorSpec((), tf.string), tf.TensorSpec((), tf.int32)))
        ds = ds.map(lambda b, y: (preprocess(decode_image(b), size), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((np.asarray(images), labels))
        if shuffle:
            ds = ds.shuffle(len(labels), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(lambda x, y: (preprocess(x, size), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    if training and augmentation:
        augment = build_augmentation(augmentation)
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def _cache_path(variant: str, split: str, seed: int) -> str:
    """Cache file prefix for one split. The train cache is written in a seeded
    order, so it is keyed by seed; val and test are stored in split order.
    Delete data/cache/ whenever data/splits/ changes."""
    suffix = f"_s{seed}" if split == "train" else ""
    return os.path.join(CACHE_DIR, f"{variant}_{split}{suffix}")


def load_plantvillage(variant: str = "color", cfg: dict | None = None,
                      splits_dir: str = SPLITS_DIR):
    """Load PlantVillage (color | segmented | grayscale) with the fixed leaf-grouped split.

    Reads the Hugging Face copy from data/raw/plantvillage_<variant> and the
    committed split in data/splits/. Test = the official test split,
    untouched; validation = ~15% of all images carved from the official train
    split by leaf (StratifiedGroupKFold, seed 42, see `make_splits`). The
    leakage check runs on every call.

    Args:
        variant: PlantVillage variant saved by data/download.py.
        cfg: experiment config (defaults to configs/base.yaml). Uses
            data.image_size, data.augmentation, train.batch_size and seed.
        splits_dir: folder holding train/val/test.csv.

    Returns:
        (train_ds, val_ds, test_ds, class_names): batched tf.data.Datasets of
        (image in [0, 1], int label); only train_ds is shuffled and augmented
        (pass a cfg with `data.augmentation: []` for an unaugmented train set).
        val_ds and test_ds have a fixed order with the classes mixed (sorted
        by file hash), so `.take(n)` gives a representative subset.
        For the label arrays (e.g. class weights) use `load_split_tables()`.
    """
    cfg = cfg or _default_config()
    splits, class_names = _rows_for_splits(variant, splits_dir)
    check_no_leakage(splits["train"], splits["val"], splits["test"])

    hf = _load_hf(variant)
    index = {name: i for i, name in enumerate(class_names)}
    out = []
    for name, source in (("train", "train"), ("val", "train"), ("test", "test")):
        meta = splits[name]
        if name != "train":
            # The source order is grouped by class; a fixed hash order mixes the
            # classes, so taking the first N examples is still representative.
            meta = meta.sort_values("file_hash", kind="stable").reset_index(drop=True)
        part = hf[source]
        rows = meta["row"].to_numpy()

        def fetch(i, part=part, rows=rows):
            """Return the encoded image bytes of the i-th example of this split."""
            return _image_bytes(part[int(rows[i])]["image"])

        ds = make_dataset(
            fetch, meta["label"].map(index).to_numpy(), training=(name == "train"),
            image_size=cfg["data"]["image_size"], batch_size=cfg["train"]["batch_size"],
            augmentation=cfg["data"].get("augmentation") or [], seed=cfg["seed"],
            cache=(_cache_path(variant, name, cfg["seed"]) if cfg["data"].get("cache") else None))
        out.append(ds)
    return out[0], out[1], out[2], class_names


def compute_class_weights(train_labels) -> dict[int, float]:
    """Inverse-frequency ("balanced") class weights from the TRAIN labels only.

    weight[c] = n_samples / (n_classes * count[c]), the same formula as
    scikit-learn's compute_class_weight("balanced"). Pass the result as
    `class_weight=` to `model.fit`.

    Args:
        train_labels: int class index per training image.

    Returns:
        {class index: weight} for every class present in `train_labels`.
    """
    labels = np.asarray(train_labels).astype(int).ravel()
    classes, counts = np.unique(labels, return_counts=True)
    weights = len(labels) / (len(classes) * counts)
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ---------------------------------------------------------------------------
# PlantDoc (external test set)
# ---------------------------------------------------------------------------
def safe_name(name: str) -> str:
    """Replace characters Windows forbids in file names by `_` (Windows only).

    data/download.py stores PlantDoc files under these names on Windows, so
    the exclusion list and the box CSVs are mapped the same way before
    matching. On Linux/Colab names are unchanged.
    """
    return _UNSAFE_CHARS.sub("_", name) if os.name == "nt" else name


def _safe_path(rel_path: str) -> str:
    """Apply `safe_name` to every component of a "/"-separated relative path."""
    return "/".join(safe_name(p) for p in rel_path.replace("\\", "/").split("/"))


def read_plantdoc_exclusions(path: str = PLANTDOC_EXCLUDE) -> set[str]:
    """Return the PlantDoc paths ("train/<class>/<file>") to drop.

    The list is made by the Evaluation & Deployment engineer. If the file does
    not exist yet, a warning is raised and nothing is excluded.
    """
    if not os.path.exists(path):
        warnings.warn(f"{path} not found: no PlantDoc images excluded")
        return set()
    return {_safe_path(p) for p in pd.read_csv(path)["path"]}


def _read_plantdoc_boxes(boxes_dir: str) -> pd.DataFrame:
    """Read the PlantDoc object-detection boxes, normalised to [0, 1].

    Returns:
        DataFrame with filename, class, box_w, box_h (size of the image the
        box was drawn on) and x0, y0, x1, y1 (fractions of that image).
    """
    parts = []
    for name in ("train_labels.csv", "test_labels.csv"):
        path = os.path.join(boxes_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing; run `python data/download.py` "
                                    "or use mode='full'")
        parts.append(pd.read_csv(path))
    boxes = pd.concat(parts, ignore_index=True)
    for col in ("width", "height", "xmin", "ymin", "xmax", "ymax"):
        boxes[col] = pd.to_numeric(boxes[col], errors="coerce")
    boxes = boxes.dropna(subset=["width", "height", "xmin", "ymin", "xmax", "ymax"])
    boxes = boxes[(boxes["width"] > 0) & (boxes["height"] > 0)]
    out = pd.DataFrame({"filename": boxes["filename"].map(safe_name), "class": boxes["class"],
                        "box_w": boxes["width"], "box_h": boxes["height"]})
    out["x0"] = (boxes["xmin"] / boxes["width"]).clip(0, 1)
    out["y0"] = (boxes["ymin"] / boxes["height"]).clip(0, 1)
    out["x1"] = (boxes["xmax"] / boxes["width"]).clip(0, 1)
    out["y1"] = (boxes["ymax"] / boxes["height"]).clip(0, 1)
    return out[(out["x1"] > out["x0"]) & (out["y1"] > out["y0"])].drop_duplicates()


def list_plantdoc_examples(class_names: list[str], mode: str = "crops",
                           plantdoc_dir: str = PLANTDOC_DIR,
                           boxes_dir: str = PLANTDOC_BOXES_DIR,
                           class_map_path: str = PLANTDOC_CLASS_MAP,
                           exclude_path: str = PLANTDOC_EXCLUDE,
                           min_box_fraction: float = 0.05) -> tuple[pd.DataFrame, dict[str, str]]:
    """List the PlantDoc examples to score, one row per image (full) or per box (crops).

    Steps: map PlantDoc classes to PlantVillage names via the class map (only
    classes in `class_names` are kept), drop every file in the exclusion list
    and unreadable files; in crops mode, look up the detection boxes of each
    image (same file name and class), skip images whose box file was drawn on
    a differently shaped (rotated) image, and keep boxes at least
    `min_box_fraction` of the image on each side.

    Returns:
        (examples, mapping): examples has path, rel_path, plantdoc_class,
        label (PlantVillage name), label_index, width, height and, in crops
        mode, x0, y0, x1, y1; mapping is {PlantDoc class: PlantVillage class}.
    """
    from PIL import Image

    if mode not in ("crops", "full"):
        raise ValueError("mode must be 'crops' or 'full'")
    class_map = pd.read_csv(class_map_path)
    mapping = {r.plantdoc_class: r.plantvillage_class for r in class_map.itertuples()
               if r.plantvillage_class in class_names}
    excluded = read_plantdoc_exclusions(exclude_path)
    index = {name: i for i, name in enumerate(class_names)}

    rows, unreadable = [], 0
    for split in ("train", "test"):
        for pd_class, pv_class in mapping.items():
            folder = os.path.join(plantdoc_dir, split, pd_class)
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                rel = f"{split}/{pd_class}/{fname}"
                if rel in excluded or not fname.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                path = os.path.join(folder, fname)
                try:
                    with Image.open(path) as im:
                        ok = im.format in ("JPEG", "PNG", "BMP", "GIF")
                        width, height = im.size
                except Exception:
                    ok = False
                if not ok:
                    unreadable += 1
                    continue
                rows.append({"path": path, "rel_path": rel, "filename": fname,
                             "plantdoc_class": pd_class, "label": pv_class,
                             "label_index": index[pv_class], "width": width, "height": height})
    if unreadable:
        warnings.warn(f"PlantDoc: skipped {unreadable} unreadable/unsupported files")
    examples = pd.DataFrame(rows, columns=["path", "rel_path", "filename", "plantdoc_class",
                                           "label", "label_index", "width", "height"])
    if mode == "crops":
        boxes = _read_plantdoc_boxes(boxes_dir)
        merged = examples.merge(boxes, left_on=["filename", "plantdoc_class"],
                                right_on=["filename", "class"], how="left")
        no_box = merged["x0"].isna()
        if no_box.any():
            warnings.warn(f"PlantDoc crops: {merged.loc[no_box, 'rel_path'].nunique()} images "
                          "have no detection box and are skipped")
        merged = merged[~no_box]
        # boxes drawn on a differently shaped image (e.g. EXIF-rotated photos,
        # which TensorFlow decodes unrotated) would land in the wrong place
        aspect_ok = ((merged["width"] / merged["height"]) /
                     (merged["box_w"] / merged["box_h"]) - 1).abs() <= 0.02
        if (~aspect_ok).any():
            warnings.warn(f"PlantDoc crops: {merged.loc[~aspect_ok, 'rel_path'].nunique()} images "
                          "skipped: box file drawn on a differently shaped (rotated) image")
        merged = merged[aspect_ok]
        big = ((merged["x1"] - merged["x0"]) >= min_box_fraction) & \
              ((merged["y1"] - merged["y0"]) >= min_box_fraction)
        examples = merged[big].drop(columns=["class", "box_w", "box_h"]).reset_index(drop=True)
    return examples, mapping


def _crop_fraction(image, box):
    """Crop a decoded image to a box given as fractions (x0, y0, x1, y1)."""
    import tensorflow as tf

    shape = tf.cast(tf.shape(image)[:2], tf.float32)
    y0 = tf.cast(tf.floor(box[1] * shape[0]), tf.int32)
    x0 = tf.cast(tf.floor(box[0] * shape[1]), tf.int32)
    y1 = tf.maximum(tf.cast(tf.math.ceil(box[3] * shape[0]), tf.int32), y0 + 1)
    x1 = tf.maximum(tf.cast(tf.math.ceil(box[2] * shape[1]), tf.int32), x0 + 1)
    return image[y0:y1, x0:x1, :]


def load_plantdoc_shared_classes(class_names: list[str], mode: str = "crops",
                                 cfg: dict | None = None, **list_kwargs):
    """Load PlantDoc images of the classes shared with PlantVillage, as a TEST-ONLY set.

    Uses data/plantdoc_class_map.csv for the class mapping and drops every
    file in data/plantdoc_exclude.csv (skipped with a warning if missing).
    PlantDoc is never used for training. Train and test folders are both used
    because we only evaluate on it.

    Args:
        class_names: PlantVillage class names, in model output order.
        mode: "crops" (default) scores each annotated leaf, cut out with the
            boxes of the PlantDoc object-detection release (CC BY 4.0,
            downloaded by data/download.py); "full" scores whole photos.
        cfg: config for image_size and batch size (defaults to base.yaml).
        **list_kwargs: passed to `list_plantdoc_examples` (paths, box filter).

    Returns:
        (test_ds, mapping): a batched, unshuffled, unaugmented
        tf.data.Dataset of (image in [0, 1], int label), in the row order of
        `list_plantdoc_examples` (same arguments); and
        {PlantDoc class: PlantVillage class}.
    """
    import tensorflow as tf

    cfg = cfg or _default_config()
    size = int(cfg["data"]["image_size"])
    examples, mapping = list_plantdoc_examples(class_names, mode=mode, **list_kwargs)
    labels = examples["label_index"].to_numpy(np.int32)
    paths = examples["path"].to_numpy(str)
    if mode == "crops":
        boxes = examples[["x0", "y0", "x1", "y1"]].to_numpy(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, boxes, labels)).map(
            lambda p, b, y: (preprocess(_crop_fraction(decode_image(tf.io.read_file(p)), b), size), y),
            num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels)).map(
            lambda p, y: (preprocess(decode_image(tf.io.read_file(p)), size), y),
            num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(cfg["train"]["batch_size"]).prefetch(tf.data.AUTOTUNE), mapping


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main() -> None:
    """CLI: `make-splits` builds data/splits/, `check` runs the leakage check on it."""
    parser = argparse.ArgumentParser(description="FrostLeaf data utilities")
    parser.add_argument("command", choices=["make-splits", "check"])
    args = parser.parse_args()
    if args.command == "make-splits":
        summary = make_plantvillage_splits()
        print("sizes:", summary["sizes"])
        print("dropped from train:", summary["dropped"])
        print(summary["per_class"].to_string())
    tables = load_split_tables()
    check_no_leakage(tables["train"], tables["val"], tables["test"])
    print("check_no_leakage: OK (no shared leaf_id or file hash between splits)",
          {name: len(t) for name, t in tables.items()})


if __name__ == "__main__":
    main()
