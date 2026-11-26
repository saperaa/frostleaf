"""Tests for src/data.py. They use tiny synthetic data, so no download is needed."""

import os

import numpy as np
import pandas as pd
import pytest

from src.data import (
    SPLITS_DIR,
    build_augmentation,
    check_no_leakage,
    compute_class_weights,
    list_plantdoc_examples,
    load_split_tables,
    make_dataset,
    make_splits,
    preprocess,
)


def _split(prefix, leaves, start=0):
    """Return a small split table whose rows are one photo per given leaf."""
    return pd.DataFrame({
        "image_id": [f"{prefix}/{i}.jpg" for i in range(start, start + len(leaves))],
        "leaf_id": leaves,
        "label": ["a"] * len(leaves),
        "file_hash": [f"{prefix}-hash-{i}" for i in range(start, start + len(leaves))],
    })


def _synthetic_plantvillage(n_classes=6, leaves_per_class=40, seed=0):
    """Return (official_train, official_test) with 1-4 photos per leaf and a class imbalance."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_classes):
        for leaf in range(leaves_per_class * (c + 1) // 2):
            for photo in range(rng.integers(1, 5)):
                rows.append({"image_id": f"c{c}/l{leaf}_{photo}.jpg", "leaf_id": f"c{c}:{leaf}",
                             "label": f"class_{c}", "file_hash": f"h{c}_{leaf}_{photo}"})
    df = pd.DataFrame(rows)
    test_leaves = set(df["leaf_id"].drop_duplicates().sample(frac=0.2, random_state=seed))
    return (df[~df["leaf_id"].isin(test_leaves)].reset_index(drop=True),
            df[df["leaf_id"].isin(test_leaves)].reset_index(drop=True))


# --- check_no_leakage -------------------------------------------------------
def test_check_no_leakage_passes_on_clean_split():
    """Disjoint leaves and hashes pass silently."""
    check_no_leakage(_split("tr", ["L1", "L2"]), _split("va", ["L3"], 10), _split("te", ["L4"], 20))


def test_check_no_leakage_catches_planted_leaf_overlap():
    """A leaf planted in both train and test is caught."""
    with pytest.raises(AssertionError, match=r"leaf_id.*train and test"):
        check_no_leakage(_split("tr", ["L1", "L2"]), _split("va", ["L3"], 10),
                         _split("te", ["L4", "L2"], 20))


def test_check_no_leakage_catches_planted_identical_file():
    """The same image bytes (hash) under a different leaf ID are caught."""
    train, val, test = _split("tr", ["L1", "L2"]), _split("va", ["L3"], 10), _split("te", ["L4"], 20)
    val.loc[0, "file_hash"] = train.loc[1, "file_hash"]
    with pytest.raises(AssertionError, match=r"file_hash.*train and val"):
        check_no_leakage(train, val, test)


def test_check_no_leakage_accepts_plain_leaf_id_lists():
    """Plain lists of leaf IDs work too (no hash check then)."""
    check_no_leakage(["L1", "L2"], ["L3"], ["L4"])
    with pytest.raises(AssertionError):
        check_no_leakage(["L1", "L2"], ["L2"], ["L4"])


def test_committed_splits_have_no_leakage():
    """The CSVs in data/splits/ (if generated) pass the leakage check."""
    if not os.path.exists(os.path.join(SPLITS_DIR, "train.csv")):
        pytest.skip("data/splits/ not generated yet")
    tables = load_split_tables()
    check_no_leakage(tables["train"], tables["val"], tables["test"])


# --- make_splits ------------------------------------------------------------
def test_make_splits_is_grouped_stratified_and_reproducible():
    """Validation is ~val_fraction of all images, by leaf, with every class, and repeatable."""
    official_train, official_test = _synthetic_plantvillage()
    train, val, test, _ = make_splits(official_train, official_test, val_fraction=0.15, seed=42)
    check_no_leakage(train, val, test)
    n_total = len(official_train) + len(official_test)
    assert 0.10 < len(val) / n_total < 0.22
    assert set(val["label"]) == set(official_train["label"])
    assert len(train) + len(val) == len(official_train)
    pd.testing.assert_frame_equal(test, official_test)  # test untouched
    train2, val2, _, _ = make_splits(official_train, official_test, val_fraction=0.15, seed=42)
    assert set(val2["image_id"]) == set(val["image_id"])


def test_make_splits_removes_train_rows_that_leak_into_test():
    """Train rows sharing a leaf or a file with test, and duplicate copies, are dropped."""
    official_train, official_test = _synthetic_plantvillage()
    official_train.loc[0, "leaf_id"] = official_test.loc[0, "leaf_id"]      # leaf in test
    official_train.loc[1, "file_hash"] = official_test.loc[1, "file_hash"]  # same file as test
    official_train.loc[3, "file_hash"] = official_train.loc[2, "file_hash"]  # duplicate in train
    train, val, test, dropped = make_splits(official_train, official_test)
    check_no_leakage(train, val, test)
    assert dropped["train_leaf_in_test"] >= 1
    assert dropped["train_file_in_test"] == 1
    assert dropped["train_duplicate_copy"] + dropped["train_duplicate_label_conflict"] >= 1
    assert not pd.concat([train, val])["file_hash"].duplicated().any()


# --- class weights ----------------------------------------------------------
def test_compute_class_weights_is_balanced_inverse_frequency():
    """Matches n / (k * count_c) and scikit-learn's "balanced" weights."""
    from sklearn.utils.class_weight import compute_class_weight

    labels = np.array([0] * 8 + [1] * 2 + [2] * 6)
    weights = compute_class_weights(labels)
    expected = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=labels)
    assert weights == pytest.approx({0: expected[0], 1: expected[1], 2: expected[2]})
    assert weights[1] == pytest.approx(16 / (3 * 2))
    assert sum(weights[c] * n for c, n in zip((0, 1, 2), (8, 2, 6))) == pytest.approx(16)


# --- preprocessing and augmentation ------------------------------------------
def _images(n=4, size=40, seed=0):
    """Return n random uint8 RGB images with a clear left/right asymmetry."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 256, size=(n, size, size, 3), dtype=np.uint8)
    x[:, :, : size // 2] //= 4
    return x


def test_preprocess_resizes_and_scales_to_unit_range():
    """Output has the requested size, float32 dtype and values in [0, 1]."""
    out = preprocess(_images(), image_size=32).numpy()
    assert out.shape == (4, 32, 32, 3) and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert preprocess(np.full((10, 10, 3), 255, np.uint8), image_size=8).numpy().max() == 1.0


def test_build_augmentation_has_requested_layers_and_rejects_unknown():
    """flip + rotate give two random layers; an unknown name raises."""
    aug = build_augmentation(["flip", "rotate"])
    assert len(aug.layers) == 2
    with pytest.raises(ValueError):
        build_augmentation(["flip", "cutmix"])


def test_augmentation_is_not_applied_to_validation():
    """A validation pipeline equals plain preprocessing; the train one changes the images."""
    x, y = _images(), np.arange(4)
    expected = preprocess(x, image_size=32).numpy()
    names = ["flip", "rotate", "brightness", "contrast"]

    val_ds = make_dataset(x, y, training=False, image_size=32, batch_size=4, augmentation=names)
    for _ in range(2):  # every epoch identical and in order
        images, labels = next(iter(val_ds))
        np.testing.assert_array_equal(labels.numpy(), y)
        np.testing.assert_allclose(images.numpy(), expected, atol=1e-6)

    train_ds = make_dataset(x, y, training=True, image_size=32, batch_size=4,
                            augmentation=names, shuffle=False)
    images, _ = next(iter(train_ds))
    assert not np.allclose(images.numpy(), expected, atol=1e-3)


def test_make_dataset_decodes_bytes_from_a_fetch_function():
    """The bytes path (used for the Hugging Face data) decodes and preprocesses."""
    import tensorflow as tf

    x = _images(n=3)
    encoded = [tf.io.encode_png(img).numpy() for img in x]
    ds = make_dataset(lambda i: encoded[i], [0, 1, 2], training=False, image_size=40, batch_size=3)
    images, _ = next(iter(ds))
    np.testing.assert_allclose(images.numpy(), preprocess(x, image_size=40).numpy(), atol=1e-6)


# --- PlantDoc ---------------------------------------------------------------
def _fake_plantdoc(root):
    """Write a 2-class PlantDoc folder tree, a class map and a box CSV under root."""
    from PIL import Image

    for split, cls, name in [("train", "Apple leaf", "a1.jpg"), ("train", "Apple leaf", "a2.jpg"),
                             ("test", "Apple leaf", "a3.jpg"), ("train", "Corn rust leaf", "c1.jpg")]:
        os.makedirs(root / "plantdoc" / split / cls, exist_ok=True)
        Image.new("RGB", (100, 80), (0, 120, 0)).save(root / "plantdoc" / split / cls / name)
    (root / "plantdoc" / "train" / "Apple leaf" / "notes.txt").write_text("not an image")
    pd.DataFrame({"plantdoc_class": ["Apple leaf", "Corn rust leaf", "Unmapped leaf"],
                  "plantvillage_class": ["Apple___healthy", "Corn_(maize)___Common_rust_", "Nope"]}
                 ).to_csv(root / "map.csv", index=False)
    pd.DataFrame({"path": ["train/Apple leaf/a2.jpg"], "plantdoc_class": ["Apple leaf"],
                  "reason": ["watermark"]}).to_csv(root / "exclude.csv", index=False)
    os.makedirs(root / "boxes")
    pd.DataFrame({"filename": ["a1.jpg", "a1.jpg", "a3.jpg", "c1.jpg"],
                  "width": [200, 200, 200, 200], "height": [160, 160, 160, 160],
                  "class": ["Apple leaf"] * 3 + ["Corn rust leaf"],
                  "xmin": [0, 100, 10, 0], "ymin": [0, 80, 10, 0],
                  "xmax": [100, 200, 12, 200], "ymax": [80, 160, 12, 160]}
                 ).to_csv(root / "boxes" / "train_labels.csv", index=False)
    pd.DataFrame(columns=["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
                 ).to_csv(root / "boxes" / "test_labels.csv", index=False)


CLASSES = ["Apple___healthy", "Corn_(maize)___Common_rust_", "Tomato___healthy"]


def test_plantdoc_listing_maps_classes_and_drops_excluded_files(tmp_path):
    """Full mode: mapped classes only, excluded and non-image files dropped."""
    _fake_plantdoc(tmp_path)
    examples, mapping = list_plantdoc_examples(
        CLASSES, mode="full", plantdoc_dir=tmp_path / "plantdoc", class_map_path=tmp_path / "map.csv",
        exclude_path=tmp_path / "exclude.csv")
    assert mapping == {"Apple leaf": "Apple___healthy", "Corn rust leaf": "Corn_(maize)___Common_rust_"}
    assert sorted(examples["rel_path"]) == ["test/Apple leaf/a3.jpg", "train/Apple leaf/a1.jpg",
                                            "train/Corn rust leaf/c1.jpg"]
    assert set(examples["label_index"]) == {0, 1}


def test_plantdoc_missing_exclusion_file_only_warns(tmp_path):
    """Without plantdoc_exclude.csv the loader warns and keeps every image."""
    _fake_plantdoc(tmp_path)
    with pytest.warns(UserWarning, match="no PlantDoc images excluded"):
        examples, _ = list_plantdoc_examples(
            CLASSES, mode="full", plantdoc_dir=tmp_path / "plantdoc",
            class_map_path=tmp_path / "map.csv", exclude_path=tmp_path / "missing.csv")
    assert len(examples) == 4


def test_plantdoc_crops_one_row_per_box_and_tiny_boxes_dropped(tmp_path):
    """Crops mode: one example per box, boxes normalised, a 2-pixel box filtered out."""
    _fake_plantdoc(tmp_path)
    examples, _ = list_plantdoc_examples(
        CLASSES, mode="crops", plantdoc_dir=tmp_path / "plantdoc", boxes_dir=tmp_path / "boxes",
        class_map_path=tmp_path / "map.csv", exclude_path=tmp_path / "exclude.csv")
    assert sorted(examples["rel_path"]) == ["train/Apple leaf/a1.jpg", "train/Apple leaf/a1.jpg",
                                            "train/Corn rust leaf/c1.jpg"]
    first = examples.sort_values(["rel_path", "x0"]).iloc[0]
    assert (first.x0, first.y0, first.x1, first.y1) == pytest.approx((0, 0, 0.5, 0.5))


def test_plantdoc_crops_skip_boxes_drawn_on_a_rotated_image(tmp_path):
    """A box file whose width/height are swapped (EXIF rotation) is not used."""
    _fake_plantdoc(tmp_path)
    boxes = pd.read_csv(tmp_path / "boxes" / "train_labels.csv")
    boxes.loc[boxes["filename"] == "c1.jpg", ["width", "height"]] = [160, 200]
    boxes.to_csv(tmp_path / "boxes" / "train_labels.csv", index=False)
    with pytest.warns(UserWarning, match="rotated"):
        examples, _ = list_plantdoc_examples(
            CLASSES, mode="crops", plantdoc_dir=tmp_path / "plantdoc", boxes_dir=tmp_path / "boxes",
            class_map_path=tmp_path / "map.csv", exclude_path=tmp_path / "exclude.csv")
    assert "train/Corn rust leaf/c1.jpg" not in set(examples["rel_path"])


def test_cached_pipeline_matches_uncached(tmp_path):
    """The decoded-image cache must not change what the model sees."""
    import io

    import numpy as np
    from PIL import Image

    from src.data import make_dataset

    rng = np.random.default_rng(0)
    blobs = []
    for _ in range(12):
        buf = io.BytesIO()
        Image.fromarray(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(buf, "PNG")
        blobs.append(buf.getvalue())
    labels = np.arange(12) % 3
    plain = make_dataset(lambda i: blobs[i], labels, training=False, image_size=16, batch_size=5)
    cached = make_dataset(lambda i: blobs[i], labels, training=False, image_size=16,
                          batch_size=5, cache=str(tmp_path / "c" / "val"))
    for _ in range(2):  # second pass reads from the cache file
        a = np.concatenate([x.numpy() for x, _ in plain])
        b = np.concatenate([x.numpy() for x, _ in cached])
        assert np.array_equal(a, b)


def test_cached_training_pipeline_reshuffles_each_epoch(tmp_path):
    """With a cache, training data still comes in a different order every epoch."""
    import numpy as np

    from src.data import make_dataset

    labels = np.arange(64)
    blobs = []
    import io

    from PIL import Image
    for i in range(64):
        buf = io.BytesIO()
        Image.fromarray(np.full((8, 8, 3), i, np.uint8)).save(buf, "PNG")
        blobs.append(buf.getvalue())
    ds = make_dataset(lambda i: blobs[i], labels, training=True, image_size=8, batch_size=64,
                      cache=str(tmp_path / "train"), shuffle_buffer=64)
    first = next(iter(ds))[1].numpy()
    second = next(iter(ds))[1].numpy()
    assert sorted(first) == list(range(64)) and not np.array_equal(first, second)
