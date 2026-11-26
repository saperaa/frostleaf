"""Download the datasets into data/raw/ (not committed to git).

Usage:
    python data/download.py                    # PlantVillage color + segmented, PlantDoc + boxes
    python data/download.py --variants color

Sources (cite both in the report):
* PlantVillage: Hugging Face `mohanty/PlantVillage` (configs: color,
  grayscale, segmented), with leaf-grouped 80/20 splits. Hughes & Salathe 2015,
  arXiv:1511.08060; Mohanty et al. 2016.
* PlantDoc: https://github.com/pratikkayal/PlantDoc-Dataset (CC BY 4.0) and,
  for leaf bounding boxes, https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset
  (CC BY 4.0). Singh et al. 2019, arXiv:1911.10317.

Notes:
* The Hugging Face repo is a loading script (plant_village.py), which
  `datasets` >= 4 no longer runs. `build_plantvillage` downloads the same
  files at a pinned revision and repeats that script's logic line for line
  (including how `leaf_id` is derived), then saves a DatasetDict with the same
  features via `save_to_disk`. Image bytes are embedded, so the saved copy is
  self-contained.
* PlantDoc is fetched as a GitHub zip at a pinned commit instead of
  `git clone`: 86 of its file names contain `?`, which Windows cannot create.
  On Windows those characters are replaced by `_` (see `safe_name`); the
  loader in src/data.py applies the same mapping. Six PlantDoc files differ
  from another file only by letter case (e.g. CAR1.jpg / car1.jpg); Windows
  cannot hold both, so there the second is skipped with a warning. Run on
  Colab/Linux for the complete set.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import urllib.request
import zipfile

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")

PLANTVILLAGE_REPO = "mohanty/PlantVillage"
PLANTVILLAGE_REVISION = "a12d9d3ebb66dd0f7668e577194d6057e80f3b16"
PLANTDOC_ZIP = ("https://github.com/pratikkayal/PlantDoc-Dataset/archive/"
                "5467f6012d78d1c446145d5f582da6096f852ae8.zip")
PLANTDOC_BOXES_URL = ("https://raw.githubusercontent.com/pratikkayal/PlantDoc-Object-Detection-Dataset/"
                      "4730a233a555b30ee98e0879c63ad25d82407455/{name}")

_UNSAFE = re.compile(r'[<>:"|?*]')


def safe_name(name: str) -> str:
    """Return `name` with characters Windows forbids in file names replaced by `_`.

    Only applied on Windows; elsewhere names are kept as they are.
    """
    return _UNSAFE.sub("_", name) if os.name == "nt" else name


def _leaf_id(file_name: str, class_name: str, leaf_map: dict) -> str:
    """Derive `leaf_id` exactly as the Hugging Face plant_village.py script does."""
    image_identifier = file_name.replace("_final_masked", "")
    if "___" in image_identifier:
        image_identifier = image_identifier.split("___")[-1]
    image_identifier = image_identifier.split("copy")[0]
    for ext in (".jpg", ".JPG", ".png", ".PNG"):
        image_identifier = image_identifier.replace(ext, "")
    image_identifier = image_identifier.strip()

    lookup_key = image_identifier.lower().strip()
    if lookup_key not in leaf_map:
        return f"fallback_{image_identifier}"
    suggestions = leaf_map[lookup_key]
    if len(suggestions) == 1:
        return suggestions[0]
    for suggestion in suggestions:
        if class_name in suggestion:
            return suggestion
    return f"fallback_{image_identifier}"


def build_plantvillage(variant: str, out: str) -> None:
    """Build one PlantVillage variant from the Hugging Face repo files and save it to `out`."""
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Value
    from huggingface_hub import hf_hub_download

    def fetch(filename: str) -> str:
        """Download one file of the dataset repo (cached) and return its local path."""
        return hf_hub_download(PLANTVILLAGE_REPO, filename, repo_type="dataset",
                               revision=PLANTVILLAGE_REVISION)

    script = open(fetch("plant_village.py"), encoding="utf-8").read()
    class_names = re.findall(r'^\s+"([^"]+___[^"]+)",$', script, flags=re.M)
    assert len(class_names) == 38, f"expected 38 classes in plant_village.py, got {len(class_names)}"
    with open(fetch("leaf_grouping/leaf-map.json"), encoding="utf-8") as f:
        leaf_map = json.load(f)
    archive = zipfile.ZipFile(fetch("data.zip"))
    members = {name for name in archive.namelist() if not name.endswith("/")}

    features = Features({
        "image": Image(),
        "image_path": Value("string"),
        "label": ClassLabel(names=class_names),
        "crop": Value("string"),
        "disease": Value("string"),
        "leaf_id": Value("string"),
    })

    def examples(split_file: str):
        """Yield the rows of one split, in the order of its split file."""
        with open(split_file, encoding="utf-8") as f:
            file_list = [line.strip() for line in f if line.strip()]
        prefix = next(m[: -len(file_list[0])] for m in members if m.endswith(file_list[0]))
        for rel in file_list:
            parts = rel.split("/")
            if len(parts) < 4:
                continue
            class_name, file_name = parts[2], parts[3]
            crop, _, disease = class_name.partition("___")
            yield {
                "image": {"bytes": archive.read(prefix + rel), "path": rel},
                "image_path": rel,
                "label": class_name,
                "crop": crop,
                "disease": disease or "unknown",
                "leaf_id": _leaf_id(file_name, class_name, leaf_map),
            }

    splits = {}
    for split in ("train", "test"):
        split_file = fetch(f"splits/{variant}_{split}.txt")
        splits[split] = Dataset.from_generator(
            examples, gen_kwargs={"split_file": split_file}, features=features)
    DatasetDict(splits).save_to_disk(out)


def download_plantvillage(variant: str) -> None:
    """Download one PlantVillage variant and save it in Hugging Face disk format."""
    out = os.path.join(RAW_DIR, f"plantvillage_{variant}")
    if os.path.exists(out):
        print(f"exists, skipping: {out}")
        return
    build_plantvillage(variant, out)
    from datasets import load_from_disk

    print(f"saved {variant}: {load_from_disk(out)}")


def download_plantdoc() -> None:
    """Download the PlantDoc classification images (pinned commit) into data/raw/plantdoc."""
    out = os.path.join(RAW_DIR, "plantdoc")
    if os.path.exists(out):
        print(f"exists, skipping: {out}")
        return
    print(f"downloading {PLANTDOC_ZIP}")
    with urllib.request.urlopen(PLANTDOC_ZIP) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    tmp = out + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    written, skipped = set(), []
    for info in archive.infolist():
        rel = info.filename.split("/", 1)[1] if "/" in info.filename else ""
        if not rel or info.is_dir():
            continue
        target = os.path.join(tmp, *[safe_name(p) for p in rel.split("/")])
        if os.name == "nt":
            # Windows file names ignore case: `CAR1.jpg` and `car1.jpg` would
            # overwrite each other, so keep the first and skip the clash.
            if target.lower() in written:
                skipped.append(rel)
                continue
            written.add(target.lower())
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(archive.read(info))
    os.replace(tmp, out)
    print(f"saved PlantDoc: {out}")
    if skipped:
        print(f"WARNING: skipped {len(skipped)} files whose names differ only by case "
              f"(Windows cannot store both): {skipped}")


def download_plantdoc_boxes() -> None:
    """Download the PlantDoc object-detection box CSVs into data/raw/plantdoc_boxes."""
    out = os.path.join(RAW_DIR, "plantdoc_boxes")
    os.makedirs(out, exist_ok=True)
    for name in ("train_labels.csv", "test_labels.csv"):
        target = os.path.join(out, name)
        if os.path.exists(target):
            print(f"exists, skipping: {target}")
            continue
        urllib.request.urlretrieve(PLANTDOC_BOXES_URL.format(name=name), target)
        print(f"saved {target}")


def main() -> None:
    """Parse arguments and download everything requested."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["color", "segmented"])
    parser.add_argument("--skip-plantdoc", action="store_true")
    args = parser.parse_args()
    os.makedirs(RAW_DIR, exist_ok=True)
    for variant in args.variants:
        download_plantvillage(variant)
    if not args.skip_plantdoc:
        download_plantdoc()
        download_plantdoc_boxes()


if __name__ == "__main__":
    main()
