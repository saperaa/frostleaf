"""Download the datasets into data/raw/ (not committed to git).

Usage:
    python data/download.py                    # PlantVillage color + segmented, PlantDoc
    python data/download.py --variants color

Sources (cite both in the report):
* PlantVillage: Hugging Face `mohanty/PlantVillage` (configs: color,
  grayscale, segmented), with leaf-grouped 80/20 splits. Hughes & Salathe 2015,
  arXiv:1511.08060; Mohanty et al. 2016.
* PlantDoc: https://github.com/pratikkayal/PlantDoc-Dataset (CC BY 4.0).
  Singh et al. 2019, arXiv:1911.10317.
"""

from __future__ import annotations

import argparse
import os
import subprocess

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
PLANTDOC_REPO = "https://github.com/pratikkayal/PlantDoc-Dataset.git"


def download_plantvillage(variant: str) -> None:
    """Download one PlantVillage variant and save it in Hugging Face disk format."""
    from datasets import load_dataset

    out = os.path.join(RAW_DIR, f"plantvillage_{variant}")
    if os.path.exists(out):
        print(f"exists, skipping: {out}")
        return
    ds = load_dataset("mohanty/PlantVillage", variant)
    ds.save_to_disk(out)
    print(f"saved {variant}: {ds}")


def download_plantdoc() -> None:
    """Shallow-clone the PlantDoc classification images."""
    out = os.path.join(RAW_DIR, "plantdoc")
    if os.path.exists(out):
        print(f"exists, skipping: {out}")
        return
    subprocess.run(["git", "clone", "--depth", "1", PLANTDOC_REPO, out], check=True)


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


if __name__ == "__main__":
    main()
