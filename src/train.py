"""Training entry point: one config in, one logged run out.

Owners: Tech Lead (entry point) and Experimentation Engineer (logging, curves).

Usage (once implemented):
    python -m src.train --config abl_precision_bf16_rtn.yaml --seed 42 --author mohamed

Responsibilities:
* load the config (src.config.load_config) and fix all random seeds;
* build data (src.data) and the model (src.models);
* train with EarlyStopping on val_loss;
* append one row to experiments/log.csv (see its header for the columns,
  including mean_freeze_rate and mean_update_ratio from PrecisionModel);
* save learning curves to experiments/curves/<run_id>.png.
"""

from __future__ import annotations

import argparse


def main() -> None:
    """Parse arguments and run one training job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--author", required=True)
    parser.parse_args()
    raise NotImplementedError("Tech Lead: implement in Week 11")


if __name__ == "__main__":
    main()
