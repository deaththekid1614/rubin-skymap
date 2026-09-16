"""scripts/train_model.py — CLI wrapper around the LightGBM training pipeline.

Usage
-----
    python scripts/train_model.py
    python scripts/train_model.py --data data/raw/plasticc_synthetic.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rubin_skymap.config import cfg_get, load_config
from rubin_skymap.logging_setup import setup_logging
from rubin_skymap.models.train import train_model

_log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LightGBM transient classifier.")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to the training parquet file. Defaults to config value.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for train_model.py."""
    args = _parse_args()
    cfg = load_config()
    setup_logging(cfg_get(cfg, "app.log_level", "INFO"))

    data_path: str
    if args.data:
        data_path = args.data
    else:
        data_path = cfg_get(cfg, "data.parquet_synthetic", "data/raw/plasticc_synthetic.parquet")
        # Fall back to processed train parquet if synthetic does not exist.
        if not Path(data_path).exists():
            data_path = cfg_get(cfg, "data.parquet_train", "data/processed/plasticc_train.parquet")

    if not Path(data_path).exists():
        print(
            f"ERROR: Training data not found at '{data_path}'.\n"
            "Run: python scripts/download_plasticc.py --synthetic",
            file=sys.stderr,
        )
        sys.exit(1)

    model_path = cfg_get(cfg, "model.model_path", "models/lgbm_v1.txt")
    label_map_path = cfg_get(cfg, "model.label_map_path", "models/label_map.json")

    params = {
        "num_leaves": cfg_get(cfg, "model.num_leaves", 31),
        "learning_rate": cfg_get(cfg, "model.learning_rate", 0.05),
        "n_estimators": cfg_get(cfg, "model.n_estimators", 300),
        "random_state": cfg_get(cfg, "model.random_state", 42),
    }

    print(f"Training on: {data_path}")
    print(f"Model will be saved to: {model_path}")

    metrics = train_model(
        data_parquet=data_path,
        model_path=model_path,
        label_map_path=label_map_path,
        params=params,
    )

    print("\n=== Training Results ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nMacro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
