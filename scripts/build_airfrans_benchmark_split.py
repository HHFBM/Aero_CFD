"""Create a benchmark-aware AirfRANS split dataset.

This script takes an existing AirfRANS-style NPZ dataset, removes any existing
split indices, and writes a new dataset with:

- train_indices
- val_indices
- test_indices
- benchmark_holdout_indices

The benchmark_holdout split is frozen and excluded from the normal development
train/val/test loop.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from cfd_operator.data.splitting import build_frozen_benchmark_splits


def _load_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def _drop_existing_split_keys(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value for key, value in payload.items() if not key.endswith("_indices")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a frozen benchmark_holdout split for an AirfRANS-style NPZ dataset.")
    parser.add_argument("--input", required=True, help="Source NPZ path.")
    parser.add_argument("--output", required=True, help="Output NPZ path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splitting.")
    parser.add_argument("--benchmark-holdout-ratio", type=float, default=0.1, help="Fraction of full dataset reserved as frozen benchmark_holdout.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio within the remaining 90%% development pool.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio within the remaining 90%% development pool.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _drop_existing_split_keys(_load_payload(input_path))
    geometry_ids = np.asarray(payload["airfoil_id"])
    rng = np.random.default_rng(args.seed)
    split_payload = build_frozen_benchmark_splits(
        geometry_ids=geometry_ids,
        benchmark_holdout_ratio=args.benchmark_holdout_ratio,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        rng=rng,
    )
    payload.update(split_payload)
    np.savez_compressed(output_path, **payload)

    manifest = {
        "dataset_path": str(output_path),
        "source": "airfrans_benchmark_split",
        "based_on": str(input_path),
        "total_sample_count": int(geometry_ids.shape[0]),
        "main_pool_count": int(geometry_ids.shape[0] - split_payload["benchmark_holdout_indices"].shape[0]),
        "train_size": int(split_payload["train_indices"].shape[0]),
        "val_size": int(split_payload["val_indices"].shape[0]),
        "test_size": int(split_payload["test_indices"].shape[0]),
        "benchmark_holdout_size": int(split_payload["benchmark_holdout_indices"].shape[0]),
        "random_seed": int(args.seed),
        "split_ratios": {
            "benchmark_holdout_ratio": float(args.benchmark_holdout_ratio),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(max(0.0, 1.0 - args.train_ratio - args.val_ratio)),
        },
        "split_strategy": "benchmark_holdout_then_main_pool_train_val_test",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved benchmark split dataset to {output_path}")
    print(f"Saved split manifest to {output_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
