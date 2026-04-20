"""Evaluate a trained surrogate model."""

from __future__ import annotations

import argparse
from pathlib import Path

from cfd_operator.config import load_config
from cfd_operator.evaluators import run_split_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CFD operator surrogate.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default=None, help="Optional split override, e.g. test or benchmark_holdout.")
    parser.add_argument("--override", action="append", default=[], help="Override config keys")
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.override)
    checkpoint_path = args.checkpoint or config.serve.checkpoint_path
    split_name = args.split or config.eval.split_name
    result = run_split_evaluation(
        config=config,
        checkpoint_path=checkpoint_path,
        split_name=split_name,
        output_dir=Path(config.experiment.run_dir) / "eval" / split_name,
    )
    print(result["metrics"])


if __name__ == "__main__":
    main()
