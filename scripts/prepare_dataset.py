"""Generate a toy CFD-like airfoil dataset."""

from __future__ import annotations

import argparse

from cfd_operator.config import load_config
from cfd_operator.data.synthetic import SyntheticAirfoilDatasetGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare toy airfoil dataset.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", action="append", default=[], help="Override config keys, e.g. data.num_samples=64")
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.override)
    generator = SyntheticAirfoilDatasetGenerator(config=config.data, seed=config.experiment.seed)
    output_path = generator.save(config.data.dataset_path)
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main()

