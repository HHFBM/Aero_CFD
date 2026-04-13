"""Evaluate a trained surrogate model."""

from __future__ import annotations

import argparse
from pathlib import Path

from cfd_operator.config import load_config
from cfd_operator.data import CFDDataModule
from cfd_operator.evaluators import Evaluator
from cfd_operator.inference import Predictor
from cfd_operator.visualization import plot_loss_curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CFD operator surrogate.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--override", action="append", default=[], help="Override config keys")
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.override)
    checkpoint_path = args.checkpoint or config.serve.checkpoint_path
    predictor = Predictor.from_checkpoint(checkpoint_path, device=config.experiment.device)

    data_module = CFDDataModule(config=config.data, batch_size=config.eval.batch_size)
    data_module.setup()
    evaluator = Evaluator(
        config=config.eval,
        model=predictor.model,
        data_module=data_module,
        normalizers=predictor.normalizers,
        device=config.experiment.device,
    )
    output_dir = Path(config.experiment.run_dir) / "eval"
    metrics = evaluator.evaluate(output_dir=output_dir)
    history_csv = Path(config.experiment.run_dir) / "reports" / "history.csv"
    if history_csv.exists() and config.eval.save_plots:
        plot_loss_curves(history_csv, output_dir / "loss_curve.png")
    print(metrics)


if __name__ == "__main__":
    main()
