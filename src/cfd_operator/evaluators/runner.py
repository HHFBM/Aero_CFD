"""Shared helpers for single-split evaluation runs."""

from __future__ import annotations

from pathlib import Path

from cfd_operator.config.schemas import ProjectConfig
from cfd_operator.data import CFDDataModule
from cfd_operator.evaluators.evaluator import Evaluator
from cfd_operator.inference import Predictor
from cfd_operator.visualization import plot_loss_curves


def run_split_evaluation(
    *,
    config: ProjectConfig,
    checkpoint_path: str | Path,
    split_name: str,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    predictor = Predictor.from_checkpoint(checkpoint_path, device=config.experiment.device)

    data_module = CFDDataModule(config=config.data, batch_size=config.eval.batch_size)
    data_module.setup()
    evaluator = Evaluator(
        config=config.eval.model_copy(update={"split_name": split_name}),
        model=predictor.model,
        data_module=data_module,
        normalizers=predictor.normalizers,
        device=config.experiment.device,
    )
    resolved_output_dir = Path(output_dir) if output_dir is not None else Path(config.experiment.run_dir) / "eval" / split_name
    metrics = evaluator.evaluate(output_dir=resolved_output_dir)
    history_csv = Path(config.experiment.run_dir) / "reports" / "history.csv"
    if history_csv.exists() and config.eval.save_plots:
        plot_loss_curves(history_csv, resolved_output_dir / "loss_curve.png")
    return {
        "split_name": split_name,
        "metrics": metrics,
        "output_dir": str(resolved_output_dir),
    }
