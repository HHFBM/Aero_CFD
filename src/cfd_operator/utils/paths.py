"""Output path conventions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cfd_operator.config.schemas import ExperimentConfig
from cfd_operator.utils.io import ensure_dir


@dataclass
class RunPaths:
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    figures_dir: Path
    reports_dir: Path


def build_run_paths(experiment: ExperimentConfig) -> RunPaths:
    run_dir = ensure_dir(experiment.run_dir)
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    logs_dir = ensure_dir(run_dir / "logs")
    figures_dir = ensure_dir(run_dir / "figures")
    reports_dir = ensure_dir(run_dir / "reports")
    return RunPaths(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
        figures_dir=figures_dir,
        reports_dir=reports_dir,
    )
