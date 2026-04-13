"""Model evaluation and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import torch

from cfd_operator.config.schemas import EvalConfig
from cfd_operator.data.module import CFDDataModule, NormalizerBundle
from cfd_operator.evaluators.metrics import mae, mse, relative_error, rmse
from cfd_operator.losses import pressure_to_cp
from cfd_operator.models.base import BaseOperatorModel
from cfd_operator.utils.io import save_json
from cfd_operator.visualization import (
    plot_cp_comparison,
    plot_field_scatter,
    plot_loss_curves,
    plot_scalar_scatter,
)


@dataclass
class Evaluator:
    config: EvalConfig
    model: BaseOperatorModel
    data_module: CFDDataModule
    normalizers: NormalizerBundle
    device: str = "cpu"

    def evaluate(self, output_dir: Union[str, Path]) -> dict[str, float]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        loader = self.data_module.test_dataloader(batch_size=self.config.batch_size, split_name=self.config.split_name)
        device = torch.device(self.device)
        self.model.to(device)
        self.model.eval()
        plot_paths: List[str] = []

        field_true_all: List[np.ndarray] = []
        field_pred_all: List[np.ndarray] = []
        scalar_true_all: List[np.ndarray] = []
        scalar_pred_all: List[np.ndarray] = []
        cp_true_all: List[np.ndarray] = []
        cp_pred_all: List[np.ndarray] = []
        saved_samples = 0

        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
                outputs = self.model.loss_outputs(batch["branch_inputs"], batch["query_points"])
                field_pred = self.normalizers.fields.inverse_transform_tensor(outputs["fields"]).cpu().numpy()
                field_true = batch["field_targets_raw"].cpu().numpy()
                scalar_pred = self.normalizers.scalars.inverse_transform_tensor(outputs["scalars"]).cpu().numpy()
                scalar_true = batch["scalar_targets_raw"].cpu().numpy()

                surface_outputs = self.model.loss_outputs(batch["branch_inputs"], batch["surface_points"])
                surface_fields = self.normalizers.fields.inverse_transform_tensor(surface_outputs["fields"])
                surface_pressure = surface_fields[..., 2:3]
                cp_reference = batch["cp_reference"].unsqueeze(1)
                cp_pred = pressure_to_cp(surface_pressure, cp_reference=cp_reference).cpu().numpy()
                cp_true = batch["surface_cp"].cpu().numpy()

                field_true_all.append(field_true.reshape(-1, field_true.shape[-1]))
                field_pred_all.append(field_pred.reshape(-1, field_pred.shape[-1]))
                scalar_true_all.append(scalar_true)
                scalar_pred_all.append(scalar_pred)
                cp_true_all.append(cp_true.reshape(-1, cp_true.shape[-1]))
                cp_pred_all.append(cp_pred.reshape(-1, cp_pred.shape[-1]))

                if self.config.save_plots and saved_samples < self.config.num_visualization_samples:
                    for local_index in range(min(batch["query_points_raw"].shape[0], self.config.num_visualization_samples - saved_samples)):
                        sample_points = batch["query_points_raw"][local_index].cpu().numpy()
                        plot_field_scatter(
                            sample_points,
                            field_true[local_index, :, 2],
                            title=f"True pressure sample {saved_samples}",
                            save_path=output_dir / f"field_true_{saved_samples:02d}.png",
                        )
                        plot_paths.append(str(output_dir / f"field_true_{saved_samples:02d}.png"))
                        plot_field_scatter(
                            sample_points,
                            field_pred[local_index, :, 2],
                            title=f"Pred pressure sample {saved_samples}",
                            save_path=output_dir / f"field_pred_{saved_samples:02d}.png",
                        )
                        plot_paths.append(str(output_dir / f"field_pred_{saved_samples:02d}.png"))
                        plot_cp_comparison(
                            batch["surface_points_raw"][local_index].cpu().numpy(),
                            cp_true[local_index, :, 0],
                            cp_pred[local_index, :, 0],
                            save_path=output_dir / f"cp_{saved_samples:02d}.png",
                        )
                        plot_paths.append(str(output_dir / f"cp_{saved_samples:02d}.png"))
                        saved_samples += 1
                        if saved_samples >= self.config.num_visualization_samples:
                            break

        field_true_np = np.concatenate(field_true_all, axis=0)
        field_pred_np = np.concatenate(field_pred_all, axis=0)
        scalar_true_np = np.concatenate(scalar_true_all, axis=0)
        scalar_pred_np = np.concatenate(scalar_pred_all, axis=0)
        cp_true_np = np.concatenate(cp_true_all, axis=0)
        cp_pred_np = np.concatenate(cp_pred_all, axis=0)

        metrics = {
            "field_mse": mse(field_true_np, field_pred_np),
            "field_rmse": rmse(field_true_np, field_pred_np),
            "field_relative_error": relative_error(field_true_np, field_pred_np),
            "cp_mae": mae(cp_true_np, cp_pred_np),
            "cp_relative_error": relative_error(cp_true_np, cp_pred_np),
            "cl_mae": mae(scalar_true_np[:, 0], scalar_pred_np[:, 0]),
            "cl_relative_error": relative_error(scalar_true_np[:, 0], scalar_pred_np[:, 0]),
            "cd_mae": mae(scalar_true_np[:, 1], scalar_pred_np[:, 1]),
            "cd_relative_error": relative_error(scalar_true_np[:, 1], scalar_pred_np[:, 1]),
        }

        if self.config.save_plots:
            plot_scalar_scatter(
                scalar_true_np[:, 0],
                scalar_pred_np[:, 0],
                label="Cl",
                save_path=output_dir / "cl_scatter.png",
            )
            plot_paths.append(str(output_dir / "cl_scatter.png"))
            plot_scalar_scatter(
                scalar_true_np[:, 1],
                scalar_pred_np[:, 1],
                label="Cd",
                save_path=output_dir / "cd_scatter.png",
            )
            plot_paths.append(str(output_dir / "cd_scatter.png"))

        save_json(output_dir / "metrics.json", metrics)
        self._write_report(output_dir=output_dir, metrics=metrics, plot_paths=plot_paths)
        return metrics

    def _write_report(self, output_dir: Path, metrics: dict[str, float], plot_paths: List[str]) -> None:
        report_path = output_dir / "report.md"
        report_json_path = output_dir / "report.json"
        lines = [
            "# Evaluation Report",
            "",
            "## Metrics",
            "",
        ]
        for key, value in metrics.items():
            lines.append(f"- {key}: {value:.6f}")
        lines.extend(
            [
                "",
                "## Notes",
                "",
                "- Dataset: toy or file-based dataset mapped to the CFDOperatorDataset interface.",
                "- Physics residuals use a simplified steady 2D compressible Euler approximation.",
                f"- Evaluated split: `{self.config.split_name}`.",
            ]
        )
        if plot_paths:
            lines.extend(["", "## Figures", ""])
            for path in plot_paths:
                lines.append(f"- {path}")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        save_json(
            report_json_path,
            {
                "config": {
                    "batch_size": self.config.batch_size,
                    "num_visualization_samples": self.config.num_visualization_samples,
                    "save_plots": self.config.save_plots,
                    "split_name": self.config.split_name,
                },
                "metrics": metrics,
                "plots": plot_paths,
            },
        )
