"""Model evaluation and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import pandas as pd
import torch

from cfd_operator.config.schemas import EvalConfig
from cfd_operator.data.module import CFDDataModule, NormalizerBundle
from cfd_operator.evaluators.metrics import mae
from cfd_operator.evaluators.registry import build_default_metric_registry
from cfd_operator.models.base import BaseOperatorModel
from cfd_operator.output_semantics import build_output_semantics, flatten_metric_groups
from cfd_operator.postprocess import (
    compute_gradient_indicators,
    estimate_shock_location,
    export_analysis_bundle,
    resolve_pressure_channel,
    resolve_surface_cp,
)
from cfd_operator.utils.io import save_json
from cfd_operator.visualization import (
    plot_field_scatter,
    plot_high_gradient_regions,
    plot_loss_curves,
    plot_scalar_scatter,
    plot_scalar_summary,
    plot_slice_field,
    plot_surface_cp,
    plot_surface_pressure,
)
from cfd_operator.tasks.capabilities import TaskRequest, resolve_effective_tasks


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

        plot_paths: list[str] = []
        exported_files: list[str] = []
        metric_groups: dict[str, dict[str, float]] = {}
        metric_registry = build_default_metric_registry()
        capability = self.data_module.dataset_capability
        effective_tasks = (
            resolve_effective_tasks(capability, TaskRequest())
            if capability is not None
            else TaskRequest()
        )

        field_true_all: list[np.ndarray] = []
        field_pred_all: list[np.ndarray] = []
        scalar_true_all: list[np.ndarray] = []
        scalar_pred_all: list[np.ndarray] = []
        cp_true_all: list[np.ndarray] = []
        cp_pred_all: list[np.ndarray] = []
        pressure_true_all: list[np.ndarray] = []
        pressure_pred_all: list[np.ndarray] = []
        slice_true_all: list[np.ndarray] = []
        slice_pred_all: list[np.ndarray] = []
        shock_true_all: list[np.ndarray] = []
        shock_pred_all: list[np.ndarray] = []
        high_true_all: list[np.ndarray] = []
        high_pred_all: list[np.ndarray] = []
        pressure_gradient_true_all: list[np.ndarray] = []
        pressure_gradient_pred_all: list[np.ndarray] = []
        shock_location_true_all: list[np.ndarray] = []
        shock_location_pred_all: list[np.ndarray] = []

        saved_samples = 0

        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
                outputs = self.model.loss_outputs(batch["branch_inputs"], batch["query_points"])
                field_pred = self.normalizers.fields.inverse_transform_tensor(outputs["fields"]).cpu().numpy()
                field_true = batch["field_targets_raw"].cpu().numpy()
                scalar_pred = self.normalizers.scalars.inverse_transform_tensor(outputs["scalars"]).cpu().numpy()
                scalar_true = batch["scalar_targets_raw"].cpu().numpy()

                surface_fields = None
                surface_pressure_pred = None
                surface_pressure_true = None
                cp_pred = None
                cp_true = None
                if effective_tasks.surface and batch["surface_points"].shape[1] > 0:
                    surface_outputs = self.model.loss_outputs(batch["branch_inputs"], batch["surface_points"])
                    surface_fields = self.normalizers.fields.inverse_transform_tensor(surface_outputs["fields"])
                    cp_reference_np = batch["cp_reference"].cpu().numpy()
                    cp_pred = np.stack(
                        [
                            resolve_surface_cp(
                                pressure_channel_values=surface_fields[batch_index, :, 2:3].cpu().numpy(),
                                pressure_target_mode=self.data_module.config.pressure_target_mode,
                                cp_reference=cp_reference_np[batch_index],
                            )
                            for batch_index in range(surface_fields.shape[0])
                        ]
                    )
                    surface_pressure_pred = np.stack(
                        [
                            resolve_pressure_channel(
                                pressure_channel_values=surface_fields[batch_index, :, 2:3].cpu().numpy(),
                                pressure_target_mode=self.data_module.config.pressure_target_mode,
                                cp_reference=cp_reference_np[batch_index],
                            )
                            for batch_index in range(surface_fields.shape[0])
                        ]
                    )
                    surface_pressure_true = batch["surface_pressure"].cpu().numpy()
                    cp_true = batch["surface_cp"].cpu().numpy()

                slice_pred = None
                slice_true = None
                if effective_tasks.slice and batch["slice_points"].shape[1] > 0:
                    slice_outputs = self.model.loss_outputs(batch["branch_inputs"], batch["slice_points"])
                    slice_pred = self.normalizers.fields.inverse_transform_tensor(slice_outputs["fields"]).cpu().numpy()
                    slice_true = batch["slice_fields_raw"].cpu().numpy()

                if effective_tasks.feature and "features" in outputs:
                    feature_logits = outputs["features"].cpu().numpy()
                    pressure_gradient_pred = 1.0 / (1.0 + np.exp(-feature_logits[..., 0:1]))
                    high_pred = (
                        1.0 / (1.0 + np.exp(-feature_logits[..., 1:2]))
                        if feature_logits.shape[-1] > 1
                        else pressure_gradient_pred
                    )
                    gradient_pred = high_pred
                else:
                    pressure_gradient_pred_list = []
                    high_pred_list = []
                    gradient_pred_list = []
                    for batch_index in range(field_pred.shape[0]):
                        derived = compute_gradient_indicators(batch["query_points_raw"][batch_index].cpu().numpy(), field_pred[batch_index])
                        pressure_gradient_pred_list.append(derived["pressure_gradient_indicator"])
                        high_pred_list.append(derived["high_gradient_mask"])
                        gradient_pred_list.append(derived["gradient_magnitude"])
                    pressure_gradient_pred = np.stack(pressure_gradient_pred_list)
                    high_pred = np.stack(high_pred_list)
                    gradient_pred = np.stack(gradient_pred_list)

                for batch_index in range(field_pred.shape[0]):
                    summary = estimate_shock_location(
                        points=batch["query_points_raw"][batch_index].cpu().numpy(),
                        shock_indicator=pressure_gradient_pred[batch_index],
                        gradient_magnitude=gradient_pred[batch_index],
                    )
                    shock_location_true_all.append(batch["shock_location"][batch_index].cpu().numpy())
                    if summary["centroid"] is None:
                        shock_location_pred_all.append(np.asarray([np.nan, np.nan], dtype=np.float32))
                    else:
                        shock_location_pred_all.append(np.asarray(summary["centroid"], dtype=np.float32))

                field_true_all.append(field_true.reshape(-1, field_true.shape[-1]))
                field_pred_all.append(field_pred.reshape(-1, field_pred.shape[-1]))
                scalar_true_all.append(scalar_true)
                scalar_pred_all.append(scalar_pred)
                if cp_true is not None and cp_pred is not None and surface_pressure_true is not None and surface_pressure_pred is not None:
                    cp_true_all.append(cp_true.reshape(-1, cp_true.shape[-1]))
                    cp_pred_all.append(cp_pred.reshape(-1, cp_pred.shape[-1]))
                    pressure_true_all.append(surface_pressure_true.reshape(-1, 1))
                    pressure_pred_all.append(surface_pressure_pred.reshape(-1, 1))
                if slice_true is not None and slice_pred is not None:
                    slice_true_all.append(slice_true.reshape(-1, slice_true.shape[-1]))
                    slice_pred_all.append(slice_pred.reshape(-1, slice_pred.shape[-1]))
                if effective_tasks.feature:
                    pressure_gradient_true_all.append(batch["pressure_gradient_indicator"].cpu().numpy().reshape(-1, 1))
                    pressure_gradient_pred_all.append(pressure_gradient_pred.reshape(-1, 1))
                    shock_true_all.append(batch["shock_indicator"].cpu().numpy().reshape(-1, 1))
                    shock_pred_all.append(pressure_gradient_pred.reshape(-1, 1))
                    high_true_all.append(batch["high_gradient_mask"].cpu().numpy().reshape(-1, 1))
                    high_pred_all.append(high_pred.reshape(-1, 1))

                if (self.config.save_plots or self.config.export_analysis) and saved_samples < self.config.num_visualization_samples:
                    remaining = self.config.num_visualization_samples - saved_samples
                    for local_index in range(min(batch["query_points_raw"].shape[0], remaining)):
                        sample_dir = output_dir / f"sample_{saved_samples:02d}"
                        sample_dir.mkdir(parents=True, exist_ok=True)
                        field_names = list(self.data_module.config.field_names)

                        if self.config.export_analysis:
                            semantics = build_output_semantics(
                                field_names=field_names,
                                pressure_target_mode=self.data_module.config.pressure_target_mode,
                                scalar_names=("cl", "cd"),
                            )
                            analysis_payload = {
                                "query_points": batch["query_points_raw"][local_index].cpu().numpy(),
                                "predicted_fields": field_pred[local_index],
                                "predicted_scalars": {
                                    "cl": float(scalar_pred[local_index, 0]),
                                    "cd": float(scalar_pred[local_index, 1]),
                                },
                                "metadata": {
                                    "field_names": field_names,
                                    "scalar_names": ["cl", "cd"],
                                    "pressure_target_mode": self.data_module.config.pressure_target_mode,
                                    "pressure_semantics": semantics["pressure"],
                                    "task_semantics": semantics["tasks"],
                                    "dataset_capability": (
                                        self.data_module.dataset_capability.as_dict()
                                        if self.data_module.dataset_capability is not None
                                        else None
                                    ),
                                    "source": "evaluation_export",
                                },
                            }
                            if cp_pred is not None and surface_pressure_pred is not None and surface_fields is not None:
                                analysis_payload["surface_predictions"] = {
                                    "surface_points": batch["surface_points_raw"][local_index].cpu().numpy(),
                                    "cp_surface": cp_pred[local_index],
                                    "pressure_surface": surface_pressure_pred[local_index],
                                    "velocity_surface": surface_fields[local_index, :, :2].cpu().numpy(),
                                    "nut_surface": surface_fields[local_index, :, 3:4].cpu().numpy(),
                                }
                            else:
                                analysis_payload.pop("surface_predictions", None)
                            analysis_payload["slice_predictions"] = (
                                [
                                    {
                                        "slice_definition": {"type": "dataset_slice"},
                                        "slice_points": batch["slice_points_raw"][local_index].cpu().numpy(),
                                        "slice_fields": slice_pred[local_index],
                                    }
                                ]
                                if slice_pred is not None
                                else []
                            )
                            if effective_tasks.feature:
                                analysis_payload["feature_predictions"] = {
                                    "pressure_gradient_indicator": pressure_gradient_pred[local_index],
                                    "high_gradient_mask": high_pred[local_index],
                                    "gradient_magnitude": gradient_pred[local_index],
                                    "high_gradient_region_summary": {
                                        "mean_gradient": float(np.mean(gradient_pred[local_index])),
                                        "max_gradient": float(np.max(gradient_pred[local_index])),
                                        "high_gradient_fraction": float(np.mean(high_pred[local_index] >= 0.5)),
                                        "pressure_gradient_fraction": float(np.mean(pressure_gradient_pred[local_index] >= 0.5)),
                                    },
                                }
                            export_analysis_bundle(sample_dir, analysis_payload)
                            exported_files.extend(
                                [
                                    str(sample_dir / "predictions.json"),
                                    str(sample_dir / "scalar_summary.json"),
                                    str(sample_dir / "task_semantics.json"),
                                    str(sample_dir / "surface_values.csv"),
                                    str(sample_dir / "slice_values.csv"),
                                    str(sample_dir / "feature_summary.json"),
                                ]
                            )

                        if self.config.save_plots:
                            sample_points = batch["query_points_raw"][local_index].cpu().numpy()
                            plot_field_scatter(
                                sample_points,
                                field_true[local_index, :, 2],
                                title=f"True raw pressure sample {saved_samples}",
                                save_path=sample_dir / "field_true_pressure.png",
                            )
                            plot_paths.append(str(sample_dir / "field_true_pressure.png"))
                            plot_field_scatter(
                                sample_points,
                                field_pred[local_index, :, 2],
                                title=f"Pred raw pressure sample {saved_samples}",
                                save_path=sample_dir / "field_pred_pressure.png",
                            )
                            plot_paths.append(str(sample_dir / "field_pred_pressure.png"))
                            if cp_pred is not None and cp_true is not None and surface_pressure_pred is not None and surface_pressure_true is not None:
                                plot_surface_cp(
                                    batch["surface_points_raw"][local_index].cpu().numpy(),
                                    cp_pred=cp_pred[local_index, :, 0],
                                    cp_true=cp_true[local_index, :, 0],
                                    save_path=sample_dir / "surface_cp.png",
                                )
                                plot_paths.append(str(sample_dir / "surface_cp.png"))
                                plot_surface_pressure(
                                    batch["surface_points_raw"][local_index].cpu().numpy(),
                                    pressure_pred=surface_pressure_pred[local_index, :, 0],
                                    pressure_true=surface_pressure_true[local_index, :, 0],
                                    save_path=sample_dir / "surface_pressure.png",
                                )
                                plot_paths.append(str(sample_dir / "surface_pressure.png"))
                            if slice_pred is not None and slice_true is not None:
                                for field_index, field_name in enumerate(field_names):
                                    plot_slice_field(
                                        batch["slice_points_raw"][local_index].cpu().numpy(),
                                        pred_values=slice_pred[local_index, :, field_index],
                                        true_values=slice_true[local_index, :, field_index],
                                        variable_name=str(field_name),
                                        save_path=sample_dir / f"slice_{field_name}.png",
                                    )
                                    plot_paths.append(str(sample_dir / f"slice_{field_name}.png"))
                            if effective_tasks.feature:
                                plot_high_gradient_regions(
                                    points=sample_points,
                                    indicator=high_pred[local_index, :, 0],
                                    save_path=sample_dir / "high_gradient_regions.png",
                                )
                                plot_paths.append(str(sample_dir / "high_gradient_regions.png"))
                            plot_scalar_summary(
                                scalar_values={"cl": float(scalar_pred[local_index, 0]), "cd": float(scalar_pred[local_index, 1])},
                                true_values={"cl": float(scalar_true[local_index, 0]), "cd": float(scalar_true[local_index, 1])},
                                save_path=sample_dir / "scalar_summary.png",
                            )
                            plot_paths.append(str(sample_dir / "scalar_summary.png"))

                        saved_samples += 1
                        if saved_samples >= self.config.num_visualization_samples:
                            break

        field_true_np = np.concatenate(field_true_all, axis=0)
        field_pred_np = np.concatenate(field_pred_all, axis=0)
        scalar_true_np = np.concatenate(scalar_true_all, axis=0)
        scalar_pred_np = np.concatenate(scalar_pred_all, axis=0)
        cp_true_np = np.concatenate(cp_true_all, axis=0) if cp_true_all else None
        cp_pred_np = np.concatenate(cp_pred_all, axis=0) if cp_pred_all else None
        pressure_true_np = np.concatenate(pressure_true_all, axis=0) if pressure_true_all else None
        pressure_pred_np = np.concatenate(pressure_pred_all, axis=0) if pressure_pred_all else None
        slice_true_np = np.concatenate(slice_true_all, axis=0) if slice_true_all else None
        slice_pred_np = np.concatenate(slice_pred_all, axis=0) if slice_pred_all else None
        pressure_gradient_true_np = np.concatenate(pressure_gradient_true_all, axis=0) if pressure_gradient_true_all else None
        pressure_gradient_pred_np = np.concatenate(pressure_gradient_pred_all, axis=0) if pressure_gradient_pred_all else None
        shock_true_np = np.concatenate(shock_true_all, axis=0) if shock_true_all else None
        shock_pred_np = np.concatenate(shock_pred_all, axis=0) if shock_pred_all else None
        high_true_np = np.concatenate(high_true_all, axis=0) if high_true_all else None
        high_pred_np = np.concatenate(high_pred_all, axis=0) if high_pred_all else None
        shock_location_true_np = np.stack(shock_location_true_all) if shock_location_true_all else None
        shock_location_pred_np = np.stack(shock_location_pred_all) if shock_location_pred_all else None
        shock_location_valid = (
            np.isfinite(shock_location_true_np).all(axis=1) & np.isfinite(shock_location_pred_np).all(axis=1)
            if shock_location_true_np is not None and shock_location_pred_np is not None
            else np.zeros((0,), dtype=bool)
        )

        for group_name, spec in metric_registry.items():
            if group_name == "field_metrics" and effective_tasks.field:
                metric_groups[group_name] = spec.computer(
                    field_true_np,
                    field_pred_np,
                    aux_name=str(self.data_module.config.field_names[3]),
                )
            elif group_name == "scalar_metrics" and effective_tasks.scalar:
                metric_groups[group_name] = spec.computer(
                    scalar_true_np,
                    scalar_pred_np,
                    scalar_names=("cl", "cd"),
                )
            elif (
                group_name == "surface_metrics"
                and effective_tasks.surface
                and cp_true_np is not None
                and cp_pred_np is not None
                and pressure_true_np is not None
                and pressure_pred_np is not None
            ):
                metric_groups[group_name] = spec.computer(
                    cp_true_np,
                    cp_pred_np,
                    pressure_true_np,
                    pressure_pred_np,
                )
            elif group_name == "slice_metrics" and effective_tasks.slice and slice_true_np is not None and slice_pred_np is not None:
                metric_groups[group_name] = spec.computer(slice_true_np, slice_pred_np)
            elif (
                group_name == "feature_metrics"
                and effective_tasks.feature
                and pressure_gradient_true_np is not None
                and pressure_gradient_pred_np is not None
                and high_true_np is not None
                and high_pred_np is not None
            ):
                metric_groups[group_name] = spec.computer(
                    pressure_gradient_true_np,
                    pressure_gradient_pred_np,
                    high_true_np,
                    high_pred_np,
                )
        metrics = flatten_metric_groups(metric_groups)
        if np.any(shock_location_valid):
            metrics["shock_location_mae"] = mae(
                shock_location_true_np[shock_location_valid],
                shock_location_pred_np[shock_location_valid],
            )
            metric_groups.setdefault("feature_metrics", {})["shock_location_mae"] = metrics["shock_location_mae"]

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
            history_csv = output_dir.parent.parent / "reports" / "history.csv"
            if history_csv.exists():
                plot_loss_curves(history_csv, output_dir / "loss_curve.png")
                plot_paths.append(str(output_dir / "loss_curve.png"))

        save_json(output_dir / "metrics.json", metrics)
        self._write_report(
            output_dir=output_dir,
            metrics=metrics,
            metric_groups=metric_groups,
            plot_paths=plot_paths,
            exported_files=exported_files,
        )
        return metrics

    def _write_report(
        self,
        output_dir: Path,
        metrics: dict[str, float],
        metric_groups: dict[str, dict[str, float]],
        plot_paths: list[str],
        exported_files: list[str],
    ) -> None:
        report_path = output_dir / "report.md"
        report_json_path = output_dir / "report.json"
        semantics = build_output_semantics(
            field_names=list(self.data_module.config.field_names),
            pressure_target_mode=self.data_module.config.pressure_target_mode,
            scalar_names=("cl", "cd"),
        )
        lines = [
            "# Evaluation Report",
            "",
            "## Metrics",
            "",
        ]
        for group_name, group_metrics in metric_groups.items():
            lines.append(f"### {group_name}")
            lines.append("")
            for key, value in group_metrics.items():
                lines.append(f"- {key}: {value:.6f}")
            lines.append("")
        lines.extend(
            [
                "## Notes",
                "",
                "- Field outputs and scalar outputs are direct model predictions.",
                f"- Pressure channel semantics: {semantics['pressure']['training_pressure_description']}",
                "- Exported pressure metrics always use raw pressure after any needed cp_like reconstruction.",
                "- Surface Cp metrics always use Cp reconstructed from the surface pressure channel and Cp reference.",
                "- Heat flux and wall shear remain placeholder/experimental proxies and are not benchmark metrics.",
                "- Feature metrics mix explicit feature-head outputs and derived gradient indicators, depending on checkpoint capability.",
                f"- Evaluated split: `{self.config.split_name}`.",
            ]
        )
        if exported_files:
            lines.extend(["", "## Exported Analysis Files", ""])
            for path in exported_files:
                lines.append(f"- {path}")
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
                "metric_groups": metric_groups,
                "dataset_capability": (
                    self.data_module.dataset_capability.as_dict()
                    if self.data_module.dataset_capability is not None
                    else None
                ),
                "task_semantics": semantics["tasks"],
                "pressure_semantics": semantics["pressure"],
                "plots": plot_paths,
                "exports": exported_files,
            },
        )
