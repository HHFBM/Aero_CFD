"""Checkpoint-backed inference helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from cfd_operator.config.schemas import ProjectConfig
from cfd_operator.data.module import NormalizerBundle
from cfd_operator.geometry import NACA4Airfoil, build_branch_features
from cfd_operator.losses import pressure_to_cp
from cfd_operator.models import create_model
from cfd_operator.postprocess import (
    build_slice_points,
    compute_gradient_indicators,
    compute_surface_cp,
    compute_surface_heat_flux,
    compute_surface_pressure,
    compute_wall_shear,
    estimate_shock_location,
    export_analysis_bundle,
)
from cfd_operator.visualization import (
    plot_field_scatter,
    plot_high_gradient_regions,
    plot_scalar_summary,
    plot_slice_field,
    plot_surface_cp,
    plot_surface_pressure,
)


@dataclass
class Predictor:
    config: ProjectConfig
    normalizers: NormalizerBundle
    model: torch.nn.Module
    device: str = "cpu"

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Union[str, Path], device: str = "cpu") -> "Predictor":
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = ProjectConfig(**checkpoint["config"])
        normalizers = NormalizerBundle.from_dict(checkpoint["normalizers"])
        model = create_model(config.model)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        model.to(device)
        model.eval()
        return cls(config=config, normalizers=normalizers, model=model, device=device)

    def predict(
        self,
        branch_inputs_raw: np.ndarray,
        query_points_raw: np.ndarray,
        flow_conditions: np.ndarray,
        surface_points_raw: Optional[np.ndarray] = None,
        slice_definitions: Optional[List[Dict[str, Any]]] = None,
        cp_reference: Optional[np.ndarray] = None,
        freestream_velocity: Optional[float] = None,
        include_surface: bool = True,
        include_slices: bool = True,
        include_features: bool = True,
        export_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        device = torch.device(self.device)
        query_points_raw = query_points_raw.astype(np.float32)
        branch_inputs = torch.from_numpy(self.normalizers.branch.transform(branch_inputs_raw.astype(np.float32))).to(device).unsqueeze(0)
        query_points = torch.from_numpy(self.normalizers.coordinates.transform(query_points_raw)).to(device).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model.loss_outputs(branch_inputs, query_points)
            predicted_fields_internal = self.normalizers.fields.inverse_transform_tensor(outputs["fields"]).cpu().numpy()[0]
            predicted_scalars = self.normalizers.scalars.inverse_transform_tensor(outputs["scalars"]).cpu().numpy()[0]

        cp_reference_used = self._resolve_cp_reference(
            flow_conditions=flow_conditions,
            cp_reference=cp_reference,
            freestream_velocity=freestream_velocity,
        )
        predicted_fields = self._export_field_values(
            field_values=predicted_fields_internal.astype(np.float32),
            cp_reference=cp_reference_used,
        )

        scalar_names = list(self._scalar_names())
        predicted_scalar_map = {
            scalar_names[index]: float(predicted_scalars[index])
            for index in range(min(len(predicted_scalars), len(scalar_names)))
        }

        result: Dict[str, Any] = {
            "query_points": query_points_raw,
            "predicted_fields": predicted_fields.astype(np.float32),
            "predicted_scalars": predicted_scalar_map,
            "metadata": {
                "model_name": self.config.model.name,
                "field_names": list(self.config.data.field_names),
                "scalar_names": scalar_names[: len(predicted_scalars)],
                "pressure_target_mode": self.config.data.pressure_target_mode,
                "flow_conditions": {
                    "mach": float(flow_conditions[0]),
                    "aoa": float(flow_conditions[1]),
                },
                "availability": {
                    "field_outputs": "predicted",
                    "scalar_outputs": "cl/cd predicted; optional scalar components reserved unless the checkpoint was trained with extra scalar dimensions",
                    "surface_outputs": "predicted/derived when requested",
                    "slice_outputs": "predicted when slice definitions are supplied",
                    "feature_outputs": "predicted_or_derived when requested",
                },
            },
        }
        if flow_conditions.shape[0] > 2:
            result["metadata"]["flow_conditions"]["reynolds"] = float(flow_conditions[2])

        if include_surface and surface_points_raw is not None:
            result["surface_predictions"] = self._predict_surface(
                branch_inputs=branch_inputs,
                surface_points_raw=surface_points_raw.astype(np.float32),
                flow_conditions=flow_conditions.astype(np.float32),
                cp_reference=cp_reference_used,
                freestream_velocity=freestream_velocity,
            )

        if include_slices and slice_definitions:
            result["slice_predictions"] = self._predict_slices(
                branch_inputs=branch_inputs,
                slice_definitions=slice_definitions,
                cp_reference=cp_reference_used,
            )

        if include_features:
            result["feature_predictions"] = self._predict_features(
                query_points_raw=query_points_raw,
                outputs=outputs,
                predicted_fields=predicted_fields.astype(np.float32),
            )

        if export_dir is not None:
            self.export_analysis_bundle(result, output_dir=export_dir)

        return result

    def predict_from_geometry(
        self,
        geometry_params: np.ndarray,
        mach: float,
        aoa_deg: float,
        query_points: np.ndarray,
        surface_points: Optional[np.ndarray] = None,
        slice_definitions: Optional[List[Dict[str, Any]]] = None,
        reynolds: Optional[float] = None,
        cp_reference: Optional[np.ndarray] = None,
        freestream_velocity: Optional[float] = None,
        include_surface: bool = True,
        include_slices: bool = True,
        include_features: bool = True,
        export_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        if geometry_params.shape[0] < 3:
            raise ValueError("geometry_params must contain at least [max_camber, camber_position, thickness]")
        chord = float(geometry_params[3]) if geometry_params.shape[0] > 3 else 1.0
        airfoil = NACA4Airfoil(
            max_camber=float(geometry_params[0]),
            camber_position=float(geometry_params[1]),
            thickness=float(geometry_params[2]),
            chord=chord,
        )
        branch_inputs = self._build_inference_branch_inputs(
            airfoil=airfoil,
            mach=mach,
            aoa_deg=aoa_deg,
            reynolds=reynolds,
        )
        if surface_points is None:
            surface_points = airfoil.surface_points(self.config.data.num_surface_points)
        return self.predict(
            branch_inputs_raw=branch_inputs.astype(np.float32),
            query_points_raw=query_points.astype(np.float32),
            flow_conditions=(
                np.asarray([mach, aoa_deg, reynolds], dtype=np.float32)
                if reynolds is not None
                else np.asarray([mach, aoa_deg], dtype=np.float32)
            ),
            surface_points_raw=surface_points.astype(np.float32),
            slice_definitions=slice_definitions,
            cp_reference=cp_reference,
            freestream_velocity=freestream_velocity,
            include_surface=include_surface,
            include_slices=include_slices,
            include_features=include_features,
            export_dir=export_dir,
        )

    def _build_inference_branch_inputs(
        self,
        airfoil: NACA4Airfoil,
        mach: float,
        aoa_deg: float,
        reynolds: Optional[float] = None,
    ) -> np.ndarray:
        """Build branch features compatible with the training-time branch schema.

        For the synthetic/NACA path we use the configured geometry encoder. For the raw
        AirfRANS path, historical datasets store a flattened surface signature followed
        by flow conditions. This fallback reconstructs that format from the expected
        branch dimension in the fitted normalizer.
        """

        expected_dim = int(self.normalizers.branch.mean.shape[0])
        default_branch = build_branch_features(
            airfoil,
            mach=mach,
            aoa_deg=aoa_deg,
            reynolds=reynolds if self.config.data.include_reynolds else None,
            mode=self.config.data.branch_feature_mode,
        )
        if default_branch.shape[0] == expected_dim:
            return default_branch.astype(np.float32)

        flow_dim = 3 if self.config.data.include_reynolds else 2
        signature_dim = expected_dim - flow_dim
        if signature_dim > 0 and signature_dim % 2 == 0:
            num_surface_points = signature_dim // 2
            surface_points = airfoil.surface_points(num_surface_points)
            if surface_points.shape[0] > num_surface_points:
                indices = np.linspace(0, surface_points.shape[0] - 1, num_surface_points, dtype=np.int64)
                surface_points = surface_points[indices]
            elif surface_points.shape[0] < num_surface_points:
                pad_count = num_surface_points - surface_points.shape[0]
                surface_points = np.concatenate([surface_points, np.repeat(surface_points[-1:], pad_count, axis=0)], axis=0)
            surface_signature = surface_points.reshape(-1).astype(np.float32)
            flow = [mach, aoa_deg]
            if self.config.data.include_reynolds and reynolds is not None:
                flow.append(reynolds)
            branch_inputs = np.concatenate([surface_signature, np.asarray(flow, dtype=np.float32)], axis=0)
            if branch_inputs.shape[0] == expected_dim:
                return branch_inputs.astype(np.float32)

        return default_branch.astype(np.float32)

    def _predict_surface(
        self,
        branch_inputs: torch.Tensor,
        surface_points_raw: np.ndarray,
        flow_conditions: np.ndarray,
        cp_reference: Optional[np.ndarray] = None,
        freestream_velocity: Optional[float] = None,
    ) -> Dict[str, Any]:
        device = torch.device(self.device)
        surface_points = torch.from_numpy(self.normalizers.coordinates.transform(surface_points_raw.astype(np.float32))).to(device).unsqueeze(0)
        with torch.no_grad():
            surface_outputs = self.model.loss_outputs(branch_inputs, surface_points)
            surface_fields = self.normalizers.fields.inverse_transform_tensor(surface_outputs["fields"]).cpu().numpy()[0]

        cp_reference_used = self._resolve_cp_reference(flow_conditions=flow_conditions, cp_reference=cp_reference, freestream_velocity=freestream_velocity)
        cp_surface = self._surface_cp_output(surface_fields=surface_fields, cp_reference=cp_reference_used)
        pressure_surface = self._surface_pressure_output(surface_fields=surface_fields, cp_reference=cp_reference_used)
        heat_flux_surface = compute_surface_heat_flux(surface_points_raw, pressure_surface)
        wall_shear_surface = compute_wall_shear(surface_points_raw, surface_fields)
        return {
            "surface_points": surface_points_raw.astype(np.float32),
            "cp_surface": cp_surface.astype(np.float32),
            "pressure_surface": pressure_surface.astype(np.float32),
            "velocity_surface": surface_fields[..., :2].astype(np.float32),
            "nut_surface": surface_fields[..., 3:4].astype(np.float32),
            "heat_flux_surface": heat_flux_surface.astype(np.float32),
            "wall_shear_surface": wall_shear_surface.astype(np.float32),
            "cp_reference": cp_reference_used.astype(np.float32),
            "availability": {
                "cp_surface": "predicted/derived",
                "pressure_surface": "predicted",
                "velocity_surface": "predicted",
                "nut_surface": "predicted for AirfRANS-style checkpoints; placeholder semantics for datasets that do not supervise the 4th field as nut",
                "heat_flux_surface": "approximate_postprocess",
                "wall_shear_surface": "approximate_postprocess",
            },
        }

    def _predict_slices(
        self,
        branch_inputs: torch.Tensor,
        slice_definitions: List[Dict[str, Any]],
        cp_reference: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        device = torch.device(self.device)
        predictions: list[dict[str, Any]] = []
        for slice_definition in slice_definitions:
            slice_points_raw = build_slice_points(slice_definition)
            slice_points = torch.from_numpy(self.normalizers.coordinates.transform(slice_points_raw.astype(np.float32))).to(device).unsqueeze(0)
            with torch.no_grad():
                slice_outputs = self.model.loss_outputs(branch_inputs, slice_points)
                slice_fields_internal = self.normalizers.fields.inverse_transform_tensor(slice_outputs["fields"]).cpu().numpy()[0]
            slice_fields = self._export_field_values(field_values=slice_fields_internal.astype(np.float32), cp_reference=cp_reference)
            predictions.append(
                {
                    "slice_definition": slice_definition,
                    "slice_points": slice_points_raw.astype(np.float32),
                    "slice_fields": slice_fields.astype(np.float32),
                }
            )
        return predictions

    def _predict_features(
        self,
        query_points_raw: np.ndarray,
        outputs: dict[str, torch.Tensor],
        predicted_fields: np.ndarray,
    ) -> Dict[str, Any]:
        if "features" in outputs:
            feature_logits = outputs["features"].detach().cpu().numpy()[0]
            pressure_gradient_indicator = 1.0 / (1.0 + np.exp(-feature_logits[:, 0:1]))
            high_gradient_mask = (
                1.0 / (1.0 + np.exp(-feature_logits[:, 1:2]))
                if feature_logits.shape[-1] > 1
                else pressure_gradient_indicator
            )
            gradient_magnitude = high_gradient_mask.astype(np.float32)
            region_summary = {
                "mean_gradient": float(np.mean(gradient_magnitude)),
                "max_gradient": float(np.max(gradient_magnitude, initial=0.0)),
                "high_gradient_fraction": float(np.mean(high_gradient_mask > 0.5)),
                "pressure_gradient_fraction": float(np.mean(pressure_gradient_indicator > 0.5)),
            }
        else:
            derived = compute_gradient_indicators(query_points_raw, predicted_fields)
            pressure_gradient_indicator = derived["pressure_gradient_indicator"]
            high_gradient_mask = derived["high_gradient_mask"]
            gradient_magnitude = derived["gradient_magnitude"]
            region_summary = derived["high_gradient_region_summary"]
        shock_summary = estimate_shock_location(query_points_raw, pressure_gradient_indicator, gradient_magnitude)
        return {
            "pressure_gradient_indicator": pressure_gradient_indicator.astype(np.float32),
            "high_gradient_mask": high_gradient_mask.astype(np.float32),
            "gradient_magnitude": gradient_magnitude.astype(np.float32),
            "high_gradient_region_summary": region_summary,
            "shock_indicator": pressure_gradient_indicator.astype(np.float32),
            "shock_location_summary": shock_summary,
        }

    def export_analysis_bundle(self, result: Dict[str, Any], output_dir: Union[str, Path]) -> Path:
        output_dir = export_analysis_bundle(output_dir=output_dir, payload=result)
        field_names = list(result.get("metadata", {}).get("field_names", self.config.data.field_names))

        plot_field_scatter(
            points=np.asarray(result["query_points"], dtype=np.float32),
            values=np.asarray(result["predicted_fields"], dtype=np.float32)[:, 2],
            title="Predicted pressure field",
            save_path=Path(output_dir) / "predicted_pressure_field.png",
        )
        plot_scalar_summary(result["predicted_scalars"], save_path=Path(output_dir) / "scalar_summary.png")

        if "surface_predictions" in result:
            surface = result["surface_predictions"]
            plot_surface_pressure(
                surface_points=np.asarray(surface["surface_points"], dtype=np.float32),
                pressure_pred=np.asarray(surface["pressure_surface"], dtype=np.float32).reshape(-1),
                save_path=Path(output_dir) / "surface_pressure.png",
            )
            plot_surface_cp(
                surface_points=np.asarray(surface["surface_points"], dtype=np.float32),
                cp_pred=np.asarray(surface["cp_surface"], dtype=np.float32).reshape(-1),
                save_path=Path(output_dir) / "surface_cp.png",
            )

        if result.get("slice_predictions"):
            first_slice = result["slice_predictions"][0]
            for field_index, field_name in enumerate(field_names[: np.asarray(first_slice["slice_fields"]).shape[-1]]):
                plot_slice_field(
                    slice_points=np.asarray(first_slice["slice_points"], dtype=np.float32),
                    pred_values=np.asarray(first_slice["slice_fields"], dtype=np.float32)[:, field_index],
                    variable_name=str(field_name),
                    save_path=Path(output_dir) / f"slice_{field_name}.png",
                )

        if "feature_predictions" in result:
            plot_high_gradient_regions(
                points=np.asarray(result["query_points"], dtype=np.float32),
                indicator=np.asarray(result["feature_predictions"]["high_gradient_mask"], dtype=np.float32).reshape(-1),
                save_path=Path(output_dir) / "high_gradient_regions.png",
            )
        return Path(output_dir)

    def _resolve_cp_reference(
        self,
        flow_conditions: np.ndarray,
        cp_reference: Optional[np.ndarray] = None,
        freestream_velocity: Optional[float] = None,
    ) -> np.ndarray:
        if cp_reference is not None:
            return np.asarray(cp_reference, dtype=np.float32).reshape(2)
        if freestream_velocity is not None:
            return np.asarray([0.0, 0.5 * float(freestream_velocity) ** 2], dtype=np.float32)
        mach = float(flow_conditions[0])
        return np.asarray([1.0, max(0.5 * 1.4 * mach**2, 1.0e-4)], dtype=np.float32)

    def _scalar_names(self) -> tuple[str, ...]:
        return ("cl", "cd", "cdp", "cdv", "clp", "clv", "cm")

    def _surface_cp_output(self, surface_fields: np.ndarray, cp_reference: np.ndarray) -> np.ndarray:
        if self.config.data.pressure_target_mode == "cp_like":
            return np.asarray(surface_fields[..., 2:3], dtype=np.float32)
        pressure_surface = compute_surface_pressure(surface_fields=surface_fields)
        return compute_surface_cp(surface_pressure=pressure_surface, cp_reference=cp_reference)

    def _surface_pressure_output(self, surface_fields: np.ndarray, cp_reference: np.ndarray) -> np.ndarray:
        if self.config.data.pressure_target_mode != "cp_like":
            return compute_surface_pressure(surface_fields=surface_fields)
        cp_surface = np.asarray(surface_fields[..., 2:3], dtype=np.float32)
        return compute_surface_pressure(surface_cp=cp_surface, cp_reference=cp_reference)

    def _export_field_values(self, field_values: np.ndarray, cp_reference: Optional[np.ndarray]) -> np.ndarray:
        export_fields = np.asarray(field_values, dtype=np.float32).copy()
        if self.config.data.pressure_target_mode == "cp_like" and cp_reference is not None:
            export_fields[:, 2:3] = compute_surface_pressure(
                surface_cp=export_fields[:, 2:3],
                cp_reference=cp_reference,
            )
        return export_fields
