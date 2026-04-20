"""Checkpoint-backed inference helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings

import numpy as np
import torch

from cfd_operator.config.schemas import ProjectConfig
from cfd_operator.data.module import NormalizerBundle
from cfd_operator.geometry import (
    BranchInputAdapter,
    BranchInputContract,
    GeometryInputError,
    NACA4Airfoil,
    resolve_geometry_input,
)
from cfd_operator.geometry.preprocess import CanonicalGeometry2D, maybe_warn_geometry_adapter
from cfd_operator.geometry.semantics import build_inference_geometry_semantics
from cfd_operator.models import create_model
from cfd_operator.models.geometry_backbone import GeometryBackboneContract, build_geometry_backbone_contract
from cfd_operator.output_semantics import build_output_semantics, default_scalar_names
from cfd_operator.postprocess import (
    build_slice_points,
    compute_gradient_indicators,
    compute_surface_heat_flux,
    compute_wall_shear,
    estimate_shock_location,
    export_analysis_bundle,
    resolve_pressure_channel,
    resolve_surface_cp,
)
from cfd_operator.visualization import (
    plot_field_scatter,
    plot_high_gradient_regions,
    plot_scalar_summary,
    plot_slice_field,
    plot_surface_cp,
    plot_surface_pressure,
)
from cfd_operator.tasks.capabilities import DatasetCapability


@dataclass
class Predictor:
    config: ProjectConfig
    normalizers: NormalizerBundle
    model: torch.nn.Module
    dataset_capability: DatasetCapability | None = None
    branch_contract: BranchInputContract | None = None
    geometry_backbone_contract: GeometryBackboneContract | None = None
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
        dataset_capability = None
        if checkpoint.get("dataset_capability") is not None:
            dataset_capability = DatasetCapability.from_dict(checkpoint["dataset_capability"])
        branch_contract = None
        if checkpoint.get("branch_contract") is not None:
            branch_contract = BranchInputContract.from_dict(checkpoint["branch_contract"])
        geometry_backbone_contract = None
        if checkpoint.get("geometry_backbone_contract") is not None:
            geometry_backbone_contract = GeometryBackboneContract.from_dict(checkpoint["geometry_backbone_contract"])
        else:
            geometry_backbone_contract = build_geometry_backbone_contract(config.model)
        return cls(
            config=config,
            normalizers=normalizers,
            model=model,
            dataset_capability=dataset_capability,
            branch_contract=branch_contract,
            geometry_backbone_contract=geometry_backbone_contract,
            device=device,
        )

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
        semantics = build_output_semantics(
            field_names=list(self.config.data.field_names),
            pressure_target_mode=self.config.data.pressure_target_mode,
            scalar_names=scalar_names,
        )

        result: Dict[str, Any] = {
            "query_points": query_points_raw,
            "predicted_fields": predicted_fields.astype(np.float32),
            "predicted_scalars": predicted_scalar_map,
            "metadata": {
                "model_name": self.config.model.name,
                "field_names": list(self.config.data.field_names),
                "scalar_names": scalar_names[: len(predicted_scalars)],
                "branch_input_mode": self.config.data.branch_input_mode,
                "pressure_target_mode": self.config.data.pressure_target_mode,
                "pressure_semantics": semantics["pressure"],
                "task_semantics": semantics["tasks"],
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
                "dataset_capability": (
                    self.dataset_capability.as_dict() if self.dataset_capability is not None else None
                ),
                "branch_contract": self.branch_contract.as_dict() if self.branch_contract is not None else None,
                "geometry_backbone_contract": (
                    self.geometry_backbone_contract.as_dict() if self.geometry_backbone_contract is not None else None
                ),
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
        geometry_params: Optional[np.ndarray],
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
        geometry_mode: Optional[str] = None,
        geometry_points: Optional[np.ndarray] = None,
        upper_surface_points: Optional[np.ndarray] = None,
        lower_surface_points: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        try:
            canonical_geometry = resolve_geometry_input(
                geometry_mode=geometry_mode,
                geometry_params=geometry_params,
                geometry_points=geometry_points,
                upper_surface_points=upper_surface_points,
                lower_surface_points=lower_surface_points,
                num_points=self.config.data.num_surface_points,
            )
        except GeometryInputError as exc:
            raise ValueError(str(exc)) from exc

        branch_inputs, geometry_metadata = self._build_inference_branch_inputs(
            canonical_geometry=canonical_geometry,
            mach=mach,
            aoa_deg=aoa_deg,
            reynolds=reynolds,
        )
        if surface_points is None:
            surface_points = canonical_geometry.canonical_surface_points
        result = self.predict(
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
            export_dir=None,
        )
        result["metadata"]["geometry_semantics"] = geometry_metadata
        if export_dir is not None:
            self.export_analysis_bundle(result, output_dir=export_dir)
        return result

    def _build_inference_branch_inputs(
        self,
        canonical_geometry: CanonicalGeometry2D,
        mach: float,
        aoa_deg: float,
        reynolds: Optional[float] = None,
    ) -> tuple[np.ndarray, dict[str, str]]:
        """Build branch features compatible with the training-time branch schema.

        For the synthetic/NACA path we use the configured geometry encoder. For the raw
        AirfRANS path, historical datasets store a flattened surface signature followed
        by flow conditions. This fallback reconstructs that format from the expected
        branch dimension in the fitted normalizer.
        """

        expected_dim = int(self.normalizers.branch.mean.shape[0])
        contract = self.branch_contract or BranchInputContract(
            branch_input_mode=self.config.data.branch_input_mode,
            branch_feature_mode=self.config.data.branch_feature_mode,
            branch_input_dim=expected_dim,
            geometry_representation=canonical_geometry.geometry_representation,
            branch_encoding_type=(
                "encoded_geometry_compatible_features"
                if self.config.data.branch_input_mode == "encoded_geometry"
                else canonical_geometry.branch_encoding_type
            ),
            include_reynolds=self.config.data.include_reynolds,
            num_surface_points=self.config.data.num_surface_points,
            encoded_geometry_latent_dim=self.config.data.encoded_geometry_latent_dim,
        )
        adapter = BranchInputAdapter(
            branch_input_mode=contract.branch_input_mode,
            branch_feature_mode=contract.branch_feature_mode,
            signature_points=contract.num_surface_points,
            encoded_geometry_latent_dim=contract.encoded_geometry_latent_dim,
        )
        try:
            branch_inputs = adapter.build_for_checkpoint(
                canonical_geometry,
                mach=mach,
                aoa_deg=aoa_deg,
                reynolds=reynolds,
                contract=contract,
            )
        except ValueError as exc:
            raise ValueError(
                "Could not build branch_inputs for this checkpoint from the provided geometry input. "
                f"expected_branch_dim={expected_dim}, "
                f"branch_input_mode={contract.branch_input_mode}, "
                f"branch_feature_mode={contract.branch_feature_mode}, "
                f"geometry_representation={contract.geometry_representation}, "
                f"branch_encoding_type={contract.branch_encoding_type}. "
                "This usually means the checkpoint expects a branch contract that cannot be safely reconstructed "
                "from the selected geometry input. Provide compatible geometry_params or surface_points."
            ) from exc

        if contract.branch_input_mode == "encoded_geometry":
            if canonical_geometry.adapter_note:
                maybe_warn_geometry_adapter(canonical_geometry.adapter_note)
            return branch_inputs, build_inference_geometry_semantics(
                geometry_mode=canonical_geometry.geometry_mode,
                geometry_representation=contract.geometry_representation,
                branch_encoding_type=contract.branch_encoding_type,
                geometry_reconstructability=canonical_geometry.reconstructability,
                geometry_params_semantics=canonical_geometry.geometry_params_semantics,
                legacy_param_source="encoded_geometry_adapter",
                notes=(
                    canonical_geometry.adapter_note
                    or "Runtime geometry was encoded through the lightweight GeometryEncoder and remapped to a fixed branch-compatible representation."
                ),
            )

        representation = (
            "parameterized_geometry"
            if canonical_geometry.airfoil is not None
            else (
                "sampled_surface_signature"
                if canonical_geometry.geometry_mode == "generic_surface_points"
                else "geometry_summary"
            )
        )
        notes = "Runtime geometry was reconstructed from legacy NACA parameters." if canonical_geometry.airfoil is not None else (
            canonical_geometry.adapter_note
            or "Runtime geometry was adapted through the legacy fixed-feature compatibility path."
        )
        if canonical_geometry.airfoil is None and canonical_geometry.adapter_note:
            maybe_warn_geometry_adapter(canonical_geometry.adapter_note)
        return branch_inputs, build_inference_geometry_semantics(
            geometry_mode=canonical_geometry.geometry_mode,
            geometry_representation=representation,
            branch_encoding_type=contract.branch_encoding_type,
            geometry_reconstructability=canonical_geometry.reconstructability,
            geometry_params_semantics=canonical_geometry.geometry_params_semantics,
            legacy_param_source="naca4_parameter_vector" if canonical_geometry.airfoil is not None else "geometry_adapter",
            notes=notes,
        )

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
                "cp_surface": "derived_from_pressure_channel_and_cp_reference",
                "pressure_surface": "predicted_raw_pressure",
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
            title="Predicted raw pressure field",
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
        return default_scalar_names()

    def _surface_cp_output(self, surface_fields: np.ndarray, cp_reference: np.ndarray) -> np.ndarray:
        return resolve_surface_cp(
            pressure_channel_values=np.asarray(surface_fields[..., 2:3], dtype=np.float32),
            pressure_target_mode=self.config.data.pressure_target_mode,
            cp_reference=cp_reference,
        )

    def _surface_pressure_output(self, surface_fields: np.ndarray, cp_reference: np.ndarray) -> np.ndarray:
        return resolve_pressure_channel(
            pressure_channel_values=np.asarray(surface_fields[..., 2:3], dtype=np.float32),
            pressure_target_mode=self.config.data.pressure_target_mode,
            cp_reference=cp_reference,
        )

    def _export_field_values(self, field_values: np.ndarray, cp_reference: Optional[np.ndarray]) -> np.ndarray:
        export_fields = np.asarray(field_values, dtype=np.float32).copy()
        if self.config.data.pressure_target_mode == "cp_like":
            if cp_reference is None:
                warnings.warn(
                    "pressure_target_mode='cp_like' but cp_reference is unavailable; "
                    "exported predicted_fields[:, 2] remains a cp-like pressure quantity.",
                    stacklevel=2,
                )
                return export_fields
            export_fields[:, 2:3] = resolve_pressure_channel(
                pressure_channel_values=export_fields[:, 2:3],
                pressure_target_mode=self.config.data.pressure_target_mode,
                cp_reference=cp_reference,
            )
        return export_fields
