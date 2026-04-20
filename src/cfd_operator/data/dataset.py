"""PyTorch datasets for CFD operator training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from cfd_operator.data.normalization import StandardNormalizer
from cfd_operator.output_semantics import format_missing_fields_message
from cfd_operator.schema import CFDSurrogateSample


def _as_float_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def _text_from_value(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value)
    return text if text else default


def _scalar_target_vector(sample: CFDSurrogateSample) -> np.ndarray:
    scalar_targets = sample.targets.scalar_targets
    return np.asarray(
        [
            float(scalar_targets.get("cl", 0.0)),
            float(scalar_targets.get("cd", 0.0)),
        ],
        dtype=np.float32,
    )


def _surface_target(sample: CFDSurrogateSample, name: str, coord_dim: int) -> np.ndarray:
    value = sample.targets.surface_targets.get(name)
    surface_points = sample.geometry.surface_points
    num_surface_points = 0 if surface_points is None else int(surface_points.shape[0])
    if value is not None:
        return np.asarray(value, dtype=np.float32)
    default_shape = (num_surface_points, 1)
    if name == "surface_velocity":
        default_shape = (num_surface_points, coord_dim)
    return np.zeros(default_shape, dtype=np.float32)


def _feature_target(sample: CFDSurrogateSample, name: str, num_query_points: int) -> np.ndarray:
    value = sample.targets.feature_targets.get(name)
    if value is not None:
        return np.asarray(value, dtype=np.float32)
    if name == "shock_location":
        return np.zeros((1,), dtype=np.float32)
    return np.zeros((num_query_points, 1), dtype=np.float32)


def _availability_value(sample: CFDSurrogateSample, name: str, length: int) -> np.ndarray:
    status = sample.availability.available.get(name, False)
    return np.full((length, 1), 1.0 if status else 0.0, dtype=np.float32)


def _sample_to_training_view(
    sample: CFDSurrogateSample,
    *,
    branch_normalizer: StandardNormalizer,
    coordinate_normalizer: StandardNormalizer,
    field_normalizer: StandardNormalizer,
    scalar_normalizer: StandardNormalizer,
    field_names: list[str] | tuple[str, ...] | None = None,
    include_reynolds: bool = False,
) -> dict[str, torch.Tensor | str]:
    field_names = list(field_names or ["u", "v", "p", "aux"])
    query_set = sample.query_sets["field_query_points"]
    query_points = np.asarray(query_set.points, dtype=np.float32)
    coord_dim = int(query_points.shape[-1])

    branch_inputs = np.asarray(sample.geometry.geometry_features.get("branch_inputs"), dtype=np.float32).reshape(-1)
    geometry_params = sample.geometry.geometry_repr
    if geometry_params is None:
        geometry_params = np.zeros((0,), dtype=np.float32)
    else:
        geometry_params = np.asarray(geometry_params, dtype=np.float32).reshape(-1)

    flow_vector = [float(sample.condition.mach or 0.0), float(sample.condition.aoa or 0.0)]
    if include_reynolds:
        flow_vector.append(float(sample.condition.reynolds or 0.0))
    flow_conditions = np.asarray(flow_vector, dtype=np.float32)

    field_targets = sample.targets.field_targets
    if field_targets is None:
        field_targets = np.zeros((query_points.shape[0], len(field_names)), dtype=np.float32)
    else:
        field_targets = np.asarray(field_targets, dtype=np.float32)

    surface_points = sample.geometry.surface_points
    if surface_points is None:
        surface_points = np.asarray(sample.query_sets.get("surface_query_points", query_set).points, dtype=np.float32)
    else:
        surface_points = np.asarray(surface_points, dtype=np.float32)
    surface_normals = sample.geometry.surface_normals
    if surface_normals is None:
        surface_normals = np.zeros_like(surface_points, dtype=np.float32)
    else:
        surface_normals = np.asarray(surface_normals, dtype=np.float32)
    surface_arc_length = np.zeros((surface_points.shape[0], 1), dtype=np.float32)

    slice_query = sample.query_sets.get("slice_query_points")
    if slice_query is None:
        slice_points = np.zeros((0, coord_dim), dtype=np.float32)
    else:
        slice_points = np.asarray(slice_query.points, dtype=np.float32)
    slice_fields = np.asarray(sample.targets.query_targets.get("slice_fields"), dtype=np.float32) if "slice_fields" in sample.targets.query_targets else np.zeros((slice_points.shape[0], len(field_names)), dtype=np.float32)

    cp_reference = np.asarray(sample.legacy_payload.get("cp_reference", np.asarray([1.0, 1.0], dtype=np.float32)), dtype=np.float32).reshape(2)
    farfield_targets = np.asarray(
        sample.legacy_payload.get("farfield_targets", np.zeros((len(field_names),), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    farfield_mask = np.ones((query_points.shape[0],), dtype=np.float32)

    surface_cp = _surface_target(sample, "surface_cp", coord_dim)
    surface_pressure = _surface_target(sample, "surface_pressure", coord_dim)
    surface_velocity = _surface_target(sample, "surface_velocity", coord_dim)
    surface_nut = _surface_target(sample, "surface_nut", coord_dim)
    surface_heat_flux = _surface_target(sample, "surface_heat_flux", coord_dim)
    surface_wall_shear = _surface_target(sample, "surface_wall_shear", coord_dim)

    pressure_gradient_indicator = _feature_target(sample, "pressure_gradient_indicator", query_points.shape[0])
    shock_indicator = _feature_target(sample, "shock_indicator", query_points.shape[0])
    high_gradient_mask = _feature_target(sample, "high_gradient_mask", query_points.shape[0])
    shock_location = _feature_target(sample, "shock_location", query_points.shape[0])

    surface_pressure_available = _availability_value(sample, "surface_pressure", surface_points.shape[0])
    surface_velocity_available = _availability_value(sample, "surface_velocity", surface_points.shape[0])
    surface_nut_available = _availability_value(sample, "surface_nut", surface_points.shape[0])
    surface_heat_flux_available = _availability_value(sample, "surface_heat_flux", surface_points.shape[0])
    surface_wall_shear_available = _availability_value(sample, "surface_wall_shear", surface_points.shape[0])
    slice_available = _availability_value(sample, "slice_fields", slice_points.shape[0])
    feature_available = _availability_value(sample, "pressure_gradient_indicator", query_points.shape[0])
    nut_available = np.full((query_points.shape[0], 1), 1.0 if "nut" in field_names else 0.0, dtype=np.float32)
    shock_location_available = np.asarray(
        [1.0 if sample.availability.available.get("shock_location", False) else 0.0],
        dtype=np.float32,
    )

    scalar_targets = _scalar_target_vector(sample)
    scalar_component_targets = np.zeros((5,), dtype=np.float32)
    scalar_component_available = np.zeros((5,), dtype=np.float32)

    query_mask = np.ones(query_points.shape[0], dtype=np.float32)
    surface_mask = np.ones(surface_points.shape[0], dtype=np.float32)
    slice_mask = np.ones(slice_points.shape[0], dtype=np.float32)

    return {
        "airfoil_id": sample.metadata.case_id,
        "geometry_mode": _text_from_value(sample.geometry.geometry_mode),
        "geometry_source": _text_from_value(sample.geometry.geometry_source),
        "geometry_representation": _text_from_value(sample.geometry.geometry_representation),
        "branch_encoding_type": _text_from_value(sample.geometry.branch_encoding_type),
        "geometry_reconstructability": _text_from_value(sample.geometry.geometry_reconstructability),
        "geometry_params_semantics": _text_from_value(sample.geometry.geometry_params_semantics),
        "legacy_param_source": _text_from_value(sample.geometry.legacy_param_source, default="none"),
        "branch_input_mode": _text_from_value(sample.legacy_payload.get("branch_input_mode"), default="legacy_fixed_features"),
        "branch_input_source": _text_from_value(sample.legacy_payload.get("branch_input_source"), default="legacy_fixed_features"),
        "geometry_encoding_meta": _text_from_value(sample.geometry.geometry_encoding_meta, default=""),
        "surface_sampling_info": _text_from_value(sample.geometry.surface_sampling_info, default=""),
        "geometry_params": _as_float_tensor(geometry_params),
        "flow_conditions": _as_float_tensor(flow_conditions),
        "branch_inputs_raw": _as_float_tensor(branch_inputs),
        "branch_inputs": _as_float_tensor(branch_normalizer.transform(branch_inputs)),
        "geometry_points_raw": _as_float_tensor(np.asarray(sample.geometry.surface_points if sample.geometry.surface_points is not None else np.zeros((0, coord_dim), dtype=np.float32), dtype=np.float32)),
        "query_points_raw": _as_float_tensor(query_points),
        "query_points": _as_float_tensor(coordinate_normalizer.transform(query_points)),
        "field_targets_raw": _as_float_tensor(field_targets),
        "field_targets": _as_float_tensor(field_normalizer.transform(field_targets)),
        "farfield_mask": _as_float_tensor(farfield_mask),
        "farfield_targets": _as_float_tensor(farfield_targets),
        "surface_points_raw": _as_float_tensor(surface_points),
        "surface_points": _as_float_tensor(coordinate_normalizer.transform(surface_points)),
        "surface_normals": _as_float_tensor(surface_normals),
        "surface_arc_length": _as_float_tensor(surface_arc_length),
        "cp_reference": _as_float_tensor(cp_reference),
        "surface_cp": _as_float_tensor(surface_cp),
        "surface_pressure": _as_float_tensor(surface_pressure),
        "surface_velocity": _as_float_tensor(surface_velocity),
        "surface_nut": _as_float_tensor(surface_nut),
        "surface_heat_flux": _as_float_tensor(surface_heat_flux),
        "surface_wall_shear": _as_float_tensor(surface_wall_shear),
        "surface_pressure_available": _as_float_tensor(surface_pressure_available),
        "surface_velocity_available": _as_float_tensor(surface_velocity_available),
        "surface_nut_available": _as_float_tensor(surface_nut_available),
        "surface_heat_flux_available": _as_float_tensor(surface_heat_flux_available),
        "surface_wall_shear_available": _as_float_tensor(surface_wall_shear_available),
        "slice_points_raw": _as_float_tensor(slice_points),
        "slice_points": _as_float_tensor(coordinate_normalizer.transform(slice_points) if slice_points.shape[0] > 0 else slice_points),
        "slice_fields_raw": _as_float_tensor(slice_fields),
        "slice_fields": _as_float_tensor(field_normalizer.transform(slice_fields) if slice_fields.shape[0] > 0 else slice_fields),
        "slice_available": _as_float_tensor(slice_available),
        "pressure_gradient_indicator": _as_float_tensor(pressure_gradient_indicator),
        "shock_indicator": _as_float_tensor(shock_indicator),
        "high_gradient_mask": _as_float_tensor(high_gradient_mask),
        "feature_available": _as_float_tensor(feature_available),
        "nut_available": _as_float_tensor(nut_available),
        "shock_location": _as_float_tensor(shock_location),
        "shock_location_available": _as_float_tensor(shock_location_available),
        "scalar_targets_raw": _as_float_tensor(scalar_targets),
        "scalar_targets": _as_float_tensor(scalar_normalizer.transform(scalar_targets)),
        "scalar_component_targets": _as_float_tensor(scalar_component_targets),
        "scalar_component_available": _as_float_tensor(scalar_component_available),
        "query_mask": _as_float_tensor(query_mask),
        "surface_mask": _as_float_tensor(surface_mask),
        "slice_mask": _as_float_tensor(slice_mask),
        "field_names": field_names,
        "fidelity_level": torch.tensor(int(sample.legacy_payload.get("fidelity_level", 0)), dtype=torch.long),
        "convergence_flag": torch.tensor(int(sample.legacy_payload.get("convergence_flag", 1)), dtype=torch.long),
    }


class CFDOperatorDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset backed by dense per-sample arrays."""

    def __init__(
        self,
        payload: dict[str, Any],
        indices: np.ndarray,
        branch_normalizer: StandardNormalizer,
        coordinate_normalizer: StandardNormalizer,
        field_normalizer: StandardNormalizer,
        scalar_normalizer: StandardNormalizer,
    ) -> None:
        self.payload = payload
        self.indices = indices.astype(np.int64)
        self.branch_normalizer = branch_normalizer
        self.coordinate_normalizer = coordinate_normalizer
        self.field_normalizer = field_normalizer
        self.scalar_normalizer = scalar_normalizer

    def _payload_array(self, key: str, index: int, dtype: np.dtype = np.float32) -> np.ndarray:
        if key not in self.payload:
            raise KeyError(format_missing_fields_message("CFDOperatorDataset payload", [key]))
        return np.asarray(self.payload[key][index], dtype=dtype)

    def _payload_text(self, key: str, index: int, default: str) -> str:
        values = self.payload.get(key)
        if values is None:
            return default
        if isinstance(values, np.ndarray):
            if values.ndim == 0 or values.shape[0] <= index:
                return default
            return str(values[index])
        if isinstance(values, (list, tuple)):
            if len(values) <= index:
                return default
            return str(values[index])
        return str(values)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        index = int(self.indices[item])
        branch_inputs = self._payload_array("branch_inputs", index)
        query_points = self._payload_array("query_points", index)
        field_targets = self._payload_array("field_targets", index)
        farfield_mask = self._payload_array("farfield_mask", index)
        farfield_targets = self._payload_array("farfield_targets", index)
        surface_points = self._payload_array("surface_points", index)
        surface_normals = self._payload_array("surface_normals", index)
        surface_arc_length = self._payload_array("surface_arc_length", index)
        cp_reference = self._payload_array("cp_reference", index)
        surface_cp = self._payload_array("surface_cp", index)
        surface_pressure = self._payload_array("surface_pressure", index)
        surface_velocity = self._payload_array("surface_velocity", index)
        surface_nut = self._payload_array("surface_nut", index)
        surface_heat_flux = self._payload_array("surface_heat_flux", index)
        surface_wall_shear = self._payload_array("surface_wall_shear", index)
        slice_points = self._payload_array("slice_points", index)
        slice_fields = self._payload_array("slice_fields", index)
        pressure_gradient_indicator = self._payload_array("pressure_gradient_indicator", index)
        shock_indicator = self._payload_array("shock_indicator", index)
        high_gradient_mask = self._payload_array("high_gradient_mask", index)
        shock_location = self._payload_array("shock_location", index)
        surface_pressure_available = self._payload_array("surface_pressure_available", index)
        surface_velocity_available = self._payload_array("surface_velocity_available", index)
        surface_nut_available = self._payload_array("surface_nut_available", index)
        surface_heat_flux_available = self._payload_array("surface_heat_flux_available", index)
        surface_wall_shear_available = self._payload_array("surface_wall_shear_available", index)
        slice_available = self._payload_array("slice_available", index)
        feature_available = self._payload_array("feature_available", index)
        nut_available = self._payload_array("nut_available", index)
        shock_location_available = np.asarray(self.payload["shock_location_available"][index], dtype=np.float32).reshape(1)
        scalar_targets = self._payload_array("scalar_targets", index)
        scalar_component_targets = self._payload_array("scalar_component_targets", index)
        scalar_component_available = self._payload_array("scalar_component_available", index)

        query_mask = np.ones(query_points.shape[0], dtype=np.float32)
        surface_mask = np.ones(surface_points.shape[0], dtype=np.float32)
        slice_mask = np.ones(slice_points.shape[0], dtype=np.float32)

        sample: dict[str, torch.Tensor | str] = {
            "airfoil_id": str(self.payload["airfoil_id"][index]),
            "geometry_mode": self._payload_text("geometry_mode", index, "unknown"),
            "geometry_source": self._payload_text("geometry_source", index, "unknown"),
            "geometry_representation": self._payload_text("geometry_representation", index, "unknown"),
            "branch_encoding_type": self._payload_text("branch_encoding_type", index, "unknown"),
            "geometry_reconstructability": self._payload_text("geometry_reconstructability", index, "unknown"),
            "geometry_params_semantics": self._payload_text("geometry_params_semantics", index, "unknown"),
            "legacy_param_source": self._payload_text("legacy_param_source", index, "unknown"),
            "branch_input_mode": self._payload_text("branch_input_mode", index, "legacy_fixed_features"),
            "branch_input_source": self._payload_text("branch_input_source", index, "legacy_fixed_features"),
            "geometry_encoding_meta": self._payload_text("geometry_encoding_meta", index, ""),
            "surface_sampling_info": self._payload_text("surface_sampling_info", index, ""),
            "geometry_params": torch.from_numpy(self._payload_array("geometry_params", index)),
            "flow_conditions": torch.from_numpy(self._payload_array("flow_conditions", index)),
            "branch_inputs_raw": torch.from_numpy(branch_inputs),
            "branch_inputs": torch.from_numpy(self.branch_normalizer.transform(branch_inputs)),
            "geometry_points_raw": torch.from_numpy(self._payload_array("geometry_points", index)),
            "query_points_raw": torch.from_numpy(query_points),
            "query_points": torch.from_numpy(self.coordinate_normalizer.transform(query_points)),
            "field_targets_raw": torch.from_numpy(field_targets),
            "field_targets": torch.from_numpy(self.field_normalizer.transform(field_targets)),
            "farfield_mask": torch.from_numpy(farfield_mask),
            "farfield_targets": torch.from_numpy(farfield_targets),
            "surface_points_raw": torch.from_numpy(surface_points),
            "surface_points": torch.from_numpy(self.coordinate_normalizer.transform(surface_points)),
            "surface_normals": torch.from_numpy(surface_normals),
            "surface_arc_length": torch.from_numpy(surface_arc_length),
            "cp_reference": torch.from_numpy(cp_reference),
            "surface_cp": torch.from_numpy(surface_cp),
            "surface_pressure": torch.from_numpy(surface_pressure),
            "surface_velocity": torch.from_numpy(surface_velocity),
            "surface_nut": torch.from_numpy(surface_nut),
            "surface_heat_flux": torch.from_numpy(surface_heat_flux),
            "surface_wall_shear": torch.from_numpy(surface_wall_shear),
            "surface_pressure_available": torch.from_numpy(surface_pressure_available),
            "surface_velocity_available": torch.from_numpy(surface_velocity_available),
            "surface_nut_available": torch.from_numpy(surface_nut_available),
            "surface_heat_flux_available": torch.from_numpy(surface_heat_flux_available),
            "surface_wall_shear_available": torch.from_numpy(surface_wall_shear_available),
            "slice_points_raw": torch.from_numpy(slice_points),
            "slice_points": torch.from_numpy(self.coordinate_normalizer.transform(slice_points)),
            "slice_fields_raw": torch.from_numpy(slice_fields),
            "slice_fields": torch.from_numpy(self.field_normalizer.transform(slice_fields)),
            "slice_available": torch.from_numpy(slice_available),
            "pressure_gradient_indicator": torch.from_numpy(pressure_gradient_indicator),
            "shock_indicator": torch.from_numpy(shock_indicator),
            "high_gradient_mask": torch.from_numpy(high_gradient_mask),
            "feature_available": torch.from_numpy(feature_available),
            "nut_available": torch.from_numpy(nut_available),
            "shock_location": torch.from_numpy(shock_location),
            "shock_location_available": torch.from_numpy(shock_location_available),
            "scalar_targets_raw": torch.from_numpy(scalar_targets),
            "scalar_targets": torch.from_numpy(self.scalar_normalizer.transform(scalar_targets)),
            "scalar_component_targets": torch.from_numpy(scalar_component_targets),
            "scalar_component_available": torch.from_numpy(scalar_component_available),
            "query_mask": torch.from_numpy(query_mask),
            "surface_mask": torch.from_numpy(surface_mask),
            "slice_mask": torch.from_numpy(slice_mask),
            "field_names": list(self.payload.get("field_names", ["u", "v", "p", "aux"])),
            "fidelity_level": torch.tensor(self.payload["fidelity_level"][index], dtype=torch.long),
            "convergence_flag": torch.tensor(self.payload["convergence_flag"][index], dtype=torch.long),
        }
        return sample


class SchemaBackedCFDOperatorDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset backed by unified CFDSurrogateSample objects."""

    def __init__(
        self,
        samples: list[CFDSurrogateSample],
        indices: np.ndarray,
        branch_normalizer: StandardNormalizer,
        coordinate_normalizer: StandardNormalizer,
        field_normalizer: StandardNormalizer,
        scalar_normalizer: StandardNormalizer,
        *,
        field_names: list[str] | tuple[str, ...],
        include_reynolds: bool,
    ) -> None:
        self.samples = samples
        self.indices = indices.astype(np.int64)
        self.branch_normalizer = branch_normalizer
        self.coordinate_normalizer = coordinate_normalizer
        self.field_normalizer = field_normalizer
        self.scalar_normalizer = scalar_normalizer
        self.field_names = list(field_names)
        self.include_reynolds = include_reynolds

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        index = int(self.indices[item])
        sample = self.samples[index]
        return _sample_to_training_view(
            sample,
            branch_normalizer=self.branch_normalizer,
            coordinate_normalizer=self.coordinate_normalizer,
            field_normalizer=self.field_normalizer,
            scalar_normalizer=self.scalar_normalizer,
            field_names=self.field_names,
            include_reynolds=self.include_reynolds,
        )
