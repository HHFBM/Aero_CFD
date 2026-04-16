"""PyTorch datasets for CFD operator training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from cfd_operator.data.normalization import StandardNormalizer
from cfd_operator.output_semantics import format_missing_fields_message


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
