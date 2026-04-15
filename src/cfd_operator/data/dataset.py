"""PyTorch datasets for CFD operator training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from cfd_operator.data.normalization import StandardNormalizer


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

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        index = int(self.indices[item])
        branch_inputs = self.payload["branch_inputs"][index].astype(np.float32)
        query_points = self.payload["query_points"][index].astype(np.float32)
        field_targets = self.payload["field_targets"][index].astype(np.float32)
        farfield_mask = self.payload["farfield_mask"][index].astype(np.float32)
        farfield_targets = self.payload["farfield_targets"][index].astype(np.float32)
        surface_points = self.payload["surface_points"][index].astype(np.float32)
        surface_normals = self.payload["surface_normals"][index].astype(np.float32)
        surface_arc_length = self.payload["surface_arc_length"][index].astype(np.float32)
        cp_reference = self.payload["cp_reference"][index].astype(np.float32)
        surface_cp = self.payload["surface_cp"][index].astype(np.float32)
        surface_pressure = self.payload["surface_pressure"][index].astype(np.float32)
        surface_velocity = self.payload["surface_velocity"][index].astype(np.float32)
        surface_nut = self.payload["surface_nut"][index].astype(np.float32)
        surface_heat_flux = self.payload["surface_heat_flux"][index].astype(np.float32)
        surface_wall_shear = self.payload["surface_wall_shear"][index].astype(np.float32)
        slice_points = self.payload["slice_points"][index].astype(np.float32)
        slice_fields = self.payload["slice_fields"][index].astype(np.float32)
        pressure_gradient_indicator = self.payload["pressure_gradient_indicator"][index].astype(np.float32)
        shock_indicator = self.payload["shock_indicator"][index].astype(np.float32)
        high_gradient_mask = self.payload["high_gradient_mask"][index].astype(np.float32)
        shock_location = self.payload["shock_location"][index].astype(np.float32)
        surface_pressure_available = self.payload["surface_pressure_available"][index].astype(np.float32)
        surface_velocity_available = self.payload["surface_velocity_available"][index].astype(np.float32)
        surface_nut_available = self.payload["surface_nut_available"][index].astype(np.float32)
        surface_heat_flux_available = self.payload["surface_heat_flux_available"][index].astype(np.float32)
        surface_wall_shear_available = self.payload["surface_wall_shear_available"][index].astype(np.float32)
        slice_available = self.payload["slice_available"][index].astype(np.float32)
        feature_available = self.payload["feature_available"][index].astype(np.float32)
        nut_available = self.payload["nut_available"][index].astype(np.float32)
        shock_location_available = np.asarray(self.payload["shock_location_available"][index], dtype=np.float32).reshape(1)
        scalar_targets = self.payload["scalar_targets"][index].astype(np.float32)
        scalar_component_targets = self.payload["scalar_component_targets"][index].astype(np.float32)
        scalar_component_available = self.payload["scalar_component_available"][index].astype(np.float32)

        query_mask = np.ones(query_points.shape[0], dtype=np.float32)
        surface_mask = np.ones(surface_points.shape[0], dtype=np.float32)
        slice_mask = np.ones(slice_points.shape[0], dtype=np.float32)

        sample: dict[str, torch.Tensor | str] = {
            "airfoil_id": str(self.payload["airfoil_id"][index]),
            "geometry_params": torch.from_numpy(self.payload["geometry_params"][index].astype(np.float32)),
            "flow_conditions": torch.from_numpy(self.payload["flow_conditions"][index].astype(np.float32)),
            "branch_inputs_raw": torch.from_numpy(branch_inputs),
            "branch_inputs": torch.from_numpy(self.branch_normalizer.transform(branch_inputs)),
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
