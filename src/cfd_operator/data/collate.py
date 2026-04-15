"""Batch collation for variable-length fields."""

from __future__ import annotations

from typing import Any

import torch


def _pad_tensor_sequence(values: list[torch.Tensor], padding_value: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(value.shape[0] for value in values)
    batch_size = len(values)
    trailing_shape = values[0].shape[1:]
    output = values[0].new_full((batch_size, max_len, *trailing_shape), fill_value=padding_value)
    mask = values[0].new_zeros((batch_size, max_len), dtype=torch.float32)
    for index, value in enumerate(values):
        length = value.shape[0]
        output[index, :length] = value
        mask[index, :length] = 1.0
    return output, mask


def cfd_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {
        "airfoil_id": [item["airfoil_id"] for item in batch],
        "geometry_params": torch.stack([item["geometry_params"] for item in batch]),
        "flow_conditions": torch.stack([item["flow_conditions"] for item in batch]),
        "branch_inputs_raw": torch.stack([item["branch_inputs_raw"] for item in batch]),
        "branch_inputs": torch.stack([item["branch_inputs"] for item in batch]),
        "farfield_targets": torch.stack([item["farfield_targets"] for item in batch]),
        "cp_reference": torch.stack([item["cp_reference"] for item in batch]),
        "shock_location": torch.stack([item["shock_location"] for item in batch]),
        "shock_location_available": torch.stack([item["shock_location_available"] for item in batch]),
        "scalar_targets": torch.stack([item["scalar_targets"] for item in batch]),
        "scalar_targets_raw": torch.stack([item["scalar_targets_raw"] for item in batch]),
        "scalar_component_targets": torch.stack([item["scalar_component_targets"] for item in batch]),
        "scalar_component_available": torch.stack([item["scalar_component_available"] for item in batch]),
        "field_names": batch[0]["field_names"],
        "fidelity_level": torch.stack([item["fidelity_level"] for item in batch]),
        "convergence_flag": torch.stack([item["convergence_flag"] for item in batch]),
    }

    for key in [
        "query_points",
        "query_points_raw",
        "field_targets",
        "field_targets_raw",
        "pressure_gradient_indicator",
        "shock_indicator",
        "high_gradient_mask",
        "feature_available",
        "nut_available",
        "surface_points",
        "surface_points_raw",
        "surface_normals",
        "surface_arc_length",
        "surface_cp",
        "surface_pressure",
        "surface_velocity",
        "surface_nut",
        "surface_heat_flux",
        "surface_wall_shear",
        "surface_pressure_available",
        "surface_velocity_available",
        "surface_nut_available",
        "surface_heat_flux_available",
        "surface_wall_shear_available",
        "slice_points",
        "slice_points_raw",
        "slice_fields",
        "slice_fields_raw",
        "slice_available",
    ]:
        collated[key], mask = _pad_tensor_sequence([item[key] for item in batch])
        if key.startswith("query_points"):
            collated["query_mask"] = mask
        if key in {"field_targets", "field_targets_raw", "pressure_gradient_indicator", "shock_indicator", "high_gradient_mask", "feature_available", "nut_available"}:
            collated["query_mask"] = mask
        if key.startswith("surface_points") or key in {
            "surface_cp",
            "surface_normals",
            "surface_pressure",
            "surface_velocity",
            "surface_nut",
            "surface_heat_flux",
            "surface_wall_shear",
            "surface_pressure_available",
            "surface_velocity_available",
            "surface_nut_available",
            "surface_heat_flux_available",
            "surface_wall_shear_available",
            "surface_arc_length",
        }:
            collated["surface_mask"] = mask
        if key.startswith("slice_points") or key in {"slice_fields", "slice_fields_raw", "slice_available"}:
            collated["slice_mask"] = mask
    collated["farfield_mask"], _ = _pad_tensor_sequence([item["farfield_mask"].unsqueeze(-1) for item in batch])
    collated["farfield_mask"] = collated["farfield_mask"].squeeze(-1)
    return collated
