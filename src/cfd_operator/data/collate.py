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
        "scalar_targets": torch.stack([item["scalar_targets"] for item in batch]),
        "scalar_targets_raw": torch.stack([item["scalar_targets_raw"] for item in batch]),
        "fidelity_level": torch.stack([item["fidelity_level"] for item in batch]),
        "convergence_flag": torch.stack([item["convergence_flag"] for item in batch]),
    }

    for key in [
        "query_points",
        "query_points_raw",
        "field_targets",
        "field_targets_raw",
        "surface_points",
        "surface_points_raw",
        "surface_normals",
        "surface_cp",
    ]:
        collated[key], mask = _pad_tensor_sequence([item[key] for item in batch])
        if key.startswith("query_points"):
            collated["query_mask"] = mask
        if key.startswith("surface_points") or key in {"surface_cp", "surface_normals"}:
            collated["surface_mask"] = mask
    collated["farfield_mask"], _ = _pad_tensor_sequence([item["farfield_mask"].unsqueeze(-1) for item in batch])
    collated["farfield_mask"] = collated["farfield_mask"].squeeze(-1)
    return collated
