"""Dataset quality validation utilities."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np


REQUIRED_KEYS = {
    "airfoil_id",
    "geometry_params",
    "flow_conditions",
    "branch_inputs",
    "query_points",
    "field_targets",
    "farfield_mask",
    "farfield_targets",
    "surface_points",
    "surface_normals",
    "cp_reference",
    "surface_cp",
    "surface_pressure",
    "surface_velocity",
    "surface_nut",
    "surface_heat_flux",
    "surface_wall_shear",
    "surface_arc_length",
    "slice_points",
    "slice_fields",
    "pressure_gradient_indicator",
    "shock_indicator",
    "high_gradient_mask",
    "shock_location",
    "scalar_targets",
    "fidelity_level",
    "source",
    "convergence_flag",
}


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_dataset_payload(payload: Dict[str, Any], strict: bool = True) -> None:
    missing = REQUIRED_KEYS.difference(payload.keys())
    _ensure(not missing, f"Dataset payload missing required keys: {sorted(missing)}")

    num_samples = len(payload["airfoil_id"])
    _ensure(num_samples > 0, "Dataset payload contains no samples")

    for key in REQUIRED_KEYS:
        values = payload[key]
        _ensure(len(values) == num_samples, f"Key '{key}' has inconsistent sample count")

    for array_key in [
        "geometry_params",
        "flow_conditions",
        "branch_inputs",
        "query_points",
        "field_targets",
        "farfield_mask",
        "farfield_targets",
        "surface_points",
        "surface_normals",
        "cp_reference",
        "surface_cp",
        "surface_pressure",
        "surface_velocity",
        "surface_nut",
        "surface_heat_flux",
        "surface_wall_shear",
        "surface_arc_length",
        "slice_points",
        "slice_fields",
        "pressure_gradient_indicator",
        "shock_indicator",
        "high_gradient_mask",
        "scalar_targets",
    ]:
        values = np.asarray(payload[array_key])
        _ensure(np.isfinite(values).all(), f"Key '{array_key}' contains non-finite values")

    query_points = np.asarray(payload["query_points"])
    field_targets = np.asarray(payload["field_targets"])
    surface_points = np.asarray(payload["surface_points"])
    surface_normals = np.asarray(payload["surface_normals"])
    cp_reference = np.asarray(payload["cp_reference"])
    farfield_mask = np.asarray(payload["farfield_mask"])
    slice_points = np.asarray(payload["slice_points"])
    slice_fields = np.asarray(payload["slice_fields"])
    pressure_gradient_indicator = np.asarray(payload["pressure_gradient_indicator"])
    shock_indicator = np.asarray(payload["shock_indicator"])
    high_gradient_mask = np.asarray(payload["high_gradient_mask"])
    shock_location = np.asarray(payload["shock_location"])
    shock_location_available = np.asarray(
        payload.get(
            "shock_location_available",
            np.isfinite(shock_location[:, 0]).astype(np.float32),
        )
    ).reshape(-1)

    _ensure(query_points.ndim == 3 and query_points.shape[-1] == 2, "query_points must have shape [N, Q, 2]")
    _ensure(field_targets.ndim == 3 and field_targets.shape[-1] == 4, "field_targets must have shape [N, Q, 4]")
    _ensure(surface_points.ndim == 3 and surface_points.shape[-1] == 2, "surface_points must have shape [N, S, 2]")
    _ensure(surface_normals.shape == surface_points.shape, "surface_normals shape must match surface_points")
    _ensure(cp_reference.ndim == 2 and cp_reference.shape[-1] == 2, "cp_reference must have shape [N, 2]")
    _ensure(farfield_mask.shape == query_points.shape[:2], "farfield_mask must have shape [N, Q]")
    _ensure(slice_points.ndim == 3 and slice_points.shape[-1] == 2, "slice_points must have shape [N, L, 2]")
    _ensure(slice_fields.ndim == 3 and slice_fields.shape[-1] == 4, "slice_fields must have shape [N, L, 4]")
    _ensure(pressure_gradient_indicator.shape == query_points.shape[:2] + (1,), "pressure_gradient_indicator must have shape [N, Q, 1]")
    _ensure(shock_indicator.shape == query_points.shape[:2] + (1,), "shock_indicator must have shape [N, Q, 1]")
    _ensure(high_gradient_mask.shape == query_points.shape[:2] + (1,), "high_gradient_mask must have shape [N, Q, 1]")
    _ensure(shock_location.shape == (num_samples, 2), "shock_location must have shape [N, 2]")
    if np.any(shock_location_available > 0.5):
        _ensure(
            np.isfinite(shock_location[shock_location_available > 0.5]).all(),
            "Available shock_location entries must be finite",
        )

    normal_norm = np.linalg.norm(surface_normals, axis=-1)
    _ensure(np.all(normal_norm > 1.0e-4), "surface_normals contain near-zero vectors")

    farfield_counts = farfield_mask.sum(axis=1)
    _ensure(np.all(farfield_counts > 0), "Each sample must contain at least one farfield query point")

    if strict:
        for split_name in [
            "train_indices",
            "val_indices",
            "test_indices",
            "test_unseen_geometry_indices",
            "test_unseen_condition_indices",
        ]:
            if split_name in payload:
                indices = np.asarray(payload[split_name], dtype=np.int64)
                _ensure(indices.size > 0, f"Split '{split_name}' is empty")
                _ensure(indices.min() >= 0 and indices.max() < num_samples, f"Split '{split_name}' contains out-of-range indices")

        _validate_split_disjointness(payload)
        _validate_unseen_geometry_split(payload)
        _validate_unseen_condition_split(payload)


def _validate_split_disjointness(payload: Dict[str, Any]) -> None:
    split_keys = [key for key in payload.keys() if key.endswith("_indices")]
    seen: Dict[int, str] = {}
    for split_key in split_keys:
        for index in np.asarray(payload[split_key], dtype=np.int64):
            previous = seen.get(int(index))
            _ensure(previous is None, f"Index {index} appears in both '{previous}' and '{split_key}'")
            seen[int(index)] = split_key


def _validate_unseen_geometry_split(payload: Dict[str, Any]) -> None:
    if "test_unseen_geometry_indices" not in payload:
        return
    airfoil_ids = np.asarray(payload["airfoil_id"])
    train_geometry_ids = set(airfoil_ids[np.asarray(payload["train_indices"], dtype=np.int64)].tolist())
    held_out_ids = set(airfoil_ids[np.asarray(payload["test_unseen_geometry_indices"], dtype=np.int64)].tolist())
    _ensure(train_geometry_ids.isdisjoint(held_out_ids), "Unseen geometry split leaks geometry IDs into training")


def _validate_unseen_condition_split(payload: Dict[str, Any]) -> None:
    if "test_unseen_condition_indices" not in payload:
        return
    train_indices = np.asarray(payload["train_indices"], dtype=np.int64)
    seen_pool_indices = np.concatenate(
        [
            np.asarray(payload["train_indices"], dtype=np.int64),
            np.asarray(payload["val_indices"], dtype=np.int64),
            np.asarray(payload["test_indices"], dtype=np.int64),
        ]
    )
    test_indices = np.asarray(payload["test_unseen_condition_indices"], dtype=np.int64)
    train_airfoil_ids = set(np.asarray(payload["airfoil_id"])[seen_pool_indices].tolist())
    held_out_airfoil_ids = set(np.asarray(payload["airfoil_id"])[test_indices].tolist())
    _ensure(held_out_airfoil_ids.issubset(train_airfoil_ids), "Unseen condition split should reuse seen geometries")

    train_flow = np.asarray(payload["flow_conditions"])[train_indices]
    held_out_flow = np.asarray(payload["flow_conditions"])[test_indices]
    train_max_mach = float(train_flow[:, 0].max())
    train_max_aoa = float(train_flow[:, 1].max())
    held_out_has_new_region = bool(
        np.any(held_out_flow[:, 0] > train_max_mach + 1.0e-6)
        or np.any(held_out_flow[:, 1] > train_max_aoa + 1.0e-6)
    )
    _ensure(held_out_has_new_region, "Unseen condition split should contain conditions outside the training range")
