"""Unified CFD surrogate sample schema.

This module intentionally adds a lightweight, dataset-agnostic schema without
replacing the existing dense payload dict consumed by the trainer/evaluator.
Adapters normalize source-specific records into ``CFDSurrogateSample`` and the
rest of the stack can progressively adopt this schema while the legacy payload
bridge remains in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


Dimensionality = Literal["2d", "3d"]


@dataclass(frozen=True)
class SampleMetadata:
    dataset_name: str
    case_id: str
    split: str | None = None
    source_format: str = "unknown"
    dimensionality: Dimensionality = "2d"
    mesh_type: str = "point_cloud"


@dataclass(frozen=True)
class GeometryData:
    geometry_type: str
    geometry_repr: np.ndarray | None = None
    geometry_features: dict[str, np.ndarray] = field(default_factory=dict)
    surface_points: np.ndarray | None = None
    surface_normals: np.ndarray | None = None
    geometry_mode: str = "unknown"
    geometry_source: str = "unknown"
    geometry_representation: str = "unknown"
    branch_encoding_type: str = "unknown"
    geometry_reconstructability: str = "unknown"
    geometry_params_semantics: str = "unknown"
    legacy_param_source: str = "none"
    geometry_encoding_meta: str | None = None
    surface_sampling_info: str | None = None


@dataclass(frozen=True)
class FlowCondition:
    reynolds: float | None = None
    mach: float | None = None
    aoa: float | None = None
    extras: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class QuerySet:
    name: str
    points: np.ndarray
    point_features: dict[str, np.ndarray] = field(default_factory=dict)
    point_type: str | None = None


@dataclass(frozen=True)
class TargetBundle:
    field_targets: np.ndarray | None = None
    surface_targets: dict[str, np.ndarray] = field(default_factory=dict)
    scalar_targets: dict[str, float] = field(default_factory=dict)
    feature_targets: dict[str, np.ndarray] = field(default_factory=dict)
    query_targets: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class AvailabilityBundle:
    available: dict[str, bool] = field(default_factory=dict)
    derived: dict[str, bool] = field(default_factory=dict)
    trainable: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class CFDSurrogateSample:
    metadata: SampleMetadata
    geometry: GeometryData
    condition: FlowCondition
    query_sets: dict[str, QuerySet]
    targets: TargetBundle
    availability: AvailabilityBundle
    legacy_payload: dict[str, Any] = field(default_factory=dict)

    def dimensionality(self) -> int:
        return 2 if self.metadata.dimensionality == "2d" else 3


def _safe_optional_array(payload: dict[str, Any], key: str, index: int) -> np.ndarray | None:
    if key not in payload:
        return None
    value = np.asarray(payload[key][index])
    return value.copy()


def _safe_float(array: np.ndarray | None) -> float | None:
    if array is None:
        return None
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return None
    value = float(flat[0])
    if not np.isfinite(value):
        return None
    return value


def sample_from_legacy_payload(
    payload: dict[str, Any],
    index: int,
    *,
    dataset_name: str,
    split: str | None = None,
    source_format: str = "legacy_payload",
) -> CFDSurrogateSample:
    """Build a unified sample from the existing dense payload schema."""

    geometry_points = _safe_optional_array(payload, "geometry_points", index)
    surface_points = _safe_optional_array(payload, "surface_points", index)
    if geometry_points is None:
        geometry_points = surface_points

    query_points = np.asarray(payload["query_points"][index], dtype=np.float32)
    surface_query_points = surface_points if surface_points is not None else np.zeros((0, query_points.shape[-1]), dtype=np.float32)
    slice_points = _safe_optional_array(payload, "slice_points", index)
    if slice_points is None:
        slice_points = np.zeros((0, query_points.shape[-1]), dtype=np.float32)

    field_targets = _safe_optional_array(payload, "field_targets", index)
    scalar_values = np.asarray(payload["scalar_targets"][index], dtype=np.float32).reshape(-1)
    scalar_targets = {}
    if scalar_values.size >= 1:
        scalar_targets["cl"] = float(scalar_values[0])
    if scalar_values.size >= 2:
        scalar_targets["cd"] = float(scalar_values[1])

    surface_targets = {}
    for key in [
        "surface_cp",
        "surface_pressure",
        "surface_velocity",
        "surface_nut",
        "surface_heat_flux",
        "surface_wall_shear",
    ]:
        value = _safe_optional_array(payload, key, index)
        if value is not None:
            surface_targets[key] = value

    feature_targets = {}
    for key in ["pressure_gradient_indicator", "high_gradient_mask", "shock_indicator", "shock_location"]:
        value = _safe_optional_array(payload, key, index)
        if value is not None:
            feature_targets[key] = value

    query_targets = {}
    slice_fields = _safe_optional_array(payload, "slice_fields", index)
    if slice_fields is not None:
        query_targets["slice_fields"] = slice_fields

    available: dict[str, bool] = {
        "field_targets": field_targets is not None,
        "scalar_targets": scalar_values.size > 0,
        "surface_pressure": "surface_pressure" in surface_targets,
        "surface_cp": "surface_cp" in surface_targets,
        "slice_fields": "slice_fields" in query_targets,
        "pressure_gradient_indicator": "pressure_gradient_indicator" in feature_targets,
        "high_gradient_mask": "high_gradient_mask" in feature_targets,
        "shock_indicator": "shock_indicator" in feature_targets,
        "shock_location": "shock_location" in feature_targets,
        "surface_heat_flux": "surface_heat_flux" in surface_targets,
        "surface_wall_shear": "surface_wall_shear" in surface_targets,
    }
    derived = {
        "surface_cp": True,
        "slice_fields": True,
        "pressure_gradient_indicator": True,
        "high_gradient_mask": True,
        "shock_indicator": True,
        "shock_location": True,
        "surface_heat_flux": True,
        "surface_wall_shear": True,
    }
    trainable = {
        "field_targets": True,
        "scalar_targets": True,
        "surface_pressure": True,
        "surface_cp": True,
        "slice_fields": True,
        "pressure_gradient_indicator": True,
        "high_gradient_mask": True,
        "shock_indicator": False,
        "shock_location": False,
        "surface_heat_flux": False,
        "surface_wall_shear": False,
    }

    metadata = SampleMetadata(
        dataset_name=dataset_name,
        case_id=str(payload["airfoil_id"][index]),
        split=split,
        source_format=source_format,
        dimensionality="3d" if query_points.shape[-1] == 3 else "2d",
        mesh_type="point_cloud",
    )
    geometry = GeometryData(
        geometry_type=str(payload.get("geometry_representation", np.asarray(["unknown"]))[index]),
        geometry_repr=np.asarray(payload["geometry_params"][index], dtype=np.float32).reshape(-1),
        geometry_features={"branch_inputs": np.asarray(payload["branch_inputs"][index], dtype=np.float32).reshape(-1)},
        surface_points=surface_points,
        surface_normals=_safe_optional_array(payload, "surface_normals", index),
        geometry_mode=str(payload.get("geometry_mode", np.asarray(["unknown"]))[index]),
        geometry_source=str(payload.get("geometry_source", np.asarray(["unknown"]))[index]),
        geometry_representation=str(payload.get("geometry_representation", np.asarray(["unknown"]))[index]),
        branch_encoding_type=str(payload.get("branch_encoding_type", np.asarray(["unknown"]))[index]),
        geometry_reconstructability=str(payload.get("geometry_reconstructability", np.asarray(["unknown"]))[index]),
        geometry_params_semantics=str(payload.get("geometry_params_semantics", np.asarray(["unknown"]))[index]),
        legacy_param_source=str(payload.get("legacy_param_source", np.asarray(["none"]))[index]),
        geometry_encoding_meta=str(payload.get("geometry_encoding_meta", np.asarray([""]))[index]) or None,
        surface_sampling_info=str(payload.get("surface_sampling_info", np.asarray([""]))[index]) or None,
    )
    flow_conditions = np.asarray(payload["flow_conditions"][index], dtype=np.float32).reshape(-1)
    condition = FlowCondition(
        reynolds=_safe_float(flow_conditions[2:3]) if flow_conditions.size >= 3 else None,
        mach=_safe_float(flow_conditions[0:1]) if flow_conditions.size >= 1 else None,
        aoa=_safe_float(flow_conditions[1:2]) if flow_conditions.size >= 2 else None,
    )
    query_sets = {
        "field_query_points": QuerySet(name="field_query_points", points=query_points, point_type="field"),
        "surface_query_points": QuerySet(name="surface_query_points", points=surface_query_points, point_type="surface"),
        "slice_query_points": QuerySet(name="slice_query_points", points=slice_points, point_type="slice"),
    }
    return CFDSurrogateSample(
        metadata=metadata,
        geometry=geometry,
        condition=condition,
        query_sets=query_sets,
        targets=TargetBundle(
            field_targets=field_targets,
            surface_targets=surface_targets,
            scalar_targets=scalar_targets,
            feature_targets=feature_targets,
            query_targets=query_targets,
        ),
        availability=AvailabilityBundle(available=available, derived=derived, trainable=trainable),
        legacy_payload={
            "branch_inputs": np.asarray(payload["branch_inputs"][index], dtype=np.float32),
            "farfield_targets": np.asarray(payload["farfield_targets"][index], dtype=np.float32),
            "cp_reference": np.asarray(payload["cp_reference"][index], dtype=np.float32),
            "source": payload["source"][index],
            "fidelity_level": int(payload["fidelity_level"][index]),
            "convergence_flag": int(payload["convergence_flag"][index]),
        },
    )
