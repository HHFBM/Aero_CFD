"""Geometry semantics helpers for 2D airfoil inputs.

This layer does not change the model input tensor layout. It only makes the
geometry source/representation explicit so datasets, inference and validation
can reason about which inputs are safe to reconstruct and which are only
metadata or branch-encoding surrogates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GeometrySemantics:
    geometry_source: str
    geometry_representation: str
    branch_encoding_type: str
    geometry_reconstructability: str
    geometry_mode: str
    geometry_params_semantics: str
    legacy_param_source: str = "none"
    notes: str = ""

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    def as_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True)


def synthetic_geometry_semantics(branch_feature_mode: str) -> GeometrySemantics:
    encoding = "naca_parameter_vector_plus_flow"
    notes = "Synthetic data stores a NACA-like parameter vector."
    if branch_feature_mode == "points":
        encoding = "naca_parameter_vector_plus_flow_plus_surface_signature"
        notes += " branch_feature_mode='points' appends a sampled surface signature."
    return GeometrySemantics(
        geometry_source="synthetic_generator",
        geometry_representation="parameterized_geometry",
        branch_encoding_type=encoding,
        geometry_reconstructability="safe_from_geometry_params",
        geometry_mode="legacy_naca_params",
        geometry_params_semantics="naca4_parameter_vector",
        legacy_param_source="naca4_parameter_vector",
        notes=notes,
    )


def airfrans_geometry_semantics(include_reynolds: bool) -> GeometrySemantics:
    return GeometrySemantics(
        geometry_source="airfrans_simulation_name",
        geometry_representation="parameterized_geometry",
        branch_encoding_type=(
            "structured_parameter_vector_plus_flow_with_reynolds"
            if include_reynolds
            else "structured_parameter_vector_plus_flow"
        ),
        geometry_reconstructability="metadata_only",
        geometry_mode="structured_param_vector",
        geometry_params_semantics="airfrans_structured_geometry_params",
        legacy_param_source="airfrans_simulation_name",
        notes=(
            "AirfRANS parsed geometry_params are structured metadata from the simulation name; "
            "they should not be assumed to be equivalent to the legacy NACA4 predictor interface."
        ),
    )


def airfrans_original_geometry_semantics() -> GeometrySemantics:
    return GeometrySemantics(
        geometry_source="airfrans_raw_surface_sampling",
        geometry_representation="geometry_summary",
        branch_encoding_type="normalized_surface_signature_plus_flow",
        geometry_reconstructability="surface_points_only",
        geometry_mode="generic_surface_points",
        geometry_params_semantics="normalized_geometry_summary",
        legacy_param_source="none",
        notes=(
            "geometry_params are a compact summary of normalized surface points, "
            "not a parameterized geometry that can be strictly reconstructed."
        ),
    )


def infer_payload_geometry_semantics(payload: dict[str, Any], branch_feature_mode: str = "params") -> GeometrySemantics:
    source_values = payload.get("source")
    first_source = ""
    if source_values is not None and len(source_values) > 0:
        first_source = str(source_values[0])
    if first_source.startswith("airfrans_original:"):
        return airfrans_original_geometry_semantics()
    if first_source.startswith("airfrans:"):
        return airfrans_geometry_semantics(include_reynolds=np.asarray(payload.get("flow_conditions")).shape[-1] > 2)
    return synthetic_geometry_semantics(branch_feature_mode=branch_feature_mode)


def _repeat_string(value: str, count: int) -> np.ndarray:
    return np.asarray([value] * count)


def ensure_geometry_payload_metadata(payload: dict[str, Any], branch_feature_mode: str = "params") -> dict[str, Any]:
    """Add geometry metadata fields without changing legacy numeric payloads."""

    if "airfoil_id" not in payload:
        return payload

    semantics = infer_payload_geometry_semantics(payload, branch_feature_mode=branch_feature_mode)
    num_samples = int(len(payload["airfoil_id"]))
    geometry_points = np.asarray(payload.get("geometry_points", payload.get("surface_points", np.zeros((num_samples, 0, 2), dtype=np.float32))), dtype=np.float32)
    payload.setdefault("geometry_points", geometry_points)
    payload.setdefault("geometry_mode", _repeat_string(semantics.geometry_mode, num_samples))
    payload.setdefault("geometry_source", _repeat_string(semantics.geometry_source, num_samples))
    payload.setdefault("geometry_representation", _repeat_string(semantics.geometry_representation, num_samples))
    payload.setdefault("branch_encoding_type", _repeat_string(semantics.branch_encoding_type, num_samples))
    payload.setdefault("geometry_reconstructability", _repeat_string(semantics.geometry_reconstructability, num_samples))
    payload.setdefault("geometry_params_semantics", _repeat_string(semantics.geometry_params_semantics, num_samples))
    payload.setdefault("legacy_param_source", _repeat_string(semantics.legacy_param_source, num_samples))
    payload.setdefault("geometry_encoding_meta", _repeat_string(semantics.as_json(), num_samples))
    payload.setdefault(
        "surface_sampling_info",
        _repeat_string(
            json.dumps(
                {
                    "geometry_points_field": "geometry_points",
                    "surface_points_field": "surface_points",
                    "notes": "geometry_points defaults to the stored 2D airfoil contour when no separate geometry field exists.",
                },
                sort_keys=True,
            ),
            num_samples,
        ),
    )
    return payload


def build_inference_geometry_semantics(
    geometry_mode: str,
    geometry_representation: str,
    branch_encoding_type: str,
    geometry_reconstructability: str,
    notes: str,
    geometry_params_semantics: str = "runtime_input",
    geometry_source: str = "runtime_input",
    legacy_param_source: str = "runtime_input",
) -> dict[str, str]:
    return GeometrySemantics(
        geometry_source=geometry_source,
        geometry_representation=geometry_representation,
        branch_encoding_type=branch_encoding_type,
        geometry_reconstructability=geometry_reconstructability,
        geometry_mode=geometry_mode,
        geometry_params_semantics=geometry_params_semantics,
        legacy_param_source=legacy_param_source,
        notes=notes,
    ).as_dict()
