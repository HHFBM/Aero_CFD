"""Dataset enrichment for analysis-oriented outputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from cfd_operator.output_semantics import format_missing_fields_message
from cfd_operator.postprocess import (
    build_slice_points,
    compute_gradient_indicators,
    compute_surface_cp,
    compute_surface_heat_flux,
    compute_surface_pressure,
    compute_wall_shear,
    estimate_shock_location,
    extract_slice_field,
)


def ensure_analysis_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Fill optional surface/slice/feature fields if they are missing.

    Existing data is preserved. Missing values are derived from existing pressure/field
    outputs or created as explicit placeholders so the rest of the pipeline can rely on
    a stable schema.
    """

    required_core_keys = {
        "airfoil_id",
        "query_points",
        "field_targets",
        "surface_points",
        "cp_reference",
        "surface_cp",
    }
    missing_core_keys = required_core_keys.difference(payload.keys())
    if missing_core_keys:
        raise ValueError(format_missing_fields_message("ensure_analysis_payload input", missing_core_keys))

    num_samples = len(payload["airfoil_id"])
    num_surface_points = int(np.asarray(payload["surface_points"]).shape[1])
    num_query_points = int(np.asarray(payload["query_points"]).shape[1])
    default_aux_name = "nut" if "surface_nut" in payload or "surface_nut_available" in payload else "rho"
    field_names = np.asarray(payload.get("field_names", np.asarray(["u", "v", "p", default_aux_name])))
    payload["field_names"] = field_names
    auxiliary_name = str(field_names[3]) if field_names.shape[0] > 3 else "aux"
    has_nut_labels = auxiliary_name == "nut"

    if "surface_arc_length" not in payload:
        payload["surface_arc_length"] = np.stack(
            [_surface_arc_length(np.asarray(points, dtype=np.float32)) for points in payload["surface_points"]]
        )

    if "surface_pressure" not in payload:
        payload["surface_pressure"] = np.stack(
            [
                compute_surface_pressure(
                    pressure_surface=None,
                    surface_fields=None,
                    surface_cp=np.asarray(payload["surface_cp"][index], dtype=np.float32),
                    cp_reference=np.asarray(payload["cp_reference"][index], dtype=np.float32),
                )
                for index in range(num_samples)
            ]
        )

    if "surface_velocity" not in payload:
        payload["surface_velocity"] = np.stack(
            [
                _nearest_interpolate(
                    np.asarray(payload["query_points"][index], dtype=np.float32),
                    np.asarray(payload["field_targets"][index], dtype=np.float32)[..., :2],
                    np.asarray(payload["surface_points"][index], dtype=np.float32),
                )
                for index in range(num_samples)
            ]
        )

    if "surface_nut" not in payload:
        if has_nut_labels:
            aux_index = 3
            payload["surface_nut"] = np.stack(
                [
                    _nearest_interpolate(
                        np.asarray(payload["query_points"][index], dtype=np.float32),
                        np.asarray(payload["field_targets"][index], dtype=np.float32)[..., aux_index : aux_index + 1],
                        np.asarray(payload["surface_points"][index], dtype=np.float32),
                    )
                    for index in range(num_samples)
                ]
            )
        else:
            payload["surface_nut"] = np.zeros((num_samples, num_surface_points, 1), dtype=np.float32)

    if "surface_heat_flux" not in payload:
        payload["surface_heat_flux"] = np.stack(
            [
                compute_surface_heat_flux(
                    np.asarray(payload["surface_points"][index], dtype=np.float32),
                    np.asarray(payload["surface_pressure"][index], dtype=np.float32),
                )
                for index in range(num_samples)
            ]
        )

    if "surface_wall_shear" not in payload:
        surface_fields = payload.get("surface_field_targets")
        if surface_fields is None:
            surface_fields = np.stack(
                [
                    _nearest_interpolate(
                        np.asarray(payload["query_points"][index], dtype=np.float32),
                        np.asarray(payload["field_targets"][index], dtype=np.float32),
                        np.asarray(payload["surface_points"][index], dtype=np.float32),
                    )
                    for index in range(num_samples)
                ]
            )
        payload["surface_wall_shear"] = np.stack(
            [
                compute_wall_shear(
                    np.asarray(payload["surface_points"][index], dtype=np.float32),
                    np.asarray(surface_fields[index], dtype=np.float32),
                )
                for index in range(num_samples)
            ]
        )

    if "slice_points" not in payload or "slice_fields" not in payload:
        default_slice = {"type": "y_const", "value": 0.0, "x_min": -0.5, "x_max": 1.5, "num_points": 96}
        slice_points = []
        slice_fields = []
        for index in range(num_samples):
            extracted = extract_slice_field(
                query_points=np.asarray(payload["query_points"][index], dtype=np.float32),
                field_values=np.asarray(payload["field_targets"][index], dtype=np.float32),
                slice_definition=default_slice,
            )
            slice_points.append(extracted["slice_points"])
            slice_fields.append(extracted["slice_fields"])
        payload["slice_points"] = np.stack(slice_points)
        payload["slice_fields"] = np.stack(slice_fields)

    if "pressure_gradient_indicator" not in payload or "high_gradient_mask" not in payload:
        pressure_gradient_all = []
        shock_all = []
        high_gradient_all = []
        shock_locations = []
        gradient_magnitudes = []
        region_summaries = []
        for index in range(num_samples):
            indicators = compute_gradient_indicators(
                points=np.asarray(payload["query_points"][index], dtype=np.float32),
                field_values=np.asarray(payload["field_targets"][index], dtype=np.float32),
            )
            summary = estimate_shock_location(
                points=np.asarray(payload["query_points"][index], dtype=np.float32),
                shock_indicator=indicators["shock_indicator"],
                gradient_magnitude=indicators["gradient_magnitude"],
            )
            pressure_gradient_all.append(indicators["pressure_gradient_indicator"])
            shock_all.append(indicators["shock_indicator"])
            high_gradient_all.append(indicators["high_gradient_mask"])
            gradient_magnitudes.append(indicators["gradient_magnitude"])
            region_summaries.append(indicators["high_gradient_region_summary"])
            if summary["centroid"] is None:
                shock_locations.append(np.asarray([np.nan, np.nan], dtype=np.float32))
            else:
                shock_locations.append(np.asarray(summary["centroid"], dtype=np.float32))
        payload["pressure_gradient_indicator"] = np.stack(pressure_gradient_all)
        payload["shock_indicator"] = np.stack(shock_all)
        payload["high_gradient_mask"] = np.stack(high_gradient_all)
        payload["gradient_magnitude"] = np.stack(gradient_magnitudes)
        payload["shock_location"] = np.stack(shock_locations)
        payload["high_gradient_region_summary"] = np.asarray(region_summaries, dtype=object)

    payload.setdefault(
        "surface_heat_flux_available",
        np.zeros((num_samples, num_surface_points), dtype=np.float32),
    )
    payload.setdefault(
        "surface_wall_shear_available",
        np.zeros((num_samples, num_surface_points), dtype=np.float32),
    )
    payload.setdefault(
        "surface_pressure_available",
        np.ones((num_samples, num_surface_points), dtype=np.float32),
    )
    payload.setdefault(
        "surface_velocity_available",
        np.ones((num_samples, num_surface_points), dtype=np.float32),
    )
    payload.setdefault(
        "surface_nut_available",
        np.full((num_samples, num_surface_points), 1.0 if has_nut_labels else 0.0, dtype=np.float32),
    )
    payload.setdefault(
        "slice_available",
        np.ones((num_samples, np.asarray(payload["slice_points"]).shape[1]), dtype=np.float32),
    )
    payload.setdefault(
        "feature_available",
        np.ones((num_samples, num_query_points), dtype=np.float32),
    )
    payload.setdefault(
        "nut_available",
        np.full((num_samples, num_query_points), 1.0 if has_nut_labels else 0.0, dtype=np.float32),
    )
    payload.setdefault(
        "scalar_component_targets",
        np.zeros((num_samples, 5), dtype=np.float32),
    )
    payload.setdefault(
        "scalar_component_available",
        np.zeros((num_samples, 5), dtype=np.float32),
    )
    payload.setdefault(
        "shock_location_available",
        np.isfinite(np.asarray(payload["shock_location"])[:, 0]).astype(np.float32),
    )
    return payload


def _surface_arc_length(points: np.ndarray) -> np.ndarray:
    delta = np.diff(points, axis=0, prepend=points[:1])
    segment = np.linalg.norm(delta, axis=1, keepdims=True)
    return np.cumsum(segment, axis=0).astype(np.float32)


def _nearest_interpolate(source_points: np.ndarray, source_values: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(source_points[None, :, :] - target_points[:, None, :], axis=-1)
    indices = np.argmin(distances, axis=1)
    return source_values[indices].astype(np.float32)
