"""Shared output/task semantics for training, evaluation, inference and export.

This module keeps output categorization and pressure/Cp semantics centralized so the
rest of the codebase does not need to re-encode the same rules in multiple places.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence


SUPERVISED_TASK = "supervised"
DERIVED_TASK = "derived"
PLACEHOLDER_TASK = "placeholder_or_experimental"


def default_scalar_names() -> tuple[str, ...]:
    return ("cl", "cd", "cdp", "cdv", "clp", "clv", "cm")


def pressure_semantics(pressure_target_mode: str) -> dict[str, str]:
    if pressure_target_mode == "cp_like":
        return {
            "training_pressure_channel": "cp_like_pressure",
            "training_pressure_description": (
                "The internal field target at index 2 is a Cp-like pressure quantity "
                "stored as (p - p_ref) / q_ref."
            ),
            "exported_pressure_channel": "raw_pressure",
            "surface_cp_dependency": "surface pressure is reconstructed from cp_like_pressure and cp_reference",
        }
    return {
        "training_pressure_channel": "raw_pressure",
        "training_pressure_description": "The internal field target at index 2 is raw pressure.",
        "exported_pressure_channel": "raw_pressure",
        "surface_cp_dependency": "surface Cp is derived from raw pressure and cp_reference",
    }


def build_output_semantics(
    field_names: Sequence[str],
    pressure_target_mode: str,
    scalar_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    scalar_names = tuple(scalar_names or default_scalar_names())
    semantics = pressure_semantics(pressure_target_mode)
    aux_name = str(field_names[3]) if len(field_names) > 3 else "aux"

    tasks: dict[str, dict[str, Any]] = {
        "field_outputs": {
            "category": SUPERVISED_TASK,
            "names": list(field_names),
            "description": "Pointwise field supervision targets.",
        },
        "scalar_outputs": {
            "category": SUPERVISED_TASK,
            "names": list(scalar_names[:2]),
            "reserved_names": list(scalar_names[2:]),
            "description": "Global scalar supervision targets.",
        },
        "surface_pressure": {
            "category": SUPERVISED_TASK,
            "description": "Surface pressure target or pressure-equivalent quantity supported by the dataset.",
        },
        "surface_cp": {
            "category": DERIVED_TASK,
            "description": "Derived from raw pressure and Cp reference. When pressure_target_mode=cp_like, the training label already stores a Cp-like pressure quantity.",
        },
        "slice_outputs": {
            "category": DERIVED_TASK,
            "description": "Slice values sampled from predicted pointwise fields.",
        },
        "feature_outputs": {
            "category": DERIVED_TASK,
            "names": ["pressure_gradient_indicator", "high_gradient_mask", "high_gradient_region_summary"],
            "description": "Feature outputs predicted by a feature head when available, otherwise derived from field gradients.",
        },
        "placeholder_outputs": {
            "category": PLACEHOLDER_TASK,
            "names": ["heat_flux_surface", "wall_shear_surface", "shock_indicator", "shock_location", aux_name if aux_name == "rho" else None],
            "description": "Outputs kept for pipeline completeness but not treated as strict supervised benchmark targets.",
        },
    }
    tasks["placeholder_outputs"]["names"] = [name for name in tasks["placeholder_outputs"]["names"] if name is not None]
    return {
        "pressure": semantics,
        "tasks": tasks,
    }


def grouped_output_names(field_names: Sequence[str], scalar_names: Sequence[str] | None = None) -> dict[str, list[str]]:
    scalar_names = tuple(scalar_names or default_scalar_names())
    return {
        "supervised": list(field_names) + list(scalar_names[:2]) + ["surface_pressure"],
        "derived": ["cp_surface", "slice_outputs", "pressure_gradient_indicator", "high_gradient_mask", "high_gradient_region_summary"],
        "placeholder_or_experimental": ["heat_flux_surface", "wall_shear_surface", "shock_indicator", "shock_location"],
    }


def flatten_metric_groups(metric_groups: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for group_metrics in metric_groups.values():
        flat.update(group_metrics)
    return flat


def format_missing_fields_message(context: str, missing_fields: Iterable[str]) -> str:
    missing = sorted(set(str(field) for field in missing_fields))
    return f"{context} is missing required fields: {missing}"
