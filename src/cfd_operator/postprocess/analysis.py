"""Postprocessing helpers for analysis-oriented surrogate outputs.

The functions in this module intentionally separate:
- physically meaningful outputs that can be derived from predicted pressure/velocity
- approximate placeholders used to keep the engineering pipeline complete

Current limitations are documented in function docstrings and mirrored in README.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - fallback for environments without scipy
    cKDTree = None


def compute_surface_cp(
    surface_pressure: Optional[np.ndarray] = None,
    cp_reference: Optional[np.ndarray] = None,
    mach: Optional[float] = None,
    gamma: float = 1.4,
    p_inf: float = 1.0,
) -> np.ndarray:
    """Compute surface Cp from raw surface pressure.

    Preferred path uses ``cp_reference = [p_ref, q_ref]``. If unavailable, a Mach-based
    reference is used. This is a standard nondimensional pressure conversion.
    """

    if surface_pressure is None:
        raise ValueError("surface_pressure is required to compute surface Cp.")
    pressure = np.asarray(surface_pressure, dtype=np.float32).reshape(-1, 1)
    if cp_reference is not None:
        cp_reference = np.asarray(cp_reference, dtype=np.float32).reshape(-1)
        p_ref = float(cp_reference[0])
        q_ref = max(float(cp_reference[1]), 1.0e-4)
        return ((pressure - p_ref) / q_ref).astype(np.float32)
    if mach is None:
        raise ValueError("Either cp_reference or mach must be provided.")
    q_inf = max(0.5 * gamma * p_inf * mach**2, 1.0e-4)
    return ((pressure - p_inf) / q_inf).astype(np.float32)


def compute_surface_pressure(
    surface_fields: Optional[np.ndarray] = None,
    pressure_surface: Optional[np.ndarray] = None,
    surface_cp: Optional[np.ndarray] = None,
    cp_reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return raw surface pressure.

    If pressure is not explicitly available but Cp and reference pressure are known,
    reconstruct pressure using ``p = p_ref + Cp * q_ref``.
    """

    if pressure_surface is not None:
        return np.asarray(pressure_surface, dtype=np.float32).reshape(-1, 1)
    if surface_fields is not None:
        return np.asarray(surface_fields, dtype=np.float32)[..., 2:3]
    if surface_cp is not None and cp_reference is not None:
        cp_reference = np.asarray(cp_reference, dtype=np.float32).reshape(-1)
        p_ref = float(cp_reference[0])
        q_ref = max(float(cp_reference[1]), 1.0e-4)
        return (p_ref + np.asarray(surface_cp, dtype=np.float32).reshape(-1, 1) * q_ref).astype(np.float32)
    raise ValueError("Insufficient inputs to compute surface pressure.")


def resolve_pressure_channel(
    pressure_channel_values: np.ndarray,
    pressure_target_mode: str,
    cp_reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Resolve the internal pressure-like channel to exported raw pressure.

    ``pressure_target_mode='raw'`` means values are already raw pressure.
    ``pressure_target_mode='cp_like'`` means values store (p - p_ref) / q_ref and need
    ``cp_reference`` to reconstruct raw pressure.
    """

    values = np.asarray(pressure_channel_values, dtype=np.float32)
    if pressure_target_mode == "raw":
        return values.astype(np.float32)
    if cp_reference is None:
        raise ValueError("cp_reference is required to reconstruct raw pressure from cp_like pressure values.")
    return compute_surface_pressure(surface_cp=values, cp_reference=cp_reference)


def resolve_surface_cp(
    pressure_channel_values: np.ndarray,
    pressure_target_mode: str,
    cp_reference: np.ndarray,
) -> np.ndarray:
    """Resolve the internal pressure-like channel to surface Cp."""

    values = np.asarray(pressure_channel_values, dtype=np.float32)
    if pressure_target_mode == "cp_like":
        return values.astype(np.float32)
    return compute_surface_cp(surface_pressure=values, cp_reference=cp_reference)


def compute_surface_heat_flux(
    surface_points: np.ndarray,
    surface_pressure: np.ndarray,
    gamma: float = 1.4,
    p_inf: float = 1.0,
    t_inf: float = 1.0,
    conductivity: float = 0.02,
) -> np.ndarray:
    """Approximate wall heat flux proxy along the surface.

    This is a placeholder proxy, not a high-fidelity CFD wall heat flux model.
    We infer an isentropic temperature-like quantity from pressure and compute a
    tangential gradient magnitude. It is useful for qualitative analysis/export only.
    """

    points = np.asarray(surface_points, dtype=np.float32)
    pressure = np.asarray(surface_pressure, dtype=np.float32).reshape(-1)
    arc = _arc_length(points)
    pressure_ratio = np.clip(pressure / max(p_inf, 1.0e-4), 1.0e-4, None)
    temp_ratio = np.clip(pressure_ratio ** ((gamma - 1.0) / gamma), 0.2, 5.0)
    temperature = t_inf * temp_ratio
    grad_t = np.gradient(temperature, np.maximum(arc, 1.0e-6), edge_order=1)
    return (-conductivity * np.abs(grad_t)).reshape(-1, 1).astype(np.float32)


def compute_heat_flux(*args: Any, **kwargs: Any) -> np.ndarray:
    """Placeholder heat-flux interface.

    AirfRANS does not provide wall heat-flux supervision. This function keeps the
    interface available and delegates to the same approximate proxy used for
    surface-level qualitative export.
    """

    return compute_surface_heat_flux(*args, **kwargs)


def compute_wall_shear(
    surface_points: np.ndarray,
    surface_fields: np.ndarray,
    dynamic_viscosity: float = 1.8e-5,
) -> np.ndarray:
    """Approximate wall shear proxy from tangential velocity variation.

    This is a placeholder shear estimate based on tangential velocity changes along
    the surface. It is not a resolved viscous wall-shear model.
    """

    points = np.asarray(surface_points, dtype=np.float32)
    fields = np.asarray(surface_fields, dtype=np.float32)
    velocity = fields[..., :2]
    tangent = _surface_tangent(points)
    tangential_velocity = np.sum(velocity * tangent, axis=-1)
    arc = _arc_length(points)
    dv_ds = np.gradient(tangential_velocity, np.maximum(arc, 1.0e-6), edge_order=1)
    return (dynamic_viscosity * np.abs(dv_ds)).reshape(-1, 1).astype(np.float32)


def build_slice_points(slice_definition: Dict[str, Any], num_points: int = 128) -> np.ndarray:
    """Construct 2D sample points for a slice definition.

    Supported definitions:
    - {"type": "x_const", "value": 0.4, "y_min": -0.5, "y_max": 0.5}
    - {"type": "y_const", "value": 0.0, "x_min": -0.5, "x_max": 1.5}
    - {"type": "line", "start": [x0, y0], "end": [x1, y1]}
    """

    slice_type = str(slice_definition.get("type", "y_const"))
    num = int(slice_definition.get("num_points", num_points))
    if slice_type == "x_const":
        x_value = float(slice_definition["value"])
        y_min = float(slice_definition.get("y_min", -0.6))
        y_max = float(slice_definition.get("y_max", 0.6))
        y = np.linspace(y_min, y_max, num=num, dtype=np.float32)
        x = np.full_like(y, fill_value=x_value, dtype=np.float32)
        return np.stack([x, y], axis=1)
    if slice_type == "y_const":
        y_value = float(slice_definition["value"])
        x_min = float(slice_definition.get("x_min", -0.5))
        x_max = float(slice_definition.get("x_max", 1.5))
        x = np.linspace(x_min, x_max, num=num, dtype=np.float32)
        y = np.full_like(x, fill_value=y_value, dtype=np.float32)
        return np.stack([x, y], axis=1)
    if slice_type == "line":
        start = np.asarray(slice_definition["start"], dtype=np.float32)
        end = np.asarray(slice_definition["end"], dtype=np.float32)
        alpha = np.linspace(0.0, 1.0, num=num, dtype=np.float32)[:, None]
        return start[None, :] * (1.0 - alpha) + end[None, :] * alpha
    raise ValueError(f"Unsupported slice definition type: {slice_type}")


def extract_slice_field(
    query_points: np.ndarray,
    field_values: np.ndarray,
    slice_definition: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Extract slice fields by interpolating predicted/query-point outputs."""

    slice_points = build_slice_points(slice_definition)
    interpolated = _nearest_interpolate(
        source_points=np.asarray(query_points, dtype=np.float32),
        source_values=np.asarray(field_values, dtype=np.float32),
        target_points=slice_points,
    )
    return {
        "slice_definition": slice_definition,
        "slice_points": slice_points.astype(np.float32),
        "slice_fields": interpolated.astype(np.float32),
    }


def compute_gradient_indicators(
    points: np.ndarray,
    field_values: np.ndarray,
    primary_index: int = 2,
    high_gradient_quantile: float = 0.9,
    indicator_quantile: float = 0.97,
) -> Dict[str, np.ndarray]:
    """Compute gradient-based AirfRANS analysis indicators.

    Primary indicator is pressure-gradient based. A broader high-gradient mask uses a
    lower threshold and is meant for flow-structure analysis, not shock ground truth.
    """

    pts = np.asarray(points, dtype=np.float32)
    fields = np.asarray(field_values, dtype=np.float32)
    primary = fields[:, primary_index]

    if pts.shape[0] < 4:
        gradient_magnitude = np.zeros((pts.shape[0],), dtype=np.float32)
    else:
        gradient_magnitude = _local_gradient_norm(pts, primary)

    high_threshold = float(np.quantile(gradient_magnitude, high_gradient_quantile))
    indicator_threshold = float(np.quantile(gradient_magnitude, indicator_quantile))
    high_gradient_mask = (gradient_magnitude >= high_threshold).astype(np.float32)
    pressure_gradient_indicator = (gradient_magnitude >= indicator_threshold).astype(np.float32)
    summary = {
        "mean_gradient": float(np.mean(gradient_magnitude)),
        "max_gradient": float(np.max(gradient_magnitude, initial=0.0)),
        "high_gradient_fraction": float(np.mean(high_gradient_mask)),
        "pressure_gradient_fraction": float(np.mean(pressure_gradient_indicator)),
    }
    return {
        "gradient_magnitude": gradient_magnitude.reshape(-1, 1).astype(np.float32),
        "high_gradient_mask": high_gradient_mask.reshape(-1, 1).astype(np.float32),
        "pressure_gradient_indicator": pressure_gradient_indicator.reshape(-1, 1).astype(np.float32),
        "high_gradient_region_summary": summary,
        "shock_indicator": pressure_gradient_indicator.reshape(-1, 1).astype(np.float32),
    }


def estimate_shock_location(
    points: np.ndarray,
    shock_indicator: np.ndarray,
    gradient_magnitude: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Estimate a coarse shock summary from indicator points."""

    pts = np.asarray(points, dtype=np.float32)
    indicator = np.asarray(shock_indicator, dtype=np.float32).reshape(-1) > 0.5
    active_points = pts[indicator]
    if active_points.size == 0:
        return {
            "count": 0,
            "centroid": None,
            "x_range": None,
            "y_range": None,
            "peak_gradient": 0.0,
        }

    peak_gradient = 0.0
    if gradient_magnitude is not None:
        peak_gradient = float(np.asarray(gradient_magnitude, dtype=np.float32).reshape(-1)[indicator].max(initial=0.0))

    return {
        "count": int(active_points.shape[0]),
        "centroid": active_points.mean(axis=0).astype(float).tolist(),
        "x_range": [float(active_points[:, 0].min()), float(active_points[:, 0].max())],
        "y_range": [float(active_points[:, 1].min()), float(active_points[:, 1].max())],
        "peak_gradient": peak_gradient,
    }


def _arc_length(points: np.ndarray) -> np.ndarray:
    diffs = np.diff(points, axis=0, prepend=points[:1])
    segment = np.linalg.norm(diffs, axis=1)
    return np.cumsum(segment).astype(np.float32)


def _surface_tangent(points: np.ndarray) -> np.ndarray:
    forward = np.roll(points, -1, axis=0) - points
    backward = points - np.roll(points, 1, axis=0)
    tangent = forward + backward
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    norm = np.where(norm < 1.0e-6, 1.0, norm)
    return (tangent / norm).astype(np.float32)


def _nearest_interpolate(source_points: np.ndarray, source_values: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    if source_points.shape[0] == 0:
        return np.zeros((target_points.shape[0], source_values.shape[-1]), dtype=np.float32)
    if cKDTree is not None:
        _, indices = cKDTree(source_points).query(target_points, k=1)
        return source_values[np.asarray(indices, dtype=np.int64)]
    distances = np.linalg.norm(source_points[None, :, :] - target_points[:, None, :], axis=-1)
    indices = np.argmin(distances, axis=1)
    return source_values[indices]


def _local_gradient_norm(points: np.ndarray, values: np.ndarray, neighbors: int = 6) -> np.ndarray:
    if cKDTree is not None:
        distances, indices = cKDTree(points).query(points, k=min(neighbors, points.shape[0]))
        distances = np.asarray(distances, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int64)
    else:  # pragma: no cover - slow fallback
        deltas = points[:, None, :] - points[None, :, :]
        pairwise = np.linalg.norm(deltas, axis=-1)
        indices = np.argsort(pairwise, axis=1)[:, : min(neighbors, points.shape[0])]
        distances = np.take_along_axis(pairwise, indices, axis=1)

    base = values[:, None]
    neighbor_values = values[indices]
    delta_values = np.abs(neighbor_values - base)
    safe_distance = np.maximum(distances, 1.0e-5)
    local_slopes = delta_values / safe_distance
    return np.mean(local_slopes[:, 1:], axis=1).astype(np.float32)
