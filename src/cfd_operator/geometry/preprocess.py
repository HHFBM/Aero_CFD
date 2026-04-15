"""Geometry preprocessing helpers for 2D airfoil inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np

from .naca import NACA4Airfoil


class GeometryInputError(ValueError):
    """Raised when a geometry input payload is incomplete or malformed."""


@dataclass
class CanonicalGeometry2D:
    geometry_mode: str
    raw_surface_points: np.ndarray
    canonical_surface_points: np.ndarray
    normalized_surface_points: np.ndarray
    geometry_params: np.ndarray | None
    geometry_params_semantics: str
    reconstructability: str
    adapter_note: str = ""
    airfoil: NACA4Airfoil | None = None


def normalize_surface_points(surface_points: np.ndarray) -> np.ndarray:
    points = np.asarray(surface_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise GeometryInputError("surface points must have shape [N, 2].")
    x = points[:, 0]
    y = points[:, 1]
    chord = max(float(x.max() - x.min()), 1.0e-6)
    x_norm = (x - float(x.min())) / chord
    y_center = 0.5 * (float(y.max()) + float(y.min()))
    y_norm = (y - y_center) / chord
    return np.stack([x_norm, y_norm], axis=1).astype(np.float32)


def _validate_points(points: np.ndarray, field_name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise GeometryInputError(f"{field_name} must have shape [N, 2].")
    if array.shape[0] < 4:
        raise GeometryInputError(f"{field_name} must contain at least 4 points.")
    if not np.isfinite(array).all():
        raise GeometryInputError(f"{field_name} contains non-finite coordinates.")
    return array


def _close_if_needed(points: np.ndarray) -> np.ndarray:
    if np.linalg.norm(points[0] - points[-1]) < 1.0e-6:
        return points[:-1]
    return points


def _resample_closed_contour(points: np.ndarray, num_points: int) -> np.ndarray:
    points = _close_if_needed(points)
    if points.shape[0] == num_points:
        return points.astype(np.float32)
    extended = np.concatenate([points, points[:1]], axis=0)
    segment = np.linalg.norm(np.diff(extended, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    total = float(cumulative[-1])
    if total < 1.0e-6:
        raise GeometryInputError("surface contour perimeter is too small to resample.")
    samples = np.linspace(0.0, total, num=num_points + 1, dtype=np.float64)[:-1]
    x = np.interp(samples, cumulative, extended[:, 0])
    y = np.interp(samples, cumulative, extended[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def _signed_area(points: np.ndarray) -> float:
    rolled = np.roll(points, -1, axis=0)
    return 0.5 * float(np.sum(points[:, 0] * rolled[:, 1] - rolled[:, 0] * points[:, 1]))


def _rotate_to_trailing_edge(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    trailing_candidates = np.where(np.abs(x - x.max()) < 1.0e-5)[0]
    if trailing_candidates.size == 0:
        start_idx = int(np.argmax(x))
    else:
        y = points[trailing_candidates, 1]
        start_idx = int(trailing_candidates[np.argmin(np.abs(y))])
    return np.roll(points, -start_idx, axis=0)


def _enforce_upper_first_orientation(points: np.ndarray) -> np.ndarray:
    points = _rotate_to_trailing_edge(points)
    leading_idx = int(np.argmin(points[:, 0]))
    upper_segment = points[: leading_idx + 1]
    lower_segment = points[leading_idx:]
    if upper_segment.size == 0 or lower_segment.size == 0:
        return points
    if float(np.mean(upper_segment[:, 1])) < float(np.mean(lower_segment[:, 1])):
        reversed_points = np.flip(points, axis=0)
        return _rotate_to_trailing_edge(reversed_points)
    return points


def canonicalize_closed_contour(surface_points: np.ndarray, num_points: int) -> np.ndarray:
    points = _validate_points(surface_points, "geometry_points")
    points = _close_if_needed(points)
    points = _resample_closed_contour(points, num_points=num_points)
    if _signed_area(points) < 0.0:
        points = np.flip(points, axis=0)
    return _enforce_upper_first_orientation(points).astype(np.float32)


def build_closed_contour_from_upper_lower(
    upper_surface_points: np.ndarray,
    lower_surface_points: np.ndarray,
    num_points: int,
) -> np.ndarray:
    upper = _validate_points(upper_surface_points, "upper_surface_points")
    lower = _validate_points(lower_surface_points, "lower_surface_points")
    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]
    contour = np.concatenate([upper[::-1], lower[1:]], axis=0)
    return canonicalize_closed_contour(contour, num_points=num_points)


def estimate_parameter_vector_from_surface_points(surface_points: np.ndarray) -> np.ndarray:
    """Return a NACA-like summary adapter for legacy parameter branches.

    This is intentionally a geometric summary, not a strict inverse map from a
    generic airfoil contour back to a true NACA parameterization.
    """

    normalized = normalize_surface_points(surface_points)
    points = _enforce_upper_first_orientation(normalized)
    leading_idx = int(np.argmin(points[:, 0]))
    upper = points[: leading_idx + 1]
    lower = points[leading_idx:]
    if upper.shape[0] < 2 or lower.shape[0] < 2:
        raise GeometryInputError("Could not split the contour into upper/lower surfaces for parameter summary estimation.")
    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]
    grid = np.linspace(0.0, 1.0, num=128, dtype=np.float32)
    upper_y = np.interp(grid, upper[:, 0], upper[:, 1])
    lower_y = np.interp(grid, lower[:, 0], lower[:, 1])
    camber = 0.5 * (upper_y + lower_y)
    thickness = upper_y - lower_y
    max_camber_index = int(np.argmax(np.abs(camber)))
    max_camber = float(np.clip(np.abs(camber[max_camber_index]), 0.0, 0.09))
    camber_position = float(np.clip(grid[max_camber_index], 0.05, 0.9))
    max_thickness = float(np.clip(np.max(thickness), 0.04, 0.24))
    return np.asarray([max_camber, camber_position, max_thickness, 1.0], dtype=np.float32)


def resolve_geometry_input(
    geometry_mode: Optional[str],
    geometry_params: Optional[np.ndarray],
    geometry_points: Optional[np.ndarray],
    upper_surface_points: Optional[np.ndarray],
    lower_surface_points: Optional[np.ndarray],
    num_points: int,
) -> CanonicalGeometry2D:
    if geometry_mode is None:
        if geometry_points is not None or (upper_surface_points is not None and lower_surface_points is not None):
            geometry_mode = "generic_surface_points"
        elif geometry_params is not None:
            geometry_mode = "legacy_naca_params"
        else:
            raise GeometryInputError(
                "Geometry input is missing. Provide geometry_mode with either geometry_params or geometry_points."
            )

    if geometry_mode == "legacy_naca_params":
        if geometry_params is None:
            raise GeometryInputError("geometry_mode='legacy_naca_params' requires geometry_params.")
        params = np.asarray(geometry_params, dtype=np.float32).reshape(-1)
        if params.shape[0] < 3:
            raise GeometryInputError(
                "geometry_params for legacy_naca_params must contain at least [max_camber, camber_position, thickness]."
            )
        chord = float(params[3]) if params.shape[0] > 3 else 1.0
        airfoil = NACA4Airfoil(
            max_camber=float(params[0]),
            camber_position=float(params[1]),
            thickness=float(params[2]),
            chord=chord,
        )
        raw_surface_points = airfoil.surface_points(num_points)
        normalized = normalize_surface_points(raw_surface_points)
        return CanonicalGeometry2D(
            geometry_mode=geometry_mode,
            raw_surface_points=raw_surface_points.astype(np.float32),
            canonical_surface_points=raw_surface_points.astype(np.float32),
            normalized_surface_points=normalized,
            geometry_params=airfoil.parameter_vector(),
            geometry_params_semantics="naca4_parameter_vector",
            reconstructability="safe_from_geometry_params",
            airfoil=airfoil,
        )

    if geometry_mode == "structured_param_vector":
        if geometry_params is None:
            raise GeometryInputError("geometry_mode='structured_param_vector' requires geometry_params.")
        params = np.asarray(geometry_params, dtype=np.float32).reshape(-1)
        if params.shape[0] < 4:
            raise GeometryInputError("structured_param_vector requires at least 4 geometry parameters.")
        if geometry_points is None:
            raise GeometryInputError(
                "structured_param_vector checkpoints still require geometry_points for surface querying because "
                "the project does not provide a strict generic reconstruction path for these parameters yet."
            )
        canonical = canonicalize_closed_contour(geometry_points, num_points=num_points)
        return CanonicalGeometry2D(
            geometry_mode=geometry_mode,
            raw_surface_points=np.asarray(geometry_points, dtype=np.float32),
            canonical_surface_points=canonical,
            normalized_surface_points=normalize_surface_points(canonical),
            geometry_params=params,
            geometry_params_semantics="structured_parameter_vector",
            reconstructability="metadata_only",
            adapter_note=(
                "structured_param_vector preserves the provided geometry_params as metadata only; "
                "generic reconstruction is not solved for this checkpoint family."
            ),
        )

    if geometry_mode != "generic_surface_points":
        raise GeometryInputError(
            "Unsupported geometry_mode. Expected one of: legacy_naca_params, generic_surface_points, structured_param_vector."
        )

    if geometry_points is not None:
        raw_surface_points = _validate_points(geometry_points, "geometry_points")
        canonical = canonicalize_closed_contour(raw_surface_points, num_points=num_points)
    elif upper_surface_points is not None and lower_surface_points is not None:
        canonical = build_closed_contour_from_upper_lower(
            upper_surface_points=upper_surface_points,
            lower_surface_points=lower_surface_points,
            num_points=num_points,
        )
        raw_surface_points = canonical
    else:
        raise GeometryInputError(
            "geometry_mode='generic_surface_points' requires geometry_points, or both upper_surface_points and lower_surface_points."
        )
    return CanonicalGeometry2D(
        geometry_mode=geometry_mode,
        raw_surface_points=np.asarray(raw_surface_points, dtype=np.float32),
        canonical_surface_points=canonical.astype(np.float32),
        normalized_surface_points=normalize_surface_points(canonical),
        geometry_params=estimate_parameter_vector_from_surface_points(canonical),
        geometry_params_semantics="derived_geometry_summary",
        reconstructability="surface_points_only",
        adapter_note=(
            "Generic surface points were canonicalized and summarized for compatibility; "
            "this is not a strict inverse reconstruction to a named parametric family."
        ),
    )


def maybe_warn_geometry_adapter(message: str) -> None:
    if message:
        warnings.warn(message, stacklevel=2)
