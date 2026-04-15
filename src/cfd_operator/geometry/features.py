"""Geometry feature encoders.

`branch_feature_mode` only governs parameterized geometry paths that call these
helpers directly. Some dataset converters, such as the raw AirfRANS path, store
their own branch encoding and bypass this helper by design.
"""

from typing import Optional
import numpy as np

from .base import AirfoilParameterization
from .preprocess import canonicalize_closed_contour


def sample_surface_signature(airfoil: AirfoilParameterization, num_points: int = 32) -> np.ndarray:
    points = airfoil.surface_points(num_points)
    return points.reshape(-1).astype(np.float32)


def sample_surface_signature_from_points(surface_points: np.ndarray, num_points: int = 32) -> np.ndarray:
    points = canonicalize_closed_contour(surface_points, num_points=num_points)
    return points.reshape(-1).astype(np.float32)


def build_branch_features(
    airfoil: AirfoilParameterization,
    mach: float,
    aoa_deg: float,
    reynolds: Optional[float] = None,
    mode: str = "params",
) -> np.ndarray:
    base = airfoil.parameter_vector()
    flow = [mach, aoa_deg]
    if reynolds is not None:
        flow.append(reynolds)
    feature = np.concatenate([base, np.asarray(flow, dtype=np.float32)], axis=0)
    if mode == "params":
        return feature.astype(np.float32)
    if mode == "points":
        signature = sample_surface_signature(airfoil)
        return np.concatenate([feature, signature], axis=0).astype(np.float32)
    raise ValueError(f"Unsupported branch feature mode: {mode}")
