"""Geometry feature encoders."""

import numpy as np
from typing import Optional

from .base import AirfoilParameterization


def sample_surface_signature(airfoil: AirfoilParameterization, num_points: int = 32) -> np.ndarray:
    points = airfoil.surface_points(num_points)
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
