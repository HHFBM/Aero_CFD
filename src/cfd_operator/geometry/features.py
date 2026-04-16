"""Geometry feature encoders.

Legacy fixed-dimension branch feature construction is centralized here via the
shared GeometryFeatureBuilder compatibility layer.
"""

from typing import Optional
import numpy as np

from .base import AirfoilParameterization
from .branch_adapter import GeometryFeatureBuilder, sample_surface_signature, sample_surface_signature_from_points


def build_branch_features_from_surface_points(
    surface_points: np.ndarray,
    mach: float,
    aoa_deg: float,
    reynolds: Optional[float] = None,
    mode: str = "params",
    signature_points: int = 32,
) -> np.ndarray:
    builder = GeometryFeatureBuilder(branch_feature_mode=mode, signature_points=signature_points)
    return builder.build_from_surface_points(
        surface_points,
        mach=mach,
        aoa_deg=aoa_deg,
        reynolds=reynolds,
    )


def build_branch_features(
    airfoil: AirfoilParameterization,
    mach: float,
    aoa_deg: float,
    reynolds: Optional[float] = None,
    mode: str = "params",
) -> np.ndarray:
    builder = GeometryFeatureBuilder(branch_feature_mode=mode)
    return builder.build_from_airfoil(
        airfoil,
        mach=mach,
        aoa_deg=aoa_deg,
        reynolds=reynolds,
    )
