"""Airfoil geometry parameterizations."""

from .base import AirfoilParameterization
from .branch_adapter import (
    BranchInputAdapter,
    BranchInputMode,
    GeometryEncoder,
    GeometryFeatureBuilder,
)
from .cst import CSTParameterization
from .features import build_branch_features, build_branch_features_from_surface_points
from .naca import NACA4Airfoil
from .preprocess import CanonicalGeometry2D, GeometryInputError, resolve_geometry_input
from .semantics import GeometrySemantics, ensure_geometry_payload_metadata

__all__ = [
    "AirfoilParameterization",
    "BranchInputAdapter",
    "BranchInputMode",
    "CSTParameterization",
    "NACA4Airfoil",
    "CanonicalGeometry2D",
    "GeometryEncoder",
    "GeometryFeatureBuilder",
    "GeometryInputError",
    "GeometrySemantics",
    "build_branch_features",
    "build_branch_features_from_surface_points",
    "ensure_geometry_payload_metadata",
    "resolve_geometry_input",
]
