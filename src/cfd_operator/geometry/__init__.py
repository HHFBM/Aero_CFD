"""Airfoil geometry parameterizations."""

from .base import AirfoilParameterization
from .cst import CSTParameterization
from .features import build_branch_features
from .naca import NACA4Airfoil
from .preprocess import CanonicalGeometry2D, GeometryInputError, resolve_geometry_input
from .semantics import GeometrySemantics, ensure_geometry_payload_metadata

__all__ = [
    "AirfoilParameterization",
    "CSTParameterization",
    "NACA4Airfoil",
    "CanonicalGeometry2D",
    "GeometryInputError",
    "GeometrySemantics",
    "build_branch_features",
    "ensure_geometry_payload_metadata",
    "resolve_geometry_input",
]
