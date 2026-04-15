from __future__ import annotations

import numpy as np
import pytest

from cfd_operator.geometry import NACA4Airfoil, resolve_geometry_input
from cfd_operator.geometry.semantics import ensure_geometry_payload_metadata


def test_naca4_surface_points_shape() -> None:
    airfoil = NACA4Airfoil(max_camber=0.02, camber_position=0.4, thickness=0.12)
    points = airfoil.surface_points(101)
    assert points.ndim == 2
    assert points.shape[1] == 2
    assert np.isfinite(points).all()
    assert points[:, 0].min() >= -1.0e-3
    assert points[:, 0].max() <= 1.01


def test_generic_surface_points_preprocess_is_resampling_robust() -> None:
    airfoil = NACA4Airfoil(max_camber=0.02, camber_position=0.4, thickness=0.12)
    dense_points = airfoil.surface_points(101)
    sparse_points = dense_points[::3]

    dense = resolve_geometry_input(
        geometry_mode="generic_surface_points",
        geometry_params=None,
        geometry_points=dense_points,
        upper_surface_points=None,
        lower_surface_points=None,
        num_points=64,
    )
    sparse = resolve_geometry_input(
        geometry_mode="generic_surface_points",
        geometry_params=None,
        geometry_points=sparse_points,
        upper_surface_points=None,
        lower_surface_points=None,
        num_points=64,
    )

    assert dense.canonical_surface_points.shape == (64, 2)
    assert sparse.canonical_surface_points.shape == (64, 2)
    assert np.max(np.abs(dense.normalized_surface_points - sparse.normalized_surface_points)) < 0.15


def test_invalid_generic_geometry_input_has_clear_error() -> None:
    with pytest.raises(ValueError, match="must contain at least 4 points"):
        resolve_geometry_input(
            geometry_mode="generic_surface_points",
            geometry_params=None,
            geometry_points=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.5, 0.1]], dtype=np.float32),
            upper_surface_points=None,
            lower_surface_points=None,
            num_points=32,
        )


def test_legacy_payload_gets_geometry_metadata_fallback() -> None:
    payload = {
        "airfoil_id": np.asarray(["sample-0"]),
        "surface_points": np.zeros((1, 8, 2), dtype=np.float32),
        "source": np.asarray(["synthetic_rule"]),
        "flow_conditions": np.zeros((1, 2), dtype=np.float32),
    }
    enriched = ensure_geometry_payload_metadata(payload)
    assert "geometry_mode" in enriched
    assert "geometry_points" in enriched
    assert enriched["geometry_mode"][0] == "legacy_naca_params"


def test_airfrans_original_metadata_is_not_marked_reconstructable() -> None:
    payload = {
        "airfoil_id": np.asarray(["sample-0"]),
        "surface_points": np.zeros((1, 8, 2), dtype=np.float32),
        "source": np.asarray(["airfrans_original:sample_0001"]),
        "flow_conditions": np.zeros((1, 3), dtype=np.float32),
    }
    enriched = ensure_geometry_payload_metadata(payload)
    assert enriched["geometry_reconstructability"][0] == "surface_points_only"
    assert enriched["geometry_representation"][0] == "geometry_summary"
