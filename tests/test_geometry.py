from __future__ import annotations

import numpy as np

from cfd_operator.geometry import NACA4Airfoil


def test_naca4_surface_points_shape() -> None:
    airfoil = NACA4Airfoil(max_camber=0.02, camber_position=0.4, thickness=0.12)
    points = airfoil.surface_points(101)
    assert points.ndim == 2
    assert points.shape[1] == 2
    assert np.isfinite(points).all()
    assert points[:, 0].min() >= -1.0e-3
    assert points[:, 0].max() <= 1.01

