from __future__ import annotations

import numpy as np

from cfd_operator.postprocess import (
    compute_gradient_indicators,
    compute_surface_cp,
    compute_surface_heat_flux,
    compute_surface_pressure,
    compute_wall_shear,
    estimate_shock_location,
    extract_slice_field,
    resolve_pressure_channel,
    resolve_surface_cp,
)


def test_postprocess_functions_run() -> None:
    surface_points = np.asarray([[0.0, 0.0], [0.3, 0.05], [0.6, 0.02], [1.0, 0.0]], dtype=np.float32)
    surface_pressure = np.asarray([[1.1], [0.9], [0.95], [1.02]], dtype=np.float32)
    surface_fields = np.asarray(
        [[1.0, 0.0, 1.1, 1.0], [1.1, 0.1, 0.9, 1.0], [0.9, -0.1, 0.95, 1.0], [1.0, 0.0, 1.02, 1.0]],
        dtype=np.float32,
    )
    cp = compute_surface_cp(surface_pressure=surface_pressure, cp_reference=np.asarray([1.0, 0.2], dtype=np.float32))
    pressure = compute_surface_pressure(surface_fields=surface_fields)
    heat_flux = compute_surface_heat_flux(surface_points, pressure)
    wall_shear = compute_wall_shear(surface_points, surface_fields)

    assert cp.shape == (4, 1)
    assert pressure.shape == (4, 1)
    assert heat_flux.shape == (4, 1)
    assert wall_shear.shape == (4, 1)


def test_pressure_channel_resolution_helpers() -> None:
    cp_like = np.asarray([[0.5], [-0.25]], dtype=np.float32)
    cp_reference = np.asarray([1.0, 2.0], dtype=np.float32)
    pressure = resolve_pressure_channel(cp_like, pressure_target_mode="cp_like", cp_reference=cp_reference)
    cp = resolve_surface_cp(cp_like, pressure_target_mode="cp_like", cp_reference=cp_reference)

    np.testing.assert_allclose(pressure, np.asarray([[2.0], [0.5]], dtype=np.float32))
    np.testing.assert_allclose(cp, cp_like)


def test_slice_and_feature_postprocess() -> None:
    points = np.asarray(
        [[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [0.5, 0.2], [0.5, -0.2]],
        dtype=np.float32,
    )
    fields = np.asarray(
        [[1.0, 0.0, 1.0, 1.0], [1.1, 0.0, 1.2, 1.0], [1.2, 0.0, 1.8, 1.0], [1.0, 0.0, 1.1, 1.0], [1.0, 0.0, 1.0, 1.0], [1.0, 0.1, 1.6, 1.0], [1.0, -0.1, 1.5, 1.0]],
        dtype=np.float32,
    )
    extracted = extract_slice_field(points, fields, {"type": "y_const", "value": 0.0, "x_min": -0.5, "x_max": 1.5, "num_points": 8})
    indicators = compute_gradient_indicators(points, fields)
    summary = estimate_shock_location(points, indicators["shock_indicator"], indicators["gradient_magnitude"])

    assert extracted["slice_points"].shape == (8, 2)
    assert extracted["slice_fields"].shape == (8, 4)
    assert indicators["high_gradient_mask"].shape == (7, 1)
    assert "count" in summary
