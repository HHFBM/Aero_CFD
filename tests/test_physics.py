from __future__ import annotations

import torch

from cfd_operator.physics import compressible_euler_residuals


def test_physics_residual_shapes() -> None:
    coords = torch.randn(2, 12, 2, requires_grad=True)
    predicted_fields = torch.randn(2, 12, 4, requires_grad=True)
    residuals = compressible_euler_residuals(predicted_fields=predicted_fields, coords=coords)
    assert residuals["continuity"].shape == (2, 12)
    assert residuals["momentum_x"].shape == (2, 12)
    assert residuals["momentum_y"].shape == (2, 12)

