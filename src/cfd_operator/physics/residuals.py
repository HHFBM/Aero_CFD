"""Autograd-based residuals for simplified steady 2D compressible Euler flow.

Assumptions:
- 2D steady external flow
- Inviscid approximation
- Predictions are pointwise fields [u, v, p, rho]
- Coordinates may be normalized before entering the model; derivative scaling is
  corrected using the coordinate normalizer standard deviation
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def compute_gradients(
    values: torch.Tensor,
    coords: torch.Tensor,
    coord_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return gradients d(values)/d(x,y) with shape [B, N, 2]."""

    gradients = torch.autograd.grad(
        outputs=values,
        inputs=coords,
        grad_outputs=torch.ones_like(values),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]
    if gradients is None:
        gradients = torch.zeros_like(coords)
    if coord_scale is not None:
        gradients = gradients / coord_scale.view(1, 1, -1)
    return gradients


def continuity_residual(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    coords: torch.Tensor,
    coord_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    rho_u = rho * u
    rho_v = rho * v
    grad_rho_u = compute_gradients(rho_u, coords, coord_scale=coord_scale)
    grad_rho_v = compute_gradients(rho_v, coords, coord_scale=coord_scale)
    return grad_rho_u[..., 0] + grad_rho_v[..., 1]


def momentum_residual(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    coords: torch.Tensor,
    coord_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rho_u2_plus_p = rho * u * u + p
    rho_v2_plus_p = rho * v * v + p
    rho_uv = rho * u * v

    grad_x = compute_gradients(rho_u2_plus_p, coords, coord_scale=coord_scale)
    grad_y = compute_gradients(rho_v2_plus_p, coords, coord_scale=coord_scale)
    grad_uv = compute_gradients(rho_uv, coords, coord_scale=coord_scale)

    residual_x = grad_x[..., 0] + grad_uv[..., 1]
    residual_y = grad_uv[..., 0] + grad_y[..., 1]
    return residual_x, residual_y


def energy_residual(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    coords: torch.Tensor,
    gamma: float = 1.4,
    coord_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    kinetic = 0.5 * rho * (u * u + v * v)
    internal = p / (gamma - 1.0)
    total_energy = kinetic + internal
    flux_x = u * (total_energy + p)
    flux_y = v * (total_energy + p)
    grad_flux_x = compute_gradients(flux_x, coords, coord_scale=coord_scale)
    grad_flux_y = compute_gradients(flux_y, coords, coord_scale=coord_scale)
    return grad_flux_x[..., 0] + grad_flux_y[..., 1]


def compressible_euler_residuals(
    predicted_fields: torch.Tensor,
    coords: torch.Tensor,
    coord_scale: Optional[torch.Tensor] = None,
    gamma: float = 1.4,
    include_energy: bool = False,
) -> Dict[str, torch.Tensor]:
    """Compute steady compressible Euler residuals.

    Parameters
    ----------
    predicted_fields:
        Tensor with shape [B, N, 4] containing [u, v, p, rho].
    coords:
        Tensor with shape [B, N, 2]. Must require gradients.
    coord_scale:
        Coordinate normalization standard deviation used to map derivatives back
        to physical coordinates.
    """

    u = predicted_fields[..., 0]
    v = predicted_fields[..., 1]
    p = predicted_fields[..., 2]
    rho = predicted_fields[..., 3]

    continuity = continuity_residual(rho=rho, u=u, v=v, coords=coords, coord_scale=coord_scale)
    momentum_x, momentum_y = momentum_residual(
        rho=rho,
        u=u,
        v=v,
        p=p,
        coords=coords,
        coord_scale=coord_scale,
    )
    outputs: Dict[str, torch.Tensor] = {
        "continuity": continuity,
        "momentum_x": momentum_x,
        "momentum_y": momentum_y,
    }
    if include_energy:
        outputs["energy"] = energy_residual(
            rho=rho,
            u=u,
            v=v,
            p=p,
            coords=coords,
            gamma=gamma,
            coord_scale=coord_scale,
        )
    return outputs
