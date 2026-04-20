"""Autograd-based residuals for simplified steady 2D flow surrogates.

Assumptions:
- 2D steady external flow
- Supports two primary modes:
  - compressible Euler-like residuals for fields [u, v, p, rho]
  - incompressible RANS-like proxy residuals for fields [u, v, p, nut]
- Coordinates may be normalized before entering the model; derivative scaling is
  corrected using the coordinate normalizer standard deviation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ResidualOutputs:
    continuity: torch.Tensor
    momentum_x: torch.Tensor
    momentum_y: torch.Tensor
    nut_transport_proxy: torch.Tensor | None = None
    energy: torch.Tensor | None = None
    mode: str = "unknown"
    strict_pde: bool = False
    proxy_terms: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {
            "continuity": self.continuity,
            "momentum_x": self.momentum_x,
            "momentum_y": self.momentum_y,
        }
        if self.nut_transport_proxy is not None:
            outputs["nut_transport_proxy"] = self.nut_transport_proxy
        if self.energy is not None:
            outputs["energy"] = self.energy
        return outputs


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


def laplacian(
    values: torch.Tensor,
    coords: torch.Tensor,
    coord_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return scalar Laplacian with shape [B, N]."""

    gradients = compute_gradients(values, coords, coord_scale=coord_scale)
    grad_x = compute_gradients(gradients[..., 0], coords, coord_scale=coord_scale)
    grad_y = compute_gradients(gradients[..., 1], coords, coord_scale=coord_scale)
    return grad_x[..., 0] + grad_y[..., 1]


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


def x_momentum_residual(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    coords: torch.Tensor,
    coord_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return only the x-momentum residual for the steady 2D Euler system."""

    residual_x, _ = momentum_residual(
        rho=rho,
        u=u,
        v=v,
        p=p,
        coords=coords,
        coord_scale=coord_scale,
    )
    return residual_x


def y_momentum_residual(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    coords: torch.Tensor,
    coord_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return only the y-momentum residual for the steady 2D Euler system."""

    _, residual_y = momentum_residual(
        rho=rho,
        u=u,
        v=v,
        p=p,
        coords=coords,
        coord_scale=coord_scale,
    )
    return residual_y


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


def incompressible_continuity_residual(
    u: torch.Tensor,
    v: torch.Tensor,
    coords: torch.Tensor,
    coord_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    grad_u = compute_gradients(u, coords, coord_scale=coord_scale)
    grad_v = compute_gradients(v, coords, coord_scale=coord_scale)
    return grad_u[..., 0] + grad_v[..., 1]


def incompressible_rans_proxy_residuals(
    predicted_fields: torch.Tensor,
    coords: torch.Tensor,
    coord_scale: Optional[torch.Tensor] = None,
    laminar_viscosity: float = 1.0e-3,
    nut_diffusivity: float = 5.0e-2,
) -> Dict[str, torch.Tensor]:
    """Compute a lightweight steady 2D incompressible RANS-like proxy residual.

    This is not a full turbulence-model residual. It provides a physics-informed
    regularizer compatible with AirfRANS-style fields [u, v, p, nut]:

    - continuity: div(u) = 0
    - momentum: convective acceleration + pressure gradient - nu_eff * Laplacian(u)
    - nut transport proxy: advection - diffusivity * Laplacian(nut)
    """

    u = predicted_fields[..., 0]
    v = predicted_fields[..., 1]
    p = predicted_fields[..., 2]
    nut = predicted_fields[..., 3]

    grad_u = compute_gradients(u, coords, coord_scale=coord_scale)
    grad_v = compute_gradients(v, coords, coord_scale=coord_scale)
    grad_p = compute_gradients(p, coords, coord_scale=coord_scale)
    grad_nut = compute_gradients(nut, coords, coord_scale=coord_scale)

    continuity = grad_u[..., 0] + grad_v[..., 1]

    nu_eff = laminar_viscosity + F.softplus(nut)
    lap_u = laplacian(u, coords, coord_scale=coord_scale)
    lap_v = laplacian(v, coords, coord_scale=coord_scale)
    lap_nut = laplacian(nut, coords, coord_scale=coord_scale)

    convective_u = u * grad_u[..., 0] + v * grad_u[..., 1]
    convective_v = u * grad_v[..., 0] + v * grad_v[..., 1]

    momentum_x = convective_u + grad_p[..., 0] - nu_eff * lap_u
    momentum_y = convective_v + grad_p[..., 1] - nu_eff * lap_v
    nut_transport = u * grad_nut[..., 0] + v * grad_nut[..., 1] - nut_diffusivity * lap_nut

    return {
        "continuity": continuity,
        "momentum_x": momentum_x,
        "momentum_y": momentum_y,
        "nut_transport": nut_transport,
    }


def compute_residual_outputs(
    predicted_fields: torch.Tensor,
    coords: torch.Tensor,
    field_names: tuple[str, ...],
    coord_scale: Optional[torch.Tensor] = None,
    gamma: float = 1.4,
    include_energy: bool = False,
) -> ResidualOutputs:
    """Unified wrapper returning structured residual outputs.

    The current project supports two main field conventions:
    - ``[u, v, p, rho]``: Euler-like residuals
    - ``[u, v, p, nut]``: incompressible RANS proxy residuals

    Any other 4th channel is treated as unsupported and produces zero-valued
    placeholder residuals rather than silently applying an invalid PDE.
    """

    aux_name = field_names[3] if len(field_names) >= 4 else "unknown"
    zero = torch.zeros_like(predicted_fields[..., 0])
    if aux_name == "rho":
        outputs = compressible_euler_residuals(
            predicted_fields=predicted_fields,
            coords=coords,
            coord_scale=coord_scale,
            gamma=gamma,
            include_energy=include_energy,
        )
        return ResidualOutputs(
            continuity=outputs["continuity"],
            momentum_x=outputs["momentum_x"],
            momentum_y=outputs["momentum_y"],
            energy=outputs.get("energy"),
            mode="compressible_euler",
            strict_pde=True,
            proxy_terms=(),
        )
    if aux_name == "nut":
        outputs = incompressible_rans_proxy_residuals(
            predicted_fields=predicted_fields,
            coords=coords,
            coord_scale=coord_scale,
        )
        return ResidualOutputs(
            continuity=outputs["continuity"],
            momentum_x=outputs["momentum_x"],
            momentum_y=outputs["momentum_y"],
            nut_transport_proxy=outputs["nut_transport"],
            mode="incompressible_rans_proxy",
            strict_pde=False,
            proxy_terms=("momentum_proxy", "nut_transport_proxy"),
        )
    return ResidualOutputs(
        continuity=zero,
        momentum_x=zero,
        momentum_y=zero,
        mode=f"unsupported_aux_{aux_name}",
        strict_pde=False,
        proxy_terms=("unsupported_field_layout",),
    )
