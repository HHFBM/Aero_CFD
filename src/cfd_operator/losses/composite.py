"""Composite supervised + physics-informed loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from cfd_operator.config.schemas import LossConfig
from cfd_operator.data.module import NormalizerBundle
from cfd_operator.models.base import BaseOperatorModel
from cfd_operator.physics import compressible_euler_residuals


def masked_reduce(loss: torch.Tensor, mask: Optional[torch.Tensor], reduction: str = "mean") -> torch.Tensor:
    if mask is None:
        return loss.mean() if reduction == "mean" else loss.sum()
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    weighted = loss * mask
    denominator = mask.sum().clamp_min(1.0)
    if reduction == "mean":
        return weighted.sum() / denominator
    return weighted.sum()


def regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if loss_type == "mse":
        return masked_reduce((prediction - target) ** 2, mask=mask, reduction="mean")
    if loss_type == "mae":
        return masked_reduce((prediction - target).abs(), mask=mask, reduction="mean")
    raise ValueError(f"Unsupported loss type: {loss_type}")


def pressure_to_cp(pressure: torch.Tensor, mach: torch.Tensor, gamma: float = 1.4, p_inf: float = 1.0) -> torch.Tensor:
    q_inf = 0.5 * gamma * p_inf * mach**2
    q_inf = torch.clamp(q_inf, min=1.0e-4)
    return (pressure - p_inf) / q_inf


@dataclass
class CompositeLoss:
    config: LossConfig
    normalizers: NormalizerBundle

    def __call__(
        self,
        model: BaseOperatorModel,
        batch: dict[str, Any],
        outputs: dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        field_loss = regression_loss(
            outputs["fields"],
            batch["field_targets"],
            loss_type=self.config.field_loss_type,
            mask=batch["query_mask"],
        )
        scalar_loss = regression_loss(
            outputs["scalars"],
            batch["scalar_targets"],
            loss_type=self.config.scalar_loss_type,
            mask=None,
        )
        surface_loss = self._surface_cp_loss(model=model, batch=batch)
        physics_loss = self._physics_loss(model=model, batch=batch) if self.config.use_physics else torch.zeros_like(field_loss)
        boundary_loss = torch.zeros_like(field_loss)

        total_loss = (
            self.config.field_weight * field_loss
            + self.config.scalar_weight * scalar_loss
            + self.config.surface_weight * surface_loss
            + self.config.physics_weight * physics_loss
            + self.config.boundary_weight * boundary_loss
        )
        metrics = {
            "loss_total": float(total_loss.detach().cpu()),
            "loss_field": float(field_loss.detach().cpu()),
            "loss_scalar": float(scalar_loss.detach().cpu()),
            "loss_surface": float(surface_loss.detach().cpu()),
            "loss_physics": float(physics_loss.detach().cpu()),
            "loss_boundary": float(boundary_loss.detach().cpu()),
        }
        return total_loss, metrics

    def _surface_cp_loss(self, model: BaseOperatorModel, batch: dict[str, Any]) -> torch.Tensor:
        surface_outputs = model.loss_outputs(batch["branch_inputs"], batch["surface_points"])
        surface_fields = self.normalizers.fields.inverse_transform_tensor(surface_outputs["fields"])
        pressure = surface_fields[..., 2:3]
        mach = batch["flow_conditions"][:, 0].unsqueeze(1).unsqueeze(2)
        cp_pred = pressure_to_cp(pressure=pressure, mach=mach)
        return regression_loss(cp_pred, batch["surface_cp"], loss_type="mse", mask=batch["surface_mask"])

    def _physics_loss(self, model: BaseOperatorModel, batch: dict[str, Any]) -> torch.Tensor:
        coords = batch["query_points"].detach().clone().requires_grad_(True)
        physics_outputs = model.loss_outputs(batch["branch_inputs"], coords)
        fields = self.normalizers.fields.inverse_transform_tensor(physics_outputs["fields"])
        coord_scale = self.normalizers.coordinates.gradient_scale_tensor(device=coords.device, dtype=coords.dtype)
        residuals = compressible_euler_residuals(
            predicted_fields=fields,
            coords=coords,
            coord_scale=coord_scale,
            include_energy=self.config.use_energy_residual,
        )
        losses = []
        for residual in residuals.values():
            losses.append(masked_reduce(residual**2, mask=batch["query_mask"], reduction="mean"))
        return torch.stack(losses).mean()
