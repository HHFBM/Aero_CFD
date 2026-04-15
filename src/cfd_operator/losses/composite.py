"""Composite supervised + physics-informed loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from cfd_operator.config.schemas import LossConfig
from cfd_operator.data.module import NormalizerBundle
from cfd_operator.models.base import BaseOperatorModel
from cfd_operator.physics import compressible_euler_residuals, incompressible_rans_proxy_residuals


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


def pressure_to_cp(
    pressure: torch.Tensor,
    mach: Optional[torch.Tensor] = None,
    gamma: float = 1.4,
    p_inf: float = 1.0,
    cp_reference: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if cp_reference is not None:
        p_ref = cp_reference[..., 0:1]
        q_ref = torch.clamp(cp_reference[..., 1:2], min=1.0e-4)
        return (pressure - p_ref) / q_ref
    if mach is None:
        raise ValueError("Either mach or cp_reference must be provided.")
    q_inf = 0.5 * gamma * p_inf * mach**2
    q_inf = torch.clamp(q_inf, min=1.0e-4)
    return (pressure - p_inf) / q_inf


@dataclass
class CompositeLoss:
    """Multi-task loss with explicit pressure-channel semantics.

    ``pressure_target_mode`` controls field channel 2:
    - ``raw``: channel 2 stores raw pressure
    - ``cp_like``: channel 2 stores a Cp-like quantity ``(p - p_ref) / q_ref``
    """

    config: LossConfig
    normalizers: NormalizerBundle
    field_names: tuple[str, ...] = ("u", "v", "p", "rho")
    pressure_target_mode: str = "raw"

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
        surface_pressure_loss = self._surface_pressure_loss(model=model, batch=batch)
        heat_flux_loss = self._surface_heat_flux_loss(model=model, batch=batch)
        wall_shear_loss = self._surface_wall_shear_loss(model=model, batch=batch)
        physics_loss = self._physics_loss(model=model, batch=batch) if self.config.use_physics else torch.zeros_like(field_loss)
        boundary_loss = self._boundary_loss(model=model, batch=batch, outputs=outputs)
        slice_loss = self._slice_loss(model=model, batch=batch)
        feature_loss = self._feature_loss(batch=batch, outputs=outputs)
        shock_location_loss = self._shock_location_loss(batch=batch, outputs=outputs)

        total_loss = (
            self.config.field_weight * field_loss
            + self.config.scalar_weight * scalar_loss
            + self.config.surface_weight * surface_loss
            + self.config.surface_pressure_weight * surface_pressure_loss
            + self.config.heat_flux_weight * heat_flux_loss
            + self.config.wall_shear_weight * wall_shear_loss
            + self.config.slice_weight * slice_loss
            + self.config.feature_weight * feature_loss
            + self.config.shock_location_weight * shock_location_loss
            + self.config.physics_weight * physics_loss
            + self.config.boundary_weight * boundary_loss
        )
        metrics = {
            "loss_total": float(total_loss.detach().cpu()),
            "loss_field": float(field_loss.detach().cpu()),
            "loss_scalar": float(scalar_loss.detach().cpu()),
            "loss_surface": float(surface_loss.detach().cpu()),
            "loss_surface_pressure": float(surface_pressure_loss.detach().cpu()),
            "loss_heat_flux": float(heat_flux_loss.detach().cpu()),
            "loss_wall_shear": float(wall_shear_loss.detach().cpu()),
            "loss_slice": float(slice_loss.detach().cpu()),
            "loss_feature": float(feature_loss.detach().cpu()),
            "loss_shock_location": float(shock_location_loss.detach().cpu()),
            "loss_physics": float(physics_loss.detach().cpu()),
            "loss_boundary": float(boundary_loss.detach().cpu()),
        }
        return total_loss, metrics

    def _surface_cp_loss(self, model: BaseOperatorModel, batch: dict[str, Any]) -> torch.Tensor:
        surface_outputs = model.loss_outputs(batch["branch_inputs"], batch["surface_points"])
        surface_fields = self.normalizers.fields.inverse_transform_tensor(surface_outputs["fields"])
        cp_pred = self._surface_cp_prediction(surface_fields=surface_fields, batch=batch)
        return regression_loss(cp_pred, batch["surface_cp"], loss_type=self.config.surface_loss_type, mask=batch["surface_mask"])

    def _surface_pressure_loss(self, model: BaseOperatorModel, batch: dict[str, Any]) -> torch.Tensor:
        if not self.config.use_surface_pressure_loss or self.config.surface_pressure_weight <= 0.0:
            return batch["field_targets"].new_zeros(())
        surface_outputs = model.loss_outputs(batch["branch_inputs"], batch["surface_points"])
        surface_fields = self.normalizers.fields.inverse_transform_tensor(surface_outputs["fields"])
        pressure = self._surface_pressure_prediction(surface_fields=surface_fields, batch=batch)
        return regression_loss(
            pressure,
            batch["surface_pressure"],
            loss_type=self.config.surface_loss_type,
            mask=batch["surface_pressure_available"] * batch["surface_mask"],
        )

    def _surface_cp_prediction(self, surface_fields: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        # Field channel 2 is pressure-like. Convert it to surface Cp using the configured semantics.
        pressure_like = surface_fields[..., 2:3]
        if self.pressure_target_mode == "cp_like":
            return pressure_like
        cp_reference = batch["cp_reference"].unsqueeze(1)
        return pressure_to_cp(pressure=pressure_like, cp_reference=cp_reference)

    def _surface_pressure_prediction(self, surface_fields: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        # Export/evaluation pressure metrics always use raw pressure, even if the training target is cp_like.
        pressure_like = surface_fields[..., 2:3]
        if self.pressure_target_mode != "cp_like":
            return pressure_like
        cp_reference = batch["cp_reference"].unsqueeze(1)
        p_ref = cp_reference[..., 0:1]
        q_ref = torch.clamp(cp_reference[..., 1:2], min=1.0e-4)
        return p_ref + pressure_like * q_ref

    def _surface_heat_flux_loss(self, model: BaseOperatorModel, batch: dict[str, Any]) -> torch.Tensor:
        if not self.config.use_heat_flux_loss or self.config.heat_flux_weight <= 0.0:
            return batch["field_targets"].new_zeros(())
        surface_outputs = model.loss_outputs(batch["branch_inputs"], batch["surface_points"])
        surface_fields = self.normalizers.fields.inverse_transform_tensor(surface_outputs["fields"])
        arc = batch["surface_arc_length"]
        tangential_velocity = torch.linalg.norm(surface_fields[..., :2], dim=-1, keepdim=True)
        grad = torch.diff(tangential_velocity, dim=1, prepend=tangential_velocity[:, :1])
        ds = torch.diff(arc, dim=1, prepend=arc[:, :1]).clamp_min(1.0e-4)
        proxy = (grad.abs() / ds).to(surface_fields.dtype)
        return regression_loss(
            proxy,
            batch["surface_heat_flux"],
            loss_type="mse",
            mask=batch["surface_heat_flux_available"] * batch["surface_mask"],
        )

    def _surface_wall_shear_loss(self, model: BaseOperatorModel, batch: dict[str, Any]) -> torch.Tensor:
        if not self.config.use_wall_shear_loss or self.config.wall_shear_weight <= 0.0:
            return batch["field_targets"].new_zeros(())
        surface_outputs = model.loss_outputs(batch["branch_inputs"], batch["surface_points"])
        surface_fields = self.normalizers.fields.inverse_transform_tensor(surface_outputs["fields"])
        tangent = torch.stack([batch["surface_normals"][..., 1], -batch["surface_normals"][..., 0]], dim=-1)
        tangential_velocity = (surface_fields[..., :2] * tangent).sum(dim=-1, keepdim=True)
        grad = torch.diff(tangential_velocity, dim=1, prepend=tangential_velocity[:, :1])
        ds = torch.diff(batch["surface_arc_length"], dim=1, prepend=batch["surface_arc_length"][:, :1]).clamp_min(1.0e-4)
        proxy = (1.8e-5 * grad.abs() / ds).to(surface_fields.dtype)
        return regression_loss(
            proxy,
            batch["surface_wall_shear"],
            loss_type="mse",
            mask=batch["surface_wall_shear_available"] * batch["surface_mask"],
        )

    def _slice_loss(self, model: BaseOperatorModel, batch: dict[str, Any]) -> torch.Tensor:
        if not self.config.use_slice_loss or self.config.slice_weight <= 0.0:
            return batch["field_targets"].new_zeros(())
        slice_outputs = model.loss_outputs(batch["branch_inputs"], batch["slice_points"])
        return regression_loss(
            slice_outputs["fields"],
            batch["slice_fields"],
            loss_type=self.config.field_loss_type,
            mask=batch["slice_available"] * batch["slice_mask"],
        )

    def _feature_loss(self, batch: dict[str, Any], outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.config.use_feature_loss or self.config.feature_weight <= 0.0:
            return batch["field_targets"].new_zeros(())
        if "features" not in outputs:
            return batch["field_targets"].new_zeros(())
        feature_target = torch.cat([batch["pressure_gradient_indicator"], batch["high_gradient_mask"]], dim=-1)
        if self.config.feature_loss_type == "bce":
            feature_loss = F.binary_cross_entropy_with_logits(outputs["features"], feature_target, reduction="none")
        else:
            feature_loss = (outputs["features"] - feature_target) ** 2
        return masked_reduce(feature_loss, mask=batch["feature_available"] * batch["query_mask"], reduction="mean")

    def _shock_location_loss(self, batch: dict[str, Any], outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.config.use_shock_location_loss or self.config.shock_location_weight <= 0.0:
            return batch["field_targets"].new_zeros(())
        if "features" not in outputs:
            return batch["field_targets"].new_zeros(())
        shock_prob = torch.sigmoid(outputs["features"][..., 0])
        weights = shock_prob / shock_prob.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        coords = batch["query_points_raw"]
        centroid = (coords * weights.unsqueeze(-1)).sum(dim=1)
        loss = (centroid - batch["shock_location"]) ** 2
        return masked_reduce(loss, mask=batch["shock_location_available"], reduction="mean")

    def _physics_loss(self, model: BaseOperatorModel, batch: dict[str, Any]) -> torch.Tensor:
        if len(self.field_names) < 4:
            return batch["field_targets"].new_zeros(())
        coords = batch["query_points"].detach().clone().requires_grad_(True)
        physics_outputs = model.loss_outputs(batch["branch_inputs"], coords)
        fields = self.normalizers.fields.inverse_transform_tensor(physics_outputs["fields"])
        coord_scale = self.normalizers.coordinates.gradient_scale_tensor(device=coords.device, dtype=coords.dtype)
        if self.field_names[3] == "rho":
            residuals = compressible_euler_residuals(
                predicted_fields=fields,
                coords=coords,
                coord_scale=coord_scale,
                include_energy=self.config.use_energy_residual,
            )
        elif self.field_names[3] == "nut":
            residuals = incompressible_rans_proxy_residuals(
                predicted_fields=fields,
                coords=coords,
                coord_scale=coord_scale,
            )
        else:
            return batch["field_targets"].new_zeros(())
        losses = []
        for residual in residuals.values():
            losses.append(masked_reduce(residual**2, mask=batch["query_mask"], reduction="mean"))
        return torch.stack(losses).mean()

    def _boundary_loss(
        self,
        model: BaseOperatorModel,
        batch: dict[str, Any],
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.config.boundary_weight <= 0.0:
            return outputs["fields"].new_zeros(())
        query_fields = self.normalizers.fields.inverse_transform_tensor(outputs["fields"])
        farfield_targets = batch["farfield_targets"].unsqueeze(1).expand_as(query_fields)
        farfield_loss = regression_loss(
            query_fields,
            farfield_targets,
            loss_type="mse",
            mask=batch["farfield_mask"] * batch["query_mask"],
        )

        surface_outputs = model.loss_outputs(batch["branch_inputs"], batch["surface_points"])
        surface_fields = self.normalizers.fields.inverse_transform_tensor(surface_outputs["fields"])
        velocity = surface_fields[..., :2]
        surface_normals = batch["surface_normals"]
        normal_velocity = (velocity * surface_normals).sum(dim=-1)
        wall_loss = masked_reduce(normal_velocity**2, mask=batch["surface_mask"], reduction="mean")
        return 0.5 * (farfield_loss + wall_loss)
