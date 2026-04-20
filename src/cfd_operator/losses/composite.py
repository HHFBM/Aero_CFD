"""Composite supervised + physics-informed loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from cfd_operator.config.schemas import LossConfig
from cfd_operator.data.module import NormalizerBundle
from cfd_operator.hard_regions import query_region_masks_torch, surface_region_masks_torch
from cfd_operator.losses.physics_losses import compute_physics_informed_loss
from cfd_operator.models.base import BaseOperatorModel
from cfd_operator.tasks.capabilities import DatasetCapability, EffectiveTaskSet, TaskRequest, resolve_effective_tasks


def masked_reduce(
    loss: torch.Tensor,
    mask: Optional[torch.Tensor],
    reduction: str = "mean",
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mask is None and weight is None:
        return loss.mean() if reduction == "mean" else loss.sum()
    effective = None
    if mask is not None:
        effective = mask
        while effective.ndim < loss.ndim:
            effective = effective.unsqueeze(-1)
    if weight is not None:
        effective_weight = weight
        while effective_weight.ndim < loss.ndim:
            effective_weight = effective_weight.unsqueeze(-1)
        effective = effective_weight if effective is None else (effective * effective_weight)
    assert effective is not None
    weighted = loss * effective
    denominator = effective.sum().clamp_min(1.0)
    if reduction == "mean":
        return weighted.sum() / denominator
    return weighted.sum()


def regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
    mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if loss_type == "mse":
        return masked_reduce((prediction - target) ** 2, mask=mask, reduction="mean", weight=weight)
    if loss_type == "mae":
        return masked_reduce((prediction - target).abs(), mask=mask, reduction="mean", weight=weight)
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
    dataset_capability: DatasetCapability | None = None
    task_request: TaskRequest | None = None
    current_epoch: int = 0
    current_step: int = 0

    def set_task_context(
        self,
        *,
        dataset_capability: DatasetCapability | None,
        task_request: TaskRequest | None,
    ) -> None:
        self.dataset_capability = dataset_capability
        self.task_request = task_request

    def set_schedule_context(self, *, epoch: int, global_step: int) -> None:
        self.current_epoch = int(epoch)
        self.current_step = int(global_step)

    def _effective_tasks(self) -> EffectiveTaskSet | None:
        if self.dataset_capability is None or self.task_request is None:
            return None
        return resolve_effective_tasks(self.dataset_capability, self.task_request)

    def __call__(
        self,
        model: BaseOperatorModel,
        batch: dict[str, Any],
        outputs: dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        effective_tasks = self._effective_tasks()
        query_region_weights = self._query_region_weights(batch=batch)
        surface_region_weights = self._surface_region_weights(batch=batch)
        slice_region_weights = self._slice_region_weights(batch=batch)
        field_loss = regression_loss(
            outputs["fields"],
            batch["field_targets"],
            loss_type=self.config.field_loss_type,
            mask=batch["query_mask"],
            weight=query_region_weights,
        )
        if effective_tasks is not None and not effective_tasks.field:
            field_loss = batch["field_targets"].new_zeros(())
        scalar_loss = regression_loss(
            outputs["scalars"],
            batch["scalar_targets"],
            loss_type=self.config.scalar_loss_type,
            mask=None,
        )
        if effective_tasks is not None and not effective_tasks.scalar:
            scalar_loss = batch["field_targets"].new_zeros(())
        surface_loss = self._surface_cp_loss(model=model, batch=batch) if effective_tasks is None or effective_tasks.surface else batch["field_targets"].new_zeros(())
        surface_pressure_loss = self._surface_pressure_loss(model=model, batch=batch) if effective_tasks is None or effective_tasks.surface else batch["field_targets"].new_zeros(())
        heat_flux_loss = self._surface_heat_flux_loss(model=model, batch=batch) if effective_tasks is None or effective_tasks.surface else batch["field_targets"].new_zeros(())
        wall_shear_loss = self._surface_wall_shear_loss(model=model, batch=batch) if effective_tasks is None or effective_tasks.surface else batch["field_targets"].new_zeros(())
        physics_bundle = (
            compute_physics_informed_loss(
                model=model,
                batch=batch,
                config=self.config,
                normalizers=self.normalizers,
                field_names=self.field_names,
                pressure_target_mode=self.pressure_target_mode,
                current_epoch=self.current_epoch,
                current_step=self.current_step,
            )
            if self.config.use_physics or self.config.boundary_weight > 0.0 or self.config.consistency_weight > 0.0
            else None
        )
        physics_loss = (
            physics_bundle["pde_total"]
            if physics_bundle is not None and self.config.use_physics and (effective_tasks is None or effective_tasks.consistency)
            else torch.zeros_like(field_loss)
        )
        boundary_loss = (
            physics_bundle["boundary"]
            if physics_bundle is not None and (effective_tasks is None or effective_tasks.consistency)
            else batch["field_targets"].new_zeros(())
        )
        consistency_loss = (
            physics_bundle["consistency"]
            if physics_bundle is not None and (effective_tasks is None or effective_tasks.consistency)
            else batch["field_targets"].new_zeros(())
        )
        slice_loss = self._slice_loss(model=model, batch=batch) if effective_tasks is None or effective_tasks.slice else batch["field_targets"].new_zeros(())
        feature_loss = self._feature_loss(batch=batch, outputs=outputs) if effective_tasks is None or effective_tasks.feature else batch["field_targets"].new_zeros(())
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
            + self.config.consistency_weight * consistency_loss
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
            "loss_consistency": float(consistency_loss.detach().cpu()),
            "query_hard_weight_mean": float(query_region_weights.mean().detach().cpu()),
            "surface_hard_weight_mean": float(surface_region_weights.mean().detach().cpu()) if surface_region_weights.numel() > 0 else 1.0,
            "slice_hard_weight_mean": float(slice_region_weights.mean().detach().cpu()) if slice_region_weights.numel() > 0 else 1.0,
        }
        if physics_bundle is not None:
            metrics["loss_continuity"] = float(physics_bundle["continuity"].detach().cpu())
            metrics["loss_momentum_x"] = float(physics_bundle["momentum_x"].detach().cpu())
            metrics["loss_momentum_y"] = float(physics_bundle["momentum_y"].detach().cpu())
            metrics["loss_nut_transport_proxy"] = float(physics_bundle["nut_transport_proxy"].detach().cpu())
            schedule_multiplier = physics_bundle["diagnostics"].get("schedule_multiplier", 1.0)
            metrics["physics_schedule_multiplier"] = float(schedule_multiplier)
        return total_loss, metrics

    def _surface_cp_loss(self, model: BaseOperatorModel, batch: dict[str, Any]) -> torch.Tensor:
        surface_outputs = model.loss_outputs(batch["branch_inputs"], batch["surface_points"])
        surface_fields = self.normalizers.fields.inverse_transform_tensor(surface_outputs["fields"])
        cp_pred = self._surface_cp_prediction(surface_fields=surface_fields, batch=batch)
        return regression_loss(
            cp_pred,
            batch["surface_cp"],
            loss_type=self.config.surface_loss_type,
            mask=batch["surface_mask"],
            weight=self._surface_region_weights(batch=batch),
        )

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
            weight=self._surface_region_weights(batch=batch),
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
            weight=self._surface_region_weights(batch=batch),
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
            weight=self._surface_region_weights(batch=batch),
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
            weight=self._slice_region_weights(batch=batch),
        )

    def _feature_loss(self, batch: dict[str, Any], outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.config.use_feature_loss or self.config.feature_weight <= 0.0:
            return batch["field_targets"].new_zeros(())
        if "features" not in outputs:
            return batch["field_targets"].new_zeros(())
        feature_target = torch.cat([batch["pressure_gradient_indicator"], batch["high_gradient_mask"]], dim=-1)
        if self.config.feature_loss_type == "bce":
            pos_weight = None
            if self.config.use_feature_class_balancing:
                positive_fraction = feature_target.mean(dim=(0, 1))
                negative_fraction = 1.0 - positive_fraction
                dynamic_pos_weight = negative_fraction / positive_fraction.clamp_min(1.0e-4)
                if self.config.feature_positive_weight > 1.0:
                    pos_weight = torch.full_like(dynamic_pos_weight, self.config.feature_positive_weight)
                else:
                    pos_weight = dynamic_pos_weight.clamp(1.0, self.config.feature_max_positive_weight)
            feature_loss = F.binary_cross_entropy_with_logits(
                outputs["features"],
                feature_target,
                reduction="none",
                pos_weight=pos_weight,
            )
            if self.config.feature_focal_gamma > 0.0:
                probs = torch.sigmoid(outputs["features"])
                pt = probs * feature_target + (1.0 - probs) * (1.0 - feature_target)
                feature_loss = feature_loss * ((1.0 - pt).clamp_min(1.0e-6) ** self.config.feature_focal_gamma)
        else:
            feature_loss = (outputs["features"] - feature_target) ** 2
        return masked_reduce(
            feature_loss,
            mask=batch["feature_available"] * batch["query_mask"],
            reduction="mean",
            weight=self._query_region_weights(batch=batch),
        )

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

    def _query_region_weights(self, batch: dict[str, Any]) -> torch.Tensor:
        query_mask = batch["query_mask"]
        weights = torch.ones_like(query_mask)
        if not self.config.use_hard_region_weighting:
            return weights
        masks = query_region_masks_torch(
            batch["query_points_raw"],
            batch["surface_points_raw"],
            surface_mask=batch.get("surface_mask"),
            high_gradient_mask=batch.get("high_gradient_mask"),
            near_wall_distance_fraction=self.config.near_wall_distance_fraction,
            wake_halfwidth_fraction=self.config.wake_halfwidth_fraction,
        )
        if self.config.high_gradient_region_weight > 0.0:
            weights = weights + self.config.high_gradient_region_weight * masks["high_gradient"]
        if self.config.near_wall_region_weight > 0.0:
            weights = weights + self.config.near_wall_region_weight * masks["near_wall"]
        if self.config.wake_region_weight > 0.0:
            weights = weights + self.config.wake_region_weight * masks["wake"]
        return weights

    def _slice_region_weights(self, batch: dict[str, Any]) -> torch.Tensor:
        slice_mask = batch["slice_mask"]
        weights = torch.ones_like(slice_mask)
        if not self.config.use_hard_region_weighting or batch["slice_points_raw"].shape[1] == 0:
            return weights
        masks = query_region_masks_torch(
            batch["slice_points_raw"],
            batch["surface_points_raw"],
            surface_mask=batch.get("surface_mask"),
            high_gradient_mask=None,
            near_wall_distance_fraction=self.config.near_wall_distance_fraction,
            wake_halfwidth_fraction=self.config.wake_halfwidth_fraction,
        )
        if self.config.near_wall_region_weight > 0.0:
            weights = weights + self.config.near_wall_region_weight * masks["near_wall"]
        if self.config.wake_region_weight > 0.0:
            weights = weights + self.config.wake_region_weight * masks["wake"]
        return weights

    def _surface_region_weights(self, batch: dict[str, Any]) -> torch.Tensor:
        surface_mask = batch["surface_mask"]
        weights = torch.ones_like(surface_mask)
        if not self.config.use_hard_region_weighting or self.config.surface_leading_edge_weight <= 0.0:
            return weights
        masks = surface_region_masks_torch(
            batch["surface_points_raw"],
            surface_mask=batch.get("surface_mask"),
            leading_edge_fraction=self.config.leading_edge_fraction,
        )
        return weights + self.config.surface_leading_edge_weight * masks["leading_edge"]
