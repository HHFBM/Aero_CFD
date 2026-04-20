"""Unified physics-informed loss entrypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from cfd_operator.config.schemas import LossConfig
from cfd_operator.data.module import NormalizerBundle
from cfd_operator.losses.loss_scheduler import LossScheduleState, build_physics_loss_scheduler
from cfd_operator.models.base import BaseOperatorModel
from cfd_operator.physics.boundary_conditions import compute_boundary_condition_loss
from cfd_operator.physics.consistency import compute_consistency_loss
from cfd_operator.physics.residuals import ResidualOutputs, compute_residual_outputs


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    weight = mask
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    numerator = (values * weight).sum()
    denominator = weight.sum().clamp_min(1.0)
    return numerator / denominator


@dataclass
class PhysicsTargets:
    field_targets: torch.Tensor | None = None
    scalar_targets: torch.Tensor | None = None
    surface_cp: torch.Tensor | None = None
    surface_pressure: torch.Tensor | None = None
    farfield_targets: torch.Tensor | None = None


@dataclass
class PhysicsBatch:
    branch_inputs: torch.Tensor
    supervised_points: torch.Tensor
    supervised_mask: torch.Tensor
    collocation_points: torch.Tensor
    collocation_mask: torch.Tensor
    collocation_fields: torch.Tensor | None = None
    surface_points: torch.Tensor | None = None
    surface_mask: torch.Tensor | None = None
    surface_normals: torch.Tensor | None = None
    farfield_mask: torch.Tensor | None = None
    cp_reference: torch.Tensor | None = None
    targets: PhysicsTargets = field(default_factory=PhysicsTargets)
    field_names: tuple[str, ...] = ("u", "v", "p", "rho")
    pressure_target_mode: str = "raw"
    metadata: dict[str, Any] = field(default_factory=dict)


def build_physics_batch(
    *,
    batch: dict[str, Any],
    collocation_points: torch.Tensor | None = None,
    collocation_mask: torch.Tensor | None = None,
    field_names: tuple[str, ...],
    pressure_target_mode: str,
) -> PhysicsBatch:
    resolved_collocation_points = collocation_points if collocation_points is not None else batch["query_points"]
    resolved_collocation_mask = collocation_mask if collocation_mask is not None else batch["query_mask"]
    return PhysicsBatch(
        branch_inputs=batch["branch_inputs"],
        supervised_points=batch["query_points"],
        supervised_mask=batch["query_mask"],
        collocation_points=resolved_collocation_points,
        collocation_mask=resolved_collocation_mask,
        surface_points=batch.get("surface_points"),
        surface_mask=batch.get("surface_mask"),
        surface_normals=batch.get("surface_normals"),
        farfield_mask=batch.get("farfield_mask"),
        cp_reference=batch.get("cp_reference"),
        targets=PhysicsTargets(
            field_targets=batch.get("field_targets"),
            scalar_targets=batch.get("scalar_targets"),
            surface_cp=batch.get("surface_cp"),
            surface_pressure=batch.get("surface_pressure"),
            farfield_targets=batch.get("farfield_targets"),
        ),
        field_names=field_names,
        pressure_target_mode=pressure_target_mode,
        metadata={
            "geometry_representation": batch.get("geometry_representation"),
            "branch_encoding_type": batch.get("branch_encoding_type"),
            "collocation_source": "query_points" if collocation_points is None else "explicit",
        },
    )


def compute_physics_informed_loss(
    *,
    model: BaseOperatorModel,
    batch: dict[str, Any],
    config: LossConfig,
    normalizers: NormalizerBundle,
    field_names: tuple[str, ...],
    pressure_target_mode: str,
    current_epoch: int = 0,
    current_step: int = 0,
) -> dict[str, Any]:
    physics_batch = build_physics_batch(
        batch=batch,
        field_names=field_names,
        pressure_target_mode=pressure_target_mode,
    )
    device = physics_batch.branch_inputs.device
    dtype = physics_batch.branch_inputs.dtype
    zero = torch.zeros((), device=device, dtype=dtype)

    scheduler = build_physics_loss_scheduler(
        warmup_epochs=config.physics_warmup_epochs,
        ramp_epochs=config.physics_ramp_epochs,
        max_weight=config.physics_schedule_max_weight,
    )
    schedule_state = LossScheduleState(epoch=current_epoch, global_step=current_step)
    schedule_multiplier = scheduler.multiplier(schedule_state)

    if config.use_physics:
        coords = physics_batch.collocation_points.detach().clone().requires_grad_(True)
    else:
        coords = physics_batch.collocation_points
    physics_outputs = model.loss_outputs(physics_batch.branch_inputs, coords)
    collocation_fields = normalizers.fields.inverse_transform_tensor(physics_outputs["fields"])
    physics_batch.collocation_fields = collocation_fields
    if config.use_physics:
        coord_scale = normalizers.coordinates.gradient_scale_tensor(device=coords.device, dtype=coords.dtype)
        residuals: ResidualOutputs = compute_residual_outputs(
            predicted_fields=collocation_fields,
            coords=coords,
            field_names=field_names,
            coord_scale=coord_scale,
            include_energy=config.use_energy_residual,
        )

        continuity_loss = _masked_mean(residuals.continuity.square(), physics_batch.collocation_mask)
        momentum_x_loss = _masked_mean(residuals.momentum_x.square(), physics_batch.collocation_mask)
        momentum_y_loss = _masked_mean(residuals.momentum_y.square(), physics_batch.collocation_mask)
        nut_loss = (
            _masked_mean(residuals.nut_transport_proxy.square(), physics_batch.collocation_mask)
            if residuals.nut_transport_proxy is not None
            else zero
        )
        pde_total = schedule_multiplier * (
            config.lambda_continuity * continuity_loss
            + config.lambda_momentum * 0.5 * (momentum_x_loss + momentum_y_loss)
            + config.lambda_nut * nut_loss
        )
    else:
        residuals = ResidualOutputs(
            continuity=torch.zeros_like(collocation_fields[..., 0]),
            momentum_x=torch.zeros_like(collocation_fields[..., 0]),
            momentum_y=torch.zeros_like(collocation_fields[..., 0]),
            mode="disabled",
            strict_pde=False,
            proxy_terms=("disabled",),
        )
        continuity_loss = zero
        momentum_x_loss = zero
        momentum_y_loss = zero
        nut_loss = zero
        pde_total = zero

    boundary_outputs = compute_boundary_condition_loss(
        model=model,
        physics_batch=physics_batch,
        normalizers=normalizers,
    )
    boundary_total = schedule_multiplier * config.lambda_bc * boundary_outputs.total

    consistency_outputs = compute_consistency_loss(
        physics_batch=physics_batch,
        predicted_fields=collocation_fields,
        predicted_scalars=physics_outputs.get("scalars"),
    )
    consistency_total = schedule_multiplier * config.lambda_consistency * consistency_outputs.total

    data_placeholder = zero
    total = config.lambda_data * data_placeholder + pde_total + boundary_total + consistency_total

    diagnostics = {
        "schedule_multiplier": float(schedule_multiplier),
        "residual_mode": residuals.mode,
        "collocation_source": physics_batch.metadata["collocation_source"],
        "boundary": boundary_outputs.diagnostics,
        "consistency": consistency_outputs.diagnostics,
        "strict_pde": residuals.strict_pde,
        "proxy_terms": list(residuals.proxy_terms),
    }
    return {
        "total": total,
        "data": data_placeholder,
        "continuity": continuity_loss,
        "momentum_x": momentum_x_loss,
        "momentum_y": momentum_y_loss,
        "nut_transport_proxy": nut_loss,
        "pde_total": pde_total,
        "boundary": boundary_total,
        "consistency": consistency_total,
        "diagnostics": diagnostics,
    }
