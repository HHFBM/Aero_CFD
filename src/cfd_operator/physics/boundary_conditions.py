"""Boundary-condition loss helpers for weak physics regularization.

These losses are intentionally conservative:
- wall / no-penetration uses surface normal velocity when normals exist
- farfield supervision reuses available farfield targets when the batch carries
  them

The module does not invent boundary labels that do not exist in the current
dataset. Missing boundary metadata results in zero-valued placeholders plus
diagnostics/TODO notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from cfd_operator.data.module import NormalizerBundle
from cfd_operator.models.base import BaseOperatorModel


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
class BoundaryConditionLossOutputs:
    total: torch.Tensor
    wall: torch.Tensor
    farfield: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


def compute_boundary_condition_loss(
    *,
    model: BaseOperatorModel,
    physics_batch: Any,
    normalizers: NormalizerBundle,
) -> BoundaryConditionLossOutputs:
    """Compute conservative BC losses from the currently available metadata.

    Notes
    -----
    - This is a weak-BC implementation for the current surrogate framework.
    - If the batch lacks boundary normals, farfield masks, or explicit boundary
      points, the corresponding term is skipped and recorded in diagnostics.
    """

    device = physics_batch.branch_inputs.device
    dtype = physics_batch.branch_inputs.dtype
    zero = torch.zeros((), device=device, dtype=dtype)
    diagnostics: dict[str, Any] = {
        "wall_bc_active": False,
        "farfield_bc_active": False,
        "todo": [],
    }

    wall_loss = zero
    if (
        physics_batch.surface_points is not None
        and physics_batch.surface_points.numel() > 0
        and physics_batch.surface_mask is not None
    ):
        if physics_batch.surface_normals is None or physics_batch.surface_normals.numel() == 0:
            diagnostics["todo"].append("surface_normals_missing_for_wall_bc")
        else:
            surface_outputs = model.loss_outputs(physics_batch.branch_inputs, physics_batch.surface_points)
            surface_fields = normalizers.fields.inverse_transform_tensor(surface_outputs["fields"])
            velocity = surface_fields[..., :2]
            normal_velocity = (velocity * physics_batch.surface_normals).sum(dim=-1)
            wall_loss = _masked_mean(normal_velocity.square(), physics_batch.surface_mask)
            diagnostics["wall_bc_active"] = True

    farfield_loss = zero
    if (
        physics_batch.targets.farfield_targets is not None
        and physics_batch.farfield_mask is not None
        and physics_batch.collocation_fields is not None
    ):
        farfield_targets = physics_batch.targets.farfield_targets.unsqueeze(1).expand_as(physics_batch.collocation_fields)
        farfield_loss = _masked_mean(
            (physics_batch.collocation_fields - farfield_targets).square(),
            physics_batch.farfield_mask * physics_batch.collocation_mask,
        )
        diagnostics["farfield_bc_active"] = True
    elif physics_batch.farfield_mask is not None:
        diagnostics["todo"].append("farfield_targets_missing_for_farfield_bc")

    total = 0.5 * (wall_loss + farfield_loss)
    return BoundaryConditionLossOutputs(
        total=total,
        wall=wall_loss,
        farfield=farfield_loss,
        diagnostics=diagnostics,
    )
