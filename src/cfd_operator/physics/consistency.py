"""Derived consistency losses for analysis-oriented outputs.

Current implementation intentionally stays conservative. The API exists now so
future stages can plug in:
- surface pressure / Cp consistency
- surface integral -> Cl/Cd consistency
- multi-fidelity or cross-head consistency

At the moment we only expose a structured zero-loss placeholder with explicit
diagnostics. This is preferable to silently pretending a physically rigorous
surface-integral consistency term already exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ConsistencyLossOutputs:
    total: torch.Tensor
    diagnostics: dict[str, Any] = field(default_factory=dict)


def compute_consistency_loss(
    *,
    physics_batch: Any,
    predicted_fields: torch.Tensor | None = None,
    predicted_scalars: torch.Tensor | None = None,
) -> ConsistencyLossOutputs:
    device = physics_batch.branch_inputs.device
    dtype = physics_batch.branch_inputs.dtype
    zero = torch.zeros((), device=device, dtype=dtype)
    diagnostics = {
        "active": False,
        "todo": [
            "surface_integral_consistency_not_yet_enabled",
            "cp_pressure_scalar_consistency_requires_nontrivial_surface_integral_or_auxiliary_targets",
        ],
        "has_surface_points": bool(physics_batch.surface_points is not None and physics_batch.surface_points.numel() > 0),
        "has_scalar_targets": bool(physics_batch.targets.scalar_targets is not None),
    }
    return ConsistencyLossOutputs(total=zero, diagnostics=diagnostics)
