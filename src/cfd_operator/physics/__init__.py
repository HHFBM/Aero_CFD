"""Physics-informed residual utilities."""

from .boundary_conditions import BoundaryConditionLossOutputs, compute_boundary_condition_loss
from .consistency import ConsistencyLossOutputs, compute_consistency_loss
from .residuals import (
    ResidualOutputs,
    compressible_euler_residuals,
    compute_residual_outputs,
    compute_gradients,
    continuity_residual,
    incompressible_continuity_residual,
    incompressible_rans_proxy_residuals,
    laplacian,
    momentum_residual,
    x_momentum_residual,
    y_momentum_residual,
)

__all__ = [
    "BoundaryConditionLossOutputs",
    "ConsistencyLossOutputs",
    "ResidualOutputs",
    "compute_boundary_condition_loss",
    "compute_consistency_loss",
    "compute_residual_outputs",
    "compressible_euler_residuals",
    "compute_gradients",
    "continuity_residual",
    "incompressible_continuity_residual",
    "incompressible_rans_proxy_residuals",
    "laplacian",
    "momentum_residual",
    "x_momentum_residual",
    "y_momentum_residual",
]
