"""Data container types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class CFDSample:
    airfoil_id: str
    geometry_params: np.ndarray
    flow_conditions: np.ndarray
    branch_inputs: np.ndarray
    query_points: np.ndarray
    field_targets: np.ndarray
    surface_points: np.ndarray
    surface_cp: np.ndarray
    scalar_targets: np.ndarray
    fidelity_level: int
    source: str
    convergence_flag: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "airfoil_id": self.airfoil_id,
            "geometry_params": self.geometry_params.astype(np.float32),
            "flow_conditions": self.flow_conditions.astype(np.float32),
            "branch_inputs": self.branch_inputs.astype(np.float32),
            "query_points": self.query_points.astype(np.float32),
            "field_targets": self.field_targets.astype(np.float32),
            "surface_points": self.surface_points.astype(np.float32),
            "surface_cp": self.surface_cp.astype(np.float32),
            "scalar_targets": self.scalar_targets.astype(np.float32),
            "fidelity_level": np.int64(self.fidelity_level),
            "source": self.source,
            "convergence_flag": np.int64(self.convergence_flag),
        }

