"""Data container types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CFDSample:
    airfoil_id: str
    geometry_params: np.ndarray
    flow_conditions: np.ndarray
    branch_inputs: np.ndarray
    query_points: np.ndarray
    field_targets: np.ndarray
    farfield_mask: np.ndarray
    farfield_targets: np.ndarray
    surface_points: np.ndarray
    surface_normals: np.ndarray
    cp_reference: np.ndarray
    surface_cp: np.ndarray
    scalar_targets: np.ndarray
    fidelity_level: int
    source: str
    convergence_flag: int
    surface_arc_length: np.ndarray | None = None
    surface_pressure: np.ndarray | None = None
    surface_velocity: np.ndarray | None = None
    surface_nut: np.ndarray | None = None
    surface_heat_flux: np.ndarray | None = None
    surface_wall_shear: np.ndarray | None = None
    slice_points: np.ndarray | None = None
    slice_fields: np.ndarray | None = None
    pressure_gradient_indicator: np.ndarray | None = None
    shock_indicator: np.ndarray | None = None
    high_gradient_mask: np.ndarray | None = None
    shock_location: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        surface_count = self.surface_points.shape[0]
        query_count = self.query_points.shape[0]
        return {
            "airfoil_id": self.airfoil_id,
            "geometry_params": self.geometry_params.astype(np.float32),
            "flow_conditions": self.flow_conditions.astype(np.float32),
            "branch_inputs": self.branch_inputs.astype(np.float32),
            "query_points": self.query_points.astype(np.float32),
            "field_targets": self.field_targets.astype(np.float32),
            "farfield_mask": self.farfield_mask.astype(np.float32),
            "farfield_targets": self.farfield_targets.astype(np.float32),
            "surface_points": self.surface_points.astype(np.float32),
            "surface_normals": self.surface_normals.astype(np.float32),
            "surface_arc_length": (
                self.surface_arc_length.astype(np.float32)
                if self.surface_arc_length is not None
                else np.zeros((surface_count, 1), dtype=np.float32)
            ),
            "cp_reference": self.cp_reference.astype(np.float32),
            "surface_cp": self.surface_cp.astype(np.float32),
            "surface_pressure": (
                self.surface_pressure.astype(np.float32)
                if self.surface_pressure is not None
                else np.zeros((surface_count, 1), dtype=np.float32)
            ),
            "surface_velocity": (
                self.surface_velocity.astype(np.float32)
                if self.surface_velocity is not None
                else np.zeros((surface_count, 2), dtype=np.float32)
            ),
            "surface_nut": (
                self.surface_nut.astype(np.float32)
                if self.surface_nut is not None
                else np.zeros((surface_count, 1), dtype=np.float32)
            ),
            "surface_heat_flux": (
                self.surface_heat_flux.astype(np.float32)
                if self.surface_heat_flux is not None
                else np.zeros((surface_count, 1), dtype=np.float32)
            ),
            "surface_wall_shear": (
                self.surface_wall_shear.astype(np.float32)
                if self.surface_wall_shear is not None
                else np.zeros((surface_count, 1), dtype=np.float32)
            ),
            "slice_points": (
                self.slice_points.astype(np.float32)
                if self.slice_points is not None
                else np.zeros((0, 2), dtype=np.float32)
            ),
            "slice_fields": (
                self.slice_fields.astype(np.float32)
                if self.slice_fields is not None
                else np.zeros((0, self.field_targets.shape[-1]), dtype=np.float32)
            ),
            "pressure_gradient_indicator": (
                self.pressure_gradient_indicator.astype(np.float32)
                if self.pressure_gradient_indicator is not None
                else np.zeros((query_count, 1), dtype=np.float32)
            ),
            "shock_indicator": (
                self.shock_indicator.astype(np.float32)
                if self.shock_indicator is not None
                else np.zeros((query_count, 1), dtype=np.float32)
            ),
            "high_gradient_mask": (
                self.high_gradient_mask.astype(np.float32)
                if self.high_gradient_mask is not None
                else np.zeros((query_count, 1), dtype=np.float32)
            ),
            "shock_location": (
                self.shock_location.astype(np.float32)
                if self.shock_location is not None
                else np.asarray([np.nan, np.nan], dtype=np.float32)
            ),
            "scalar_targets": self.scalar_targets.astype(np.float32),
            "fidelity_level": np.int64(self.fidelity_level),
            "source": self.source,
            "convergence_flag": np.int64(self.convergence_flag),
        }
