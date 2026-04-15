"""API request and response models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class PredictionRequest(BaseModel):
    geometry_mode: Optional[str] = Field(
        default=None,
        description="One of: legacy_naca_params, generic_surface_points, structured_param_vector.",
    )
    geometry_params: Optional[List[float]] = Field(default=None, description="[max_camber, camber_position, thickness, chord?]")
    geometry_points: Optional[List[List[float]]] = Field(default=None, description="Ordered closed airfoil contour points [N,2].")
    upper_surface_points: Optional[List[List[float]]] = None
    lower_surface_points: Optional[List[List[float]]] = None
    mach: float
    aoa: float
    query_points: List[List[float]]
    surface_points: Optional[List[List[float]]] = None
    reynolds: Optional[float] = None

    @model_validator(mode="after")
    def validate_geometry_input(self) -> "PredictionRequest":
        if self.geometry_mode is None:
            if self.geometry_points is not None or (self.upper_surface_points is not None and self.lower_surface_points is not None):
                self.geometry_mode = "generic_surface_points"
            elif self.geometry_params is not None:
                self.geometry_mode = "legacy_naca_params"
        if self.geometry_mode is None:
            raise ValueError(
                "PredictionRequest requires geometry_params, geometry_points, or upper/lower surface points."
            )
        if self.geometry_mode == "legacy_naca_params" and self.geometry_params is None:
            raise ValueError("geometry_mode='legacy_naca_params' requires geometry_params.")
        if self.geometry_mode == "structured_param_vector" and self.geometry_params is None:
            raise ValueError("geometry_mode='structured_param_vector' requires geometry_params.")
        if self.geometry_mode == "generic_surface_points":
            if self.geometry_points is None and not (self.upper_surface_points is not None and self.lower_surface_points is not None):
                raise ValueError(
                    "geometry_mode='generic_surface_points' requires geometry_points or both upper_surface_points and lower_surface_points."
                )
        if (self.upper_surface_points is None) != (self.lower_surface_points is None):
            raise ValueError("upper_surface_points and lower_surface_points must be provided together.")
        return self


class PredictionResponse(BaseModel):
    predicted_fields: List[List[float]]
    predicted_scalars: Dict[str, float]
    surface_cp: Optional[List[float]] = None
    metadata: Dict[str, Any]


class BatchPredictionRequest(BaseModel):
    items: List[PredictionRequest]
