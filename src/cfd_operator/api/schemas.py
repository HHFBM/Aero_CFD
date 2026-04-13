"""API request and response models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    geometry_params: List[float] = Field(..., description="[max_camber, camber_position, thickness, chord?]")
    mach: float
    aoa: float
    query_points: List[List[float]]
    surface_points: Optional[List[List[float]]] = None
    reynolds: Optional[float] = None


class PredictionResponse(BaseModel):
    predicted_fields: List[List[float]]
    predicted_scalars: Dict[str, float]
    surface_cp: Optional[List[float]] = None
    metadata: Dict[str, Any]


class BatchPredictionRequest(BaseModel):
    items: List[PredictionRequest]
