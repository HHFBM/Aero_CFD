"""FastAPI application factory."""

from __future__ import annotations

import os

import numpy as np
from fastapi import FastAPI

from cfd_operator.api.schemas import BatchPredictionRequest, PredictionRequest, PredictionResponse
from cfd_operator.inference import Predictor


def create_app(checkpoint_path: str, device: str = "cpu") -> FastAPI:
    predictor = Predictor.from_checkpoint(checkpoint_path=checkpoint_path, device=device)
    app = FastAPI(title="CFD Operator API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        result = predictor.predict_from_geometry(
            geometry_params=np.asarray(request.geometry_params, dtype=np.float32) if request.geometry_params is not None else None,
            mach=request.mach,
            aoa_deg=request.aoa,
            query_points=np.asarray(request.query_points, dtype=np.float32),
            surface_points=np.asarray(request.surface_points, dtype=np.float32) if request.surface_points is not None else None,
            reynolds=request.reynolds,
            geometry_mode=request.geometry_mode,
            geometry_points=np.asarray(request.geometry_points, dtype=np.float32) if request.geometry_points is not None else None,
            upper_surface_points=(
                np.asarray(request.upper_surface_points, dtype=np.float32)
                if request.upper_surface_points is not None
                else None
            ),
            lower_surface_points=(
                np.asarray(request.lower_surface_points, dtype=np.float32)
                if request.lower_surface_points is not None
                else None
            ),
        )
        return PredictionResponse(
            predicted_fields=result["predicted_fields"].tolist(),
            predicted_scalars=result["predicted_scalars"],
            surface_cp=result.get("surface_cp").tolist() if result.get("surface_cp") is not None else None,
            metadata=result["metadata"],
        )

    @app.post("/predict_batch")
    def predict_batch(request: BatchPredictionRequest) -> dict[str, list[dict[str, object]]]:
        outputs = []
        for item in request.items:
            result = predictor.predict_from_geometry(
                geometry_params=np.asarray(item.geometry_params, dtype=np.float32) if item.geometry_params is not None else None,
                mach=item.mach,
                aoa_deg=item.aoa,
                query_points=np.asarray(item.query_points, dtype=np.float32),
                surface_points=np.asarray(item.surface_points, dtype=np.float32) if item.surface_points is not None else None,
                reynolds=item.reynolds,
                geometry_mode=item.geometry_mode,
                geometry_points=np.asarray(item.geometry_points, dtype=np.float32) if item.geometry_points is not None else None,
                upper_surface_points=(
                    np.asarray(item.upper_surface_points, dtype=np.float32)
                    if item.upper_surface_points is not None
                    else None
                ),
                lower_surface_points=(
                    np.asarray(item.lower_surface_points, dtype=np.float32)
                    if item.lower_surface_points is not None
                    else None
                ),
            )
            outputs.append(
                {
                    "predicted_fields": result["predicted_fields"].tolist(),
                    "predicted_scalars": result["predicted_scalars"],
                    "surface_cp": result.get("surface_cp").tolist() if result.get("surface_cp") is not None else None,
                    "metadata": result["metadata"],
                }
            )
        return {"items": outputs}

    return app


def create_app_from_env() -> FastAPI:
    checkpoint_path = os.environ.get("CFD_OPERATOR_CHECKPOINT")
    if not checkpoint_path:
        raise RuntimeError("CFD_OPERATOR_CHECKPOINT environment variable is required.")
    device = os.environ.get("CFD_OPERATOR_DEVICE", "cpu")
    return create_app(checkpoint_path=checkpoint_path, device=device)
