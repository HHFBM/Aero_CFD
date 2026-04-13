"""Checkpoint-backed inference helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from cfd_operator.config.schemas import ProjectConfig
from cfd_operator.data.module import NormalizerBundle
from cfd_operator.geometry import NACA4Airfoil, build_branch_features
from cfd_operator.losses import pressure_to_cp
from cfd_operator.models import create_model


@dataclass
class Predictor:
    config: ProjectConfig
    normalizers: NormalizerBundle
    model: torch.nn.Module
    device: str = "cpu"

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Union[str, Path], device: str = "cpu") -> "Predictor":
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = ProjectConfig(**checkpoint["config"])
        normalizers = NormalizerBundle.from_dict(checkpoint["normalizers"])
        model = create_model(config.model)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        return cls(config=config, normalizers=normalizers, model=model, device=device)

    def predict(
        self,
        branch_inputs_raw: np.ndarray,
        query_points_raw: np.ndarray,
        flow_conditions: np.ndarray,
        surface_points_raw: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        device = torch.device(self.device)
        branch_inputs = torch.from_numpy(self.normalizers.branch.transform(branch_inputs_raw.astype(np.float32))).to(device).unsqueeze(0)
        query_points = torch.from_numpy(self.normalizers.coordinates.transform(query_points_raw.astype(np.float32))).to(device).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model.loss_outputs(branch_inputs, query_points)
            predicted_fields = self.normalizers.fields.inverse_transform_tensor(outputs["fields"]).cpu().numpy()[0]
            predicted_scalars = self.normalizers.scalars.inverse_transform_tensor(outputs["scalars"]).cpu().numpy()[0]

            result: Dict[str, Any] = {
                "predicted_fields": predicted_fields,
                "predicted_scalars": {"cl": float(predicted_scalars[0]), "cd": float(predicted_scalars[1])},
                "metadata": {
                    "model_name": self.config.model.name,
                    "flow_conditions": {
                        "mach": float(flow_conditions[0]),
                        "aoa": float(flow_conditions[1]),
                    },
                },
            }

            if surface_points_raw is not None:
                surface_points = torch.from_numpy(self.normalizers.coordinates.transform(surface_points_raw.astype(np.float32))).to(device).unsqueeze(0)
                surface_outputs = self.model.loss_outputs(branch_inputs, surface_points)
                surface_fields = self.normalizers.fields.inverse_transform_tensor(surface_outputs["fields"])
                pressure = surface_fields[..., 2:3]
                mach = torch.tensor([[[float(flow_conditions[0])]]], device=device, dtype=pressure.dtype)
                cp = pressure_to_cp(pressure=pressure, mach=mach).cpu().numpy()[0, :, 0]
                result["surface_cp"] = cp
        return result

    def predict_from_geometry(
        self,
        geometry_params: np.ndarray,
        mach: float,
        aoa_deg: float,
        query_points: np.ndarray,
        surface_points: Optional[np.ndarray] = None,
        reynolds: Optional[float] = None,
    ) -> Dict[str, Any]:
        if geometry_params.shape[0] < 3:
            raise ValueError("geometry_params must contain at least [max_camber, camber_position, thickness]")
        chord = float(geometry_params[3]) if geometry_params.shape[0] > 3 else 1.0
        airfoil = NACA4Airfoil(
            max_camber=float(geometry_params[0]),
            camber_position=float(geometry_params[1]),
            thickness=float(geometry_params[2]),
            chord=chord,
        )
        branch_inputs = build_branch_features(
            airfoil,
            mach=mach,
            aoa_deg=aoa_deg,
            reynolds=reynolds if self.config.data.include_reynolds else None,
            mode=self.config.data.branch_feature_mode,
        )
        if surface_points is None:
            surface_points = airfoil.surface_points(self.config.data.num_surface_points)
        return self.predict(
            branch_inputs_raw=branch_inputs.astype(np.float32),
            query_points_raw=query_points.astype(np.float32),
            flow_conditions=np.asarray([mach, aoa_deg], dtype=np.float32),
            surface_points_raw=surface_points.astype(np.float32),
        )
