"""Export helpers for analysis-oriented inference bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from cfd_operator.utils.io import ensure_dir, save_json


def export_analysis_bundle(output_dir: str | Path, payload: Dict[str, Any]) -> Path:
    """Export a compact analysis bundle to a directory."""

    bundle_dir = ensure_dir(output_dir)
    predictions_json = bundle_dir / "predictions.json"
    save_json(predictions_json, _jsonify(payload))
    if "predicted_scalars" in payload:
        save_json(bundle_dir / "scalar_summary.json", _jsonify(payload["predicted_scalars"]))

    if "surface_predictions" in payload:
        surface = payload["surface_predictions"]
        frame_payload = {
            "x": np.asarray(surface["surface_points"])[:, 0],
            "y": np.asarray(surface["surface_points"])[:, 1],
            "cp_surface": np.asarray(surface["cp_surface"]).reshape(-1),
            "pressure_surface": np.asarray(surface["pressure_surface"]).reshape(-1),
        }
        if "heat_flux_surface" in surface:
            frame_payload["heat_flux_surface"] = np.asarray(surface["heat_flux_surface"]).reshape(-1)
        if "wall_shear_surface" in surface:
            frame_payload["wall_shear_surface"] = np.asarray(surface["wall_shear_surface"]).reshape(-1)
        if "velocity_surface" in surface:
            frame_payload["u_surface"] = np.asarray(surface["velocity_surface"])[:, 0]
            frame_payload["v_surface"] = np.asarray(surface["velocity_surface"])[:, 1]
        if "nut_surface" in surface:
            frame_payload["nut_surface"] = np.asarray(surface["nut_surface"]).reshape(-1)
        surface_frame = pd.DataFrame(frame_payload)
        surface_frame.to_csv(bundle_dir / "surface_values.csv", index=False)

    if payload.get("slice_predictions"):
        field_names = payload.get("metadata", {}).get("field_names", ["u", "v", "p", "nut"])
        rows: list[dict[str, float | str | int]] = []
        for index, slice_payload in enumerate(payload["slice_predictions"]):
            fields = np.asarray(slice_payload["slice_fields"], dtype=np.float32)
            points = np.asarray(slice_payload["slice_points"], dtype=np.float32)
            for point_index in range(points.shape[0]):
                row: dict[str, float | str | int] = {
                    "slice_id": index,
                    "point_id": point_index,
                    "x": float(points[point_index, 0]),
                    "y": float(points[point_index, 1]),
                }
                for field_index, field_name in enumerate(field_names):
                    row[str(field_name)] = float(fields[point_index, field_index])
                rows.append(row)
        pd.DataFrame(rows).to_csv(bundle_dir / "slice_values.csv", index=False)

    if "feature_predictions" in payload:
        save_json(bundle_dir / "feature_summary.json", _jsonify(payload["feature_predictions"]))

    return bundle_dir


def _jsonify(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _jsonify(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_jsonify(value) for value in payload]
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, (np.floating, np.integer)):
        return payload.item()
    return payload
