"""Run single-sample inference from a JSON/YAML input file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml

from cfd_operator.inference import Predictor


def _load_input(path: Union[str, Path]) -> Dict[str, Any]:
    input_path = Path(path)
    if input_path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(input_path.read_text(encoding="utf-8"))
    return json.loads(input_path.read_text(encoding="utf-8"))


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


def _save_output(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    suffix = output_path.suffix.lower()
    field_names = payload.get("metadata", {}).get("field_names", ["u", "v", "p", "nut"])
    if suffix == ".json":
        output_path.write_text(json.dumps(_jsonify(payload), indent=2), encoding="utf-8")
        return
    if suffix == ".npz":
        np.savez(
            output_path,
            predicted_fields=np.asarray(payload["predicted_fields"], dtype=np.float32),
            **{
                key: np.float32(value)
                for key, value in payload["predicted_scalars"].items()
            },
        )
        return
    if suffix == ".csv":
        import pandas as pd

        frame = pd.DataFrame(payload["predicted_fields"], columns=field_names)
        frame.to_csv(output_path, index=False)
        return
    raise ValueError(f"Unsupported output format: {suffix}")


def _default_query_points(num_x: int = 32, num_y: int = 24) -> np.ndarray:
    x = np.linspace(-0.5, 1.5, num=num_x, dtype=np.float32)
    y = np.linspace(-0.6, 0.6, num=num_y, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)


def _resolve_geometry_mode(payload: Dict[str, Any]) -> Optional[str]:
    if payload.get("geometry_mode") is not None:
        return str(payload["geometry_mode"])
    if payload.get("geometry_points") is not None or payload.get("airfoil_surface_points") is not None:
        return "generic_surface_points"
    if payload.get("upper_surface_points") is not None or payload.get("lower_surface_points") is not None:
        return "generic_surface_points"
    if payload.get("geometry_params") is not None or payload.get("geometry") is not None:
        return "legacy_naca_params"
    return None


def _resolve_geometry_params(payload: Dict[str, Any]) -> Optional[np.ndarray]:
    geometry = payload.get("geometry_params", payload.get("geometry"))
    if geometry is None:
        return None
    return np.asarray(geometry, dtype=np.float32)


def _resolve_geometry_points(payload: Dict[str, Any]) -> Optional[np.ndarray]:
    geometry_points = payload.get("geometry_points", payload.get("airfoil_surface_points"))
    if geometry_points is None:
        return None
    return np.asarray(geometry_points, dtype=np.float32)


def _validate_geometry_payload(payload: Dict[str, Any], geometry_mode: Optional[str]) -> None:
    has_upper = payload.get("upper_surface_points") is not None
    has_lower = payload.get("lower_surface_points") is not None
    if has_upper != has_lower:
        raise ValueError("upper_surface_points and lower_surface_points must be provided together.")
    if geometry_mode == "legacy_naca_params" and payload.get("geometry_params") is None and payload.get("geometry") is None:
        raise ValueError("geometry_mode='legacy_naca_params' requires geometry_params.")
    if geometry_mode == "structured_param_vector" and payload.get("geometry_params") is None and payload.get("geometry") is None:
        raise ValueError("geometry_mode='structured_param_vector' requires geometry_params.")


def _resolve_aoa(payload: Dict[str, Any]) -> float:
    if payload.get("aoa") is not None:
        return float(payload["aoa"])
    if payload.get("aoa_deg") is not None:
        return float(payload["aoa_deg"])
    raise ValueError("Input must provide 'aoa' or 'aoa_deg'.")


def _resolve_mach(payload: Dict[str, Any]) -> float:
    if payload.get("mach") is not None:
        return float(payload["mach"])
    if payload.get("freestream_velocity") is not None and payload.get("speed_of_sound") is not None:
        return float(payload["freestream_velocity"]) / max(float(payload["speed_of_sound"]), 1.0e-6)
    if (
        payload.get("reynolds") is not None
        and payload.get("kinematic_viscosity") is not None
        and payload.get("speed_of_sound") is not None
    ):
        freestream_velocity = float(payload["reynolds"]) * float(payload["kinematic_viscosity"])
        return freestream_velocity / max(float(payload["speed_of_sound"]), 1.0e-6)
    raise ValueError(
        "Input must provide 'mach', or enough freestream data to derive it: "
        "'freestream_velocity' + 'speed_of_sound', or "
        "'reynolds' + 'kinematic_viscosity' + 'speed_of_sound'."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single prediction.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--export-dir", default=None, help="Optional directory for analysis bundle export.")
    parser.add_argument("--no-surface", action="store_true", help="Disable surface output generation.")
    parser.add_argument("--no-slices", action="store_true", help="Disable slice output generation.")
    parser.add_argument("--no-features", action="store_true", help="Disable feature indicator generation.")
    args = parser.parse_args()

    predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)
    payload = _load_input(args.input)
    geometry_mode = _resolve_geometry_mode(payload)
    geometry_params = _resolve_geometry_params(payload)
    geometry_points = _resolve_geometry_points(payload)
    aoa_deg = _resolve_aoa(payload)
    mach = _resolve_mach(payload)
    if geometry_mode is None:
        raise ValueError(
            "Input must provide geometry_mode or enough geometry fields for inference: "
            "geometry_params for legacy_naca_params, or geometry_points / upper_surface_points + lower_surface_points "
            "for generic_surface_points."
        )
    _validate_geometry_payload(payload, geometry_mode)
    query_points = (
        np.asarray(payload["query_points"], dtype=np.float32)
        if payload.get("query_points") is not None
        else _default_query_points()
    )
    if payload.get("query_points") is None:
        print("query_points not provided; using a default 2D analysis grid on [-0.5, 1.5] x [-0.6, 0.6].")

    if payload.get("slice_definitions") is None and not args.no_slices:
        print("slice_definitions not provided; slice outputs are omitted.")

    result = predictor.predict_from_geometry(
        geometry_params=geometry_params,
        mach=mach,
        aoa_deg=aoa_deg,
        query_points=query_points,
        surface_points=np.asarray(payload["surface_points"], dtype=np.float32) if payload.get("surface_points") is not None else None,
        slice_definitions=payload.get("slice_definitions"),
        reynolds=payload.get("reynolds"),
        cp_reference=np.asarray(payload["cp_reference"], dtype=np.float32) if payload.get("cp_reference") is not None else None,
        freestream_velocity=payload.get("freestream_velocity"),
        include_surface=not args.no_surface,
        include_slices=not args.no_slices,
        include_features=not args.no_features,
        export_dir=args.export_dir,
        geometry_mode=geometry_mode,
        geometry_points=geometry_points,
        upper_surface_points=(
            np.asarray(payload["upper_surface_points"], dtype=np.float32)
            if payload.get("upper_surface_points") is not None
            else None
        ),
        lower_surface_points=(
            np.asarray(payload["lower_surface_points"], dtype=np.float32)
            if payload.get("lower_surface_points") is not None
            else None
        ),
    )
    if "surface_predictions" not in result and not args.no_surface:
        print("surface outputs were requested but are unavailable for this input/checkpoint combination.")
    if "slice_predictions" not in result and not args.no_slices:
        print("slice outputs were requested but no slice definitions were provided.")
    if "feature_predictions" not in result and not args.no_features:
        print("feature outputs were requested but the checkpoint/result does not provide them.")
    _save_output(args.output, result)
    print(f"Saved inference results to {args.output}")
    if args.export_dir:
        print(f"Saved analysis bundle to {args.export_dir}")


if __name__ == "__main__":
    main()
