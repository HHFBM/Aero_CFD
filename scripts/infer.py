"""Run single-sample inference from a JSON/YAML input file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import yaml

from cfd_operator.inference import Predictor


def _load_input(path: Union[str, Path]) -> Dict[str, Any]:
    input_path = Path(path)
    if input_path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(input_path.read_text(encoding="utf-8"))
    return json.loads(input_path.read_text(encoding="utf-8"))


def _save_output(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    suffix = output_path.suffix.lower()
    if suffix == ".json":
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return
    if suffix == ".npz":
        np.savez(
            output_path,
            predicted_fields=np.asarray(payload["predicted_fields"], dtype=np.float32),
            cl=np.float32(payload["predicted_scalars"]["cl"]),
            cd=np.float32(payload["predicted_scalars"]["cd"]),
            surface_cp=np.asarray(payload.get("surface_cp", []), dtype=np.float32),
        )
        return
    if suffix == ".csv":
        import pandas as pd

        frame = pd.DataFrame(payload["predicted_fields"], columns=["u", "v", "p", "rho"])
        frame.to_csv(output_path, index=False)
        return
    raise ValueError(f"Unsupported output format: {suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single prediction.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)
    payload = _load_input(args.input)
    result = predictor.predict_from_geometry(
        geometry_params=np.asarray(payload["geometry_params"], dtype=np.float32),
        mach=float(payload["mach"]),
        aoa_deg=float(payload["aoa"]),
        query_points=np.asarray(payload["query_points"], dtype=np.float32),
        surface_points=np.asarray(payload["surface_points"], dtype=np.float32) if payload.get("surface_points") is not None else None,
        reynolds=payload.get("reynolds"),
    )
    result["predicted_fields"] = result["predicted_fields"].tolist()
    if "surface_cp" in result:
        result["surface_cp"] = result["surface_cp"].tolist()
    _save_output(args.output, result)
    print(f"Saved inference results to {args.output}")


if __name__ == "__main__":
    main()
