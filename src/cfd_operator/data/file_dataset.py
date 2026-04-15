"""File-based CFD dataset readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_dataset_payload(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()

    if suffix == ".npz":
        with np.load(input_path, allow_pickle=True) as payload:
            return {key: payload[key] for key in payload.files}

    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(input_path)

    if suffix == ".parquet":
        table = pd.read_parquet(input_path)
        return _group_tabular_records(table)

    if suffix == ".csv":
        table = pd.read_csv(input_path)
        return _group_tabular_records(table)

    raise ValueError(f"Unsupported dataset format: {suffix}")


def _group_tabular_records(table: pd.DataFrame) -> dict[str, Any]:
    if "sample_id" not in table.columns:
        raise ValueError("CSV/Parquet datasets require a 'sample_id' column.")
    field_name_candidates = ["u", "v", "p"]
    if "nut" in table.columns:
        field_name_candidates.append("nut")
    elif "rho" in table.columns:
        field_name_candidates.append("rho")
    else:
        field_name_candidates.append("aux")

    grouped = table.groupby("sample_id")
    samples = []
    for sample_id, frame in grouped:
        if field_name_candidates[3] == "aux":
            auxiliary_values = np.zeros((frame.shape[0],), dtype=np.float32)
        else:
            auxiliary_values = frame[field_name_candidates[3]].to_numpy(dtype=np.float32)
        samples.append(
            {
                "airfoil_id": str(sample_id),
                "geometry_params": frame[["max_camber", "camber_position", "thickness", "chord"]].iloc[0].to_numpy(dtype=np.float32),
                "flow_conditions": frame[["mach", "aoa"]].iloc[0].to_numpy(dtype=np.float32),
                "branch_inputs": frame[
                    ["max_camber", "camber_position", "thickness", "chord", "mach", "aoa"]
                ].iloc[0].to_numpy(dtype=np.float32),
                "query_points": frame[["x", "y"]].to_numpy(dtype=np.float32),
                "field_targets": np.stack(
                    [
                        frame["u"].to_numpy(dtype=np.float32),
                        frame["v"].to_numpy(dtype=np.float32),
                        frame["p"].to_numpy(dtype=np.float32),
                        auxiliary_values,
                    ],
                    axis=1,
                ),
                "surface_points": frame.loc[frame["surface_flag"] == 1, ["x", "y"]].to_numpy(dtype=np.float32),
                "surface_cp": frame.loc[frame["surface_flag"] == 1, ["cp"]].to_numpy(dtype=np.float32),
                "scalar_targets": frame[["cl", "cd"]].iloc[0].to_numpy(dtype=np.float32),
                "fidelity_level": int(frame["fidelity_level"].iloc[0]),
                "source": str(frame["source"].iloc[0]),
                "convergence_flag": int(frame["convergence_flag"].iloc[0]),
            }
        )
    return {"samples": samples, "field_names": np.asarray(field_name_candidates)}
