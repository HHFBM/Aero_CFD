"""File-based CFD dataset readers."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd

from cfd_operator.config.schemas import DataConfig
from cfd_operator.geometry import BranchInputAdapter
from cfd_operator.output_semantics import format_missing_fields_message


def load_dataset_payload(path: str | Path, config: DataConfig | None = None) -> dict[str, Any]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {input_path}")

    if suffix == ".npz":
        with np.load(input_path, allow_pickle=True) as payload:
            return {key: payload[key] for key in payload.files}

    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(input_path)

    if suffix == ".parquet":
        table = pd.read_parquet(input_path)
        return _group_tabular_records(table, config=config)

    if suffix == ".csv":
        table = pd.read_csv(input_path)
        return _group_tabular_records(table, config=config)

    raise ValueError(f"Unsupported dataset format: {suffix}")


def _optional_distinct_points(frame: pd.DataFrame, x_col: str, y_col: str) -> np.ndarray | None:
    if x_col not in frame.columns or y_col not in frame.columns:
        return None
    points = frame[[x_col, y_col]].dropna().drop_duplicates().to_numpy(dtype=np.float32)
    if points.size == 0:
        return None
    return points


def _estimate_surface_normals(surface_points: np.ndarray) -> np.ndarray:
    if surface_points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    prev_points = np.roll(surface_points, 1, axis=0)
    next_points = np.roll(surface_points, -1, axis=0)
    tangent = next_points - prev_points
    normals = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm = np.maximum(norm, 1.0e-6)
    normals = normals / norm
    centroid = surface_points.mean(axis=0, keepdims=True)
    direction = surface_points - centroid
    flip_mask = (normals * direction).sum(axis=1, keepdims=True) < 0.0
    normals = np.where(flip_mask, -normals, normals)
    return normals.astype(np.float32)


def _build_farfield_mask(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    x = points[:, 0]
    y = points[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_pad = 0.08 * max(x_max - x_min, 1.0e-6)
    y_pad = 0.08 * max(y_max - y_min, 1.0e-6)
    mask = (
        (x <= x_min + x_pad)
        | (x >= x_max - x_pad)
        | (y <= y_min + y_pad)
        | (y >= y_max - y_pad)
    )
    if not np.any(mask):
        centroid = points.mean(axis=0, keepdims=True)
        scores = np.linalg.norm(points - centroid, axis=1)
        mask[np.argmax(scores)] = True
    return mask.astype(np.float32)


def _parse_geometry_params(frame: pd.DataFrame) -> np.ndarray:
    parameter_columns = sorted(column for column in frame.columns if column.startswith("geom_param_"))
    if parameter_columns:
        return frame[parameter_columns].iloc[0].to_numpy(dtype=np.float32)
    if "geometry_params_json" in frame.columns:
        raw_value = frame["geometry_params_json"].iloc[0]
        if isinstance(raw_value, str) and raw_value.strip():
            return np.asarray(json.loads(raw_value), dtype=np.float32).reshape(-1)
    return np.zeros((4,), dtype=np.float32)


def _sample_scalar_targets(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((2,), dtype=np.float32)
    available = np.zeros((2,), dtype=np.float32)
    if "cl" in frame.columns:
        values[0] = float(frame["cl"].iloc[0])
        available[0] = 1.0
    if "cd" in frame.columns:
        values[1] = float(frame["cd"].iloc[0])
        available[1] = 1.0
    return values, available


def _infer_geometry_payload(
    frame: pd.DataFrame,
    config: DataConfig | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str, str, str]:
    geometry_mode = str(frame["geometry_mode"].iloc[0]) if "geometry_mode" in frame.columns else ""
    branch_columns = sorted(column for column in frame.columns if column.startswith("branch_"))
    has_legacy_params = {"max_camber", "camber_position", "thickness", "chord"}.issubset(frame.columns)
    branch_encoding_type = "precomputed_branch_columns"
    geometry_points = _optional_distinct_points(frame, "geometry_x", "geometry_y")
    if geometry_points is None and "surface_flag" in frame.columns:
        geometry_points = frame.loc[frame["surface_flag"] == 1, ["x", "y"]].to_numpy(dtype=np.float32)

    if branch_columns:
        branch_inputs = frame[branch_columns].iloc[0].to_numpy(dtype=np.float32)
    elif has_legacy_params:
        geometry_params = frame[["max_camber", "camber_position", "thickness", "chord"]].iloc[0].to_numpy(dtype=np.float32)
        if config is not None:
            adapter = BranchInputAdapter(
                branch_input_mode=config.branch_input_mode,
                branch_feature_mode=config.branch_feature_mode,
                signature_points=config.num_surface_points,
                encoded_geometry_latent_dim=config.encoded_geometry_latent_dim,
            )
            branch_inputs = adapter.build_from_geometry_params(
                geometry_params,
                mach=float(frame["mach"].iloc[0]),
                aoa_deg=float(frame["aoa"].iloc[0]),
                reynolds=float(frame["reynolds"].iloc[0]) if config.include_reynolds and "reynolds" in frame.columns else None,
                surface_points=geometry_points,
            )
        else:
            branch_inputs = frame[
                ["max_camber", "camber_position", "thickness", "chord", "mach", "aoa"]
            ].iloc[0].to_numpy(dtype=np.float32)
    elif geometry_points is not None and geometry_points.shape[0] >= 4 and config is not None:
        adapter = BranchInputAdapter(
            branch_input_mode=config.branch_input_mode,
            branch_feature_mode=config.branch_feature_mode,
            signature_points=config.num_surface_points,
            encoded_geometry_latent_dim=config.encoded_geometry_latent_dim,
        )
        branch_inputs = adapter.build_from_surface_points(
            geometry_points,
            mach=float(frame["mach"].iloc[0]),
            aoa_deg=float(frame["aoa"].iloc[0]),
            reynolds=float(frame["reynolds"].iloc[0]) if config.include_reynolds and "reynolds" in frame.columns else None,
        )
        if config.branch_input_mode == "encoded_geometry":
            branch_encoding_type = "encoded_geometry_compatible_features"
        else:
            branch_encoding_type = (
                "derived_geometry_summary_plus_flow"
                if config.branch_feature_mode == "params"
                else "derived_surface_signature_plus_flow"
            )
    else:
        raise ValueError(
            "Generic tabular file datasets without legacy geometry_params require either precomputed branch_* columns, "
            "or a DataConfig-driven geometry encoder path with usable geometry points."
        )

    if has_legacy_params:
        geometry_params = frame[["max_camber", "camber_position", "thickness", "chord"]].iloc[0].to_numpy(dtype=np.float32)
        default_mode = "legacy_naca_params"
        representation = "parameterized_geometry"
        params_semantics = "naca4_parameter_vector"
        branch_encoding_type = (
            "encoded_geometry_compatible_features"
            if config is not None and config.branch_input_mode == "encoded_geometry"
            else "naca_parameter_vector_plus_flow"
        )
        reconstructability = "safe_from_geometry_params"
    else:
        geometry_params = _parse_geometry_params(frame)
        default_mode = "generic_surface_points"
        representation = "sampled_surface_signature"
        params_semantics = "generic_geometry_metadata_only"
        reconstructability = "surface_points_only"

    return (
        geometry_params,
        branch_inputs,
        geometry_points if geometry_points is not None else np.zeros((0, 2), dtype=np.float32),
        geometry_mode or default_mode,
        representation,
        params_semantics,
        branch_encoding_type,
        reconstructability,
    )


def _group_tabular_records(table: pd.DataFrame, config: DataConfig | None = None) -> dict[str, Any]:
    if "sample_id" not in table.columns:
        raise ValueError("CSV/Parquet datasets require a 'sample_id' column.")
    required_columns = {
        "mach",
        "aoa",
        "x",
        "y",
        "u",
        "v",
        "p",
    }
    missing_columns = required_columns.difference(table.columns)
    if missing_columns:
        raise ValueError(format_missing_fields_message("Tabular CFD dataset", missing_columns))
    field_name_candidates = ["u", "v", "p"]
    if "nut" in table.columns:
        field_name_candidates.append("nut")
    elif "rho" in table.columns:
        field_name_candidates.append("rho")
    else:
        field_name_candidates.append("aux")

    grouped = list(table.groupby("sample_id"))
    if not grouped:
        raise ValueError("Tabular CFD dataset contains no grouped samples.")

    samples = []
    for sample_id, frame in grouped:
        has_precomputed_branch = any(column.startswith("branch_") for column in frame.columns)
        if field_name_candidates[3] == "aux":
            auxiliary_values = np.zeros((frame.shape[0],), dtype=np.float32)
        else:
            auxiliary_values = frame[field_name_candidates[3]].to_numpy(dtype=np.float32)

        geometry_params, branch_inputs, geometry_points, geometry_mode, geometry_representation, geometry_params_semantics, branch_encoding_type, geometry_reconstructability = _infer_geometry_payload(frame, config=config)
        surface_mask = (
            frame["surface_flag"].to_numpy(dtype=np.float32) > 0.5
            if "surface_flag" in frame.columns
            else np.zeros((frame.shape[0],), dtype=bool)
        )
        surface_points = frame.loc[surface_mask, ["x", "y"]].to_numpy(dtype=np.float32)
        if surface_points.shape[0] == 0 and geometry_points.shape[0] > 0:
            surface_points = geometry_points
        surface_count = surface_points.shape[0]
        surface_normals = _estimate_surface_normals(surface_points)
        query_points = frame[["x", "y"]].to_numpy(dtype=np.float32)
        query_count = query_points.shape[0]
        farfield_mask = _build_farfield_mask(query_points)
        scalar_targets, scalar_targets_available = _sample_scalar_targets(frame)
        if "cp" in frame.columns and np.any(surface_mask):
            surface_cp = frame.loc[surface_mask, ["cp"]].to_numpy(dtype=np.float32)
            surface_cp_available = np.ones((surface_count,), dtype=np.float32)
        else:
            surface_cp = np.zeros((surface_count, 1), dtype=np.float32)
            surface_cp_available = np.zeros((surface_count,), dtype=np.float32)
        samples.append(
            {
                "airfoil_id": str(sample_id),
                "geometry_params": geometry_params,
                "geometry_mode": geometry_mode,
                "geometry_source": str(frame["source"].iloc[0]) if "source" in frame.columns else "generic_tabular",
                "geometry_representation": geometry_representation,
                "branch_encoding_type": (
                    f"geometry_preprocessed_{config.branch_feature_mode}"
                    if branch_encoding_type == "precomputed_branch_columns" and config is not None and not any(column.startswith("branch_") for column in frame.columns)
                    else branch_encoding_type
                ),
                "geometry_reconstructability": geometry_reconstructability,
                "geometry_params_semantics": geometry_params_semantics,
                "legacy_param_source": "tabular_file",
                "branch_input_mode": (
                    config.branch_input_mode if config is not None else "legacy_fixed_features"
                ),
                "branch_input_source": (
                    "precomputed_branch_columns"
                    if has_precomputed_branch
                    else (
                        "encoded_geometry"
                    if config is not None and config.branch_input_mode == "encoded_geometry"
                    else "legacy_fixed_features"
                    )
                ),
                "geometry_points": geometry_points if geometry_points.size > 0 else surface_points,
                "geometry_encoding_meta": "",
                "surface_sampling_info": "",
                "flow_conditions": frame[["mach", "aoa"]].iloc[0].to_numpy(dtype=np.float32),
                "branch_inputs": branch_inputs,
                "query_points": query_points,
                "field_targets": np.stack(
                    [
                        frame["u"].to_numpy(dtype=np.float32),
                        frame["v"].to_numpy(dtype=np.float32),
                        frame["p"].to_numpy(dtype=np.float32),
                        auxiliary_values,
                    ],
                    axis=1,
                ),
                "farfield_mask": farfield_mask,
                "farfield_targets": np.zeros((4,), dtype=np.float32),
                "surface_points": surface_points,
                "surface_normals": surface_normals,
                "surface_arc_length": np.zeros((surface_count, 1), dtype=np.float32),
                "cp_reference": np.asarray([0.0, 1.0], dtype=np.float32),
                "surface_cp": surface_cp,
                "surface_pressure": np.zeros((surface_count, 1), dtype=np.float32),
                "surface_velocity": np.zeros((surface_count, 2), dtype=np.float32),
                "surface_nut": np.zeros((surface_count, 1), dtype=np.float32),
                "surface_heat_flux": np.zeros((surface_count, 1), dtype=np.float32),
                "surface_wall_shear": np.zeros((surface_count, 1), dtype=np.float32),
                "slice_points": np.zeros((0, 2), dtype=np.float32),
                "slice_fields": np.zeros((0, 4), dtype=np.float32),
                "pressure_gradient_indicator": np.zeros((query_count, 1), dtype=np.float32),
                "shock_indicator": np.zeros((query_count, 1), dtype=np.float32),
                "high_gradient_mask": np.zeros((query_count, 1), dtype=np.float32),
                "shock_location": np.asarray([np.nan, np.nan], dtype=np.float32),
                "surface_cp_available": surface_cp_available,
                "surface_pressure_available": np.zeros((surface_count,), dtype=np.float32),
                "surface_velocity_available": np.zeros((surface_count,), dtype=np.float32),
                "surface_nut_available": np.zeros((surface_count,), dtype=np.float32),
                "surface_heat_flux_available": np.zeros((surface_count,), dtype=np.float32),
                "surface_wall_shear_available": np.zeros((surface_count,), dtype=np.float32),
                "slice_available": np.zeros((0,), dtype=np.float32),
                "feature_available": np.zeros((query_count,), dtype=np.float32),
                "nut_available": np.zeros((query_count,), dtype=np.float32),
                "shock_location_available": np.asarray([0.0], dtype=np.float32),
                "scalar_targets": scalar_targets,
                "scalar_targets_available": scalar_targets_available,
                "scalar_component_targets": np.zeros((5,), dtype=np.float32),
                "scalar_component_available": np.zeros((5,), dtype=np.float32),
                "fidelity_level": int(frame["fidelity_level"].iloc[0]) if "fidelity_level" in frame.columns else 0,
                "source": str(frame["source"].iloc[0]) if "source" in frame.columns else "generic_tabular",
                "convergence_flag": int(frame["convergence_flag"].iloc[0]) if "convergence_flag" in frame.columns else 1,
            }
        )

    branch_dim = int(samples[0]["branch_inputs"].shape[-1])
    query_count = int(samples[0]["query_points"].shape[0])
    surface_count = int(samples[0]["surface_points"].shape[0])
    geometry_point_count = int(samples[0]["geometry_points"].shape[0])
    payload = {
        "airfoil_id": np.asarray([sample["airfoil_id"] for sample in samples]),
        "geometry_params": np.stack([sample["geometry_params"] for sample in samples]),
        "geometry_mode": np.asarray([sample["geometry_mode"] for sample in samples]),
        "geometry_source": np.asarray([sample["geometry_source"] for sample in samples]),
        "geometry_representation": np.asarray([sample["geometry_representation"] for sample in samples]),
        "branch_encoding_type": np.asarray([sample["branch_encoding_type"] for sample in samples]),
        "geometry_reconstructability": np.asarray([sample["geometry_reconstructability"] for sample in samples]),
        "geometry_params_semantics": np.asarray([sample["geometry_params_semantics"] for sample in samples]),
        "legacy_param_source": np.asarray([sample["legacy_param_source"] for sample in samples]),
        "geometry_points": np.stack([sample["geometry_points"] for sample in samples]).reshape(len(samples), geometry_point_count, 2),
        "geometry_encoding_meta": np.asarray([sample["geometry_encoding_meta"] for sample in samples]),
        "surface_sampling_info": np.asarray([sample["surface_sampling_info"] for sample in samples]),
        "flow_conditions": np.stack([sample["flow_conditions"] for sample in samples]),
        "branch_inputs": np.stack([sample["branch_inputs"] for sample in samples]).reshape(len(samples), branch_dim),
        "query_points": np.stack([sample["query_points"] for sample in samples]).reshape(len(samples), query_count, 2),
        "field_targets": np.stack([sample["field_targets"] for sample in samples]),
        "farfield_mask": np.stack([sample["farfield_mask"] for sample in samples]).reshape(len(samples), query_count),
        "farfield_targets": np.stack([sample["farfield_targets"] for sample in samples]),
        "surface_points": np.stack([sample["surface_points"] for sample in samples]).reshape(len(samples), surface_count, 2),
        "surface_normals": np.stack([sample["surface_normals"] for sample in samples]).reshape(len(samples), surface_count, 2),
        "surface_arc_length": np.stack([sample["surface_arc_length"] for sample in samples]).reshape(len(samples), surface_count, 1),
        "cp_reference": np.stack([sample["cp_reference"] for sample in samples]),
        "surface_cp": np.stack([sample["surface_cp"] for sample in samples]).reshape(len(samples), surface_count, 1),
        "surface_cp_available": np.stack([sample["surface_cp_available"] for sample in samples]).reshape(len(samples), surface_count),
        "surface_pressure": np.stack([sample["surface_pressure"] for sample in samples]).reshape(len(samples), surface_count, 1),
        "surface_velocity": np.stack([sample["surface_velocity"] for sample in samples]).reshape(len(samples), surface_count, 2),
        "surface_nut": np.stack([sample["surface_nut"] for sample in samples]).reshape(len(samples), surface_count, 1),
        "surface_heat_flux": np.stack([sample["surface_heat_flux"] for sample in samples]).reshape(len(samples), surface_count, 1),
        "surface_wall_shear": np.stack([sample["surface_wall_shear"] for sample in samples]).reshape(len(samples), surface_count, 1),
        "slice_points": np.zeros((len(samples), 0, 2), dtype=np.float32),
        "slice_fields": np.zeros((len(samples), 0, 4), dtype=np.float32),
        "pressure_gradient_indicator": np.stack([sample["pressure_gradient_indicator"] for sample in samples]).reshape(len(samples), query_count, 1),
        "shock_indicator": np.stack([sample["shock_indicator"] for sample in samples]).reshape(len(samples), query_count, 1),
        "high_gradient_mask": np.stack([sample["high_gradient_mask"] for sample in samples]).reshape(len(samples), query_count, 1),
        "shock_location": np.stack([sample["shock_location"] for sample in samples]).reshape(len(samples), 2),
        "surface_pressure_available": np.stack([sample["surface_pressure_available"] for sample in samples]).reshape(len(samples), surface_count),
        "surface_velocity_available": np.stack([sample["surface_velocity_available"] for sample in samples]).reshape(len(samples), surface_count),
        "surface_nut_available": np.stack([sample["surface_nut_available"] for sample in samples]).reshape(len(samples), surface_count),
        "surface_heat_flux_available": np.stack([sample["surface_heat_flux_available"] for sample in samples]).reshape(len(samples), surface_count),
        "surface_wall_shear_available": np.stack([sample["surface_wall_shear_available"] for sample in samples]).reshape(len(samples), surface_count),
        "slice_available": np.zeros((len(samples), 0), dtype=np.float32),
        "feature_available": np.stack([sample["feature_available"] for sample in samples]).reshape(len(samples), query_count),
        "nut_available": np.stack([sample["nut_available"] for sample in samples]).reshape(len(samples), query_count),
        "shock_location_available": np.stack([sample["shock_location_available"] for sample in samples]).reshape(len(samples), 1),
        "scalar_targets": np.stack([sample["scalar_targets"] for sample in samples]),
        "scalar_targets_available": np.stack([sample["scalar_targets_available"] for sample in samples]),
        "scalar_component_targets": np.stack([sample["scalar_component_targets"] for sample in samples]),
        "scalar_component_available": np.stack([sample["scalar_component_available"] for sample in samples]),
        "field_names": np.asarray(field_name_candidates),
        "fidelity_level": np.asarray([sample["fidelity_level"] for sample in samples], dtype=np.int64),
        "source": np.asarray([sample["source"] for sample in samples]),
        "convergence_flag": np.asarray([sample["convergence_flag"] for sample in samples], dtype=np.int64),
    }
    num_samples = len(samples)
    indices = np.arange(num_samples, dtype=np.int64)
    train_end = max(1, int(round(num_samples * 0.7)))
    val_end = min(num_samples, train_end + max(1, int(round(num_samples * 0.15))))
    payload["train_indices"] = indices[:train_end]
    payload["val_indices"] = indices[train_end:val_end]
    payload["test_indices"] = indices[val_end:]
    return payload
