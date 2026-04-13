"""Converter for raw AirfRANS tar archives."""

from __future__ import annotations

import csv
import hashlib
import io
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

from cfd_operator.config.schemas import DataConfig
from cfd_operator.data.airfrans import _build_farfield_mask
from cfd_operator.data.schemas import CFDSample
from cfd_operator.utils.io import ensure_dir, save_json


def _read_scalars_csv(blob: bytes) -> dict[str, float]:
    reader = csv.DictReader(io.StringIO(blob.decode("utf-8")))
    row = next(reader)
    return {key: float(value) for key, value in row.items()}


def _sample_ordered_surface(points: np.ndarray, num_points: int) -> np.ndarray:
    center = points.mean(axis=0, keepdims=True)
    angles = np.arctan2(points[:, 1] - center[0, 1], points[:, 0] - center[0, 0])
    order = np.argsort(angles)
    ordered = points[order]
    if ordered.shape[0] <= num_points:
        return ordered.astype(np.float32)
    indices = np.linspace(0, ordered.shape[0] - 1, num_points, dtype=np.int64)
    return ordered[indices].astype(np.float32)


def _normalize_airfoil_surface(surface_points: np.ndarray) -> np.ndarray:
    x = surface_points[:, 0]
    y = surface_points[:, 1]
    chord = max(float(x.max() - x.min()), 1.0e-6)
    x_norm = (x - float(x.min())) / chord
    y_center = 0.5 * (float(y.max()) + float(y.min()))
    y_norm = (y - y_center) / chord
    return np.stack([x_norm, y_norm], axis=1).astype(np.float32)


def _estimate_normals(surface_points: np.ndarray) -> np.ndarray:
    prev_points = np.roll(surface_points, 1, axis=0)
    next_points = np.roll(surface_points, -1, axis=0)
    tangent = next_points - prev_points
    normals = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm = np.maximum(norm, 1.0e-6)
    return (normals / norm).astype(np.float32)


def _geometry_summary(surface_points: np.ndarray) -> np.ndarray:
    x = surface_points[:, 0]
    y = surface_points[:, 1]
    return np.asarray(
        [
            float(x.min()),
            float(x.max()),
            float(y.min()),
            float(y.max()),
            float(y.max() - y.min()),
            float(y.mean()),
        ],
        dtype=np.float32,
    )


def _geometry_id(surface_points: np.ndarray) -> str:
    normalized = _normalize_airfoil_surface(surface_points)
    rounded = np.round(normalized, 2).astype(np.float32)
    digest = hashlib.md5(rounded.tobytes(), usedforsecurity=False).hexdigest()[:12]
    return f"raw-{digest}"


def _load_cgns_arrays(blob: bytes) -> dict[str, np.ndarray]:
    with tempfile.NamedTemporaryFile(suffix=".cgns") as handle:
        handle.write(blob)
        handle.flush()
        with h5py.File(handle.name, "r") as h5_file:
            def read(path: str) -> np.ndarray:
                return h5_file[path][()].astype(np.float32)

            return {
                "x": read("Base_2_2/Zone/GridCoordinates/CoordinateX/ data"),
                "y": read("Base_2_2/Zone/GridCoordinates/CoordinateY/ data"),
                "u": read("Base_2_2/Zone/VertexFields/Ux/ data"),
                "v": read("Base_2_2/Zone/VertexFields/Uy/ data"),
                "p": read("Base_2_2/Zone/VertexFields/p/ data"),
                "implicit_distance": read("Base_2_2/Zone/VertexFields/implicit_distance/ data"),
            }


@dataclass
class AirfRANSOriginalDatasetConverter:
    config: DataConfig
    seed: int = 42

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def _build_sample(self, sample_name: str, scalar_blob: bytes, mesh_blob: bytes, rng: np.random.Generator) -> CFDSample:
        scalar_values = _read_scalars_csv(scalar_blob)
        arrays = _load_cgns_arrays(mesh_blob)

        coords = np.stack([arrays["x"], arrays["y"]], axis=1).astype(np.float32)
        velocity = np.stack([arrays["u"], arrays["v"]], axis=1).astype(np.float32)
        pressure = arrays["p"].reshape(-1, 1).astype(np.float32)
        rho = np.ones((coords.shape[0], 1), dtype=np.float32)
        implicit_distance = arrays["implicit_distance"]

        query_count = min(self.config.num_query_points, coords.shape[0])
        query_indices = rng.choice(coords.shape[0], size=query_count, replace=False)
        query_points = coords[query_indices]

        surface_pool_count = min(max(self.config.num_surface_points * 8, self.config.num_surface_points), coords.shape[0])
        closest_indices = np.argsort(np.abs(implicit_distance))[:surface_pool_count]
        surface_pool = coords[closest_indices]
        surface_points = _sample_ordered_surface(surface_pool, self.config.num_surface_points)
        surface_points_norm = _normalize_airfoil_surface(surface_points)
        surface_normals = _estimate_normals(surface_points_norm)
        surface_geometry_id = _geometry_id(surface_points)

        # Use nearest sampled surface vertices to define Cp targets.
        surface_source = surface_pool if surface_pool.shape[0] > 0 else coords
        surface_dist = np.linalg.norm(surface_source[:, None, :] - surface_points[None, :, :], axis=2)
        nearest_surface_indices = np.argmin(surface_dist, axis=0)
        surface_reference_indices = closest_indices[nearest_surface_indices] if surface_pool.shape[0] > 0 else nearest_surface_indices

        aoa_deg = float(np.rad2deg(scalar_values["angle_of_attack"]))
        inlet_velocity = float(scalar_values["inlet_velocity"])
        mach = inlet_velocity / 340.0
        reynolds_proxy = inlet_velocity

        full_farfield_mask = _build_farfield_mask(coords)
        if np.any(full_farfield_mask > 0.5):
            p_inf = float(np.mean(pressure[full_farfield_mask > 0.5, 0]))
        else:
            p_inf = float(np.mean(pressure[:, 0]))
        dynamic_pressure = max(0.5 * inlet_velocity**2, 1.0e-6)
        cp_reference = np.asarray([p_inf, dynamic_pressure], dtype=np.float32)

        velocity_nd = velocity / max(inlet_velocity, 1.0e-6)
        pressure_nd = (pressure - p_inf) / dynamic_pressure
        field_targets = np.concatenate([velocity_nd[query_indices], pressure_nd[query_indices], rho[query_indices]], axis=1)
        farfield_mask = _build_farfield_mask(query_points)
        surface_pressure = pressure[surface_reference_indices]
        surface_cp = ((surface_pressure - p_inf) / dynamic_pressure).astype(np.float32)
        farfield_targets = np.asarray(
            [
                np.cos(np.deg2rad(aoa_deg)),
                np.sin(np.deg2rad(aoa_deg)),
                0.0,
                1.0,
            ],
            dtype=np.float32,
        )

        geometry_params = _geometry_summary(surface_points_norm)
        branch_inputs = np.concatenate(
            [
                surface_points_norm.reshape(-1).astype(np.float32),
                np.asarray([mach, aoa_deg], dtype=np.float32),
            ],
            axis=0,
        )

        flow_conditions = np.asarray([mach, aoa_deg, reynolds_proxy], dtype=np.float32)
        scalar_targets = np.asarray([scalar_values["C_L"], scalar_values["C_D"]], dtype=np.float32)

        return CFDSample(
            airfoil_id=surface_geometry_id,
            geometry_params=geometry_params,
            flow_conditions=flow_conditions,
            branch_inputs=branch_inputs,
            query_points=query_points,
            field_targets=field_targets,
            farfield_mask=farfield_mask,
            farfield_targets=farfield_targets,
            surface_points=surface_points_norm,
            surface_normals=surface_normals,
            cp_reference=cp_reference,
            surface_cp=surface_cp,
            scalar_targets=scalar_targets,
            fidelity_level=0,
            source=f"airfrans_original:{sample_name}",
            convergence_flag=1,
        )

    def _stream_samples(self) -> List[CFDSample]:
        archive_path = Path(self.config.airfrans_archive_path)
        if not archive_path.exists():
            raise FileNotFoundError(f"Raw AirfRANS archive not found: {archive_path}")

        rng = self._rng()
        pending: Dict[str, dict[str, bytes]] = {}
        samples: List[CFDSample] = []
        max_samples = self.config.airfrans_max_samples

        with tarfile.open(archive_path, "r") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = member.name
                if "/dataset/samples/" not in name:
                    continue
                if not (name.endswith("/scalars.csv") or name.endswith("/meshes/mesh_000000000.cgns")):
                    continue

                parts = Path(name).parts
                try:
                    sample_name = next(part for part in parts if part.startswith("sample_"))
                except StopIteration:
                    continue

                handle = archive.extractfile(member)
                if handle is None:
                    continue
                blob = handle.read()
                bucket = pending.setdefault(sample_name, {})
                if name.endswith("/scalars.csv"):
                    bucket["scalars"] = blob
                else:
                    bucket["mesh"] = blob

                if "scalars" in bucket and "mesh" in bucket:
                    samples.append(self._build_sample(sample_name, bucket["scalars"], bucket["mesh"], rng))
                    del pending[sample_name]
                    if max_samples is not None and len(samples) >= max_samples:
                        break

        if not samples:
            raise ValueError("No samples were extracted from the raw AirfRANS archive.")
        return samples

    def _split_indices(self, samples: List[CFDSample]) -> dict[str, np.ndarray]:
        rng = self._rng()
        geometry_ids = np.asarray([sample.airfoil_id for sample in samples])
        condition_primary = np.asarray([sample.flow_conditions[0] for sample in samples], dtype=np.float32)
        condition_secondary = np.asarray([sample.flow_conditions[1] for sample in samples], dtype=np.float32)
        unique_geometry_ids, counts = np.unique(geometry_ids, return_counts=True)
        num_holdout = max(1, int(round(unique_geometry_ids.shape[0] * self.config.unseen_geometry_ratio)))
        unseen_geometry_ids = rng.choice(unique_geometry_ids, size=min(num_holdout, unique_geometry_ids.shape[0]), replace=False)
        unseen_geometry_mask = np.isin(geometry_ids, unseen_geometry_ids)

        multi_geometry_ids = unique_geometry_ids[counts > 1]
        unseen_condition_mask = np.zeros(len(samples), dtype=bool)
        if multi_geometry_ids.size > 0 and self.config.unseen_condition_ratio > 0.0:
            repeated_mask = np.isin(geometry_ids, multi_geometry_ids) & ~unseen_geometry_mask
            repeated_indices = np.where(repeated_mask)[0]
            if repeated_indices.size > 0:
                sorted_repeated = repeated_indices[np.argsort(condition_primary[repeated_indices] + 0.1 * condition_secondary[repeated_indices])]
                num_condition_holdout = max(1, int(round(sorted_repeated.size * self.config.unseen_condition_ratio)))
                unseen_condition_mask[sorted_repeated[-num_condition_holdout:]] = True

        remaining = np.where(~unseen_geometry_mask & ~unseen_condition_mask)[0]
        rng.shuffle(remaining)
        total = remaining.size
        train_end = max(1, int(round(total * self.config.train_ratio)))
        val_end = min(total - 1, train_end + max(1, int(round(total * self.config.val_ratio))))

        train_indices = np.sort(remaining[:train_end])
        val_indices = np.sort(remaining[train_end:val_end])
        test_indices = np.sort(remaining[val_end:])

        split_payload = {
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
        }
        unseen_geometry_indices = np.sort(np.where(unseen_geometry_mask)[0])
        unseen_condition_indices = np.sort(np.where(unseen_condition_mask)[0])
        if unseen_geometry_indices.size > 0:
            split_payload["test_unseen_geometry_indices"] = unseen_geometry_indices
        if unseen_condition_indices.size > 0:
            split_payload["test_unseen_condition_indices"] = unseen_condition_indices
        return split_payload

    def save(self, path: str | Path) -> Path:
        samples = self._stream_samples()
        output_path = Path(path)
        ensure_dir(output_path.parent)
        num_samples = len(samples)
        branch_dim = samples[0].branch_inputs.shape[-1]
        num_query_points = samples[0].query_points.shape[0]
        num_surface_points = samples[0].surface_points.shape[0]

        payload = {
            "airfoil_id": np.asarray([sample.airfoil_id for sample in samples]),
            "geometry_params": np.stack([sample.geometry_params for sample in samples]),
            "flow_conditions": np.stack([sample.flow_conditions for sample in samples]),
            "branch_inputs": np.stack([sample.branch_inputs for sample in samples]).reshape(num_samples, branch_dim),
            "query_points": np.stack([sample.query_points for sample in samples]).reshape(num_samples, num_query_points, 2),
            "field_targets": np.stack([sample.field_targets for sample in samples]),
            "farfield_mask": np.stack([sample.farfield_mask for sample in samples]).reshape(num_samples, num_query_points),
            "farfield_targets": np.stack([sample.farfield_targets for sample in samples]),
            "surface_points": np.stack([sample.surface_points for sample in samples]).reshape(num_samples, num_surface_points, 2),
            "surface_normals": np.stack([sample.surface_normals for sample in samples]).reshape(num_samples, num_surface_points, 2),
            "cp_reference": np.stack([sample.cp_reference for sample in samples]),
            "surface_cp": np.stack([sample.surface_cp for sample in samples]).reshape(num_samples, num_surface_points, 1),
            "scalar_targets": np.stack([sample.scalar_targets for sample in samples]),
            "fidelity_level": np.asarray([sample.fidelity_level for sample in samples], dtype=np.int64),
            "source": np.asarray([sample.source for sample in samples]),
            "convergence_flag": np.asarray([sample.convergence_flag for sample in samples], dtype=np.int64),
        }
        payload.update(self._split_indices(samples))
        np.savez(output_path, **payload)
        save_json(
            output_path.with_suffix(".json"),
            {
                "dataset_path": str(output_path),
                "source": "airfrans_original",
                "archive_path": str(self.config.airfrans_archive_path),
                "num_samples": num_samples,
                "num_query_points": num_query_points,
                "num_surface_points": num_surface_points,
            },
        )
        return output_path
