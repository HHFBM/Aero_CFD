"""AirfRANS dataset conversion into the project NPZ schema."""

from __future__ import annotations

import json
import shutil
import ssl
import subprocess
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from cfd_operator.config.schemas import DataConfig
from cfd_operator.data.schemas import CFDSample
from cfd_operator.data.splitting import build_generalization_splits
from cfd_operator.geometry.semantics import airfrans_geometry_semantics
from cfd_operator.utils.io import ensure_dir, save_json


def _require_airfrans():
    try:
        import airfrans  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "AirfRANS support requires the optional dependency 'airfrans'. "
            "Install it with: python -m pip install airfrans"
        ) from exc
    return airfrans


def parse_airfrans_simulation_name(name: str) -> Dict[str, object]:
    parts = name.split("_")
    if len(parts) < 7:
        raise ValueError(f"Unexpected AirfRANS simulation name: {name}")

    inlet_velocity = float(parts[2])
    aoa_deg = float(parts[3])
    geometry_values = tuple(float(part) for part in parts[4:])
    if len(geometry_values) == 3:
        family_code = 4.0
        geometry_params = np.asarray(
            [family_code, geometry_values[0], geometry_values[1], geometry_values[2], 0.0],
            dtype=np.float32,
        )
    elif len(geometry_values) == 4:
        family_code = 5.0
        geometry_params = np.asarray(
            [family_code, geometry_values[0], geometry_values[1], geometry_values[2], geometry_values[3]],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unsupported NACA family in simulation name: {name}")

    geometry_id = f"naca-{int(family_code)}-" + "-".join(f"{value:.6f}" for value in geometry_values)
    return {
        "geometry_id": geometry_id,
        "geometry_params": geometry_params,
        "aoa_deg": aoa_deg,
        "inlet_velocity": inlet_velocity,
    }


def _build_farfield_mask(points: np.ndarray) -> np.ndarray:
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


def _cp_reference_for_airfrans(inlet_velocity: float) -> np.ndarray:
    return np.asarray([0.0, 0.5 * inlet_velocity**2], dtype=np.float32)


def _scalar_components_placeholder() -> np.ndarray:
    return np.zeros((5,), dtype=np.float32)


def _download_file_with_fallback(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        context = ssl.create_default_context()
        with urllib.request.urlopen(url, context=context) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        return
    except Exception:
        pass

    try:
        insecure_context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=insecure_context) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        return
    except Exception:
        pass

    curl_path = shutil.which("curl")
    if curl_path is None:
        raise RuntimeError("Failed to download AirfRANS dataset via urllib, and curl is not available.")
    subprocess.run([curl_path, "-L", "-k", url, "-o", str(destination)], check=True)


def _download_airfrans_dataset(root: Path, unzip: bool = True) -> None:
    archive_path = root / "Dataset.zip"
    url = "https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip"
    _download_file_with_fallback(url=url, destination=archive_path)
    if unzip:
        with zipfile.ZipFile(archive_path, "r") as zipf:
            zipf.extractall(root)


@dataclass
class AirfRANSDatasetConverter:
    config: DataConfig
    seed: int = 42

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def prepare_raw_dataset(self) -> Path:
        root = ensure_dir(self.config.airfrans_root)
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            return root

        if not self.config.airfrans_download:
            raise FileNotFoundError(
                f"AirfRANS manifest not found at {manifest_path}. "
                "Set data.airfrans_download=true or place the raw dataset there."
            )

        _download_airfrans_dataset(root=root, unzip=self.config.airfrans_unzip)
        if not manifest_path.exists():
            raise FileNotFoundError(f"AirfRANS download completed but manifest was not found at {manifest_path}")
        return root

    def _load_manifest_names(self, root: Path) -> List[str]:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        task = self.config.airfrans_task
        names = list(manifest[f"{task}_train"]) + list(manifest[f"{task}_test"])
        if self.config.airfrans_max_samples is not None:
            names = names[: self.config.airfrans_max_samples]
        return names

    def _build_sample(self, simulation_name: str, root: Path, seed: int) -> CFDSample:
        airfrans = _require_airfrans()
        simulation = airfrans.Simulation(root=str(root), name=simulation_name)
        info = parse_airfrans_simulation_name(simulation_name)
        geometry_semantics = airfrans_geometry_semantics(include_reynolds=self.config.include_reynolds)

        reynolds = float(simulation.inlet_velocity / simulation.NU)
        mach = float(simulation.inlet_velocity / simulation.C)
        aoa_deg = float(info["aoa_deg"])
        geometry_params = np.asarray(info["geometry_params"], dtype=np.float32)
        geometry_id = str(info["geometry_id"])

        mesh_sample = simulation.sampling_mesh(seed=seed, n=self.config.num_query_points, targets=True)
        query_points = mesh_sample[:, 0:2].astype(np.float32)
        velocity = mesh_sample[:, 8:10].astype(np.float32)
        pressure = mesh_sample[:, 10:11].astype(np.float32)
        nut = mesh_sample[:, 11:12].astype(np.float32) if mesh_sample.shape[1] >= 12 else np.zeros((mesh_sample.shape[0], 1), dtype=np.float32)
        field_targets = np.concatenate([velocity, pressure, nut], axis=1).astype(np.float32)

        surface_sample = simulation.sampling_surface(seed=seed + 10_000, n=self.config.num_surface_points, targets=True)
        surface_points = surface_sample[:, 0:2].astype(np.float32)
        surface_normals = surface_sample[:, 2:4].astype(np.float32)
        surface_velocity = surface_sample[:, 4:6].astype(np.float32) if surface_sample.shape[1] >= 6 else np.zeros((surface_sample.shape[0], 2), dtype=np.float32)
        surface_pressure = surface_sample[:, 6:7].astype(np.float32) if surface_sample.shape[1] >= 7 else np.zeros((surface_sample.shape[0], 1), dtype=np.float32)
        surface_nut = surface_sample[:, 7:8].astype(np.float32) if surface_sample.shape[1] >= 8 else np.zeros((surface_sample.shape[0], 1), dtype=np.float32)
        cp_reference = _cp_reference_for_airfrans(simulation.inlet_velocity)
        surface_cp = (surface_pressure - cp_reference[0]) / max(float(cp_reference[1]), 1.0e-6)
        surface_cp = surface_cp.astype(np.float32)

        cd_components, cl_components = simulation.force_coefficient(reference=True)
        scalar_targets = np.asarray([cl_components[0], cd_components[0]], dtype=np.float32)

        flow_conditions = np.asarray([mach, aoa_deg, reynolds], dtype=np.float32)
        if self.config.include_reynolds:
            branch_inputs = np.concatenate([geometry_params, flow_conditions], axis=0).astype(np.float32)
        else:
            branch_inputs = np.concatenate([geometry_params, flow_conditions[:2]], axis=0).astype(np.float32)

        alpha = np.deg2rad(aoa_deg)
        farfield_targets = np.asarray(
            [
                simulation.inlet_velocity * np.cos(alpha),
                simulation.inlet_velocity * np.sin(alpha),
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        return CFDSample(
            airfoil_id=geometry_id,
            geometry_params=geometry_params,
            flow_conditions=flow_conditions,
            branch_inputs=branch_inputs,
            query_points=query_points,
            field_targets=field_targets,
            farfield_mask=_build_farfield_mask(query_points),
            farfield_targets=farfield_targets,
            surface_points=surface_points,
            surface_normals=surface_normals,
            cp_reference=cp_reference,
            surface_cp=surface_cp,
            surface_pressure=surface_pressure,
            surface_velocity=surface_velocity,
            surface_nut=surface_nut,
            scalar_targets=scalar_targets,
            fidelity_level=1,
            source=f"airfrans:{self.config.airfrans_task}",
            convergence_flag=1,
            geometry_mode=geometry_semantics.geometry_mode,
            geometry_source=geometry_semantics.geometry_source,
            geometry_representation=geometry_semantics.geometry_representation,
            branch_encoding_type=geometry_semantics.branch_encoding_type,
            geometry_reconstructability=geometry_semantics.geometry_reconstructability,
            geometry_params_semantics=geometry_semantics.geometry_params_semantics,
            legacy_param_source=geometry_semantics.legacy_param_source,
            geometry_points=surface_points,
            geometry_encoding_meta=geometry_semantics.as_json(),
            surface_sampling_info='{"source":"airfrans_surface_sample","ordering":"dataset_native","normalized":false}',
        )

    def _split_indices(self, samples: Sequence[CFDSample]) -> Dict[str, np.ndarray]:
        rng = self._rng()
        geometry_ids = np.asarray([sample.airfoil_id for sample in samples])
        reynolds = np.asarray([sample.flow_conditions[2] for sample in samples], dtype=np.float32)
        aoa_values = np.asarray([sample.flow_conditions[1] for sample in samples], dtype=np.float32)
        return build_generalization_splits(
            geometry_ids=geometry_ids,
            primary_condition_values=reynolds,
            secondary_condition_values=aoa_values,
            unseen_geometry_ratio=self.config.unseen_geometry_ratio,
            unseen_condition_ratio=self.config.unseen_condition_ratio,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            rng=rng,
        )

    def generate_samples(self) -> List[CFDSample]:
        root = self.prepare_raw_dataset()
        names = self._load_manifest_names(root)
        samples = []
        for index, name in enumerate(names):
            samples.append(self._build_sample(name, root=root, seed=self.seed + index))
        return samples

    def save(self, path: str | Path) -> Path:
        samples = self.generate_samples()
        output_path = Path(path)
        ensure_dir(output_path.parent)
        num_samples = len(samples)
        branch_dim = samples[0].branch_inputs.shape[-1]
        num_query_points = samples[0].query_points.shape[0]
        num_surface_points = samples[0].surface_points.shape[0]
        payload = {
            "airfoil_id": np.asarray([sample.airfoil_id for sample in samples]),
            "geometry_params": np.stack([sample.geometry_params for sample in samples]),
            "geometry_mode": np.asarray([sample.geometry_mode for sample in samples]),
            "geometry_source": np.asarray([sample.geometry_source for sample in samples]),
            "geometry_representation": np.asarray([sample.geometry_representation for sample in samples]),
            "branch_encoding_type": np.asarray([sample.branch_encoding_type for sample in samples]),
            "geometry_reconstructability": np.asarray([sample.geometry_reconstructability for sample in samples]),
            "geometry_params_semantics": np.asarray([sample.geometry_params_semantics for sample in samples]),
            "legacy_param_source": np.asarray([sample.legacy_param_source for sample in samples]),
            "geometry_points": np.stack([sample.geometry_points for sample in samples]).reshape(num_samples, num_surface_points, 2),
            "geometry_encoding_meta": np.asarray([sample.geometry_encoding_meta for sample in samples]),
            "surface_sampling_info": np.asarray([sample.surface_sampling_info for sample in samples]),
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
            "surface_pressure": np.stack([sample.surface_pressure for sample in samples]).reshape(num_samples, num_surface_points, 1),
            "surface_velocity": np.stack([sample.surface_velocity for sample in samples]).reshape(num_samples, num_surface_points, 2),
            "surface_nut": np.stack([sample.surface_nut for sample in samples]).reshape(num_samples, num_surface_points, 1),
            "scalar_targets": np.stack([sample.scalar_targets for sample in samples]),
            "scalar_component_targets": np.zeros((num_samples, 5), dtype=np.float32),
            "scalar_component_available": np.zeros((num_samples, 5), dtype=np.float32),
            "surface_pressure_available": np.ones((num_samples, num_surface_points), dtype=np.float32),
            "surface_velocity_available": np.ones((num_samples, num_surface_points), dtype=np.float32),
            "surface_nut_available": np.ones((num_samples, num_surface_points), dtype=np.float32),
            "field_names": np.asarray(self.config.field_names),
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
                "source": "airfrans",
                "airfrans_task": self.config.airfrans_task,
                "num_samples": num_samples,
                "num_query_points": num_query_points,
                "num_surface_points": num_surface_points,
            },
        )
        return output_path
