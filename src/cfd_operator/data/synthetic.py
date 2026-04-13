"""Synthetic toy CFD-like dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cfd_operator.config.schemas import DataConfig
from cfd_operator.data.schemas import CFDSample
from cfd_operator.geometry import NACA4Airfoil, build_branch_features
from cfd_operator.utils.io import ensure_dir, save_json


def _safe_beta(x: np.ndarray, scale: float) -> np.ndarray:
    return np.exp(-(x**2) / max(scale, 1.0e-6))


def _surface_cp_distribution(
    airfoil: NACA4Airfoil,
    surface_points: np.ndarray,
    mach: float,
    aoa_deg: float,
) -> np.ndarray:
    x = np.clip(surface_points[:, 0], 1.0e-3, 1.0)
    yc = airfoil.camber_line(x)
    upper_mask = surface_points[:, 1] >= yc
    alpha_eff = np.deg2rad(aoa_deg + 25.0 * airfoil.max_camber - 4.0 * (airfoil.camber_position - 0.4))
    compressibility = 1.0 / np.sqrt(max(1.0 - mach**2, 0.08))
    leading = np.clip(1.0 / np.sqrt(x), 0.0, 8.0)
    thickness_term = airfoil.thickness * np.cos(np.pi * x)
    camber_term = airfoil.max_camber * (1.0 - x)

    cp_upper = -compressibility * (
        0.9 * alpha_eff * leading * np.exp(-1.5 * x)
        + 0.6 * camber_term
        + 0.15 * thickness_term
    )
    cp_lower = compressibility * (
        0.55 * alpha_eff * leading * np.exp(-1.2 * x)
        - 0.25 * camber_term
        + 0.08 * thickness_term
    )
    cp = np.where(upper_mask, cp_upper, cp_lower)
    return cp[:, None].astype(np.float32)


def _lift_drag_coefficients(airfoil: NACA4Airfoil, mach: float, aoa_deg: float) -> np.ndarray:
    alpha = np.deg2rad(aoa_deg)
    beta = np.sqrt(max(1.0 - mach**2, 0.12))
    cl = (2.0 * np.pi * alpha / beta) * (1.0 + 3.5 * airfoil.max_camber + 0.3 * airfoil.thickness)
    drag_rise = max(mach - 0.72, 0.0) ** 2
    cd = 0.008 + 0.1 * airfoil.thickness + 0.018 * cl**2 + 0.35 * drag_rise
    return np.asarray([cl, cd], dtype=np.float32)


def _field_solution(
    airfoil: NACA4Airfoil,
    points: np.ndarray,
    mach: float,
    aoa_deg: float,
) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    x_airfoil = np.clip(x, 0.0, 1.0)
    yc = airfoil.camber_line(x_airfoil)
    yt = airfoil.thickness_distribution(x_airfoil)

    alpha = np.deg2rad(aoa_deg)
    gamma = 1.4
    v_inf = 1.0 + 0.6 * mach
    u_inf = v_inf * np.cos(alpha)
    v_inf_y = v_inf * np.sin(alpha)

    envelope = np.exp(-((x - 0.35) ** 2 / (0.08 + airfoil.thickness) ** 2 + (y - yc) ** 2 / (0.05 + 2.0 * yt) ** 2))
    wake = np.exp(-((x - 1.05) ** 2 / 0.18 + y**2 / 0.025))
    circulation = (0.5 * mach + 2.5 * alpha + 3.0 * airfoil.max_camber) * np.exp(-((x - 0.2) ** 2 / 0.25))

    u = u_inf * (1.0 - 0.28 * envelope + 0.04 * airfoil.thickness * np.sin(2.0 * np.pi * x)) - 0.06 * wake
    v = v_inf_y + circulation * y * np.exp(-np.abs(y) / (0.08 + 2.5 * yt + 1.0e-3))

    cp_like = (
        -0.85 * envelope
        + 0.25 * np.sin(alpha) * np.exp(-((x - 0.2) ** 2 / 0.12))
        - 0.15 * wake
        + 0.4 * airfoil.max_camber * np.exp(-((x - 0.45) ** 2 / 0.15))
    )
    p_inf = 1.0
    p = p_inf * np.clip(1.0 + 0.5 * gamma * mach**2 * cp_like, 0.35, 1.8)
    rho = np.clip((p / p_inf) ** (1.0 / gamma), 0.45, 1.6)
    return np.stack([u, v, p, rho], axis=1).astype(np.float32)


@dataclass(slots=True)
class SyntheticAirfoilDatasetGenerator:
    config: DataConfig
    seed: int = 42

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def generate_samples(self) -> list[CFDSample]:
        rng = self._rng()
        samples: list[CFDSample] = []
        for index in range(self.config.num_samples):
            max_camber = float(rng.uniform(0.0, 0.06))
            camber_position = float(rng.uniform(0.2, 0.6))
            thickness = float(rng.uniform(0.08, 0.18))
            mach = float(rng.uniform(*self.config.mach_range))
            aoa = float(rng.uniform(*self.config.aoa_range))

            airfoil = NACA4Airfoil(
                max_camber=max_camber,
                camber_position=camber_position,
                thickness=thickness,
            )

            query_points = np.empty((self.config.num_query_points, 2), dtype=np.float32)
            query_points[:, 0] = rng.uniform(-0.5, 1.5, size=self.config.num_query_points)
            query_points[:, 1] = rng.uniform(-0.6, 0.6, size=self.config.num_query_points)

            surface_points = airfoil.surface_points(self.config.num_surface_points)
            surface_cp = _surface_cp_distribution(airfoil, surface_points, mach=mach, aoa_deg=aoa)
            scalar_targets = _lift_drag_coefficients(airfoil, mach=mach, aoa_deg=aoa)
            field_targets = _field_solution(airfoil, query_points, mach=mach, aoa_deg=aoa)

            geometry_params = airfoil.parameter_vector()
            flow_conditions = np.asarray([mach, aoa], dtype=np.float32)
            branch_inputs = build_branch_features(
                airfoil,
                mach=mach,
                aoa_deg=aoa,
                reynolds=1.0e6 if self.config.include_reynolds else None,
                mode=self.config.branch_feature_mode,
            )

            samples.append(
                CFDSample(
                    airfoil_id=f"naca-like-{index:05d}",
                    geometry_params=geometry_params,
                    flow_conditions=flow_conditions,
                    branch_inputs=branch_inputs,
                    query_points=query_points,
                    field_targets=field_targets,
                    surface_points=surface_points,
                    surface_cp=surface_cp,
                    scalar_targets=scalar_targets,
                    fidelity_level=0,
                    source="synthetic_rule",
                    convergence_flag=1,
                )
            )
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
            "flow_conditions": np.stack([sample.flow_conditions for sample in samples]),
            "branch_inputs": np.stack([sample.branch_inputs for sample in samples]).reshape(num_samples, branch_dim),
            "query_points": np.stack([sample.query_points for sample in samples]).reshape(num_samples, num_query_points, 2),
            "field_targets": np.stack([sample.field_targets for sample in samples]),
            "surface_points": np.stack([sample.surface_points for sample in samples]).reshape(num_samples, num_surface_points, 2),
            "surface_cp": np.stack([sample.surface_cp for sample in samples]).reshape(num_samples, num_surface_points, 1),
            "scalar_targets": np.stack([sample.scalar_targets for sample in samples]),
            "fidelity_level": np.asarray([sample.fidelity_level for sample in samples], dtype=np.int64),
            "source": np.asarray([sample.source for sample in samples]),
            "convergence_flag": np.asarray([sample.convergence_flag for sample in samples], dtype=np.int64),
        }

        indices = np.arange(num_samples)
        rng = self._rng()
        rng.shuffle(indices)
        train_end = int(num_samples * self.config.train_ratio)
        val_end = train_end + int(num_samples * self.config.val_ratio)
        payload["train_indices"] = indices[:train_end]
        payload["val_indices"] = indices[train_end:val_end]
        payload["test_indices"] = indices[val_end:]

        np.savez(output_path, **payload)
        save_json(
            output_path.with_suffix(".json"),
            {
                "dataset_path": str(output_path),
                "num_samples": num_samples,
                "num_query_points": num_query_points,
                "num_surface_points": num_surface_points,
            },
        )
        return output_path

