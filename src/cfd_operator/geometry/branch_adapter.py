"""Branch input compatibility helpers.

This module centralizes fixed-dimension ``branch_inputs`` construction so the
project can keep the current branch-net contract while making it explicit that
fixed branch features are only one compatibility representation of geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
import warnings

import numpy as np

from .base import AirfoilParameterization
from .preprocess import CanonicalGeometry2D, canonicalize_closed_contour, estimate_parameter_vector_from_surface_points


BranchInputMode = Literal["legacy_fixed_features", "encoded_geometry"]


def sample_surface_signature(airfoil: AirfoilParameterization, num_points: int = 32) -> np.ndarray:
    points = airfoil.surface_points(max(num_points, 8))
    canonical = canonicalize_closed_contour(points, num_points=num_points)
    return canonical.reshape(-1).astype(np.float32)


def sample_surface_signature_from_points(surface_points: np.ndarray, num_points: int = 32) -> np.ndarray:
    points = canonicalize_closed_contour(surface_points, num_points=num_points)
    return points.reshape(-1).astype(np.float32)


def _flow_array(mach: float, aoa_deg: float, reynolds: Optional[float] = None) -> np.ndarray:
    flow = [mach, aoa_deg]
    if reynolds is not None:
        flow.append(reynolds)
    return np.asarray(flow, dtype=np.float32)


def _resample_feature_vector(values: np.ndarray, target_dim: int) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float32).reshape(-1)
    if target_dim <= 0:
        raise ValueError("target_dim must be positive")
    if flattened.shape[0] == target_dim:
        return flattened.astype(np.float32)
    if flattened.shape[0] == 1:
        return np.full((target_dim,), float(flattened[0]), dtype=np.float32)
    source = np.linspace(0.0, 1.0, num=flattened.shape[0], dtype=np.float32)
    target = np.linspace(0.0, 1.0, num=target_dim, dtype=np.float32)
    return np.interp(target, source, flattened).astype(np.float32)


@dataclass(frozen=True)
class GeometryFeatureBuilder:
    branch_feature_mode: str = "params"
    signature_points: int = 32

    def build_from_airfoil(
        self,
        airfoil: AirfoilParameterization,
        *,
        mach: float,
        aoa_deg: float,
        reynolds: Optional[float] = None,
    ) -> np.ndarray:
        base = airfoil.parameter_vector()
        flow = _flow_array(mach=mach, aoa_deg=aoa_deg, reynolds=reynolds)
        feature = np.concatenate([base, flow], axis=0)
        if self.branch_feature_mode == "params":
            return feature.astype(np.float32)
        if self.branch_feature_mode == "points":
            signature = sample_surface_signature(airfoil, num_points=self.signature_points)
            # Preserve the historical parameterized-airfoil contract: points mode
            # augments the fixed parameter+flow vector with a sampled surface
            # signature instead of replacing it.
            return np.concatenate([feature, signature], axis=0).astype(np.float32)
        raise ValueError(f"Unsupported branch feature mode: {self.branch_feature_mode}")

    def build_from_surface_points(
        self,
        surface_points: np.ndarray,
        *,
        mach: float,
        aoa_deg: float,
        reynolds: Optional[float] = None,
    ) -> np.ndarray:
        flow = _flow_array(mach=mach, aoa_deg=aoa_deg, reynolds=reynolds)
        if self.branch_feature_mode == "params":
            geometry_summary = estimate_parameter_vector_from_surface_points(surface_points)
            return np.concatenate([geometry_summary, flow], axis=0).astype(np.float32)
        if self.branch_feature_mode == "points":
            signature = sample_surface_signature_from_points(surface_points, num_points=self.signature_points)
            return np.concatenate([signature, flow], axis=0).astype(np.float32)
        raise ValueError(f"Unsupported branch feature mode: {self.branch_feature_mode}")

    @staticmethod
    def build_from_geometry_params(
        geometry_params: np.ndarray,
        *,
        mach: float,
        aoa_deg: float,
        reynolds: Optional[float] = None,
    ) -> np.ndarray:
        return np.concatenate(
            [np.asarray(geometry_params, dtype=np.float32).reshape(-1), _flow_array(mach=mach, aoa_deg=aoa_deg, reynolds=reynolds)],
            axis=0,
        ).astype(np.float32)

    @staticmethod
    def build_airfrans_original_legacy_signature(
        normalized_surface_points: np.ndarray,
        *,
        mach: float,
        aoa_deg: float,
    ) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(normalized_surface_points, dtype=np.float32).reshape(-1),
                np.asarray([mach, aoa_deg], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)


@dataclass(frozen=True)
class GeometryEncoder:
    latent_dim: int = 16
    signature_points: int = 32

    def encode_surface_points(self, surface_points: np.ndarray) -> np.ndarray:
        canonical = canonicalize_closed_contour(surface_points, num_points=max(self.signature_points, 16))
        summary = estimate_parameter_vector_from_surface_points(canonical)
        signature = sample_surface_signature_from_points(canonical, num_points=max(self.signature_points, 16))
        base = np.concatenate(
            [
                summary,
                np.asarray(
                    [
                        float(canonical[:, 0].min()),
                        float(canonical[:, 0].max()),
                        float(canonical[:, 1].min()),
                        float(canonical[:, 1].max()),
                        float(np.ptp(canonical[:, 1])),
                        float(np.mean(canonical[:, 1])),
                    ],
                    dtype=np.float32,
                ),
                signature,
            ],
            axis=0,
        )
        return _resample_feature_vector(base, target_dim=self.latent_dim)

    def encode_canonical_geometry(self, geometry: CanonicalGeometry2D) -> np.ndarray:
        return self.encode_surface_points(geometry.canonical_surface_points)


@dataclass(frozen=True)
class BranchInputAdapter:
    branch_input_mode: BranchInputMode = "legacy_fixed_features"
    branch_feature_mode: str = "params"
    signature_points: int = 32
    encoded_geometry_latent_dim: int = 16

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_builder", GeometryFeatureBuilder(self.branch_feature_mode, self.signature_points))
        object.__setattr__(self, "geometry_encoder", GeometryEncoder(self.encoded_geometry_latent_dim, self.signature_points))

    def build_from_airfoil(
        self,
        airfoil: AirfoilParameterization,
        *,
        mach: float,
        aoa_deg: float,
        reynolds: Optional[float] = None,
        target_dim: Optional[int] = None,
    ) -> np.ndarray:
        legacy = self.feature_builder.build_from_airfoil(airfoil, mach=mach, aoa_deg=aoa_deg, reynolds=reynolds)
        if self.branch_input_mode == "legacy_fixed_features":
            return legacy
        surface_points = airfoil.surface_points(max(self.signature_points, 32))
        return self._build_encoded(
            surface_points=surface_points,
            mach=mach,
            aoa_deg=aoa_deg,
            reynolds=reynolds,
            target_dim=target_dim or legacy.shape[0],
        )

    def build_from_surface_points(
        self,
        surface_points: np.ndarray,
        *,
        mach: float,
        aoa_deg: float,
        reynolds: Optional[float] = None,
        target_dim: Optional[int] = None,
    ) -> np.ndarray:
        legacy = self.feature_builder.build_from_surface_points(
            surface_points,
            mach=mach,
            aoa_deg=aoa_deg,
            reynolds=reynolds,
        )
        if self.branch_input_mode == "legacy_fixed_features":
            return legacy
        return self._build_encoded(
            surface_points=surface_points,
            mach=mach,
            aoa_deg=aoa_deg,
            reynolds=reynolds,
            target_dim=target_dim or legacy.shape[0],
        )

    def build_from_geometry_params(
        self,
        geometry_params: np.ndarray,
        *,
        mach: float,
        aoa_deg: float,
        reynolds: Optional[float] = None,
        surface_points: np.ndarray | None = None,
        target_dim: Optional[int] = None,
    ) -> np.ndarray:
        legacy = self.feature_builder.build_from_geometry_params(
            geometry_params,
            mach=mach,
            aoa_deg=aoa_deg,
            reynolds=reynolds,
        )
        if self.branch_input_mode == "legacy_fixed_features":
            return legacy
        if surface_points is None:
            warnings.warn(
                "encoded_geometry mode requested but raw geometry surface points are unavailable; falling back to legacy_fixed_features.",
                stacklevel=2,
            )
            return legacy
        return self._build_encoded(
            surface_points=surface_points,
            mach=mach,
            aoa_deg=aoa_deg,
            reynolds=reynolds,
            target_dim=target_dim or legacy.shape[0],
        )

    def build_airfrans_original(
        self,
        normalized_surface_points: np.ndarray,
        *,
        mach: float,
        aoa_deg: float,
        target_dim: Optional[int] = None,
    ) -> np.ndarray:
        legacy = self.feature_builder.build_airfrans_original_legacy_signature(
            normalized_surface_points,
            mach=mach,
            aoa_deg=aoa_deg,
        )
        if self.branch_input_mode == "legacy_fixed_features":
            return legacy
        return self._build_encoded(
            surface_points=normalized_surface_points,
            mach=mach,
            aoa_deg=aoa_deg,
            reynolds=None,
            target_dim=target_dim or legacy.shape[0],
        )

    def build_for_checkpoint(
        self,
        canonical_geometry: CanonicalGeometry2D,
        *,
        mach: float,
        aoa_deg: float,
        reynolds: Optional[float],
        expected_dim: int,
        include_reynolds: bool,
    ) -> np.ndarray:
        if self.branch_input_mode == "encoded_geometry":
            return self._build_encoded(
                surface_points=canonical_geometry.normalized_surface_points,
                mach=mach,
                aoa_deg=aoa_deg,
                reynolds=reynolds if include_reynolds else None,
                target_dim=expected_dim,
            )

        flow_array = _flow_array(mach=mach, aoa_deg=aoa_deg, reynolds=reynolds if include_reynolds else None)
        if canonical_geometry.airfoil is not None:
            default_branch = self.feature_builder.build_from_airfoil(
                canonical_geometry.airfoil,
                mach=mach,
                aoa_deg=aoa_deg,
                reynolds=reynolds if include_reynolds else None,
            ).astype(np.float32)
            if default_branch.shape[0] == expected_dim:
                return default_branch

        signature_dim = expected_dim - min(flow_array.shape[0], 3 if include_reynolds else 2)
        if signature_dim > 0 and signature_dim % 2 == 0:
            num_surface_points = signature_dim // 2
            surface_signature = sample_surface_signature_from_points(
                canonical_geometry.normalized_surface_points,
                num_points=num_surface_points,
            )
            branch_inputs = np.concatenate([surface_signature, flow_array[:2]], axis=0).astype(np.float32)
            if branch_inputs.shape[0] == expected_dim:
                return branch_inputs

        geometry_params = None if canonical_geometry.geometry_params is None else np.asarray(canonical_geometry.geometry_params, dtype=np.float32).reshape(-1)
        if geometry_params is not None and geometry_params.shape[0] + flow_array.shape[0] == expected_dim:
            return np.concatenate([geometry_params, flow_array], axis=0).astype(np.float32)
        if geometry_params is not None and geometry_params.shape[0] + 2 == expected_dim:
            return np.concatenate([geometry_params, np.asarray([mach, aoa_deg], dtype=np.float32)], axis=0).astype(np.float32)

        raise ValueError(
            "Could not build legacy fixed branch_inputs for this checkpoint from the provided geometry input."
        )

    def _build_encoded(
        self,
        *,
        surface_points: np.ndarray,
        mach: float,
        aoa_deg: float,
        reynolds: Optional[float],
        target_dim: int,
    ) -> np.ndarray:
        flow = _flow_array(mach=mach, aoa_deg=aoa_deg, reynolds=reynolds)
        geometry_dim = max(target_dim - flow.shape[0], 1)
        geometry_latent = self.geometry_encoder.encode_surface_points(surface_points)
        projected_geometry = _resample_feature_vector(geometry_latent, target_dim=geometry_dim)
        return np.concatenate([projected_geometry, flow], axis=0).astype(np.float32)
