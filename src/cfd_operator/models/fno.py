"""Fourier operator variants.

The standard grid FNO baseline remains intentionally unavailable because the
current AirfRANS data path uses irregular point clouds rather than a shared
Cartesian lattice. GeoFNOModel below provides a geometry-aware spectral
operator that works directly on irregular query points while preserving the
project's field/scalar/features interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from cfd_operator.config.schemas import ModelConfig
from cfd_operator.models.base import BaseOperatorModel
from cfd_operator.models.deeponet import MLP, _activation
from cfd_operator.models.heads import FeatureDecoderHead, FieldDecoderHead, ScalarDecoderHead, SurfaceDecoderHead


def _build_frequency_lattice(num_modes: int, coord_dim: int) -> torch.Tensor:
    if coord_dim != 2:
        raise ValueError("GeoFNOModel currently supports 2D coordinates only.")
    if num_modes <= 0:
        raise ValueError("num_modes must be positive.")

    candidates: list[tuple[int, int, int]] = []
    radius = 1
    while len(candidates) < num_modes * 2:
        for kx in range(-radius, radius + 1):
            for ky in range(-radius, radius + 1):
                if kx == 0 and ky == 0:
                    continue
                norm = abs(kx) + abs(ky)
                candidates.append((norm, kx, ky))
        radius += 1

    candidates.sort(key=lambda item: (item[0], abs(item[1]), abs(item[2]), item[1], item[2]))
    unique: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for _, kx, ky in candidates:
        pair = (kx, ky)
        if pair in seen:
            continue
        seen.add(pair)
        unique.append(pair)
        if len(unique) >= num_modes:
            break
    return torch.tensor(unique, dtype=torch.float32)


class SpectralPointMixing(nn.Module):
    """Low-mode spectral mixing on irregular 2D point sets."""

    def __init__(self, channels: int, num_modes: int, coord_dim: int) -> None:
        super().__init__()
        self.channels = channels
        self.num_modes = num_modes
        self.coord_dim = coord_dim
        frequencies = _build_frequency_lattice(num_modes=num_modes, coord_dim=coord_dim)
        self.register_buffer("frequencies", frequencies, persistent=False)
        scale = 1.0 / max(channels, 1)
        weight = scale * torch.randn(num_modes, channels, dtype=torch.cfloat)
        self.weight = nn.Parameter(weight)

    def forward(self, inputs: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        # inputs: [B, N, C], coords: [B, N, 2]
        phase = 2.0 * torch.pi * torch.einsum("bnc,mc->bnm", coords, self.frequencies)
        basis = torch.exp(1j * phase)
        coeffs = torch.einsum("bnm,bnc->bmc", basis.conj(), inputs.to(torch.cfloat))
        coeffs = coeffs / max(inputs.shape[1], 1)
        mixed = coeffs * self.weight.unsqueeze(0)
        reconstructed = torch.einsum("bnm,bmd->bnd", basis, mixed)
        return reconstructed.real


class GeoFNOBlock(nn.Module):
    def __init__(self, channels: int, num_modes: int, coord_dim: int, activation: str, dropout: float) -> None:
        super().__init__()
        self.spectral = SpectralPointMixing(channels=channels, num_modes=num_modes, coord_dim=coord_dim)
        self.local = nn.Sequential(
            nn.Linear(channels, channels),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(channels, channels),
        )
        self.norm = nn.LayerNorm(channels)
        self.activation = _activation(activation)

    def forward(self, inputs: torch.Tensor, coords: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        conditioned = inputs + conditioning.unsqueeze(1)
        update = self.spectral(conditioned, coords) + self.local(conditioned)
        return self.norm(inputs + self.activation(update))


@dataclass
class GeoFNOOutput:
    fields: torch.Tensor
    scalars: torch.Tensor
    features: torch.Tensor | None
    point_latent: torch.Tensor
    branch_latent: torch.Tensor


class FNOModel(BaseOperatorModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, branch_inputs, query_points):  # type: ignore[override]
        raise NotImplementedError(
            "FNOModel requires a shared regular grid. "
            "The current AirfRANS point-cloud pipeline should use model.name=geofno instead."
        )


class GeoFNOModel(BaseOperatorModel):
    """Geometry-aware Fourier operator for irregular 2D query points."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.coord_dim = config.trunk_input_dim
        self.channels = config.latent_dim
        self.num_modes = max(4, config.fourier_features_dim // 4)

        if self.coord_dim != 2:
            raise ValueError("GeoFNOModel currently expects 2D query points.")

        self.branch_net = MLP(
            input_dim=config.branch_input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=self.channels,
            depth=config.branch_layers,
            activation=config.activation,
            dropout=config.dropout,
        )
        lift_input_dim = config.trunk_input_dim + 2 * self.num_modes
        self.point_lift = MLP(
            input_dim=lift_input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=self.channels,
            depth=2,
            activation=config.activation,
            dropout=config.dropout,
        )
        self.blocks = nn.ModuleList(
            [
                GeoFNOBlock(
                    channels=self.channels,
                    num_modes=self.num_modes,
                    coord_dim=self.coord_dim,
                    activation=config.activation,
                    dropout=config.dropout,
                )
                for _ in range(max(config.trunk_layers, 1))
            ]
        )
        self.condition_proj = nn.ModuleList(
            [nn.Linear(self.channels, self.channels) for _ in range(max(config.trunk_layers, 1))]
        )
        self.field_head = FieldDecoderHead(
            nn.Sequential(
                nn.Linear(self.channels, config.hidden_dim),
                _activation(config.activation),
                nn.Linear(config.hidden_dim, config.field_output_dim),
            ),
            output_dim=config.field_output_dim,
        )
        self.surface_head = SurfaceDecoderHead(nn.Identity(), output_dim=config.field_output_dim)
        self.scalar_head = ScalarDecoderHead(
            nn.Sequential(
                nn.Linear(self.channels * 2, config.scalar_head_hidden_dim),
                _activation(config.activation),
                nn.Linear(config.scalar_head_hidden_dim, config.scalar_output_dim),
            ),
            output_dim=config.scalar_output_dim,
        )
        if config.feature_output_dim > 0:
            self.feature_head: Optional[FeatureDecoderHead] = FeatureDecoderHead(
                nn.Sequential(
                    nn.Linear(self.channels, config.hidden_dim),
                    _activation(config.activation),
                    nn.Linear(config.hidden_dim, config.feature_output_dim),
                ),
                output_dim=config.feature_output_dim,
            )
        else:
            self.feature_head = None

        frequencies = _build_frequency_lattice(num_modes=self.num_modes, coord_dim=self.coord_dim)
        self.register_buffer("lift_frequencies", frequencies, persistent=False)

    def _coord_embedding(self, query_points: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * torch.pi * torch.einsum("bnc,mc->bnm", query_points, self.lift_frequencies)
        return torch.cat([query_points, torch.sin(phase), torch.cos(phase)], dim=-1)

    def forward(self, branch_inputs: torch.Tensor, query_points: torch.Tensor) -> dict[str, torch.Tensor]:
        branch_latent = self.branch_net(branch_inputs)
        point_latent = self.point_lift(self._coord_embedding(query_points))

        hidden = point_latent + branch_latent.unsqueeze(1)
        for block, condition_proj in zip(self.blocks, self.condition_proj):
            hidden = block(hidden, query_points, condition_proj(branch_latent))

        fields = self.field_head(hidden)
        pooled = hidden.mean(dim=1)
        scalars = self.scalar_head(torch.cat([branch_latent, pooled], dim=-1))
        outputs: dict[str, torch.Tensor] = {
            "fields": fields,
            "scalars": scalars,
            "branch_latent": branch_latent,
            "point_latent": hidden,
        }
        if self.feature_head is not None:
            outputs["features"] = self.feature_head(hidden)
        return outputs

    def decoder_head_metadata(self) -> dict[str, dict[str, object]]:
        metadata = {
            "field": self.field_head.metadata(),
            "surface": self.surface_head.metadata(),
            "scalar": self.scalar_head.metadata(),
        }
        if self.feature_head is not None:
            metadata["feature"] = self.feature_head.metadata()
        return metadata
