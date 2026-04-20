"""DeepONet-style surrogate model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

from cfd_operator.config.schemas import ModelConfig
from cfd_operator.models.base import BaseOperatorModel
from cfd_operator.models.geometry_backbone import build_geometry_backbone_contract
from cfd_operator.models.heads import FeatureDecoderHead, FieldDecoderHead, ScalarDecoderHead, SurfaceDecoderHead


def _activation(name: str) -> nn.Module:
    mapping = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(),
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int, activation: str, dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = input_dim
        act = _activation(activation)
        for _ in range(max(depth - 1, 1)):
            layers.extend([nn.Linear(current_dim, hidden_dim), act, nn.Dropout(dropout)])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class FourierFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, num_features: int) -> None:
        super().__init__()
        if num_features % 2 != 0:
            raise ValueError("fourier_features_dim must be even")
        self.register_buffer("projection", torch.randn(input_dim, num_features // 2))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        projected = 2.0 * torch.pi * coords @ self.projection
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


@dataclass
class DeepONetOutput:
    fields: torch.Tensor
    scalars: torch.Tensor
    features: torch.Tensor | None
    branch_latent: torch.Tensor
    trunk_latent: torch.Tensor


class DeepONetModel(BaseOperatorModel):
    """Branch/trunk operator surrogate with scalar head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        trunk_input_dim = config.trunk_input_dim
        self.fourier_encoder: Optional[FourierFeatureEncoder] = None
        if config.use_fourier_features:
            self.fourier_encoder = FourierFeatureEncoder(config.trunk_input_dim, config.fourier_features_dim)
            trunk_input_dim = config.trunk_input_dim + config.fourier_features_dim

        self.branch_net = MLP(
            input_dim=config.branch_input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.latent_dim,
            depth=config.branch_layers,
            activation=config.activation,
            dropout=config.dropout,
        )
        self.trunk_net = MLP(
            input_dim=trunk_input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.latent_dim,
            depth=config.trunk_layers,
            activation=config.activation,
            dropout=config.dropout,
        )
        self.field_head = FieldDecoderHead(
            nn.Linear(config.latent_dim, config.field_output_dim),
            output_dim=config.field_output_dim,
        )
        self.surface_head = SurfaceDecoderHead(nn.Identity(), output_dim=config.field_output_dim)
        self.scalar_head = ScalarDecoderHead(
            nn.Sequential(
                nn.Linear(config.latent_dim, config.scalar_head_hidden_dim),
                _activation(config.activation),
                nn.Linear(config.scalar_head_hidden_dim, config.scalar_output_dim),
            ),
            output_dim=config.scalar_output_dim,
        )
        self.feature_head: Optional[FeatureDecoderHead]
        if config.feature_output_dim > 0:
            self.feature_head = FeatureDecoderHead(
                nn.Linear(config.latent_dim, config.feature_output_dim),
                output_dim=config.feature_output_dim,
            )
        else:
            self.feature_head = None

    def _encode_trunk(self, query_points: torch.Tensor) -> torch.Tensor:
        if self.fourier_encoder is not None:
            encoded = self.fourier_encoder(query_points)
            query_points = torch.cat([query_points, encoded], dim=-1)
        return self.trunk_net(query_points)

    def forward(self, branch_inputs: torch.Tensor, query_points: torch.Tensor) -> dict[str, torch.Tensor]:
        branch_latent = self.branch_net(branch_inputs)
        trunk_latent = self._encode_trunk(query_points)
        combined = branch_latent.unsqueeze(1) * trunk_latent
        fields = self.field_head(combined)
        scalars = self.scalar_head(branch_latent)
        outputs: dict[str, torch.Tensor] = {
            "fields": fields,
            "scalars": scalars,
            "branch_latent": branch_latent,
            "trunk_latent": trunk_latent,
        }
        if self.feature_head is not None:
            outputs["features"] = self.feature_head(combined)
        return outputs

    def decoder_head_metadata(self) -> dict[str, dict[str, object]]:
        metadata = {
            "field": self.field_head.metadata(),
            "surface": self.surface_head.metadata(),
            "scalar": self.scalar_head.metadata(),
        }
        metadata["scalar"]["aggregation"] = (
            "branch_only"
            if self.config.scalar_pooling_mode == "default"
            else self.config.scalar_pooling_mode
        )
        if self.feature_head is not None:
            metadata["feature"] = self.feature_head.metadata()
        return metadata

    def geometry_backbone_metadata(self) -> dict[str, object] | None:
        return build_geometry_backbone_contract(self.config).as_dict()
