"""Reusable decoder heads for CFD surrogate outputs."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn


class SupportsForward(Protocol):
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor: ...


class DecoderHead(nn.Module):
    head_name = "decoder"

    def __init__(self, module: nn.Module, output_dim: int) -> None:
        super().__init__()
        self.module = module
        self.output_dim = output_dim

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.module(latent)

    def metadata(self) -> dict[str, object]:
        return {"name": self.head_name, "output_dim": self.output_dim}


class FieldDecoderHead(DecoderHead):
    head_name = "field"


class SurfaceDecoderHead(DecoderHead):
    head_name = "surface"


class ScalarDecoderHead(DecoderHead):
    head_name = "scalar"


class FeatureDecoderHead(DecoderHead):
    head_name = "feature"
