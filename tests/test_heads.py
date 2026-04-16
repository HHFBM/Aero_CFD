from __future__ import annotations

import torch
from torch import nn

from cfd_operator.models.heads import FeatureDecoderHead, FieldDecoderHead, ScalarDecoderHead


def test_decoder_heads_route_outputs() -> None:
    field_head = FieldDecoderHead(nn.Linear(8, 4), output_dim=4)
    scalar_head = ScalarDecoderHead(nn.Linear(8, 2), output_dim=2)
    feature_head = FeatureDecoderHead(nn.Linear(8, 3), output_dim=3)

    latent = torch.randn(5, 7, 8)
    pooled = torch.randn(5, 8)

    assert field_head(latent).shape == (5, 7, 4)
    assert scalar_head(pooled).shape == (5, 2)
    assert feature_head(latent).shape == (5, 7, 3)
    assert field_head.metadata()["name"] == "field"
