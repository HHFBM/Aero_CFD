from __future__ import annotations

import torch

from cfd_operator.config.schemas import ModelConfig
from cfd_operator.models import create_model


def test_deeponet_forward_shapes() -> None:
    config = ModelConfig(
        name="deeponet",
        branch_input_dim=6,
        trunk_input_dim=2,
        field_output_dim=4,
        scalar_output_dim=2,
        hidden_dim=32,
        latent_dim=32,
        fourier_features_dim=8,
    )
    model = create_model(config)
    batch_size = 3
    num_points = 17
    outputs = model(
        branch_inputs=torch.randn(batch_size, 6),
        query_points=torch.randn(batch_size, num_points, 2),
    )
    assert outputs["fields"].shape == (batch_size, num_points, 4)
    assert outputs["scalars"].shape == (batch_size, 2)

