from __future__ import annotations

from pathlib import Path

import torch

from cfd_operator.config.schemas import DataConfig, LossConfig, ModelConfig
from cfd_operator.data import CFDDataModule
from cfd_operator.losses import CompositeLoss
from cfd_operator.models import create_model


def test_composite_loss_runs(tmp_path: Path) -> None:
    data_config = DataConfig(
        dataset_type="synthetic",
        dataset_path=str(tmp_path / "toy_dataset.npz"),
        num_samples=16,
        num_geometries=4,
        conditions_per_geometry=4,
        num_query_points=24,
        num_surface_points=20,
    )
    data_module = CFDDataModule(config=data_config, batch_size=2)
    data_module.setup()
    batch = next(iter(data_module.train_dataloader()))

    model_config = ModelConfig(
        branch_input_dim=batch["branch_inputs"].shape[-1],
        trunk_input_dim=2,
        field_output_dim=4,
        scalar_output_dim=2,
        hidden_dim=32,
        latent_dim=32,
        fourier_features_dim=8,
    )
    model = create_model(model_config)
    outputs = model(batch["branch_inputs"], batch["query_points"])
    loss_fn = CompositeLoss(config=LossConfig(use_physics=True, boundary_weight=0.2), normalizers=data_module.normalizers)
    total_loss, metrics = loss_fn(model=model, batch=batch, outputs=outputs)
    assert torch.isfinite(total_loss)
    assert metrics["loss_total"] >= 0.0
    assert metrics["loss_boundary"] >= 0.0
