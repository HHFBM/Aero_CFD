from __future__ import annotations

from pathlib import Path

from cfd_operator.config.schemas import DataConfig, LossConfig, ModelConfig
from cfd_operator.data import CFDDataModule
from cfd_operator.models import create_model
from cfd_operator.postprocess import compute_gradient_indicators, estimate_shock_location


def test_feature_outputs_and_summary(tmp_path: Path) -> None:
    data_config = DataConfig(
        dataset_type="synthetic",
        dataset_path=str(tmp_path / "toy_dataset.npz"),
        num_geometries=5,
        conditions_per_geometry=4,
        num_query_points=24,
        num_surface_points=18,
        field_names=("u", "v", "p", "nut"),
    )
    data_module = CFDDataModule(config=data_config, batch_size=2)
    data_module.setup()
    batch = next(iter(data_module.train_dataloader()))

    indicators = compute_gradient_indicators(
        batch["query_points_raw"][0].numpy(),
        batch["field_targets_raw"][0].numpy(),
    )
    summary = estimate_shock_location(
        batch["query_points_raw"][0].numpy(),
        indicators["shock_indicator"],
        indicators["gradient_magnitude"],
    )

    model = create_model(
        ModelConfig(
            branch_input_dim=batch["branch_inputs"].shape[-1],
            trunk_input_dim=2,
            field_output_dim=4,
            scalar_output_dim=2,
            feature_output_dim=2,
            hidden_dim=32,
            latent_dim=32,
            fourier_features_dim=8,
        )
    )
    outputs = model(batch["branch_inputs"], batch["query_points"])

    assert outputs["features"].shape[-1] == 2
    assert "count" in summary
