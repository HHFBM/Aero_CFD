from __future__ import annotations

from pathlib import Path

from cfd_operator.config.schemas import DataConfig, EvalConfig, LossConfig, ModelConfig
from cfd_operator.data import CFDDataModule
from cfd_operator.evaluators import Evaluator
from cfd_operator.losses import CompositeLoss
from cfd_operator.models import create_model


def test_scalar_outputs_shape_and_placeholder_losses(tmp_path: Path) -> None:
    data_config = DataConfig(
        dataset_type="synthetic",
        dataset_path=str(tmp_path / "toy_dataset.npz"),
        num_geometries=5,
        conditions_per_geometry=4,
        num_query_points=24,
        num_surface_points=18,
    )
    data_module = CFDDataModule(config=data_config, batch_size=2)
    data_module.setup()
    batch = next(iter(data_module.train_dataloader()))

    model = create_model(
        ModelConfig(
            branch_input_dim=batch["branch_inputs"].shape[-1],
            trunk_input_dim=2,
            field_output_dim=4,
            scalar_output_dim=2,
            hidden_dim=32,
            latent_dim=32,
            fourier_features_dim=8,
        )
    )
    outputs = model(batch["branch_inputs"], batch["query_points"])
    loss_fn = CompositeLoss(
        config=LossConfig(use_physics=False, boundary_weight=0.0),
        normalizers=data_module.normalizers,
    )
    _, metrics = loss_fn(model=model, batch=batch, outputs=outputs)

    assert batch["scalar_targets"].shape[-1] == 2
    assert outputs["scalars"].shape[-1] == 2
    assert metrics["loss_heat_flux"] == 0.0
    assert metrics["loss_wall_shear"] == 0.0
    assert metrics["loss_shock_location"] == 0.0


def test_placeholder_outputs_not_reported_as_benchmarks(tmp_path: Path) -> None:
    data_config = DataConfig(
        dataset_type="synthetic",
        dataset_path=str(tmp_path / "toy_dataset.npz"),
        num_geometries=5,
        conditions_per_geometry=4,
        num_query_points=24,
        num_surface_points=18,
    )
    data_module = CFDDataModule(config=data_config, batch_size=2)
    data_module.setup()
    assert data_module.payload is not None

    model = create_model(
        ModelConfig(
            branch_input_dim=int(data_module.payload["branch_inputs"].shape[-1]),
            trunk_input_dim=2,
            field_output_dim=4,
            scalar_output_dim=2,
            hidden_dim=32,
            latent_dim=32,
            fourier_features_dim=8,
        )
    )
    evaluator = Evaluator(
        config=EvalConfig(batch_size=2, save_plots=False, export_analysis=False),
        model=model,
        data_module=data_module,
        normalizers=data_module.normalizers,
        device="cpu",
    )
    metrics = evaluator.evaluate(tmp_path / "eval_scalar_only")

    assert "cl_mae" in metrics
    assert "cd_mae" in metrics
    assert "pressure_surface_rmse" in metrics
    assert "cp_surface_rmse" in metrics
    assert "slice_rmse" in metrics
    assert "loss_heat_flux" not in metrics
    assert "heat_flux_surface_rmse" not in metrics
    assert "wall_shear_surface_rmse" not in metrics
