from __future__ import annotations

from pathlib import Path

import torch

from cfd_operator.config.schemas import DataConfig, LossConfig, ModelConfig
from cfd_operator.data import CFDDataModule
from cfd_operator.losses import CompositeLoss, LossScheduleState, build_physics_loss_scheduler, compute_physics_informed_loss
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


def test_composite_loss_runs_with_nut_physics(tmp_path: Path) -> None:
    data_config = DataConfig(
        dataset_type="synthetic",
        dataset_path=str(tmp_path / "toy_dataset_nut.npz"),
        num_samples=16,
        num_geometries=4,
        conditions_per_geometry=4,
        num_query_points=24,
        num_surface_points=20,
        field_names=("u", "v", "p", "nut"),
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
    loss_fn = CompositeLoss(
        config=LossConfig(use_physics=True, physics_weight=0.1, boundary_weight=0.2),
        normalizers=data_module.normalizers,
        field_names=("u", "v", "p", "nut"),
    )
    total_loss, metrics = loss_fn(model=model, batch=batch, outputs=outputs)
    assert torch.isfinite(total_loss)
    assert metrics["loss_physics"] >= 0.0


def test_composite_loss_runs_with_hard_region_weighting_and_feature_balancing(tmp_path: Path) -> None:
    data_config = DataConfig(
        dataset_type="synthetic",
        dataset_path=str(tmp_path / "toy_dataset_weighted.npz"),
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
        feature_output_dim=2,
        hidden_dim=32,
        latent_dim=32,
        fourier_features_dim=8,
    )
    model = create_model(model_config)
    outputs = model.loss_outputs(batch["branch_inputs"], batch["query_points"])
    loss_fn = CompositeLoss(
        config=LossConfig(
            use_physics=False,
            boundary_weight=0.0,
            use_feature_loss=True,
            feature_weight=0.05,
            use_hard_region_weighting=True,
            high_gradient_region_weight=1.0,
            near_wall_region_weight=0.5,
            wake_region_weight=0.5,
            surface_leading_edge_weight=0.5,
            use_feature_class_balancing=True,
            feature_focal_gamma=1.5,
        ),
        normalizers=data_module.normalizers,
    )
    total_loss, metrics = loss_fn(model=model, batch=batch, outputs=outputs)
    assert torch.isfinite(total_loss)
    assert metrics["query_hard_weight_mean"] >= 1.0
    assert metrics["surface_hard_weight_mean"] >= 1.0


def test_unified_physics_loss_bundle_exposes_expected_terms(tmp_path: Path) -> None:
    data_config = DataConfig(
        dataset_type="synthetic",
        dataset_path=str(tmp_path / "toy_dataset_physics_bundle.npz"),
        num_samples=12,
        num_geometries=3,
        conditions_per_geometry=4,
        num_query_points=16,
        num_surface_points=12,
        field_names=("u", "v", "p", "nut"),
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
    physics_bundle = compute_physics_informed_loss(
        model=model,
        batch=batch,
        config=LossConfig(use_physics=True, lambda_bc=1.0, lambda_consistency=1.0),
        normalizers=data_module.normalizers,
        field_names=("u", "v", "p", "nut"),
        pressure_target_mode="raw",
    )
    for key in [
        "total",
        "data",
        "continuity",
        "momentum_x",
        "momentum_y",
        "nut_transport_proxy",
        "pde_total",
        "boundary",
        "consistency",
        "diagnostics",
    ]:
        assert key in physics_bundle
    assert torch.isfinite(physics_bundle["total"])


def test_physics_loss_scheduler_warmup_and_ramp() -> None:
    scheduler = build_physics_loss_scheduler(warmup_epochs=2, ramp_epochs=3, max_weight=1.0)
    assert scheduler.multiplier(LossScheduleState(epoch=0)) == 0.0
    assert scheduler.multiplier(LossScheduleState(epoch=1)) == 0.0
    assert 0.0 < scheduler.multiplier(LossScheduleState(epoch=2)) <= 1.0
    assert scheduler.multiplier(LossScheduleState(epoch=10)) == 1.0
