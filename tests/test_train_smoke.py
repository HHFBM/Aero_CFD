from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from cfd_operator.config.schemas import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    ProjectConfig,
    ServeConfig,
    TrainConfig,
)
from cfd_operator.data import CFDDataModule
from cfd_operator.losses import CompositeLoss
from cfd_operator.models import create_model
from cfd_operator.trainers import Trainer


def test_train_smoke(tmp_path: Path) -> None:
    config = ProjectConfig(
        experiment=ExperimentConfig(name="smoke", output_root=str(tmp_path), device="cpu"),
        data=DataConfig(
            dataset_type="synthetic",
            dataset_path=str(tmp_path / "toy_dataset.npz"),
            num_samples=20,
            num_geometries=5,
            conditions_per_geometry=4,
            num_query_points=24,
            num_surface_points=20,
        ),
        model=ModelConfig(hidden_dim=32, latent_dim=32, fourier_features_dim=8),
        train=TrainConfig(epochs=2, batch_size=4, early_stopping_patience=3),
        loss=LossConfig(use_physics=True, boundary_weight=0.1),
        eval=EvalConfig(),
        serve=ServeConfig(),
    )
    data_module = CFDDataModule(config=config.data, batch_size=config.train.batch_size)
    data_module.setup()
    assert data_module.payload is not None

    config.model.branch_input_dim = int(data_module.payload["branch_inputs"].shape[-1])
    config.model.trunk_input_dim = int(data_module.payload["query_points"].shape[-1])
    config.model.field_output_dim = int(data_module.payload["field_targets"].shape[-1])
    config.model.scalar_output_dim = int(data_module.payload["scalar_targets"].shape[-1])

    model = create_model(config.model)
    trainer = Trainer(
        config=config,
        model=model,
        data_module=data_module,
        loss_fn=CompositeLoss(config=config.loss, normalizers=data_module.normalizers),
    )
    history = trainer.fit()
    assert "train_loss_total" in history
    assert (tmp_path / "smoke" / "checkpoints" / "best.pt").exists()
    checkpoint = torch.load(tmp_path / "smoke" / "checkpoints" / "best.pt", map_location="cpu")
    assert checkpoint["branch_contract"]["branch_input_mode"] == "legacy_fixed_features"
    assert checkpoint["branch_contract"]["branch_input_dim"] == int(data_module.payload["branch_inputs"].shape[-1])
    assert checkpoint["geometry_backbone_contract"]["mode"] == "fixed_branch_vector"
    assert checkpoint["geometry_backbone_contract"]["backbone_type"] == "none"


def test_train_smoke_schema_dataset(tmp_path: Path) -> None:
    config = ProjectConfig(
        experiment=ExperimentConfig(name="smoke_schema", output_root=str(tmp_path), device="cpu"),
        data=DataConfig(
            dataset_type="synthetic",
            dataset_view_mode="schema",
            dataset_path=str(tmp_path / "toy_dataset_schema.npz"),
            num_samples=20,
            num_geometries=5,
            conditions_per_geometry=4,
            num_query_points=24,
            num_surface_points=20,
        ),
        model=ModelConfig(hidden_dim=32, latent_dim=32, fourier_features_dim=8),
        train=TrainConfig(epochs=1, batch_size=4, early_stopping_patience=2),
        loss=LossConfig(use_physics=True, boundary_weight=0.1),
        eval=EvalConfig(),
        serve=ServeConfig(),
    )
    data_module = CFDDataModule(config=config.data, batch_size=config.train.batch_size)
    data_module.setup()
    assert data_module.payload is not None

    config.model.branch_input_dim = int(data_module.payload["branch_inputs"].shape[-1])
    config.model.trunk_input_dim = int(data_module.payload["query_points"].shape[-1])
    config.model.field_output_dim = int(data_module.payload["field_targets"].shape[-1])
    config.model.scalar_output_dim = int(data_module.payload["scalar_targets"].shape[-1])

    model = create_model(config.model)
    trainer = Trainer(
        config=config,
        model=model,
        data_module=data_module,
        loss_fn=CompositeLoss(config=config.loss, normalizers=data_module.normalizers),
    )
    history = trainer.fit()
    assert "train_loss_total" in history
    assert (tmp_path / "smoke_schema" / "checkpoints" / "best.pt").exists()


def test_train_smoke_geofno(tmp_path: Path) -> None:
    config = ProjectConfig(
        experiment=ExperimentConfig(name="smoke_geofno", output_root=str(tmp_path), device="cpu"),
        data=DataConfig(
            dataset_type="synthetic",
            dataset_path=str(tmp_path / "toy_dataset_geofno.npz"),
            num_samples=20,
            num_geometries=5,
            conditions_per_geometry=4,
            num_query_points=24,
            num_surface_points=20,
        ),
        model=ModelConfig(name="geofno", hidden_dim=32, latent_dim=32, fourier_features_dim=16, feature_output_dim=2),
        train=TrainConfig(epochs=1, batch_size=4, early_stopping_patience=2),
        loss=LossConfig(use_physics=True, boundary_weight=0.1, use_feature_loss=True, feature_weight=0.02),
        eval=EvalConfig(),
        serve=ServeConfig(),
    )
    data_module = CFDDataModule(config=config.data, batch_size=config.train.batch_size)
    data_module.setup()
    assert data_module.payload is not None

    config.model.branch_input_dim = int(data_module.payload["branch_inputs"].shape[-1])
    config.model.trunk_input_dim = int(data_module.payload["query_points"].shape[-1])
    config.model.field_output_dim = int(data_module.payload["field_targets"].shape[-1])
    config.model.scalar_output_dim = int(data_module.payload["scalar_targets"].shape[-1])

    model = create_model(config.model)
    trainer = Trainer(
        config=config,
        model=model,
        data_module=data_module,
        loss_fn=CompositeLoss(config=config.loss, normalizers=data_module.normalizers),
    )
    history = trainer.fit()
    assert "train_loss_total" in history
    assert (tmp_path / "smoke_geofno" / "checkpoints" / "best.pt").exists()


def test_train_smoke_generic_file_geometry(tmp_path: Path) -> None:
    rows = []
    for sample_id in range(6):
        mach = 0.25 + 0.01 * sample_id
        aoa = -1.0 + 0.5 * sample_id
        contour = [
            (1.0, 0.0),
            (0.5, 0.08 + 0.005 * sample_id),
            (0.0, 0.0),
            (0.5, -0.08 - 0.005 * sample_id),
        ]
        for x, y in contour:
            rows.append(
                {
                    "sample_id": sample_id,
                    "geometry_mode": "generic_surface_points",
                    "geometry_x": x,
                    "geometry_y": y,
                    "mach": mach,
                    "aoa": aoa,
                    "x": x,
                    "y": y,
                    "u": 1.0,
                    "v": 0.0,
                    "p": 1.0,
                    "surface_flag": 1,
                    "cp": 0.0,
                    "cl": 0.1 + 0.01 * sample_id,
                    "cd": 0.01 + 0.001 * sample_id,
                    "fidelity_level": 0,
                    "source": "generic_tabular",
                    "convergence_flag": 1,
                }
            )
    csv_path = tmp_path / "generic_train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    config = ProjectConfig(
        experiment=ExperimentConfig(name="smoke_generic_file", output_root=str(tmp_path), device="cpu"),
        data=DataConfig(
            dataset_type="file",
            dataset_path=str(csv_path),
            branch_feature_mode="points",
            num_query_points=4,
            num_surface_points=4,
            train_ratio=0.67,
            val_ratio=0.17,
            test_ratio=0.16,
        ),
        model=ModelConfig(hidden_dim=32, latent_dim=32, fourier_features_dim=8),
        train=TrainConfig(epochs=1, batch_size=2, early_stopping_patience=2),
        loss=LossConfig(use_physics=False, boundary_weight=0.0),
        eval=EvalConfig(),
        serve=ServeConfig(),
    )
    data_module = CFDDataModule(config=config.data, batch_size=config.train.batch_size)
    data_module.setup()
    assert data_module.payload is not None
    assert data_module.payload["branch_encoding_type"][0] == "derived_surface_signature_plus_flow"

    config.model.branch_input_dim = int(data_module.payload["branch_inputs"].shape[-1])
    config.model.trunk_input_dim = int(data_module.payload["query_points"].shape[-1])
    config.model.field_output_dim = int(data_module.payload["field_targets"].shape[-1])
    config.model.scalar_output_dim = int(data_module.payload["scalar_targets"].shape[-1])

    model = create_model(config.model)
    trainer = Trainer(
        config=config,
        model=model,
        data_module=data_module,
        loss_fn=CompositeLoss(config=config.loss, normalizers=data_module.normalizers),
    )
    history = trainer.fit()
    assert "train_loss_total" in history
    assert (tmp_path / "smoke_generic_file" / "checkpoints" / "best.pt").exists()


def test_train_smoke_generic_file_minimum_contract(tmp_path: Path) -> None:
    rows = []
    for sample_id in range(6):
        mach = 0.25 + 0.01 * sample_id
        aoa = -1.0 + 0.5 * sample_id
        contour = [
            (1.0, 0.0),
            (0.5, 0.08 + 0.005 * sample_id),
            (0.0, 0.0),
            (0.5, -0.08 - 0.005 * sample_id),
        ]
        for x, y in contour:
            rows.append(
                {
                    "sample_id": sample_id,
                    "geometry_mode": "generic_surface_points",
                    "geometry_x": x,
                    "geometry_y": y,
                    "mach": mach,
                    "aoa": aoa,
                    "x": x,
                    "y": y,
                    "u": 1.0,
                    "v": 0.0,
                    "p": 1.0,
                    "cl": 0.1 + 0.01 * sample_id,
                    "cd": 0.01 + 0.001 * sample_id,
                }
            )
    csv_path = tmp_path / "generic_train_minimum.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    config = ProjectConfig(
        experiment=ExperimentConfig(name="smoke_generic_minimum", output_root=str(tmp_path), device="cpu"),
        data=DataConfig(
            dataset_type="file",
            dataset_path=str(csv_path),
            branch_feature_mode="points",
            num_query_points=4,
            num_surface_points=4,
            train_ratio=0.67,
            val_ratio=0.17,
            test_ratio=0.16,
        ),
        model=ModelConfig(hidden_dim=32, latent_dim=32, fourier_features_dim=8),
        train=TrainConfig(epochs=1, batch_size=2, early_stopping_patience=2),
        loss=LossConfig(
            use_physics=False,
            boundary_weight=0.0,
            surface_weight=0.0,
            use_slice_loss=False,
            use_feature_loss=False,
        ),
        eval=EvalConfig(),
        serve=ServeConfig(),
    )
    data_module = CFDDataModule(config=config.data, batch_size=config.train.batch_size)
    data_module.setup()
    assert data_module.payload is not None
    assert data_module.dataset_capability is not None
    assert not data_module.dataset_capability.available("surface_cp")
    assert not data_module.dataset_capability.available("surface_pressure")
    assert not data_module.dataset_capability.available("slice_fields")
    assert not data_module.dataset_capability.available("pressure_gradient_indicator")

    config.model.branch_input_dim = int(data_module.payload["branch_inputs"].shape[-1])
    config.model.trunk_input_dim = int(data_module.payload["query_points"].shape[-1])
    config.model.field_output_dim = int(data_module.payload["field_targets"].shape[-1])
    config.model.scalar_output_dim = int(data_module.payload["scalar_targets"].shape[-1])

    model = create_model(config.model)
    trainer = Trainer(
        config=config,
        model=model,
        data_module=data_module,
        loss_fn=CompositeLoss(config=config.loss, normalizers=data_module.normalizers),
    )
    history = trainer.fit()
    assert "train_loss_total" in history
    assert (tmp_path / "smoke_generic_minimum" / "checkpoints" / "best.pt").exists()
