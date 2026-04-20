from __future__ import annotations

from pathlib import Path

import numpy as np
import json
import pandas as pd

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
from cfd_operator.geometry import NACA4Airfoil
from cfd_operator.inference import Predictor
from cfd_operator.losses import CompositeLoss
from cfd_operator.models import create_model
from cfd_operator.trainers import Trainer


def test_analysis_infer_bundle_exports(tmp_path: Path) -> None:
    config = ProjectConfig(
        experiment=ExperimentConfig(name="analysis_smoke", output_root=str(tmp_path), device="cpu"),
        data=DataConfig(
            dataset_type="synthetic",
            dataset_path=str(tmp_path / "toy_dataset.npz"),
            num_geometries=5,
            conditions_per_geometry=4,
            num_query_points=24,
            num_surface_points=20,
        ),
        model=ModelConfig(hidden_dim=32, latent_dim=32, fourier_features_dim=8, feature_output_dim=2),
        train=TrainConfig(epochs=1, batch_size=4, early_stopping_patience=2),
        loss=LossConfig(use_physics=False, boundary_weight=0.1, use_feature_loss=True, feature_weight=0.1),
        eval=EvalConfig(save_plots=False),
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
    trainer.fit()

    checkpoint_path = tmp_path / "analysis_smoke" / "checkpoints" / "best.pt"
    predictor = Predictor.from_checkpoint(checkpoint_path, device="cpu")
    export_dir = tmp_path / "analysis_bundle"
    result = predictor.predict_from_geometry(
        geometry_params=np.asarray([0.02, 0.4, 0.12, 1.0], dtype=np.float32),
        mach=0.45,
        aoa_deg=3.0,
        query_points=np.asarray([[0.0, 0.0], [0.4, 0.1], [0.8, -0.05], [1.2, 0.0]], dtype=np.float32),
        slice_definitions=[{"type": "y_const", "value": 0.0, "x_min": -0.2, "x_max": 1.2, "num_points": 16}],
        export_dir=export_dir,
    )

    assert "surface_predictions" in result
    assert "slice_predictions" in result
    assert "feature_predictions" in result
    assert "task_semantics" in result["metadata"]
    assert "pressure_semantics" in result["metadata"]
    assert "geometry_backbone_contract" in result["metadata"]
    assert (export_dir / "predictions.json").exists()
    assert (export_dir / "scalar_summary.png").exists()
    assert (export_dir / "surface_cp.png").exists()
    assert (export_dir / "surface_pressure.png").exists()
    assert (export_dir / "surface_values.csv").exists()
    assert (export_dir / "slice_values.csv").exists()
    assert (export_dir / "feature_summary.json").exists()
    assert (export_dir / "task_semantics.json").exists()
    assert (export_dir / "dataset_capability.json").exists()
    assert (export_dir / "high_gradient_regions.png").exists()

    exported_predictions = json.loads((export_dir / "predictions.json").read_text(encoding="utf-8"))
    assert "pressure_semantics" in exported_predictions["metadata"]
    assert exported_predictions["metadata"]["geometry_backbone_contract"]["mode"] == "fixed_branch_vector"
    assert exported_predictions["metadata"]["task_semantics"]["surface_cp"]["category"] == "derived"


def test_analysis_infer_bundle_respects_disabled_outputs(tmp_path: Path) -> None:
    config = ProjectConfig(
        experiment=ExperimentConfig(name="analysis_smoke_disabled", output_root=str(tmp_path), device="cpu"),
        data=DataConfig(
            dataset_type="synthetic",
            dataset_path=str(tmp_path / "toy_dataset_disabled.npz"),
            num_geometries=4,
            conditions_per_geometry=4,
            num_query_points=20,
            num_surface_points=18,
        ),
        model=ModelConfig(hidden_dim=32, latent_dim=32, fourier_features_dim=8),
        train=TrainConfig(epochs=1, batch_size=4, early_stopping_patience=2),
        loss=LossConfig(use_physics=False, boundary_weight=0.1),
        eval=EvalConfig(save_plots=False),
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
    trainer.fit()

    checkpoint_path = tmp_path / "analysis_smoke_disabled" / "checkpoints" / "best.pt"
    predictor = Predictor.from_checkpoint(checkpoint_path, device="cpu")
    export_dir = tmp_path / "analysis_bundle_disabled"
    result = predictor.predict_from_geometry(
        geometry_params=np.asarray([0.02, 0.4, 0.12, 1.0], dtype=np.float32),
        mach=0.45,
        aoa_deg=3.0,
        query_points=np.asarray([[0.0, 0.0], [0.4, 0.1], [0.8, -0.05], [1.2, 0.0]], dtype=np.float32),
        include_surface=False,
        include_slices=False,
        include_features=False,
        export_dir=export_dir,
    )

    assert "surface_predictions" not in result
    assert "slice_predictions" not in result
    assert "feature_predictions" not in result
    assert (export_dir / "predictions.json").exists()
    assert not (export_dir / "surface_values.csv").exists()
    assert not (export_dir / "slice_values.csv").exists()
    assert not (export_dir / "feature_summary.json").exists()


def test_generic_surface_points_input_runs_inference_and_exports_bundle(tmp_path: Path) -> None:
    config = ProjectConfig(
        experiment=ExperimentConfig(name="analysis_smoke_generic", output_root=str(tmp_path), device="cpu"),
        data=DataConfig(
            dataset_type="synthetic",
            dataset_path=str(tmp_path / "toy_dataset_generic.npz"),
            num_geometries=4,
            conditions_per_geometry=4,
            num_query_points=24,
            num_surface_points=24,
        ),
        model=ModelConfig(hidden_dim=32, latent_dim=32, fourier_features_dim=8),
        train=TrainConfig(epochs=1, batch_size=4, early_stopping_patience=2),
        loss=LossConfig(use_physics=False, boundary_weight=0.1),
        eval=EvalConfig(save_plots=False),
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
    trainer.fit()

    checkpoint_path = tmp_path / "analysis_smoke_generic" / "checkpoints" / "best.pt"
    predictor = Predictor.from_checkpoint(checkpoint_path, device="cpu")
    airfoil = NACA4Airfoil(max_camber=0.03, camber_position=0.4, thickness=0.12)
    geometry_points = airfoil.surface_points(61)
    export_dir = tmp_path / "analysis_bundle_generic"
    result = predictor.predict_from_geometry(
        geometry_params=None,
        geometry_mode="generic_surface_points",
        geometry_points=geometry_points,
        mach=0.42,
        aoa_deg=2.0,
        query_points=np.asarray([[0.0, 0.0], [0.4, 0.1], [0.8, -0.05], [1.2, 0.0]], dtype=np.float32),
        export_dir=export_dir,
    )

    assert "geometry_semantics" in result["metadata"]
    assert result["metadata"]["geometry_semantics"]["geometry_mode"] == "generic_surface_points"
    assert (export_dir / "predictions.json").exists()


def test_generic_minimum_file_dataset_can_train_and_run_inference(tmp_path: Path) -> None:
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
    csv_path = tmp_path / "generic_infer_minimum.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    config = ProjectConfig(
        experiment=ExperimentConfig(name="analysis_generic_minimum", output_root=str(tmp_path), device="cpu"),
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
        eval=EvalConfig(save_plots=False),
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
    trainer.fit()

    checkpoint_path = tmp_path / "analysis_generic_minimum" / "checkpoints" / "best.pt"
    predictor = Predictor.from_checkpoint(checkpoint_path, device="cpu")
    export_dir = tmp_path / "analysis_bundle_generic_minimum"
    geometry_points = np.asarray(
        [[1.0, 0.0], [0.5, 0.08], [0.0, 0.0], [0.5, -0.08]],
        dtype=np.float32,
    )
    result = predictor.predict_from_geometry(
        geometry_params=None,
        geometry_mode="generic_surface_points",
        geometry_points=geometry_points,
        mach=0.31,
        aoa_deg=2.0,
        query_points=np.asarray([[0.0, 0.0], [0.4, 0.1], [0.8, -0.05], [1.2, 0.0]], dtype=np.float32),
        include_surface=False,
        include_slices=False,
        include_features=False,
        export_dir=export_dir,
    )

    assert result["metadata"]["geometry_semantics"]["geometry_mode"] == "generic_surface_points"
    assert (export_dir / "predictions.json").exists()
    assert not (export_dir / "surface_values.csv").exists()
    assert not (export_dir / "slice_values.csv").exists()
    assert not (export_dir / "feature_summary.json").exists()
