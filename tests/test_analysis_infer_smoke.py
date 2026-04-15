from __future__ import annotations

from pathlib import Path

import numpy as np

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
    assert (export_dir / "predictions.json").exists()
    assert (export_dir / "scalar_summary.png").exists()
    assert (export_dir / "surface_cp.png").exists()
    assert (export_dir / "surface_pressure.png").exists()
    assert (export_dir / "surface_values.csv").exists()
    assert (export_dir / "slice_values.csv").exists()
    assert (export_dir / "feature_summary.json").exists()
    assert (export_dir / "high_gradient_regions.png").exists()
