from __future__ import annotations

from pathlib import Path

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
from cfd_operator.evaluators import Evaluator
from cfd_operator.inference import Predictor
from cfd_operator.losses import CompositeLoss
from cfd_operator.models import create_model
from cfd_operator.trainers import Trainer


def test_evaluator_smoke_runs(tmp_path: Path) -> None:
    config = ProjectConfig(
        experiment=ExperimentConfig(name="eval_smoke", output_root=str(tmp_path), device="cpu"),
        data=DataConfig(
            dataset_type="synthetic",
            dataset_path=str(tmp_path / "toy_dataset.npz"),
            num_geometries=5,
            conditions_per_geometry=4,
            num_query_points=24,
            num_surface_points=20,
        ),
        model=ModelConfig(hidden_dim=32, latent_dim=32, fourier_features_dim=8),
        train=TrainConfig(epochs=1, batch_size=4, early_stopping_patience=3),
        loss=LossConfig(use_physics=True, boundary_weight=0.1),
        eval=EvalConfig(batch_size=4, save_plots=False, split_name="test"),
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

    checkpoint_path = tmp_path / "eval_smoke" / "checkpoints" / "best.pt"
    predictor = Predictor.from_checkpoint(checkpoint_path, device="cpu")
    evaluator = Evaluator(
        config=config.eval,
        model=predictor.model,
        data_module=data_module,
        normalizers=predictor.normalizers,
        device="cpu",
    )
    output_dir = tmp_path / "eval_report"
    metrics = evaluator.evaluate(output_dir)
    assert "field_rmse" in metrics
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "report.json").exists()
