from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

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
from cfd_operator.data.splitting import build_frozen_benchmark_splits
from cfd_operator.evaluators import Evaluator
from cfd_operator.inference import Predictor
from cfd_operator.losses import CompositeLoss
from cfd_operator.models import create_model
from cfd_operator.tasks import CapabilityStatus, DatasetCapability
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


def test_evaluator_smoke_runs_with_schema_dataset(tmp_path: Path) -> None:
    config = ProjectConfig(
        experiment=ExperimentConfig(name="eval_smoke_schema", output_root=str(tmp_path), device="cpu"),
        data=DataConfig(
            dataset_type="synthetic",
            dataset_view_mode="schema",
            dataset_path=str(tmp_path / "toy_dataset_schema.npz"),
            num_geometries=5,
            conditions_per_geometry=4,
            num_query_points=24,
            num_surface_points=20,
        ),
        model=ModelConfig(hidden_dim=32, latent_dim=32, fourier_features_dim=8),
        train=TrainConfig(epochs=1, batch_size=4, early_stopping_patience=2),
        loss=LossConfig(use_physics=False, boundary_weight=0.0),
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

    checkpoint_path = tmp_path / "eval_smoke_schema" / "checkpoints" / "best.pt"
    predictor = Predictor.from_checkpoint(checkpoint_path, device="cpu")
    evaluator = Evaluator(
        config=config.eval,
        model=predictor.model,
        data_module=data_module,
        normalizers=predictor.normalizers,
        device="cpu",
    )
    output_dir = tmp_path / "eval_report_schema"
    metrics = evaluator.evaluate(output_dir)
    assert "field_rmse" in metrics
    assert (output_dir / "metrics.json").exists()


def test_evaluator_gracefully_degrades_with_limited_capability(tmp_path: Path) -> None:
    config = ProjectConfig(
        experiment=ExperimentConfig(name="eval_smoke_degraded", output_root=str(tmp_path), device="cpu"),
        data=DataConfig(
            dataset_type="synthetic",
            dataset_path=str(tmp_path / "toy_dataset_degraded.npz"),
            num_geometries=5,
            conditions_per_geometry=4,
            num_query_points=24,
            num_surface_points=20,
        ),
        model=ModelConfig(hidden_dim=32, latent_dim=32, fourier_features_dim=8),
        train=TrainConfig(epochs=1, batch_size=4, early_stopping_patience=2),
        loss=LossConfig(use_physics=False, boundary_weight=0.0),
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

    data_module.dataset_capability = DatasetCapability(
        dataset_name="degraded_eval",
        target_capabilities={
            "field_targets": CapabilityStatus("supervised", True, True, True, True),
            "scalar_targets": CapabilityStatus("supervised", True, True, True, True),
            "surface_pressure": CapabilityStatus("unavailable", False, False, False, False),
            "surface_cp": CapabilityStatus("unavailable", False, False, False, False),
            "slice_fields": CapabilityStatus("unavailable", False, False, False, False),
            "pressure_gradient_indicator": CapabilityStatus("unavailable", False, False, False, False),
            "high_gradient_mask": CapabilityStatus("unavailable", False, False, False, False),
        },
    )

    checkpoint_path = tmp_path / "eval_smoke_degraded" / "checkpoints" / "best.pt"
    predictor = Predictor.from_checkpoint(checkpoint_path, device="cpu")
    evaluator = Evaluator(
        config=config.eval,
        model=predictor.model,
        data_module=data_module,
        normalizers=predictor.normalizers,
        device="cpu",
    )
    metrics = evaluator.evaluate(tmp_path / "eval_report_degraded")
    assert "field_rmse" in metrics
    assert "cl_mae" in metrics
    assert "cp_surface_rmse" not in metrics
    assert "high_gradient_iou" not in metrics


def test_evaluator_generic_minimum_dataset_skips_optional_metric_groups(tmp_path: Path) -> None:
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
    csv_path = tmp_path / "generic_eval_minimum.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    config = ProjectConfig(
        experiment=ExperimentConfig(name="eval_generic_minimum", output_root=str(tmp_path), device="cpu"),
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
        eval=EvalConfig(batch_size=2, save_plots=False, split_name="test"),
        serve=ServeConfig(),
    )
    data_module = CFDDataModule(config=config.data, batch_size=config.train.batch_size)
    data_module.setup()

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

    checkpoint_path = tmp_path / "eval_generic_minimum" / "checkpoints" / "best.pt"
    predictor = Predictor.from_checkpoint(checkpoint_path, device="cpu")
    evaluator = Evaluator(
        config=config.eval,
        model=predictor.model,
        data_module=data_module,
        normalizers=predictor.normalizers,
        device="cpu",
    )
    output_dir = tmp_path / "eval_report_generic_minimum"
    metrics = evaluator.evaluate(output_dir)
    assert "field_rmse" in metrics
    assert "cl_mae" in metrics
    assert "cp_surface_rmse" not in metrics
    assert "high_gradient_iou" not in metrics
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert "surface_metrics" in report["skipped_metric_groups"]
    assert "feature_metrics" in report["skipped_metric_groups"]


def test_evaluator_reports_local_region_metrics(tmp_path: Path) -> None:
    config = ProjectConfig(
        experiment=ExperimentConfig(name="eval_local_regions", output_root=str(tmp_path), device="cpu"),
        data=DataConfig(
            dataset_type="synthetic",
            dataset_path=str(tmp_path / "toy_dataset_local_regions.npz"),
            num_geometries=5,
            conditions_per_geometry=4,
            num_query_points=24,
            num_surface_points=20,
        ),
        model=ModelConfig(hidden_dim=32, latent_dim=32, fourier_features_dim=8, feature_output_dim=2),
        train=TrainConfig(epochs=1, batch_size=4, early_stopping_patience=2),
        loss=LossConfig(
            use_physics=False,
            boundary_weight=0.0,
            use_feature_loss=True,
            feature_weight=0.02,
            use_hard_region_weighting=True,
            high_gradient_region_weight=1.0,
            near_wall_region_weight=0.5,
            wake_region_weight=0.5,
            surface_leading_edge_weight=0.5,
        ),
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

    checkpoint_path = tmp_path / "eval_local_regions" / "checkpoints" / "best.pt"
    predictor = Predictor.from_checkpoint(checkpoint_path, device="cpu")
    evaluator = Evaluator(
        config=config.eval,
        model=predictor.model,
        data_module=data_module,
        normalizers=predictor.normalizers,
        device="cpu",
    )
    output_dir = tmp_path / "eval_report_local_regions"
    metrics = evaluator.evaluate(output_dir)
    assert "high_gradient_field_rmse" in metrics
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert "local_region_metrics" in report["metric_groups"]
    local_metrics = report["metric_groups"]["local_region_metrics"]
    assert any(
        key in local_metrics
        for key in ("near_wall_field_rmse", "wake_field_rmse", "leading_edge_cp_rmse")
    )


def test_evaluator_runs_on_benchmark_holdout_split(tmp_path: Path) -> None:
    source_config = ProjectConfig(
        experiment=ExperimentConfig(name="eval_benchmark_source", output_root=str(tmp_path), device="cpu"),
        data=DataConfig(
            dataset_type="synthetic",
            dataset_path=str(tmp_path / "eval_benchmark_source.npz"),
            num_geometries=5,
            conditions_per_geometry=4,
            num_query_points=24,
            num_surface_points=20,
        ),
        model=ModelConfig(hidden_dim=32, latent_dim=32, fourier_features_dim=8),
        train=TrainConfig(epochs=1, batch_size=4, early_stopping_patience=2),
        loss=LossConfig(use_physics=False, boundary_weight=0.0),
        eval=EvalConfig(batch_size=4, save_plots=False, split_name="benchmark_holdout"),
        serve=ServeConfig(),
    )
    source_module = CFDDataModule(config=source_config.data, batch_size=source_config.train.batch_size)
    source_module.setup()
    assert source_module.payload is not None
    payload = {key: value for key, value in source_module.payload.items() if not key.endswith("_indices")}
    payload.update(
        build_frozen_benchmark_splits(
            geometry_ids=np.asarray(payload["airfoil_id"]),
            benchmark_holdout_ratio=0.1,
            train_ratio=0.7,
            val_ratio=0.15,
            rng=np.random.default_rng(42),
        )
    )
    benchmark_path = tmp_path / "eval_benchmark_dataset.npz"
    np.savez_compressed(benchmark_path, **payload)

    config = ProjectConfig(
        experiment=ExperimentConfig(name="eval_benchmark", output_root=str(tmp_path), device="cpu"),
        data=DataConfig(dataset_type="file", dataset_path=str(benchmark_path)),
        model=ModelConfig(hidden_dim=32, latent_dim=32, fourier_features_dim=8),
        train=TrainConfig(epochs=1, batch_size=4, early_stopping_patience=2),
        loss=LossConfig(use_physics=False, boundary_weight=0.0),
        eval=EvalConfig(batch_size=4, save_plots=False, split_name="benchmark_holdout"),
        serve=ServeConfig(),
    )
    data_module = CFDDataModule(config=config.data, batch_size=config.train.batch_size)
    data_module.setup()

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

    checkpoint_path = tmp_path / "eval_benchmark" / "checkpoints" / "best.pt"
    predictor = Predictor.from_checkpoint(checkpoint_path, device="cpu")
    evaluator = Evaluator(
        config=config.eval,
        model=predictor.model,
        data_module=data_module,
        normalizers=predictor.normalizers,
        device="cpu",
    )
    output_dir = tmp_path / "eval_report_benchmark_holdout"
    metrics = evaluator.evaluate(output_dir)
    assert "field_rmse" in metrics
    assert (output_dir / "metrics_flat.csv").exists()
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "frozen benchmark split" in report
    selected_cases = json.loads((output_dir / "selected_cases.json").read_text(encoding="utf-8"))
    assert {"best", "median", "worst"}.issubset(selected_cases.keys())


def test_benchmark_script_summarizes_test_and_benchmark_holdout(tmp_path: Path) -> None:
    source_config = ProjectConfig(
        experiment=ExperimentConfig(name="benchmark_script_source", output_root=str(tmp_path), device="cpu"),
        data=DataConfig(
            dataset_type="synthetic",
            dataset_path=str(tmp_path / "benchmark_script_source.npz"),
            num_geometries=5,
            conditions_per_geometry=4,
            num_query_points=24,
            num_surface_points=20,
        ),
        model=ModelConfig(hidden_dim=32, latent_dim=32, fourier_features_dim=8),
        train=TrainConfig(epochs=1, batch_size=4, early_stopping_patience=2),
        loss=LossConfig(use_physics=False, boundary_weight=0.0),
        eval=EvalConfig(batch_size=4, save_plots=False, split_name="test"),
        serve=ServeConfig(),
    )
    source_module = CFDDataModule(config=source_config.data, batch_size=source_config.train.batch_size)
    source_module.setup()
    assert source_module.payload is not None
    payload = {key: value for key, value in source_module.payload.items() if not key.endswith("_indices")}
    payload.update(
        build_frozen_benchmark_splits(
            geometry_ids=np.asarray(payload["airfoil_id"]),
            benchmark_holdout_ratio=0.1,
            train_ratio=0.7,
            val_ratio=0.15,
            rng=np.random.default_rng(42),
        )
    )
    benchmark_path = tmp_path / "benchmark_script_dataset.npz"
    np.savez_compressed(benchmark_path, **payload)

    config = ProjectConfig(
        experiment=ExperimentConfig(name="benchmark_script_run", output_root=str(tmp_path), device="cpu"),
        data=DataConfig(dataset_type="file", dataset_path=str(benchmark_path)),
        model=ModelConfig(hidden_dim=32, latent_dim=32, fourier_features_dim=8),
        train=TrainConfig(epochs=1, batch_size=4, early_stopping_patience=2),
        loss=LossConfig(use_physics=False, boundary_weight=0.0),
        eval=EvalConfig(batch_size=4, save_plots=False, split_name="test"),
        serve=ServeConfig(),
    )
    data_module = CFDDataModule(config=config.data, batch_size=config.train.batch_size)
    data_module.setup()

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

    config_path = tmp_path / "benchmark_script_config.yaml"
    config_path.write_text(yaml.safe_dump(config.as_dict()), encoding="utf-8")
    checkpoint_path = tmp_path / "benchmark_script_run" / "checkpoints" / "best.pt"
    summary_dir = tmp_path / "benchmark_summary"
    subprocess.run(
        [
            sys.executable,
            "scripts/benchmark.py",
            "--run",
            "deeponet_smoke",
            str(config_path),
            str(checkpoint_path),
            "--output-dir",
            str(summary_dir),
        ],
        check=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert (summary_dir / "benchmark_summary.json").exists()
    assert (summary_dir / "benchmark_summary.csv").exists()
    assert (summary_dir / "benchmark_summary.md").exists()
    summary = json.loads((summary_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary[0]["gaps"]
