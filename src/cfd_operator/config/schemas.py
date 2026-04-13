"""Typed configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Tuple

from pydantic import BaseModel, Field, model_validator


class ExperimentConfig(BaseModel):
    name: str = "default_run"
    seed: int = 42
    output_root: str = "outputs"
    device: str = "cpu"

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.name


class NormalizationConfig(BaseModel):
    branch: Literal["none", "standard"] = "standard"
    coordinates: Literal["none", "standard"] = "standard"
    fields: Literal["none", "standard"] = "standard"
    scalars: Literal["none", "standard"] = "standard"


class DataConfig(BaseModel):
    name: str = "toy_airfoil"
    dataset_type: Literal["synthetic", "file"] = "synthetic"
    dataset_path: str = "outputs/data/toy_airfoil_dataset.npz"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_samples: int = 160
    num_query_points: int = 192
    num_surface_points: int = 96
    mach_range: Tuple[float, float] = (0.2, 0.78)
    aoa_range: Tuple[float, float] = (-2.0, 8.0)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    include_reynolds: bool = False
    branch_feature_mode: Literal["params", "points"] = "params"
    low_fidelity_enabled: bool = False

    @model_validator(mode="after")
    def validate_ratios(self) -> "DataConfig":
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1.0e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        return self


class ModelConfig(BaseModel):
    name: str = "deeponet"
    branch_input_dim: int = 6
    trunk_input_dim: int = 2
    field_output_dim: int = 4
    scalar_output_dim: int = 2
    hidden_dim: int = 128
    latent_dim: int = 128
    branch_layers: int = 3
    trunk_layers: int = 3
    activation: Literal["relu", "gelu", "tanh", "silu"] = "gelu"
    dropout: float = 0.0
    use_fourier_features: bool = True
    fourier_features_dim: int = 32
    scalar_head_hidden_dim: int = 64


class SchedulerConfig(BaseModel):
    name: Literal["none", "cosine", "step"] = "cosine"
    t_max: int = 20
    min_lr: float = 1.0e-5
    step_size: int = 10
    gamma: float = 0.5


class TrainConfig(BaseModel):
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-6
    optimizer: Literal["adam", "adamw"] = "adam"
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    mixed_precision: bool = False
    grad_clip_norm: Optional[float] = 1.0
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 1
    early_stopping_patience: int = 10
    resume_from: Optional[str] = None


class LossConfig(BaseModel):
    field_weight: float = 1.0
    surface_weight: float = 0.5
    scalar_weight: float = 0.5
    physics_weight: float = 0.1
    boundary_weight: float = 0.0
    field_loss_type: Literal["mse", "mae"] = "mse"
    scalar_loss_type: Literal["mse", "mae"] = "mse"
    use_physics: bool = True
    use_energy_residual: bool = False


class EvalConfig(BaseModel):
    batch_size: int = 8
    num_visualization_samples: int = 3
    save_plots: bool = True
    metrics_path: str = "outputs/eval/metrics.json"
    report_path: str = "outputs/eval/report.md"


class ServeConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    checkpoint_path: str = "outputs/default_run/checkpoints/best.pt"


class ProjectConfig(BaseModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump()
