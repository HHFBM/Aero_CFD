"""Typed configuration models.

Descriptions intentionally call out which knobs affect supervised, derived or
placeholder/experimental outputs so the YAML remains readable even without a
separate schema browser.
"""

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
    dataset_type: Literal["synthetic", "file", "airfrans", "airfrans_original"] = "synthetic"
    dataset_path: str = "outputs/data/toy_airfoil_dataset.npz"
    airfrans_root: str = "outputs/data/airfrans_raw"
    airfrans_archive_path: str = "outputs/data/AirfRANS_original.tar"
    airfrans_task: Literal["full", "scarce", "reynolds", "aoa"] = "full"
    airfrans_download: bool = True
    airfrans_unzip: bool = True
    airfrans_max_samples: Optional[int] = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_samples: int = 160
    num_geometries: int = 20
    conditions_per_geometry: int = 8
    num_query_points: int = 192
    num_surface_points: int = 96
    mach_range: Tuple[float, float] = (0.2, 0.78)
    aoa_range: Tuple[float, float] = (-2.0, 8.0)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    include_reynolds: bool = False
    branch_input_mode: Literal["legacy_fixed_features", "encoded_geometry"] = Field(
        default="legacy_fixed_features",
        description=(
            "Controls how fixed-dimension branch_inputs are produced. "
            "'legacy_fixed_features' preserves the current path. "
            "'encoded_geometry' first encodes raw geometry and then adapts it back to a fixed branch-compatible vector."
        ),
    )
    branch_feature_mode: Literal["params", "points"] = Field(
        default="params",
        description=(
            "Geometry encoding mode for parameterized or generic geometry preprocessing helpers. "
            "File-backed generic 2D airfoil datasets can use this to derive branch_inputs from geometry_points. "
            "This does not override dataset-specific encoders such as the raw AirfRANS surface-signature path."
        ),
    )
    encoded_geometry_latent_dim: int = Field(
        default=16,
        description=(
            "Latent size used by the lightweight GeometryEncoder when branch_input_mode='encoded_geometry'. "
            "This mode remains optional and defaults to the legacy fixed-feature path."
        ),
    )
    field_names: Tuple[str, str, str, str] = Field(
        default=("u", "v", "p", "rho"),
        description="Supervised pointwise field names. The third channel is always the pressure-like channel.",
    )
    pressure_target_mode: Literal["raw", "cp_like"] = Field(
        default="raw",
        description=(
            "'raw' means field channel 2 stores raw pressure. "
            "'cp_like' means field channel 2 stores a Cp-like pressure quantity (p - p_ref) / q_ref."
        ),
    )
    low_fidelity_enabled: bool = False
    unseen_geometry_ratio: float = 0.15
    unseen_condition_ratio: float = 0.15
    strict_quality_checks: bool = True

    @model_validator(mode="after")
    def validate_ratios(self) -> "DataConfig":
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1.0e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        if self.num_geometries <= 0:
            raise ValueError("num_geometries must be positive")
        if self.conditions_per_geometry <= 0:
            raise ValueError("conditions_per_geometry must be positive")
        if not (0.0 <= self.unseen_geometry_ratio < 0.5):
            raise ValueError("unseen_geometry_ratio must be in [0, 0.5)")
        if not (0.0 <= self.unseen_condition_ratio < 0.5):
            raise ValueError("unseen_condition_ratio must be in [0, 0.5)")
        if len(self.field_names) != 4:
            raise ValueError("field_names must contain exactly four entries: [u, v, pressure_like, aux].")
        if self.field_names[2] not in {"p", "pressure"}:
            raise ValueError("field_names[2] must describe the pressure-like channel, typically 'p'.")
        if self.encoded_geometry_latent_dim <= 0:
            raise ValueError("encoded_geometry_latent_dim must be positive.")
        return self


class ModelConfig(BaseModel):
    name: str = "deeponet"
    branch_input_dim: int = 6
    trunk_input_dim: int = 2
    field_output_dim: int = 4
    scalar_output_dim: int = 2
    feature_output_dim: int = 0
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
    field_weight: float = Field(default=1.0, description="Weight for supervised pointwise field loss.")
    surface_weight: float = Field(default=0.5, description="Weight for derived/supervised surface Cp loss.")
    surface_pressure_weight: float = Field(default=0.0, description="Weight for supervised surface-pressure loss.")
    heat_flux_weight: float = Field(default=0.0, description="Weight for placeholder heat-flux proxy loss.")
    wall_shear_weight: float = Field(default=0.0, description="Weight for placeholder wall-shear proxy loss.")
    scalar_weight: float = Field(default=0.5, description="Weight for supervised scalar loss.")
    slice_weight: float = Field(default=0.0, description="Weight for derived slice loss sampled from fields.")
    feature_weight: float = Field(default=0.0, description="Weight for derived feature-loss terms.")
    shock_location_weight: float = Field(default=0.0, description="Weight for experimental shock-location loss.")
    physics_weight: float = Field(default=0.1, description="Weight for physics regularization loss.")
    boundary_weight: float = Field(default=0.05, description="Weight for boundary consistency loss.")
    field_loss_type: Literal["mse", "mae"] = "mse"
    scalar_loss_type: Literal["mse", "mae"] = "mse"
    surface_loss_type: Literal["mse", "mae"] = "mse"
    feature_loss_type: Literal["bce", "mse"] = "bce"
    use_surface_pressure_loss: bool = False
    use_heat_flux_loss: bool = False
    use_wall_shear_loss: bool = False
    use_slice_loss: bool = False
    use_feature_loss: bool = False
    use_shock_location_loss: bool = False
    use_physics: bool = True
    use_energy_residual: bool = False

    @model_validator(mode="after")
    def validate_loss_flags(self) -> "LossConfig":
        weighted_flags = [
            ("use_surface_pressure_loss", self.use_surface_pressure_loss, "surface_pressure_weight", self.surface_pressure_weight),
            ("use_heat_flux_loss", self.use_heat_flux_loss, "heat_flux_weight", self.heat_flux_weight),
            ("use_wall_shear_loss", self.use_wall_shear_loss, "wall_shear_weight", self.wall_shear_weight),
            ("use_slice_loss", self.use_slice_loss, "slice_weight", self.slice_weight),
            ("use_feature_loss", self.use_feature_loss, "feature_weight", self.feature_weight),
            ("use_shock_location_loss", self.use_shock_location_loss, "shock_location_weight", self.shock_location_weight),
            ("use_physics", self.use_physics, "physics_weight", self.physics_weight),
        ]
        for flag_name, enabled, weight_name, weight in weighted_flags:
            if enabled and weight <= 0.0:
                raise ValueError(f"{flag_name}=true requires {weight_name} > 0.")
        return self


class EvalConfig(BaseModel):
    batch_size: int = 8
    num_visualization_samples: int = 3
    save_plots: bool = True
    metrics_path: str = "outputs/eval/metrics.json"
    report_path: str = "outputs/eval/report.md"
    split_name: str = "test"
    export_analysis: bool = True


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
