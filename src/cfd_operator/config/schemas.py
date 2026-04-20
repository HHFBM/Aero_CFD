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
    dataset_view_mode: Literal["payload", "schema"] = Field(
        default="payload",
        description=(
            "Controls how the datamodule builds dataset samples. "
            "'payload' keeps the legacy dense payload-backed path. "
            "'schema' builds training views from unified CFDSurrogateSample objects."
        ),
    )
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
    scalar_pooling_mode: Literal["default", "branch_only", "mean", "mean_max_residual"] = "default"
    geometry_backbone_mode: Literal["fixed_branch_vector", "native_geometry_latent_reserved"] = Field(
        default="fixed_branch_vector",
        description=(
            "Controls the geometry conditioning contract seen by the backbone. "
            "'fixed_branch_vector' keeps the current stable path. "
            "'native_geometry_latent_reserved' only reserves metadata/interface slots for a future native variable-length geometry backbone."
        ),
    )
    geometry_backbone_type: Literal["none", "reserved_interface"] = Field(
        default="none",
        description=(
            "Type tag for the geometry backbone. 'none' means the current fixed branch-vector path. "
            "'reserved_interface' means the checkpoint/config explicitly reserves a future native geometry backbone interface without making it active in the main training path."
        ),
    )
    native_geometry_latent_dim: int = Field(
        default=128,
        description="Reserved latent size for a future native geometry backbone. Not active in the default branch-vector path.",
    )
    native_geometry_token_dim: int = Field(
        default=64,
        description="Reserved token feature size for a future native geometry backbone. Not active in the default branch-vector path.",
    )
    native_geometry_max_tokens: int = Field(
        default=128,
        description="Reserved token budget for a future native geometry backbone. Not active in the default branch-vector path.",
    )

    @model_validator(mode="after")
    def validate_geometry_backbone(self) -> "ModelConfig":
        if self.native_geometry_latent_dim <= 0:
            raise ValueError("native_geometry_latent_dim must be positive.")
        if self.native_geometry_token_dim <= 0:
            raise ValueError("native_geometry_token_dim must be positive.")
        if self.native_geometry_max_tokens <= 0:
            raise ValueError("native_geometry_max_tokens must be positive.")
        if self.geometry_backbone_mode == "fixed_branch_vector" and self.geometry_backbone_type == "reserved_interface":
            # Keep this combination legal for explicit metadata reservation while making the current path remain active.
            return self
        if self.geometry_backbone_mode == "native_geometry_latent_reserved" and self.geometry_backbone_type == "none":
            raise ValueError(
                "geometry_backbone_mode='native_geometry_latent_reserved' requires geometry_backbone_type='reserved_interface'."
            )
        return self


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
    consistency_weight: float = Field(default=0.0, description="Weight for derived consistency loss family.")
    lambda_data: float = Field(default=1.0, description="Inner-family weight for supervised data terms in the unified physics-loss API.")
    lambda_continuity: float = Field(default=1.0, description="Inner-family weight for continuity residual loss.")
    lambda_momentum: float = Field(default=1.0, description="Inner-family weight for momentum residual loss.")
    lambda_nut: float = Field(default=1.0, description="Inner-family weight for nut transport/smoothness proxy residuals.")
    lambda_bc: float = Field(default=1.0, description="Inner-family weight for boundary condition loss terms.")
    lambda_consistency: float = Field(default=1.0, description="Inner-family weight for derived consistency loss terms.")
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
    physics_warmup_epochs: int = Field(
        default=0,
        description="Number of epochs to keep PDE/BC/consistency families off before ramping them in.",
    )
    physics_ramp_epochs: int = Field(
        default=0,
        description="Linear ramp duration for physics-family weights after warmup.",
    )
    physics_schedule_max_weight: float = Field(
        default=1.0,
        description="Maximum multiplier applied by the physics-family scheduler after warmup/ramp.",
    )
    use_hard_region_weighting: bool = Field(
        default=False,
        description="Enable additional weighting for high-gradient, near-wall, wake, and leading-edge regions.",
    )
    high_gradient_region_weight: float = Field(
        default=0.0,
        description="Extra weight added on high-gradient query regions for field/feature supervision.",
    )
    near_wall_region_weight: float = Field(
        default=0.0,
        description="Extra weight added on near-wall query and slice regions.",
    )
    wake_region_weight: float = Field(
        default=0.0,
        description="Extra weight added on wake-region query and slice regions.",
    )
    surface_leading_edge_weight: float = Field(
        default=0.0,
        description="Extra weight added on leading-edge surface regions for surface losses.",
    )
    near_wall_distance_fraction: float = Field(
        default=0.08,
        description="Near-wall distance threshold as a fraction of chord.",
    )
    wake_halfwidth_fraction: float = Field(
        default=0.15,
        description="Wake half-width as a fraction of chord when constructing wake masks.",
    )
    leading_edge_fraction: float = Field(
        default=0.15,
        description="Leading-edge region width as a fraction of chord for surface weighting.",
    )
    use_feature_class_balancing: bool = Field(
        default=False,
        description="Enable positive-class reweighting for BCE-style feature supervision.",
    )
    feature_positive_weight: float = Field(
        default=1.0,
        description="Static positive-class weight for feature BCE. Values <= 1 fall back to dynamic class balancing when enabled.",
    )
    feature_max_positive_weight: float = Field(
        default=10.0,
        description="Upper bound for dynamic feature positive-class weighting.",
    )
    feature_focal_gamma: float = Field(
        default=0.0,
        description="Optional focal-style gamma applied to feature BCE loss.",
    )

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
        if self.use_hard_region_weighting:
            region_weights = (
                self.high_gradient_region_weight,
                self.near_wall_region_weight,
                self.wake_region_weight,
                self.surface_leading_edge_weight,
            )
            if max(region_weights) <= 0.0:
                raise ValueError(
                    "use_hard_region_weighting=true requires at least one of "
                    "high_gradient_region_weight, near_wall_region_weight, wake_region_weight, "
                    "or surface_leading_edge_weight to be > 0."
                )
        for name, value in [
            ("near_wall_distance_fraction", self.near_wall_distance_fraction),
            ("wake_halfwidth_fraction", self.wake_halfwidth_fraction),
            ("leading_edge_fraction", self.leading_edge_fraction),
        ]:
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0.")
        if self.feature_positive_weight <= 0.0:
            raise ValueError("feature_positive_weight must be > 0.")
        if self.feature_max_positive_weight <= 0.0:
            raise ValueError("feature_max_positive_weight must be > 0.")
        if self.feature_focal_gamma < 0.0:
            raise ValueError("feature_focal_gamma must be >= 0.")
        for name, value in [
            ("consistency_weight", self.consistency_weight),
            ("lambda_data", self.lambda_data),
            ("lambda_continuity", self.lambda_continuity),
            ("lambda_momentum", self.lambda_momentum),
            ("lambda_nut", self.lambda_nut),
            ("lambda_bc", self.lambda_bc),
            ("lambda_consistency", self.lambda_consistency),
            ("physics_schedule_max_weight", self.physics_schedule_max_weight),
        ]:
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0.")
        if self.physics_warmup_epochs < 0:
            raise ValueError("physics_warmup_epochs must be >= 0.")
        if self.physics_ramp_epochs < 0:
            raise ValueError("physics_ramp_epochs must be >= 0.")
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
