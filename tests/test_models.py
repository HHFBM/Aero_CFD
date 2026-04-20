from __future__ import annotations

import torch

from cfd_operator.config.schemas import ModelConfig
from cfd_operator.models import create_model, create_reserved_geometry_backbone


def test_deeponet_forward_shapes() -> None:
    config = ModelConfig(
        name="deeponet",
        branch_input_dim=6,
        trunk_input_dim=2,
        field_output_dim=4,
        scalar_output_dim=2,
        hidden_dim=32,
        latent_dim=32,
        fourier_features_dim=8,
    )
    model = create_model(config)
    batch_size = 3
    num_points = 17
    outputs = model(
        branch_inputs=torch.randn(batch_size, 6),
        query_points=torch.randn(batch_size, num_points, 2),
    )
    assert outputs["fields"].shape == (batch_size, num_points, 4)
    assert outputs["scalars"].shape == (batch_size, 2)


def test_geofno_forward_shapes() -> None:
    config = ModelConfig(
        name="geofno",
        branch_input_dim=6,
        trunk_input_dim=2,
        field_output_dim=4,
        scalar_output_dim=2,
        feature_output_dim=2,
        hidden_dim=32,
        latent_dim=32,
        branch_layers=3,
        trunk_layers=3,
        fourier_features_dim=16,
    )
    model = create_model(config)
    batch_size = 3
    num_points = 17
    outputs = model(
        branch_inputs=torch.randn(batch_size, 6),
        query_points=torch.randn(batch_size, num_points, 2),
    )
    assert outputs["fields"].shape == (batch_size, num_points, 4)
    assert outputs["scalars"].shape == (batch_size, 2)
    assert outputs["features"].shape == (batch_size, num_points, 2)


def test_backbone_scalar_metadata_is_consistent() -> None:
    deeponet = create_model(
        ModelConfig(
            name="deeponet",
            branch_input_dim=6,
            trunk_input_dim=2,
            field_output_dim=4,
            scalar_output_dim=2,
            hidden_dim=32,
            latent_dim=32,
            fourier_features_dim=8,
        )
    )
    geofno = create_model(
        ModelConfig(
            name="geofno",
            branch_input_dim=6,
            trunk_input_dim=2,
            field_output_dim=4,
            scalar_output_dim=2,
            feature_output_dim=2,
            hidden_dim=32,
            latent_dim=32,
            branch_layers=3,
            trunk_layers=3,
            fourier_features_dim=16,
        )
    )
    deeponet_meta = deeponet.decoder_head_metadata()
    geofno_meta = geofno.decoder_head_metadata()
    assert deeponet_meta["scalar"]["name"] == "scalar"
    assert geofno_meta["scalar"]["name"] == "scalar"
    assert deeponet_meta["scalar"]["aggregation"] == "branch_only"
    assert geofno_meta["scalar"]["aggregation"] == "mean_max_residual"


def test_default_geometry_backbone_metadata_remains_fixed_branch_vector() -> None:
    model = create_model(
        ModelConfig(
            name="deeponet",
            branch_input_dim=6,
            trunk_input_dim=2,
            field_output_dim=4,
            scalar_output_dim=2,
            hidden_dim=32,
            latent_dim=32,
        )
    )
    metadata = model.model_metadata()
    assert metadata["geometry_backbone"]["mode"] == "fixed_branch_vector"
    assert metadata["geometry_backbone"]["active_in_main_path"] is True


def test_reserved_native_geometry_backbone_interface_encodes_surface_points() -> None:
    config = ModelConfig(
        name="deeponet",
        branch_input_dim=6,
        trunk_input_dim=2,
        field_output_dim=4,
        scalar_output_dim=2,
        hidden_dim=32,
        latent_dim=32,
        geometry_backbone_mode="native_geometry_latent_reserved",
        geometry_backbone_type="reserved_interface",
        native_geometry_latent_dim=48,
        native_geometry_token_dim=24,
        native_geometry_max_tokens=32,
    )
    backbone = create_reserved_geometry_backbone(config)
    assert backbone is not None
    surface_points = torch.randn(2, 40, 2)
    encoded = backbone.encode(surface_points)
    assert encoded.global_latent.shape == (2, 48)
    assert encoded.tokens is not None
    assert encoded.tokens.shape == (2, 32, 24)
    assert encoded.metadata is not None
    assert encoded.metadata["status"] == "experimental_reserved_interface"
