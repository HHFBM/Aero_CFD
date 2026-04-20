from __future__ import annotations

import numpy as np
import pytest
import torch

from cfd_operator.config.schemas import DataConfig, ModelConfig
from cfd_operator.data import CFDDataModule
from cfd_operator.geometry import BranchInputAdapter, BranchInputContract, GeometryFeatureBuilder, NACA4Airfoil
from cfd_operator.models import create_model


def test_legacy_geometry_feature_builder_matches_manual_logic() -> None:
    airfoil = NACA4Airfoil(max_camber=0.02, camber_position=0.4, thickness=0.12)
    builder = GeometryFeatureBuilder(branch_feature_mode="params", signature_points=32)
    features = builder.build_from_airfoil(airfoil, mach=0.4, aoa_deg=3.0, reynolds=1.0e6)
    expected = np.concatenate([airfoil.parameter_vector(), np.asarray([0.4, 3.0, 1.0e6], dtype=np.float32)], axis=0)
    assert np.allclose(features, expected.astype(np.float32))


def test_geometry_feature_builder_points_mode_dimension_matches_legacy_pattern() -> None:
    airfoil = NACA4Airfoil(max_camber=0.02, camber_position=0.4, thickness=0.12)
    builder = GeometryFeatureBuilder(branch_feature_mode="points", signature_points=24)
    features = builder.build_from_airfoil(airfoil, mach=0.35, aoa_deg=2.0)
    expected_dim = airfoil.parameter_vector().shape[0] + 2 + 24 * 2
    assert features.shape == (expected_dim,)


def test_branch_input_adapter_dual_routing_shapes_match() -> None:
    airfoil = NACA4Airfoil(max_camber=0.03, camber_position=0.4, thickness=0.13)
    surface_points = airfoil.surface_points(61)
    legacy_adapter = BranchInputAdapter(
        branch_input_mode="legacy_fixed_features",
        branch_feature_mode="params",
        signature_points=32,
    )
    encoded_adapter = BranchInputAdapter(
        branch_input_mode="encoded_geometry",
        branch_feature_mode="params",
        signature_points=32,
        encoded_geometry_latent_dim=12,
    )
    legacy = legacy_adapter.build_from_surface_points(surface_points, mach=0.42, aoa_deg=4.0)
    encoded = encoded_adapter.build_from_surface_points(surface_points, mach=0.42, aoa_deg=4.0)
    assert legacy.shape == encoded.shape
    assert not np.allclose(legacy, encoded)


def test_encoded_geometry_mode_minimal_forward() -> None:
    airfoil = NACA4Airfoil(max_camber=0.02, camber_position=0.4, thickness=0.12)
    adapter = BranchInputAdapter(
        branch_input_mode="encoded_geometry",
        branch_feature_mode="params",
        signature_points=32,
        encoded_geometry_latent_dim=10,
    )
    branch_inputs = adapter.build_from_airfoil(airfoil, mach=0.4, aoa_deg=2.0)
    model = create_model(
        ModelConfig(
            name="deeponet",
            branch_input_dim=int(branch_inputs.shape[0]),
            trunk_input_dim=2,
            field_output_dim=4,
            scalar_output_dim=2,
            hidden_dim=16,
            latent_dim=16,
            fourier_features_dim=8,
        )
    )
    outputs = model(
        branch_inputs=torch.from_numpy(branch_inputs).unsqueeze(0),
        query_points=torch.randn(1, 6, 2),
    )
    assert outputs["fields"].shape == (1, 6, 4)
    assert outputs["scalars"].shape == (1, 2)


def test_encoded_geometry_missing_raw_surface_points_falls_back_to_legacy() -> None:
    geometry_params = np.asarray([0.02, 0.4, 0.12, 1.0], dtype=np.float32)
    adapter = BranchInputAdapter(
        branch_input_mode="encoded_geometry",
        branch_feature_mode="params",
        signature_points=32,
    )
    with pytest.warns(UserWarning, match="falling back to legacy_fixed_features"):
        branch_inputs = adapter.build_from_geometry_params(
            geometry_params,
            mach=0.3,
            aoa_deg=1.0,
            surface_points=None,
        )
    expected = np.concatenate([geometry_params, np.asarray([0.3, 1.0], dtype=np.float32)], axis=0)
    assert np.allclose(branch_inputs, expected)


def test_default_config_keeps_legacy_branch_mode(tmp_path) -> None:
    config = DataConfig(
        dataset_type="synthetic",
        dataset_path=str(tmp_path / "branch_mode_dataset.npz"),
        num_geometries=4,
        conditions_per_geometry=3,
        num_query_points=16,
        num_surface_points=12,
    )
    assert config.branch_input_mode == "legacy_fixed_features"
    data_module = CFDDataModule(config=config, batch_size=2)
    data_module.setup()
    batch = next(iter(data_module.train_dataloader()))
    assert batch["branch_input_mode"][0] == "legacy_fixed_features"


def test_branch_contract_roundtrip() -> None:
    contract = BranchInputContract(
        branch_input_mode="legacy_fixed_features",
        branch_feature_mode="params",
        branch_input_dim=6,
        geometry_representation="parameterized_geometry",
        branch_encoding_type="naca_parameter_vector_plus_flow",
        include_reynolds=False,
        num_surface_points=32,
        encoded_geometry_latent_dim=16,
    )
    restored = BranchInputContract.from_dict(contract.as_dict())
    assert restored == contract
