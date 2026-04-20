from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cfd_operator.config.schemas import DataConfig
from cfd_operator.data import CFDDataModule
from cfd_operator.data.file_dataset import load_dataset_payload
from cfd_operator.data.splitting import build_frozen_benchmark_splits


def test_data_module_builds_synthetic_dataset(tmp_path: Path) -> None:
    config = DataConfig(
        dataset_type="synthetic",
        dataset_path=str(tmp_path / "toy_dataset.npz"),
        num_samples=12,
        num_geometries=4,
        conditions_per_geometry=3,
        num_query_points=32,
        num_surface_points=24,
    )
    data_module = CFDDataModule(config=config, batch_size=4)
    data_module.setup()

    batch = next(iter(data_module.train_dataloader()))
    assert 1 <= batch["branch_inputs"].shape[0] <= 4
    assert batch["query_points"].shape[-1] == 2
    assert batch["field_targets"].shape[-1] == 4
    assert batch["surface_cp"].shape[-1] == 1
    assert batch["surface_normals"].shape[-1] == 2
    assert batch["farfield_targets"].shape[-1] == 4
    assert "geometry_mode" in batch
    assert batch["geometry_mode"][0] == "legacy_naca_params"
    assert "test_unseen_geometry" in data_module.available_splits()
    assert "test_unseen_condition" in data_module.available_splits()


def test_data_module_builds_schema_backed_synthetic_dataset(tmp_path: Path) -> None:
    config = DataConfig(
        dataset_type="synthetic",
        dataset_view_mode="schema",
        dataset_path=str(tmp_path / "toy_dataset_schema.npz"),
        num_samples=12,
        num_geometries=4,
        conditions_per_geometry=3,
        num_query_points=32,
        num_surface_points=24,
    )
    data_module = CFDDataModule(config=config, batch_size=4)
    data_module.setup()

    assert data_module.unified_samples
    batch = next(iter(data_module.train_dataloader()))
    assert 1 <= batch["branch_inputs"].shape[0] <= 4
    assert batch["query_points"].shape[-1] == 2
    assert batch["field_targets"].shape[-1] == 4
    assert batch["surface_cp"].shape[-1] == 1
    assert batch["geometry_mode"][0] == "legacy_naca_params"
    assert "test_unseen_geometry" in data_module.available_splits()
    assert "test_unseen_condition" in data_module.available_splits()


def test_unseen_geometry_split_is_disjoint(tmp_path: Path) -> None:
    config = DataConfig(
        dataset_type="synthetic",
        dataset_path=str(tmp_path / "toy_dataset.npz"),
        num_geometries=8,
        conditions_per_geometry=4,
        num_query_points=24,
        num_surface_points=20,
    )
    data_module = CFDDataModule(config=config, batch_size=4)
    data_module.setup()

    assert data_module.payload is not None
    airfoil_ids = data_module.payload["airfoil_id"]
    train_ids = set(airfoil_ids[data_module.payload["train_indices"]].tolist())
    unseen_ids = set(airfoil_ids[data_module.payload["test_unseen_geometry_indices"]].tolist())
    assert train_ids.isdisjoint(unseen_ids)


def test_file_dataset_generic_geometry_requires_branch_encoding_without_config(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "sample_id": [0, 0, 0, 0],
            "geometry_mode": ["generic_surface_points"] * 4,
            "geometry_x": [1.0, 0.5, 0.0, 0.5],
            "geometry_y": [0.0, 0.1, 0.0, -0.1],
            "mach": [0.3] * 4,
            "aoa": [2.0] * 4,
            "x": [0.0, 0.2, 0.4, 0.6],
            "y": [0.0, 0.1, 0.0, -0.1],
            "u": [1.0, 1.0, 1.0, 1.0],
            "v": [0.0, 0.0, 0.0, 0.0],
            "p": [1.0, 1.0, 1.0, 1.0],
            "surface_flag": [1, 1, 1, 1],
            "cp": [0.0, 0.0, 0.0, 0.0],
            "cl": [0.1] * 4,
            "cd": [0.01] * 4,
            "fidelity_level": [0] * 4,
            "source": ["generic_tabular"] * 4,
            "convergence_flag": [1] * 4,
        }
    )
    path = tmp_path / "generic.csv"
    frame.to_csv(path, index=False)

    try:
        load_dataset_payload(path)
    except ValueError as exc:
        assert "branch_* columns" in str(exc) or "DataConfig-driven geometry encoder" in str(exc)
    else:
        raise AssertionError("Expected generic tabular dataset without branch encoding to raise a clear error.")


def test_file_dataset_generic_geometry_with_precomputed_branch_loads(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "sample_id": [0, 0, 0, 0],
            "geometry_mode": ["generic_surface_points"] * 4,
            "geometry_x": [1.0, 0.5, 0.0, 0.5],
            "geometry_y": [0.0, 0.1, 0.0, -0.1],
            "branch_0": [0.1] * 4,
            "branch_1": [0.2] * 4,
            "branch_2": [0.3] * 4,
            "branch_3": [0.4] * 4,
            "mach": [0.3] * 4,
            "aoa": [2.0] * 4,
            "x": [0.0, 0.2, 0.4, 0.6],
            "y": [0.0, 0.1, 0.0, -0.1],
            "u": [1.0, 1.0, 1.0, 1.0],
            "v": [0.0, 0.0, 0.0, 0.0],
            "p": [1.0, 1.0, 1.0, 1.0],
            "surface_flag": [1, 1, 1, 1],
            "cp": [0.0, 0.0, 0.0, 0.0],
            "cl": [0.1] * 4,
            "cd": [0.01] * 4,
            "fidelity_level": [0] * 4,
            "source": ["generic_tabular"] * 4,
            "convergence_flag": [1] * 4,
        }
    )
    path = tmp_path / "generic_with_branch.csv"
    frame.to_csv(path, index=False)
    payload = load_dataset_payload(path)
    assert payload["geometry_mode"][0] == "generic_surface_points"
    assert payload["branch_inputs"].shape == (1, 4)


def test_file_dataset_generic_geometry_can_derive_branch_inputs_with_config(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "sample_id": [0, 0, 0, 0],
            "geometry_mode": ["generic_surface_points"] * 4,
            "geometry_x": [1.0, 0.5, 0.0, 0.5],
            "geometry_y": [0.0, 0.1, 0.0, -0.1],
            "mach": [0.3] * 4,
            "aoa": [2.0] * 4,
            "x": [0.0, 0.2, 0.4, 0.6],
            "y": [0.0, 0.1, 0.0, -0.1],
            "u": [1.0, 1.0, 1.0, 1.0],
            "v": [0.0, 0.0, 0.0, 0.0],
            "p": [1.0, 1.0, 1.0, 1.0],
            "surface_flag": [1, 1, 1, 1],
            "cp": [0.0, 0.0, 0.0, 0.0],
            "cl": [0.1] * 4,
            "cd": [0.01] * 4,
            "fidelity_level": [0] * 4,
            "source": ["generic_tabular"] * 4,
            "convergence_flag": [1] * 4,
        }
    )
    path = tmp_path / "generic_derived.csv"
    frame.to_csv(path, index=False)
    config = DataConfig(dataset_type="file", dataset_path=str(path), branch_feature_mode="points", num_surface_points=4)
    payload = load_dataset_payload(path, config=config)
    assert payload["geometry_mode"][0] == "generic_surface_points"
    assert payload["branch_encoding_type"][0] == "derived_surface_signature_plus_flow"
    assert payload["branch_inputs"].shape == (1, 10)


def test_file_dataset_generic_minimum_contract_loads_without_surface_targets(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "sample_id": [0, 0, 0, 0],
            "geometry_mode": ["generic_surface_points"] * 4,
            "geometry_x": [1.0, 0.5, 0.0, 0.5],
            "geometry_y": [0.0, 0.1, 0.0, -0.1],
            "mach": [0.3] * 4,
            "aoa": [2.0] * 4,
            "x": [0.0, 0.2, 0.4, 0.6],
            "y": [0.0, 0.1, 0.0, -0.1],
            "u": [1.0, 1.0, 1.0, 1.0],
            "v": [0.0, 0.0, 0.0, 0.0],
            "p": [1.0, 1.0, 1.0, 1.0],
            "cl": [0.1] * 4,
            "cd": [0.01] * 4,
        }
    )
    path = tmp_path / "generic_minimum.csv"
    frame.to_csv(path, index=False)
    config = DataConfig(dataset_type="file", dataset_path=str(path), branch_feature_mode="points", num_surface_points=4)
    payload = load_dataset_payload(path, config=config)
    assert payload["branch_inputs"].shape == (1, 10)
    assert payload["surface_cp_available"].shape == (1, 4)
    assert float(payload["surface_cp_available"][0].sum()) == 0.0
    assert float(payload["surface_pressure_available"][0].sum()) == 0.0
    assert float(payload["feature_available"][0].sum()) == 0.0
    assert payload["source"][0] == "generic_tabular"


def test_data_module_discovers_benchmark_holdout_split(tmp_path: Path) -> None:
    source_config = DataConfig(
        dataset_type="synthetic",
        dataset_path=str(tmp_path / "source_synthetic.npz"),
        num_samples=20,
        num_geometries=5,
        conditions_per_geometry=4,
        num_query_points=24,
        num_surface_points=20,
    )
    source_data_module = CFDDataModule(config=source_config, batch_size=4)
    source_data_module.setup()
    assert source_data_module.payload is not None
    payload = {
        key: value
        for key, value in source_data_module.payload.items()
        if not key.endswith("_indices")
    }
    splits = build_frozen_benchmark_splits(
        geometry_ids=np.asarray(payload["airfoil_id"]),
        benchmark_holdout_ratio=0.1,
        train_ratio=0.7,
        val_ratio=0.15,
        rng=np.random.default_rng(42),
    )
    payload.update(splits)
    benchmark_path = tmp_path / "synthetic_benchmarksplit.npz"
    np.savez_compressed(benchmark_path, **payload)

    benchmark_config = DataConfig(dataset_type="file", dataset_path=str(benchmark_path))
    benchmark_data_module = CFDDataModule(config=benchmark_config, batch_size=4)
    benchmark_data_module.setup()

    assert "benchmark_holdout" in benchmark_data_module.available_splits()
    assert benchmark_data_module.datasets["benchmark_holdout"] is not None
    benchmark_indices = np.asarray(benchmark_data_module.payload["benchmark_holdout_indices"], dtype=np.int64)
    train_indices = np.asarray(benchmark_data_module.payload["train_indices"], dtype=np.int64)
    assert np.intersect1d(benchmark_indices, train_indices).size == 0
