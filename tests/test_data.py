from __future__ import annotations

from pathlib import Path

from cfd_operator.config.schemas import DataConfig
from cfd_operator.data import CFDDataModule


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
