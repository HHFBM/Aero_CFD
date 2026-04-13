from __future__ import annotations

from pathlib import Path

from cfd_operator.config.schemas import DataConfig
from cfd_operator.data import CFDDataModule


def test_data_module_builds_synthetic_dataset(tmp_path: Path) -> None:
    config = DataConfig(
        dataset_type="synthetic",
        dataset_path=str(tmp_path / "toy_dataset.npz"),
        num_samples=12,
        num_query_points=32,
        num_surface_points=24,
    )
    data_module = CFDDataModule(config=config, batch_size=4)
    data_module.setup()

    batch = next(iter(data_module.train_dataloader()))
    assert batch["branch_inputs"].shape[0] == 4
    assert batch["query_points"].shape[-1] == 2
    assert batch["field_targets"].shape[-1] == 4
    assert batch["surface_cp"].shape[-1] == 1

