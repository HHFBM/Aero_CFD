from __future__ import annotations

from pathlib import Path

from cfd_operator.config.schemas import DataConfig
from cfd_operator.data import CFDDataModule


def test_surface_outputs_present(tmp_path: Path) -> None:
    data_config = DataConfig(
        dataset_type="synthetic",
        dataset_path=str(tmp_path / "toy_dataset.npz"),
        num_geometries=5,
        conditions_per_geometry=4,
        num_query_points=20,
        num_surface_points=18,
        field_names=("u", "v", "p", "nut"),
    )
    data_module = CFDDataModule(config=data_config, batch_size=2)
    data_module.setup()
    batch = next(iter(data_module.train_dataloader()))

    assert batch["surface_cp"].shape[-1] == 1
    assert batch["surface_pressure"].shape[-1] == 1
    assert batch["surface_nut"].shape[-1] == 1
    assert batch["surface_heat_flux"].shape[-1] == 1
    assert batch["surface_wall_shear"].shape[-1] == 1
    assert batch["surface_arc_length"].shape[-1] == 1
