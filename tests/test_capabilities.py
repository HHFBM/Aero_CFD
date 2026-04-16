from __future__ import annotations

from pathlib import Path

from cfd_operator.config.schemas import DataConfig, LossConfig
from cfd_operator.data import CFDDataModule
from cfd_operator.tasks import (
    build_task_request_from_loss_config,
    resolve_effective_tasks,
)


def test_capability_and_task_request_intersection(tmp_path: Path) -> None:
    config = DataConfig(
        dataset_type="synthetic",
        dataset_path=str(tmp_path / "capability_dataset.npz"),
        num_geometries=3,
        conditions_per_geometry=3,
        num_query_points=16,
        num_surface_points=12,
    )
    data_module = CFDDataModule(config=config, batch_size=2)
    data_module.setup()
    assert data_module.dataset_capability is not None

    request = build_task_request_from_loss_config(
        LossConfig(
            use_physics=True,
            use_feature_loss=True,
            feature_weight=0.02,
            use_slice_loss=True,
            slice_weight=0.02,
        )
    )
    effective = resolve_effective_tasks(data_module.dataset_capability, request)

    assert effective.field is True
    assert effective.scalar is True
    assert effective.surface is True
    assert effective.slice is True
    assert effective.feature is True
    assert effective.consistency is True
