from __future__ import annotations

from pathlib import Path

from cfd_operator.config.schemas import DataConfig
from cfd_operator.data import CFDDataModule
from cfd_operator.schema import sample_from_legacy_payload


def test_unified_schema_builds_from_legacy_payload(tmp_path: Path) -> None:
    config = DataConfig(
        dataset_type="synthetic",
        dataset_path=str(tmp_path / "schema_dataset.npz"),
        num_geometries=4,
        conditions_per_geometry=3,
        num_query_points=16,
        num_surface_points=12,
    )
    data_module = CFDDataModule(config=config, batch_size=2)
    data_module.setup()
    assert data_module.payload is not None

    sample = sample_from_legacy_payload(
        data_module.payload,
        0,
        dataset_name=config.name,
        split="train",
        source_format="synthetic_npz",
    )
    assert sample.metadata.dataset_name == config.name
    assert sample.metadata.dimensionality == "2d"
    assert sample.geometry.geometry_mode == "legacy_naca_params"
    assert "field_query_points" in sample.query_sets
    assert sample.targets.field_targets is not None
    assert "surface_cp" in sample.targets.surface_targets
    assert sample.availability.available["field_targets"] is True
