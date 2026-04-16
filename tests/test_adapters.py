from __future__ import annotations

from pathlib import Path

from cfd_operator.config.schemas import DataConfig
from cfd_operator.data import CFDDataModule, build_adapter
from cfd_operator.data.adapters import AirfRANSAdapter, NPZFileAdapter


def test_airfrans_adapter_outputs_unified_samples(tmp_path: Path) -> None:
    synthetic_config = DataConfig(
        dataset_type="synthetic",
        dataset_path=str(tmp_path / "adapter_dataset.npz"),
        num_geometries=5,
        conditions_per_geometry=4,
        num_query_points=12,
        num_surface_points=10,
    )
    data_module = CFDDataModule(config=synthetic_config, batch_size=2)
    dataset_path = data_module.prepare_data()
    adapter = AirfRANSAdapter(
        DataConfig(
            dataset_type="airfrans",
            dataset_path=str(dataset_path),
            num_query_points=12,
            num_surface_points=10,
        )
    )
    result = adapter.load(dataset_path)

    assert result.dataset_name == "toy_airfoil"
    assert result.samples
    assert result.samples[0].metadata.source_format == "airfrans_npz"
    assert result.capability.available("field_targets") is True


def test_npz_file_adapter_outputs_unified_samples(tmp_path: Path) -> None:
    synthetic_config = DataConfig(
        dataset_type="synthetic",
        dataset_path=str(tmp_path / "base_dataset.npz"),
        num_geometries=5,
        conditions_per_geometry=4,
        num_query_points=12,
        num_surface_points=10,
    )
    data_module = CFDDataModule(config=synthetic_config, batch_size=2)
    dataset_path = data_module.prepare_data()

    file_config = DataConfig(
        dataset_type="file",
        dataset_path=str(dataset_path),
        num_query_points=12,
        num_surface_points=10,
    )
    adapter = NPZFileAdapter(file_config)
    result = adapter.load(dataset_path)

    assert result.samples
    assert result.source_format == "file_payload"
    assert result.samples[0].geometry.surface_points is not None
