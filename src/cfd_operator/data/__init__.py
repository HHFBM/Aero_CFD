"""Dataset and datamodule utilities."""

from .adapters import (
    AirfRANSAdapter,
    AirfRANSOriginalAdapter,
    NPZFileAdapter,
    SyntheticAdapter,
    build_adapter,
)
from .airfrans import AirfRANSDatasetConverter
from .module import CFDDataModule
from .synthetic import SyntheticAirfoilDatasetGenerator

__all__ = [
    "AirfRANSAdapter",
    "AirfRANSDatasetConverter",
    "AirfRANSOriginalAdapter",
    "CFDDataModule",
    "NPZFileAdapter",
    "SyntheticAdapter",
    "SyntheticAirfoilDatasetGenerator",
    "build_adapter",
]
