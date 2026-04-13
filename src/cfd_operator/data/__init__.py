"""Dataset and datamodule utilities."""

from .airfrans import AirfRANSDatasetConverter
from .module import CFDDataModule
from .synthetic import SyntheticAirfoilDatasetGenerator

__all__ = ["AirfRANSDatasetConverter", "CFDDataModule", "SyntheticAirfoilDatasetGenerator"]
