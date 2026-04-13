"""Dataset and datamodule utilities."""

from .module import CFDDataModule
from .synthetic import SyntheticAirfoilDatasetGenerator

__all__ = ["CFDDataModule", "SyntheticAirfoilDatasetGenerator"]

