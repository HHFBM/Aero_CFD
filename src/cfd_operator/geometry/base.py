"""Geometry abstractions.

The abstract geometry object is only one possible upstream representation. The
training pipeline ultimately consumes branch inputs, so dataset metadata may
describe geometry summaries or sampled contours that are not safely
reconstructable into a parameterized geometry object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AirfoilParameterization(ABC):
    """Abstract interface for parameterized airfoil geometry."""

    @abstractmethod
    def surface_points(self, num_points: int = 200) -> np.ndarray:
        """Return closed airfoil surface points with shape [num_points, 2]."""

    @abstractmethod
    def parameter_vector(self) -> np.ndarray:
        """Return fixed-length geometry parameters for branch encoding."""

    @abstractmethod
    def camber_line(self, x: np.ndarray) -> np.ndarray:
        """Return camber line y-values for the provided x-coordinates."""

    @abstractmethod
    def thickness_distribution(self, x: np.ndarray) -> np.ndarray:
        """Return half-thickness values for the provided x-coordinates."""
