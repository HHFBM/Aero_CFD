"""Placeholder CST airfoil interface."""

from __future__ import annotations

import numpy as np

from .base import AirfoilParameterization


class CSTParameterization(AirfoilParameterization):
    """Reserved interface for future CST-based parameterization."""

    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = np.asarray(coefficients, dtype=np.float32)

    def surface_points(self, num_points: int = 200) -> np.ndarray:
        raise NotImplementedError("CSTParameterization is reserved for future extension.")

    def parameter_vector(self) -> np.ndarray:
        return self.coefficients

    def camber_line(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("CSTParameterization is reserved for future extension.")

    def thickness_distribution(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("CSTParameterization is reserved for future extension.")

