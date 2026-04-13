"""NACA 4-digit parameterization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import AirfoilParameterization


@dataclass
class NACA4Airfoil(AirfoilParameterization):
    """Parameterized NACA 4-digit airfoil."""

    max_camber: float
    camber_position: float
    thickness: float
    chord: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.max_camber <= 0.09):
            raise ValueError("max_camber should be in [0, 0.09]")
        if not (0.05 <= self.camber_position <= 0.9):
            raise ValueError("camber_position should be in [0.05, 0.9]")
        if not (0.04 <= self.thickness <= 0.24):
            raise ValueError("thickness should be in [0.04, 0.24]")

    def parameter_vector(self) -> np.ndarray:
        return np.asarray(
            [self.max_camber, self.camber_position, self.thickness, self.chord],
            dtype=np.float32,
        )

    def thickness_distribution(self, x: np.ndarray) -> np.ndarray:
        xc = np.clip(x / self.chord, 1.0e-6, 1.0)
        yt = (
            5.0
            * self.thickness
            * self.chord
            * (
                0.2969 * np.sqrt(xc)
                - 0.1260 * xc
                - 0.3516 * xc**2
                + 0.2843 * xc**3
                - 0.1015 * xc**4
            )
        )
        return yt

    def camber_line(self, x: np.ndarray) -> np.ndarray:
        xc = np.clip(x / self.chord, 0.0, 1.0)
        m = self.max_camber
        p = self.camber_position

        yc = np.where(
            xc < p,
            m / (p**2) * (2 * p * xc - xc**2),
            m / ((1 - p) ** 2) * ((1 - 2 * p) + 2 * p * xc - xc**2),
        )
        return yc * self.chord

    def camber_slope(self, x: np.ndarray) -> np.ndarray:
        xc = np.clip(x / self.chord, 0.0, 1.0)
        m = self.max_camber
        p = self.camber_position
        dyc_dx = np.where(
            xc < p,
            2 * m / (p**2) * (p - xc),
            2 * m / ((1 - p) ** 2) * (p - xc),
        )
        return dyc_dx

    def upper_lower_surfaces(self, num_points: int = 200):
        beta = np.linspace(0.0, np.pi, num_points, dtype=np.float64)
        x = 0.5 * self.chord * (1.0 - np.cos(beta))
        yc = self.camber_line(x)
        yt = self.thickness_distribution(x)
        theta = np.arctan(self.camber_slope(x))

        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)
        upper = np.stack([xu, yu], axis=1)
        lower = np.stack([xl, yl], axis=1)
        return upper.astype(np.float32), lower.astype(np.float32)

    def surface_points(self, num_points: int = 200) -> np.ndarray:
        upper, lower = self.upper_lower_surfaces(num_points=max(2, num_points // 2 + 1))
        # Closed contour: TE -> LE on upper, LE -> TE on lower.
        return np.concatenate([upper[::-1], lower[1:]], axis=0).astype(np.float32)
