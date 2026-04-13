"""Geometry plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt

from .base import AirfoilParameterization


def plot_airfoil(airfoil: AirfoilParameterization, save_path: Optional[Union[str, Path]] = None) -> None:
    points = airfoil.surface_points(200)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(points[:, 0], points[:, 1], color="tab:blue")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Airfoil geometry")
    ax.grid(True, alpha=0.3)
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
