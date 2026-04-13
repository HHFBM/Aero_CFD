"""Plotting utilities for training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_scalar_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    save_path: Union[str, Path],
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolor="none")
    min_value = float(min(y_true.min(), y_pred.min()))
    max_value = float(max(y_true.max(), y_pred.max()))
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="black")
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(f"{label} parity plot")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_field_scatter(
    points: np.ndarray,
    values: np.ndarray,
    title: str,
    save_path: Union[str, Path],
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=values, s=20, cmap="viridis")
    fig.colorbar(scatter, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_cp_comparison(
    surface_points: np.ndarray,
    cp_true: np.ndarray,
    cp_pred: np.ndarray,
    save_path: Union[str, Path],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = surface_points[:, 0]
    ax.plot(x, cp_true, label="true", color="tab:blue")
    ax.plot(x, cp_pred, label="pred", color="tab:orange")
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("Cp")
    ax.set_title("Surface Cp comparison")
    ax.legend()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(history_csv: Union[str, Path], save_path: Union[str, Path]) -> None:
    history = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["train_loss_total"], label="train")
    ax.plot(history["epoch"], history["val_loss_total"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training history")
    ax.legend()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
