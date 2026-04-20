"""Plotting utilities for training, evaluation, and inference exports."""

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
    plot_surface_cp(
        surface_points=surface_points,
        cp_pred=cp_pred,
        cp_true=cp_true,
        save_path=save_path,
    )


def plot_surface_cp(
    surface_points: np.ndarray,
    cp_pred: np.ndarray,
    save_path: Union[str, Path],
    cp_true: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = surface_points[:, 0]
    if cp_true is not None:
        ax.plot(x, cp_true, label="true", color="tab:blue")
    ax.plot(x, cp_pred, label="pred" if cp_true is not None else "value", color="tab:orange")
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("Cp")
    ax.set_title("Surface Cp")
    if cp_true is not None:
        ax.legend()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_surface_pressure(
    surface_points: np.ndarray,
    pressure_pred: np.ndarray,
    save_path: Union[str, Path],
    pressure_true: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = surface_points[:, 0]
    if pressure_true is not None:
        ax.plot(x, pressure_true, label="true", color="tab:blue")
    ax.plot(x, pressure_pred, label="pred" if pressure_true is not None else "value", color="tab:red")
    ax.set_xlabel("x")
    ax.set_ylabel("Raw pressure")
    ax.set_title("Surface raw pressure")
    if pressure_true is not None:
        ax.legend()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_slice_field(
    slice_points: np.ndarray,
    pred_values: np.ndarray,
    variable_name: str,
    save_path: Union[str, Path],
    true_values: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    arc = np.cumsum(np.linalg.norm(np.diff(slice_points, axis=0, prepend=slice_points[:1]), axis=1))
    if true_values is not None:
        ax.plot(arc, true_values, label="true", color="tab:blue")
    ax.plot(arc, pred_values, label="pred" if true_values is not None else "value", color="tab:orange")
    ax.set_xlabel("slice distance")
    ax.set_ylabel(variable_name)
    ax.set_title(f"Slice {variable_name}")
    if true_values is not None:
        ax.legend()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_high_gradient_regions(
    points: np.ndarray,
    indicator: np.ndarray,
    save_path: Union[str, Path],
    title: str = "High-gradient / shock regions",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    indicator = np.asarray(indicator).reshape(-1)
    scatter = ax.scatter(points[:, 0], points[:, 1], c=indicator, s=18, cmap="inferno")
    fig.colorbar(scatter, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_scalar_summary(
    scalar_values: dict[str, float],
    save_path: Union[str, Path],
    true_values: dict[str, float] | None = None,
) -> None:
    labels = list(scalar_values.keys())
    values = [float(scalar_values[key]) for key in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    if true_values is None:
        ax.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green", "tab:red"][: len(labels)])
    else:
        true_bar = [float(true_values.get(key, np.nan)) for key in labels]
        positions = np.arange(len(labels))
        width = 0.35
        ax.bar(positions - width / 2.0, true_bar, width=width, label="true", color="tab:blue")
        ax.bar(positions + width / 2.0, values, width=width, label="pred", color="tab:orange")
        ax.set_xticks(positions, labels)
        ax.legend()
    ax.set_ylabel("value")
    ax.set_title("Scalar summary")
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


def plot_split_metric_bars(
    metric_values_by_split: dict[str, float],
    title: str,
    ylabel: str,
    save_path: Union[str, Path],
) -> None:
    labels = list(metric_values_by_split.keys())
    values = [float(metric_values_by_split[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green"][: len(labels)])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_gap_bars(
    gap_values: dict[str, float],
    title: str,
    save_path: Union[str, Path],
) -> None:
    labels = list(gap_values.keys())
    values = [float(gap_values[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color="tab:red")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel("gap")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_multi_experiment_metric_comparison(
    experiment_names: list[str],
    test_values: list[float],
    benchmark_values: list[float],
    metric_name: str,
    save_path: Union[str, Path],
) -> None:
    positions = np.arange(len(experiment_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(positions - width / 2.0, test_values, width=width, label="test", color="tab:blue")
    ax.bar(positions + width / 2.0, benchmark_values, width=width, label="benchmark_holdout", color="tab:orange")
    ax.set_xticks(positions, experiment_names, rotation=20)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name}: test vs benchmark_holdout")
    ax.legend()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
