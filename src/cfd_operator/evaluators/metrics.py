"""Evaluation metrics."""

from __future__ import annotations

import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def relative_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0e-8) -> float:
    numerator = np.linalg.norm(y_true - y_pred)
    denominator = np.linalg.norm(y_true) + eps
    return float(numerator / denominator)


def binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    true_label = np.asarray(y_true).reshape(-1) >= threshold
    pred_label = np.asarray(y_pred).reshape(-1) >= threshold
    return float(np.mean(true_label == pred_label))


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, eps: float = 1.0e-8) -> float:
    true_label = np.asarray(y_true).reshape(-1) >= threshold
    pred_label = np.asarray(y_pred).reshape(-1) >= threshold
    tp = np.sum(true_label & pred_label)
    fp = np.sum(~true_label & pred_label)
    fn = np.sum(true_label & ~pred_label)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return float(2.0 * precision * recall / (precision + recall + eps))


def binary_iou(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, eps: float = 1.0e-8) -> float:
    true_label = np.asarray(y_true).reshape(-1) >= threshold
    pred_label = np.asarray(y_pred).reshape(-1) >= threshold
    intersection = np.sum(true_label & pred_label)
    union = np.sum(true_label | pred_label)
    return float(intersection / (union + eps))
