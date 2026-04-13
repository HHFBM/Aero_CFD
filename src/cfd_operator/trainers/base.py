"""Base trainer definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    @abstractmethod
    def fit(self) -> dict[str, list[float]]:
        """Run the training loop and return history."""

