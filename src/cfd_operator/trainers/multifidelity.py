"""Reserved interfaces for future multifidelity training."""

from __future__ import annotations

from cfd_operator.trainers.base import BaseTrainer


class MultiFidelityTrainer(BaseTrainer):
    def fit(self) -> dict[str, list[float]]:
        raise NotImplementedError("Multifidelity training is reserved for a future release.")

