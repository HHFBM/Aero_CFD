"""Schedulers for gradually enabling physics-informed losses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LossScheduleState:
    epoch: int = 0
    global_step: int = 0


@dataclass(frozen=True)
class PhysicsLossScheduler:
    warmup_epochs: int = 0
    ramp_epochs: int = 0
    max_weight: float = 1.0

    def multiplier(self, state: LossScheduleState) -> float:
        if self.max_weight <= 0.0:
            return 0.0
        if state.epoch < self.warmup_epochs:
            return 0.0
        if self.ramp_epochs <= 0:
            return float(self.max_weight)
        progress_epoch = state.epoch - self.warmup_epochs + 1
        progress = min(max(progress_epoch / float(self.ramp_epochs), 0.0), 1.0)
        return float(self.max_weight * progress)


def build_physics_loss_scheduler(
    *,
    warmup_epochs: int,
    ramp_epochs: int,
    max_weight: float,
) -> PhysicsLossScheduler:
    return PhysicsLossScheduler(
        warmup_epochs=max(int(warmup_epochs), 0),
        ramp_epochs=max(int(ramp_epochs), 0),
        max_weight=max(float(max_weight), 0.0),
    )
