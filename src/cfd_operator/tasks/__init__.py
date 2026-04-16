"""Task and capability abstractions for CFD surrogate training/evaluation."""

from .capabilities import (
    CapabilityStatus,
    DatasetCapability,
    EffectiveTaskSet,
    TaskRequest,
    build_task_request_from_loss_config,
    infer_dataset_capability,
    resolve_effective_tasks,
)

__all__ = [
    "CapabilityStatus",
    "DatasetCapability",
    "EffectiveTaskSet",
    "TaskRequest",
    "build_task_request_from_loss_config",
    "infer_dataset_capability",
    "resolve_effective_tasks",
]
