"""Primary supervised / mixed-physics trainer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from cfd_operator.config.schemas import ProjectConfig
from cfd_operator.data.module import CFDDataModule
from cfd_operator.losses import CompositeLoss
from cfd_operator.models.base import BaseOperatorModel
from cfd_operator.tasks import build_task_request_from_loss_config
from cfd_operator.utils.io import save_json, save_yaml
from cfd_operator.utils.logging import setup_logger
from cfd_operator.utils.paths import RunPaths, build_run_paths


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    epochs_without_improvement: int = 0


@dataclass
class Trainer:
    config: ProjectConfig
    model: BaseOperatorModel
    data_module: CFDDataModule
    loss_fn: CompositeLoss
    run_paths: Optional[RunPaths] = None
    state: TrainerState = field(default_factory=TrainerState)

    def __post_init__(self) -> None:
        self.device = torch.device(self.config.experiment.device)
        self.model.to(self.device)
        self.run_paths = self.run_paths or build_run_paths(self.config.experiment)
        self.logger = setup_logger("train", self.run_paths.logs_dir)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.config.train.mixed_precision and self.device.type == "cuda"
        )
        self.history: list[dict[str, float]] = []
        if hasattr(self.loss_fn, "set_task_context"):
            self.loss_fn.set_task_context(
                dataset_capability=self.data_module.dataset_capability,
                task_request=build_task_request_from_loss_config(self.config.loss),
            )
        save_yaml(self.run_paths.run_dir / "config.yaml", self.config.as_dict())

    def _build_optimizer(self) -> torch.optim.Optimizer:
        optimizer_cls = Adam if self.config.train.optimizer == "adam" else AdamW
        return optimizer_cls(
            self.model.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay,
        )

    def _build_scheduler(self):
        scheduler_config = self.config.train.scheduler
        if scheduler_config.name == "none":
            return None
        if scheduler_config.name == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.t_max,
                eta_min=scheduler_config.min_lr,
            )
        if scheduler_config.name == "step":
            return StepLR(
                self.optimizer,
                step_size=scheduler_config.step_size,
                gamma=scheduler_config.gamma,
            )
        raise ValueError(f"Unsupported scheduler: {scheduler_config.name}")

    def fit(self) -> dict[str, list[float]]:
        if self.config.train.resume_from:
            self.load_checkpoint(self.config.train.resume_from)

        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()

        for epoch in range(self.state.epoch, self.config.train.epochs):
            self.state.epoch = epoch
            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics = self._run_epoch(val_loader, training=False)

            if self.scheduler is not None:
                self.scheduler.step()

            record = {"epoch": epoch + 1, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
            self.history.append(record)
            self._save_history()
            self.logger.info(
                (
                    "epoch=%d train_total=%.6f train_field=%.6f train_surface=%.6f "
                    "train_surface_pressure=%.6f train_slice=%.6f train_feature=%.6f "
                    "train_scalar=%.6f train_physics=%.6f train_boundary=%.6f "
                    "val_total=%.6f val_field=%.6f val_surface=%.6f val_surface_pressure=%.6f "
                    "val_slice=%.6f val_feature=%.6f val_scalar=%.6f val_physics=%.6f"
                ),
                epoch + 1,
                train_metrics["loss_total"],
                train_metrics["loss_field"],
                train_metrics["loss_surface"],
                train_metrics["loss_surface_pressure"],
                train_metrics["loss_slice"],
                train_metrics["loss_feature"],
                train_metrics["loss_scalar"],
                train_metrics["loss_physics"],
                train_metrics["loss_boundary"],
                val_metrics["loss_total"],
                val_metrics["loss_field"],
                val_metrics["loss_surface"],
                val_metrics["loss_surface_pressure"],
                val_metrics["loss_slice"],
                val_metrics["loss_feature"],
                val_metrics["loss_scalar"],
                val_metrics["loss_physics"],
            )

            if (epoch + 1) % self.config.train.checkpoint_every_n_epochs == 0:
                self.save_checkpoint(self.run_paths.checkpoints_dir / f"epoch_{epoch + 1:03d}.pt", is_best=False)

            improved = val_metrics["loss_total"] < self.state.best_val_loss
            if improved:
                self.state.best_val_loss = val_metrics["loss_total"]
                self.state.epochs_without_improvement = 0
                self.save_checkpoint(self.run_paths.checkpoints_dir / "best.pt", is_best=True)
            else:
                self.state.epochs_without_improvement += 1

            if self.state.epochs_without_improvement >= self.config.train.early_stopping_patience:
                self.logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

        return self._history_dict()

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(training)
        aggregate: dict[str, float] = {
            "loss_total": 0.0,
            "loss_field": 0.0,
            "loss_scalar": 0.0,
            "loss_surface": 0.0,
            "loss_surface_pressure": 0.0,
            "loss_heat_flux": 0.0,
            "loss_wall_shear": 0.0,
            "loss_slice": 0.0,
            "loss_feature": 0.0,
            "loss_shock_location": 0.0,
            "loss_physics": 0.0,
            "loss_boundary": 0.0,
        }
        num_batches = 0

        for batch in loader:
            batch = self._move_batch(batch)
            requires_grad = training or self.config.loss.use_physics
            with torch.set_grad_enabled(requires_grad):
                with torch.cuda.amp.autocast(
                    enabled=self.config.train.mixed_precision and self.device.type == "cuda"
                ):
                    outputs = self.model.loss_outputs(batch["branch_inputs"], batch["query_points"])
                    loss, metrics = self.loss_fn(model=self.model, batch=batch, outputs=outputs)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    if self.config.train.grad_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.state.global_step += 1

            for key, value in metrics.items():
                aggregate[key] += value
            num_batches += 1

        if num_batches == 0:
            return aggregate
        return {key: value / num_batches for key, value in aggregate.items()}

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    def save_checkpoint(self, path: Union[str, Path], is_best: bool) -> None:
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "config": self.config.as_dict(),
            "normalizers": self.data_module.normalizers.to_dict() if self.data_module.normalizers is not None else None,
            "dataset_capability": (
                self.data_module.dataset_capability.as_dict()
                if self.data_module.dataset_capability is not None
                else None
            ),
            "trainer_state": {
                "epoch": self.state.epoch,
                "global_step": self.state.global_step,
                "best_val_loss": self.state.best_val_loss,
                "epochs_without_improvement": self.state.epochs_without_improvement,
            },
            "is_best": is_best,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.scheduler is not None and checkpoint.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        trainer_state = checkpoint.get("trainer_state", {})
        self.state = TrainerState(
            epoch=trainer_state.get("epoch", 0),
            global_step=trainer_state.get("global_step", 0),
            best_val_loss=trainer_state.get("best_val_loss", float("inf")),
            epochs_without_improvement=trainer_state.get("epochs_without_improvement", 0),
        )

    def _save_history(self) -> None:
        history_path = self.run_paths.reports_dir / "history.csv"
        pd.DataFrame(self.history).to_csv(history_path, index=False)
        save_json(self.run_paths.reports_dir / "latest_metrics.json", self.history[-1])

    def _history_dict(self) -> dict[str, list[float]]:
        if not self.history:
            return {}
        keys = self.history[0].keys()
        return {key: [record[key] for record in self.history] for key in keys}
