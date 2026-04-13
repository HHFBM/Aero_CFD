"""One-command demo for the CFD operator surrogate.

This script builds a tiny synthetic dataset, trains a small DeepONet model for a
few epochs, runs one prediction on a held-out sample, and writes the outputs to
the demo run directory. It is meant to be the fastest way to verify that the
project can train and infer end-to-end on a local machine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cfd_operator.config import load_config
from cfd_operator.data import CFDDataModule
from cfd_operator.inference import Predictor
from cfd_operator.losses import CompositeLoss
from cfd_operator.models import create_model
from cfd_operator.trainers import Trainer
from cfd_operator.utils.io import save_json
from cfd_operator.utils.seed import set_seed


def build_demo_config(config_path: str):
    config = load_config(
        config_path,
        overrides=[
            "experiment.name=demo_run",
            "experiment.device=cpu",
            "data.dataset_type=synthetic",
            "data.dataset_path=outputs/data/demo_dataset.npz",
            "data.num_geometries=4",
            "data.conditions_per_geometry=4",
            "data.num_query_points=48",
            "data.num_surface_points=32",
            "train.epochs=2",
            "train.batch_size=4",
            "train.early_stopping_patience=3",
            "loss.use_physics=true",
            "loss.physics_weight=0.05",
            "loss.boundary_weight=0.1",
            "eval.save_plots=false",
        ],
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a complete tiny surrogate demo.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = build_demo_config(args.config)
    set_seed(config.experiment.seed)

    data_module = CFDDataModule(config=config.data, batch_size=config.train.batch_size)
    data_module.setup()
    assert data_module.payload is not None
    assert data_module.normalizers is not None

    config.model.branch_input_dim = int(data_module.payload["branch_inputs"].shape[-1])
    config.model.field_output_dim = int(data_module.payload["field_targets"].shape[-1])
    config.model.scalar_output_dim = int(data_module.payload["scalar_targets"].shape[-1])
    config.model.trunk_input_dim = int(data_module.payload["query_points"].shape[-1])

    model = create_model(config.model)
    trainer = Trainer(
        config=config,
        model=model,
        data_module=data_module,
        loss_fn=CompositeLoss(config=config.loss, normalizers=data_module.normalizers),
    )
    trainer.fit()

    checkpoint_path = Path(config.experiment.run_dir) / "checkpoints" / "best.pt"
    predictor = Predictor.from_checkpoint(checkpoint_path, device="cpu")

    demo_dataset = data_module.datasets["test"]
    sample = demo_dataset[0]
    result = predictor.predict(
        branch_inputs_raw=sample["branch_inputs_raw"].numpy(),
        query_points_raw=sample["query_points_raw"].numpy(),
        flow_conditions=sample["flow_conditions"].numpy(),
        surface_points_raw=sample["surface_points_raw"].numpy(),
    )

    output_dir = Path(config.experiment.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "demo_prediction.json"
    serializable_result = {
        "predicted_fields_first_5": np.asarray(result["predicted_fields"][:5], dtype=np.float32).tolist(),
        "predicted_scalars": result["predicted_scalars"],
        "surface_cp_first_10": np.asarray(result["surface_cp"][:10], dtype=np.float32).tolist() if "surface_cp" in result else [],
        "metadata": result["metadata"],
    }
    save_json(result_path, serializable_result)

    print("Demo run completed.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Prediction JSON: {result_path}")
    print(f"Predicted Cl: {result['predicted_scalars']['cl']:.6f}")
    print(f"Predicted Cd: {result['predicted_scalars']['cd']:.6f}")
    print("First 3 predicted field rows:")
    for row in np.asarray(result["predicted_fields"][:3], dtype=np.float32):
        print("  ", row.tolist())


if __name__ == "__main__":
    main()
