"""Train a neural operator surrogate."""

from __future__ import annotations

import argparse

from cfd_operator.config import load_config
from cfd_operator.data import CFDDataModule
from cfd_operator.losses import CompositeLoss
from cfd_operator.models import create_model
from cfd_operator.trainers import Trainer
from cfd_operator.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CFD operator surrogate.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", action="append", default=[], help="Override config keys, e.g. train.epochs=5")
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.override)
    set_seed(config.experiment.seed)

    data_module = CFDDataModule(config=config.data, batch_size=config.train.batch_size)
    data_module.setup()

    assert data_module.payload is not None
    config.model.branch_input_dim = int(data_module.payload["branch_inputs"].shape[-1])
    config.model.field_output_dim = int(data_module.payload["field_targets"].shape[-1])
    config.model.scalar_output_dim = int(data_module.payload["scalar_targets"].shape[-1])
    config.model.trunk_input_dim = int(data_module.payload["query_points"].shape[-1])

    model = create_model(config.model)
    loss_fn = CompositeLoss(config=config.loss, normalizers=data_module.normalizers)
    trainer = Trainer(config=config, model=model, data_module=data_module, loss_fn=loss_fn)
    trainer.fit()


if __name__ == "__main__":
    main()

