import argparse
import datetime
import gzip
import logging
import os
import sys

import orbax.checkpoint as ocp
from flax import nnx
from google.protobuf import text_format

from lczero_training.commands import configure_root_logging
from lczero_training.convert.jax_to_leela import (
    LeelaExportOptions,
    jax_to_leela,
)
from lczero_training.dataloader import make_dataloader
from lczero_training.model.loss_function import LczeroLoss
from lczero_training.model.model import LczeroModel
from lczero_training.training.lr_schedule import make_lr_schedule
from lczero_training.training.optimizer import make_gradient_transformation
from lczero_training.training.state import TrainingState
from lczero_training.training.training import Training, from_dataloader
from proto.root_config_pb2 import RootConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start a training run.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config file.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save a checkpoint every N steps (0 = only at end).",
    )
    parser.add_argument(
        "--export-every",
        type=int,
        default=0,
        help="Export a network file every N steps (0 = only at end).",
    )
    parser.add_argument(
        "--from-network",
        type=str,
        default=None,
        help="Initialize from an exported lc0 network (.pb.gz) instead of "
        "requiring an existing checkpoint. Runs lc0-init automatically.",
    )
    parser.add_argument(
        "--override-steps",
        type=int,
        default=None,
        help="Override the starting step number (use with --from-network "
        "to start from step 0 instead of the network's embedded step).",
    )
    parser.add_argument(
        "--ignore-config-mismatch",
        action="store_true",
        help="Ignore model config mismatch when loading from --from-network.",
    )
    return parser


def _export_network(config, training_state, jit_state):
    """Export the network to file(s) based on config."""
    if not config.export.destination_filename:
        return
    date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.info("Exporting network")
    options = LeelaExportOptions(
        min_version="0.28",
        num_heads=training_state.num_heads,
        license=None,
    )
    export_state = (
        jit_state.swa_state
        if config.export.export_swa_model
        else jit_state.model_state
    )
    assert isinstance(export_state, nnx.State)
    net = jax_to_leela(jax_weights=export_state, export_options=options)
    network_bytes = gzip.compress(net.SerializeToString())
    step_value = int(jit_state.step)
    for destination_template in config.export.destination_filename:
        destination = destination_template.format(
            datetime=date_str, step=step_value
        )
        logging.info(f"Writing network to {destination}")
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as f:
            f.write(network_bytes)
        logging.info(f"Finished writing network to {destination}")


def train(
    config_filename: str,
    checkpoint_every: int = 0,
    export_every: int = 0,
    from_network: str | None = None,
    override_steps: int | None = None,
    ignore_config_mismatch: bool = False,
) -> None:
    config = RootConfig()
    logging.info("Reading configuration from proto file")
    with open(config_filename, "r") as f:
        text_format.Parse(f.read(), config)

    if config.training.checkpoint.path is None:
        logging.error("Checkpoint path must be set in the configuration.")
        sys.exit(1)

    # If --from-network is given, run init to create the checkpoint first
    if from_network is not None:
        logging.info(f"Initializing from network: {from_network}")
        from lczero_training.training.init import init

        init_kwargs = dict(
            config_filename=config_filename,
            lczero_model=from_network,
            seed=42,
            dry_run=False,
            swa_initial_nets=0,
            override_training_steps=override_steps,
            overwrite=True,
            no_copy_swa=False,
            ignore_config_mismatch=ignore_config_mismatch,
        )
        init(**init_kwargs)
        logging.info("Initialization from network complete")

    checkpoint_mgr = ocp.CheckpointManager(
        config.training.checkpoint.path,
        options=ocp.CheckpointManagerOptions(
            create=True,
        ),
    )

    logging.info("Creating state from configuration")
    empty_state = TrainingState.new_from_config(
        model_config=config.model,
        training_config=config.training,
    )
    logging.info("Restoring checkpoint")
    try:
        training_state = checkpoint_mgr.restore(
            None, args=ocp.args.PyTreeRestore(empty_state)
        )
    except ValueError as e:
        if "tree structures do not match" in str(e):
            logging.warning(
                "Checkpoint tree structure mismatch — retrying with "
                "partial_restore=True: %s", e
            )
            training_state = checkpoint_mgr.restore(
                None,
                args=ocp.args.PyTreeRestore(
                    empty_state, partial_restore=True
                ),
            )
        else:
            raise
    logging.info("Restored checkpoint")

    model, _ = nnx.split(
        LczeroModel(config=config.model, rngs=nnx.Rngs(params=42))
    )

    assert isinstance(training_state, TrainingState)

    jit_state = training_state.jit_state
    lr_sched = make_lr_schedule(config.training.lr_schedule)
    optimizer_tx = make_gradient_transformation(
        config.training.optimizer,
        max_grad_norm=getattr(config.training, "max_grad_norm", 0.0),
        lr_schedule=lr_sched,
    )
    training = Training(
        optimizer_tx=optimizer_tx,
        graphdef=model,
        loss_fn=LczeroLoss(config=config.training.losses),
        swa_config=(
            config.training.swa if config.training.HasField("swa") else None
        ),
    )

    def step_hook(hook_data):
        step = hook_data.global_step
        if checkpoint_every > 0 and step % checkpoint_every == 0:
            logging.info(f"Periodic checkpoint at step {step}")
            save_state = training_state.replace(
                jit_state=hook_data.jit_state
            )
            checkpoint_mgr.save(step, args=ocp.args.PyTreeSave(save_state))
            checkpoint_mgr.wait_until_finished()
            logging.info(f"Checkpoint saved at step {step}")
        if export_every > 0 and step % export_every == 0:
            logging.info(f"Periodic export at step {step}")
            _export_network(config, training_state, hook_data.jit_state)

    new_state = training.run(
        jit_state,
        from_dataloader(make_dataloader(config.data_loader)),
        config.training.schedule.steps_per_network,
        step_hook=step_hook if (checkpoint_every or export_every) else None,
    )

    # Final checkpoint
    step_value = int(new_state.step)
    logging.info(f"Saving final checkpoint at step {step_value}")
    save_state = training_state.replace(jit_state=new_state)
    checkpoint_mgr.save(step_value, args=ocp.args.PyTreeSave(save_state))
    checkpoint_mgr.wait_until_finished()
    logging.info("Final checkpoint saved")

    # Final export
    _export_network(config, training_state, new_state)


def main(argv: list[str] | None = None) -> int:
    configure_root_logging(logging.INFO)

    parser = _build_parser()
    args = parser.parse_args(argv)

    train(
        config_filename=args.config,
        checkpoint_every=args.checkpoint_every,
        export_every=args.export_every,
        from_network=args.from_network,
        override_steps=args.override_steps,
        ignore_config_mismatch=args.ignore_config_mismatch,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
