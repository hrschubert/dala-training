import dataclasses
import logging
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp
import jax.sharding as jshard
import numpy as np
import optax
from flax import nnx
from flax.struct import dataclass

from lczero_training.model.model import LczeroModel
from lczero_training.training.lr_schedule import make_lr_schedule
from lczero_training.training.optimizer import (
    make_gradient_transformation,
    update_optimizer_step,
)
from proto.model_config_pb2 import ModelConfig
from proto.training_config_pb2 import TrainingConfig

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TrainingSample:
    """Single training sample without batch dimension.

    Used for vmap over individual samples in loss computation.

    Fields:
        inputs: Input planes tensor [112, 8, 8]
        probabilities: Policy probabilities tensor [1858]
        values: Combined values tensor [6, 3] where:
            - Index 0: result [result_q, result_d, plies_left]
            - Index 1: best [best_q, best_d, best_m]
            - Index 2: played [played_q, played_d, played_m]
            - Index 3: orig [orig_q, orig_d, orig_m] (may contain NaN)
            - Index 4: root [root_q, root_d, root_m]
            - Index 5: st [q_st, d_st, NaN]
        teacher_policies: Optional dict mapping teacher policy-head name to
            logits tensor [1858]. Empty when no teacher network is
            configured. Populated per-batch by the training loop before
            being vmapped down to per-sample tensors.
    """

    inputs: jax.Array
    probabilities: jax.Array
    values: jax.Array
    teacher_policies: Dict[str, jax.Array] = dataclasses.field(
        default_factory=dict
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class TrainingBatch:
    """Batch of training data with inputs, probabilities, and values tensors.

    Fields:
        inputs: Input planes tensor [batch, 112, 8, 8]
        probabilities: Policy probabilities tensor [batch, 1858]
        values: Combined values tensor [batch, 6, 3] where:
            - Index 0: result [result_q, result_d, plies_left]
            - Index 1: best [best_q, best_d, best_m]
            - Index 2: played [played_q, played_d, played_m]
            - Index 3: orig [orig_q, orig_d, orig_m] (may contain NaN)
            - Index 4: root [root_q, root_d, root_m]
            - Index 5: st [q_st, d_st, NaN]
        teacher_policies: Optional dict mapping teacher policy-head name to
            policy logits tensor [batch, 1858]. Empty when no teacher
            network is configured.
    """

    inputs: Union[jax.Array, jshard.NamedSharding]
    probabilities: Union[jax.Array, jshard.NamedSharding]
    values: Union[jax.Array, jshard.NamedSharding]
    teacher_policies: Dict[str, Union[jax.Array, jshard.NamedSharding]] = (
        dataclasses.field(default_factory=dict)
    )

    def replace(self, **changes: Any) -> "TrainingBatch":
        """Returns a new instance of the class with the specified changes."""
        return dataclasses.replace(self, **changes)

    @classmethod
    def from_tuple(
        cls, tensor_tuple: tuple[np.ndarray, ...]
    ) -> "TrainingBatch":
        """Create TrainingBatch from tuple returned by DataLoader."""
        if len(tensor_tuple) != 3:
            raise ValueError(
                f"Expected tuple of 3 tensors, got {len(tensor_tuple)}"
            )
        return cls(
            inputs=jnp.asarray(tensor_tuple[0]),
            probabilities=jnp.asarray(tensor_tuple[1]),
            values=jnp.asarray(tensor_tuple[2]),
            teacher_policies={},
        )


@dataclass
class JitTrainingState:
    step: int
    model_state: nnx.State
    opt_state: Optional[optax.OptState]
    # SWA state mirrors model_state structure when enabled; None otherwise.
    # Marked non-pytree to exclude from JIT/pjit inputs and device transfers.
    swa_state: Optional[nnx.State]
    # Effective number of model snapshots accumulated into SWA (can be fractional).
    num_averages: float

    def replace(self, **changes: Any) -> "JitTrainingState":
        """Returns a new instance of the class with the specified changes."""
        return dataclasses.replace(self, **changes)


@dataclass
class TrainingState:
    jit_state: JitTrainingState
    # Last chunk source that was available when the last epoch started training.
    num_heads: int
    last_chunk_source: str = ""

    def replace(self, **changes: Any) -> "TrainingState":
        """Returns a new instance of the class with the specified changes."""
        return dataclasses.replace(self, **changes)

    def with_updated_step(self, step: int) -> "TrainingState":
        """Returns a copy with updated step in both jit_state and optimizer."""
        updated_opt_state = (
            update_optimizer_step(self.jit_state.opt_state, step)
            if self.jit_state.opt_state is not None
            else None
        )
        return self.replace(
            jit_state=self.jit_state.replace(
                step=step,
                opt_state=updated_opt_state,
            )
        )

    @staticmethod
    def new_from_config(
        model_config: ModelConfig, training_config: TrainingConfig
    ) -> "TrainingState":
        rngs = nnx.Rngs(params=42)
        model_state = nnx.state(LczeroModel(config=model_config, rngs=rngs))
        lr_sched = make_lr_schedule(training_config.lr_schedule)
        opt_state = make_gradient_transformation(
            training_config.optimizer,
            max_grad_norm=getattr(training_config, "max_grad_norm", 0.0),
            lr_schedule=lr_sched,
        ).init(model_state)
        import jax
        swa_copy = jax.tree.map(lambda x: x.copy(), model_state)
        jit_state = JitTrainingState(
            step=0,
            model_state=model_state,
            opt_state=opt_state,
            swa_state=swa_copy,
            num_averages=0.0,
        )
        return TrainingState(
            jit_state=jit_state,
            num_heads=model_config.encoder.heads,
        )
