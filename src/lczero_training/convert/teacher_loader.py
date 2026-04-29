"""Loader for a frozen teacher network used in policy distillation.

Loads an exported lc0 ``.pb.gz`` weights file, builds an ``LczeroModel``
from its embedded config, copies weights in via the standard converter,
and returns a JIT-compiled batch forward function that yields the
teacher's policy logits per head.

Usage::

    teacher_fn = load_teacher_predictions_fn(
        path="/path/to/teacher.pb.gz",
        compute_dtype=hlo_pb2.XlaShapeProto.F32,
    )
    teacher_logits = teacher_fn(batch_inputs)
    # -> {"vanilla": jnp.ndarray[batch, 1858], ...}

The returned function is pure and JIT-friendly; the teacher's weights
are captured as constants in its closure, so each invocation is just an
inference pass.
"""

import gzip
import logging
from typing import Callable, Dict

import jax
from flax import nnx

from lczero_training.convert.leela_to_jax import (
    LeelaImportOptions,
    fix_older_weights_file,
    leela_to_jax,
)
from lczero_training.convert.leela_to_modelconfig import leela_to_modelconfig
from lczero_training.model.model import LczeroModel
from proto import hlo_pb2, net_pb2

logger = logging.getLogger(__name__)


TeacherPredictFn = Callable[[jax.Array], Dict[str, jax.Array]]


def load_teacher_predictions_fn(
    path: str,
    compute_dtype: hlo_pb2.XlaShapeProto.Type,
) -> TeacherPredictFn:
    """Load a teacher .pb.gz and return a JIT-compiled batch forward fn.

    The returned callable takes a batch of inputs of shape
    ``[batch, 112, 8, 8]`` and returns a ``dict`` mapping each policy-head
    name in the teacher to its logits of shape ``[batch, 1858]``.

    The teacher uses its own embedded model config — it does NOT need to
    match the student's architecture. This is the whole point of
    distillation: a strong, large teacher transferring knowledge to a
    smaller, human-aligned student.
    """
    logger.info(f"Loading teacher network: {path}")
    leela_net = net_pb2.Net()
    with gzip.open(path, "rb") as f:
        leela_net.ParseFromString(f.read())
    fix_older_weights_file(leela_net)

    teacher_config = leela_to_modelconfig(
        leela_net,
        weights_dtype=hlo_pb2.XlaShapeProto.F32,
        compute_dtype=compute_dtype,
    )

    import_options = LeelaImportOptions(
        weights_dtype=hlo_pb2.XlaShapeProto.F32, compute_dtype=compute_dtype
    )
    teacher_state = leela_to_jax(
        leela_net, import_options, target_config=teacher_config
    )

    teacher_model = LczeroModel(config=teacher_config, rngs=nnx.Rngs(params=0))
    nnx.update(teacher_model, teacher_state)
    graphdef, state = nnx.split(teacher_model)

    policy_head_names = [h.name for h in teacher_config.policy_head]
    logger.info(
        f"Teacher loaded with policy heads: {policy_head_names} "
        f"(d_model={teacher_config.encoder.d_model}, "
        f"blocks={teacher_config.encoder.num_blocks})"
    )

    @jax.jit
    def _teacher_forward(inputs: jax.Array) -> Dict[str, jax.Array]:
        """Run a forward pass over a batch and return policy logits.

        ``inputs`` shape: ``[batch, 112, 8, 8]``.
        Returns a dict ``{head_name: [batch, 1858]}``.
        """
        teacher = nnx.merge(graphdef, state)

        def _per_sample(x: jax.Array) -> Dict[str, jax.Array]:
            return teacher(x).policy

        return jax.vmap(_per_sample)(inputs)

    return _teacher_forward
