from proto import hlo_pb2, model_config_pb2, net_pb2


def _defaultactivation_to_activation(
    activation: net_pb2.NetworkFormat.DefaultActivation,
) -> net_pb2.NetworkFormat.ActivationFunction:
    return {
        net_pb2.NetworkFormat.DEFAULT_ACTIVATION_RELU: net_pb2.NetworkFormat.ACTIVATION_RELU,
        net_pb2.NetworkFormat.DEFAULT_ACTIVATION_MISH: net_pb2.NetworkFormat.ACTIVATION_MISH,
    }[activation]


def leela_to_modelconfig(
    leela_net: net_pb2.Net,
    weights_dtype: hlo_pb2.XlaShapeProto.Type,
    compute_dtype: hlo_pb2.XlaShapeProto.Type,
) -> model_config_pb2.ModelConfig:
    assert weights_dtype == hlo_pb2.XlaShapeProto.F32, (
        "Only float32 weights are supported."
    )
    assert leela_net.format.weights_encoding == net_pb2.Format.LINEAR16
    leela_net_format = leela_net.format.network_format
    model_config = model_config_pb2.ModelConfig()

    model_config.defaults.compute_dtype = compute_dtype
    model_config.defaults.activation = _defaultactivation_to_activation(
        leela_net_format.default_activation
    )
    model_config.defaults.ffn_activation = (
        leela_net_format.ffn_activation or model_config.defaults.activation
    )
    assert (
        leela_net_format.input_embedding
        == net_pb2.NetworkFormat.INPUT_EMBEDDING_PE_DENSE
    ), "Only dense positional embedding is supported, got {}".format(
        net_pb2.NetworkFormat.InputEmbeddingFormat.Name(
            leela_net_format.input_embedding
        )
    )
    assert leela_net_format.policy == net_pb2.NetworkFormat.POLICY_ATTENTION, (
        "Only attention policy is supported, got {}".format(
            net_pb2.NetworkFormat.PolicyFormat.Name(leela_net_format.policy)
        )
    )
    assert leela_net_format.value == net_pb2.NetworkFormat.VALUE_WDL, (
        "Only WDL value is supported, got {}".format(
            net_pb2.NetworkFormat.ValueFormat.Name(leela_net_format.value)
        )
    )
    has_moves_left = (
        leela_net_format.moves_left == net_pb2.NetworkFormat.MOVES_LEFT_V1
    )

    def size(x: net_pb2.Weights.Layer) -> int:
        return len(x.params) // 2

    assert (
        leela_net_format.network
        == net_pb2.NetworkFormat.NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT
    )
    weights = leela_net.weights
    model_config.embedding.dense_size = size(weights.ip_emb_preproc_b) // 64
    model_config.embedding.embedding_size = size(weights.ip_emb_b)
    assert size(weights.ip_mult_gate) > 0
    assert size(weights.ip_add_gate) > 0
    model_config.embedding.dff = size(weights.ip_emb_ffn.dense1_b)

    model_config.encoder.num_blocks = len(weights.encoder)
    assert model_config.encoder.num_blocks > 0
    encoder = weights.encoder[0]
    model_config.encoder.d_model = size(encoder.mha.q_b)
    model_config.encoder.heads = weights.headcount
    model_config.encoder.dff = size(encoder.ffn.dense1_b)

    if weights.HasField("smolgen_w"):
        model_config.encoder.smolgen.activation = (
            leela_net_format.smolgen_activation
            or model_config.defaults.activation
        )
        model_config.encoder.smolgen.hidden_channels = (
            size(encoder.mha.smolgen.compress)
            // model_config.embedding.embedding_size
        )
        model_config.encoder.smolgen.gen_size = (
            size(encoder.mha.smolgen.dense2_b) // weights.headcount
        )
        model_config.encoder.smolgen.hidden_size = size(
            encoder.mha.smolgen.dense1_b
        )

    if weights.policy_heads.HasField("ip_pol_w"):
        model_config.shared_policy_embedding_size = size(
            weights.policy_heads.ip_pol_b
        )

    for head_name in ["vanilla", "optimistic_st", "soft", "opponent"]:
        if weights.policy_heads.HasField(head_name):
            head = getattr(weights.policy_heads, head_name)
            assert size(head.ip2_pol_b) > 0
            policy_head = model_config.policy_head.add()
            policy_head.name = head_name
            if head.HasField("ip_pol_w"):
                # Per-head policy embedding
                policy_head.embedding_size = size(head.ip_pol_b)
            elif not model_config.HasField("shared_policy_embedding_size"):
                policy_head.embedding_size = size(head.ip_pol_b)
            policy_head.d_model = size(head.ip2_pol_b)

    for head_name in ["winner", "q", "st"]:
        if weights.value_heads.HasField(head_name):
            head = getattr(weights.value_heads, head_name)
            assert size(head.ip_val_b) > 0
            value_head = model_config.value_head.add()
            value_head.name = head_name
            value_head.num_channels = size(head.ip_val_b)
            if head.HasField("ip_val_err_w"):
                value_head.has_error_output = True
            if head.HasField("ip_val_cat_b"):
                value_head.num_categorical_buckets = size(head.ip_val_cat_b)

    if has_moves_left and size(weights.ip_mov_b) > 0:
        movesleft_head = model_config.movesleft_head.add()
        movesleft_head.name = "main"
        movesleft_head.num_channels = size(weights.ip_mov_b)

    return model_config


def is_compatible_subset(
    source: model_config_pb2.ModelConfig,
    target: model_config_pb2.ModelConfig,
) -> tuple[bool, str]:
    """Return (is_compatible, reason).

    Configs are compatible if every layer/head in ``source`` also exists with
    identical configuration in ``target``. ``target`` may declare additional
    heads that are missing in ``source`` — those are allowed and will be left
    at their random initialization when importing weights.

    Use this to allow loading a network with fewer heads than the training
    textproto declares (e.g. importing a 1-policy-head network into a
    2-policy-head training config).
    """
    if source.defaults != target.defaults:
        return False, "model.defaults differ"
    if source.embedding != target.embedding:
        return False, "model.embedding differs"
    if source.encoder != target.encoder:
        return False, "model.encoder differs"
    if source.HasField(
        "shared_policy_embedding_size"
    ) != target.HasField("shared_policy_embedding_size"):
        return False, "shared_policy_embedding_size presence differs"
    if (
        source.HasField("shared_policy_embedding_size")
        and source.shared_policy_embedding_size
        != target.shared_policy_embedding_size
    ):
        return False, "shared_policy_embedding_size value differs"

    def _check_heads(field_name: str) -> tuple[bool, str]:
        target_by_name = {h.name: h for h in getattr(target, field_name)}
        for src_head in getattr(source, field_name):
            if src_head.name not in target_by_name:
                return (
                    False,
                    f"{field_name} '{src_head.name}' present in source "
                    f"but not in target",
                )
            if src_head != target_by_name[src_head.name]:
                return (
                    False,
                    f"{field_name} '{src_head.name}' configuration differs",
                )
        return True, ""

    for field in ("policy_head", "value_head", "movesleft_head"):
        ok, reason = _check_heads(field)
        if not ok:
            return False, reason
    return True, "compatible"
