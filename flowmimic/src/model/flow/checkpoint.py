RELATIVE_TIME_STATE_MARKER = ".relative_time_bias."
LATENT_SLOT_ADAPTER_STATE_MARKER = "slot_condition_adapter."


def flow_state_uses_relative_time_bias(state_dict):
    return any(RELATIVE_TIME_STATE_MARKER in key for key in state_dict)


def infer_relative_time_hidden_dim(state_dict, default=32):
    suffix = ".relative_time_bias.net.0.weight"
    for key, value in state_dict.items():
        if key.endswith(suffix):
            return int(value.shape[0])
    return int(default)


def flow_state_uses_latent_slot_adapter(state_dict):
    return any(LATENT_SLOT_ADAPTER_STATE_MARKER in key for key in state_dict)


def infer_latent_slot_adapter_config(
    state_dict,
    *,
    default_latent_len=16,
    default_ffn_dim=1024,
):
    latent_len = int(default_latent_len)
    ffn_dim = int(default_ffn_dim)
    for key, value in state_dict.items():
        if key.endswith("slot_condition_adapter.slot_queries"):
            latent_len = int(value.shape[1])
        elif key.endswith("slot_condition_adapter.ffn.0.weight"):
            ffn_dim = int(value.shape[0])
    return {"latent_len": latent_len, "ffn_dim": ffn_dim}


def load_flow_state_dict(
    model,
    state_dict,
    allow_relative_time_migration=False,
    allow_latent_slot_adapter_migration=False,
):
    """Load a flow state, optionally initializing new conditioning branches."""
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if allow_relative_time_migration:
        missing = [
            key for key in missing if RELATIVE_TIME_STATE_MARKER not in key
        ]
    if allow_latent_slot_adapter_migration:
        missing = [
            key
            for key in missing
            if LATENT_SLOT_ADAPTER_STATE_MARKER not in key
        ]
    if missing or unexpected:
        raise RuntimeError(
            "Flow checkpoint is incompatible: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return incompatible
