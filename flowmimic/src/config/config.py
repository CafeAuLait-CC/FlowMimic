import json
import os


def load_config(path=None):
    if path is None:
        path = os.path.join("flowmimic", "src", "config", "config.json")

    with open(path, "r", encoding="utf-8") as f:
        return normalize_config(json.load(f))


def normalize_config(config):
    """Return a config dict with grouped sections plus legacy flat aliases.

    Older scripts in this repo read keys such as ``config["seq_len"]`` and
    ``config["aist_motions_dir"]`` directly.  The JSON file is now grouped, so
    this function injects those flat aliases after load instead of duplicating
    values in the file.
    """
    cfg = dict(config)

    paths = cfg.get("paths", {})
    data = cfg.get("data", {})
    motion = data.get("motion", {})
    splits = data.get("splits", {})
    openpose = data.get("openpose", {})
    cameras = data.get("cameras", {})
    model = cfg.get("model", {})
    vae = cfg.get("vae", {})
    evaluator = cfg.get("evaluator", {})
    runtime = cfg.get("runtime", {})

    _set_missing(cfg, "aist_motions_dir", paths.get("aist_motions_dir"))
    _set_missing(cfg, "mvhumannet_root", paths.get("mvhumannet_root"))
    _set_missing(cfg, "cache_root", paths.get("cache_root"))
    _set_missing(cfg, "cond_cache_root", paths.get("cond_cache_root"))
    _set_missing(cfg, "smpl45_to_body25_def", paths.get("smpl45_to_body25_def"))
    _set_missing(cfg, "stats_path", paths.get("stats_path"))
    _set_missing(cfg, "latent_stats_path", paths.get("latent_stats_path"))
    _set_missing(cfg, "openpose_stats_path", paths.get("openpose_stats_path"))

    _set_missing(cfg, "seq_len", motion.get("seq_len"))
    _set_missing(cfg, "target_fps", motion.get("target_fps"))
    _set_missing(cfg, "aist_fps", motion.get("aist_fps"))
    _set_missing(cfg, "mvh_fps", motion.get("mvh_fps"))
    _set_missing(cfg, "aist_genres", data.get("aist_genres"))
    _set_missing(cfg, "num_styles", data.get("num_styles", model.get("num_styles")))
    _set_missing(cfg, "d_in", model.get("d_in"))
    _set_missing(cfg, "d_z", model.get("d_z"))

    _set_missing(cfg, "aist_split_train", splits.get("aist_train"))
    _set_missing(cfg, "aist_split_val", splits.get("aist_val"))
    _set_missing(cfg, "aist_split_test", splits.get("aist_test"))
    _set_missing(cfg, "mvh_split_train", splits.get("mvh_train"))
    _set_missing(cfg, "mvh_split_val", splits.get("mvh_val"))

    _set_missing(cfg, "aist_openpose_dir", openpose.get("aist_dir"))
    _set_missing(cfg, "mvh_openpose_root", openpose.get("mvh_root"))
    _set_missing(cfg, "aist_cameras", cameras.get("aist"))
    _set_missing(cfg, "mvh_cameras", cameras.get("mvh"))

    _set_missing(cfg, "vae_ckpt", vae.get("ckpt"))
    losses = vae.get("losses", {})
    for key in (
        "kl_target_weight",
        "kl_warmup_steps",
        "w_vel",
        "w_acc",
        "w_contact",
        "w_root",
        "w_root_late_start",
        "w_root_late_factor",
        "w_style",
        "style_dropout_p",
    ):
        _set_missing(cfg, key, losses.get(key))

    _set_missing(
        cfg,
        "t2m_motion_encoder_ckpt",
        evaluator.get("t2m_motion_encoder_ckpt"),
    )
    _set_missing(cfg, "t2m_eval_mean_path", evaluator.get("t2m_eval_mean_path"))
    _set_missing(cfg, "t2m_eval_std_path", evaluator.get("t2m_eval_std_path"))

    for key in (
        "seed",
        "train_batch_size",
        "eval_batch_size",
        "num_workers",
        "pin_memory",
        "prefetch_factor",
        "persistent_workers",
        "grad_clip_norm",
        "val_every_epochs",
    ):
        _set_missing(cfg, key, runtime.get(key))

    cfg["flow"] = _merge_nested_section(cfg.get("flow", {}))
    return cfg


def _set_missing(config, key, value):
    if value is not None and key not in config:
        config[key] = value


def _merge_nested_section(section):
    """Merge scalar values from one nested level into a section copy."""
    merged = {}
    for key, value in section.items():
        if isinstance(value, dict):
            merged[key] = value
            for child_key, child_value in value.items():
                if not isinstance(child_value, dict):
                    merged.setdefault(child_key, child_value)
        else:
            merged[key] = value
    return merged
