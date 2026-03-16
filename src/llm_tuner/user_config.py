from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


# Compatible with sft-rl-pipeline style config.py keys.
# Users can still pass llm-tuner CLI flags to override these values.
SUPPORTED_KEYS = {
    # model/common
    "model",
    "model_name",
    "renderer_name",
    "lora_rank",
    "base_url",
    "wandb_project",
    "wandb_exe_project",
    "wandb_exe_url",
    # output/run
    "output_root",
    "run_name",
    "log_dir",
    # sft
    "sft_dataset",
    "dataset_path",
    "sft_lr",
    "sft_batch_size",
    "sft_max_length",
    "sft_epochs",
    "sft_max_steps",
    "sft_save_every",
    "sft_wandb_name",
    # rl
    "rl_env",
    "rl_lr",
    "rl_group_size",
    "rl_groups_per_batch",
    "rl_max_tokens",
    "rl_max_steps",
    "rl_save_every",
    "rl_wandb_name",
    "rl_from_checkpoint",
    "from_checkpoint",
}


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("_llm_tuner_user_config", path)

    # spec_from_file_location can return None for non-.py suffixes.
    if spec is None or spec.loader is None:
        from importlib.machinery import SourceFileLoader

        loader = SourceFileLoader("_llm_tuner_user_config", str(path))
        spec = importlib.util.spec_from_loader("_llm_tuner_user_config", loader)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config module from {path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_user_config(path: str | None = "config.py") -> tuple[dict[str, Any], Path | None]:
    """Load optional user config.py.

    Returns (config_dict, resolved_path_or_none).
    Missing file is treated as "no config" and not an error.
    """
    if not path:
        return {}, None

    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if not p.exists():
        return {}, None

    mod = _load_module(p)
    cfg: dict[str, Any] = {}
    for key in SUPPORTED_KEYS:
        if hasattr(mod, key):
            cfg[key] = getattr(mod, key)

    return cfg, p


def normalize_sft_dataset(value: str) -> str:
    """Map common dataset aliases to llm-tuner dataset files when available."""
    raw = str(value).strip()

    aliases = {
        "no_robots": Path("data/no_robots_sft_3000.jsonl"),
    }
    if raw in aliases:
        candidate = aliases[raw]
        if candidate.exists():
            return str(candidate)

    return raw
