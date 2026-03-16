from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.recipes.math_rl import arithmetic_env, math_env
from tinker_cookbook.rl import train as rl_train

from llm_tuner.config import RLConfig


@dataclass
class RLRunResult:
    log_dir: str
    state_path: str
    sampler_path: str


def build_rl_dataset(config: RLConfig):
    if config.env == "arithmetic":
        return arithmetic_env.ArithmeticDatasetBuilder(
            batch_size=config.groups_per_batch,
            model_name_for_tokenizer=config.model_name,
            renderer_name=config.renderer_name,
            n_batches=max(config.max_steps + 5, 20),
            group_size=config.group_size,
            include_fewshot=True,
        )

    if config.env in {"gsm8k", "math", "polaris", "deepmath"}:
        return math_env.get_math_dataset_builder(
            dataset_name=config.env,
            batch_size=config.groups_per_batch,
            model_name_for_tokenizer=config.model_name,
            renderer_name=config.renderer_name,
            group_size=config.group_size,
            seed=0,
        )

    raise ValueError(
        f"Unsupported env={config.env!r}. Use one of: arithmetic, gsm8k, math, polaris, deepmath"
    )


def run_rl(
    config: RLConfig,
    log_dir: str | Path,
    load_checkpoint_path: str,
) -> RLRunResult:
    log_dir = str(log_dir)

    rl_cfg = rl_train.Config(
        learning_rate=config.learning_rate,
        dataset_builder=build_rl_dataset(config),
        model_name=config.model_name,
        renderer_name=config.renderer_name,
        lora_rank=config.lora_rank,
        max_tokens=config.max_tokens,
        log_path=log_dir,
        load_checkpoint_path=load_checkpoint_path,
        eval_every=0,
        save_every=config.save_every,
        max_steps=config.max_steps,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        base_url=config.base_url,
    )

    asyncio.run(rl_train.main(rl_cfg))

    ckpt = checkpoint_utils.get_last_checkpoint(log_dir, required_key="sampler_path")
    if ckpt is None or ckpt.state_path is None or ckpt.sampler_path is None:
        raise RuntimeError("RL finished but no state/sampler checkpoint was found.")

    return RLRunResult(log_dir=log_dir, state_path=ckpt.state_path, sampler_path=ckpt.sampler_path)
