from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train as supervised_train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from llm_tuner.config import SFTConfig


@dataclass
class SFTRunResult:
    log_dir: str
    state_path: str
    sampler_path: str


def run_sft(config: SFTConfig, log_dir: str | Path) -> SFTRunResult:
    log_dir = str(log_dir)

    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=config.model_name,
        renderer_name=config.renderer_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )

    dataset_builder = FromConversationFileBuilder(
        common_config=common,
        file_path=config.dataset_path,
    )

    sft_cfg = supervised_train.Config(
        log_path=log_dir,
        model_name=config.model_name,
        renderer_name=config.renderer_name,
        dataset_builder=dataset_builder,
        learning_rate=config.learning_rate,
        lr_schedule="linear",
        num_epochs=config.num_epochs,
        lora_rank=config.lora_rank,
        save_every=config.save_every,
        eval_every=0,
        max_steps=config.max_steps,
        base_url=config.base_url,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
    )

    asyncio.run(supervised_train.main(sft_cfg))

    ckpt = checkpoint_utils.get_last_checkpoint(log_dir, required_key="state_path")
    if ckpt is None or ckpt.state_path is None or ckpt.sampler_path is None:
        raise RuntimeError("SFT finished but no state/sampler checkpoint was found.")

    return SFTRunResult(log_dir=log_dir, state_path=ckpt.state_path, sampler_path=ckpt.sampler_path)
