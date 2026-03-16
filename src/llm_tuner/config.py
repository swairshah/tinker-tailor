from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SFTConfig:
    model_name: str = "Qwen/Qwen3.5-4B"
    renderer_name: str = "qwen3_5_disable_thinking"
    dataset_path: str = "data/sft_conversations.jsonl"
    lora_rank: int = 32
    max_length: int = 4096
    batch_size: int = 64
    learning_rate: float = 2e-4
    num_epochs: int = 1
    max_steps: int = 20
    save_every: int = 5
    base_url: str | None = None
    wandb_project: str | None = None
    wandb_name: str = "qwen35-4b-sft"


@dataclass
class RLConfig:
    model_name: str = "Qwen/Qwen3.5-4B"
    renderer_name: str = "qwen3_5_disable_thinking"
    lora_rank: int = 32
    env: str = "arithmetic"  # arithmetic | gsm8k | math | polaris | deepmath
    group_size: int = 4
    groups_per_batch: int = 32
    learning_rate: float = 1e-5
    max_tokens: int = 64
    max_steps: int = 30
    save_every: int = 5
    base_url: str | None = None
    wandb_project: str | None = None
    wandb_name: str = "qwen35-4b-rl"


@dataclass
class PipelineConfig:
    output_root: str = "runs"
    run_name: str | None = None
    download_final_adapter: bool = True
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
